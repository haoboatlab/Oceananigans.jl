using SparseArrays, LinearAlgebra, Statistics
using CUDA
using CUDA.CUSPARSE
using Oceananigans.Architectures
using Oceananigans.Architectures: arch_array

mutable struct SpaiIterator{VF<:AbstractVector, SV<:AbstractVector, VI<:AbstractVector, SM}
     mhat :: VF
        e :: SV
        r :: SV
        J :: VI
        I :: VI
        J̃ :: VI
        Ĩ :: VI
        Q :: SM
        R :: SM
end

function SpaiIterator(e, r, J, Q)
    mhat = deepcopy(e)
    I    = deepcopy(J)
    Ĩ    = deepcopy(J)
    J̃    = deepcopy(J)
    R    = deepcopy(Q)
    return SpaiIterator(mhat, e, r, J, I, Ĩ, J̃, Q, R)
end

"""
the sparse_approximate_inverse function calculates a SParse Approximate Inverse M ≈ A⁻¹ to be used as a preconditioner
Since it can be applied to the residual with just a matrix multiplication instead of the solution
of a triangular linear problem it makes it very appealing to GPU use

The algorithm implemeted here calculates M following the specifications found in

Grote M. J. & Huckle T, "Parallel Preconditioning with sparse approximate inverses" 

In particular, the algorithm tries to minimize the Frobenius norm of

‖ A mⱼ - eⱼ ‖ where mⱼ and eⱼ are the jₜₕ column of matrix M and the identity matrix I, respectively

Since we are solving for an "sparse approximate" inverse (i.e. a sparse version of the actually non-sparse A⁻¹),
we start assuming that mⱼ has a sparsity pattern J, which means that

mⱼ(k) = 0 ∀k ∉ J 

we call m̂ⱼ = mⱼ(J). From here we calculate the set of row indices I for which

A(i, J) !=0 for i ∈ I 

we call Â = A(I, J). The problem is now reduced to a much smaller minimization problem which can be solved 
with QR decomposition (which luckily we have neatly implemented in julia: Hooray! but not on GPUs... booo)

once solved for m̂ⱼ we compute the residuals of the minimization problem

r = eⱼ - A[:, J] * m̂

we can repeat the computation on the indices for which r != 0 (J̃ and respective Ĩ on the rows),
so that we have Â = A(I U Ĩ, J U J̃) and m̂ = mⱼ(J U J̃)

(... in practice we choose only the more proficuous of the J̃, the ones that will have the larger
change in residual value ...)

To do that we do not need to recompute the entire QR factorization but just update it by appending the
new terms (and recomputing QR for a small part of Â).

sparse_approximate_inverse(A::AbstractMatrix; ε, nzrel)

returns M ≈ A⁻¹ where `|| AM - I ||` ≈ ε and `nnz(M) ≈ nnz(A) * nzrel`

if we choose a sufficiently large `nzrel` (`nzrel = size(A, 1)` for example), then
`sparse_approximate_inverse(A, 0.0, nzrel) = A⁻¹ ± machine_precision`

"""
function sparse_approximate_inverse(A::AbstractMatrix; ε, nzrel)
   
    FT = eltype(A)
    n  = size(A, 1)

    if CUDA.has_cuda_gpu()
        arch = GPU()
    else
        arch = CPU()
    end
     
    A_arch = arch_sparse_matrix(arch, A)
    
    iterators = SpaiIterator[]

    for j in 1:n
        e    = spzeros(FT, n)
        r    = spzeros(FT, n)
        e[j] = FT(1)
        J    = Int64[1]

        Q  = spzeros(FT, 1, 1)

        Q = arch_sparse_matrix(arch, Q)
        J = arch_array(arch, J)
        e = arch_sparse_vector(arch, e)
        r = arch_sparse_vector(arch, r)
        push!(iterators, SpaiIterator(e, r, J, Q))
    end
    
    M  = spzeros(FT, n, n)
    M = arch_sparse_matrix(arch, M)

    maximum_threads = 256
    threads         = max(maximum_threads, n)

    build_kernel = build_approximate_inverse!(Architectures.device(arch), threads, n)
    build_event  = build_kernel(M, A_arch, iterators, ε, nzrel, n, FT)

    wait(device(arch), build_event)

    return arch_sparse_matrix(architecture(A), M)
end

@kernel function build_approximate_inverse!(M, A, iterators, ε, nzrel, n, FT)
    j = @index(Global, Linear)

    iterator = iterators[j]

    # maximum number of elements in a column
    ncolmax = nzrel * nnz(A[:, j])

    set_j_column!(iterator, A, j, ε, ncolmax, n, FT)
    mj             = spzeros(FT, n, 1)
    mj[iterator.J] = iterator.mhat
    M[:, j]        = mj
end

@inline function set_j_column!(iterator, A, j, ε, ncolmax, n, FT)
    @inbounds begin
        # the initial sparsity pattern is assumed to be mⱼ = eⱼ
        initial_sparsity_pattern!(iterator, j)

        # find the initial solution with mⱼ = eⱼ
        find_mhat_given_col!(iterator, A, n)

        # calculate the residuals and locations where r != 0
        calc_residuals!(iterator, A)
        iterator.J̃ = setdiff(iterator.r.nzind, iterator.J)

        # we do not need to select the residuals here as our sparsity pattern is quite large 
        # (only 13 elements maximum in a column). Therefore it gives no benefit to reduce the number of
        # selected iterator.J̃ versus the computational time required by select_residuals. It is nice to switch
        # on this function if we have to calculate the sparse inverse of a much more dense matrix
        # select_residuals!(iterator, A, n, FT)
        
        # iterate until a certain tolerance is met or the maximum number of fill is reached
        while norm(iterator.r) > ε && length(iterator.mhat) < ncolmax
            if isempty(iterator.J̃)
                iterator.r .= 0
            else
                update_mhat_given_col!(iterator, A, FT)
                calc_residuals!(iterator, A)
                iterator.J̃ = setdiff(iterator.r.nzind, iterator.J)
                # select_residuals!(iterator, A, n, FT)
            end
        end
    end
end    

function initial_sparsity_pattern!(iterator, j)
    iterator.J = [j]
end

function update_mhat_given_col!(iterator, A, FT)
    @inbounds begin
        A1  = A[:, iterator.J̃]
        A1I = A1[iterator.I, :]      

        n₁ = length(iterator.I)
        n₂ = length(iterator.J)
        ñ₂ = length(iterator.J̃)

        push!(iterator.J, iterator.J̃...)
        Atmp = A[:, iterator.J]

        iterator.Ĩ = setdiff(unique(Atmp.rowval), iterator.I)

        A1Ĩ = A1[iterator.Ĩ, :]
    
        ñ₁ = length(iterator.Ĩ)

        B1 = spzeros(n₂, ñ₂)
        mul!(B1, iterator.Q[:,1:n₂]', A1I)
        B2 = iterator.Q[:,n₂+1:end]' * A1I
        B2 = sparse(vcat(B2, A1Ĩ))

        # update_QR_decomposition!(iterator.Q, iterator.R, B1, B2, n₁, n₂, ñ₁, ñ₂)
        F = qr(B2, ordering = false)

        Iₙ₁ = speye(FT, ñ₁)
        Iₙ₂ = speye(FT, n₂)
        hm  = spzeros(n₁, ñ₁)
        iterator.Q = vcat(hcat(iterator.Q, hm), hcat(hm', Iₙ₁))
        hm  = spzeros(ñ₁ + n₁ - n₂, n₂)
        iterator.Q = iterator.Q * vcat(hcat(Iₙ₂, hm'), hcat(hm, F.Q))
        
        hm = spzeros(ñ₂, n₂)
        iterator.R = vcat(hcat(iterator.R, B1), hcat(hm, F.R))    
        
        push!(iterator.I, iterator.Ĩ...)

        bj = zeros(length(iterator.I))
        copyto!(bj, iterator.e[iterator.I])
        minimize!(iterator, bj)
    end
end

function find_mhat_given_col!(iterator, A, n)
    
    A1 = spzeros(n, length(iterator.J))
    copyto!(A1, A[:, iterator.J])
    
    iterator.I = unique(A1.rowval)

    bj = zeros(length(iterator.I))
    copyto!(bj, iterator.e[iterator.I])
    
    F = qr(A1[iterator.I, :], ordering = false)
    iterator.Q = sparse(F.Q)
    iterator.R = sparse(F.R)
    
    minimize!(iterator, bj)
end

function select_residuals!(iterator, A, n, FT)
    ρ = zeros(length(iterator.J̃))
    @inbounds for (t, k) in enumerate(iterator.J̃)
        ek   = speyecolumn(FT, k, n)
        ρ[t] = norm(iterator.r)^2 - norm(iterator.r' * A * ek)^2 / norm(A * ek)^2
    end
    iterator.J̃ = iterator.J̃[ ρ .< mean(ρ) ] 
end

@inline calc_residuals!(i::SpaiIterator, A) = copyto!(i.r, i.e - A[:, i.J] * i.mhat)
@inline minimize!(i::SpaiIterator, bj)      = i.mhat = (i.R \ (i.Q' * bj)[1:length(i.J)])
@inline speye(FT, n) = spdiagm(0=>ones(FT, n))
