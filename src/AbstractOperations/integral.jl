using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!

import Oceananigans.Fields: Field, compute!, location

struct IntegralOperation{LX, LY, LZ, G, T, F, A, D} <: AbstractOperation{LX, LY, LZ, G, T}
    field :: F
    axis :: A
    direction :: D
    grid :: G
    
    function IntegralOperation{LX, LY, LZ}(field::F; axis::A = :z, 
                                direction::D = :+, grid::G = field.grid) where {LX, LY, LZ, F, A, D, G}
        T = eltype(G)

        return new{LX, LY, LZ, G, T, F, A, D}(field, axis, direction, grid)
    end
end

operation_name(::IntegralOperation) = "IntegralOperation"
@inbounds location(integral::IntegralOperation) = location(integral.field)

const IntegralField = Field{<:Any, <:Any, <:Any, <:IntegralOperation}

function Field(integral::IntegralOperation;)
    grid = integral.field.grid
    loc = location(integral.field)
    indices = integral.field.indices

    data = new_data(grid, loc, indices)

    return Field(loc, grid, data, nothing, indices, integral, nothing)
end

@inline integral_layout(::Val{:x}) = :yz
@inline integral_layout(::Val{:y}) = :xz
@inline integral_layout(::Val{:z}) = :xy

@inline integral_location(loc, ::Val{:x}) = loc[1]
@inline integral_location(loc, ::Val{:y}) = loc[2]
@inline integral_location(loc, ::Val{:z}) = loc[3]

function compute!(comp::IntegralField, time=nothing)
    axis = comp.operand.axis

    arch = architecture(comp.grid)
    event = launch!(arch, comp.grid, integral_layout(Val(axis)), _integrate!, 
                comp.data, 
                comp.operand.field, 
                Val(integral_location(location(comp.operand), Val(comp.operand.axis))), 
                Val(comp.operand.axis), 
                Val(comp.operand.direction))

    wait(event)
end

@kernel function _integrate!(data, field, ::Val{Center}, ::Val{:z}, ::Val{:-})
    i, j = @index(Global, NTuple)

    grid = field.grid

    zᶜ = znodes(Center, grid)
    zᶠ = znodes(Face, grid)

    @inbounds data[i, j, grid.Nz] = (zᶜ[grid.Nz] - zᶠ[grid.Nz])*field[i, j, grid.Nz]

    for k in grid.Nz-1:-1:1
        @inbounds data[i, j, k] = data[i, j, k+1] + (zᶜ[k + 1] - zᶠ[k])*field[i, j, k + 1] + (zᶠ[k] - zᶜ[k])*field[i, j, k]
    end
end

∫⁺dx(af::AF{LX, LY, LZ}) where {LX, LY, LZ} = nothing
∫⁻dx(af::AF{LX, LY, LZ}) where {LX, LY, LZ} = nothing
∫⁺dy(af::AF{LX, LY, LZ}) where {LX, LY, LZ} = nothing
∫⁻dy(af::AF{LX, LY, LZ}) where {LX, LY, LZ} = nothing
∫⁺dz(af::AF{LX, LY, LZ}) where {LX, LY, LZ} = nothing

∫⁻dz(af::AF{LX, LY, LZ}) where {LX, LY, LZ} = IntegralOperation{LX, LY, LZ}(af, axis = :z, direction = :-)