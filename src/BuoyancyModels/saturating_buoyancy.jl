struct SaturatingBuoyancy{FT} <: AbstractBuoyancyModel{Nothing}
    background_buoyancy_frequency :: FT
end

"""
    SaturatingBuoyancy(; background_buoyancy_frequency)

Returns a buoyancy model with piecewise constant saturtation.
SaturatingBuoyancy requries two tracers, a "dry buoyancy tracer" D
and a "moist buoyancy tracer", M.
"""
SaturatingBuoyancy(; background_buoyancy_frequency) =
    SaturatingBuoyancy(background_buoyancy_frequency)

required_tracers(::SaturatingBuoyancy) = (:D, :M)

Base.nameof(::Type{SaturatingBuoyancy}) = "SaturatingBuoyancy"
Base.summary(b::SaturatingBuoyancy) = "SaturatingBuoyancy"
Base.show(io::IO, b::SaturatingBuoyancy) = print(io, summary(b))

@inline function buoyancy_perturbation(i, j, k, grid, b::SaturatingBuoyancy, C)
    D = @inbounds C.D[i, j, k]
    M = @inbounds C.M[i, j, k]
    z = znode(Center(), Center(), Center(), i, j, k, grid)
    N²ₛ = b.background_buoyancy_frequency
    return max(M, D - N²ₛ * z)
end

#####
##### Buoyancy gradient components
#####

# TODO
