module Biogeochemistry

using Oceananigans.Grids: Center, xnode, ynode, znode
using Oceananigans.Advection: div_Uc, CenteredSecondOrder
using Oceananigans.Utils: tupleit
using Oceananigans.Fields: Field
using Oceananigans.Architectures: device, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel

import Oceananigans.Fields: location, CenterField
import Oceananigans.Forcings: regularize_forcing, maybe_constant_field

#####
##### Generic fallbacks for biogeochemistry
#####

"""
Update tendencies.

Called at the end of calculate_tendencies!
"""
update_tendencies!(bgc, model) = nothing

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_biogeochemical_state!(bgc, model) = nothing

@inline biogeochemical_drift_velocity(bgc, val_tracer_name) = nothing
@inline biogeochemical_advection_scheme(bgc, val_tracer_name) = nothing
@inline biogeochemical_auxiliary_fieilds(bgc) = NamedTuple()

#####
##### Default (discrete form) biogeochemical source
#####

abstract type AbstractBiogeochemistry end

@inline function biogeochemistry_rhs(i, j, k, grid, bgc, val_tracer_name::Val{tracer_name}, clock, fields) where tracer_name
    U_drift = biogeochemical_drift_velocity(bgc, val_tracer_name)
    scheme = biogeochemical_advection_scheme(bgc, val_tracer_name)
    src = biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields)
    c = @inbounds fields[tracer_name]
        
    return src - div_Uc(i, j, k, grid, scheme, U_drift, c)
end

@inline biogeochemical_transition(i, j, k, grid, bgc, val_tracer_name, clock, fields) =
    bgc(i, j, k, grid, val_tracer_name, clock, fields)

@inline (bgc::AbstractBiogeochemistry)(i, j, k, grid, val_tracer_name, clock, fields) = zero(grid)

#####
##### Continuous form biogeochemical source
#####
 
"""Return the biogeochemical forcing for `val_tracer_name` when model is called."""
abstract type AbstractContinuousFormBiogeochemistry <: AbstractBiogeochemistry end

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{1}) =
    @inbounds (fields[names[1]][i, j, k],)

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{2}) =
    @inbounds (fields[names[1]][i, j, k],
               fields[names[2]][i, j, k])

@inline extract_biogeochemical_fields(i, j, k, grid, fields, names::NTuple{N}) where N =
    @inbounds ntuple(n -> fields[names[n]][i, j, k], Val(N))

@inline function biogeochemical_transition(i, j, k, grid, bgc::AbstractContinuousFormBiogeochemistry,
                                           val_tracer_name, clock, fields)

    names_to_extract = tuple(required_biogeochemical_tracers(bgc)...,
                             required_biogeochemical_auxiliary_fields(bgc)...)

    fields_ijk = extract_biogeochemical_fields(i, j, k, grid, fields, names_to_extract)

    x = xnode(Center(), Center(), Center(), i, j, k, grid)
    y = ynode(Center(), Center(), Center(), i, j, k, grid)
    z = znode(Center(), Center(), Center(), i, j, k, grid)

    return bgc(val_tracer_name, x, y, z, clock.time, fields_ijk...)
end

@inline (bgc::AbstractContinuousFormBiogeochemistry)(val_tracer_name, x, y, z, t, fields...) = zero(x)

struct NoBiogeochemistry <: AbstractBiogeochemistry end

tracernames(tracers) = keys(tracers)
tracernames(tracers::Tuple) = tracers

@inline function has_biogeochemical_tracers(fields::NamedTuple, required_fields, grid)

    for field_name in required_fields
        if !(field_name in keys(fields))
            fields = merge(fields, NamedTuple{(field_name, )}((CenterField(grid), )))
        end
    end

    return fields
end

@inline has_biogeochemical_tracers(user_tracer_names::Tuple, required_names, grid) =
    tuple(user_tracer_names..., required_names...)

"""
    validate_biogeochemistry(tracers, auxiliary_fields, bgc, grid, clock)

Ensure that `tracers` contains biogeochemical tracers and `auxiliary_fields`
contains biogeochemical auxiliary fields (e.g. PAR).
"""
@inline function validate_biogeochemistry(tracers, auxiliary_fields, bgc, grid, clock)
    req_tracers = required_biogeochemical_tracers(bgc)
    tracers = has_biogeochemical_tracers(tracers, req_tracers, grid)
    req_auxiliary_fields = required_biogeochemical_auxiliary_fields(bgc)

    all(field ∈ tracernames(auxiliary_fields) for field in req_auxiliary_fields) ||
        error("$(req_auxiliary_fields) must be among the list of auxiliary fields to use $(typeof(bgc).name.wrapper)")

    # Return tracers and aux fields so that users may overload and
    # define their own special auxiliary fields (e.g. PAR in test)
    return tracers, auxiliary_fields 
end

required_biogeochemical_tracers(::NoBiogeochemistry) = ()
required_biogeochemical_auxiliary_fields(bgc::AbstractBiogeochemistry) = ()

#=
"""
    BasicBiogeochemistry <: AbstractContinuousFormBiogeochemistry

Sets up a tracer based biogeochemical model in a similar way to SeawaterBuoyancy.

Example
=======

@inline growth(x, y, z, t, P, μ₀, λ, m) = (μ₀ * exp(z / λ) - m) * P 

biogeochemistry = Biogeochemistry(tracers = :P, transitions = (; P=growth))
"""
struct BasicBiogeochemistry{T, S, U, A, P} <: AbstractContinuousFormBiogeochemistry
    biogeochemical_tracers :: NTuple{N, Symbol} where N
    transitions :: T
    advection_schemes :: S
    drift_velocities :: U
    auxiliary_fields :: A
    parameters :: P
end

@inline required_biogeochemical_tracers(bgc::BasicBiogeochemistry) = bgc.biogeochemical_tracers
@inline required_biogeochemical_auxiliary_fields(bgc::BasicBiogeochemistry) = bgc.auxiliary_fields

@inline biogeochemical_drift_velocity(bgc::BasicBiogeochemistry, ::Val{tracer_name}) where tracer_name = tracer_name in keys(bgc.drift_velocities) ? bgc.drift_velocities[tracer_name] : 0
@inline biogeochemical_advection_scheme(bgc::BasicBiogeochemistry, ::Val{tracer_name}) where tracer_name = tracer_name in keys(bgc.advection_schemes) ? bgc.advection_schemes[tracer_name] : nothing
@inline biogeochemical_auxiliary_fieilds(bgc::BasicBiogeochemistry) = bgc.auxiliary_fields

@inline (bgc::BasicBiogeochemistry)(::Val{tracer_name}, x, y, z, t, fields_ijk...) where tracer_name = tracer_name in bgc.biogeochemical_tracers ? bgc.transitions[tracer_name](x, y, z, t, fields_ijk..., bgc.parameters...) : 0.0

"""
    maybe_velocity_fields(drift_speeds)

Returns converts a `NamdedTuple` containing the `u`, `v`, and `w` components of velocity 
either as scalars or fields, and returns them all as fields.
"""
function maybe_velocity_fields(drift_speeds)
    drift_velocities = []
    for (u, v, w) in values(drift_speeds) 
        if !all(isa.(values(w), Field))
            u, v, w = maybe_constant_field.((u, v, w))
            push!(drift_velocities, (; u, v, w))
        else
            push!(drift_velocities, w)
        end
    end

    return NamedTuple{keys(drift_speeds)}(drift_velocities)
end


function BasicBiogeochemistry(;tracers,
                                   transitions, 
                                   drift_velocities = NamedTuple(), 
                                   adveciton_schemes = NamedTuple{keys(drift_velocities)}(repeat([CenteredSecondOrder()], 
                                                                                                 length(drift_velocities))),  
                                   auxiliary_fields = NamedTuple(),
                                   parameters = NamedTuple())

    drift_velocities = maybe_velocity_fields(drift_velocities)

    return BasicBiogeochemistry(tupleit(tracers), transitions, adveciton_schemes, drift_velocities, tupleit(auxiliary_fields), parameters)
end
=#

end # module
