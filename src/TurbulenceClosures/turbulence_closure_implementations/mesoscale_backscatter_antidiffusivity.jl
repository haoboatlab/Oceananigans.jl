using Oceananigans.Operators

struct MesoscaleBackscatterAntiDiffusivity{B, DA, CA} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, HorizontalFormulation}
    backscattered_fraction :: B
    diffusing_advection :: DA
    conserving_advection :: CA
end

const MBAD = MesoscaleBackscatterAntiDiffusivity

DiffusivityFields(grid, tracer_names, bcs, ::MBAD) =  (; νₑ = CenterField(grid))

function calculate_diffusivities!(diffusivity_fields, closure::MBAD, model)
    arch     = model.architecture
    grid     = model.grid
    tracers  = model.tracers

    event = launch!(arch, grid, :xyz,
                    calculate_viscosities!,
                    diffusivity_fields, grid, closure, tracers,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@inline viscosity(::MesoscaleBackscatterAntiDiffusivity, K) = K.νₑ

@inline diffusive_flux_x(i, j, k, grid, cl::MBAD, args...) = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, cl::MBAD, args...) = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, cl::MBAD, args...) = zero(grid)

@inline backscattered_energy(i, j, k, grid,args...) = zero(grid)

@inline backscattered_energy(i, j, k, grid, cl::MBAD, args...) = - ( ℑxᶜᵃᵃ(i, j, k, grid, explicit_backscatter_U, cl, args...) +
                                                                     ℑyᵃᶜᵃ(i, j, k, grid, explicit_backscatter_V, cl, args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any}, Ks, args...) =
          backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any}, Ks, args...) = (
          backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
        + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, Ks, args...) = (
          backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
        + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...) 
        + backscattered_energy(i, j, k, grid, closures[3], Ks[3], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, args...) = (
          backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
        + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...) 
        + backscattered_energy(i, j, k, grid, closures[3], Ks[3], args...) 
        + backscattered_energy(i, j, k, grid, closures[4], Ks[4], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any, <:Any}, Ks, args...) = (
          backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
        + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...) 
        + backscattered_energy(i, j, k, grid, closures[3], Ks[3], args...) 
        + backscattered_energy(i, j, k, grid, closures[4], Ks[4], args...)
        + backscattered_energy(i, j, k, grid, closures[5], Ks[5], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple, Ks, args...) = (
          backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
        + backscattered_energy(i, j, k, grid, closures[2:end], Ks[2:end], args...))

@inline explicit_backscatter_U(i, j, k, grid, cl, K, clock, U, b) = U.u[i, j, k] * ∂ⱼ_τ₁ⱼ(i, j, k, grid, cl, K, clock, U, b)
@inline explicit_backscatter_V(i, j, k, grid, cl, K, clock, U, b) = U.v[i, j, k] * ∂ⱼ_τ₂ⱼ(i, j, k, grid, cl, K, clock, U, b)

@kernel function calculate_viscosities!(diffusivity, grid, cl, tracers)
    i, j, k = @index(Global, NTuple)

    visc = sqrt(Azᶜᶜᶜ(i, j, k, grid) * max(2 * tracers.E[i, j, k], 0.0))
    diffusivity.νₑ[i, j, k] = - cl.backscattered_fraction * visc
end

using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v

@inline implicit_dissipation_U(i, j, k, grid, cl, U) = U.u[i, j, k] * (U_dot_∇u(i, j, k, grid, cl.diffusing_advection, U) - 
                                                                       U_dot_∇u(i, j, k, grid, cl.conserving_advection, U))

@inline implicit_dissipation_V(i, j, k, grid, cl, U) = U.v[i, j, k] * (U_dot_∇v(i, j, k, grid, cl.diffusing_advection, U) - 
                                                                       U_dot_∇v(i, j, k, grid, cl.conserving_advection, U))

@inline implicit_dissipation(i, j, k, grid, cl::MBAD, U) = ℑxᶜᵃᵃ(i, j, k, grid, implicit_dissipation_U, cl, U) +
                                                     ℑyᵃᶜᵃ(i, j, k, grid, implicit_dissipation_V, cl, U)

@inline implicit_dissipation(i, j, k, grid, args...) = zero(grid)

@inline implicit_dissipation(i, j, k, grid, closures::Tuple{<:Any}, args...) =
        implicit_dissipation(i, j, k, grid, closures[1], args...)

@inline implicit_dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any}, args...) = (
        implicit_dissipation(i, j, k, grid, closures[1], args...)
      + implicit_dissipation(i, j, k, grid, closures[2], args...))

@inline implicit_dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, args...) = (
        implicit_dissipation(i, j, k, grid, closures[1], args...)
      + implicit_dissipation(i, j, k, grid, closures[2], args...) 
      + implicit_dissipation(i, j, k, grid, closures[3], args...))

@inline implicit_dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any}, args...) = (
        implicit_dissipation(i, j, k, grid, closures[1], args...)
      + implicit_dissipation(i, j, k, grid, closures[2], args...) 
      + implicit_dissipation(i, j, k, grid, closures[3], args...) 
      + implicit_dissipation(i, j, k, grid, closures[4], args...))

@inline implicit_dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any, <:Any}, args...) = (
        implicit_dissipation(i, j, k, grid, closures[1], args...)
      + implicit_dissipation(i, j, k, grid, closures[2], args...) 
      + implicit_dissipation(i, j, k, grid, closures[3], args...) 
      + implicit_dissipation(i, j, k, grid, closures[4], args...)
      + implicit_dissipation(i, j, k, grid, closures[5], args...))

@inline implicit_dissipation(i, j, k, grid, closures::Tuple, args...) = (
        implicit_dissipation(i, j, k, grid, closures[1], args...)
      + implicit_dissipation(i, j, k, grid, closures[2:end], args...))

@inline function hydrostatic_subgrid_kinetic_energy_tendency(i, j, k, grid,
                                                             val_tracer_index::Val{tracer_index},
                                                             advection,
                                                             closure,
                                                             e_immersed_bc,
                                                             buoyancy,
                                                             backgound_fields,
                                                             velocities,
                                                             tracers,
                                                             auxiliary_fields,
                                                             diffusivities,
                                                             forcing,
                                                             clock) where tracer_index
         
    @inbounds E = tracers[tracer_index]

    model_fields = merge(velocities, tracers, auxiliary_fields)

    return (- div_Uc(i, j, k, grid, advection, velocities, E)
            + implicit_dissipation(i, j, k, grid, closure, velocities)
            - backscattered_energy(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy))
end      
