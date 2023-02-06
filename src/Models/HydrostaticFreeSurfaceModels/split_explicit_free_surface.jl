using Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures
using Oceananigans.AbstractOperations: Δz, GridMetricOperation
using KernelAbstractions: @index, @kernel
using Adapt

import Oceananigans.TimeSteppers: reset!
import Base.show

"""
    struct SplitExplicitFreeSurface

The split-explicit free surface solver.

$(TYPEDFIELDS)
"""
struct SplitExplicitFreeSurface{𝒩, 𝒮, ℱ, 𝒫 ,ℰ} <: AbstractFreeSurface{𝒩, 𝒫}
    "The instantaneous free surface (`ReducedField`)"
    η :: 𝒩
    "The entire state for the split-explicit (`SplitExplicitState`)"
    state :: 𝒮
    "Parameters for timestepping split-explicit (`NamedTuple`)"
    auxiliary :: ℱ
    "Gravitational acceleration"
    gravitational_acceleration :: 𝒫
    "Settings for the split-explicit scheme (`NamedTuple`)"
    settings :: ℰ
end

# use as a trait for dispatch purposes
"""
    SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth, kwargs...) 

Constructor for the `SplitExplicitFreeSurface`, for more information on the available `kwargs...`
see `SplitExplicitSettings` 
"""
SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth, kwargs...) =
    SplitExplicitFreeSurface(nothing, nothing, nothing,
                             gravitational_acceleration, SplitExplicitSettings(; kwargs...))

# The new constructor is defined later on after the state, settings, auxiliary have been defined
function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid)
    η =  FreeSurfaceDisplacementField(velocities, free_surface, grid)

    return SplitExplicitFreeSurface(η, SplitExplicitState(grid),
                                    SplitExplicitAuxiliary(grid),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

function SplitExplicitFreeSurface(grid; gravitational_acceleration = g_Earth,
                                        settings = SplitExplicitSettings(; substeps = 200))

η = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))

    return SplitExplicitFreeSurface(η,
                                    SplitExplicitState(grid),
                                    SplitExplicitAuxiliary(grid),
                                    gravitational_acceleration,
                                    settings
                                    )
end

"""
    struct SplitExplicitState

A struct containing the state fields for the split-explicit free surface.

$(TYPEDFIELDS)
"""
Base.@kwdef struct SplitExplicitState{𝒞𝒞, ℱ𝒞, 𝒞ℱ}
    "The free surface at times at times `m`, `m-1` and `m-2`. (`ReducedField`)"
    ηᵐ   :: 𝒞𝒞
    ηᵐ⁻¹ :: 𝒞𝒞
    ηᵐ⁻² :: 𝒞𝒞
    "The instantaneous barotropic component of the zonal velocity at times `m`, `m-1` and `m-2`. (`ReducedField`)"
    U    :: ℱ𝒞
    Uᵐ⁻¹ :: ℱ𝒞
    Uᵐ⁻² :: ℱ𝒞
    "The instantaneous barotropic component of the meridional velocity at times `m`, `m-1` and `m-2`. (`ReducedField`)"
    V    :: 𝒞ℱ
    Vᵐ⁻¹ :: 𝒞ℱ
    Vᵐ⁻² :: 𝒞ℱ
    "The time-filtered free surface. (`ReducedField`)"
    η̅    :: 𝒞𝒞
    "The time-filtered barotropic component of the zonal velocity. (`ReducedField`)"
    U̅    :: ℱ𝒞
    "The time-filtered barotropic component of the meridional velocity. (`ReducedField`)"
    V̅    :: 𝒞ℱ    
end

"""
    SplitExplicitState(grid::AbstractGrid)

Return the split-explicit state. Note that `η̅` is solely used for setting the `η`
at the next substep iteration -- it essentially acts as a filter for `η`.
Values at `ᵐ⁻¹` and `ᵐ⁻²` are previous stored time steps to allow using a higher order
time stepping scheme (`AdamsBashforth3Scheme`)
"""
function SplitExplicitState(grid::AbstractGrid)
    η̅ = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))

    ηᵐ   = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
    ηᵐ⁻¹ = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
    ηᵐ⁻² = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
          
    U    = Field((Face, Center, Nothing), grid)
    V    = Field((Center, Face, Nothing), grid)

    Uᵐ⁻¹ = Field((Face, Center, Nothing), grid)
    Vᵐ⁻¹ = Field((Center, Face, Nothing), grid)
          
    Uᵐ⁻² = Field((Face, Center, Nothing), grid)
    Vᵐ⁻² = Field((Center, Face, Nothing), grid)
          
    U̅    = Field((Face, Center, Nothing), grid)
    V̅    = Field((Center, Face, Nothing), grid)
    
    return SplitExplicitState(; ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, Uᵐ⁻¹, Uᵐ⁻², V, Vᵐ⁻¹, Vᵐ⁻², η̅, U̅, V̅)
end

"""
    SplitExplicitAuxiliary

A struct containing auxiliary fields for the split-explicit free surface.

The Barotropic time stepping will be launched on a grid `(kernel_size[1], kernel_size[2])`
large (or `:xy` in case of a serial computation),  and start computing from 
`(i - kernel_offsets[1], j - kernel_offsets[2])`

$(TYPEDFIELDS)
"""
Base.@kwdef struct SplitExplicitAuxiliary{𝒞ℱ, ℱ𝒞, 𝒞𝒞, 𝒦, 𝒪}
    "Vertically integrated slow barotropic forcing function for `U` (`ReducedField`)"
    Gᵁ :: ℱ𝒞
    "Vertically integrated slow barotropic forcing function for `V` (`ReducedField`)"
    Gⱽ :: 𝒞ℱ
    "Depth at `(Face, Center)` (`ReducedField`)"
    Hᶠᶜ :: ℱ𝒞
    "Depth at `(Center, Face)` (`ReducedField`)"
    Hᶜᶠ :: 𝒞ℱ
    "Depth at `(Center, Center)` (`ReducedField`)"
    Hᶜᶜ :: 𝒞𝒞
    "kernel size for barotropic time stepping"
    kernel_size :: 𝒦
    "index offsets for halo calculations"
    kernel_offsets :: 𝒪
end

function SplitExplicitAuxiliary(grid::AbstractGrid)

    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)

    Hᶠᶜ = Field((Face,   Center, Nothing), grid)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid)
    Hᶜᶜ = Field((Center, Center, Nothing), grid)

    dz = GridMetricOperation((Face, Center, Center), Δz, grid)
    sum!(Hᶠᶜ, dz)
   
    dz = GridMetricOperation((Center, Face, Center), Δz, grid)
    sum!(Hᶜᶠ, dz)

    dz = GridMetricOperation((Center, Center, Center), Δz, grid)
    sum!(Hᶜᶜ, dz)

    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ, Hᶜᶜ))

    kernel_size    = :xy
    kernel_offsets = (0, 0)

    return SplitExplicitAuxiliary(; Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, kernel_size, kernel_offsets)
end

"""
    struct SplitExplicitSettings

A struct containing settings for the split-explicit free surface.

$(TYPEDFIELDS)
"""
struct SplitExplicitSettings{𝒩, ℳ, 𝒯, 𝒮}
    "substeps: (`Int`)"
    substeps :: 𝒩
    "averaging_weights : (`Vector`)"
    averaging_weights :: ℳ
    "mass_flux_weights : (`Vector`)"
    mass_flux_weights :: ℳ
    "fractional step: (`Number`), the barotropic time step will be (Δτ ⋅ Δt)" 
    Δτ :: 𝒯
    "time stepping scheme"
    timestepper :: 𝒮
end

"""
    Possible barotropic time-stepping scheme. 

- `AdamsBashforth3Scheme`: η = f(U, Uᵐ⁻¹, Uᵐ⁻²) then U = f(η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
- `ForwardBackwardScheme`: η = f(U)             then U = f(η)
"""
struct AdamsBashforth3Scheme end
struct ForwardBackwardScheme end

# Weights that minimize dispersion error from http://falk.ucsd.edu/roms_class/shchepetkin04.pdf (p = 2, q = 4, r = 0.18927)
@inline function averaging_shape_function(τ; p = 2, q = 4, r = 0.18927) 
    τ₀ = (p + 2) * (p + q + 2) / (p + 1) / (p + q + 1) 
    return (τ / τ₀)^p * (1 - (τ / τ₀)^q) - r * (τ / τ₀)
end

@inline averaging_cosine_function(τ) = τ >= 0.5 && τ <= 1.5 ? 1 + cos(2π * (τ - 1)) : 0.0

@inline averaging_fixed_function(τ) = 1.0

"""
    SplitExplicitSettings(; substeps = 200, 
                            averaging_function = averaging_shape_function,
                            timestepper = ForwardBackwardScheme())

constructor for `SpliExplicitSettings`. The keyword arguments can be provided
directly to the `SplitExplicitFreeSurface` constructor

Keyword Arguments
=================

- `substeps`: The number of substeps that divide the range `(t, t + 2Δt)`. NOTE: not all averaging functions
              require to substep till `2Δt`. The number of substeps will be reduced automatically to the last index
              of `averaging_weights` for which `averaging_weights > 0`
- `averaging_function`: function of `τ` used to average `U` and `η` within the barotropic advancement.
                        `τ` is the fractional substep going from 0 to 2 with the baroclinic time step `t + Δt`
                        located at `τ = 1`. This function should be centered at `τ = 1` (i.e. ∑(aₘ⋅m/M) = 1)
- `timestepper`: Time stepping scheme used, either `ForwardBackwardScheme` or `AdamsBashforth3Scheme`
"""
function SplitExplicitSettings(; substeps = 200, 
                                 averaging_function = averaging_shape_function,
                                 timestepper = ForwardBackwardScheme())

    τᶠ = range(0.0, 2.0, length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]

    averaging_weights = averaging_function.(τᶠ[2:end]) 
    idx = searchsortedlast(averaging_weights, 0.0, rev=true)
    substeps = idx

    averaging_weights = averaging_weights[1:idx]
    mass_flux_weights = similar(averaging_weights)
    
    M = searchsortedfirst(τᶠ, 1.0) - 1

    averaging_weights ./= sum(averaging_weights)

    for i in substeps:-1:1
        mass_flux_weights[i] = 1 / M * sum(averaging_weights[i:substeps]) 
    end

    mass_flux_weights ./= sum(mass_flux_weights)

    return SplitExplicitSettings(substeps,
                                 averaging_weights,
                                 mass_flux_weights, 
                                 Δτ, 
                                 timestepper)
end

# Convenience Functions for grabbing free surface
free_surface(free_surface::SplitExplicitFreeSurface) = free_surface.η

# extend 
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = 0

# convenience functor
function (sefs::SplitExplicitFreeSurface)(settings::SplitExplicitSettings)
    return SplitExplicitFreeSurface(sefs.η, sefs.state, sefs.auxiliary, sefs.gravitational_acceleration, settings)
end

Base.summary(sefs::SplitExplicitFreeSurface) = string("SplitExplicitFreeSurface with $(sefs.settings.substeps) steps")

Base.show(io::IO, sefs::SplitExplicitFreeSurface) = print(io, "$(summary(sefs))\n")

function reset!(sefs::SplitExplicitFreeSurface)
    for name in propertynames(sefs.state)
        var = getproperty(sefs.state, name)
        fill!(var, 0.0)
    end
end

# Adapt
Adapt.adapt_structure(to, free_surface::SplitExplicitFreeSurface) =
    SplitExplicitFreeSurface(Adapt.adapt(to, free_surface.η), nothing, nothing,
                             free_surface.gravitational_acceleration, nothing)
