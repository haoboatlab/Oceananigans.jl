using Printf
using Statistics
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
# using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, solid_node, solid_interface
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, inactive_node, peripheral_node
# solid_node is inactive_node and solid_interface is peripheral_node
using Oceananigans.Operators: Δzᵃᵃᶜ
#using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

using Random
Random.seed!(1234)

arch = GPU()
with_ridge = false

filename = "eddying_channel_with_beta_test"

# Domain
const Lx = 4000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = 3kilometers    # depth [m]

# number of grid points
Nx = 200
Ny = 100
Nz = 15

save_fields_interval = 50years
stop_time = 20years + 1day
Δt₀ = 7.5minutes * 2

grid = RectilinearGrid(arch;
    topology=(Periodic, Bounded, Bounded),
    size=(Nx, Ny, Nz),
    halo=(4, 4, 4),
    x=(0, Lx),
    y=(0, Ly),
    z=(-Lz, 0)
)

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α = 2e-4     # [K⁻¹] thermal expansion coefficient
g = 9.8061   # [m s⁻²] gravitational constant
cᵖ = 3994.0   # [J K⁻¹] heat capacity
ρ = 1024.0   # [kg m⁻³] reference density

# Different bottom drags depending on with or without ridge 
# accounting for form drag
if with_ridge
    μ = 2e-3
else
    μ = 1e-2
end

parameters = (Ly=Ly,
    Lz=Lz,
    Qᵇ=10 / (ρ * cᵖ) * α * g,         # buoyancy flux magnitude [m² s⁻³]
    y_shutoff=5 / 6 * Ly,             # shutoff location for buoyancy flux [m]
    τ=0.15 / ρ,                       # surface kinematic wind stress [m² s⁻²]
    μ=μ,                            # quadratic bottom drag coefficient []
    ΔB=8 * α * g,                     # surface vertical buoyancy gradient [s⁻²]
    H=Lz,                             # domain depth [m]
    h=1000.0,                         # exponential decay scale of stable stratification [m]
    y_sponge=19 / 20 * Ly,            # southern boundary of sponge layer [m]
    λt=7days,                         # relaxation time scale for the northen sponge [s]
    λs=2e-4,                          # relaxation time scale for the surface [s]
)

@inline relaxation_profile(y, p) = p.ΔB * (y / p.Ly)
@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return @inbounds p.λs * (model_fields.b[i, j, grid.Nz] - relaxation_profile(y, p))
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=parameters)


@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return -p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

if with_ridge
    @inline is_immersed_drag_u(i, j, k, grid) = Int(peripheral_node(Face(), Center(), Center(), i, j, k - 1, grid) & !inactive_node(Face(), Center(), Center(), i, j, k, grid))
    @inline is_immersed_drag_v(i, j, k, grid) = Int(peripheral_node(Center(), Face(), Center(), i, j, k - 1, grid) & !inactive_node(Center(), Face(), Center(), i, j, k, grid))

    # Keep a constant linear drag parameter independent on vertical level
    @inline u_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * sqrt(model_fields.u[i, j, k]^2 + model_fields.v[i, j, k]^2) * is_immersed_drag_u(i, j, k, grid) * fields.u[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)
    @inline v_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * sqrt(model_fields.u[i, j, k]^2 + model_fields.v[i, j, k]^2) * is_immersed_drag_v(i, j, k, grid) * fields.v[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)

    Fu = Forcing(u_drag, discrete_form=true, parameters=parameters)
    Fv = Forcing(v_drag, discrete_form=true, parameters=parameters)

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

    u_bcs = FieldBoundaryConditions(top=u_stress_bc, bottom=u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom=v_drag_bc)

else
    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * model_fields.u[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * model_fields.v[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)
    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)
    u_bcs = FieldBoundaryConditions(top=u_stress_bc, bottom=u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom=v_drag_bc)
end

b_bcs = FieldBoundaryConditions(top=buoyancy_flux_bc)

#####
##### Coriolis
#####

const f = -1e-4     # [s⁻¹]
const β = 1e-11    # [m⁻¹ s⁻¹]
coriolis = BetaPlane(f₀=f, β=β)

#####
##### Forcing and initial condition
#####
@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

@inline function buoyancy_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]
    return -1 / timescale * mask(y, p) * (b - target_b)
end

Fb = Forcing(buoyancy_sponge, discrete_form=true, parameters=parameters)

ridge(x, y) = 1e3 * exp(-(x - 1e6)^2 / (2e5)^2) - 3e3
@inline function u_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = 30 * 60 # 30 minutes
    x = xnode(Face(), i, grid)
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    u = @inbounds model_fields.u[i, j, k]
    mm = ridge(x, y) > z # e.g. if (x = 1e6, y = 0, z = -3000) this returns true
    return -1 / timescale * mm * u
end

@inline function v_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = 30 * 60 # 30 minutes
    x = xnode(Center(), i, grid)
    y = ynode(Face(), j, grid)
    z = znode(Center(), k, grid)
    v = @inbounds model_fields.v[i, j, k]
    mm = ridge(x, y) > z # e.g. if (x = 1e6, y = 0, z = 0) this returns false
    return -1 / timescale * mm * v
end

Fu_sponge = Forcing(u_sponge, discrete_form=true, parameters=parameters)
Fv_sponge = Forcing(v_sponge, discrete_form=true, parameters=parameters)



# Turbulence closures

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity

vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νz, κ=κz)

horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν=νh, κ=κh)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1.0,
    convective_νz=0.0)

# catke = CATKEVerticalDiffusivity()

#####
##### Model building
#####

@info "Building a model..."

ridge(x, y) = 1e3*exp(-(x-1e6)^2/(2e5)^2) - 3e3 


if with_ridge
    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(ridge))
    grid = immersed_grid
    boundary_conditions = (; b=b_bcs, u=u_bcs)
    forcings = (; b=Fb, u=Fu, v=Fv)
else
    boundary_conditions = (; b=b_bcs, u=u_bcs, v=v_bcs)
    forcings = (; b=Fb, u = Fu_sponge, v = Fv_sponge)
end

free_surface = ImplicitFreeSurface()
# free_surface = ImplicitFreeSurface(preconditioner_method = :SparseInverse, preconditioner_settings = (ε = 0.01, nzrel = 10))

model = HydrostaticFreeSurfaceModel(grid=grid,
    free_surface=free_surface,
    momentum_advection=WENO(),
    tracer_advection=WENO(),
    buoyancy=BuoyancyTracer(),
    coriolis=coriolis,
    closure=(horizontal_diffusive_closure, vertical_diffusive_closure, convective_adjustment),
    tracers=(:b, :c, :e),
    boundary_conditions=(b=b_bcs, u=u_bcs, v=v_bcs),
    forcing=(; b=Fb))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)
uᵢ(x, y, z) = ε(1e-8)
vᵢ(x, y, z) = ε(1e-8)
wᵢ(x, y, z) = ε(1e-8)

Δy = 100kilometers
Δz = 100
Δc = 2Δy
cᵢ(x, y, z) = exp(-(y - Ly / 2)^2 / 2Δc^2) * exp(-(z + Lz / 4)^2 / 2Δz^2)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ, w=wᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=20minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.v),
        maximum(abs, sim.model.velocities.w),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))


#####
##### Diagnostics
#####
#=
u, v, w = model.velocities
b, c = model.tracers.b, model.tracers.c
η = model.free_surface.η

ζ = Field(∂x(v) - ∂y(u))

B = Field(Average(b, dims=1))
C = Field(Average(c, dims=1))
U = Field(Average(u, dims=1))
η̄ = Field(Average(η, dims=1))
V = Field(Average(v, dims=1))
W = Field(Average(w, dims=1))

uu_op = @at (Center, Center, Center) u * u
vv_op = @at (Center, Center, Center) v * v
ww_op = @at (Center, Center, Center) w * w

uv_op = @at (Center, Center, Center) u * v
vw_op = @at (Center, Center, Center) v * w
uw_op = @at (Center, Center, Center) u * w

uu = Field(Average(uu_op, dims=1))
vv = Field(Average(vv_op, dims=1))
ww = Field(Average(ww_op, dims=1))

uv = Field(Average(uv_op, dims=1))
vw = Field(Average(vw_op, dims=1))
uw = Field(Average(uw_op, dims=1))

bb = Field(Average(b * b, dims=1))
vb = Field(Average(b * v, dims=1))
wb = Field(Average(b * w, dims=1))

cc = Field(Average(c * c, dims=1))
vc = Field(Average(c * v, dims=1))
wc = Field(Average(c * w, dims=1))

outputs = (; b, c, ζ, u, v, w)

zonally_averaged_outputs = (b=B, u=U, v=V, w=W, c=C, η=η̄, uu=uu, vv=vv, ww=ww, uv=uv, vw=vw, uw=uw, bb=bb, vb=vb, wb=wb, cc=cc, vc=vc, wc=wc)
#=
 vb=v′b′, wb=w′b′, vc=v′c′, wc=w′c′, bb=b′b′,
    tke=tke, uv=u′v′, vw=v′w′, uw=u′w′, cc=c′c′
=#
#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
    schedule=TimeInterval(10years),
    prefix=filename,
    force=true)

slicers = (west=(1, :, :),
    east=(grid.Nx, :, :),
    south=(:, 1, :),
    north=(:, grid.Ny, :),
    bottom=(:, :, 1),
    top=(:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, outputs;
        schedule=TimeInterval(save_fields_interval),
        indices,
        prefix=filename * "_$(side)_slice",
        force=true)
end


simulation.output_writers[:zonal] = JLD2OutputWriter(model, zonally_averaged_outputs,
    schedule=AveragedTimeInterval(10 * 365days, window=10 * 365days, stride=10),
    prefix=filename * "_zonal_time_average",
    force=true)

simulation.output_writers[:averaged_stats_nc] =
    NetCDFOutputWriter(model, zonally_averaged_outputs,
        mode="c",
        filepath=filename * "_zonal_time_averaged_statistics.nc",
        schedule=AveragedTimeInterval(10 * 365days, window=10 * 365days, stride=10),
        )
=#

#=
simulation.output_writers[:zonal] = JLD2OutputWriter(model, zonally_averaged_outputs;
                                                     schedule = TimeInterval(save_fields_interval),
                                                     prefix = filename * "_zonal_average",
                                                     force = true)
simulation.output_writers[:zonal] = JLD2OutputWriter(model, zonally_averaged_outputs,
                                                     schedule = AveragedTimeInterval(60days),
                                                     prefix = filename * "_zonal_time_average",
                                                     force = true)
simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
                                                        schedule = AveragedTimeInterval(1days, window=1days, stride=1),
                                                        prefix = "eddying_channel_averages",
                                                        verbose = true,
                                                        force = true)
=#

@info "Running the simulation..."

run!(simulation, pickup=false)

#=
# simulation.stop_time += 61days
# run!(simulation, pickup=true)
=#

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
