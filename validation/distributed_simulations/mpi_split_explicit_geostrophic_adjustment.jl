using MPI
using Oceananigans
using Oceananigans.Distributed
using Oceananigans.Units: kilometers, meters

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

topology = (Periodic, Periodic, Bounded)
arch = MultiArch(CPU(); topology, ranks=(Nranks, 1, 1))

Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(arch,
                       size = (80, 3, 1),
                       x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
                       topology = topology)

coriolis = FPlane(f = 1e-4)

free_surface = SplitExplicitFreeSurface(; substeps = 10)

model = HydrostaticFreeSurfaceModel(; grid,
                                      coriolis = coriolis,
                                      free_surface = free_surface)

gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = grid.Lx / 40 # gaussian width
x₀ = grid.Lx / 4 # gaussian center

vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.gravitational_acceleration
η = model.free_surface.η

η₀ = coriolis.f * U * L / g # geostrophic free surface amplitude

ηᵍ(x) = η₀ * gaussian(x - x₀, L)

ηⁱ(x, y) = 2 * ηᵍ(x)

set!(model, v = vᵍ, η = ηⁱ)

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
simulation = Simulation(model, Δt = 2wave_propagation_time_scale, stop_iteration = 300)

outputs = Dict()

outputs[:u] = Field(model.velocities.u; indices)
outputs[:v] = Field(model.velocities.v; indices)
outputs[:w] = Field(model.velocities.w; indices)
outputs[:η] = model.free_surface.η

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = IterationInterval(1),
                                                      filename = "test_output_writing_rank$rank",
                                                      overwrite_existing = true)

run!(simulation)

MPI.Finalize()