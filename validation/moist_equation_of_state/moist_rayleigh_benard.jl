using Oceananigans
using Oceananigans.Units
using Oceananigans.BuoyancyModels: SaturatingBuoyancy
using GLMakie

Nx = Nz = 64
Lx = Ly = 1kilometer
H = 500meters

grid = RectilinearGrid(size = (Nx, 1, Nz),
                      x = (0, Lx),
                      y = (0, Ly),
                      z = (0, H),
                      topology = (Periodic, Periodic, Bounded))

N²ₛ = 1e-4
buoyancy = SaturatingBuoyancy(background_buoyancy_frequency=N²ₛ)
closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
coriolis = FPlane(f=1e-4)

Dᴴ = 0.5 * N²ₛ * H

Dᴮ(z) = + Dᴴ / H * z
Mᴮ(z) = - Dᴴ / H * z
Mᴴ = Mᴮ(H)

# Necessary condition for instability: Dᴴ < N² * H
@show Dᴴ 
@show N²ₛ * H

D_bottom_bc = ValueBoundaryCondition(0.0)
M_bottom_bc = ValueBoundaryCondition(0.0)
D_top_bc = ValueBoundaryCondition(Dᴴ)
M_top_bc = ValueBoundaryCondition(Mᴴ)

D_bcs = FieldBoundaryConditions(top = D_top_bc, bottom = D_bottom_bc)
M_bcs = FieldBoundaryConditions(top = M_top_bc, bottom = M_bottom_bc)

model = NonhydrostaticModel(; grid,
                            closure,
                            buoyancy,
                            tracers = (:D, :M))


ϵ(z) = 1e-3 * randn() * z * (z - grid.Nz)
Dᵢ(x, y, z) = Dᴮ(z) * (1 + ϵ(z))
Mᵢ(x, y, z) = Mᴮ(z) * (1 + ϵ(z))

set!(model, D=Dᵢ, M=Mᵢ)

simulation = Simulation(model, Δt=1e-3, stop_iteration=100)

run!(simulation)

u, v, w = model.velocities
fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, interior(w, :, 1, :))

display(fig)
