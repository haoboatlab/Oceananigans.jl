# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its turbulent decay
# in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no tracers and no buoyancy model.
#   * How to use computed `Field`s to generate output.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# ## Model setup

# We instantiate the model with an isotropic diffusivity. We use a grid with 128² points,
# a fifth-order advection scheme, third-order Runge-Kutta time-stepping,
# and a small isotropic viscosity.  Note that we assign `Flat` to the `z` direction.

using Oceananigans

grid_base = RectilinearGrid(GPU(), size=(128, 128, 1), extent=(2π, 2π), topology=(Periodic, Periodic, Bounded))

grid  = MultiRegionGrid(grid_base, devices = (0, 1))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            closure = ScalarDiffusivity(ν=1e-5))

# ## Random initial conditions
#
# Our initial condition randomizes `model.velocities.u` and `model.velocities.v`.
# We ensure that both have zero mean for aesthetic reasons.

using Statistics

u, v, w = model.velocities

uᵢ = rand(128, 128, 1)
vᵢ = rand(128, 128, 1)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)

using Oceananigans.MultiRegion: multi_region_object_from_array

u₀ = multi_region_object_from_array(uᵢ, grid)
v₀ = multi_region_object_from_array(vᵢ, grid)

set!(model, u=u₀, v=v₀)

simulation = Simulation(model, Δt=0.2, stop_time=50)

# ## Logging simulation progress
#
# We set up a callback that logs the simulation iteration and time every 100 iterations.

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ## Output
#
# We set up an output writer for the simulation that saves vorticity and speed every 20 iterations.
#
# ### Computing vorticity and speed
#
# To make our equations prettier, we unpack `u`, `v`, and `w` from
# the `NamedTuple` model.velocities:
u, v, w = model.velocities

# Next we create two `Field`s that calculate
# _(i)_ vorticity that measures the rate at which the fluid rotates
# and is defined as
#
# ```math
# ω = ∂_x v - ∂_y u \, ,
# ```

ω = ∂x(v) - ∂y(u)

# We also calculate _(ii)_ the _speed_ of the flow,
#
# ```math
# s = \sqrt{u^2 + v^2} \, .
# ```

s = sqrt(u^2 + v^2)

# We pass these operations to an output writer below to calculate and output them during the simulation.
filename = "two_dimensional_turbulence"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s),
                                                      schedule = TimeInterval(0.6),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

# ## Running the simulation
#
# Pretty much just

run!(simulation)

# ## Visualizing the results
#
# We load the output.

# Construct the ``x, y, z`` grid for plotting purposes,
using Oceananigans.MultiRegion: reconstruct_global_field

ω_global = reconstruct_global_field(Field(ω))
s_global = reconstruct_global_field(Field(s))

xω, yω, zω = nodes(ω_global)
xs, ys, zs = nodes(s_global)
nothing # hide

ω_timeseries = []
s_timeseries = []

file = jldopen(filename * ".jld2")

iterations = keys(file["timeseries/t"])

for iter in iterations
    push!(ω_timeseries, file["timeseries/ω/" * iter])
    push!(s_timeseries, file["timeseries/s/" * iter])
end

# and animate the vorticity and fluid speed.

using CairoMakie
set_theme!(Theme(fontsize = 24))

@info "Making a neat movie of vorticity and speed..."

fig = Figure(resolution = (800, 500))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               limits = ((0, 2π), (0, 2π)),
               aspect = AxisAspect(1))

ax_ω = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
ax_s = Axis(fig[2, 2]; title = "Speed", axis_kwargs...)
nothing #hide

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

# Now let's plot the vorticity and speed.

ω = @lift ω_timeseries[$n][:, :, 1]
s = @lift s_timeseries[$n][:, :, 1]

heatmap!(ax_ω, xω, yω, ω; colormap = :balance, colorrange = (-2, 2))
heatmap!(ax_s, xs, ys, s; colormap = :speed, colorrange = (0, 0.2))

# title = @lift "t = " * string(round(times[$n], digits=2))
# Label(fig[1, 1:2], title, textsize=24, tellwidth=false)

# Finally, we record a movie.

frames = 1:length(s_timeseries)

@info "Making a neat animation of vorticity and speed..."

record(fig, filename * ".mp4", frames, framerate=24) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
