using Oceananigans.Grids: size, halo_size, topology, Flat

calculate_tendency_contributions!(model, region; kwargs...) = nothing
calculate_boundary_tendency_contributions!(model)           = nothing
update_state_actions!(model, region; kwargs...)             = nothing

"""
calculate_tendencies!(model)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model, fill_halo_events = [NoneEvent()])

    arch = model.architecture

    # We have to add to this the calculate_diffusivities and update hydrostatic pressure

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    if validate_kernel_splitting(model.grid) # Split communication and computation for large 3D simulations (for which N > 2H in every direction) # && !(fill_halo_events isa NoneEvent)

        pre_interior_events = update_state_actions!(model, :interior; dependencies = device_event(arch))
        
        wait(device(arch), MultiEvent(tuple(pre_interior_events...)))
        interior_events     = calculate_tendency_contributions!(model, :interior; dependencies = pre_interior_events[end])
        wait(device(arch), MultiEvent(tuple(interior_events...)))
        
        pre_boundary_events = []
        boundary_events = []

        dependencies    = fill_halo_events[end]

        for region in (:west, :east, :south, :north, :bottom, :top)
            push!(pre_boundary_events, update_state_actions!(model, region; dependencies)...)
            wait(device(arch), MultiEvent(tuple(pre_boundary_event[end]...)))
            push!(boundary_events, calculate_tendency_contributions!(model, region;
                                   dependencies = MultiEvent((pre_boundary_events[end]..., pre_interior_events[end])))...)
            wait(device(arch), MultiEvent(tuple(boundary_events[end]...)))
        end

        wait(device(arch), MultiEvent(tuple(fill_halo_events..., pre_interior_events..., interior_events..., pre_boundary_events..., boundary_events...)))

    else # For 2D computations, not communicating simulations, or domains that have (N < 2H) in at least one direction, launching 1 kernel is enough
        wait(device(arch), MultiEvent(tuple(fill_halo_events...)))

        pre_interior_events = update_state_actions!(model, :allfield; dependencies = device_event(arch))
        interior_events = calculate_tendency_contributions!(model, :allfield; dependencies = pre_interior_events[end])

        wait(device(arch), MultiEvent(tuple(pre_interior_events..., interior_events...)))
    end

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model)

    return nothing
end

@inline function validate_kernel_splitting(grid)

    N = size(grid)
    H = halo_size(grid)

    grid_is_3D           = all(topology(grid) .!= Flat)
    grid_is_large_enough = all(N .- 2 .* H .> 0) 

    return grid_is_3D & grid_is_large_enough
end