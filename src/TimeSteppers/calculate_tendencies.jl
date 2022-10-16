using Oceananigans.Grids: size, halo_size, topology, Flat

calculate_tendency_contributions!(model, region; kwargs...) = nothing
calculate_boundary_tendency_contributions!(model)           = nothing

"""
calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model, fill_halo_events = [NoneEvent()])

    arch = model.architecture

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    if validate_kernel_splitting(model.grid) # Split communication and computation for large 3D simulations (for which N > 2H in every direction) # && !(fill_halo_events isa NoneEvent)
        interior_events = calculate_tendency_contributions!(model, :interior; dependencies = device_event(arch))
        
        boundary_events = []
        dependencies    = fill_halo_events[end]

        boundary_events = []
        for region in (:west, :east, :south, :north, :bottom, :top)
            push!(boundary_events, calculate_tendency_contributions!(model, region; dependencies)...)
        end

        wait(device(arch), MultiEvent(tuple(fill_halo_events..., interior_events..., boundary_events...)))
    else # For 2D computations, not communicating simulations, or domains that have (N < 2H) in at least one direction, launching 1 kernel is enough
        wait(device(arch), MultiEvent(tuple(fill_halo_events...)))

        interior_events = calculate_tendency_contributions!(model, :allfield; dependencies = device_event(arch))

        wait(device(arch), MultiEvent(tuple(interior_events...)))
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

@inline tendency_kernel_size(grid, ::Val{:allfield}) = size(grid) 
@inline tendency_kernel_size(grid, ::Val{:interior}) = size(grid) .- 2 .* halo_size(grid)

@inline tendency_kernel_offset(grid, ::Val{:allfield}) = (0, 0, 0)
@inline tendency_kernel_offset(grid, ::Val{:interior}) = halo_size(grid)

## The corners and vertical edges are calculated in the x direction
## The horizontal edges in the y direction

@inline tendency_kernel_size(grid, ::Val{:west})   = (halo_size(grid, 1), size(grid, 2), size(grid, 3))
@inline tendency_kernel_size(grid, ::Val{:south})  = (size(grid, 1) - 2*halo_size(grid, 1), halo_size(grid, 2), size(grid, 3))
@inline tendency_kernel_size(grid, ::Val{:bottom}) = (size(grid, 1) - 2*halo_size(grid, 1), size(grid, 2) - 2*halo_size(grid, 2), halo_size(grid, 3))

@inline tendency_kernel_size(grid, ::Val{:east})   = tendency_kernel_size(grid, Val(:west))
@inline tendency_kernel_size(grid, ::Val{:north})  = tendency_kernel_size(grid, Val(:south))
@inline tendency_kernel_size(grid, ::Val{:top})    = tendency_kernel_size(grid, Val(:bottom))

@inline tendency_kernel_offset(grid, ::Val{:west})   = (0, 0, 0)
@inline tendency_kernel_offset(grid, ::Val{:east})   = (size(grid, 1) - halo_size(grid, 1), 0, 0)
@inline tendency_kernel_offset(grid, ::Val{:south})  = (halo_size(grid, 1), 0, 0)
@inline tendency_kernel_offset(grid, ::Val{:north})  = (halo_size(grid, 1), size(grid, 2) - halo_size(grid, 2), 0)
@inline tendency_kernel_offset(grid, ::Val{:bottom}) = (halo_size(grid, 1), halo_size(grid, 2), 0)
@inline tendency_kernel_offset(grid, ::Val{:top})    = (halo_size(grid, 1), halo_size(grid, 2), size(grid, 3) - halo_size(grid, 3))

## Auxiliary fields (such as velocity) do not fill halo but calculate values on the boundaries directly
@inline add2(H) = ifelse(H > 0,  2, 0)
@inline add1(H) = ifelse(H > 0,  1, 0)
@inline min1(H) = ifelse(H > 0, -1, 0)

@inline tendency_kernel_size_aux(grid, ::Val{:allfield}) = size(grid) .+ add2.(halo_size(grid))
@inline tendency_kernel_size_aux(grid, ::Val{:interior}) = size(grid) .- 2 .* halo_size(grid)

@inline tendency_kernel_offset_aux(grid, ::Val{:allfield}) = min1.(halo_size(grid))
@inline tendency_kernel_offset_aux(grid, ::Val{:interior}) = halo_size(grid)

@inline tendency_kernel_size_aux(grid, ::Val{:west})   = @inbounds (halo_size(grid, 1)+add1(halo_size(grid, 1)), size(grid, 2)+add2(halo_size(grid, 2)),      size(grid, 3)+add2(halo_size(grid, 3)))
@inline tendency_kernel_size_aux(grid, ::Val{:south})  = @inbounds (size(grid, 1)-2*halo_size(grid, 1),          halo_size(grid, 2)+add1(halo_size(grid, 2)), size(grid, 3)+add2(halo_size(grid, 3)))
@inline tendency_kernel_size_aux(grid, ::Val{:bottom}) = @inbounds (size(grid, 1)-2*halo_size(grid, 1),          size(grid, 2) - 2*halo_size(grid, 2),        halo_size(grid, 3)+add1(halo_size(grid, 1)))

@inline tendency_kernel_size_aux(grid, ::Val{:east})   = tendency_kernel_size(grid, Val(:west))
@inline tendency_kernel_size_aux(grid, ::Val{:north})  = tendency_kernel_size(grid, Val(:south))
@inline tendency_kernel_size_aux(grid, ::Val{:top})    = tendency_kernel_size(grid, Val(:bottom))

@inline tendency_kernel_offset_aux(grid, ::Val{:west})   = @inbounds min1.(halo_size(grid))
@inline tendency_kernel_offset_aux(grid, ::Val{:east})   = @inbounds (size(grid, 1)-halo_size(grid, 1), min1.(halo_size(grid)[[2, 3]])...)
@inline tendency_kernel_offset_aux(grid, ::Val{:south})  = @inbounds (halo_size(grid, 1),               min1.(halo_size(grid)[[2, 3]])...)
@inline tendency_kernel_offset_aux(grid, ::Val{:north})  = @inbounds (halo_size(grid, 1),               size(grid, 2)-halo_size(grid, 2), min1(halo_size(grid, 3)))
@inline tendency_kernel_offset_aux(grid, ::Val{:bottom}) = @inbounds (halo_size(grid, 1),               halo_size(grid, 2),               min1(halo_size(grid, 3)))
@inline tendency_kernel_offset_aux(grid, ::Val{:top})    = @inbounds (halo_size(grid, 1),               halo_size(grid, 2),               size(grid, 3)-halo_size(grid, 3))
