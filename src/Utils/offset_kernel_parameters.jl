using Oceananigans.Grids: size, halo_size, topology, Flat

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

## Non-prognostic fields do not fill halo but calculate values on the boundaries directly
## They are also displaced in the interior by one to allow calculation of derivatives
@inline add2(H) = ifelse(H > 0,  2, 0)
@inline add1(H) = ifelse(H > 0,  1, 0)
@inline min1(H) = ifelse(H > 0, -1, 0)

@inline tendency_kernel_size_aux(grid, ::Val{:allfield})   = size(grid) .+ add2.(halo_size(grid))
@inline tendency_kernel_offset_aux(grid, ::Val{:allfield}) = min1.(halo_size(grid))

@inline tendency_kernel_size_aux(grid, ::Val{:interior})   = size(grid) .- 2 .* halo_size(grid)
@inline tendency_kernel_offset_aux(grid, ::Val{:interior}) = halo_size(grid) .- 1

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
