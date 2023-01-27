using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface

import Oceananigans.Models.HydrostaticFreeSurfaceModels: FreeSurface, SplitExplicitAuxiliary

function SplitExplicitAuxiliary(grid::MultiRegionGrid)
    
    Gᵁ = Field{Face,   Center, Nothing}(grid)
    Gⱽ = Field{Center, Face,   Nothing}(grid)
    
    Hᶠᶜ = Field{Face,   Center, Nothing}(grid)
    Hᶜᶠ = Field{Center, Face,   Nothing}(grid)
    Hᶜᶜ = Field{Center, Center, Nothing}(grid)
    
    dz = GridMetricOperation((Face, Center, Center), Δz, grid)
    @apply_regionally sum!(Hᶠᶜ, dz)
       
    dz = GridMetricOperation((Center, Face, Center), Δz, grid)
    @apply_regionally sum!(Hᶜᶠ, dz)
    
    dz = GridMetricOperation((Center, Center, Center), Δz, grid)
    @apply_regionally sum!(Hᶜᶜ, dz)
    
    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ, Hᶜᶜ))

        # In a non-parallel grid we calculate only the interior
    @apply_regionally kernel_size    = augmented_kernel_size(grid, grid.partition)
    @apply_regionally kernel_offsets = full_offsets(grid, grid.partition)
    
    return SplitExplicitAuxiliary(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, kernel_size, kernel_offsets)
end

@inline augmented_kernel_size(grid, ::XPartition) = (size(grid, 1) + 2halo_size(grid)[1]-2, size(grid, 2))
@inline augmented_kernel_size(grid, ::YPartition) = (size(grid, 1), size(grid, 2) + 2halo_size(grid)[2]-2)

@inline full_offsets(grid, ::XPartition) = (halo_size(grid)[1]-1, 0)
@inline full_offsets(grid, ::YPartition) = (0, halo_size(grid)[2]-1)

function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid::MultiRegionGrid)

        settings  = free_surface.settings 

        switch_device!(grid.devices[1])
        old_halos = halo_size(getregion(grid, 1))

        new_halos = partitioned_halos(old_halos, settings.substeps+1, grid.partition)         
        new_grid  = with_halo(new_halos, grid)

        η = ZFaceField(new_grid, indices = (:, :, size(new_grid, 3)+1))

        return SplitExplicitFreeSurface(η,
                                        SplitExplicitState(new_grid),
                                        SplitExplicitAuxiliary(new_grid),
                                        free_surface.gravitational_acceleration,
                                        free_surface.settings)
end

@inline partitioned_halos(old_halos, step_halo, ::XPartition) = (step_halo, old_halos[2], old_halos[3])
@inline partitioned_halos(old_halos, step_halo, ::YPartition) = (old_halos[1], step_halo, old_halos[3])