using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid
using Oceananigans.Utils: tendency_kernel_size_aux, tendency_kernel_offset_aux

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbation` downwards:

    `pHY′ = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`
"""
@kernel function _update_hydrostatic_pressure!(pHY′, offsets, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    i′ = i + offsets[1]
    j′ = j + offsets[2]

    @inbounds pHY′[i′, j′, grid.Nz] = - ℑzᵃᵃᶠ(i′, j′, grid.Nz+1, grid, z_dot_g_b, buoyancy, C) * Δzᶜᶜᶠ(i′, j′, grid.Nz+1, grid)

    @unroll for k in grid.Nz-1 : -1 : 1
        @inbounds pHY′[i′, j′, k] = pHY′[i′, j′, k+1] - ℑzᵃᵃᶠ(i′, j′, k+1, grid, z_dot_g_b, buoyancy, C) * Δzᶜᶜᶠ(i′, j′, k+1, grid)
    end
end

update_hydrostatic_pressure!(model.pressures.pHY′, model.architecture, ::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, model.buoyancy, model.tracers) = nothing

# Partial cell "algorithm"
const PCB = PartialCellBottom
const PCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PCB}
update_hydrostatic_pressure!(pHY′, arch, ibg::PCBIBG, buoyancy, tracers) =
    update_hydrostatic_pressure!(pHY′, arch, ibg.underlying_grid, buoyancy, tracers)

function update_hydrostatic_pressure!(pHY′, arch, grid, buoyancy, tracers; region_to_compute, dependencies)

    kernel_size   = tendency_kernel_size_aux(grid, Val(region_to_compute))[[1, 2]]
    kernel_offset = tendency_kernel_offset_aux(grid, Val(region_to_compute))[[1, 2]]

    pressure_event = launch!(arch, grid, kernel_size, _update_hydrostatic_pressure!,
                                   pHY′, kernel_offset, grid, buoyancy, tracers,
                                   dependencies = Event(device(arch)))

    # Fill halo regions for pressure

    return pressure_event
end
