using Oceananigans.BoundaryConditions: MCBC, DCBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size, architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: MultiEvent, NoneEvent, @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

struct FieldBoundaryBuffers{W, E, S, N}
    west :: W
    east :: E
   south :: S
   north :: N
end

FieldBoundaryBuffers() = FieldBoundaryBuffers(nothing, nothing, nothing, nothing)
FieldBoundaryBuffers(grid, data, ::Missing) = nothing
FieldBoundaryBuffers(grid, data, ::Nothing) = nothing

function FieldBoundaryBuffers(grid, data, boundary_conditions)

    Hx, Hy, Hz = halo_size(grid)

    west  = create_buffer_x(architecture(grid), data, Hx, boundary_conditions.west)
    east  = create_buffer_x(architecture(grid), data, Hx, boundary_conditions.east)
    south = create_buffer_y(architecture(grid), data, Hy, boundary_conditions.south)
    north = create_buffer_y(architecture(grid), data, Hy, boundary_conditions.north)

    return FieldBoundaryBuffers(west, east, south, north)
end

create_buffer_x(arch, data, H, bc) = nothing
create_buffer_y(arch, data, H, bc) = nothing

using_buffered_communication(arch) = true

const PassingBC = Union{MCBC, DCBC}

function create_buffer_x(arch, data, H, ::PassingBC) 
    if !using_buffered_communication(arch)
        return nothing
    end
    return (send = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), H, size(parent(data), 2), size(parent(data), 3))))    
end

function create_buffer_y(arch, data, H, ::PassingBC)
    if !using_buffered_communication(arch)
        return nothing
    end
    return (send = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))), 
            recv = arch_array(arch, zeros(eltype(data), size(parent(data), 1), H, size(parent(data), 3))))
end

Adapt.adapt_structure(to, buff::FieldBoundaryBuffers) =
    FieldBoundaryBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south))

"""
    fill_send_buffers(c, buffers, arch)

fills `buffers.send` from OffsetArray `c` preparing for message passing. If we are on CPU
we do not need to fill the buffers as the transfer can happen through views
"""
function fill_west_and_east_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing)

    eventwest = fill_west_send_buffer!(parent(c), buffers.west, grid; dependencies)
    eventeast = fill_east_send_buffer!(parent(c), buffers.east, grid; dependencies)

    return MultiEvent((eventeast, eventwest))
end

function fill_south_and_north_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing)

    eventsouth = fill_south_send_buffer!(parent(c), buffers.south, grid; dependencies)
    eventnorth = fill_north_send_buffer!(parent(c), buffers.north, grid; dependencies)
    
    return MultiEvent((eventsouth, eventnorth))
end

fill_west_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_west_send_buffer!(parent(c), buffers.west, grid; dependencies)

fill_east_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_east_send_buffer!(parent(c), buffers.east, grid; dependencies)

fill_south_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_south_send_buffer!(parent(c), buffers.south, grid; dependencies)

fill_north_send_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_north_send_buffer!(parent(c), buffers.north, grid; dependencies)

"""
    fill_recv_buffers(c, buffers, arch)

fills OffsetArray `c` from `buffers.recv` after message passing occurred. If we are on CPU
we do not need to fill the buffers as the transfer can happen through views
"""

function fill_west_and_east_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing)

    eventwest = fill_west_recv_buffer!(parent(c), buffers.west, grid; dependencies)
    eventeast = fill_east_recv_buffer!(parent(c), buffers.east, grid; dependencies)

    return MultiEvent((eventeast, eventwest))
end

function fill_south_and_north_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing)

    eventsouth = fill_south_recv_buffer!(parent(c), buffers.south, grid; dependencies)
    eventnorth = fill_north_recv_buffer!(parent(c), buffers.north, grid; dependencies)
    
    return MultiEvent((eventsouth, eventnorth))
end

fill_west_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_west_send_buffer!(parent(c), buffers.west, grid; dependencies)

fill_east_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_east_send_buffer!(parent(c), buffers.east, grid; dependencies)

fill_south_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_south_send_buffer!(parent(c), buffers.south, grid; dependencies)

fill_north_recv_buffers!(c::OffsetArray, buffers::FieldBoundaryBuffers, grid; dependencies = nothing) = 
    fill_north_send_buffer!(parent(c), buffers.north, grid; dependencies)

# Actual send and recv kernels

 fill_west_send_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()
 fill_east_send_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()
fill_north_send_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()
fill_south_send_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()

 fill_west_send_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :yz,  _fill_west_send_buffer!, c, buff.send, halo_size(grid)[1], size(grid)[1]; dependencies)
 fill_east_send_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :yz,  _fill_east_send_buffer!, c, buff.send, halo_size(grid)[1], size(grid)[1]; dependencies)
fill_north_send_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :xz, _fill_north_send_buffer!, c, buff.send, halo_size(grid)[2], size(grid)[2]; dependencies)
fill_south_send_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :xz, _fill_south_send_buffer!, c, buff.send, halo_size(grid)[2], size(grid)[2]; dependencies)

 fill_west_recv_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()
 fill_east_recv_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()
fill_north_recv_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()
fill_south_recv_buffer!(c, ::Nothing, args...; kwargs...) = NoneEvent()

 fill_west_recv_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :yz,  _fill_west_recv_buffer!, c, buff.recv, halo_size(grid)[1], size(grid)[1]; dependencies)
 fill_east_recv_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :yz,  _fill_east_recv_buffer!, c, buff.recv, halo_size(grid)[1], size(grid)[1]; dependencies)
fill_north_recv_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :xz, _fill_north_recv_buffer!, c, buff.recv, halo_size(grid)[2], size(grid)[2]; dependencies)
fill_south_recv_buffer!(c, buff, grid; dependencies = nothing) = launch!(architecture(grid), grid, :xz, _fill_south_recv_buffer!, c, buff.recv, halo_size(grid)[2], size(grid)[2]; dependencies)

@kernel function _fill_west_send_buffer!(c, buff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        buff[i, j, k] = c[i+H, j, k]
    end
end
@kernel function _fill_east_send_buffer!(c, buff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        buff[i, j, k] = c[i+N, j, k]
    end
end
@kernel function _fill_south_send_buffer!(c, buff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        buff[i, j, k] = c[i, j+H, k]
    end
end
@kernel function _fill_north_send_buffer!(c, buff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        buff[i, j, k] = c[i, j+N, k]
    end
end

@kernel function _fill_west_recv_buffer!(c, buff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
       c[i, j, k] = buff[i, j, k]
    end
end
@kernel function _fill_east_recv_buffer!(c, buff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        c[i+N+H, j, k] = buff[i, j, k]
    end
end
@kernel function _fill_south_recv_buffer!(c, buff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        c[i, j, k] = buff[i, j, k]
    end
end
@kernel function _fill_north_recv_buffer!(c, buff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        c[i, j+N+H, k] = buff[i, j, k]
    end
end