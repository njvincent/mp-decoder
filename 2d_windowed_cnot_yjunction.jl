"""
Moving Y-junction CNOT decoder.

This standalone prototype implements one ideal synchronous X-sector CNOT.
Two labeled pre-gate decoder histories meet one unlabeled post-gate target
history at a moving junction.  Field messages traverse the Y graph in both
directions, while every defect remains in exactly one lane.  Once the junction
reaches the finite back wall, the two pre-gate lanes XOR-collapse into the
post-gate lane and are released.
"""

using Random
using Base.Threads

const Y_CONTROL_BLOCK = 1
const Y_TARGET_BLOCK = 2

const Y_PRE_CONTROL = UInt8(1)
const Y_PRE_TARGET = UInt8(2)
const Y_POST_TARGET = UInt8(3)
const Y_NO_BRANCH = UInt8(0)

function nonzeromin(a,b)
    if a == 0
        return b
    elseif b == 0
        return a
    end
    return min(a,b)
end

nonzeromin(a,b,c) = nonzeromin(nonzeromin(a,b),c)

function get_synds(state)
    L = size(state,1)
    ind(i) = mod1(i,L)
    synds = falses(L,L)
    for i in 1:L, j in 1:L
        synds[i,j] = state[i,j,1] ⊻ state[i,j,2] ⊻
                      state[ind(i-1),j,1] ⊻ state[i,ind(j-1),2]
    end
    return synds
end

function detect_logical_error(state)
    Lx,Ly,_ = size(state)
    xparity = false
    yparity = false
    for i in 1:Lx
        xparity ⊻= state[i,1,2]
    end
    for j in 1:Ly
        yparity ⊻= state[1,j,1]
    end
    return !xparity && !yparity
end

mutable struct YPhysicalBlock
    block::Int
    errors::BitArray{3}
    frame::BitArray{3}
    old_synds::BitArray{2}
    new_synds::BitArray{2}
    noise_rounds::Int
    measurement_rounds::Int
end

mutable struct DecoderLane
    hist::BitArray{3}
    fields::Array{Int,5}
    new_fields::Array{Int,5}
    proposals::BitArray{4}
end

mutable struct TargetYJunction
    pre_control::DecoderLane
    pre_target::DecoderLane
    post_target::DecoderLane
    junction_depth::Int
    branch_temporal_costs::Array{Int,3}
    junction_proposals::Matrix{UInt8}
end

mutable struct YJunctionCNOTState
    blocks::Vector{YPhysicalBlock}
    control_history::DecoderLane
    target_decoder::Union{DecoderLane,TargetYJunction}
    cnot_applied::Bool
    rounds::Int
    cnot_round::Int
    collapse_round::Int
    control_branch_crossings::Int
    target_branch_crossings::Int
    equal_branch_ties::Int
    max_target_lane_count::Int
end

struct YJunctionRoundMasks
    data::NTuple{2,BitArray{3}}
    measurement::NTuple{2,BitArray{2}}
end

function YJunctionRoundMasks(
    control_data::AbstractArray{Bool,3},
    control_measurement::AbstractArray{Bool,2},
    target_data::AbstractArray{Bool,3},
    target_measurement::AbstractArray{Bool,2},
)
    return YJunctionRoundMasks(
        (BitArray(control_data),BitArray(target_data)),
        (BitArray(control_measurement),BitArray(target_measurement)),
    )
end

function make_yphysical_block(block,L)
    return YPhysicalBlock(
        block,
        falses(L,L,2),
        falses(L,L,2),
        falses(L,L),
        falses(L,L),
        0,
        0,
    )
end

function make_decoder_lane(L,Z)
    return DecoderLane(
        falses(L,L,Z),
        zeros(Int,L,L,Z,3,2),
        zeros(Int,L,L,Z,3,2),
        falses(L,L,Z,3),
    )
end

function snapshot_decoder_lane(lane::DecoderLane)
    L,_,Z = size(lane.hist)
    copied = make_decoder_lane(L,Z)
    copied.hist .= lane.hist
    copied.fields .= lane.fields
    return copied
end

function initial_yjunction_state(L,Z)
    if L < 2
        error("Y-junction CNOT requires L >= 2")
    elseif Z < 2
        error("Y-junction CNOT requires Z >= 2 for a distinct back wall")
    end
    return YJunctionCNOTState(
        YPhysicalBlock[
            make_yphysical_block(Y_CONTROL_BLOCK,L),
            make_yphysical_block(Y_TARGET_BLOCK,L),
        ],
        make_decoder_lane(L,Z),
        make_decoder_lane(L,Z),
        false,
        0,
        -1,
        -1,
        0,
        0,
        0,
        1,
    )
end

function sample_yjunction_channel_mask(rng,dims,probability)
    if !(0 <= probability < 0.5)
        error("decoder probabilities must satisfy 0 <= probability < 0.5")
    elseif probability == 0
        return falses(dims...)
    end
    return BitArray(rand(rng,dims...) .< probability)
end

function sample_yjunction_round_masks(rng,L,p,q)
    return YJunctionRoundMasks(
        sample_yjunction_channel_mask(rng,(L,L,2),p),
        sample_yjunction_channel_mask(rng,(L,L),q),
        sample_yjunction_channel_mask(rng,(L,L,2),p),
        sample_yjunction_channel_mask(rng,(L,L),q),
    )
end

function validate_yjunction_round_masks(masks::YJunctionRoundMasks,L)
    for block in (Y_CONTROL_BLOCK,Y_TARGET_BLOCK)
        if size(masks.data[block]) != (L,L,2)
            error("block $block data mask must have shape ($(L),$(L),2)")
        elseif size(masks.measurement[block]) != (L,L)
            error("block $block measurement mask must have shape ($(L),$(L))")
        end
    end
    return masks
end

# Baseline synchronous decoder kernel.

function lane_site_field_values(i,j,k,fields,hist)
    values = zeros(Int,3,2)
    L,_,Z = size(hist)
    ind(index) = mod1(index,L)
    zind(index) = clamp(index,1,Z)

    for axis in 1:3, step in (-1,1)
        sign_index = step == 1 ? 1 : 2
        best = typemax(Int)
        for delta1 in -1:1, delta2 in -1:1
            if axis == 1
                ip = ind(i-step)
                jp = ind(j+delta1)
                kp = zind(k+delta2)
            elseif axis == 2
                ip = ind(i+delta1)
                jp = ind(j-step)
                kp = zind(k+delta2)
            else
                ip = ind(i+delta1)
                jp = ind(j+delta2)
                kp = zind(k-step)
            end
            distance = 1 + abs(delta1) + abs(delta2)
            if hist[ip,jp,kp]
                best = min(best,distance)
            end
            incoming = fields[ip,jp,kp,axis,sign_index]
            if incoming != 0
                best = min(best,incoming + distance)
            end
        end
        values[axis,sign_index] = best == typemax(Int) ? 0 : best
    end
    return values
end

function update_lane_fields!(lane::DecoderLane)
    L,_,Z = size(lane.hist)
    for i in 1:L, j in 1:L, k in 1:Z
        lane.new_fields[i,j,k,:,:] .= lane_site_field_values(
            i,j,k,lane.fields,lane.hist,
        )
    end
    lane.fields .= lane.new_fields
    return nothing
end

function perform_correction!(hist,proposals)
    L,_,Z = size(hist)
    ind(index) = mod1(index,L)
    for i in 1:L, j in 1:L, k in 1:Z
        if proposals[i,j,k,1]
            hist[i,j,k] ⊻= true
            hist[ind(i+1),j,k] ⊻= true
        end
        if proposals[i,j,k,2]
            hist[i,j,k] ⊻= true
            hist[i,ind(j+1),k] ⊻= true
        end
        if proposals[i,j,k,3]
            hist[i,j,k] ⊻= true
            hist[i,j,k+1] ⊻= true
        end
    end
    return nothing
end

function select_lane_proposals!(
    lane::DecoderLane;decoder_rng=Random.default_rng(),
)
    L,_,Z = size(lane.hist)
    ind(index) = mod1(index,L)
    lane.proposals .= false
    for i in 1:L, j in 1:L, k in 1:Z
        if !lane.hist[i,j,k]
            continue
        end
        if k < Z
            if !any(!iszero,@view lane.fields[i,j,k,:,:])
                continue
            end
            mindist = minimum(lane.fields[i,j,k,:,:][lane.fields[i,j,k,:,:] .> 0])
            if lane.fields[i,j,k,3,2] == mindist
                lane.proposals[i,j,k,3] = true
            elseif lane.fields[i,j,k,1,1] == mindist
                lane.proposals[ind(i-1),j,k,1] = true
            elseif lane.fields[i,j,k,2,1] == mindist
                lane.proposals[i,ind(j-1),k,2] = true
            elseif lane.fields[i,j,k,2,2] == mindist
                lane.proposals[i,j,k,2] = true
            elseif lane.fields[i,j,k,1,2] == mindist
                lane.proposals[i,j,k,1] = true
            end
        elseif any(!iszero,@view lane.fields[i,j,k,1:2,:]) &&
               rand(decoder_rng) < 0.8
            mindist = minimum(lane.fields[i,j,k,1:2,:][lane.fields[i,j,k,1:2,:] .> 0])
            if lane.fields[i,j,k,1,1] == mindist
                lane.proposals[ind(i-1),j,k,1] = true
            elseif lane.fields[i,j,k,2,1] == mindist
                lane.proposals[i,ind(j-1),k,2] = true
            elseif lane.fields[i,j,k,2,2] == mindist
                lane.proposals[i,j,k,2] = true
            elseif lane.fields[i,j,k,1,2] == mindist
                lane.proposals[i,j,k,1] = true
            end
        end
    end
    return nothing
end

function commit_lane_proposals!(lane::DecoderLane,frame)
    L,_,_ = size(lane.hist)
    for i in 1:L, j in 1:L, axis in 1:2
        frame[i,j,axis] ⊻= reduce(⊻,@view lane.proposals[i,j,:,axis])
    end
    perform_correction!(lane.hist,lane.proposals)
    return nothing
end

function rg_cycle!(lane::DecoderLane)
    _,_,Z = size(lane.hist)
    lane.hist[:,:,Z] .= xor.(lane.hist[:,:,Z],lane.hist[:,:,Z-1])
    if Z > 2
        copyto!(@view(lane.hist[:,:,2:end-1]),@view(lane.hist[:,:,1:end-2]))
    end
    lane.hist[:,:,1] .= false

    lane.fields[:,:,Z,1:2,:] .= nonzeromin.(
        lane.fields[:,:,Z-1,1:2,:],lane.fields[:,:,Z,1:2,:],
    )
    if Z > 2
        copyto!(
            @view(lane.fields[:,:,2:end-1,:,:]),
            @view(lane.fields[:,:,1:end-2,:,:]),
        )
    end
    lane.fields[:,:,1,:,:] .= 0
    lane.new_fields .= 0
    lane.proposals .= false
    return nothing
end

function apply_block_channel!(
    block::YPhysicalBlock,data_mask,measurement_mask,
)
    block.errors .⊻= data_mask
    block.old_synds .= block.new_synds
    block.new_synds .= get_synds(block.errors)
    block.new_synds .⊻= measurement_mask
    block.noise_rounds += 1
    block.measurement_rounds += 1
    return block.old_synds .⊻ block.new_synds
end

function update_live_lane!(
    block::YPhysicalBlock,lane::DecoderLane,r,data_mask,measurement_mask;
    decoder_rng=Random.default_rng(),
)
    if r < 1
        error("Y-junction CNOT requires r >= 1")
    end
    for _ in 1:r
        update_lane_fields!(lane)
    end
    select_lane_proposals!(lane;decoder_rng=decoder_rng)
    commit_lane_proposals!(lane,block.frame)
    new_event = apply_block_channel!(block,data_mask,measurement_mask)
    rg_cycle!(lane)
    lane.hist[:,:,1] .= new_event
    return nothing
end

# Y-junction field topology.

function y_lane(junction::TargetYJunction,lane_id::UInt8)
    if lane_id == Y_PRE_CONTROL
        return junction.pre_control
    elseif lane_id == Y_PRE_TARGET
        return junction.pre_target
    elseif lane_id == Y_POST_TARGET
        return junction.post_target
    end
    error("invalid Y-junction lane id $lane_id")
end

function y_candidate_lane_ids(dest_lane::UInt8,dest_k,candidate_k,g)
    if dest_lane == Y_POST_TARGET
        if candidate_k <= g
            return (Y_POST_TARGET,Y_NO_BRANCH)
        elseif dest_k == g && candidate_k == g + 1
            return (Y_PRE_CONTROL,Y_PRE_TARGET)
        end
    else
        if candidate_k > g
            return (dest_lane,Y_NO_BRANCH)
        elseif dest_k == g + 1 && candidate_k == g
            return (Y_POST_TARGET,Y_NO_BRANCH)
        end
    end
    return (Y_NO_BRANCH,Y_NO_BRANCH)
end

function update_y_candidate!(
    best,distance,junction::TargetYJunction,lane_id,ip,jp,kp,axis,sign_index,
)
    if lane_id == Y_NO_BRANCH
        return best
    end
    lane = y_lane(junction,lane_id)
    if lane.hist[ip,jp,kp]
        best = min(best,distance)
    end
    incoming = lane.fields[ip,jp,kp,axis,sign_index]
    if incoming != 0
        best = min(best,incoming + distance)
    end
    return best
end

function y_site_field_values(junction::TargetYJunction,dest_lane::UInt8,i,j,k)
    values = zeros(Int,3,2)
    L,_,Z = size(junction.post_target.hist)
    g = junction.junction_depth
    ind(index) = mod1(index,L)
    zind(index) = clamp(index,1,Z)

    for axis in 1:3, step in (-1,1)
        sign_index = step == 1 ? 1 : 2
        best = typemax(Int)
        for delta1 in -1:1, delta2 in -1:1
            if axis == 1
                ip = ind(i-step)
                jp = ind(j+delta1)
                kp = zind(k+delta2)
            elseif axis == 2
                ip = ind(i+delta1)
                jp = ind(j-step)
                kp = zind(k+delta2)
            else
                ip = ind(i+delta1)
                jp = ind(j+delta2)
                kp = zind(k-step)
            end
            distance = 1 + abs(delta1) + abs(delta2)
            lane_a,lane_b = y_candidate_lane_ids(dest_lane,k,kp,g)
            best = update_y_candidate!(
                best,distance,junction,lane_a,ip,jp,kp,axis,sign_index,
            )
            best = update_y_candidate!(
                best,distance,junction,lane_b,ip,jp,kp,axis,sign_index,
            )
        end
        values[axis,sign_index] = best == typemax(Int) ? 0 : best
    end
    return values
end

function yjunction_field_sweep!(junction::TargetYJunction)
    L,_,Z = size(junction.post_target.hist)
    g = junction.junction_depth
    for lane in (junction.pre_control,junction.pre_target,junction.post_target)
        lane.new_fields .= 0
    end

    if g > 0
        for i in 1:L, j in 1:L, k in 1:g
            junction.post_target.new_fields[i,j,k,:,:] .= y_site_field_values(
                junction,Y_POST_TARGET,i,j,k,
            )
        end
    end
    if g < Z
        for lane_id in (Y_PRE_CONTROL,Y_PRE_TARGET)
            lane = y_lane(junction,lane_id)
            for i in 1:L, j in 1:L, k in g+1:Z
                lane.new_fields[i,j,k,:,:] .= y_site_field_values(
                    junction,lane_id,i,j,k,
                )
            end
        end
    end

    # Preserve the two branch costs from the same frozen field state that
    # produced the merged post-side temporal message.  Recomputing these after
    # committing the Jacobi sweep would use a one-sweep-newer branch field and
    # could route a defect through a branch that did not supply its minimum.
    junction.branch_temporal_costs .= 0
    if 1 <= g < Z
        for i in 1:L, j in 1:L
            junction.branch_temporal_costs[i,j,1] =
                junction_branch_temporal_cost(junction,Y_PRE_CONTROL,i,j)
            junction.branch_temporal_costs[i,j,2] =
                junction_branch_temporal_cost(junction,Y_PRE_TARGET,i,j)
        end
    end

    # Commit every lane only after the complete Y graph has been evaluated.
    for lane in (junction.pre_control,junction.pre_target,junction.post_target)
        lane.fields .= lane.new_fields
    end
    return nothing
end

function junction_branch_temporal_cost(
    junction::TargetYJunction,branch::UInt8,i,j,
)
    L,_,Z = size(junction.post_target.hist)
    g = junction.junction_depth
    if !(1 <= g < Z) || !(branch in (Y_PRE_CONTROL,Y_PRE_TARGET))
        return 0
    end
    ind(index) = mod1(index,L)
    lane = y_lane(junction,branch)
    best = typemax(Int)
    for delta_i in -1:1, delta_j in -1:1
        ip = ind(i+delta_i)
        jp = ind(j+delta_j)
        distance = 1 + abs(delta_i) + abs(delta_j)
        if lane.hist[ip,jp,g+1]
            best = min(best,distance)
        end
        incoming = lane.fields[ip,jp,g+1,3,2]
        if incoming != 0
            best = min(best,incoming + distance)
        end
    end
    return best == typemax(Int) ? 0 : best
end

function choose_junction_branch!(state::YJunctionCNOTState,junction,i,j)
    control_cost = junction.branch_temporal_costs[i,j,1]
    target_cost = junction.branch_temporal_costs[i,j,2]
    if control_cost == 0
        return target_cost == 0 ? Y_NO_BRANCH : Y_PRE_TARGET
    elseif target_cost == 0
        return Y_PRE_CONTROL
    elseif control_cost == target_cost
        state.equal_branch_ties += 1
        return Y_PRE_CONTROL
    elseif control_cost < target_cost
        return Y_PRE_CONTROL
    end
    return Y_PRE_TARGET
end

function select_yjunction_proposals!(
    state::YJunctionCNOTState,junction::TargetYJunction;
    decoder_rng=Random.default_rng(),
)
    L,_,Z = size(junction.post_target.hist)
    g = junction.junction_depth
    ind(index) = mod1(index,L)
    junction.junction_proposals .= Y_NO_BRANCH
    for lane in (junction.pre_control,junction.pre_target,junction.post_target)
        lane.proposals .= false
    end

    # Pre-gate lanes retain ordinary one-way defect aging toward larger k.
    if g < Z
        for lane_id in (Y_PRE_CONTROL,Y_PRE_TARGET)
            lane = y_lane(junction,lane_id)
            for i in 1:L, j in 1:L, k in g+1:Z
                if !lane.hist[i,j,k]
                    continue
                end
                if k < Z && any(!iszero,@view lane.fields[i,j,k,:,:])
                    mindist = minimum(lane.fields[i,j,k,:,:][lane.fields[i,j,k,:,:] .> 0])
                    if lane.fields[i,j,k,3,2] == mindist
                        lane.proposals[i,j,k,3] = true
                    elseif lane.fields[i,j,k,1,1] == mindist
                        lane.proposals[ind(i-1),j,k,1] = true
                    elseif lane.fields[i,j,k,2,1] == mindist
                        lane.proposals[i,ind(j-1),k,2] = true
                    elseif lane.fields[i,j,k,2,2] == mindist
                        lane.proposals[i,j,k,2] = true
                    elseif lane.fields[i,j,k,1,2] == mindist
                        lane.proposals[i,j,k,1] = true
                    end
                elseif k == Z && any(!iszero,@view lane.fields[i,j,k,1:2,:]) &&
                       rand(decoder_rng) < 0.8
                    mindist = minimum(lane.fields[i,j,k,1:2,:][lane.fields[i,j,k,1:2,:] .> 0])
                    if lane.fields[i,j,k,1,1] == mindist
                        lane.proposals[ind(i-1),j,k,1] = true
                    elseif lane.fields[i,j,k,2,1] == mindist
                        lane.proposals[i,ind(j-1),k,2] = true
                    elseif lane.fields[i,j,k,2,2] == mindist
                        lane.proposals[i,j,k,2] = true
                    elseif lane.fields[i,j,k,1,2] == mindist
                        lane.proposals[i,j,k,1] = true
                    end
                end
            end
        end
    end

    post = junction.post_target
    if g > 0
        for i in 1:L, j in 1:L, k in 1:g
            if !post.hist[i,j,k] || !any(!iszero,@view post.fields[i,j,k,:,:])
                continue
            end
            mindist = minimum(post.fields[i,j,k,:,:][post.fields[i,j,k,:,:] .> 0])
            if post.fields[i,j,k,3,2] == mindist
                if k < g
                    post.proposals[i,j,k,3] = true
                else
                    branch = choose_junction_branch!(state,junction,i,j)
                    junction.junction_proposals[i,j] = branch
                    if branch == Y_PRE_CONTROL
                        state.control_branch_crossings += 1
                    elseif branch == Y_PRE_TARGET
                        state.target_branch_crossings += 1
                    end
                end
            elseif post.fields[i,j,k,1,1] == mindist
                post.proposals[ind(i-1),j,k,1] = true
            elseif post.fields[i,j,k,2,1] == mindist
                post.proposals[i,ind(j-1),k,2] = true
            elseif post.fields[i,j,k,2,2] == mindist
                post.proposals[i,j,k,2] = true
            elseif post.fields[i,j,k,1,2] == mindist
                post.proposals[i,j,k,1] = true
            end
        end
    end
    return nothing
end

function commit_yjunction_proposals!(junction::TargetYJunction,target_frame)
    for lane in (junction.pre_control,junction.pre_target,junction.post_target)
        commit_lane_proposals!(lane,target_frame)
    end
    g = junction.junction_depth
    _,_,Z = size(junction.post_target.hist)
    if 1 <= g < Z
        for i in axes(junction.junction_proposals,1),
            j in axes(junction.junction_proposals,2)
            branch = junction.junction_proposals[i,j]
            if branch == Y_PRE_CONTROL || branch == Y_PRE_TARGET
                junction.post_target.hist[i,j,g] ⊻= true
                y_lane(junction,branch).hist[i,j,g+1] ⊻= true
            end
        end
    end
    return nothing
end

function clear_invalid_yjunction_slices!(junction::TargetYJunction)
    _,_,Z = size(junction.post_target.hist)
    g = junction.junction_depth
    if g < Z
        junction.post_target.hist[:,:,g+1:Z] .= false
        junction.post_target.fields[:,:,g+1:Z,:,:] .= 0
        junction.post_target.new_fields[:,:,g+1:Z,:,:] .= 0
        junction.post_target.proposals[:,:,g+1:Z,:] .= false
    end
    if g > 0
        for lane in (junction.pre_control,junction.pre_target)
            lane.hist[:,:,1:g] .= false
            lane.fields[:,:,1:g,:,:] .= 0
            lane.new_fields[:,:,1:g,:,:] .= 0
            lane.proposals[:,:,1:g,:] .= false
        end
    end
    return nothing
end

function collapse_yjunction!(state::YJunctionCNOTState,junction::TargetYJunction)
    post = junction.post_target
    _,_,Z = size(post.hist)
    post.hist[:,:,Z] .⊻= junction.pre_control.hist[:,:,Z]
    post.hist[:,:,Z] .⊻= junction.pre_target.hist[:,:,Z]
    post.fields[:,:,Z,1:2,:] .= nonzeromin.(
        post.fields[:,:,Z,1:2,:],
        junction.pre_control.fields[:,:,Z,1:2,:],
        junction.pre_target.fields[:,:,Z,1:2,:],
    )
    post.fields[:,:,Z,3,:] .= 0
    post.new_fields .= 0
    post.proposals .= false
    state.target_decoder = post
    state.collapse_round = state.rounds + 1
    return nothing
end

function age_yjunction!(
    state::YJunctionCNOTState,junction::TargetYJunction,new_event,
)
    _,_,Z = size(junction.post_target.hist)
    for lane in (junction.pre_control,junction.pre_target,junction.post_target)
        rg_cycle!(lane)
    end
    junction.junction_depth += 1
    junction.post_target.hist[:,:,1] .= new_event
    if junction.junction_depth == Z
        collapse_yjunction!(state,junction)
    else
        clear_invalid_yjunction_slices!(junction)
    end
    return nothing
end

function update_yjunction_target!(
    state::YJunctionCNOTState,junction::TargetYJunction,r,
    data_mask,measurement_mask;decoder_rng=Random.default_rng(),
)
    if r < 1
        error("Y-junction CNOT requires r >= 1")
    end
    for _ in 1:r
        yjunction_field_sweep!(junction)
    end
    select_yjunction_proposals!(state,junction;decoder_rng=decoder_rng)
    target = state.blocks[Y_TARGET_BLOCK]
    commit_yjunction_proposals!(junction,target.frame)
    new_event = apply_block_channel!(target,data_mask,measurement_mask)
    age_yjunction!(state,junction,new_event)
    return nothing
end

function apply_cnot_x_yjunction!(
    state::YJunctionCNOTState,
    control_block=Y_CONTROL_BLOCK,
    target_block=Y_TARGET_BLOCK,
)
    if state.cnot_applied
        error("the Y-junction prototype supports exactly one CNOT")
    elseif control_block != Y_CONTROL_BLOCK || target_block != Y_TARGET_BLOCK ||
           length(state.blocks) != 2
        error("the Y-junction prototype supports block 1 -> block 2 only")
    elseif !(state.target_decoder isa DecoderLane)
        error("target decoder is not in its pre-gate single-lane state")
    end

    control = state.blocks[control_block]
    target = state.blocks[target_block]
    target.errors .⊻= control.errors
    target.frame .⊻= control.frame
    target.old_synds .⊻= control.old_synds
    target.new_synds .⊻= control.new_synds

    pre_control = snapshot_decoder_lane(state.control_history)
    pre_target = state.target_decoder
    pre_target.new_fields .= 0
    pre_target.proposals .= false
    L,_,Z = size(pre_target.hist)
    post_target = make_decoder_lane(L,Z)
    state.target_decoder = TargetYJunction(
        pre_control,
        pre_target,
        post_target,
        0,
        zeros(Int,L,L,2),
        fill(Y_NO_BRANCH,L,L),
    )
    state.cnot_applied = true
    state.cnot_round = state.rounds
    state.max_target_lane_count = 3
    return nothing
end

function update_yjunction_round!(
    state::YJunctionCNOTState,r,p,q;
    synch=true,pretty=false,masks=nothing,
    noise_rng=Random.default_rng(),decoder_rng=nothing,
)
    if !synch
        error("Y-junction CNOT supports synchronous updates only")
    elseif pretty
        error("Y-junction CNOT does not implement pretty updates")
    end
    L = size(state.blocks[Y_CONTROL_BLOCK].errors,1)
    round_masks = masks === nothing ?
        sample_yjunction_round_masks(noise_rng,L,p,q) :
        validate_yjunction_round_masks(masks,L)
    if decoder_rng === nothing
        decoder_rng = Random.Xoshiro(rand(noise_rng,UInt64))
    end

    update_live_lane!(
        state.blocks[Y_CONTROL_BLOCK],
        state.control_history,
        r,
        round_masks.data[Y_CONTROL_BLOCK],
        round_masks.measurement[Y_CONTROL_BLOCK];
        decoder_rng=decoder_rng,
    )

    target = state.blocks[Y_TARGET_BLOCK]
    decoder = state.target_decoder
    if decoder isa TargetYJunction
        update_yjunction_target!(
            state,decoder,r,
            round_masks.data[Y_TARGET_BLOCK],
            round_masks.measurement[Y_TARGET_BLOCK];
            decoder_rng=decoder_rng,
        )
    else
        update_live_lane!(
            target,decoder,r,
            round_masks.data[Y_TARGET_BLOCK],
            round_masks.measurement[Y_TARGET_BLOCK];
            decoder_rng=decoder_rng,
        )
    end
    state.rounds += 1
    return round_masks
end

function all_yjunction_histories_empty(state::YJunctionCNOTState)
    if any(state.control_history.hist)
        return false
    end
    decoder = state.target_decoder
    if decoder isa TargetYJunction
        return !any(decoder.pre_control.hist) &&
               !any(decoder.pre_target.hist) &&
               !any(decoder.post_target.hist)
    end
    return !any(decoder.hist)
end

yjunction_is_collapsed(state::YJunctionCNOTState) =
    state.cnot_applied && state.target_decoder isa DecoderLane

function yjunction_target_lane_count(state::YJunctionCNOTState)
    return state.target_decoder isa TargetYJunction ? 3 : 1
end

function yjunction_field_pair_count(state::YJunctionCNOTState)
    return 1 + yjunction_target_lane_count(state)
end

function decoded_yjunction_block(state::YJunctionCNOTState,block)
    return state.blocks[block].errors .⊻ state.blocks[block].frame
end

function split_cnot_timing(total_time)
    if total_time < 1
        error("CNOT total time T must be positive")
    end
    pre_time = fld(total_time,2)
    post_time = total_time - pre_time
    return pre_time,post_time,2total_time
end

function estimate_yjunction_cnot_Ft(
    L,Z,p,q,r,synch,pretty,T_PRE,T_POST,CLEANUP_TIME,
    acc_err,fixed_samps,trial_parallel,verbose,
)
    if !synch
        error("2d_windowed_cnot_yjunction.jl supports SYNCH=true only")
    elseif pretty
        error("2d_windowed_cnot_yjunction.jl does not implement pretty updates")
    elseif T_PRE < 0 || T_POST < 0 || CLEANUP_TIME < 0
        error("CNOT timing and cleanup counts must be nonnegative")
    end

    use_fixed_samps = fixed_samps > 0
    if !use_fixed_samps && p == 0 && q == 0
        use_fixed_samps = true
        fixed_samps = 1
    end
    if use_fixed_samps && fixed_samps < 1
        error("fixed_samps must be positive")
    elseif !use_fixed_samps && acc_err < 1
        error("acc_err must be positive when accumulating failures")
    end

    work_units = use_fixed_samps ? fixed_samps : acc_err
    worker_count = trial_parallel ? min(nthreads(),max(work_units,1)) : 1
    worker_results = Vector{Any}(undef,worker_count)
    wall_start = time_ns()

    function run_trials(local_samps,target_errors)
        failures = 0
        trials = 0
        control_failures = 0
        target_failures = 0
        both_failures = 0
        cleanup_failures = 0
        collapse_failures = 0
        control_crossings = 0
        target_crossings = 0
        equal_ties = 0
        collapse_delay_sum = 0
        collapsed_trials = 0
        gate_bytes_sum = 0
        final_bytes_sum = 0

        while use_fixed_samps ? trials < local_samps : failures < target_errors
            if verbose && trials % 10000 == 0
                println("thread $(threadid()) Y-junction trial: ",trials)
            end
            state = initial_yjunction_state(L,Z)
            for _ in 1:T_PRE
                update_yjunction_round!(state,r,p,q)
            end
            apply_cnot_x_yjunction!(state)
            gate_bytes_sum += Base.summarysize(state)
            for _ in 1:T_POST
                update_yjunction_round!(state,r,p,q)
            end
            for _ in 1:CLEANUP_TIME
                if all_yjunction_histories_empty(state) && yjunction_is_collapsed(state)
                    break
                end
                update_yjunction_round!(state,r,0.0,0.0)
            end

            history_cleanup_failed = !all_yjunction_histories_empty(state)
            collapse_failed = !yjunction_is_collapsed(state)
            cleanup_failures += history_cleanup_failed
            collapse_failures += collapse_failed
            if !collapse_failed
                collapse_delay_sum += state.collapse_round - state.cnot_round
                collapsed_trials += 1
            end

            decoded_control = decoded_yjunction_block(state,Y_CONTROL_BLOCK)
            decoded_target = decoded_yjunction_block(state,Y_TARGET_BLOCK)
            if !history_cleanup_failed
                @assert !any(get_synds(decoded_control)) "decoded control is not syndrome-free"
                @assert !any(get_synds(decoded_target)) "decoded target is not syndrome-free"
            end
            control_failed = !detect_logical_error(decoded_control)
            target_failed = !detect_logical_error(decoded_target)
            logical_failed = control_failed || target_failed

            failures += logical_failed
            control_failures += control_failed
            target_failures += target_failed
            both_failures += control_failed && target_failed
            control_crossings += state.control_branch_crossings
            target_crossings += state.target_branch_crossings
            equal_ties += state.equal_branch_ties
            final_bytes_sum += Base.summarysize(state)
            trials += 1
        end
        return (
            failures,trials,control_failures,target_failures,both_failures,
            cleanup_failures,collapse_failures,control_crossings,
            target_crossings,equal_ties,collapse_delay_sum,collapsed_trials,
            gate_bytes_sum,final_bytes_sum,
        )
    end

    @threads for worker in 1:worker_count
        if use_fixed_samps
            local_samps = fixed_samps ÷ worker_count +
                          (worker <= fixed_samps % worker_count ? 1 : 0)
            worker_results[worker] = run_trials(local_samps,0)
        else
            target_errors = acc_err ÷ worker_count +
                            (worker <= acc_err % worker_count ? 1 : 0)
            worker_results[worker] = run_trials(0,target_errors)
        end
    end

    total(index) = sum(result[index] for result in worker_results)
    logical_failures = total(1)
    trials = total(2)
    collapsed_trials = total(12)
    fail_rate = logical_failures / trials
    elapsed_seconds = (time_ns() - wall_start) / 1.0e9
    return Dict{String,Any}(
        "CNOT_Ft" => 1 - fail_rate,
        "CNOT_fail_rate" => fail_rate,
        "trials" => trials,
        "logical_failures" => logical_failures,
        "control_logical_failures" => total(3),
        "target_logical_failures" => total(4),
        "both_logical_failures" => total(5),
        "cleanup_failures" => total(6),
        "yjunction_collapse_failures" => total(7),
        "yjunction_control_branch_crossings" => total(8),
        "yjunction_target_branch_crossings" => total(9),
        "yjunction_equal_branch_ties" => total(10),
        "yjunction_collapse_delay_mean" =>
            collapsed_trials == 0 ? NaN : total(11) / collapsed_trials,
        "yjunction_peak_target_lane_count" => 3,
        "yjunction_peak_total_lane_count" => 4,
        "yjunction_peak_field_pair_count" => 4,
        "yjunction_final_field_pair_count" =>
            total(7) == 0 ? 2 : missing,
        "yjunction_gate_summarysize_bytes_mean" => total(13) / trials,
        "yjunction_final_summarysize_bytes_mean" => total(14) / trials,
        "yjunction_elapsed_seconds" => elapsed_seconds,
        "yjunction_physical_block_count" => 2,
    )
end

function run_yjunction_sanity_checks(L,Z,r,T_PRE,T_POST,CLEANUP_TIME)
    state = initial_yjunction_state(L,Z)
    for _ in 1:T_PRE
        update_yjunction_round!(state,r,0.0,0.0)
    end
    apply_cnot_x_yjunction!(state)
    @assert state.target_decoder isa TargetYJunction
    @assert state.target_decoder.pre_control.hist !== state.control_history.hist
    for _ in 1:T_POST
        update_yjunction_round!(state,r,0.0,0.0)
    end
    for _ in 1:CLEANUP_TIME
        if all_yjunction_histories_empty(state) && yjunction_is_collapsed(state)
            break
        end
        update_yjunction_round!(state,r,0.0,0.0)
    end
    @assert yjunction_is_collapsed(state)
    @assert all_yjunction_histories_empty(state)
    @assert detect_logical_error(decoded_yjunction_block(state,Y_CONTROL_BLOCK))
    @assert detect_logical_error(decoded_yjunction_block(state,Y_TARGET_BLOCK))
    println("Y-junction sanity checks passed")
    return nothing
end

parse_bool(value) = lowercase(value) in ("1","true","yes","on")

function write_yjunction_output(path,params,data)
    output_path = abspath(path)
    mkpath(dirname(output_path))
    open(output_path,"w") do io
        println(io,"### data ###")
        for key in sort!(collect(keys(data)))
            println(io,key," = ",repr(data[key]))
        end
        println(io)
        println(io,"### params ###")
        for key in sort!(collect(keys(params)))
            println(io,key," = ",repr(params[key]))
        end
    end
    return output_path
end

function main()
    mode = get(ENV,"MODE","CNOT_Ft")
    L = parse(Int,get(ENV,"LVAL",mode == "CNOT_DEBUG" ? "3" : "13"))
    logz = parse_bool(get(ENV,"LOGZ","true"))
    Z = parse(Int,get(ENV,"ZVAL",string(
        logz ? ceil(Int,log(1.5,L)) : ceil(Int,L/4),
    )))
    p = parse(Float64,get(ENV,"PVAL","0.011"))
    qrat = parse(Float64,get(ENV,"QRAT","1.0"))
    q = p * qrat
    r = parse(Int,get(ENV,"RVAL","3"))
    synch = parse_bool(get(ENV,"SYNCH","true"))
    pretty = parse_bool(get(ENV,"PRETTY","false"))
    total_time = parse(Int,get(ENV,"TVAL",string(L)))
    default_pre,default_post,default_cleanup = split_cnot_timing(total_time)
    T_PRE = parse(Int,get(ENV,"CNOT_T_PRE",string(default_pre)))
    T_POST = parse(Int,get(ENV,"CNOT_T_POST",string(default_post)))
    if T_PRE + T_POST != total_time
        error("CNOT_T_PRE + CNOT_T_POST must equal TVAL")
    end
    cleanup_time = parse(Int,get(ENV,"CLEANUP_TIME",string(default_cleanup)))
    fixed_samps = parse(Int,get(ENV,"SAMPS",get(ENV,"CNOT_SAMPS","0")))
    acc_err = parse(Int,get(ENV,"ACC_ERRORS","100"))
    trial_parallel = parse_bool(get(ENV,"TRIAL_PARALLEL","true"))
    verbose = parse_bool(get(ENV,"VERBOSE","false"))

    if mode == "CNOT_DEBUG"
        run_yjunction_sanity_checks(L,Z,r,T_PRE,T_POST,cleanup_time)
        return nothing
    elseif mode != "CNOT_Ft"
        error("Y-junction driver supports MODE=CNOT_Ft or MODE=CNOT_DEBUG")
    end

    data = estimate_yjunction_cnot_Ft(
        L,Z,p,q,r,synch,pretty,T_PRE,T_POST,cleanup_time,
        acc_err,fixed_samps,trial_parallel,verbose,
    )
    params = Dict{String,Any}(
        "MODE" => mode,
        "CNOT_STYLE" => "yjunction",
        "L" => L,
        "Z" => Z,
        "p" => p,
        "q" => q,
        "QRAT" => qrat,
        "r" => r,
        "SYNCH" => synch,
        "LOGZ" => logz,
        "T" => total_time,
        "T_PRE" => T_PRE,
        "T_POST" => T_POST,
        "CLEANUP_TIME" => cleanup_time,
        "SAMPS" => fixed_samps,
        "ACC_ERRORS" => acc_err,
        "TRIAL_PARALLEL" => trial_parallel,
    )
    for key in sort!(collect(keys(data)))
        println(key," = ",data[key])
    end
    output_file = get(ENV,"OUTPUT_FILE","")
    if !isempty(output_file)
        written_path = write_yjunction_output(output_file,params,data)
        println("wrote Y-junction CNOT result to ",written_path)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
