"""
Two-pass causal-junction CNOT decoder.

The synchronous memory-decoder kernel below is copied from the primitive CNOT
lineage and narrowed to the synchronous, non-visualization path supported by
this driver. This file is standalone: it imports neither the primitive driver
nor the snapshot implementation.
"""

using Random
using Base.Threads

const CONTROL_BLOCK = 1
const TARGET_BLOCK = 2

# Primitive/baseline memory-decoder kernel.

function nonzeromin(a,b)
    if a == 0
        return b
    elseif b == 0
        return a
    end
    return min(a,b)
end

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

function onesite_field_update(i,j,k,fields,hist)
    new_fields = zeros(Int,3,2)
    L,_,Z = size(hist)
    ind(index) = mod1(index,L)
    zind(index) = clamp(index,1,Z)

    function ca_update!(axis,step)
        newfield = Inf
        sign_index = step == 1 ? 1 : 2
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
                newfield = min(newfield,distance)
            end
            incoming = fields[ip,jp,kp,axis,sign_index]
            if incoming != 0
                newfield = min(newfield,incoming + distance)
            end
        end
        new_fields[axis,sign_index] = newfield == Inf ? 0 : newfield
        return nothing
    end

    for axis in 1:3, step in (-1,1)
        ca_update!(axis,step)
    end
    return new_fields
end

@views function update_2d_windowed_fields!(fields,new_fields,hist)
    L,_,Z = size(hist)
    for i in 1:L, j in 1:L, k in 1:Z
        new_fields[i,j,k,:,:] .= onesite_field_update(i,j,k,fields,hist)
    end
    fields .= new_fields
    return nothing
end

@views function rg_cycle!(hist,fields)
    _,_,Z = size(hist)
    hist[:,:,Z] .= xor.(hist[:,:,Z],hist[:,:,Z-1])
    copyto!(hist[:,:,2:end-1],hist[:,:,1:end-2])
    hist[:,:,1] .= false

    fields[:,:,Z,1:2,:] .= nonzeromin.(
        fields[:,:,Z-1,1:2,:],fields[:,:,Z,1:2,:],
    )
    copyto!(fields[:,:,2:end-1,:,:],fields[:,:,1:end-2,:,:])
    fields[:,:,1,:,:] .= 0
    return nothing
end

function perform_correction!(hist,hist_correction)
    L,_,Z = size(hist)
    ind(index) = mod1(index,L)
    for i in 1:L, j in 1:L, k in 1:Z
        if hist_correction[i,j,k,1]
            hist[i,j,k] ⊻= true
            hist[ind(i+1),j,k] ⊻= true
        end
        if hist_correction[i,j,k,2]
            hist[i,j,k] ⊻= true
            hist[i,ind(j+1),k] ⊻= true
        end
        if hist_correction[i,j,k,3]
            hist[i,j,k] ⊻= true
            hist[i,j,k+1] ⊻= true
        end
    end
    return nothing
end

function update!(
    state,state_correction,old_synds,new_synds,hist,hist_correction,
    fields,new_fields,r,p,q,synch,pretty,
)
    if !synch
        error("two-pass primitive kernel supports synchronous updates only")
    elseif pretty
        error("two-pass primitive kernel does not support pretty updates")
    end

    L,_,Z = size(hist)
    ind(index) = mod1(index,L)
    for _ in 1:r
        update_2d_windowed_fields!(fields,new_fields,hist)
    end

    hist_correction .= false
    for i in 1:L, j in 1:L, k in 1:Z
        if !hist[i,j,k]
            continue
        end
        if k < Z
            if !any(!iszero,@view fields[i,j,k,:,:])
                continue
            end
            positive_fields = fields[i,j,k,:,:][fields[i,j,k,:,:] .> 0]
            mindist = minimum(positive_fields)
            if fields[i,j,k,3,2] == mindist
                hist_correction[i,j,k,3] = true
            elseif fields[i,j,k,1,1] == mindist
                hist_correction[ind(i-1),j,k,1] = true
            elseif fields[i,j,k,2,1] == mindist
                hist_correction[i,ind(j-1),k,2] = true
            elseif fields[i,j,k,2,2] == mindist
                hist_correction[i,j,k,2] = true
            elseif fields[i,j,k,1,2] == mindist
                hist_correction[i,j,k,1] = true
            end
        elseif any(!iszero,@view fields[i,j,k,1:2,:]) && rand() < 0.8
            positive_fields = fields[i,j,k,1:2,:][fields[i,j,k,1:2,:] .> 0]
            mindist = minimum(positive_fields)
            if fields[i,j,k,1,1] == mindist
                hist_correction[ind(i-1),j,k,1] = true
            elseif fields[i,j,k,2,1] == mindist
                hist_correction[i,ind(j-1),k,2] = true
            elseif fields[i,j,k,2,2] == mindist
                hist_correction[i,j,k,2] = true
            elseif fields[i,j,k,1,2] == mindist
                hist_correction[i,j,k,1] = true
            end
        end
    end

    for i in 1:L, j in 1:L, axis in 1:2
        state_correction[i,j,axis] ⊻= reduce(
            ⊻,@view(hist_correction[i,j,:,axis]),
        )
    end
    perform_correction!(hist,hist_correction)

    if p > 0
        state .⊻= (rand(L,L,2) .< p)
    end
    old_synds .= new_synds
    new_synds .= get_synds(state)
    if q > 0
        new_synds .⊻= (rand(L,L) .< q)
    end
    rg_cycle!(hist,fields)
    hist[:,:,1] .= old_synds .⊻ new_synds
    return nothing
end

function split_cnot_timing(total_time)
    if total_time < 1
        error("CNOT total time T must be positive")
    end
    pre_time = fld(total_time,2)
    post_time = total_time - pre_time
    cleanup_time = 2total_time
    return pre_time,post_time,cleanup_time
end

const PRE_CONTROL = 1
const PRE_TARGET = 2
const POST_TARGET = 3
const TWOPASS_STREAM_COUNT = 3

const MOVE_NONE = UInt8(0)
const MOVE_TEMPORAL = UInt8(1)
const MOVE_NEG_X = UInt8(2)
const MOVE_NEG_Y = UInt8(3)
const MOVE_POS_Y = UInt8(4)
const MOVE_POS_X = UInt8(5)

const JUNCTION_NONE = UInt8(0)
const JUNCTION_PRE_CONTROL = UInt8(PRE_CONTROL)
const JUNCTION_PRE_TARGET = UInt8(PRE_TARGET)

mutable struct TwopassPhysicalBlock
    block::Int
    errors::BitArray{3}
    frame::BitArray{3}
    old_synds::BitArray{2}
    new_synds::BitArray{2}
    noise_rounds::Int
    measurement_rounds::Int
end

mutable struct TwopassHistory
    hist::BitArray{3}
    fields::Array{Int,5}
    new_fields::Array{Int,5}
    proposals::BitArray{4}
end

mutable struct TwopassTargetHistory
    hist::BitArray{4}
    fields::Array{Int,6}
    new_fields::Array{Int,6}
    proposals::BitArray{5}
    moves::Array{UInt8,4}
    route_hist::BitArray{4}
    route_fields::Array{Int,6}
    route_new_fields::Array{Int,6}
    junction_fields::Array{Int,4}
    new_junction_fields::Array{Int,4}
    route_junction_fields::Array{Int,4}
    route_new_junction_fields::Array{Int,4}
    junction_proposals::Array{UInt8,2}
    residual::TwopassHistory
    junction_depth::Int
    retired_defects::Int
end

mutable struct TwopassCNOTState
    blocks::Vector{TwopassPhysicalBlock}
    control_history::TwopassHistory
    target_history::TwopassTargetHistory
    cnot_applied::Bool
    spatial_weight::Float64
    temporal_weight::Float64
end

function twopass_log_weight(probability)
    if probability == 0
        return Inf
    elseif !(0 < probability < 0.5)
        error("decoder probabilities must satisfy 0 <= probability < 0.5")
    end
    return log((1 - probability) / probability)
end

function make_twopass_block(block,L)
    return TwopassPhysicalBlock(
        block,
        falses(L,L,2),
        falses(L,L,2),
        falses(L,L),
        falses(L,L),
        0,
        0,
    )
end

function make_twopass_history(L,Z)
    return TwopassHistory(
        falses(L,L,Z),
        zeros(Int,L,L,Z,3,2),
        zeros(Int,L,L,Z,3,2),
        falses(L,L,Z,3),
    )
end

function make_twopass_target_history(L,Z)
    return TwopassTargetHistory(
        falses(L,L,Z,TWOPASS_STREAM_COUNT),
        zeros(Int,L,L,Z,3,2,TWOPASS_STREAM_COUNT),
        zeros(Int,L,L,Z,3,2,TWOPASS_STREAM_COUNT),
        falses(L,L,Z,3,TWOPASS_STREAM_COUNT),
        zeros(UInt8,L,L,Z,TWOPASS_STREAM_COUNT),
        falses(L,L,Z,TWOPASS_STREAM_COUNT),
        zeros(Int,L,L,Z,3,2,TWOPASS_STREAM_COUNT),
        zeros(Int,L,L,Z,3,2,TWOPASS_STREAM_COUNT),
        zeros(Int,L,L,Z,2),
        zeros(Int,L,L,Z,2),
        zeros(Int,L,L,Z,2),
        zeros(Int,L,L,Z,2),
        zeros(UInt8,L,L),
        make_twopass_history(L,Z),
        0,
        0,
    )
end

function initial_twopass_state(L,Z,p=0.01,q=p)
    if Z < 2
        error("two-pass CNOT requires Z >= 2 because the inherited RG cycle uses a separate back wall")
    end
    blocks = TwopassPhysicalBlock[
        make_twopass_block(CONTROL_BLOCK,L),
        make_twopass_block(TARGET_BLOCK,L),
    ]
    return TwopassCNOTState(
        blocks,
        make_twopass_history(L,Z),
        make_twopass_target_history(L,Z),
        false,
        twopass_log_weight(p),
        twopass_log_weight(q),
    )
end

lane_hist(target::TwopassTargetHistory,stream) = @view target.hist[:,:,:,stream]
lane_fields(target::TwopassTargetHistory,stream) = @view target.fields[:,:,:,:,:,stream]
lane_new_fields(target::TwopassTargetHistory,stream) = @view target.new_fields[:,:,:,:,:,stream]
lane_proposals(target::TwopassTargetHistory,stream) = @view target.proposals[:,:,:,:,stream]
route_lane_hist(target::TwopassTargetHistory,stream) = @view target.route_hist[:,:,:,stream]
route_lane_fields(target::TwopassTargetHistory,stream) = @view target.route_fields[:,:,:,:,:,stream]
route_lane_new_fields(target::TwopassTargetHistory,stream) = @view target.route_new_fields[:,:,:,:,:,stream]

function select_twopass_move(temporal,neg_x,neg_y,pos_y,pos_x,spatial_weight,temporal_weight)
    candidates = (
        (MOVE_TEMPORAL, temporal == 0 ? Inf : temporal * temporal_weight),
        (MOVE_NEG_X, neg_x == 0 ? Inf : neg_x * spatial_weight),
        (MOVE_NEG_Y, neg_y == 0 ? Inf : neg_y * spatial_weight),
        (MOVE_POS_Y, pos_y == 0 ? Inf : pos_y * spatial_weight),
        (MOVE_POS_X, pos_x == 0 ? Inf : pos_x * spatial_weight),
    )
    selected = MOVE_NONE
    selected_cost = Inf
    for (move,cost) in candidates
        if isfinite(cost) && cost < selected_cost
            selected = move
            selected_cost = cost
        end
    end
    return selected
end

function update_twopass_live_block!(block::TwopassPhysicalBlock,hist,proposals,fields,new_fields,r,p,q)
    update!(
        block.errors,
        block.frame,
        block.old_synds,
        block.new_synds,
        hist,
        proposals,
        fields,
        new_fields,
        r,p,q,true,false,
    )
    block.noise_rounds += 1
    block.measurement_rounds += 1
    return nothing
end

function apply_cnot_x_twopass!(state::TwopassCNOTState,control_block=CONTROL_BLOCK,target_block=TARGET_BLOCK)
    if state.cnot_applied
        error("the two-pass prototype currently supports exactly one CNOT")
    elseif control_block == target_block
        error("CNOT control and target must be different blocks")
    elseif length(state.blocks) != 2 || control_block != CONTROL_BLOCK || target_block != TARGET_BLOCK
        error("the two-pass prototype supports control block 1 and target block 2 only")
    end

    control = state.blocks[control_block]
    target = state.blocks[target_block]
    histories = state.target_history

    target.errors .⊻= control.errors
    target.frame .⊻= control.frame
    target.old_synds .⊻= control.old_synds
    target.new_synds .⊻= control.new_synds

    lane_hist(histories,PRE_CONTROL) .= state.control_history.hist
    lane_fields(histories,PRE_CONTROL) .= state.control_history.fields
    lane_new_fields(histories,PRE_CONTROL) .= 0

    lane_hist(histories,POST_TARGET) .= false
    lane_fields(histories,POST_TARGET) .= 0
    histories.new_fields .= 0
    histories.proposals .= false
    histories.moves .= MOVE_NONE
    histories.route_hist .= false
    histories.route_fields .= 0
    histories.route_new_fields .= 0
    histories.junction_fields .= 0
    histories.new_junction_fields .= 0
    histories.route_junction_fields .= 0
    histories.route_new_junction_fields .= 0
    histories.junction_proposals .= JUNCTION_NONE
    histories.residual.hist .= false
    histories.residual.fields .= 0
    histories.residual.new_fields .= 0
    histories.residual.proposals .= false
    histories.junction_depth = 0
    histories.retired_defects = 0

    state.cnot_applied = true
    return nothing
end

function compute_twopass_junction_fields!(new_junction,junction,hist,fields,g)
    _,_,Z,_ = size(hist)
    new_junction .= 0
    if 1 <= g < Z
        for branch in 1:2
            pre_stream = branch == 1 ? PRE_CONTROL : PRE_TARGET
            for i in axes(hist,1), j in axes(hist,2), k in 1:g
                if k == g
                    if hist[i,j,g+1,pre_stream]
                        new_junction[i,j,k,branch] = 1
                    else
                        incoming = fields[i,j,g+1,3,2,pre_stream]
                        if incoming != 0
                            new_junction[i,j,k,branch] = incoming + 1
                        end
                    end
                else
                    incoming = junction[i,j,k+1,branch]
                    if incoming != 0
                        new_junction[i,j,k,branch] = incoming + 1
                    end
                end
            end
        end
    end
    return nothing
end

function update_twopass_fields!(target::TwopassTargetHistory,r)
    if r < 1
        error("two-pass CNOT requires r >= 1")
    end
    for _ in 1:r
        # Compute the junction buffer from the same pre-sweep fields used by
        # every lane. Commit it only after all independent lane sweeps.
        compute_twopass_junction_fields!(
            target.new_junction_fields,
            target.junction_fields,
            target.hist,
            target.fields,
            target.junction_depth,
        )
        for stream in 1:TWOPASS_STREAM_COUNT
            update_2d_windowed_fields!(
                lane_fields(target,stream),
                lane_new_fields(target,stream),
                lane_hist(target,stream),
            )
        end
        update_2d_windowed_fields!(
            target.residual.fields,
            target.residual.new_fields,
            target.residual.hist,
        )
        target.junction_fields .= target.new_junction_fields
    end
    return nothing
end

function retire_labeled_backwall!(target::TwopassTargetHistory)
    L,_,Z,_ = size(target.hist)
    affected_streams = falses(TWOPASS_STREAM_COUNT)
    transferred = 0
    for stream in 1:TWOPASS_STREAM_COUNT, i in 1:L, j in 1:L
        if target.hist[i,j,Z,stream]
            target.hist[i,j,Z,stream] = false
            target.residual.hist[i,j,Z] ⊻= true
            affected_streams[stream] = true
            transferred += 1
        end
    end
    if transferred > 0
        for stream in 1:TWOPASS_STREAM_COUNT
            if affected_streams[stream]
                lane_fields(target,stream) .= 0
                lane_new_fields(target,stream) .= 0
            end
        end
        target.residual.fields .= 0
        target.residual.new_fields .= 0
        target.junction_fields .= 0
        target.new_junction_fields .= 0
        target.route_junction_fields .= 0
        target.route_new_junction_fields .= 0
        target.retired_defects += transferred
    end
    return transferred
end

function twopass_temporal_message(target::TwopassTargetHistory,i,j,k,stream)
    return twopass_temporal_message(
        target.hist,
        target.fields,
        target.junction_fields,
        target.junction_depth,
        i,j,k,stream,
    )
end

function twopass_temporal_message(hist,fields,junction_fields,junction_depth,i,j,k,stream)
    _,_,Z,_ = size(hist)
    if k >= Z
        return 0
    end
    own_message = fields[i,j,k,3,2,stream]
    if stream != POST_TARGET || junction_depth == 0
        return own_message
    end

    g = junction_depth
    if k > g
        return own_message
    elseif k == g
        return nonzeromin(
            junction_fields[i,j,k,1],
            junction_fields[i,j,k,2],
        )
    end
    return nonzeromin(
        own_message,
        nonzeromin(
            junction_fields[i,j,k,1],
            junction_fields[i,j,k,2],
        ),
    )
end

function select_twopass_directions!(target::TwopassTargetHistory,spatial_weight,temporal_weight)
    L,_,Z,_ = size(target.hist)
    target.moves .= MOVE_NONE
    for stream in 1:TWOPASS_STREAM_COUNT, i in 1:L, j in 1:L, k in 1:Z
        if target.hist[i,j,k,stream]
            temporal = twopass_temporal_message(target,i,j,k,stream)
            target.moves[i,j,k,stream] = select_twopass_move(
                temporal,
                target.fields[i,j,k,1,1,stream],
                target.fields[i,j,k,2,1,stream],
                target.fields[i,j,k,2,2,stream],
                target.fields[i,j,k,1,2,stream],
                spatial_weight,
                temporal_weight,
            )
        end
    end
    return nothing
end

function select_junction_branch(junction_fields,g,i,j)
    if g == 0
        return JUNCTION_NONE
    end
    control_cost = junction_fields[i,j,g,1]
    target_cost = junction_fields[i,j,g,2]
    if control_cost == 0
        return target_cost == 0 ? JUNCTION_NONE : JUNCTION_PRE_TARGET
    elseif target_cost == 0 || control_cost <= target_cost
        return JUNCTION_PRE_CONTROL
    end
    return JUNCTION_PRE_TARGET
end

function populate_route_history!(target::TwopassTargetHistory)
    # The second pass is source/stream aware, not classification aware: every
    # current defect in the same legal stream graph may be the selected
    # spatial or temporal partner.
    target.route_hist .= target.hist
    return nothing
end

function recompute_route_fields!(target::TwopassTargetHistory,r;with_junction)
    # Preserve Lake's accumulated propagation radius by seeding the route bank
    # from pass one's persistent messages, then recompute on the frozen current
    # histories. Starting from zero here would permanently cap routing at r.
    target.route_fields .= target.fields
    target.route_new_fields .= 0
    target.route_junction_fields .= target.junction_fields
    target.route_new_junction_fields .= 0
    for _ in 1:r
        if with_junction
            compute_twopass_junction_fields!(
                target.route_new_junction_fields,
                target.route_junction_fields,
                target.route_hist,
                target.route_fields,
                target.junction_depth,
            )
        end
        for stream in 1:TWOPASS_STREAM_COUNT
            update_2d_windowed_fields!(
                route_lane_fields(target,stream),
                route_lane_new_fields(target,stream),
                route_lane_hist(target,stream),
            )
        end
        if with_junction
            target.route_junction_fields .= target.route_new_junction_fields
        end
    end
    return nothing
end

function route_twopass_directions!(target::TwopassTargetHistory,r=1)
    L,_,Z,_ = size(target.hist)
    ind(i) = mod1(i,L)
    target.proposals .= false
    target.junction_proposals .= JUNCTION_NONE

    # Rebuild a fresh label-aware spacetime field after pass-one classification.
    populate_route_history!(target)
    recompute_route_fields!(target,r;with_junction=true)

    # Choose every junction crossing before recording ordinary source
    # proposals. A directly selected pre-gate endpoint is then skipped as a
    # source; proposal arrays encode edges, so clearing an endpoint coordinate
    # afterward would clear the wrong edges for negative moves and could erase
    # an unrelated neighbor's proposal.
    g = target.junction_depth
    if 1 <= g < Z
        for i in 1:L, j in 1:L
            if target.hist[i,j,g,POST_TARGET] &&
               target.moves[i,j,g,POST_TARGET] == MOVE_TEMPORAL
                temporal = twopass_temporal_message(
                    target.route_hist,
                    target.route_fields,
                    target.route_junction_fields,
                    g,
                    i,j,g,POST_TARGET,
                )
                if temporal != 0
                    target.junction_proposals[i,j] = select_junction_branch(
                        target.route_junction_fields,g,i,j,
                    )
                end
            end
        end
    end

    for stream in 1:TWOPASS_STREAM_COUNT, i in 1:L, j in 1:L, k in 1:Z
        if !target.hist[i,j,k,stream]
            continue
        end
        if 1 <= g < Z && k == g + 1 &&
           target.junction_proposals[i,j] == UInt8(stream)
            continue
        end
        first_pass_move = target.moves[i,j,k,stream]
        if first_pass_move == MOVE_TEMPORAL
            temporal = twopass_temporal_message(
                target.route_hist,
                target.route_fields,
                target.route_junction_fields,
                target.junction_depth,
                i,j,k,stream,
            )
            if temporal == 0 || k == Z
                continue
            end
            if stream == POST_TARGET && k == g
                continue
            else
                target.proposals[i,j,k,3,stream] = true
            end
        elseif first_pass_move in (MOVE_NEG_X,MOVE_NEG_Y,MOVE_POS_Y,MOVE_POS_X)
            move = select_twopass_move(
                0,
                target.route_fields[i,j,k,1,1,stream],
                target.route_fields[i,j,k,2,1,stream],
                target.route_fields[i,j,k,2,2,stream],
                target.route_fields[i,j,k,1,2,stream],
                1.0,1.0,
            )
            if move == MOVE_NEG_X
                target.proposals[ind(i-1),j,k,1,stream] = true
            elseif move == MOVE_NEG_Y
                target.proposals[i,ind(j-1),k,2,stream] = true
            elseif move == MOVE_POS_Y
                target.proposals[i,j,k,2,stream] = true
            elseif move == MOVE_POS_X
                target.proposals[i,j,k,1,stream] = true
            end
        end
    end

    return nothing
end

function select_residual_proposals!(target::TwopassTargetHistory)
    residual = target.residual
    L,_,Z = size(residual.hist)
    ind(i) = mod1(i,L)
    residual.proposals .= false
    for i in 1:L, j in 1:L, k in 1:Z
        if !residual.hist[i,j,k]
            continue
        end
        if k < Z && any(!iszero,@view residual.fields[i,j,k,:,:])
            mindist = minimum(residual.fields[i,j,k,:,:][residual.fields[i,j,k,:,:] .> 0])
            if residual.fields[i,j,k,3,2] == mindist
                residual.proposals[i,j,k,3] = true
            elseif residual.fields[i,j,k,1,1] == mindist
                residual.proposals[ind(i-1),j,k,1] = true
            elseif residual.fields[i,j,k,2,1] == mindist
                residual.proposals[i,ind(j-1),k,2] = true
            elseif residual.fields[i,j,k,2,2] == mindist
                residual.proposals[i,j,k,2] = true
            elseif residual.fields[i,j,k,1,2] == mindist
                residual.proposals[i,j,k,1] = true
            end
        elseif k == Z && any(!iszero,@view residual.fields[i,j,k,1:2,:]) && rand() < 0.8
            mindist = minimum(residual.fields[i,j,k,1:2,:][residual.fields[i,j,k,1:2,:] .> 0])
            if residual.fields[i,j,k,1,1] == mindist
                residual.proposals[ind(i-1),j,k,1] = true
            elseif residual.fields[i,j,k,2,1] == mindist
                residual.proposals[i,ind(j-1),k,2] = true
            elseif residual.fields[i,j,k,2,2] == mindist
                residual.proposals[i,j,k,2] = true
            elseif residual.fields[i,j,k,1,2] == mindist
                residual.proposals[i,j,k,1] = true
            end
        end
    end
    return nothing
end

function apply_twopass_proposals!(target::TwopassTargetHistory,target_frame)
    L,_,Z,_ = size(target.hist)

    for stream in 1:TWOPASS_STREAM_COUNT
        proposals = lane_proposals(target,stream)
        for i in 1:L, j in 1:L, axis in 1:2
            target_frame[i,j,axis] ⊻= reduce(⊻,@view proposals[i,j,:,axis])
        end
        perform_correction!(lane_hist(target,stream),proposals)
    end

    residual = target.residual
    for i in 1:L, j in 1:L, axis in 1:2
        target_frame[i,j,axis] ⊻= reduce(⊻,@view residual.proposals[i,j,:,axis])
    end
    perform_correction!(residual.hist,residual.proposals)

    g = target.junction_depth
    if 1 <= g < Z
        for i in 1:L, j in 1:L
            branch = target.junction_proposals[i,j]
            if branch == JUNCTION_PRE_CONTROL || branch == JUNCTION_PRE_TARGET
                target.hist[i,j,g,POST_TARGET] ⊻= true
                target.hist[i,j,g+1,Int(branch)] ⊻= true
            end
        end
    end
    return nothing
end

function age_twopass_junction_fields!(target::TwopassTargetHistory)
    _,_,Z,_ = size(target.junction_fields)
    if Z > 2
        copyto!(
            @view(target.junction_fields[:,:,2:end-1,:]),
            @view(target.junction_fields[:,:,1:end-2,:]),
        )
    end
    target.junction_fields[:,:,1,:] .= 0
    target.junction_fields[:,:,Z,:] .= 0
    target.new_junction_fields .= 0
    return nothing
end

function age_twopass_target_histories!(target::TwopassTargetHistory,new_event)
    _,_,Z,_ = size(target.hist)
    for stream in 1:TWOPASS_STREAM_COUNT
        rg_cycle!(lane_hist(target,stream),lane_fields(target,stream))
    end
    rg_cycle!(target.residual.hist,target.residual.fields)
    age_twopass_junction_fields!(target)

    lane_hist(target,PRE_CONTROL)[:,:,1] .= false
    lane_hist(target,PRE_TARGET)[:,:,1] .= false
    lane_hist(target,POST_TARGET)[:,:,1] .= new_event
    target.residual.hist[:,:,1] .= false
    target.junction_depth = min(target.junction_depth + 1,Z - 1)
    return nothing
end

function update_twopass_target!(state::TwopassCNOTState,r,p,q)
    if !state.cnot_applied
        error("target junction update requires an applied CNOT")
    end
    target_block = state.blocks[TARGET_BLOCK]
    target = state.target_history

    update_twopass_fields!(target,r)

    # Pass one selects a direction from a frozen history. Pass two routes that
    # one edge on the labeled causal graph; histories are not mutated until all
    # proposals have been recorded.
    select_twopass_directions!(target,state.spatial_weight,state.temporal_weight)
    route_twopass_directions!(target,r)
    select_residual_proposals!(target)
    apply_twopass_proposals!(target,target_block.frame)
    retire_labeled_backwall!(target)

    if p > 0
        target_block.errors .⊻= (rand(size(target_block.errors)) .< p)
    end
    target_block.old_synds .= target_block.new_synds
    target_block.new_synds .= get_synds(target_block.errors)
    if q > 0
        target_block.new_synds .⊻= (rand(size(target_block.new_synds)) .< q)
    end
    target_block.noise_rounds += 1
    target_block.measurement_rounds += 1

    new_event = target_block.old_synds .⊻ target_block.new_synds
    age_twopass_target_histories!(target,new_event)
    return nothing
end

function update_twopass_round!(state::TwopassCNOTState,r,p,q;pretty=false)
    if pretty
        error("two-pass CNOT does not implement pretty/visualization updates")
    end
    control = state.blocks[CONTROL_BLOCK]
    update_twopass_live_block!(
        control,
        state.control_history.hist,
        state.control_history.proposals,
        state.control_history.fields,
        state.control_history.new_fields,
        r,p,q,
    )

    if state.cnot_applied
        update_twopass_target!(state,r,p,q)
    else
        target = state.blocks[TARGET_BLOCK]
        history = state.target_history
        update_twopass_live_block!(
            target,
            lane_hist(history,PRE_TARGET),
            lane_proposals(history,PRE_TARGET),
            lane_fields(history,PRE_TARGET),
            lane_new_fields(history,PRE_TARGET),
            r,p,q,
        )
    end
    return nothing
end

function all_twopass_histories_empty(state::TwopassCNOTState)
    return !any(state.control_history.hist) &&
           !any(state.target_history.hist) &&
           !any(state.target_history.residual.hist)
end

function decoded_twopass_block(state::TwopassCNOTState,block)
    return state.blocks[block].errors .⊻ state.blocks[block].frame
end

function twopass_stream_defect_counts(state::TwopassCNOTState)
    return [count(lane_hist(state.target_history,stream)) for stream in 1:TWOPASS_STREAM_COUNT]
end

function estimate_twopass_cnot_Ft(
    L,Z,p,q,r,synch,pretty,T_PRE,T_POST,CLEANUP_TIME,
    acc_err,fixed_samps,trial_parallel,verbose,
)
    if !synch
        error("2d_windowed_cnot_twopass.jl supports SYNCH=true only")
    elseif pretty
        error("2d_windowed_cnot_twopass.jl does not implement pretty updates")
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

    function run_trials(local_samps,target_errors)
        failures = 0
        trials = 0
        control_failures = 0
        target_failures = 0
        both_failures = 0
        cleanup_failures = 0
        retired_defects = 0

        while use_fixed_samps ? (trials < local_samps) : (failures < target_errors)
            if verbose && trials % 10000 == 0
                println("thread $(threadid()) two-pass CNOT trial: ",trials)
            end
            state = initial_twopass_state(L,Z,p,q)
            for _ in 1:T_PRE
                update_twopass_round!(state,r,p,q)
            end
            apply_cnot_x_twopass!(state)
            for _ in 1:T_POST
                update_twopass_round!(state,r,p,q)
            end
            for _ in 1:CLEANUP_TIME
                if all_twopass_histories_empty(state)
                    break
                end
                update_twopass_round!(state,r,0.0,0.0)
            end

            cleanup_failed = !all_twopass_histories_empty(state)
            cleanup_failures += cleanup_failed
            decoded_control = decoded_twopass_block(state,CONTROL_BLOCK)
            decoded_target = decoded_twopass_block(state,TARGET_BLOCK)
            if !cleanup_failed
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
            retired_defects += state.target_history.retired_defects
            trials += 1
        end
        return (
            failures,trials,control_failures,target_failures,both_failures,
            cleanup_failures,retired_defects,
        )
    end

    @threads for worker in 1:worker_count
        if use_fixed_samps
            local_samps = fixed_samps ÷ worker_count + (worker <= fixed_samps % worker_count ? 1 : 0)
            worker_results[worker] = run_trials(local_samps,0)
        else
            target_errors = acc_err ÷ worker_count + (worker <= acc_err % worker_count ? 1 : 0)
            worker_results[worker] = run_trials(0,target_errors)
        end
    end

    logical_failures = sum(result[1] for result in worker_results)
    total_trials = sum(result[2] for result in worker_results)
    control_logical_failures = sum(result[3] for result in worker_results)
    target_logical_failures = sum(result[4] for result in worker_results)
    both_logical_failures = sum(result[5] for result in worker_results)
    total_cleanup_failures = sum(result[6] for result in worker_results)
    total_retired_defects = sum(result[7] for result in worker_results)
    fail_rate = logical_failures / total_trials

    return Dict{String,Any}(
        "CNOT_Ft" => 1 - fail_rate,
        "CNOT_fail_rate" => fail_rate,
        "trials" => total_trials,
        "logical_failures" => logical_failures,
        "control_logical_failures" => control_logical_failures,
        "target_logical_failures" => target_logical_failures,
        "both_logical_failures" => both_logical_failures,
        "cleanup_failures" => total_cleanup_failures,
        "twopass_physical_block_count" => 2,
        "twopass_target_stream_count" => TWOPASS_STREAM_COUNT,
        "twopass_retired_defects" => total_retired_defects,
        "twopass_retired_defects_mean" => total_retired_defects / total_trials,
    )
end

function twopass_sanity_checks()
    state = initial_twopass_state(3,2,0.01,0.01)
    state.blocks[CONTROL_BLOCK].errors[1,1,1] = true
    state.blocks[TARGET_BLOCK].errors[2,2,2] = true
    control_before = copy(state.blocks[CONTROL_BLOCK].errors)
    target_before = copy(state.blocks[TARGET_BLOCK].errors)
    apply_cnot_x_twopass!(state)
    @assert state.blocks[CONTROL_BLOCK].errors == control_before
    @assert state.blocks[TARGET_BLOCK].errors == (target_before .⊻ control_before)
    @assert lane_hist(state.target_history,PRE_CONTROL) == state.control_history.hist
    @assert !Base.mightalias(lane_hist(state.target_history,PRE_CONTROL),state.control_history.hist)

    zero_noise = estimate_twopass_cnot_Ft(
        3,2,0.0,0.0,3,true,false,1,1,4,1,1,false,false,
    )
    @assert zero_noise["logical_failures"] == 0
    @assert zero_noise["cleanup_failures"] == 0
    println("two-pass CNOT sanity checks passed")
    return true
end

function twopass_main()
    mode = get(ENV,"MODE","CNOT_Ft")
    L = parse(Int,get(ENV,"LVAL","13"))
    logZ = parse(Bool,get(ENV,"LOGZ","true"))
    Z = logZ ? ceil(Int,log(1.5,L)) : ceil(Int,L/4)
    p = parse(Float64,get(ENV,"PVAL","0.011"))
    q = p * parse(Float64,get(ENV,"QRAT","1"))
    r = parse(Int,get(ENV,"RVAL","3"))
    synch = parse(Bool,get(ENV,"SYNCH","true"))
    trial_parallel = parse(Bool,get(ENV,"TRIAL_PARALLEL","true"))
    total_time = parse(Int,get(ENV,"TVAL",string(L)))
    T_PRE,T_POST,default_cleanup = split_cnot_timing(total_time)
    cleanup_env = get(ENV,"CLEANUP_TIME","auto")
    cleanup_time = cleanup_env == "auto" ? default_cleanup : parse(Int,cleanup_env)

    if mode == "CNOT_DEBUG"
        twopass_sanity_checks()
        return nothing
    elseif mode != "CNOT_Ft"
        error("two-pass driver supports MODE=CNOT_Ft or MODE=CNOT_DEBUG")
    end

    fixed_samps = parse(Int,get(ENV,"SAMPS","0"))
    acc_err = parse(Int,get(ENV,"ACC_ERRORS","1000"))
    data = estimate_twopass_cnot_Ft(
        L,Z,p,q,r,synch,false,T_PRE,T_POST,cleanup_time,
        acc_err,fixed_samps,trial_parallel,true,
    )
    println("two-pass CNOT result:")
    for key in sort!(collect(keys(data)))
        println("$key = $(repr(data[key]))")
    end

    if haskey(ENV,"OUTPUT_FILE")
        open(ENV["OUTPUT_FILE"],"w") do io
            println(io,"### data ###")
            for key in sort!(collect(keys(data)))
                println(io,"$key = $(repr(data[key]))")
            end
            println(io)
            println(io,"### params ###")
            println(io,"L = $L")
            println(io,"Z = $Z")
            println(io,"p = $p")
            println(io,"q = $q")
            println(io,"r = $r")
            println(io,"T_PRE = $T_PRE")
            println(io,"T_POST = $T_POST")
            println(io,"CLEANUP_TIME = $cleanup_time")
            println(io,"SYNCH = $synch")
        end
    end
    return data
end

if abspath(PROGRAM_FILE) == @__FILE__
    twopass_main()
end
