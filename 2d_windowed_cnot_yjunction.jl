"""
Moving Y-junction CNOT decoder.

This standalone prototype implements one ideal synchronous X-sector CNOT.
Two labeled pre-CNOT decoder lanes meet one unlabeled post-CNOT target lane at
a moving junction. Messages traverse the Y graph in both directions, while
every defect remains in exactly one lane. Once the junction reaches the finite
back wall, the two pre-CNOT lanes XOR-collapse into the post-CNOT lane and are
released.
"""

using Random
using Base.Threads

const CONTROL_BLOCK_ID = 1
const TARGET_BLOCK_ID = 2

const PRE_CNOT_CONTROL_LANE_ID = UInt8(1)
const PRE_CNOT_TARGET_LANE_ID = UInt8(2)
const POST_CNOT_TARGET_LANE_ID = UInt8(3)
const NO_LANE_ID = UInt8(0)

function nonzero_minimum(a,b)
    """
    returns the minimum positive message value, treating zero as "no message"
    inputs: a,b: integer message values
    output: the zero-aware minimum of a and b
    """
    if a == 0
        return b
    elseif b == 0
        return a
    end
    return min(a,b)
end

function nonzero_minimum(a,b,c)
    """
    three-candidate version of nonzero_minimum
    inputs: a,b,c: integer message values
    output: the zero-aware minimum of all three values
    """
    return nonzero_minimum(nonzero_minimum(a,b),c)
end

function compute_syndrome(edge_configuration)
    """
    calculates the periodic toric-code plaquette syndrome
    input: edge_configuration: L x L x 2 BitArray of X errors or decoded edges
    output: L x L BitArray of plaquette syndromes
    """
    L = size(edge_configuration,1)
    ind(i) = mod1(i,L)
    synds = falses(L,L)
    for i in 1:L, j in 1:L
        synds[i,j] = edge_configuration[i,j,1] ⊻ edge_configuration[i,j,2] ⊻
                      edge_configuration[ind(i-1),j,1] ⊻
                      edge_configuration[i,ind(j-1),2]
    end
    return synds
end

function is_logically_trivial(edge_configuration)
    """
    input: anyon-free L x L x 2 edge configuration
    output: true if both torus winding parities are trivial; false otherwise
    despite the historical name, true denotes logical success
    """
    Lx,Ly,_ = size(edge_configuration)
    xparity = false
    yparity = false
    for i in 1:Lx
        xparity ⊻= edge_configuration[i,1,2]
    end
    for j in 1:Ly
        yparity ⊻= edge_configuration[1,j,1]
    end
    return !xparity && !yparity
end

# Observable X-sector state for one toric-code block.
mutable struct PhysicalBlockState
    block_id::Int                         # observable block identifier
    errors::BitArray{3}                   # accumulated physical X errors; L x L x 2
    correction_frame::BitArray{3}         # accumulated recovery; L x L x 2
    previous_syndrome::BitArray{2}        # preceding measured syndrome; L x L
    current_syndrome::BitArray{2}         # latest measured syndrome; L x L
    data_noise_rounds::Int                # applied data-noise channels
    measurement_noise_rounds::Int         # applied measurement-noise channels
end

# Decoder evidence and messages for one spacetime lane. A lane owns no physical
# error state or separate correction frame.
mutable struct DecoderLane
    defects::BitArray{3}                  # syndrome-change events; L x L x Z
    messages::Array{Int,5}                # current messages; L x L x Z x 3 x 2
    next_messages::Array{Int,5}           # next Jacobi buffer; same shape
    correction_links::BitArray{4}         # frozen feedback links; L x L x Z x 3
end

# Transient target decoder while the CNOT event crosses the finite window.
# At depth g, k > g belongs to both pre-CNOT branches and k <= g belongs to
# post_cnot_target_lane.
mutable struct TargetJunctionDecoder
    pre_cnot_control_lane::DecoderLane       # non-aliased control snapshot
    pre_cnot_target_lane::DecoderLane        # original target lane
    post_cnot_target_lane::DecoderLane       # unlabeled post-CNOT trunk
    interface_depth::Int                    # current interface depth g
    branch_crossing_costs::Array{Int,3}      # frozen costs; L x L x 2
    branch_choices::Matrix{UInt8}            # winning pre-CNOT lane; L x L
end

# Complete two-block physical and decoder state. The target decoder changes
# from DecoderLane to TargetJunctionDecoder at the gate and returns to DecoderLane
# after collapse.
mutable struct YJunctionState
    physical_blocks::Vector{PhysicalBlockState}            # control and target
    control_decoder::DecoderLane                           # continuous control decoder
    target_decoder::Union{DecoderLane,TargetJunctionDecoder} # ordinary or active Y
    has_applied_cnot::Bool                             # one-gate guard
    completed_rounds::Int                              # completed rounds
    cnot_round::Int                                    # gate insertion round
    collapse_round::Int                                # branch-release round
    pre_cnot_control_crossings::Int                   # post-to-pre-control moves
    pre_cnot_target_crossings::Int                    # post-to-pre-target moves
    equal_branch_cost_ties::Int                       # positive equal-cost ties
    max_target_lane_count::Int                         # peak target evidence lanes
end

# One data-error mask and one measurement-error mask per observable block.
# Tuple index 1 is control and index 2 is target.
struct RoundNoiseMasks
    data_errors::NTuple{2,BitArray{3}}         # two L x L x 2 data masks
    measurement_errors::NTuple{2,BitArray{2}}  # two L x L measurement masks
end

function RoundNoiseMasks(
    control_data_errors::AbstractArray{Bool,3},
    control_measurement_errors::AbstractArray{Bool,2},
    target_data_errors::AbstractArray{Bool,3},
    target_measurement_errors::AbstractArray{Bool,2},
)
    """
    copies explicit control/target data and measurement masks into the concrete
    RoundNoiseMasks representation
    inputs: four Boolean arrays with data shape L x L x 2 and syndrome shape L x L
    output: RoundNoiseMasks for one synchronous physical round
    """
    return RoundNoiseMasks(
        (BitArray(control_data_errors),BitArray(target_data_errors)),
        (BitArray(control_measurement_errors),BitArray(target_measurement_errors)),
    )
end

function initialize_physical_block(block_id,L)
    """
    allocates a zeroed observable physical block
    inputs: block_id: block identifier; L: spatial size
    output: PhysicalBlockState
    """
    return PhysicalBlockState(
        block_id,
        falses(L,L,2),
        falses(L,L,2),
        falses(L,L),
        falses(L,L),
        0,
        0,
    )
end

function initialize_decoder_lane(L,Z)
    """
    allocates an empty decoder-evidence lane and scratch buffers
    inputs: L: spatial size; Z: buffer depth
    output: DecoderLane
    """
    return DecoderLane(
        falses(L,L,Z),
        zeros(Int,L,L,Z,3,2),
        zeros(Int,L,L,Z,3,2),
        falses(L,L,Z,3),
    )
end

function copy_decoder_lane(lane::DecoderLane)
    """
    copies persistent defects and messages without aliasing
    input: lane: source DecoderLane
    output: independent DecoderLane with empty next messages and correction links
    """
    L,_,Z = size(lane.defects)
    copied = initialize_decoder_lane(L,Z)
    copied.defects .= lane.defects
    copied.messages .= lane.messages
    return copied
end

function initialize_yjunction_state(L,Z)
    """
    creates the zeroed pre-CNOT two-block Y-junction state
    inputs: L: spatial size, L >= 2; Z: buffer depth, Z >= 2
    output: YJunctionState with one ordinary decoder lane per block
    """
    if L < 2
        error("Y-junction CNOT requires L >= 2")
    elseif Z < 2
        error("Y-junction CNOT requires Z >= 2 for a distinct back wall")
    end
    return YJunctionState(
        PhysicalBlockState[
            initialize_physical_block(CONTROL_BLOCK_ID,L),
            initialize_physical_block(TARGET_BLOCK_ID,L),
        ],
        initialize_decoder_lane(L,Z),
        initialize_decoder_lane(L,Z),
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

function sample_noise_mask(rng,dims,probability)
    """
    samples a Boolean Bernoulli channel mask
    inputs: rng; dims: output dimensions; probability in [0,0.5)
    output: BitArray with independent true entries at the supplied probability
    """
    if !(0 <= probability < 0.5)
        error("decoder probabilities must satisfy 0 <= probability < 0.5")
    elseif probability == 0
        return falses(dims...)
    end
    return BitArray(rand(rng,dims...) .< probability)
end

function sample_round_noise_masks(rng,L,p,q)
    """
    samples one complete control/target physical-round mask set
    inputs: rng; L: spatial size; p: data rate; q: measurement rate
    output: RoundNoiseMasks
    """
    return RoundNoiseMasks(
        sample_noise_mask(rng,(L,L,2),p),
        sample_noise_mask(rng,(L,L),q),
        sample_noise_mask(rng,(L,L,2),p),
        sample_noise_mask(rng,(L,L),q),
    )
end

function validate_round_noise_masks(masks::RoundNoiseMasks,L)
    """
    validates explicit data and measurement mask shapes
    inputs: masks: RoundNoiseMasks; L: spatial size
    output: the validated masks; throws an error for a shape mismatch
    """
    for block_id in (CONTROL_BLOCK_ID,TARGET_BLOCK_ID)
        if size(masks.data_errors[block_id]) != (L,L,2)
            error("block $block_id data mask must have shape ($(L),$(L),2)")
        elseif size(masks.measurement_errors[block_id]) != (L,L)
            error("block $block_id measurement mask must have shape ($(L),$(L))")
        end
    end
    return masks
end

# Baseline synchronous decoder kernel.

function compute_decoder_lane_site_messages(i,j,k,messages,defects)
    """
    computes all six Lake-style messages at one ordinary-lane site
    inputs:
        i,j,k: destination coordinates
        messages: frozen L x L x Z x 3 x 2 integer message bank
        defects: frozen L x L x Z defect array
    output: 3 x 2 matrix of updated messages; does not modify messages or defects
    """
    values = zeros(Int,3,2)
    L,_,Z = size(defects)
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
            if defects[ip,jp,kp]
                best = min(best,distance)
            end
            incoming = messages[ip,jp,kp,axis,sign_index]
            if incoming != 0
                best = min(best,incoming + distance)
            end
        end
        values[axis,sign_index] = best == typemax(Int) ? 0 : best
    end
    return values
end

function update_decoder_lane_messages!(lane::DecoderLane)
    """
    performs one globally synchronous Jacobi message sweep on an ordinary lane
    input: lane: DecoderLane to update
    output: nothing; commits lane.next_messages into lane.messages
    """
    L,_,Z = size(lane.defects)
    for i in 1:L, j in 1:L, k in 1:Z
        lane.next_messages[i,j,k,:,:] .= compute_decoder_lane_site_messages(
            i,j,k,lane.messages,lane.defects,
        )
    end
    lane.messages .= lane.next_messages
    return nothing
end

function apply_correction_links!(defects,correction_links)
    """
    applies synchronous correction-link endpoints to defects by xor
    inputs:
        defects: L x L x Z BitArray of defects
        correction_links: L x L x Z x 3 BitArray of x, y, and +k links
    output: nothing; mutates defects
    correction-link selection must keep temporal links off the back wall
    """
    L,_,Z = size(defects)
    ind(index) = mod1(index,L)
    for i in 1:L, j in 1:L, k in 1:Z
        if correction_links[i,j,k,1]
            defects[i,j,k] ⊻= true
            defects[ind(i+1),j,k] ⊻= true
        end
        if correction_links[i,j,k,2]
            defects[i,j,k] ⊻= true
            defects[i,ind(j+1),k] ⊻= true
        end
        if correction_links[i,j,k,3]
            defects[i,j,k] ⊻= true
            defects[i,j,k+1] ⊻= true
        end
    end
    return nothing
end

function select_decoder_lane_correction_links!(
    lane::DecoderLane;decoder_rng=Random.default_rng(),
)
    """
    selects primitive-priority feedback moves from frozen ordinary-lane messages
    inputs: lane: DecoderLane; decoder_rng: RNG for back-wall stochasticity
    output: nothing; fills lane.correction_links without mutating lane.defects
    back-wall defects make spatial moves only, with probability 0.8
    """
    L,_,Z = size(lane.defects)
    ind(index) = mod1(index,L)
    lane.correction_links .= false
    for i in 1:L, j in 1:L, k in 1:Z
        if !lane.defects[i,j,k]
            continue
        end
        if k < Z
            if !any(!iszero,@view lane.messages[i,j,k,:,:])
                continue
            end
            mindist = minimum(lane.messages[i,j,k,:,:][lane.messages[i,j,k,:,:] .> 0])
            if lane.messages[i,j,k,3,2] == mindist
                lane.correction_links[i,j,k,3] = true
            elseif lane.messages[i,j,k,1,1] == mindist
                lane.correction_links[ind(i-1),j,k,1] = true
            elseif lane.messages[i,j,k,2,1] == mindist
                lane.correction_links[i,ind(j-1),k,2] = true
            elseif lane.messages[i,j,k,2,2] == mindist
                lane.correction_links[i,j,k,2] = true
            elseif lane.messages[i,j,k,1,2] == mindist
                lane.correction_links[i,j,k,1] = true
            end
        elseif any(!iszero,@view lane.messages[i,j,k,1:2,:]) &&
               rand(decoder_rng) < 0.8
            mindist = minimum(lane.messages[i,j,k,1:2,:][lane.messages[i,j,k,1:2,:] .> 0])
            if lane.messages[i,j,k,1,1] == mindist
                lane.correction_links[ind(i-1),j,k,1] = true
            elseif lane.messages[i,j,k,2,1] == mindist
                lane.correction_links[i,ind(j-1),k,2] = true
            elseif lane.messages[i,j,k,2,2] == mindist
                lane.correction_links[i,j,k,2] = true
            elseif lane.messages[i,j,k,1,2] == mindist
                lane.correction_links[i,j,k,1] = true
            end
        end
    end
    return nothing
end

function commit_decoder_lane_correction_links!(lane::DecoderLane,correction_frame)
    """
    commits one lane's frozen correction_links
    inputs: lane: DecoderLane; correction_frame: shared L x L x 2 Pauli frame
    output: nothing; xors spatial parity into the frame and link endpoints into defects
    """
    L,_,_ = size(lane.defects)
    for i in 1:L, j in 1:L, axis in 1:2
        correction_frame[i,j,axis] ⊻=
            reduce(⊻,@view lane.correction_links[i,j,:,axis])
    end
    apply_correction_links!(lane.defects,lane.correction_links)
    return nothing
end

function age_decoder_lane!(lane::DecoderLane)
    """
    ages one decoder lane toward larger buffer depth
    input: lane: DecoderLane
    output: nothing; xors defects onto the back wall, zero-aware-merges spatial
        wall messages, shifts bulk slices, and clears the front and scratch buffers
    """
    _,_,Z = size(lane.defects)
    lane.defects[:,:,Z] .= xor.(lane.defects[:,:,Z],lane.defects[:,:,Z-1])
    if Z > 2
        copyto!(@view(lane.defects[:,:,2:end-1]),@view(lane.defects[:,:,1:end-2]))
    end
    lane.defects[:,:,1] .= false

    lane.messages[:,:,Z,1:2,:] .= nonzero_minimum.(
        lane.messages[:,:,Z-1,1:2,:],lane.messages[:,:,Z,1:2,:],
    )
    if Z > 2
        copyto!(
            @view(lane.messages[:,:,2:end-1,:,:]),
            @view(lane.messages[:,:,1:end-2,:,:]),
        )
    end
    lane.messages[:,:,1,:,:] .= 0
    lane.next_messages .= 0
    lane.correction_links .= false
    return nothing
end

function apply_noise_to_block!(
    physical_block::PhysicalBlockState,data_errors,measurement_errors,
)
    """
    applies one observable physical and measurement channel
    inputs:
        physical_block: PhysicalBlockState
        data_errors: L x L x 2 BitArray
        measurement_errors: L x L BitArray
    output: L x L observable syndrome-change event; mutates the block and counters
    """
    physical_block.errors .⊻= data_errors
    physical_block.previous_syndrome .= physical_block.current_syndrome
    physical_block.current_syndrome .= compute_syndrome(physical_block.errors)
    physical_block.current_syndrome .⊻= measurement_errors
    physical_block.data_noise_rounds += 1
    physical_block.measurement_noise_rounds += 1
    return physical_block.previous_syndrome .⊻ physical_block.current_syndrome
end

function update_decoder_lane!(
    physical_block::PhysicalBlockState,lane::DecoderLane,r,
    data_errors,measurement_errors;
    decoder_rng=Random.default_rng(),
)
    """
    runs one complete ordinary synchronous decoder round
    inputs: physical block, lane, message-sweep count r, explicit noise masks,
        and decoder_rng for feedback stochasticity
    output: nothing; performs messages, feedback, physical channel, aging, and
        front-slice event insertion in that order
    """
    if r < 1
        error("Y-junction CNOT requires r >= 1")
    end
    for _ in 1:r
        update_decoder_lane_messages!(lane)
    end
    select_decoder_lane_correction_links!(lane;decoder_rng=decoder_rng)
    commit_decoder_lane_correction_links!(lane,physical_block.correction_frame)
    new_event = apply_noise_to_block!(
        physical_block,data_errors,measurement_errors,
    )
    age_decoder_lane!(lane)
    lane.defects[:,:,1] .= new_event
    return nothing
end

# Y-junction message topology.

function get_target_junction_lane(junction::TargetJunctionDecoder,lane_id::UInt8)
    """
    input: junction and one PRE_CNOT_CONTROL_LANE_ID/PRE_CNOT_TARGET_LANE_ID/POST_CNOT_TARGET_LANE_ID identifier
    output: the corresponding DecoderLane; errors for an invalid identifier
    """
    if lane_id == PRE_CNOT_CONTROL_LANE_ID
        return junction.pre_cnot_control_lane
    elseif lane_id == PRE_CNOT_TARGET_LANE_ID
        return junction.pre_cnot_target_lane
    elseif lane_id == POST_CNOT_TARGET_LANE_ID
        return junction.post_cnot_target_lane
    end
    error("invalid Y-junction lane id $lane_id")
end

function get_target_junction_candidate_lane_ids(
    destination_lane_id::UInt8,destination_depth,candidate_depth,interface_depth,
)
    """
    determines which lane or lanes supply one frozen message candidate
    inputs: destination lane/depth, candidate depth, and interface depth
    output: two lane identifiers, using NO_LANE_ID for absent candidates
    post-CNOT-to-pre-CNOT crossings expose both branches; reverse crossings
    expose the single post-CNOT lane
    """
    if destination_lane_id == POST_CNOT_TARGET_LANE_ID
        if candidate_depth <= interface_depth
            return (POST_CNOT_TARGET_LANE_ID,NO_LANE_ID)
        elseif destination_depth == interface_depth &&
               candidate_depth == interface_depth + 1
            return (PRE_CNOT_CONTROL_LANE_ID,PRE_CNOT_TARGET_LANE_ID)
        end
    else
        if candidate_depth > interface_depth
            return (destination_lane_id,NO_LANE_ID)
        elseif destination_depth == interface_depth + 1 &&
               candidate_depth == interface_depth
            return (POST_CNOT_TARGET_LANE_ID,NO_LANE_ID)
        end
    end
    return (NO_LANE_ID,NO_LANE_ID)
end

function reduce_target_junction_message_candidate(
    best,distance,junction::TargetJunctionDecoder,lane_id,ip,jp,kp,axis,sign_index,
)
    """
    reduces one lane's defect or nonzero message into the current best cost
    inputs: current best, geometric distance, junction, candidate lane/site,
        and message component
    output: updated best cost; NO_LANE_ID is ignored
    """
    if lane_id == NO_LANE_ID
        return best
    end
    lane = get_target_junction_lane(junction,lane_id)
    if lane.defects[ip,jp,kp]
        best = min(best,distance)
    end
    incoming = lane.messages[ip,jp,kp,axis,sign_index]
    if incoming != 0
        best = min(best,incoming + distance)
    end
    return best
end

function compute_target_junction_site_messages(
    junction::TargetJunctionDecoder,destination_lane_id::UInt8,i,j,k,
)
    """
    computes all six messages at one valid Y-graph destination
    inputs: junction, destination lane identifier, and coordinates i,j,k
    output: 3 x 2 message matrix; does not mutate the frozen lane banks
    uses the ordinary 3 x 3 candidate planes and 1-norm distances with moving
    interface ownership
    """
    values = zeros(Int,3,2)
    L,_,Z = size(junction.post_cnot_target_lane.defects)
    g = junction.interface_depth
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
            lane_a,lane_b = get_target_junction_candidate_lane_ids(
                destination_lane_id,k,kp,g,
            )
            best = reduce_target_junction_message_candidate(
                best,distance,junction,lane_a,ip,jp,kp,axis,sign_index,
            )
            best = reduce_target_junction_message_candidate(
                best,distance,junction,lane_b,ip,jp,kp,axis,sign_index,
            )
        end
        values[axis,sign_index] = best == typemax(Int) ? 0 : best
    end
    return values
end

function update_target_junction_messages!(junction::TargetJunctionDecoder)
    """
    performs one globally synchronous Jacobi sweep over the valid Y graph
    input: junction: TargetJunctionDecoder
    output: nothing; commits all three message banks together and caches both
        branch-specific temporal costs from the same frozen state
    """
    L,_,Z = size(junction.post_cnot_target_lane.defects)
    g = junction.interface_depth
    for lane in (
        junction.pre_cnot_control_lane,
        junction.pre_cnot_target_lane,
        junction.post_cnot_target_lane,
    )
        lane.next_messages .= 0
    end

    if g > 0
        for i in 1:L, j in 1:L, k in 1:g
            junction.post_cnot_target_lane.next_messages[i,j,k,:,:] .=
                compute_target_junction_site_messages(
                    junction,POST_CNOT_TARGET_LANE_ID,i,j,k,
                )
        end
    end
    if g < Z
        for lane_id in (PRE_CNOT_CONTROL_LANE_ID,PRE_CNOT_TARGET_LANE_ID)
            lane = get_target_junction_lane(junction,lane_id)
            for i in 1:L, j in 1:L, k in g+1:Z
                lane.next_messages[i,j,k,:,:] .= compute_target_junction_site_messages(
                    junction,lane_id,i,j,k,
                )
            end
        end
    end

    # Preserve the two branch costs from the same frozen message state that
    # produced the merged post-CNOT temporal message. Recomputing these after
    # committing the Jacobi sweep would use a one-sweep-newer branch message and
    # could route a defect through a branch that did not supply its minimum.
    junction.branch_crossing_costs .= 0
    if 1 <= g < Z
        for i in 1:L, j in 1:L
            junction.branch_crossing_costs[i,j,1] =
                compute_target_junction_branch_crossing_cost(junction,PRE_CNOT_CONTROL_LANE_ID,i,j)
            junction.branch_crossing_costs[i,j,2] =
                compute_target_junction_branch_crossing_cost(junction,PRE_CNOT_TARGET_LANE_ID,i,j)
        end
    end

    # Commit every lane only after the complete Y graph has been evaluated.
    for lane in (
        junction.pre_cnot_control_lane,
        junction.pre_cnot_target_lane,
        junction.post_cnot_target_lane,
    )
        lane.messages .= lane.next_messages
    end
    return nothing
end

function compute_target_junction_branch_crossing_cost(
    junction::TargetJunctionDecoder,lane_id::UInt8,i,j,
)
    """
    computes a branch-specific +k crossing cost at the moving interface
    inputs: junction, pre-CNOT lane identifier, and post-CNOT site i,j
    output: smallest positive cost into branch slice g+1, or zero if absent
    """
    L,_,Z = size(junction.post_cnot_target_lane.defects)
    g = junction.interface_depth
    if !(1 <= g < Z) ||
       !(lane_id in (PRE_CNOT_CONTROL_LANE_ID,PRE_CNOT_TARGET_LANE_ID))
        return 0
    end
    ind(index) = mod1(index,L)
    lane = get_target_junction_lane(junction,lane_id)
    best = typemax(Int)
    for delta_i in -1:1, delta_j in -1:1
        ip = ind(i+delta_i)
        jp = ind(j+delta_j)
        distance = 1 + abs(delta_i) + abs(delta_j)
        if lane.defects[ip,jp,g+1]
            best = min(best,distance)
        end
        incoming = lane.messages[ip,jp,g+1,3,2]
        if incoming != 0
            best = min(best,incoming + distance)
        end
    end
    return best == typemax(Int) ? 0 : best
end

function select_target_junction_pre_cnot_lane!(state::YJunctionState,junction,i,j)
    """
    chooses the winning pre-CNOT lane for one post-CNOT defect crossing
    inputs: global state, active junction, and post-CNOT site i,j
    output: lane identifier or NO_LANE_ID
    zero means absent; the smaller positive cost wins; an equal positive tie
    selects pre-control and increments the tie counter
    """
    control_cost = junction.branch_crossing_costs[i,j,1]
    target_cost = junction.branch_crossing_costs[i,j,2]
    if control_cost == 0
        return target_cost == 0 ? NO_LANE_ID : PRE_CNOT_TARGET_LANE_ID
    elseif target_cost == 0
        return PRE_CNOT_CONTROL_LANE_ID
    elseif control_cost == target_cost
        state.equal_branch_cost_ties += 1
        return PRE_CNOT_CONTROL_LANE_ID
    elseif control_cost < target_cost
        return PRE_CNOT_CONTROL_LANE_ID
    end
    return PRE_CNOT_TARGET_LANE_ID
end

function select_target_junction_correction_links!(
    state::YJunctionState,junction::TargetJunctionDecoder;
    decoder_rng=Random.default_rng(),
)
    """
    selects frozen ordinary correction links and junction branch choices
    inputs: global state, active junction, and decoder_rng
    output: nothing; fills all lane correction links and junction branch choices
    post-CNOT defects cross into one pre-CNOT lane; pre-CNOT defects stay owned
    and move only toward larger k
    """
    L,_,Z = size(junction.post_cnot_target_lane.defects)
    g = junction.interface_depth
    ind(index) = mod1(index,L)
    junction.branch_choices .= NO_LANE_ID
    for lane in (
        junction.pre_cnot_control_lane,
        junction.pre_cnot_target_lane,
        junction.post_cnot_target_lane,
    )
        lane.correction_links .= false
    end

    # Pre-CNOT lanes retain ordinary one-way defect aging toward larger k.
    if g < Z
        for lane_id in (PRE_CNOT_CONTROL_LANE_ID,PRE_CNOT_TARGET_LANE_ID)
            lane = get_target_junction_lane(junction,lane_id)
            for i in 1:L, j in 1:L, k in g+1:Z
                if !lane.defects[i,j,k]
                    continue
                end
                if k < Z && any(!iszero,@view lane.messages[i,j,k,:,:])
                    mindist = minimum(lane.messages[i,j,k,:,:][lane.messages[i,j,k,:,:] .> 0])
                    if lane.messages[i,j,k,3,2] == mindist
                        lane.correction_links[i,j,k,3] = true
                    elseif lane.messages[i,j,k,1,1] == mindist
                        lane.correction_links[ind(i-1),j,k,1] = true
                    elseif lane.messages[i,j,k,2,1] == mindist
                        lane.correction_links[i,ind(j-1),k,2] = true
                    elseif lane.messages[i,j,k,2,2] == mindist
                        lane.correction_links[i,j,k,2] = true
                    elseif lane.messages[i,j,k,1,2] == mindist
                        lane.correction_links[i,j,k,1] = true
                    end
                # Backwall
                elseif k == Z && any(!iszero,@view lane.messages[i,j,k,1:2,:]) &&
                       rand(decoder_rng) < 0.8
                    mindist = minimum(lane.messages[i,j,k,1:2,:][lane.messages[i,j,k,1:2,:] .> 0])
                    if lane.messages[i,j,k,1,1] == mindist
                        lane.correction_links[ind(i-1),j,k,1] = true
                    elseif lane.messages[i,j,k,2,1] == mindist
                        lane.correction_links[i,ind(j-1),k,2] = true
                    elseif lane.messages[i,j,k,2,2] == mindist
                        lane.correction_links[i,j,k,2] = true
                    elseif lane.messages[i,j,k,1,2] == mindist
                        lane.correction_links[i,j,k,1] = true
                    end
                end
            end
        end
    end

    # Post-CNOT lane
    post_cnot_lane = junction.post_cnot_target_lane
    if g > 0
        for i in 1:L, j in 1:L, k in 1:g
            if !post_cnot_lane.defects[i,j,k] ||
               !any(!iszero,@view post_cnot_lane.messages[i,j,k,:,:])
                continue
            end
            positive_messages = post_cnot_lane.messages[i,j,k,:,:][
                post_cnot_lane.messages[i,j,k,:,:] .> 0
            ]
            mindist = minimum(positive_messages)
            # Move in time direction
            if post_cnot_lane.messages[i,j,k,3,2] == mindist
                # Inside trunk, regular move
                if k < g
                    post_cnot_lane.correction_links[i,j,k,3] = true
                # At the junction, choose a branch
                else
                    lane_id = select_target_junction_pre_cnot_lane!(state,junction,i,j)
                    junction.branch_choices[i,j] = lane_id
                    if lane_id == PRE_CNOT_CONTROL_LANE_ID
                        state.pre_cnot_control_crossings += 1
                    elseif lane_id == PRE_CNOT_TARGET_LANE_ID
                        state.pre_cnot_target_crossings += 1
                    end
                end
            elseif post_cnot_lane.messages[i,j,k,1,1] == mindist
                post_cnot_lane.correction_links[ind(i-1),j,k,1] = true
            elseif post_cnot_lane.messages[i,j,k,2,1] == mindist
                post_cnot_lane.correction_links[i,ind(j-1),k,2] = true
            elseif post_cnot_lane.messages[i,j,k,2,2] == mindist
                post_cnot_lane.correction_links[i,j,k,2] = true
            elseif post_cnot_lane.messages[i,j,k,1,2] == mindist
                post_cnot_lane.correction_links[i,j,k,1] = true
            end
        end
    end
    return nothing
end

function commit_target_junction_correction_links!(junction::TargetJunctionDecoder,target_correction_frame)
    """
    atomically commits all target-lane and junction correction_links
    inputs: active junction and the one shared target Pauli frame
    output: nothing; updates frame parity and defect endpoints
    each branch choice toggles the post endpoint and exactly one pre-CNOT endpoint
    """
    # Regular
    for lane in (
        junction.pre_cnot_control_lane,
        junction.pre_cnot_target_lane,
        junction.post_cnot_target_lane,
    )
        commit_decoder_lane_correction_links!(lane,target_correction_frame)
    end
    # Cross junction
    g = junction.interface_depth
    _,_,Z = size(junction.post_cnot_target_lane.defects)
    if 1 <= g < Z
        for i in axes(junction.branch_choices,1),
            j in axes(junction.branch_choices,2)
            lane_id = junction.branch_choices[i,j]
            if lane_id == PRE_CNOT_CONTROL_LANE_ID ||
               lane_id == PRE_CNOT_TARGET_LANE_ID
                junction.post_cnot_target_lane.defects[i,j,g] ⊻= true
                get_target_junction_lane(junction,lane_id).defects[i,j,g+1] ⊻= true
            end
        end
    end
    return nothing
end

function clear_target_junction_unowned_slices!(junction::TargetJunctionDecoder)
    """
    enforces moving-interface lane ownership
    input: active junction at depth g
    output: nothing; zeros post slices k > g and pre-branch slices k <= g,
        including defects, messages, scratch messages, and correction links
    """
    _,_,Z = size(junction.post_cnot_target_lane.defects)
    g = junction.interface_depth
    if g < Z
        junction.post_cnot_target_lane.defects[:,:,g+1:Z] .= false
        junction.post_cnot_target_lane.messages[:,:,g+1:Z,:,:] .= 0
        junction.post_cnot_target_lane.next_messages[:,:,g+1:Z,:,:] .= 0
        junction.post_cnot_target_lane.correction_links[:,:,g+1:Z,:] .= false
    end
    if g > 0
        for lane in (junction.pre_cnot_control_lane,junction.pre_cnot_target_lane)
            lane.defects[:,:,1:g] .= false
            lane.messages[:,:,1:g,:,:] .= 0
            lane.next_messages[:,:,1:g,:,:] .= 0
            lane.correction_links[:,:,1:g,:] .= false
        end
    end
    return nothing
end

function collapse_target_junction!(state::YJunctionState,junction::TargetJunctionDecoder)
    """
    collapses the active target Y-junction at g = Z
    inputs: global state and active junction
    output: nothing; xors both pre-CNOT defects into the post-CNOT back wall,
        zero-aware-minimizes spatial messages, clears temporal/scratch state,
        and replaces the target junction with the ordinary post-CNOT lane
    """
    post_cnot_lane = junction.post_cnot_target_lane
    _,_,Z = size(post_cnot_lane.defects)
    post_cnot_lane.defects[:,:,Z] .⊻=
        junction.pre_cnot_control_lane.defects[:,:,Z]
    post_cnot_lane.defects[:,:,Z] .⊻=
        junction.pre_cnot_target_lane.defects[:,:,Z]
    post_cnot_lane.messages[:,:,Z,1:2,:] .= nonzero_minimum.(
        post_cnot_lane.messages[:,:,Z,1:2,:],
        junction.pre_cnot_control_lane.messages[:,:,Z,1:2,:],
        junction.pre_cnot_target_lane.messages[:,:,Z,1:2,:],
    )
    post_cnot_lane.messages[:,:,Z,3,:] .= 0
    post_cnot_lane.next_messages .= 0
    post_cnot_lane.correction_links .= false
    state.target_decoder = post_cnot_lane
    state.collapse_round = state.completed_rounds + 1
    return nothing
end

function age_target_junction!(
    state::YJunctionState,junction::TargetJunctionDecoder,new_event,
)
    """
    ages all three target lanes and advances the junction by one round
    inputs: global state, active junction, and new observable target event
    output: nothing; cycles lanes, inserts the post event, and collapses at g = Z
    """
    _,_,Z = size(junction.post_cnot_target_lane.defects)
    for lane in (
        junction.pre_cnot_control_lane,
        junction.pre_cnot_target_lane,
        junction.post_cnot_target_lane,
    )
        age_decoder_lane!(lane)
    end
    junction.interface_depth += 1
    junction.post_cnot_target_lane.defects[:,:,1] .= new_event
    if junction.interface_depth == Z
        collapse_target_junction!(state,junction)
    else
        clear_target_junction_unowned_slices!(junction)
    end
    return nothing
end

function update_target_junction!(
    state::YJunctionState,junction::TargetJunctionDecoder,r,
    data_errors,measurement_errors;decoder_rng=Random.default_rng(),
)
    """
    runs one synchronous target round while the junction is active
    inputs: state, junction, message-sweep count r, explicit target masks,
        and decoder_rng
    output: nothing; performs Y messages, atomic feedback, one physical channel,
        and junction aging in that order
    """
    if r < 1
        error("Y-junction CNOT requires r >= 1")
    end
    for _ in 1:r
        update_target_junction_messages!(junction)
    end
    select_target_junction_correction_links!(state,junction;decoder_rng=decoder_rng)
    target_block = state.physical_blocks[TARGET_BLOCK_ID]
    commit_target_junction_correction_links!(junction,target_block.correction_frame)
    new_event = apply_noise_to_block!(
        target_block,data_errors,measurement_errors,
    )
    age_target_junction!(state,junction,new_event)
    return nothing
end

function apply_yjunction_cnot_x!(
    state::YJunctionState,
    control_block_id=CONTROL_BLOCK_ID,
    target_block_id=TARGET_BLOCK_ID,
)
    """
    applies the sole supported ideal X-sector CNOT from block 1 to block 2
    inputs: pre-CNOT state and optional fixed control/target identifiers
    output: nothing; xors the target errors, correction frame, and syndrome
        registers with the control; moves the target lane into
        pre_cnot_target_lane; snapshots control into pre_cnot_control_lane;
        and creates an empty post-CNOT lane at g = 0
    """
    if state.has_applied_cnot
        error("the Y-junction prototype supports exactly one CNOT")
    elseif control_block_id != CONTROL_BLOCK_ID ||
           target_block_id != TARGET_BLOCK_ID ||
           length(state.physical_blocks) != 2
        error("the Y-junction prototype supports block 1 -> block 2 only")
    elseif !(state.target_decoder isa DecoderLane)
        error("target decoder is not in its pre-CNOT single-lane state")
    end

    control_block = state.physical_blocks[control_block_id]
    target_block = state.physical_blocks[target_block_id]
    target_block.errors .⊻= control_block.errors
    target_block.correction_frame .⊻= control_block.correction_frame
    target_block.previous_syndrome .⊻= control_block.previous_syndrome
    target_block.current_syndrome .⊻= control_block.current_syndrome

    pre_cnot_control_lane = copy_decoder_lane(state.control_decoder)
    pre_cnot_target_lane = state.target_decoder
    pre_cnot_target_lane.next_messages .= 0
    pre_cnot_target_lane.correction_links .= false
    L,_,Z = size(pre_cnot_target_lane.defects)
    post_cnot_target_lane = initialize_decoder_lane(L,Z)
    state.target_decoder = TargetJunctionDecoder(
        pre_cnot_control_lane,
        pre_cnot_target_lane,
        post_cnot_target_lane,
        0,
        zeros(Int,L,L,2),
        fill(NO_LANE_ID,L,L),
    )
    state.has_applied_cnot = true
    state.cnot_round = state.completed_rounds
    state.max_target_lane_count = 3
    return nothing
end

function update_yjunction_state!(
    state::YJunctionState,r,p,q;
    synch=true,pretty=false,masks=nothing,
    noise_rng=Random.default_rng(),decoder_rng=nothing,
)
    """
    advances both observable blocks by one synchronous round
    inputs: state, r, data rate p, measurement rate q, synchronous/pretty flags,
        optional RoundNoiseMasks, noise_rng, and decoder_rng
    output: the masks actually applied; mutates both blocks and decoder lanes
    control uses the ordinary kernel; target uses the active Y kernel or the
    ordinary post-collapse lane
    """
    if !synch
        error("Y-junction CNOT supports synchronous updates only")
    elseif pretty
        error("Y-junction CNOT does not implement pretty updates")
    end
    L = size(state.physical_blocks[CONTROL_BLOCK_ID].errors,1)
    round_masks = masks === nothing ?
        sample_round_noise_masks(noise_rng,L,p,q) :
        validate_round_noise_masks(masks,L)
    if decoder_rng === nothing
        decoder_rng = Random.Xoshiro(rand(noise_rng,UInt64))
    end

    update_decoder_lane!(
        state.physical_blocks[CONTROL_BLOCK_ID],
        state.control_decoder,
        r,
        round_masks.data_errors[CONTROL_BLOCK_ID],
        round_masks.measurement_errors[CONTROL_BLOCK_ID];
        decoder_rng=decoder_rng,
    )

    target_block = state.physical_blocks[TARGET_BLOCK_ID]
    target_decoder = state.target_decoder
    if target_decoder isa TargetJunctionDecoder
        update_target_junction!(
            state,target_decoder,r,
            round_masks.data_errors[TARGET_BLOCK_ID],
            round_masks.measurement_errors[TARGET_BLOCK_ID];
            decoder_rng=decoder_rng,
        )
    else
        update_decoder_lane!(
            target_block,target_decoder,r,
            round_masks.data_errors[TARGET_BLOCK_ID],
            round_masks.measurement_errors[TARGET_BLOCK_ID];
            decoder_rng=decoder_rng,
        )
    end
    state.completed_rounds += 1
    return round_masks
end

function has_no_decoder_defects(state::YJunctionState)
    """
    input: YJunctionState
    output: true only when every currently owned decoder lane has no defects
    """
    if any(state.control_decoder.defects)
        return false
    end
    target_decoder = state.target_decoder
    if target_decoder isa TargetJunctionDecoder
        return !any(target_decoder.pre_cnot_control_lane.defects) &&
               !any(target_decoder.pre_cnot_target_lane.defects) &&
               !any(target_decoder.post_cnot_target_lane.defects)
    end
    return !any(target_decoder.defects)
end

function is_target_junction_collapsed(state::YJunctionState)
    """
    input: YJunctionState
    output: true if the CNOT occurred and the transient target junction collapsed
    """
    return state.has_applied_cnot && state.target_decoder isa DecoderLane
end

function count_target_decoder_lanes(state::YJunctionState)
    """
    input: YJunctionState
    output: three target lanes while active, or one ordinary target lane otherwise
    """
    return state.target_decoder isa TargetJunctionDecoder ? 3 : 1
end

function count_decoder_message_buffer_pairs(state::YJunctionState)
    """
    input: YJunctionState
    output: total control-plus-target count of messages/next_messages lane pairs
    """
    return 1 + count_target_decoder_lanes(state)
end

function compute_decoded_block_edges(state::YJunctionState,block_id)
    """
    inputs: global state and observable block identifier
    output: local decoded edge configuration: errors xor correction frame
    """
    physical_block = state.physical_blocks[block_id]
    return physical_block.errors .⊻ physical_block.correction_frame
end

function split_cnot_timing(total_time)
    """
    splits total noisy time into floor/ceiling pre/post intervals
    input: positive total noisy time T
    output: T_PRE, T_POST, and a 2T cleanup cap
    """
    if total_time < 1
        error("CNOT total time T must be positive")
    end
    pre_time = fld(total_time,2)
    post_time = total_time - pre_time
    return pre_time,post_time,2total_time
end

function estimate_yjunction_cnot_fidelity(
    L,Z,p,q,r,synch,pretty,T_PRE,T_POST,CLEANUP_TIME,
    acc_err,fixed_samps,trial_parallel,verbose,
)
    """
    estimates joint control/target CNOT fixed-time fidelity
    inputs:
        L,Z: lattice size and buffer depth
        p,q,r: data rate, measurement rate, and message-update ratio
        synch,pretty: only true,false is supported
        T_PRE,T_POST,CLEANUP_TIME: protocol schedule
        acc_err,fixed_samps: failure-accumulation or fixed-sample stopping rule
        trial_parallel,verbose: trial threading and progress controls
    output: dictionary of logical, cleanup, crossing, collapse, memory, runtime,
        and lane-count metrics
    """
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
        pre_cnot_control_crossings = 0
        pre_cnot_target_crossings = 0
        equal_branch_cost_ties = 0
        collapse_delay_sum = 0
        collapsed_trials = 0
        gate_bytes_sum = 0
        final_bytes_sum = 0

        while use_fixed_samps ? trials < local_samps : failures < target_errors
            if verbose && trials % 10000 == 0
                println("thread $(threadid()) Y-junction trial: ",trials)
            end
            state = initialize_yjunction_state(L,Z)
            for _ in 1:T_PRE
                update_yjunction_state!(state,r,p,q)
            end
            apply_yjunction_cnot_x!(state)
            gate_bytes_sum += Base.summarysize(state)
            for _ in 1:T_POST
                update_yjunction_state!(state,r,p,q)
            end
            for _ in 1:CLEANUP_TIME
                if has_no_decoder_defects(state) && is_target_junction_collapsed(state)
                    break
                end
                update_yjunction_state!(state,r,0.0,0.0)
            end

            defect_cleanup_failed = !has_no_decoder_defects(state)
            collapse_failed = !is_target_junction_collapsed(state)
            cleanup_failures += defect_cleanup_failed
            collapse_failures += collapse_failed
            if !collapse_failed
                collapse_delay_sum += state.collapse_round - state.cnot_round
                collapsed_trials += 1
            end

            decoded_control = compute_decoded_block_edges(state,CONTROL_BLOCK_ID)
            decoded_target = compute_decoded_block_edges(state,TARGET_BLOCK_ID)
            if !defect_cleanup_failed
                @assert !any(compute_syndrome(decoded_control)) "decoded control is not syndrome-free"
                @assert !any(compute_syndrome(decoded_target)) "decoded target is not syndrome-free"
            end
            control_failed = !is_logically_trivial(decoded_control)
            target_failed = !is_logically_trivial(decoded_target)
            logical_failed = control_failed || target_failed

            failures += logical_failed
            control_failures += control_failed
            target_failures += target_failed
            both_failures += control_failed && target_failed
            pre_cnot_control_crossings += state.pre_cnot_control_crossings
            pre_cnot_target_crossings += state.pre_cnot_target_crossings
            equal_branch_cost_ties += state.equal_branch_cost_ties
            final_bytes_sum += Base.summarysize(state)
            trials += 1
        end
        return (
            failures,trials,control_failures,target_failures,both_failures,
            cleanup_failures,collapse_failures,pre_cnot_control_crossings,
            pre_cnot_target_crossings,equal_branch_cost_ties,collapse_delay_sum,collapsed_trials,
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
    """
    runs a zero-noise end-to-end assertion path
    inputs: L,Z,r and the pre/post/cleanup schedule
    output: nothing; checks gate activation, non-aliasing, collapse, cleanup,
        and trivial logical readout
    """
    state = initialize_yjunction_state(L,Z)
    for _ in 1:T_PRE
        update_yjunction_state!(state,r,0.0,0.0)
    end
    apply_yjunction_cnot_x!(state)
    @assert state.target_decoder isa TargetJunctionDecoder
    @assert state.target_decoder.pre_cnot_control_lane.defects !== state.control_decoder.defects
    for _ in 1:T_POST
        update_yjunction_state!(state,r,0.0,0.0)
    end
    for _ in 1:CLEANUP_TIME
        if has_no_decoder_defects(state) && is_target_junction_collapsed(state)
            break
        end
        update_yjunction_state!(state,r,0.0,0.0)
    end
    @assert is_target_junction_collapsed(state)
    @assert has_no_decoder_defects(state)
    @assert is_logically_trivial(compute_decoded_block_edges(state,CONTROL_BLOCK_ID))
    @assert is_logically_trivial(compute_decoded_block_edges(state,TARGET_BLOCK_ID))
    println("Y-junction sanity checks passed")
    return nothing
end

function parse_boolean(value)
    """
    input: string-like environment value
    output: true for 1, true, yes, or on after lowercasing; false otherwise
    """
    return lowercase(value) in ("1","true","yes","on")
end

function write_yjunction_output(path,params,data)
    """
    writes sorted result and parameter sections to a text file
    inputs: path, parameter dictionary, and result dictionary
    output: absolute written path; creates missing parent directories
    """
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
    """
    environment-driven command-line entry point
    inputs: MODE and lattice/noise/timing/sample settings from ENV
    output: nothing; runs CNOT_Ft or CNOT_DEBUG, prints metrics, and optionally
        writes OUTPUT_FILE
    """
    mode = get(ENV,"MODE","CNOT_Ft")
    L = parse(Int,get(ENV,"LVAL",mode == "CNOT_DEBUG" ? "3" : "13"))
    logz = parse_boolean(get(ENV,"LOGZ","true"))
    Z = parse(Int,get(ENV,"ZVAL",string(
        logz ? ceil(Int,log(1.5,L)) : ceil(Int,L/4),
    )))
    p = parse(Float64,get(ENV,"PVAL","0.011"))
    qrat = parse(Float64,get(ENV,"QRAT","1.0"))
    q = p * qrat
    r = parse(Int,get(ENV,"RVAL","3"))
    synch = parse_boolean(get(ENV,"SYNCH","true"))
    pretty = parse_boolean(get(ENV,"PRETTY","false"))
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
    trial_parallel = parse_boolean(get(ENV,"TRIAL_PARALLEL","true"))
    verbose = parse_boolean(get(ENV,"VERBOSE","false"))

    if mode == "CNOT_DEBUG"
        run_yjunction_sanity_checks(L,Z,r,T_PRE,T_POST,cleanup_time)
        return nothing
    elseif mode != "CNOT_Ft"
        error("Y-junction driver supports MODE=CNOT_Ft or MODE=CNOT_DEBUG")
    end

    data = estimate_yjunction_cnot_fidelity(
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
