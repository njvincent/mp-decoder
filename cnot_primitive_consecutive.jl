"""
imported from common functions
"""

using Random
using Alert
using Dates
using Base.Threads

# Consecutive primitive CNOT driver.
#
# This file intentionally reuses the one-sector memory decoder below and adds
# a minimal many-control/one-target experiment. It is not a physically
# complete labeled-defect CNOT decoder; it only applies the X-sector rule
# c_out = c, t_out = c xor t to the tracked decoder arrays.
# 
# Modifications:
# 1. Each control and the one persistent target own independent DecoderBlock
#    storage. Every live block receives one independent noise/decoder update
#    per noisy round.
# 2. CNOT_INTERVALS specifies the independently configurable noisy intervals:
#    interval 1, C1 -> T, interval 2, C2 -> T, ..., final interval, cleanup.
# 3. The update rule for CNOT is that
#    a. state, state_correction, old_synds, new_synds, and hist of the control 
#       are unchanged, but the target are XORed with the control.
#    b. fields of the control is unchanged. fields of the target takes the 
#       non-zero-min of control and target.
#    c. new_fields for both the control and the target are set to zero.
# 4. A trial is counted as failure if any control or the target has a logical
#    failure after the cleanup.


function nonzeromin(a, b)
    if a == 0
        return b
    elseif b == 0
        return a
    else
        return min(a, b)
    end
end

function get_synds(state)
    """ 
    calculates syndromes 
    """
    L = size(state)[1]
    ind(i) = mod1(i,L)
    synds = falses(L,L)
    for i in 1:L, j in 1:L
        im1 = ind(i-1); jm1 = ind(j-1)
        synds[i,j] = state[i,j,1] ⊻ state[i,j,2] ⊻ state[im1,j,1] ⊻ state[i,jm1,2]
    end 

    return synds

end 

function detect_logical_error(state)
    """
    input: anyon-free state 
    output: true if both cycles have trivial winding; false otherwise 
    errors are X strings; detect logical error with perpendicular Z strings
    """
    Lx,Ly,_ = size(state)
    xparity = false; yparity = false
    for i in 1:Lx 
        xparity ⊻= state[i,1,2]
    end 
    for j in 1:Ly
        yparity ⊻= state[1,j,1]
    end
    return ~xparity && ~yparity 
end

"""
simulation of the 2D windowed message passing decoder
code is unfortunately still somewhat of a mess; will be cleaned up eventually 
"""

function onesite_field_update(i,j,k,fields,hist)
    """
    given fields and a space"time" point i,j,k returns the new value of the field at i,j,k 
    inputs: 
        i,j,k: indices of the space/time point to update (k is rg time)
        fields: L x L x Z x 3 x 2 of Ints (last two indices are space/time dimensions and forward/backward)
        hist: L x L x Z of Bools (history of synd-changing events)
    outputs: 
        new_fields: 3 x 2 matrix of ints, equal to the new value of the fields at site i,j 
        (does not modify fields directly)
    """

    new_fields = zeros(Int,3,2)
    L,_,Z = size(hist); ind(i) = mod1(i,L); zind(i) = i < 1 ? 1 : (i > Z ? Z : i) #if i < 1, return 1; if i > Z, return Z; else, return i
    
    function ca_update!(a,s)
        """ 
        helper function to do the CA update at site (i,j,k) in direction sa, s ∈ ±1; a ∈ 1...3 
        """
        newfield = Inf
        # s = +1 -> sind = 1, s = -1 -> sind = 2
        sind = s == 1 ? 1 : 2
        ip = 0; jp = 0; kp = 0 # neighbor coordinates being inspected
        for delta1 in -1:1, delta2 in -1:1 
            if a == 1 
                ip = ind(i-s); jp = ind(j+delta1); kp = zind(k+delta2)
            elseif a == 2 
                ip = ind(i+delta1); jp = ind(j-s); kp = zind(k+delta2)
            elseif a == 3 
                ip = ind(i+delta1); jp = ind(j+delta2); kp = zind(k-s)
            end 
            dist = abs(s) + abs(delta1) + abs(delta2) # 1-norm distance to site (ip,jp,kp)
            if hist[ip,jp,kp] 
                newfield = min(newfield,dist)
            end 
            if fields[ip,jp,kp,a,sind] != 0 
                newfield = min(newfield,fields[ip,jp,kp,a,sind]+dist) 
            end
        end 
        new_fields[a,sind] = newfield == Inf ? 0 : newfield
    end 

    for a in 1:3, s in [-1 1]
        ca_update!(a,s)
    end 

    return new_fields 
end 

@views function update_2d_windowed_fields!(fields,new_fields,hist)
    """
    performs synchronous CA updates on the fields, using the history of synd-changing events
    inputs: 
        fields, new_fields: L x L x Z x 3 x 2 of Ints (last two indices are space/time and forward/backward)
        hist: L x L x Z of Bools (history of synd-changing events)
    outputs: 
        nothing 
    """
    L,_,Z = size(hist)
    for i in 1:L, j in 1:L, k in 1:Z 
        new_fields[i,j,k,:,:] .= onesite_field_update(i,j,k,fields,hist)
    end

    fields .= new_fields 
    return nothing 
end 

@views function update_2d_windowed_fields_column!(fields,new_fields,hist,i,j) 
    """
    does asynchronous update of the fields stored in a single processor at site i,j 
    inputs: 
        fields, new_fields: L x L x Z x 3 x 2 of Ints (last two indices are space/time and forward/backward)
        hist: L x L x Z of Bools (history of synd-changing events)
        i,j: location of processor where update occurs 
    outputs: 
        nothing 
    """
    _,_,Z = size(hist)
    for k in 1:Z 
        new_fields[i,j,k,:,:] .= onesite_field_update(i,j,k,fields,hist)
    end

    fields[i,j,:,:,:] .= new_fields[i,j,:,:,:]
    return nothing 
end

@views function rg_cycle!(hist,fields)
    """
    does synchronous RG cycle by moving fields and syndromes along the RG direction
    inputs: 
        hist: L x L x Z of Bools (history of synd-changing events)
        fields: L x L x Z x 3 x 2 of Ints (fields at each RG time, and in space/rg_time and forward/backward directions)
    outputs: 
        nothing 
    """

    L,_,Z = size(hist)

    ### update history ### 
    # splatter anyons onto back wall 
    hist[:,:,Z] .= xor.(hist[:,:,Z], hist[:,:,Z-1])  
            
    # cycle anyons along the RG direction 
    copyto!(hist[:,:,2:end-1], hist[:,:,1:end-2])

    # clear the history at zero RG time 
    hist[:,:,1] .= false 

    ### update fields ### 
    # cycle fields along the RG direction 
    fields[:,:,Z,1:2,:] .= nonzeromin.(fields[:,:,Z-1,1:2,:],fields[:,:,Z,1:2,:]) # only keeping the spatial message fields on the back wall 
    copyto!(fields[:,:,2:end-1,:,:],fields[:,:,1:end-2,:,:])

    # clear the messages at zero RG time 
    fields[:,:,1,:,:] .= 0
    
    return nothing 
end

@views function rg_cycle_column!(hist,fields,i,j) # (in practice the @views basically doesn't give any speedup)
    """
    single-site variant of rg_cycle! function; updates all fields at processor i,j 
    """

    _,_,Z = size(hist)

    ### update history ### 
    # splatter anyons onto back wall 
    hist[i,j,Z] = xor(hist[i,j,Z],hist[i,j,Z-1])  

    copyto!(hist[i,j,2:end-1],hist[i,j,1:end-2])

    # clear the history at zero RG time 
    hist[i,j,1] = false 

    ### update fields ### 
    # cycle fields along the RG direction 
    fields[i,j,Z,1:2,:] .= nonzeromin.(fields[i,j,Z-1,1:2,:],fields[i,j,Z,1:2,:]) # only keeping the spatial message fields on the back wall 
    copyto!(fields[i,j,2:end-1,:,:],fields[i,j,1:end-2,:,:])

    # clear the messages at zero RG time 
    fields[i,j,1,:,:] .= 0
    
    return nothing 
end

function perform_correction!(hist,hist_correction)
    """
    performs synchronous corrections on the history of synd-changing events
    inputs: 
        hist: L x L x Z of Bools (history of synd-changing events)
        hist_correction: L x L x Z x 3 of Bools 
    outputs: 
        nothing 
    """
    L,_,Z = size(hist); ind(i) = mod1(i,L)
    for i in 1:L, j in 1:L, k in 1:Z
        if hist_correction[i,j,k,1] 
            hist[i,j,k] ⊻= true; hist[ind(i+1),j,k] ⊻= true 
        end
        if hist_correction[i,j,k,2] 
            hist[i,j,k] ⊻= true; hist[i,ind(j+1),k] ⊻= true 
        end
        if hist_correction[i,j,k,3] # this will always be false for k = Z 
            hist[i,j,k] ⊻= true; hist[i,j,k+1] ⊻= true 
        end
    end 

    return nothing 
end 

function perform_correction_column!(hist, hist_correction_col,i,j)
    """ 
    version of perform_correction! that acts on a single column (z-coords) of the history. 
    used for asynch updates. 
    only impliments the vertical (z) part of the history correction  
    """
    L, _, Z = size(hist)
    ind(k) = mod1(k, L)
    for k in 1:Z
        if hist_correction_col[k] # this will always be false for k = Z
            hist[i, j, k] ⊻= true
            hist[i, j, k+1] ⊻= true
        end
    end
    return nothing
end

function anyons_source_fields!(hist,fields) # ensures that fields are always updated in the 1-balls around the each anyon's position
    #only called later to pretty up the animations
    L,_,Z = size(hist) 
    ind(i) = mod1(i,L); zind(i) = i < 1 ? 1 : (i > Z ? Z : i) # indices for periodic boundary conditions in space and RG time
    for i in 1:L, j in 1:L, k in 1:Z 
        if hist[i,j,k]
            ip1 = ind(i+1); im1 = ind(i-1)
            jp1 = ind(j+1); jm1 = ind(j-1)
            kp1 = zind(k+1); km1 = zind(k-1)
            # nearest neighbors in 1-norm: 
            fields[ip1,j,k,1,1] = 1 
            fields[im1,j,k,1,2] = 1
            fields[i,jp1,k,2,1] = 1
            fields[i,jm1,k,2,2] = 1
            fields[i,j,kp1,3,1] = 1
            fields[i,j,km1,3,2] = 1

        end 
    end 
end

function update!(state,state_correction, old_synds,new_synds, hist,hist_correction, fields,new_fields, r,p,q, synch,pretty)
    """
    for a system of size L x L and rg depth Z: 

    state, state_correction: L x L x 2 of Bools 
    old/new_synds: L x L of Bools (old ⊻ new is used to feed synd-changing events into history)
    hist: L x L x Z of Bools (history of synd-changing events) 
    hist_correction: L x L x Z x 2 of Bools (links where corrections are applied; doing it this way just to make synchronous updates easier)
    fields, new_fields: L x L x Z x 3 x 2 of Ints 
    r: ratio of field updates to spin updates
    p: error probability 
    q: measurement error probability 
    synch: if true, updates are done synchronously 
    pretty: for animation purposes
    """
    L,_,Z = size(hist)
    ind(i) = mod1(i,L); zind(i) = i < 1 ? 1 : (i > Z ? Z : i) 

    if synch 

        # update fields r times
        for _ in 1:(r-(pretty ? 1 : 0))  
            update_2d_windowed_fields!(fields,new_fields,hist) 
        end 

        # reset all proposed corrections
        hist_correction .= false 
        for i in 1:L   
            im1 = ind(i-1); ip1 = ind(i+1) # precompute neighboring indices, m stands for minus, p stands for plus
            for j in 1:L 
                jm1 = ind(j-1); jp1 = ind(j+1)
                for k in 1:Z
                    if hist[i,j,k] 
                        if k < Z # bulk motion 
                            if any(!iszero, @view fields[i,j,k,:,:]) # check whether there is any nonzero field; if so, move somewhere 
                                @views mindist = minimum(fields[i,j,k,:,:][fields[i,j,k,:,:] .> 0])
                                
                                if fields[i,j,k,3,2] == mindist # move along +z 
                                    hist_correction[i,j,k,3] = true  
                                elseif fields[i,j,k,1,1] == mindist # && fields[i,j,k,1,2] != mindist # move along -x 
                                    hist_correction[im1,j,k,1] = true 
                                elseif fields[i,j,k,2,1] == mindist # && fields[i,j,k,2,2] != mindist # move along -y
                                    hist_correction[i,jm1,k,2] = true
                                elseif fields[i,j,k,2,2] == mindist # && fields[i,j,k,2,1] != mindist # move along +y
                                    hist_correction[i,j,k,2] = true
                                elseif fields[i,j,k,1,2] == mindist # && fields[i,j,k,1,1] != mindist # move along +x
                                    hist_correction[i,j,k,1] = true
                                end
                            end 
                        else # motion on back screen---just spatial components 
                            if any(!iszero, fields[i,j,k,1:2,:]) && rand() < .8 # move somewhere -- small stochasticity can be added to break out of doppler-locked limit cycles; not important for larger system sizes 
                                mindist = minimum(fields[i,j,k,1:2,:][fields[i,j,k,1:2,:] .> 0])
                                # spatial corrections -- correct both the state and the history (doing it in an appropriate order is important in situations where degenerate field strengths arise)
                                if fields[i,j,k,1,1] == mindist # && fields[i,j,k,1,2] != mindist # move along -x 
                                    hist_correction[im1,j,k,1] = true 
                                elseif fields[i,j,k,2,1] == mindist # && fields[i,j,k,2,2] != mindist # move along -y  
                                    hist_correction[i,jm1,k,2] = true
                                elseif fields[i,j,k,2,2] == mindist # && fields[i,j,k,2,1] != mindist # move along +y  
                                    hist_correction[i,j,k,2] = true
                                elseif fields[i,j,k,1,2] == mindist # && fields[i,j,k,1,1] != mindist # move along +x 
                                    hist_correction[i,j,k,1] = true
                                end
                            end 
                        end 
                    end 
                end 
            end
        end 

        for i in 1:L, j in 1:L, a in 1:2  
            state_correction[i,j,a] ⊻= reduce(⊻, @view hist_correction[i,j,:,a]) # state correction is modified by xoring the rg columns of hist_correction 
        end 
        perform_correction!(hist,hist_correction) # update the history 

        # apply noise to state and calculate new syndrome-changing events to feed into hist 
        if p > 0
            state .⊻= (rand(L,L,2) .< p) # apply noise to state 
        end 

        old_synds .= new_synds 
        new_synds .= get_synds(state) 
        if q > 0
            new_synds .⊻= (rand(L,L) .< q) # get new syndromes with measurement errors
        end 

        rg_cycle!(hist,fields) # cycle the history and fields to make room for the new data 
        @views hist[:,:,1] .= (old_synds .⊻ new_synds)  # store the syndromes that changed in the history and include noise 
        
        # include these and do one less field update above in order to make prettier looking animations without any shockwaves 
        if pretty 
            anyons_source_fields!(hist,fields)
            update_2d_windowed_fields!(fields,new_fields,hist)
        end 

    ### asychronous updates ### 
    else 
        # stores proposed corrections along the RG-time direction for a single spatial column (i,j).
        # reused each time a new column is updated.
        vertical_correction = falses(Z)

        for _ in 1:((r+1)*L^2) # chosen so that we have L^2 feedback / rg-cycle / noise updates in total on average 
            i = rand(1:L); j = rand(1:L)  # pick column to update 
            # probability of field update = r / (r+1); probability of feedback update = 1 / (r+1)
            field_update = ~(rand(1:(1+r)) == 1)
            if field_update # update fields (no rg cycling)
                update_2d_windowed_fields_column!(fields,new_fields,hist,i,j)

            else # apply feedback to column and do rg cycling and apply noise 
                ## feedback to entire column 
                im1 = ind(i-1); ip1 = ind(i+1)
                jm1 = ind(j-1); jp1 = ind(j+1)
                for k in 1:Z-1 # bulk motion 
                    if hist[i,j,k] 
                        if any(!iszero, @view fields[i,j,k,:,:]) # move somewhere 
                            @views mindist = minimum(fields[i,j,k,:,:][fields[i,j,k,:,:] .> 0])
                            
                            # need to update hist here? use vertical_correction 
                            if fields[i,j,k,3,2] == mindist # move along +z 
                                vertical_correction[k] = true 
                            elseif fields[i,j,k,1,1] == mindist # && fields[i,j,k,1,2] != mindist # move along -x 
                                # do the state correction and the history update
                                state_correction[im1,j,1] ⊻= true; hist[im1,j,k] ⊻= true; hist[i,j,k] ⊻= true 
                            elseif fields[i,j,k,2,1] == mindist # && fields[i,j,k,2,2] != mindist # move along -y
                                state_correction[i,jm1,2] ⊻= true; hist[i,jm1,k] ⊻= true; hist[i,j,k] ⊻= true 
                            elseif fields[i,j,k,2,2] == mindist # && fields[i,j,k,2,1] != mindist # move along +y
                                state_correction[i,j,2] ⊻= true; hist[i,j,k] ⊻= true; hist[i,jp1,k] ⊻= true 
                            elseif fields[i,j,k,1,2] == mindist # && fields[i,j,k,1,1] != mindist # move along +x
                                state_correction[i,j,1] ⊻= true; hist[i,j,k] ⊻= true; hist[ip1,j,k] ⊻= true 
                            end
                        end 
                    end 
                end 

                # back wall motion 
                if hist[i,j,Z]
                    if any(!iszero, fields[i,j,Z,1:2,:]) && rand() < .8 # move somewhere -- small stochasticity can be added to break out of doppler-locked limit cycles; not important for larger system sizes 
                        mindist = minimum(fields[i,j,Z,1:2,:][fields[i,j,Z,1:2,:] .> 0])
                        # spatial corrections -- correct both the state and the history (doing it in an appropriate order is important in situations where degenerate field strengths arise)
                        if fields[i,j,Z,1,1] == mindist # move along -x 
                            state_correction[im1,j,1] ⊻= true; hist[im1,j,Z] ⊻= true; hist[i,j,Z] ⊻= true 
                        elseif fields[i,j,Z,2,1] == mindist # move along -y  
                            state_correction[i,jm1,2] ⊻= true; hist[i,jm1,Z] ⊻= true; hist[i,j,Z] ⊻= true 
                        elseif fields[i,j,Z,2,2] == mindist # move along +y  
                            state_correction[i,j,2] ⊻= true; hist[i,j,Z] ⊻= true; hist[i,jp1,Z] ⊻= true 
                        elseif fields[i,j,Z,1,2] == mindist # move along +x 
                            state_correction[i,j,1] ⊻= true; hist[i,j,Z] ⊻= true; hist[ip1,j,Z] ⊻= true 
                        end
                    end 
                end 
    
                perform_correction_column!(hist,vertical_correction,i,j) # update the history along the z direction for this processor 
                vertical_correction .= false # reset the vertical correction 
    
                if rand() < p # noise (on a generically distinct site)
                    state[rand(1:L),rand(1:L),rand(1:2)] ⊻= true 
                end 

                old_synds[i,j] = new_synds[i,j]
                new_synds[i,j] = state[i,j,1] ⊻ state[i,j,2] ⊻ state[im1,j,1] ⊻ state[i,jm1,2] ⊻ (rand() < q) # get new syndrome with error 
                
                rg_cycle_column!(hist,fields,i,j) # cycle the history and fields to make room for the new data 
                hist[i,j,1] = (old_synds[i,j] ⊻ new_synds[i,j])  # store the syndromes that changed in the history and include noise 
            end 
        end 
    end 
end 

mutable struct DecoderBlock
    state::BitArray{3}
    state_correction::BitArray{3}
    old_synds::BitMatrix
    new_synds::BitMatrix
    hist::BitArray{3}
    hist_correction::BitArray{4}
    fields::Array{Int,5}
    new_fields::Array{Int,5}
end

function make_decoder_block(L,Z)
    L > 1 || error("decoder block size L must be greater than 1")
    Z > 1 || error("decoder block depth Z must be at least 2")
    return DecoderBlock(
        falses(L,L,2),
        falses(L,L,2),
        falses(L,L),
        falses(L,L),
        falses(L,L,Z),
        falses(L,L,Z,3),
        zeros(Int,L,L,Z,3,2),
        zeros(Int,L,L,Z,3,2),
    )
end

function reset_decoder_block!(block::DecoderBlock)
    block.state .= false
    block.state_correction .= false
    block.old_synds .= false
    block.new_synds .= false
    block.hist .= false
    block.hist_correction .= false
    block.fields .= 0
    block.new_fields .= 0
    return nothing
end

function update_block!(block::DecoderBlock,r,p,q,synch,pretty)
    update!(
        block.state,
        block.state_correction,
        block.old_synds,
        block.new_synds,
        block.hist,
        block.hist_correction,
        block.fields,
        block.new_fields,
        r,
        p,
        q,
        synch,
        pretty,
    )
    return nothing
end

function primitive_cnot_x_sector!(
    state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,fields_c,new_fields_c,
    state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,fields_t,new_fields_t)
    """
    Primitive X-sector CNOT propagation for two memory blocks.

    This is a baseline bookkeeping map, not a physically complete surface-code
    CNOT or labeled-defect decoder. The control block is left unchanged, while
    the target block receives the control state, correction, syndrome history,
    and syndrome registers by xor. Message fields combine with nonzero-min
    because 0 means "no message" in this decoder.
    """
    state_t .⊻= state_c
    state_correction_t .⊻= state_correction_c
    old_synds_t .⊻= old_synds_c
    new_synds_t .⊻= new_synds_c
    hist_t .⊻= hist_c
    fields_t .= nonzeromin.(fields_t,fields_c)
    new_fields_c .= 0
    new_fields_t .= 0
    return nothing
end

function primitive_cnot_x_sector!(
    control::DecoderBlock,
    target::DecoderBlock,
)
    primitive_cnot_x_sector!(
        control.state,
        control.state_correction,
        control.old_synds,
        control.new_synds,
        control.hist,
        control.fields,
        control.new_fields,
        target.state,
        target.state_correction,
        target.old_synds,
        target.new_synds,
        target.hist,
        target.fields,
        target.new_fields,
    )
    control.hist_correction .= false
    target.hist_correction .= false
    return nothing
end

function update_blocks!(
    controls::Vector{DecoderBlock},
    target::DecoderBlock,
    r,p,q,synch,pretty,
)
    """
    Apply one ordinary decoder round to every live block. Each call to
    update_block! samples fresh noise, while the target is updated exactly once
    regardless of the number of controls.
    """
    for control in controls
        update_block!(control,r,p,q,synch,pretty)
    end
    update_block!(target,r,p,q,synch,pretty)
    return nothing
end

function run_cnot_sequence!(
    controls::Vector{DecoderBlock},
    target::DecoderBlock,
    intervals::Vector{Int},
    r,p,q,synch,pretty,
)
    length(intervals) == length(controls) + 1 ||
        error("a sequence with N controls requires N + 1 intervals")
    all(>=(0), intervals) || error("CNOT intervals must be nonnegative")

    for gate_index in eachindex(controls)
        for _ in 1:intervals[gate_index]
            update_blocks!(controls,target,r,p,q,synch,pretty)
        end
        primitive_cnot_x_sector!(controls[gate_index],target)
    end

    for _ in 1:intervals[end]
        update_blocks!(controls,target,r,p,q,synch,pretty)
    end
    return nothing
end

function parse_cnot_intervals(spec)
    entries = [
        entry
        for entry in split(strip(spec),r"[\s,]+")
        if !isempty(entry)
    ]
    length(entries) >= 2 ||
        error("CNOT_INTERVALS must contain at least two round counts")
    intervals = parse.(Int,entries)
    all(>=(0), intervals) ||
        error("CNOT_INTERVALS entries must be nonnegative")
    return intervals
end

function decoder_histories_empty(
    controls::Vector{DecoderBlock},
    target::DecoderBlock,
)
    return !any(target.hist) && all(control -> !any(control.hist),controls)
end

function estimate_consecutive_cnot_Ft(
    L,Z,p,q,r,synch,pretty,intervals,CLEANUP_TIME,
    stop_mode,target,trial_parallel,verbose,
)
    """
    Monte Carlo estimate of a consecutive primitive-CNOT success probability.

    In failure-stopping mode, run until target logical failures have been
    accumulated. In trial-stopping mode, run exactly target trials.
    """
    stop_mode in ("failures", "trials") ||
        error("stop_mode must be either failures or trials")
    target > 0 || error("stopping target must be positive")
    stop_mode == "failures" && p == 0 && q == 0 &&
        error("failure-stopping mode cannot terminate at zero noise; use trials")
    length(intervals) >= 2 ||
        error("at least one CNOT requires at least two intervals")
    all(>=(0), intervals) || error("CNOT intervals must be nonnegative")
    CLEANUP_TIME >= 0 || error("cleanup time must be nonnegative")

    cnot_count = length(intervals) - 1

    worker_count = trial_parallel ? min(nthreads(), target) : 1
    worker_targets = [
        target ÷ worker_count + (worker <= target % worker_count ? 1 : 0)
        for worker in 1:worker_count
    ]
    verbose && println("worker targets = $worker_targets")
    worker_results = Vector{
        Tuple{Int,Int,Vector{Int},Int,Int,Int,Int}
    }(undef,worker_count)

    function run_cnot_trials(worker_target)
        controls = [
            make_decoder_block(L,Z)
            for _ in 1:cnot_count
        ]
        target_block = make_decoder_block(L,Z)
        decoded_states_c = [
            falses(L,L,2)
            for _ in 1:cnot_count
        ]
        decoded_state_t = falses(L,L,2)

        local_failures = 0
        local_trials = 0
        local_control_logical_failures = zeros(Int,cnot_count)
        local_any_control_logical_failures = 0
        local_target_logical_failures = 0
        local_control_target_logical_failures = 0
        local_cleanup_failures = 0

        while stop_mode == "failures" ?
            local_failures < worker_target : local_trials < worker_target
            if verbose && local_trials % 10000 == 0
                println("thread $(threadid()) CNOT trial: ", local_trials)
            end

            for control in controls
                reset_decoder_block!(control)
            end
            reset_decoder_block!(target_block)

            run_cnot_sequence!(
                controls,
                target_block,
                intervals,
                r,
                p,
                q,
                synch,
                pretty,
            )

            for _ in 1:CLEANUP_TIME
                update_blocks!(controls,target_block,r,0,0,true,pretty)
                if decoder_histories_empty(controls,target_block)
                    break
                end
            end

            for control_index in eachindex(controls)
                decoded_states_c[control_index] .=
                    controls[control_index].state .⊻
                    controls[control_index].state_correction
            end
            decoded_state_t .=
                target_block.state .⊻ target_block.state_correction
            cleanup_success = decoder_histories_empty(controls,target_block)
            if cleanup_success
                for control_index in eachindex(controls)
                    @assert(
                        !any(get_synds(decoded_states_c[control_index])),
                        "control $control_index decoded state is not syndrome-free!",
                    )
                end
                @assert !any(get_synds(decoded_state_t)) "target decoded state is not syndrome-free!"
            elseif verbose
                println("thread $(threadid()) CNOT cleanup did not remove all defects")
            end

            control_logical_failures = [
                !detect_logical_error(decoded_state)
                for decoded_state in decoded_states_c
            ]
            any_control_logical_failure = any(control_logical_failures)
            target_logical_failure = !detect_logical_error(decoded_state_t)
            cleanup_failure = !cleanup_success
            logical_failure =
                any_control_logical_failure || target_logical_failure

            local_failures += logical_failure ? 1 : 0
            for control_index in eachindex(controls)
                local_control_logical_failures[control_index] +=
                    control_logical_failures[control_index] ? 1 : 0
            end
            local_any_control_logical_failures +=
                any_control_logical_failure ? 1 : 0
            local_target_logical_failures += target_logical_failure ? 1 : 0
            local_control_target_logical_failures +=
                (any_control_logical_failure && target_logical_failure) ? 1 : 0
            local_cleanup_failures += cleanup_failure ? 1 : 0
            local_trials += 1

            if verbose && logical_failure
                progress = stop_mode == "failures" ?
                    local_failures / worker_target :
                    local_trials / worker_target
                println("thread $(threadid()) CNOT progress: $progress")
            end
        end

        return (
            local_failures,
            local_trials,
            local_control_logical_failures,
            local_any_control_logical_failures,
            local_target_logical_failures,
            local_control_target_logical_failures,
            local_cleanup_failures,
        )
    end

    @threads for worker in 1:worker_count
        worker_results[worker] = run_cnot_trials(worker_targets[worker])
    end
    verbose && println("worker results = $worker_results")

    logical_failures = sum(result[1] for result in worker_results)
    trials = sum(result[2] for result in worker_results)
    control_logical_failures = zeros(Int,cnot_count)
    for result in worker_results
        control_logical_failures .+= result[3]
    end
    any_control_logical_failures =
        sum(result[4] for result in worker_results)
    target_logical_failures = sum(result[5] for result in worker_results)
    control_target_logical_failures =
        sum(result[6] for result in worker_results)
    cleanup_failures = sum(result[7] for result in worker_results)
    fail_rate = logical_failures / trials

    data = Dict{String, Any}(
        "CNOT_Ft" => 1 - fail_rate,
        "CNOT_fail_rate" => fail_rate,
        "failures" => logical_failures,
        "trials" => trials,
        "logical_failures" => logical_failures,
        "control_logical_failures" => control_logical_failures,
        "any_control_logical_failures" => any_control_logical_failures,
        "target_logical_failures" => target_logical_failures,
        "control_target_logical_failures" =>
            control_target_logical_failures,
        "cleanup_failures" => cleanup_failures,
    )
    for control_index in 1:cnot_count
        data["control_$(control_index)_logical_failures"] =
            control_logical_failures[control_index]
    end
    return data
end

function main()
    mode = get(ENV, "MODE", "CNOT_Ft")
    mode == "CNOT_Ft" ||
        error("cnot_primitive_consecutive.jl supports only MODE=CNOT_Ft")

    L = parse(Int, get(ENV, "LVAL", "13"))
    L > 1 || error("LVAL must be greater than 1")

    logZ = parse(Bool, get(ENV, "LOGZ", "true"))
    Z = logZ ? ceil(Int, log(1.5, L)) : ceil(Int, L / 4)
    Z > 1 || error("resolved buffer depth Z must be at least 2")

    p = parse(Float64, get(ENV, "PVAL", "0.011"))
    qrat = parse(Float64, get(ENV, "QRAT", "1"))
    0 <= p <= 1 || error("PVAL must be between 0 and 1")
    qrat >= 0 || error("QRAT must be nonnegative")
    p * qrat <= 1 || error("PVAL * QRAT must be at most 1")

    r = parse(Int, get(ENV, "RVAL", "3"))
    r > 0 || error("RVAL must be positive")
    synch = parse(Bool, get(ENV, "SYNCH", "true"))
    trial_parallel = parse(Bool, get(ENV, "TRIAL_PARALLEL", "true"))
    verbose = parse(Bool, get(ENV, "VERBOSE", "true"))

    T = parse(Int, get(ENV, "TVAL", string(L)))
    T > 0 || error("TVAL must be positive")
    default_interval_spec = join(fill(string(T),3),",")
    interval_spec = get(ENV,"CNOT_INTERVALS",default_interval_spec)
    intervals = parse_cnot_intervals(interval_spec)
    cnot_count = length(intervals) - 1
    noisy_rounds = sum(intervals)
    default_cleanup_time = 2T

    cleanup_time_env = lowercase(strip(get(ENV, "CLEANUP_TIME", "auto")))
    cleanup_time = cleanup_time_env == "auto" ?
        default_cleanup_time : parse(Int, cleanup_time_env)
    cleanup_time >= 0 || error("CLEANUP_TIME must be nonnegative")

    cnot_style = get(ENV, "CNOT_STYLE", "primitive")
    cnot_style == "primitive" ||
        error("only CNOT_STYLE=primitive is implemented in this driver")

    legacy_samps = parse(Int, get(ENV, "SAMPS", "0"))
    legacy_samps >= 0 || error("SAMPS must be nonnegative")
    default_stop_mode = legacy_samps > 0 ? "trials" : "failures"
    stop_mode = lowercase(strip(get(ENV, "STOP_MODE", default_stop_mode)))
    stop_mode in ("failures", "trials") ||
        error("STOP_MODE must be either failures or trials")
    legacy_samps > 0 && stop_mode != "trials" &&
        error("SAMPS>0 is a legacy alias for STOP_MODE=trials")

    target = if stop_mode == "failures"
        value = parse(Int, get(ENV, "ACC_ERRORS", "1000"))
        value > 0 || error("ACC_ERRORS must be positive in failure-stopping mode")
        value
    else
        default_trials = legacy_samps > 0 ? string(legacy_samps) : "100000"
        value = parse(Int, get(ENV, "MAX_TRIALS", default_trials))
        value > 0 || error("MAX_TRIALS must be positive in trial-stopping mode")
        legacy_samps > 0 && haskey(ENV, "MAX_TRIALS") &&
            value != legacy_samps &&
            error("SAMPS and MAX_TRIALS must agree when both are set")
        value
    end

    repeat_adj = haskey(ENV, "REPEAT_INDEX") ?
        "_rep$(ENV["REPEAT_INDEX"])" : ""
    out_adj = get(ENV, "OUT_ADJ", repeat_adj)

    params = Dict{String, Any}(
        "mode" => "CNOT_Ft",
        "L" => L,
        "Ls" => [L],
        "Z" => Z,
        "Zs" => [Z],
        "logZ" => logZ,
        "p" => p,
        "ps" => [p],
        "qrat" => qrat,
        "r" => r,
        "synch" => synch,
        "trial_parallel" => trial_parallel,
        "julia_threads" => nthreads(),
        "T" => T,
        "Ts" => [T],
        "CNOT_COUNT" => cnot_count,
        "CNOT_INTERVALS_SPEC" => interval_spec,
        "CNOT_INTERVALS" => intervals,
        "NOISY_ROUNDS" => noisy_rounds,
        "CLEANUP_TIME" => cleanup_time,
        "CNOT_STYLE" => cnot_style,
        "stop_mode" => stop_mode,
    )
    if stop_mode == "failures"
        params["acc_errors"] = target
    else
        params["max_trials"] = target
    end
    legacy_samps > 0 && (params["legacy_samps"] = legacy_samps)

    println("details of simulation:")
    println("mode = CNOT_Ft")
    println("CNOT style = $cnot_style")
    println("system size = $L")
    println("RG depth = $Z")
    println("p = $p, q = $(p * qrat)")
    println("synch = $synch")
    println("field update speed = $r")
    println("T = $T (default interval and cleanup reference)")
    println("CNOT count = $cnot_count")
    println("CNOT intervals = $intervals")
    println("noisy rounds = $noisy_rounds, CLEANUP_TIME = $cleanup_time")
    println("stopping mode = $stop_mode, target = $target")
    println(
        "trial parallelism = $(trial_parallel && nthreads() > 1) " *
        "($(nthreads()) Julia threads)",
    )

    data = estimate_consecutive_cnot_Ft(
        L,
        Z,
        p,
        p * qrat,
        r,
        synch,
        false,
        intervals,
        cleanup_time,
        stop_mode,
        target,
        trial_parallel,
        verbose,
    )

    sadj = synch ? "" : "_asynch"
    qadj = qrat == 0 ? "" : "_qrat$qrat"
    padj = "_p$(round(p, sigdigits=3))to$(round(p, sigdigits=3))"
    logzadj = logZ ? "_logZ" : ""
    stop_adj = stop_mode == "failures" ? "_fail$(target)" : "_trials$(target)"
    sequence_adj = "_seq$(cnot_count)_I$(join(intervals,"-"))"
    fout = "2d_CNOT_$(cnot_style)_Ft$(qadj)$(padj)_L$(L)_Z$(Z)" *
        "$(sadj)$(logzadj)$(stop_adj)$(sequence_adj)$(out_adj).txt"

    println("writing to file: $fout")
    open(fout, "w") do io
        println(io, "### data ###")
        for key in keys(data)
            println(io, "$key = $(repr(data[key]))")
        end
        println(io)
        println(io, "### params ###")
        for key in keys(params)
            println(io, "$key = $(repr(params[key]))")
        end
    end

    if parse(Bool, get(ENV, "ENABLE_ALERT", "false"))
        try
            alert("finished | L = $L; p = $p")
        catch err
            @warn "completion notification failed" exception=(err, catch_backtrace())
        end
    end
    println("finished at time $(Dates.now())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
