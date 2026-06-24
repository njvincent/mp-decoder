"""
imported from common functions
"""

using Random 
using Statistics
using LinearAlgebra 
using JLD2 
using Alert
using ProgressMeter
using Dates
using Base.Threads

# Primitive first-version CNOT driver.
#
# This file intentionally reuses the one-sector memory decoder below and adds
# a minimal two-block control/target experiment. It is not a physically
# complete labeled-defect CNOT decoder; it only applies the X-sector rule
# c_out = c, t_out = c xor t to the tracked decoder arrays.
# 
# Modifications:
# 1. Both the control and the target first update T/2 rounds before the CNOT 
#    gate, and they update another T/2 rounds before 2T rounds of cleanup.
# 2. The update rule for CNOT is that 
#    a. state, state_correction, old_synds, new_synds, and hist of the control 
#       are unchanged, but the target are XORed with the control.
#    b. fields of the control is unchanged. fields of the target takes the 
#       non-zero-min of control and target.
#    c. new_fields for both the control and the target are set to zero.
# 3. A trial is counted as failure if either the control or the target has a 
#    logical failure after the cleanup.


function circle_distance(i,j,L)
    """
    computes the distance between two sites on a circle of length L
    """
    return min(abs(i-j),L-abs(i-j))
end

function nonzeromin(a, b)
    if a == 0
        return b
    elseif b == 0
        return a
    else
        return min(a, b)
    end
end

function init_2d(p,L,init_type,bconds)
    """
    initializes 2d system
    """
    ind(i) = mod1(i,L)
    state = falses(L,L,2)
    if init_type == "rand"
        if bconds == "periodic"
            for i in 1:L, j in 1:L, o in 1:2
                if rand() < p 
                    state[i,j,o] ⊻= true 
                end 
            end
        else # finite square lattice
            for i in 2:L-1, j in 2:L-1
                if i == L-1 && j < L-1
                    if rand() < p 
                        state[i,j,2] ⊻= true 
                    end 
                end 
                if j == L-1 && i < L-1
                    if rand() < p 
                        state[i,j,1] ⊻= true 
                    end 
                end
                if i < L-1 && j < L-1 
                    for o in 1:2 
                        if rand() < p 
                            state[i,j,o] ⊻= true 
                        end 
                    end
                end 
            end
        end
    else 
        println("init type $init_type not recognized")
    end 

    return state, get_synds(state)
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

function get_decoding_time(state,old_synds,new_synds,hist,r,synch)
    """
    runtime of offline decoding on initial history hist
    """
    t = 0
    L,_,Z = size(hist)
    hist_copy = copy(hist); fields = zeros(Int,L,L,Z,3,2); new_fields = copy(fields); state_copy = copy(state); state_correction = falses(L,L,2); hist_correction = falses(L,L,Z,3); old_synds_copy = copy(old_synds); new_synds_copy = copy(new_synds)

    max_erode_time = L^2 # only try for this long to decode the syndromes
    while any(hist_copy) && t ≤ max_erode_time # while there are any syndromes left to decode
        t += 1
        update!(state_copy,state_correction,old_synds_copy,new_synds_copy,hist_copy,hist_correction,fields,new_fields,r,0,0,synch,false)
    end
    if t == max_erode_time println("max time reached") end 
    return t
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

function update_two_blocks!(
    state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,hist_correction_c,fields_c,new_fields_c,
    state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,hist_correction_t,fields_t,new_fields_t,
    r,p,q,synch,pretty)
    """
    Apply the ordinary single-block decoder update independently to control and
    target memory blocks.
    """
    update!(state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,hist_correction_c,fields_c,new_fields_c,r,p,q,synch,pretty)
    update!(state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,hist_correction_t,fields_t,new_fields_t,r,p,q,synch,pretty)
    return nothing
end

function split_cnot_timing(T)
    """
    Split the primitive CNOT protocol's total noisy time T into
    T/2 rounds before the CNOT, T/2 rounds after the CNOT, and 2T cleanup
    rounds. For odd T, the extra noisy round is placed after the CNOT.
    """
    if T < 1
        error("CNOT total time T must be positive.")
    end
    T_PRE = fld(T,2)
    T_POST = T - T_PRE
    CLEANUP_TIME = 2T
    return T_PRE,T_POST,CLEANUP_TIME
end

function estimate_primitive_cnot_Ft(L,Z,p,q,r,synch,pretty,T_PRE,T_POST,CLEANUP_TIME,acc_err,fixed_samps,trial_parallel,verbose)
    """
    Monte Carlo estimate of a primitive CNOT fixed-time success probability.

    If fixed_samps > 0, run exactly that many samples. Otherwise sample until
    acc_err failed trials have been accumulated, following the original Ft
    mode. For p = q = 0, a single fixed sample is used to avoid an infinite
    accumulate-until-failure loop.
    """
    use_fixed_samps = fixed_samps > 0
    if !use_fixed_samps && p == 0 && q == 0
        use_fixed_samps = true
        fixed_samps = 1
    end
    if use_fixed_samps && fixed_samps < 1
        error("fixed_samps must be positive when fixed-sample CNOT sampling is requested.")
    elseif !use_fixed_samps && acc_err < 1
        error("ACC_ERRORS must be positive when CNOT_Ft is accumulating failures.")
    end

    work_units = use_fixed_samps ? fixed_samps : acc_err
    worker_count = trial_parallel ? min(nthreads(),max(work_units,1)) : 1
    worker_results = Vector{Tuple{Int,Int,Int,Int,Int,Int}}(undef, worker_count)

    function run_cnot_trials(local_samps,target_errors)
        local_hist_c = falses(L,L,Z); local_hist_correction_c = falses(L,L,Z,3)
        local_state_c = falses(L,L,2); local_state_correction_c = falses(L,L,2)
        local_fields_c = zeros(Int,L,L,Z,3,2); local_new_fields_c = zeros(Int,L,L,Z,3,2)
        local_old_synds_c = falses(L,L); local_new_synds_c = falses(L,L)

        local_hist_t = falses(L,L,Z); local_hist_correction_t = falses(L,L,Z,3)
        local_state_t = falses(L,L,2); local_state_correction_t = falses(L,L,2)
        local_fields_t = zeros(Int,L,L,Z,3,2); local_new_fields_t = zeros(Int,L,L,Z,3,2)
        local_old_synds_t = falses(L,L); local_new_synds_t = falses(L,L)

        decoded_state_c = falses(L,L,2)
        decoded_state_t = falses(L,L,2)

        local_failures = 0
        local_trials = 0
        local_control_logical_failures = 0
        local_target_logical_failures = 0
        local_both_logical_failures = 0
        local_cleanup_failures = 0

        while (use_fixed_samps ? (local_trials < local_samps) : (local_failures < target_errors))
            if verbose && local_trials % 10000 == 0
                println("thread $(threadid()) CNOT trial: ", local_trials)
            end

            local_hist_c .= false; local_hist_correction_c .= false
            local_state_c .= false; local_state_correction_c .= false
            local_fields_c .= 0; local_new_fields_c .= 0
            local_old_synds_c .= false; local_new_synds_c .= false

            local_hist_t .= false; local_hist_correction_t .= false
            local_state_t .= false; local_state_correction_t .= false
            local_fields_t .= 0; local_new_fields_t .= 0
            local_old_synds_t .= false; local_new_synds_t .= false

            for _ in 1:T_PRE
                update_two_blocks!(
                    local_state_c,local_state_correction_c,local_old_synds_c,local_new_synds_c,local_hist_c,local_hist_correction_c,local_fields_c,local_new_fields_c,
                    local_state_t,local_state_correction_t,local_old_synds_t,local_new_synds_t,local_hist_t,local_hist_correction_t,local_fields_t,local_new_fields_t,
                    r,p,q,synch,pretty)
            end

            primitive_cnot_x_sector!(
                local_state_c,local_state_correction_c,local_old_synds_c,local_new_synds_c,local_hist_c,local_fields_c,local_new_fields_c,
                local_state_t,local_state_correction_t,local_old_synds_t,local_new_synds_t,local_hist_t,local_fields_t,local_new_fields_t)
            local_hist_correction_c .= false
            local_hist_correction_t .= false

            for _ in 1:T_POST
                update_two_blocks!(
                    local_state_c,local_state_correction_c,local_old_synds_c,local_new_synds_c,local_hist_c,local_hist_correction_c,local_fields_c,local_new_fields_c,
                    local_state_t,local_state_correction_t,local_old_synds_t,local_new_synds_t,local_hist_t,local_hist_correction_t,local_fields_t,local_new_fields_t,
                    r,p,q,synch,pretty)
            end

            for _ in 1:CLEANUP_TIME
                update_two_blocks!(
                    local_state_c,local_state_correction_c,local_old_synds_c,local_new_synds_c,local_hist_c,local_hist_correction_c,local_fields_c,local_new_fields_c,
                    local_state_t,local_state_correction_t,local_old_synds_t,local_new_synds_t,local_hist_t,local_hist_correction_t,local_fields_t,local_new_fields_t,
                    r,0,0,true,pretty)
                if !any(local_hist_c) && !any(local_hist_t)
                    break
                end
            end

            decoded_state_c .= local_state_c .⊻ local_state_correction_c
            decoded_state_t .= local_state_t .⊻ local_state_correction_t
            cleanup_success = !any(local_hist_c) && !any(local_hist_t)
            if cleanup_success
                @assert !any(get_synds(decoded_state_c)) "control decoded state is not syndrome-free!"
                @assert !any(get_synds(decoded_state_t)) "target decoded state is not syndrome-free!"
            elseif verbose
                println("thread $(threadid()) CNOT cleanup did not remove all defects")
            end

            control_logical_failure = !detect_logical_error(decoded_state_c)
            target_logical_failure = !detect_logical_error(decoded_state_t)
            cleanup_failure = !cleanup_success
            logical_failure = control_logical_failure || target_logical_failure

            local_failures += logical_failure ? 1 : 0
            local_control_logical_failures += control_logical_failure ? 1 : 0
            local_target_logical_failures += target_logical_failure ? 1 : 0
            local_both_logical_failures += (control_logical_failure && target_logical_failure) ? 1 : 0
            local_cleanup_failures += cleanup_failure ? 1 : 0
            local_trials += 1

            if verbose && !use_fixed_samps && logical_failure
                println("thread $(threadid()) CNOT progress: $(local_failures / target_errors)")
            end
        end

        return (
            local_failures,
            local_trials,
            local_control_logical_failures,
            local_target_logical_failures,
            local_both_logical_failures,
            local_cleanup_failures,
        )
    end

    @threads for worker in 1:worker_count
        if use_fixed_samps
            local_samps = fixed_samps ÷ worker_count + (worker <= fixed_samps % worker_count ? 1 : 0)
            worker_results[worker] = run_cnot_trials(local_samps,0)
        else
            target_errors = acc_err ÷ worker_count + (worker <= acc_err % worker_count ? 1 : 0)
            worker_results[worker] = run_cnot_trials(0,target_errors)
        end
    end

    logical_failures = sum(result[1] for result in worker_results)
    trials = sum(result[2] for result in worker_results)
    control_logical_failures = sum(result[3] for result in worker_results)
    target_logical_failures = sum(result[4] for result in worker_results)
    both_logical_failures = sum(result[5] for result in worker_results)
    cleanup_failures = sum(result[6] for result in worker_results)
    fail_rate = logical_failures / trials

    return Dict{String, Any}(
        "CNOT_Ft" => 1 - fail_rate,
        "CNOT_fail_rate" => fail_rate,
        "trials" => trials,
        "logical_failures" => logical_failures,
        "control_logical_failures" => control_logical_failures,
        "target_logical_failures" => target_logical_failures,
        "both_logical_failures" => both_logical_failures,
        "cleanup_failures" => cleanup_failures,
    )
end

function js_string(s)
    out = replace(string(s), "\\" => "\\\\")
    out = replace(out, "\"" => "\\\"")
    out = replace(out, "\n" => "\\n")
    return "\"$out\""
end

function bool_entries_js(a)
    entries = String[]
    for I in CartesianIndices(a)
        if a[I]
            push!(entries, "[" * join(Tuple(I), ",") * "]")
        end
    end
    return "[" * join(entries, ",") * "]"
end

function hist_entries_js(hist)
    L,_,Z = size(hist)
    entries = String[]
    for i in 1:L, j in 1:L
        count = 0
        for k in 1:Z
            count += hist[i,j,k] ? 1 : 0
        end
        if count > 0
            push!(entries, "[$i,$j,$count]")
        end
    end
    return "[" * join(entries, ",") * "]"
end

function field_entries_js(fields)
    L,_,Z,_,_ = size(fields)
    entries = String[]
    for i in 1:L, j in 1:L
        min_field = typemax(Int)
        for k in 1:Z, a in 1:3, s in 1:2
            val = fields[i,j,k,a,s]
            if val > 0
                min_field = min(min_field,val)
            end
        end
        if min_field < typemax(Int)
            push!(entries, "[$i,$j,$min_field]")
        end
    end
    return "[" * join(entries, ",") * "]"
end

function cnot_demo_block_js(state,state_correction,hist,fields)
    decoded_state = state .⊻ state_correction
    synds = get_synds(decoded_state)
    logical_status = (any(synds) || any(hist)) ? "pending" : (detect_logical_error(decoded_state) ? "OK" : "FAIL")
    return "{" *
        "\"physical\":$(bool_entries_js(state))," *
        "\"correction\":$(bool_entries_js(state_correction))," *
        "\"decoded\":$(bool_entries_js(decoded_state))," *
        "\"syndromes\":$(bool_entries_js(synds))," *
        "\"logical_status\":$(js_string(logical_status))," *
        "\"hist\":$(hist_entries_js(hist))," *
        "\"fields\":$(field_entries_js(fields))" *
        "}"
end

function write_cnot_demo_html(frames_js,L,Z,p,q,r,synch,T_PRE,T_POST,CLEANUP_TIME,fout)
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Primitive CNOT X-sector demo</title>
<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f6f8fa; }
main { max-width: 1180px; margin: 0 auto; padding: 20px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.note { color: #5b6775; margin: 0 0 14px; line-height: 1.4; }
.controls { display: flex; gap: 10px; align-items: center; margin: 14px 0; flex-wrap: wrap; }
button { padding: 6px 10px; border: 1px solid #b7c0cc; border-radius: 6px; background: white; cursor: pointer; }
input[type="range"] { flex: 1 1 340px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel { background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; }
.panel h2 { font-size: 16px; margin: 0 0 8px; }
canvas { width: 100%; height: auto; border: 1px solid #d8dee4; border-radius: 6px; background: #fbfcfe; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; font-size: 13px; color: #52606d; }
.swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.meta { font-size: 13px; color: #52606d; margin-top: 8px; }
@media (max-width: 860px) { .panels { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
<h1>Primitive CNOT X-sector Demo</h1>
<p class="note">Bookkeeping view of the primitive baseline. At the CNOT frame, the control block is unchanged and the target block receives target xor control; target fields are merged by nonzero-min.</p>
<div id="frameLabel" class="meta"></div>
<div class="controls">
<button id="prev">Prev</button>
<button id="play">Play</button>
<button id="next">Next</button>
<input id="slider" type="range" min="0" max="0" value="0">
</div>
<div class="panels">
<section class="panel"><h2>Control</h2><canvas id="control" width="560" height="560"></canvas><div id="controlMeta" class="meta"></div></section>
<section class="panel"><h2>Target</h2><canvas id="target" width="560" height="560"></canvas><div id="targetMeta" class="meta"></div></section>
</div>
<div class="legend">
<span><span class="swatch" style="background:#e67e22"></span>physical X</span>
<span><span class="swatch" style="background:#2d7ff9"></span>state correction</span>
<span><span class="swatch" style="background:#111827"></span>decoded residual</span>
<span><span class="swatch" style="background:#d62828; border-radius:50%"></span>decoded syndrome</span>
<span><span class="swatch" style="background:#7b2cbf"></span>history count</span>
<span><span class="swatch" style="background:#2a9d8f"></span>field site</span>
</div>
<p class="meta">L=$L, Z=$Z, p=$p, q=$q, r=$r, synch=$synch, T=$(T_PRE + T_POST), T_PRE=$T_PRE, T_POST=$T_POST, CLEANUP_TIME=$CLEANUP_TIME</p>
</main>
<script>
const L = $L;
const frames = $frames_js;
const WIDTH = 560;
const MARGIN = 48;
const SCALE = (WIDTH - 2 * MARGIN) / Math.max(L - 1, 1);
const slider = document.getElementById("slider");
const label = document.getElementById("frameLabel");
const playButton = document.getElementById("play");
let index = 0;
let timer = null;
slider.max = Math.max(frames.length - 1, 0);

function pt(i, j) {
  return [MARGIN + (i - 1) * SCALE, MARGIN + (j - 1) * SCALE];
}

function edgeEndpoints(edge) {
  const i = edge[0], j = edge[1], o = edge[2];
  const a = pt(i, j);
  if (o === 1) return [a[0], a[1], MARGIN + (i % L) * SCALE, a[1]];
  return [a[0], a[1], a[0], MARGIN + (j % L) * SCALE];
}

function drawEdges(ctx, edges, color, width, offset) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  for (const edge of edges) {
    let [x1, y1, x2, y2] = edgeEndpoints(edge);
    if (edge[2] === 1) { y1 += offset; y2 += offset; }
    else { x1 += offset; x2 += offset; }
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawBlock(canvasId, metaId, block) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#d9e2ec";
  ctx.lineWidth = 1;
  for (let i = 1; i <= L; i++) {
    let [x1, y1] = pt(i, 1), [x2, y2] = pt(i, L);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (let j = 1; j <= L; j++) {
    let [x1, y1] = pt(1, j), [x2, y2] = pt(L, j);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }

  for (const f of block.fields) {
    const [x, y] = pt(f[0], f[1]);
    const alpha = Math.max(0.12, Math.min(0.45, 0.55 / Math.max(f[2], 1)));
    ctx.fillStyle = "rgba(42, 157, 143, " + alpha + ")";
    ctx.fillRect(x - 13, y - 13, 26, 26);
  }

  drawEdges(ctx, block.physical, "#e67e22", 7, -5);
  drawEdges(ctx, block.correction, "#2d7ff9", 5, 5);
  drawEdges(ctx, block.decoded, "#111827", 3, 0);

  for (const h of block.hist) {
    const [x, y] = pt(h[0], h[1]);
    ctx.fillStyle = "rgba(123, 44, 191, 0.72)";
    ctx.fillRect(x - 9, y - 9, 18, 18);
    ctx.fillStyle = "white";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(h[2]), x, y);
  }
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";

  for (const s of block.syndromes) {
    const [x, y] = pt(s[0], s[1]);
    ctx.fillStyle = "#d62828";
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.fillStyle = "#627d98";
  for (let i = 1; i <= L; i++) {
    for (let j = 1; j <= L; j++) {
      const [x, y] = pt(i, j);
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  document.getElementById(metaId).textContent =
    "physical: " + block.physical.length +
    " | correction: " + block.correction.length +
    " | residual: " + block.decoded.length +
    " | syndromes: " + block.syndromes.length +
    " | logical: " + block.logical_status +
    " | history sites: " + block.hist.length +
    " | field sites: " + block.fields.length;
}

function render() {
  const f = frames[index];
  slider.value = index;
  label.textContent = "frame " + (index + 1) + " / " + frames.length + ": " + f.label;
  drawBlock("control", "controlMeta", f.control);
  drawBlock("target", "targetMeta", f.target);
}

function stop() {
  if (timer !== null) clearInterval(timer);
  timer = null;
  playButton.textContent = "Play";
}

document.getElementById("prev").onclick = () => { stop(); index = Math.max(0, index - 1); render(); };
document.getElementById("next").onclick = () => { stop(); index = Math.min(frames.length - 1, index + 1); render(); };
slider.oninput = () => { stop(); index = Number(slider.value); render(); };
playButton.onclick = () => {
  if (timer !== null) { stop(); return; }
  playButton.textContent = "Pause";
  timer = setInterval(() => {
    index = (index + 1) % frames.length;
    render();
  }, 650);
};
render();
</script>
</body>
</html>
"""
    open(fout, "w") do io
        print(io, html)
    end
    return fout
end

function run_cnot_demo(L,Z,p,q,r,synch,T_PRE,T_POST,CLEANUP_TIME,out_adj)
    if L < 4
        error("CNOT_DEMO expects L >= 4 so the seeded pattern is visible.")
    end

    Random.seed!(parse(Int, get(ENV, "DEMO_SEED", "7")))
    pretty = false
    hist_frames_c = Vector{Array{Bool,3}}()
    hist_frames_t = Vector{Array{Bool,3}}()
    field_frames_c = Vector{Array{Int,5}}()
    field_frames_t = Vector{Array{Int,5}}()
    html_frames = String[]

    hist_c = falses(L,L,Z); hist_correction_c = falses(L,L,Z,3)
    state_c = falses(L,L,2); state_correction_c = falses(L,L,2)
    fields_c = zeros(Int,L,L,Z,3,2); new_fields_c = zeros(Int,L,L,Z,3,2)
    old_synds_c = falses(L,L); new_synds_c = falses(L,L)

    hist_t = falses(L,L,Z); hist_correction_t = falses(L,L,Z,3)
    state_t = falses(L,L,2); state_correction_t = falses(L,L,2)
    fields_t = zeros(Int,L,L,Z,3,2); new_fields_t = zeros(Int,L,L,Z,3,2)
    old_synds_t = falses(L,L); new_synds_t = falses(L,L)

    demo_seed_errors = parse(Int, get(ENV, "DEMO_SEED_ERRORS", string(L)))
    demo_style = get(ENV, "DEMO_STYLE", "complex")

    if demo_style == "simple"
        state_c[2,2,1] = true
        state_c[3,2,1] = true
        state_t[2,4,2] = true
    elseif demo_style == "complex"
        for _ in 1:demo_seed_errors
            state_c[rand(1:L),rand(1:L),rand(1:2)] ⊻= true
            state_t[rand(1:L),rand(1:L),rand(1:2)] ⊻= true
        end
    else
        error("DEMO_STYLE must be \"simple\" or \"complex\".")
    end

    function add_frame!(label)
        push!(hist_frames_c, copy(hist_c))
        push!(hist_frames_t, copy(hist_t))
        push!(field_frames_c, copy(fields_c))
        push!(field_frames_t, copy(fields_t))
        push!(html_frames, "{" *
            "\"label\":$(js_string(label))," *
            "\"control\":$(cnot_demo_block_js(state_c,state_correction_c,hist_c,fields_c))," *
            "\"target\":$(cnot_demo_block_js(state_t,state_correction_t,hist_t,fields_t))" *
            "}")
    end

    add_frame!("seeded initial state")

    for t in 1:T_PRE
        update_two_blocks!(
            state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,hist_correction_c,fields_c,new_fields_c,
            state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,hist_correction_t,fields_t,new_fields_t,
            r,p,q,synch,pretty)
        add_frame!("pre round $t")
    end

    add_frame!("immediately before primitive CNOT")
    primitive_cnot_x_sector!(
        state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,fields_c,new_fields_c,
        state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,fields_t,new_fields_t)
    hist_correction_c .= false
    hist_correction_t .= false
    add_frame!("after primitive CNOT: target = target xor control")

    for t in 1:T_POST
        update_two_blocks!(
            state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,hist_correction_c,fields_c,new_fields_c,
            state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,hist_correction_t,fields_t,new_fields_t,
            r,p,q,synch,pretty)
        add_frame!("post round $t")
    end

    for t in 1:CLEANUP_TIME
        update_two_blocks!(
            state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,hist_correction_c,fields_c,new_fields_c,
            state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,hist_correction_t,fields_t,new_fields_t,
            r,0,0,true,pretty)
        add_frame!("ideal cleanup round $t")
        if !any(hist_c) && !any(hist_t)
            break
        end
    end

    qadj = q == 0 ? "" : "_q$(round(q,sigdigits=3))"
    padj = "_p$(round(p,sigdigits=3))"
    sadj = ~synch ? "_asynch" : ""
    style_adj = demo_style == "simple" ? "_simple" : "_complex"
    base = "2d_CNOT_primitive_demo$style_adj$padj$qadj" * "_L$(L)_Z$(Z)$sadj$out_adj"

    function write_block_history(block_name,hist_frames,field_frames)
        T = length(hist_frames)
        # Store hist as a plain numeric array, not a Julia BitArray. JLD2 saves
        # BitArray with internal chunk metadata that h5py cannot read directly.
        hist_out = zeros(UInt8,T,L,L,Z)
        field_hist_out = zeros(Int,T,L,L,Z,3,2)
        for t in 1:T
            hist_out[t,:,:,:] .= UInt8.(hist_frames[t])
            field_hist_out[t,:,:,:,:,:] .= field_frames[t]
        end

        fout = "$(base)_$(block_name).jld2"
        jldsave(
            fout;
            hist=hist_out,
            field_hist=field_hist_out,
            L=L,
            Z=Z
        )
        return fout
    end

    control_file = write_block_history("control",hist_frames_c,field_frames_c)
    target_file = write_block_history("target",hist_frames_t,field_frames_t)
    html_file = "$(base).html"
    write_cnot_demo_html("[" * join(html_frames, ",") * "]",L,Z,p,q,r,synch,T_PRE,T_POST,CLEANUP_TIME,html_file)
    return html_file, control_file, target_file
end

function run_cnot_sanity_checks(L,Z,r,T_PRE,T_POST,CLEANUP_TIME)
    """
    Lightweight checks for the primitive CNOT bookkeeping.
    """
    @assert nonzeromin(0,7) == 7
    @assert nonzeromin(7,0) == 7
    @assert nonzeromin(4,9) == 4
    @assert nonzeromin(9,4) == 4

    fields_c = zeros(Int,3,3,2,3,2); fields_t = zeros(Int,3,3,2,3,2)
    fields_c[1,1,1,1,1] = 5; fields_t[1,1,1,1,1] = 0
    fields_c[1,1,1,1,2] = 0; fields_t[1,1,1,1,2] = 7
    fields_c[1,1,1,2,1] = 4; fields_t[1,1,1,2,1] = 9
    primitive_cnot_x_sector!(
        falses(3,3,2),falses(3,3,2),falses(3,3),falses(3,3),falses(3,3,2),fields_c,zeros(Int,3,3,2,3,2),
        falses(3,3,2),falses(3,3,2),falses(3,3),falses(3,3),falses(3,3,2),fields_t,zeros(Int,3,3,2,3,2))
    @assert fields_t[1,1,1,1,1] == 5
    @assert fields_t[1,1,1,1,2] == 7
    @assert fields_t[1,1,1,2,1] == 4

    testL = 3
    testZ = 2
    for cval in (false,true), tval in (false,true)
        state_c = falses(testL,testL,2); state_t = falses(testL,testL,2)
        state_correction_c = falses(testL,testL,2); state_correction_t = falses(testL,testL,2)
        old_synds_c = falses(testL,testL); old_synds_t = falses(testL,testL)
        new_synds_c = falses(testL,testL); new_synds_t = falses(testL,testL)
        hist_c = falses(testL,testL,testZ); hist_t = falses(testL,testL,testZ)
        fields_c = zeros(Int,testL,testL,testZ,3,2); fields_t = zeros(Int,testL,testL,testZ,3,2)
        new_fields_c = ones(Int,testL,testL,testZ,3,2); new_fields_t = ones(Int,testL,testL,testZ,3,2)

        state_c[1,1,1] = cval; state_t[1,1,1] = tval
        state_correction_c[1,1,1] = cval; state_correction_t[1,1,1] = tval
        old_synds_c[1,1] = cval; old_synds_t[1,1] = tval
        new_synds_c[1,1] = cval; new_synds_t[1,1] = tval
        hist_c[1,1,1] = cval; hist_t[1,1,1] = tval

        primitive_cnot_x_sector!(
            state_c,state_correction_c,old_synds_c,new_synds_c,hist_c,fields_c,new_fields_c,
            state_t,state_correction_t,old_synds_t,new_synds_t,hist_t,fields_t,new_fields_t)

        @assert state_c[1,1,1] == cval
        @assert state_t[1,1,1] == xor(cval,tval)
        @assert hist_c[1,1,1] == cval
        @assert hist_t[1,1,1] == xor(cval,tval)
        @assert state_correction_t[1,1,1] == xor(cval,tval)
        @assert old_synds_t[1,1] == xor(cval,tval)
        @assert new_synds_t[1,1] == xor(cval,tval)
        @assert !any(new_fields_c)
        @assert !any(new_fields_t)
    end

    zero_noise_data = estimate_primitive_cnot_Ft(3,2,0.0,0.0,r,true,false,max(T_PRE,1),max(T_POST,1),max(CLEANUP_TIME,2),1,5,false,false)
    @assert zero_noise_data["logical_failures"] == 0
    println("CNOT sanity checks passed")
    return true
end

function main()
    """
    supported modes: 
    * "hist":   get history of one evolution, manual input of evolution time T, error rate p & q, and initial state
    * "erode":  correct random anyon configuration with no further physical or measurement error, output logical error rate and correction time statistics
    * "quench": time evolution of anyon densities and decoding times starting from a random state. used for diagnosing state preparation.
    * "trel":   compute relaxation time/memory lifetime for online decoding 
    * "Ft":     get decoding fidelity (error rate after a fixed number of noisy decoding rounds) for online decoding     
    * "CNOT_Ft": primitive two-block CNOT fixed-time fidelity test
    * "CNOT_DEMO": write side-by-side HTML plus optional control/target histories
    * "CNOT_DEBUG": run small primitive CNOT sanity checks
    * "stats":  get anyon density in the long time steady state
    """

    mode = get(ENV, "MODE", "CNOT_Ft")

    L = parse(Int, get(ENV, "LVAL", mode == "CNOT_DEMO" ? "9" : "13"))
    logZ = parse(Bool, get(ENV, "LOGZ", "true"))
    Z = logZ ? ceil(Int, log(1.5, L)) : ceil(Int, L/4) # log scaling with L
    p = parse(Float64, get(ENV, "PVAL", mode == "CNOT_DEMO" ? "0.015" : "0.011"))
    qrat = parse(Float64, get(ENV, "QRAT", "1")) # ratio of measurement errors to physical errors
    vary_L = false # if true, vary system size; if false, use fixed system size and vary p
    vary_Z = false 

    r = parse(Int, get(ENV, "RVAL", "3"))             # number of field updates per spin update # poor
    synch = parse(Bool, get(ENV, "SYNCH", "true"))    # synchronous or asynchronous update
    pretty = mode == "hist"                           # makes slightly prettier animations
    verbose = true                                    # controls some printouts
    trial_parallel = parse(Bool, get(ENV, "TRIAL_PARALLEL", "true"))
    repeat_adj = haskey(ENV, "REPEAT_INDEX") ? "_rep$(ENV["REPEAT_INDEX"])" : ""
    out_adj = get(ENV, "OUT_ADJ", repeat_adj)
    T_TOTAL = parse(Int, get(ENV, "TVAL", mode == "CNOT_DEMO" ? "12" : string(L)))
    T_PRE,T_POST,default_cleanup_time = split_cnot_timing(T_TOTAL)
    cleanup_time_env = get(ENV, "CLEANUP_TIME", "auto")
    CLEANUP_TIME = cleanup_time_env == "auto" ? default_cleanup_time : parse(Int, cleanup_time_env)
    CNOT_STYLE = get(ENV, "CNOT_STYLE", "primitive")
    cnot_acc_errors_env = haskey(ENV, "ACC_ERRORS") ? parse(Int, ENV["ACC_ERRORS"]) : nothing
    cnot_samps_env = haskey(ENV, "SAMPS") ? parse(Int, ENV["SAMPS"]) : 0

    params = parameter_repository(mode,L,Z,p,qrat,r,synch,vary_L,vary_Z,logZ)
    if mode == "CNOT_Ft" || mode == "CNOT_DEMO" || mode == "CNOT_DEBUG"
        if cnot_acc_errors_env !== nothing
            params["accu_errors"] = cnot_acc_errors_env
            params["accu_errors_vec"] = [cnot_acc_errors_env]
        end
        if cnot_samps_env > 0
            params["samps"] = [cnot_samps_env]
        end
        params["T"] = T_TOTAL
        params["Ts"] = [T_TOTAL]
        params["T_PRE"] = T_PRE
        params["T_POST"] = T_POST
        params["CLEANUP_TIME"] = CLEANUP_TIME
        params["CNOT_STYLE"] = CNOT_STYLE
    end
    Ts = params["Ts"]; samps = params["samps"]; # Ts: total simulation time; samps: number of samples per simulation
    ps = params["ps"]; nps = params["nps"];     # nps: the number of error probabilities being tested; ps: list of error probabilities
    Ls = params["Ls"]; Zs = params["Zs"];       # system size L and Z to be tested
    accu_errors = params["accu_errors"]; accu_errors_vec = params["accu_errors_vec"] # stop simulation if logical error count reaches accu_errors

    data_keys = ["Ft" "CNOT_Ft" "CNOT_fail_rate" "trials" "logical_failures" "control_logical_failures" "target_logical_failures" "both_logical_failures" "cleanup_failures" "demo_files" "demo_visualizer_commands"] ∪ ["hist" "field_hist" "state_hist"] ∪ ["trels" "trel_stats"] ∪ ["erode_times" "erode_stats"] ∪ ["Ms" "binds" "chis" "anyon_densities"] ∪ ["tpreps" "quenched_anyon_densities" "dectest_times" "tprep_errors"]
    data = Dict{String, Any}(key => 0 for key in data_keys)  # dictionary whose keys are the strings in data_keys and whose initial values are all 0

    println("details of simulation: ")
    println("synch = $synch")
    if vary_L 
        println("p = $p")
        println("Ls = $Ls")
    else 
        println("system size: $L")
        println("RG depth: $Z")
        println("ps = $ps")
    end 
    println("mode = $mode")
    if haskey(ENV, "REPEAT_INDEX")
        println("repeat index = $(ENV["REPEAT_INDEX"])")
    end
    println("trial parallelism = $(trial_parallel && nthreads() > 1) ($(nthreads()) Julia threads)")
    println("field update speed = $r")
    if mode == "CNOT_Ft" || mode == "CNOT_DEMO" || mode == "CNOT_DEBUG"
        println("CNOT style = $CNOT_STYLE")
        println("T = $T_TOTAL")
        println("T_PRE = $T_PRE, T_POST = $T_POST, CLEANUP_TIME = $CLEANUP_TIME")
        if mode == "CNOT_DEMO"
            println("demo seed = $(get(ENV, "DEMO_SEED", "7"))")
            println("demo style = $(get(ENV, "DEMO_STYLE", "complex"))")
            println("demo seeded random errors per block = $(get(ENV, "DEMO_SEED_ERRORS", string(L)))")
        elseif cnot_samps_env > 0
            println("fixed CNOT samples = $cnot_samps_env")
        else
            println("CNOT failures to accumulate = $(accu_errors_vec[1])")
        end
    end
    println("")

    state = falses(L,L,2); state_correction = falses(L,L,2); fields = zeros(Int,L,L,Z,3,2); new_fields = zeros(Int,L,L,Z,3,2); hist = falses(L,L,Z); hist_correction = falses(L,L,Z,3)
    old_synds = falses(L,L); new_synds = falses(L,L) 

    ### primitive CNOT fixed-time fidelity test ###
    if mode == "CNOT_DEMO"
        if CNOT_STYLE != "primitive"
            error("Only CNOT_STYLE=\"primitive\" is implemented in this first-version CNOT simulator.")
        end
        println("writing primitive CNOT demo visualization...")
        html_demo_file, control_demo_file, target_demo_file = run_cnot_demo(L,Z,p,p*qrat,r,synch,T_PRE,T_POST,CLEANUP_TIME,out_adj)
        data["demo_files"] = [html_demo_file,control_demo_file,target_demo_file]
        data["demo_visualizer_commands"] = [
            "python3 2d_windowed_history_visualizer.py -fin $control_demo_file",
            "python3 2d_windowed_history_visualizer.py -fin $target_demo_file",
        ]
        println("open HTML demo: $html_demo_file")
        println("demo history written to: $control_demo_file")
        println("demo history written to: $target_demo_file")
        println("visualize control with: $(data["demo_visualizer_commands"][1])")
        println("visualize target with: $(data["demo_visualizer_commands"][2])")

    elseif mode == "CNOT_Ft"
        if CNOT_STYLE != "primitive"
            error("Only CNOT_STYLE=\"primitive\" is implemented in this first-version CNOT simulator.")
        end
        println("running primitive CNOT fixed-time fidelity test...")
        cnot_data = estimate_primitive_cnot_Ft(L,Z,p,p*qrat,r,synch,pretty,T_PRE,T_POST,CLEANUP_TIME,accu_errors_vec[1],cnot_samps_env,trial_parallel,verbose)
        for key in keys(cnot_data)
            data[key] = cnot_data[key]
        end

    elseif mode == "CNOT_DEBUG"
        data["sanity_checks_passed"] = run_cnot_sanity_checks(L,Z,r,T_PRE,T_POST,CLEANUP_TIME)

    ### write history of evolution ### 
    elseif mode == "hist" 
        T = round(Int,2L) 
        println("running for time T = $T")
        data["hist"] = zeros(Bool,T,L,L,Z) # history of syndrome-changing events 
        data["state_hist"] = zeros(Bool,T,L,L,2) # history of state ⊻ state_correction
        data["field_hist"] = zeros(Int,T,L,L,Z,3,2) # history of fields
        
        examples = 1 # pooexamples 
        for ex in 1:examples 
            println(ex)
            # reset 
            hist .= false; hist_correction .= false; fields .= 0; new_fields .= 0; state .= false; state_correction .= false; old_synds .= false; new_synds .= false
            state[1:4,2,1] .= true # smol 
            state[3,2,2] = true # smol 

            # evolve for T time steps
            for t in 1:T
                data["hist"][t,:,:,:] .= hist
                data["state_hist"][t,:,:,:] .= state .⊻ state_correction
                data["field_hist"][t,:,:,:,:,:] .= fields
                emult = t > L ? 0 : 1 # turn off the errors at long enough time to allow for ideal decoding 
                update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,.01*emult,0.01*emult,synch,pretty)
                if emult == 0 && ~any(hist) # if we are anyon-free and no errors will occur at later times, remaining updates are only with fields 
                    break 
                end
            end 

            if verbose 
                ### for debugging: ### 
                if any(hist)
                    println("not all anyons cleaned up!!")
                    break 
                # end 
                else
                    corrected_state = state .⊻ state_correction 
                    corrected_synds = get_synds(corrected_state)
                    if any(corrected_synds)
                        println("incorrect feedback applied!")
                        println("after correction: ",corrected_synds)
                        break 
                    end 
                end 
            end 
        end 

    ### offline decoding ### 
    elseif mode == "erode" 
        println("doing offline decoding at L = $L...")
        # nps is the number of error probabilities being tested
        data["erode_frac"] = zeros(nps)    # logical error probablity
        data["erode_times"] = zeros(nps)   # mean of erosion times
        data["erode_stats"] = zeros(nps,3) # mean, std, max of erosion times
        # for each value of error probability p
        for (pind,thisp) in enumerate(ps) 
            thisL = Ls[pind]
            thisZ = Zs[pind]

            max_erode_time = thisL^2 
            println("L = $thisL, p = $thisp")

            function run_erode_trials(target_errors)
                fields = zeros(Int,thisL,thisL,thisZ,3,2); new_fields = zeros(Int,thisL,thisL,thisZ,3,2)
                state_correction = falses(thisL,thisL,2); hist = falses(thisL,thisL,thisZ); hist_correction = falses(thisL,thisL,thisZ,3); new_synds = falses(thisL,thisL)
                local_failures = 0
                local_trials = 0
                local_tsum = 0.0
                local_t2sum = 0.0
                local_max_t = 0
                while local_failures < target_errors # sample until this worker sees its assigned failures
                    state, old_synds = init_2d(thisp,thisL,"rand","periodic") # random initial state with noise of strength p
                    fields .= 0; hist .= false; new_synds .= false; hist_correction .= false; state_correction .= false
                    t = 0
                    while t < max_erode_time
                        update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,0,0,synch,pretty)
                        t += 1
                        if ~any(hist) # stop once the syndrome-history array contains no defects
                            break
                        end
                    end
                    if t == max_erode_time
                        println("max erosion time reached! $t")
                    end
                    local_trials += 1
                    local_max_t = max(local_max_t, t)
                    local_tsum += t
                    local_t2sum += t^2

                    corrected_state = state .⊻ state_correction
                    if t < max_erode_time @assert sum(get_synds(corrected_state)) == 0 "syndromes not cleaned up: $(get_synds(corrected_state))" end  # check that syndromes are cleaned up
                    logical_error = 1-detect_logical_error(corrected_state) # 0 only if both cycles have trivial winding
                    local_failures += logical_error
                end
                return local_failures, local_trials, local_tsum, local_t2sum, local_max_t
            end

            worker_count = trial_parallel ? min(nthreads(), accu_errors) : 1
            worker_results = Vector{Tuple{Int,Int,Float64,Float64,Int}}(undef, worker_count)
            @threads for worker in 1:worker_count
                target_errors = accu_errors ÷ worker_count + (worker <= accu_errors % worker_count ? 1 : 0)
                worker_results[worker] = run_erode_trials(target_errors)
            end

            logical_failures = sum(result[1] for result in worker_results)
            trials = sum(result[2] for result in worker_results)
            longest_erosion = maximum(result[5] for result in worker_results)
            data["erode_times"][pind] = sum(result[3] for result in worker_results)
            data["erode_stats"][pind,2] = sum(result[4] for result in worker_results)
            data["erode_frac"][pind] = logical_failures / trials # average over trials
            data["erode_times"][pind] /= trials # average over trials
            data["erode_stats"][pind,1] = data["erode_times"][pind] # redundant of course
            data["erode_stats"][pind,2] = sqrt(data["erode_stats"][pind,2]/trials - data["erode_stats"][pind,1]^2 + 1e-10)
            data["erode_stats"][pind,3] = longest_erosion
        end 

    ### state preparation ### 
    elseif mode == "quench"

        println("studying quenches...")
        timesteps = min(2minimum(Ls),15)             # number of times to measure the expected decoding time, at most 15
        data["tpreps"] = zeros(nps,timesteps)        # average decoding time
        data["dectest_times"] = zeros(nps,timesteps) # simulation time t at which the decoding test was performed
        data["tprep_errors"] = zeros(nps,timesteps)  # standard deviations of decoding times

        for (pind,thisp) in enumerate(ps) 
            thisL = Ls[pind]
            thisZ = Zs[pind]

            # initialize stuff 
            fields = zeros(Int,thisL,thisL,thisZ,3,2); new_fields = zeros(Int,thisL,thisL,thisZ,3,2)
            state_correction = falses(thisL,thisL,2); hist = falses(thisL,thisL,thisZ); hist_correction = falses(thisL,thisL,thisZ,3); old_synds = falses(thisL,thisL); new_synds = falses(thisL,thisL) 
        
            println("L = $thisL, p = $thisp, Z = $thisZ")

            deltat = Ts[pind] ÷ timesteps # Ts[pind] is the total simulation time; timesteps is the number of checkpoints
            
            tdec_histogram = zeros(Int,timesteps,samps[pind]) # decoding time for each checkpoint and each sample

            @showprogress dt=1 for samp in 1:samps[pind]
                fields .= 0; hist .= false; old_synds .= false; new_synds .= false; hist_correction .= false; state_correction .= false # reset everything
                state,old_synds = init_2d(1/2,thisL,"rand","periodic") # random initial state 
                dataind = 1 
                for t in 1:Ts[pind]
                    update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,thisp,qrat*thisp,synch,pretty)

                    if t % deltat == 1 || deltat == 1 # measure the decoding time
                        tdec = get_decoding_time(state,old_synds,new_synds,hist,r,synch) 
                        tdec_histogram[dataind,samp] = tdec 
                        data["dectest_times"][pind,dataind] = t 
                        dataind += 1 
                    end 
                    if dataind > timesteps break end 
                end
            end 
            data["tpreps"][pind,:] = mean(tdec_histogram,dims=2) # average over samples
            data["tprep_errors"][pind,:] = std(tdec_histogram,dims=2) / sqrt(samps[pind]) # 
            println("ts / L = [", join(round.(data["dectest_times"][pind,:] ./ thisL, digits=2), ", "), "]")
            println("Average preparation times (⟨tprep⟩) for L = $(Ls[pind]): $(join(data["tpreps"][pind,:], ", "))")
        end 

    else ## stuff requiring monte carlo averages (logical error rates, relaxation times, steady-state statistics)

        ### initialize various things ### 
        nsteps = vary_L ? length(Ls) : length(ps)                                          # number of data points
        scalar_quantities = ["Ft" "trials" "trels" "anyon_densities"]
        for key in scalar_quantities 
            data[key] = zeros(nsteps) 
        end
        data["trel_stats"] = zeros(nsteps,3)                                               # mean, std, max of relaxation times

        ### compute things requiring monte carlo averages ### 
        function compute(p,q,samps,L,T,acc_err)
            """ 
            computes monte carlo averages of various quantities for a given value of (p,q,L)
            samps: number of sample in trel mode
            T: number of noisy decoding time step in Ft mode
            acc_err: number of logical errors needed to stop simulation in Ft mode

            returns: 
                mode = trel:  relaxation time/memory lifetime
                mode = Ft:    decoding fidelity, error rate after a fixed number of noisy decoding rounds
                mode = stats: anyon density in long-time steady state 
            uses synch, eta as global parameters (only p, q, and L may vary)
            """

            mc_keys = ["trels" "Ft" "trel_stats" "anyon_densities"] # quantities to compute
            mc_data = Dict{String, Any}(key => 0 for key in mc_keys) # store the results of the monte carlo averages

            # initialize stuff 
            hist = falses(L,L,Z); hist_correction = falses(L,L,Z,3); state = falses(L,L,2); state_correction = falses(L,L,2); fields = zeros(Int,L,L,Z,3,2); new_fields = zeros(Int,L,L,Z,3,2)
            old_synds = falses(L,L); new_synds = falses(L,L)

            # copies used to test decoding without modifying the actual ongoing noisy trajectory; matters especially in trel mode.
            dhist = falses(L,L,Z); dstate = falses(L,L,2); dstate_correction = falses(L,L,2); dfields = zeros(Int,L,L,Z,3,2)
            dold_synds = falses(L,L); dnew_synds = falses(L,L)
            corrected_dstate = falses(L,L,2)

            if mode == "trel"
                mc_data["trel_stats"] = zeros(3) # mean, std, max of relaxation times
                maxT = 50000000
                max_decode_time = 4L # maximum time allowed for offline decoding
                decode_interval = 2 # offline decoding every decode_interval steps

                function run_trel_samples(worker, worker_count)
                    local_hist = falses(L,L,Z); local_hist_correction = falses(L,L,Z,3); local_state = falses(L,L,2); local_state_correction = falses(L,L,2)
                    local_fields = zeros(Int,L,L,Z,3,2); local_new_fields = zeros(Int,L,L,Z,3,2)
                    local_old_synds = falses(L,L); local_new_synds = falses(L,L)
                    local_dhist = falses(L,L,Z); local_dstate = falses(L,L,2); local_dstate_correction = falses(L,L,2); local_dfields = zeros(Int,L,L,Z,3,2)
                    local_dold_synds = falses(L,L); local_dnew_synds = falses(L,L); local_corrected_dstate = falses(L,L,2)
                    tsum = 0.0
                    t2sum = 0.0
                    max_trel = 0

                    for samp in worker:worker_count:samps
                        local_state .= false; local_state_correction .= false; local_old_synds .= false; local_new_synds .= false
                        local_hist .= false; local_hist_correction .= false; local_fields .= 0

                        t = 1
                        while t < maxT
                            update!(local_state,local_state_correction,local_old_synds,local_new_synds,local_hist,local_hist_correction,local_fields,local_new_fields,r,p,q,synch,pretty)

                            if t%decode_interval == 0 # offline decoding on a copy at every decode_interval steps
                                local_dhist .= local_hist; local_dfields .= local_fields; local_dstate .= local_state; local_dstate_correction .= local_state_correction; local_dnew_synds .= local_new_synds; local_dold_synds .= local_old_synds

                                tdec = 0 # time spent decoding
                                while tdec < max_decode_time && any(local_dhist)
                                    update!(local_dstate,local_dstate_correction,local_dold_synds,local_dnew_synds,local_dhist,local_hist_correction,local_dfields,local_new_fields,r,0,0,true,pretty)
                                    tdec += 1
                                end
                                local_corrected_dstate .= local_dstate_correction .⊻ local_dstate
                                if any(get_synds(local_corrected_dstate)) println("decoded state is not logical!, $(tdec/max_decode_time)") end
                                logical_error = 1-detect_logical_error(local_corrected_dstate) # 0 only if both cycles have trivial winding
                                if logical_error == 1 break end
                            end
                            t += 1
                        end
                        if t == maxT
                            println("maxT reached! (sample = $samp)")
                        end
                        tsum += t
                        t2sum += t^2
                        max_trel = max(max_trel, t)
                    end
                    return tsum, t2sum, max_trel
                end

                worker_count = trial_parallel ? min(nthreads(), samps) : 1
                worker_results = Vector{Tuple{Float64,Float64,Int}}(undef, worker_count)
                @threads for worker in 1:worker_count
                    worker_results[worker] = run_trel_samples(worker, worker_count)
                end

                trel_sum = sum(result[1] for result in worker_results)
                trel2_sum = sum(result[2] for result in worker_results)
                max_trel = maximum(result[3] for result in worker_results)
                mc_data["trels"] = trel_sum / samps
                mc_data["trel_stats"][1] = mc_data["trels"]
                mc_data["trel_stats"][2] = trel2_sum / samps
                mc_data["trel_stats"][2] = sqrt(mc_data["trel_stats"][2] - mc_data["trel_stats"][1]^2)
                mc_data["trel_stats"][3] = max_trel
            end 

            if mode == "Ft"
                println("T = $T")
                println("accu errors = $acc_err")

                function run_Ft_trials(target_errors)
                    local_hist = falses(L,L,Z); local_hist_correction = falses(L,L,Z,3); local_state = falses(L,L,2); local_state_correction = falses(L,L,2)
                    local_fields = zeros(Int,L,L,Z,3,2); local_new_fields = zeros(Int,L,L,Z,3,2)
                    local_old_synds = falses(L,L); local_new_synds = falses(L,L)
                    local_failures = 0
                    local_trials = 0

                    while local_failures < target_errors # sample until this worker sees its assigned failures
                        if verbose && local_trials % 10000 == 0
                            println("thread $(threadid()) trial: ", local_trials)
                        end
                        local_hist .= false; local_fields .= 0; local_state .= false; local_state_correction .= false; local_old_synds .= false; local_new_synds .= false

                        for _ in 1:T
                            update!(local_state,local_state_correction,local_old_synds,local_new_synds,local_hist,local_hist_correction,local_fields,local_new_fields,r,p,q,synch,pretty)
                        end
                        cleanup_time = 2T
                        for _ in 1:cleanup_time # finish up by running some (synchronous) ideal decoding to get rid of anyons that remain at positive RG times
                            update!(local_state,local_state_correction,local_old_synds,local_new_synds,local_hist,local_hist_correction,local_fields,local_new_fields,r,0,0,true,pretty)
                            if ~any(local_hist) # if we are in a logical state
                                break
                            end
                        end
                        decoded_state = local_state .⊻ local_state_correction
                        if ~any(local_hist)
                            @assert ~any(get_synds(decoded_state)) "decoded state is not logical!" # check that syndromes are cleaned up
                        else
                            if verbose println("anyons not cleaned up!") end
                        end
                        logical_failure = 1-detect_logical_error(decoded_state) # 0 only if both cycles have trivial winding
                        local_failures += logical_failure
                        local_trials += 1
                        if verbose && logical_failure == 1 println("thread $(threadid()) progress: $(local_failures / target_errors)") end
                    end
                    return local_failures, local_trials
                end

                worker_count = trial_parallel ? min(nthreads(), acc_err) : 1
                worker_results = Vector{Tuple{Int,Int}}(undef, worker_count)
                @threads for worker in 1:worker_count
                    target_errors = acc_err ÷ worker_count + (worker <= acc_err % worker_count ? 1 : 0)
                    worker_results[worker] = run_Ft_trials(target_errors)
                end

                logical_failures = sum(result[1] for result in worker_results)
                trials = sum(result[2] for result in worker_results)
                mc_data["Ft"] = 1-logical_failures/trials # decoding fidelity
                mc_data["trials"] = trials # number of trials
            end

            if mode == "stats"  

                t_therm = params["t_therm"]
                @showprogress dt=1 desc="thermalizing..." for t in 1:t_therm 
                    update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,p,q,synch,false)
                end

                take_data_interval = 1
                t_data = params["t_data"]
                ρa = 0
                @showprogress dt=1 desc="taking data..." for t in 1:t_data 
                    update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,p,q,synch,false)
                    if t%take_data_interval == 0 
                        ρa += sum(hist) / L # anyon "density"
                    end 
                end

                mc_data["anyon_densities"] = ρa / (t_data ÷ take_data_interval) # average anyon density
            end 

            return mc_data
        end 

        for i in 1:nsteps 
            thisp = p; thisL = L; thissamps = samps[i]
            thisT = Ts[i]; this_accu_errors = accu_errors_vec[i]
            if vary_L   
                thisL = Ls[i]; thisT = Ts[i]
                if mode == "Ft"
                    println("L = $thisL | T = $thisT")
                else 
                    println("L = $thisL | samps : $thissamps")
                end 
            else
                thisp = ps[i]
                if mode == "Ft"
                    println("p = $thisp | T = $thisT")
                else 
                    println("p = $thisp | samps : $thissamps")
                end 
            end
            this_mc_data = compute(thisp,thisp*qrat,thissamps,thisL,thisT,this_accu_errors)
            for key in keys(this_mc_data)
                if key == "trel_stats"
                    data[key][i,:] .= this_mc_data[key]
                else
                    data[key][i] = this_mc_data[key]
                end 
            end
        end 
    end 

    # write to file 
    sadj = ~synch ? "_asynch" : ""   
    qadj = qrat == 0 ? "" : "_qrat$qrat"
    padj = ~(vary_L || vary_Z) ? "_p$(round(ps[1],sigdigits=3))to$(round(ps[end],sigdigits=3))" : "_p$(round(p,sigdigits=3))"
    zadj = vary_Z ? "_Z$(Zs[1])to$(Zs[end])" : (vary_L ? "" : "_Z$Z")
    Ladj = vary_L ? "_L$(Ls[1])to$(Ls[end])" : "_L$L"
    logzadj = logZ ? "_logZ" : ""
    if mode == "CNOT_Ft"
        fout = "2d_CNOT_$(CNOT_STYLE)_Ft$qadj$padj$Ladj$zadj$sadj$logzadj$out_adj.txt"
    elseif mode == "CNOT_DEMO"
        fout = "2d_CNOT_$(CNOT_STYLE)_demo$qadj$padj$Ladj$zadj$sadj$logzadj$out_adj.txt"
    else
        fout = "2d_$mode$qadj$padj$Ladj$zadj$sadj$logzadj$out_adj.txt"
    end

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
    if vary_L
        alert("finished | p = $p; L = $(Ls[1]) → $(Ls[end])")
    else 
        alert("finished | L = $L; p = $(ps[1]) → $(ps[end])")
    end 
    println("finished at time $(Dates.now())")
end 

function parameter_repository(mode,L,Z,p,qrat,r,synch,vary_L,vary_Z,logZ)
    nps = 1                         # number of error probabilities p to be tested
    samps = 1; samps_vec = [samps]; ps = [p]; Ls = [L]; Ts = [1]; Zs = [Z]; t_data = 1; t_therm = 1 
    accu_errors = 1                 # number of logical errors to accumulate before terminating sampling (used when computing Ft and doing erosion)
    accu_errors_vec = [accu_errors] # accumulate varying numbers of logical failures per parameter value to speed up the small p simulations

    # some heuristic parameters
    pmin = 0; pmax = 1
    pc = .08 # logZ = true and qrat = 1 
    if qrat == 0 pc = .18 end 
    if ~logZ 
        if qrat == 0 pc = .02 else pc = .09 end 
    end 

    if mode == "trel"
        ps = [p]
        nps = length(ps)

        samps = 1000
        samps_vec = [samps for _ in 1:nps]
        
        Ts = [1 for _ in 1:nps]
        accu_errors_vec = [1 for _ in 1:nps]
    end 

    if mode == "Ft"
        ps = [p]
        nps = length(ps)
          
        Ts = [L for _ in 1:nps]         
        Ls = [L for _ in 1:nps]
        samps_vec = [1 for _ in 1:nps]

        accu_errors_vec = [1000 for i in 1:nps]
        println("number of logical failures to accumulate: ", accu_errors_vec)
    end

    if mode == "CNOT_Ft"
        ps = [p]
        nps = length(ps)

        Ts = [L for _ in 1:nps]
        Ls = [L for _ in 1:nps]
        samps_vec = [1 for _ in 1:nps]

        accu_errors_vec = [1000 for i in 1:nps]
        println("number of primitive CNOT failures to accumulate: ", accu_errors_vec)
    end

    if mode == "CNOT_DEMO"
        ps = [p]
        nps = length(ps)

        Ts = [L for _ in 1:nps]
        Ls = [L for _ in 1:nps]
        samps_vec = [1 for _ in 1:nps]

        accu_errors_vec = [1 for _ in 1:nps]
    end

    if mode == "CNOT_DEBUG"
        ps = [p]
        nps = length(ps)

        Ts = [L for _ in 1:nps]
        Ls = [L for _ in 1:nps]
        samps_vec = [1 for _ in 1:nps]

        accu_errors_vec = [1 for _ in 1:nps]
    end

    params = Dict{String, Any}()
    params["samps"] = samps_vec; params["Ts"] = Ts; params["mode"] = mode; params["L"] = L; params["Ls"] = Ls; params["nps"] = nps; params["ps"] = ps; params["p"] = p; params["vary_L"] = vary_L; params["Z"] = Z; params["Zs"] = Zs; params["qrat"] = qrat; params["r"] = r; params["synch"] = synch; params["accu_errors"] = accu_errors; params["t_therm"] = t_therm; params["t_data"] = t_data; params["vary_Z"] = vary_Z; params["logZ"] = logZ; params["pc"] = pc; params["accu_errors_vec"] = accu_errors_vec

    return params 
end 

main()
