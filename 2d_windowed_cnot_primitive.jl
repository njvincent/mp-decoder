"""
imported from common functions
"""

using Random
using Alert
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

function parse_cnot_fraction(spec, name)
    text = strip(spec)
    if occursin("/", text)
        parts = split(text, "/")
        length(parts) == 2 ||
            error("$name must be a number or a fraction such as 1/4")
        numerator = parse(Int, parts[1])
        denominator = parse(Int, parts[2])
        denominator > 0 || error("$name must have a positive denominator")
        fraction = numerator // denominator
    else
        fraction = parse(Float64, text)
    end
    0 <= fraction <= 1 || error("$name must be between 0 and 1")
    return fraction
end

function split_cnot_timing(T, pre_fraction=0.5, post_fraction=0.5)
    """
    Split the primitive CNOT protocol's total noisy time T according to
    pre_fraction and post_fraction, followed by 2T cleanup rounds.

    The requested fractions must sum to one. The pre-CNOT duration is rounded
    down and the remaining noisy rounds are placed after the CNOT. Thus the
    default half split retains the original odd-T behavior.
    """
    T > 0 || error("CNOT total time T must be positive")
    isapprox(pre_fraction + post_fraction, 1; atol=1e-12, rtol=0) ||
        error("CNOT pre/post fractions must sum to 1")

    T_PRE = floor(Int, T * pre_fraction)
    T_POST = T - T_PRE
    CLEANUP_TIME = 2T
    return T_PRE, T_POST, CLEANUP_TIME
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

function main()
    mode = get(ENV, "MODE", "CNOT_Ft")
    mode == "CNOT_Ft" ||
        error("2d_windowed_cnot_primitive.jl supports only MODE=CNOT_Ft")

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
    has_round_override = haskey(ENV, "CNOT_T_PRE") ||
        haskey(ENV, "CNOT_T_POST")
    has_fraction_override = haskey(ENV, "CNOT_T_PRE_FRACTION") ||
        haskey(ENV, "CNOT_T_POST_FRACTION")
    !(has_round_override && has_fraction_override) ||
        error("use either CNOT_T_PRE/CNOT_T_POST or fraction overrides, not both")

    pre_fraction_spec = "1/2"
    post_fraction_spec = "1/2"
    requested_pre_fraction = 0.5
    requested_post_fraction = 0.5
    timing_source = "default"

    if has_fraction_override
        haskey(ENV, "CNOT_T_PRE_FRACTION") &&
            haskey(ENV, "CNOT_T_POST_FRACTION") ||
            error(
                "CNOT_T_PRE_FRACTION and CNOT_T_POST_FRACTION " *
                "must be set together",
            )
        pre_fraction_spec = ENV["CNOT_T_PRE_FRACTION"]
        post_fraction_spec = ENV["CNOT_T_POST_FRACTION"]
        requested_pre_fraction = parse_cnot_fraction(
            pre_fraction_spec,
            "CNOT_T_PRE_FRACTION",
        )
        requested_post_fraction = parse_cnot_fraction(
            post_fraction_spec,
            "CNOT_T_POST_FRACTION",
        )
        timing_source = "fractions"
    end

    T_PRE, T_POST, default_cleanup_time = split_cnot_timing(
        T,
        requested_pre_fraction,
        requested_post_fraction,
    )
    if has_round_override
        haskey(ENV, "CNOT_T_PRE") && haskey(ENV, "CNOT_T_POST") ||
            error("CNOT_T_PRE and CNOT_T_POST must be set together")
        T_PRE = parse(Int, ENV["CNOT_T_PRE"])
        T_POST = parse(Int, ENV["CNOT_T_POST"])
        T_PRE >= 0 && T_POST >= 0 ||
            error("CNOT_T_PRE and CNOT_T_POST must be nonnegative")
        T_PRE + T_POST == T ||
            error("CNOT_T_PRE + CNOT_T_POST must equal TVAL")
        requested_pre_fraction = T_PRE / T
        requested_post_fraction = T_POST / T
        pre_fraction_spec = string(requested_pre_fraction)
        post_fraction_spec = string(requested_post_fraction)
        timing_source = "rounds"
    end

    cleanup_time_env = lowercase(strip(get(ENV, "CLEANUP_TIME", "auto")))
    cleanup_time = cleanup_time_env == "auto" ?
        default_cleanup_time : parse(Int, cleanup_time_env)
    cleanup_time >= 0 || error("CLEANUP_TIME must be nonnegative")

    cnot_style = get(ENV, "CNOT_STYLE", "primitive")
    cnot_style == "primitive" ||
        error("only CNOT_STYLE=primitive is implemented in this driver")

    acc_errors = parse(Int, get(ENV, "ACC_ERRORS", "1000"))
    fixed_samps = parse(Int, get(ENV, "SAMPS", "0"))
    fixed_samps >= 0 || error("SAMPS must be nonnegative")
    if fixed_samps == 0
        acc_errors > 0 || error("ACC_ERRORS must be positive when SAMPS=0")
    end

    repeat_adj = haskey(ENV, "REPEAT_INDEX") ?
        "_rep$(ENV["REPEAT_INDEX"])" : ""
    timing_adj = has_round_override || has_fraction_override ?
        "_Tpre$(T_PRE)_Tpost$(T_POST)" : ""
    out_adj = get(ENV, "OUT_ADJ", repeat_adj * timing_adj)

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
        "T_PRE" => T_PRE,
        "T_POST" => T_POST,
        "T_PRE_FRACTION_SPEC" => pre_fraction_spec,
        "T_POST_FRACTION_SPEC" => post_fraction_spec,
        "T_PRE_FRACTION" => requested_pre_fraction,
        "T_POST_FRACTION" => requested_post_fraction,
        "T_PRE_RESOLVED_FRACTION" => T_PRE / T,
        "T_POST_RESOLVED_FRACTION" => T_POST / T,
        "timing_source" => timing_source,
        "CLEANUP_TIME" => cleanup_time,
        "CNOT_STYLE" => cnot_style,
        "samps" => [fixed_samps > 0 ? fixed_samps : 1],
        "accu_errors" => acc_errors,
        "accu_errors_vec" => [acc_errors],
    )

    println("details of simulation:")
    println("mode = CNOT_Ft")
    println("CNOT style = $cnot_style")
    println("system size = $L")
    println("RG depth = $Z")
    println("p = $p, q = $(p * qrat)")
    println("synch = $synch")
    println("field update speed = $r")
    println("T = $T")
    println(
        "requested split = $(pre_fraction_spec)T before / " *
        "$(post_fraction_spec)T after",
    )
    println("T_PRE = $T_PRE, T_POST = $T_POST, CLEANUP_TIME = $cleanup_time")
    if fixed_samps > 0
        println("fixed CNOT samples = $fixed_samps")
    else
        println("CNOT failures to accumulate = $acc_errors")
    end
    if haskey(ENV, "REPEAT_INDEX")
        println("repeat index = $(ENV["REPEAT_INDEX"])")
    end
    println(
        "trial parallelism = $(trial_parallel && nthreads() > 1) " *
        "($(nthreads()) Julia threads)",
    )

    data = estimate_primitive_cnot_Ft(
        L,
        Z,
        p,
        p * qrat,
        r,
        synch,
        false,
        T_PRE,
        T_POST,
        cleanup_time,
        acc_errors,
        fixed_samps,
        trial_parallel,
        verbose,
    )

    sadj = synch ? "" : "_asynch"
    qadj = qrat == 0 ? "" : "_qrat$qrat"
    padj = "_p$(round(p, sigdigits=3))to$(round(p, sigdigits=3))"
    logzadj = logZ ? "_logZ" : ""
    fout = "2d_CNOT_$(cnot_style)_Ft$(qadj)$(padj)_L$(L)_Z$(Z)" *
        "$(sadj)$(logzadj)$(out_adj).txt"

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

main()
