"""
imported from common functions
"""

using Random 
using Alert
using Dates
using Base.Threads

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

function main()
    mode = get(ENV,"MODE","Ft")
    mode == "Ft" || error("2d_windowed_baseline.jl supports only MODE=Ft")

    L = parse(Int,get(ENV,"LVAL","13"))
    L > 0 || error("LVAL must be positive")
    logZ = parse(Bool,get(ENV,"LOGZ","true"))
    buffer_depth_env = lowercase(strip(get(ENV,"BUFFER_DEPTH","1")))
    buffer_depth_factor = buffer_depth_env == "auto" ? 1.0 : parse(Float64,buffer_depth_env)
    buffer_depth_factor > 0 || error("BUFFER_DEPTH must be a positive multiplier")
    Z = logZ ? ceil(Int,buffer_depth_factor * log(1.5,L)) : ceil(Int,buffer_depth_factor * L/4)
    Z > 0 || error("resolved buffer depth Z must be positive")

    p = parse(Float64,get(ENV,"PVAL","0.011"))
    qrat = parse(Float64,get(ENV,"QRAT","1"))
    r = parse(Int,get(ENV,"RVAL","3"))
    synch = parse(Bool,get(ENV,"SYNCH","true"))
    trial_parallel = parse(Bool,get(ENV,"TRIAL_PARALLEL","true"))
    verbose = parse(Bool,get(ENV,"VERBOSE","true"))

    update_time_env = lowercase(strip(get(ENV,"UPDATE_TIME","1")))
    cleanup_time_env = lowercase(strip(get(ENV,"CLEANUP_TIME","2")))
    update_time_factor = update_time_env == "auto" ? 1.0 : parse(Float64,update_time_env)
    cleanup_time_factor = cleanup_time_env == "auto" ? 2.0 : parse(Float64,cleanup_time_env)
    update_time_factor >= 0 || error("UPDATE_TIME must be a nonnegative multiple of L")
    cleanup_time_factor >= 0 || error("CLEANUP_TIME must be a nonnegative multiple of L")
    update_time = round(Int,update_time_factor * L)
    cleanup_time = round(Int,cleanup_time_factor * L)

    stop_mode = lowercase(strip(get(ENV,"STOP_MODE","failures")))
    stop_mode in ("failures","trials") || error("STOP_MODE must be either failures or trials")
    target = if stop_mode == "failures"
        value = parse(Int,get(ENV,"ACC_ERRORS","1000"))
        value > 0 || error("ACC_ERRORS must be positive in failure-stopping mode")
        value
    else
        value = parse(Int,get(ENV,"MAX_TRIALS","10000"))
        value > 0 || error("MAX_TRIALS must be positive in trial-stopping mode")
        value
    end

    repeat_adj = haskey(ENV,"REPEAT_INDEX") ? "_rep$(ENV["REPEAT_INDEX"])" : ""
    out_adj = get(ENV,"OUT_ADJ",repeat_adj)

    params = Dict{String,Any}(
        "mode" => "Ft",
        "L" => L,
        "Ls" => [L],
        "Z" => Z,
        "Zs" => [Z],
        "logZ" => logZ,
        "buffer_depth_factor" => buffer_depth_factor,
        "p" => p,
        "ps" => [p],
        "qrat" => qrat,
        "r" => r,
        "synch" => synch,
        "trial_parallel" => trial_parallel,
        "julia_threads" => nthreads(),
        "update_time_factor" => update_time_factor,
        "cleanup_time_factor" => cleanup_time_factor,
        "update_time" => update_time,
        "cleanup_time" => cleanup_time,
        "stop_mode" => stop_mode,
    )
    if stop_mode == "failures"
        params["acc_errors"] = target
    else
        params["max_trials"] = target
    end

    println("details of simulation:")
    println("mode = Ft")
    println("system size = $L")
    buffer_scale_label = logZ ? "log₁.₅(L)" : "L/4"
    println("buffer depth = $(buffer_depth_factor) × $buffer_scale_label = $Z")
    println("p = $p, q = $(qrat*p)")
    println("synch = $synch")
    println("field update speed = $r")
    println("noisy update time = $(update_time_factor)L = $update_time rounds")
    println("maximum cleanup time = $(cleanup_time_factor)L = $cleanup_time rounds")
    println("stopping mode = $stop_mode, target = $target")
    println("trial parallelism = $(trial_parallel && nthreads() > 1) ($(nthreads()) Julia threads)")

    function run_Ft_trials(worker_target)
        hist = falses(L,L,Z)
        hist_correction = falses(L,L,Z,3)
        state = falses(L,L,2)
        state_correction = falses(L,L,2)
        fields = zeros(Int,L,L,Z,3,2)
        new_fields = zeros(Int,L,L,Z,3,2)
        old_synds = falses(L,L)
        new_synds = falses(L,L)
        local_failures = 0
        local_trials = 0

        while stop_mode == "failures" ? local_failures < worker_target : local_trials < worker_target
            if verbose && local_trials % 10000 == 0
                println("thread $(threadid()) trial: $local_trials")
            end

            hist .= false
            hist_correction .= false
            state .= false
            state_correction .= false
            fields .= 0
            new_fields .= 0
            old_synds .= false
            new_synds .= false

            for _ in 1:update_time
                update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,p,qrat*p,synch,false)
            end
            for _ in 1:cleanup_time
                update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,0,0,true,false)
                !any(hist) && break
            end

            decoded_state = state .⊻ state_correction
            if !any(hist)
                @assert !any(get_synds(decoded_state)) "decoded state is not logical!"
            elseif verbose
                println("anyons not cleaned up!")
            end

            logical_failure = 1-detect_logical_error(decoded_state)
            local_failures += logical_failure
            local_trials += 1
            if verbose && logical_failure == 1
                progress = stop_mode == "failures" ? local_failures/worker_target : local_trials/worker_target
                println("thread $(threadid()) progress: $progress")
            end
        end
        return local_failures,local_trials
    end

    worker_count = trial_parallel ? min(nthreads(),target) : 1
    worker_targets = [target ÷ worker_count + (worker <= target % worker_count ? 1 : 0) for worker in 1:worker_count]
    verbose && println("worker targets = $worker_targets")
    worker_results = Vector{Tuple{Int,Int}}(undef,worker_count)
    @threads for worker in 1:worker_count
        worker_results[worker] = run_Ft_trials(worker_targets[worker])
    end
    verbose && println("worker results = $worker_results")

    logical_failures = sum(result[1] for result in worker_results)
    total_trials = sum(result[2] for result in worker_results)
    data = Dict{String,Any}(
        "Ft" => [1-logical_failures/total_trials],
        "failures" => [logical_failures],
        "trials" => [total_trials],
    )

    sadj = synch ? "" : "_asynch"
    qadj = qrat == 0 ? "" : "_qrat$qrat"
    logzadj = logZ ? "_logZ" : ""
    timing_adj = "_B$(buffer_depth_factor)_Fu$(update_time_factor)_Fc$(cleanup_time_factor)_Tu$(update_time)_Tc$(cleanup_time)"
    stop_adj = stop_mode == "failures" ? "_fail$(target)" : "_trials$(target)"
    fout = "2d_Ft$(qadj)_p$(round(p,sigdigits=3))to$(round(p,sigdigits=3))_L$(L)_Z$(Z)$(sadj)$(logzadj)$(timing_adj)$(stop_adj)$(out_adj).txt"

    println("writing to file: $fout")
    open(fout,"w") do io
        println(io,"### data ###")
        for key in keys(data)
            println(io,"$key = $(repr(data[key]))")
        end
        println(io)
        println(io,"### params ###")
        for key in keys(params)
            println(io,"$key = $(repr(params[key]))")
        end
    end

    if parse(Bool,get(ENV,"ENABLE_ALERT","false"))
        try
            alert("finished | L = $L; p = $p")
        catch err
            @warn "completion notification failed" exception=(err,catch_backtrace())
        end
    end
    println("finished at time $(Dates.now())")
end

main()
