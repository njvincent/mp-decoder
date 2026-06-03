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

function main()
    """
    supported modes: 
    * "hist":   get history of one evolution, manual input of evolution time T, error rate p & q, and initial state
    * "erode":  correct random anyon configuration with no further physical or measurement error, output logical error rate and correction time statistics
    * "quench": time evolution of anyon densities and decoding times starting from a random state. used for diagnosing state preparation.
    * "trel":   compute relaxation time/memory lifetime for online decoding 
    * "Ft":     get decoding fidelity (error rate after a fixed number of noisy decoding rounds) for online decoding     
    * "stats":  get anyon density in the long time steady state
    """

    mode = get(ENV, "MODE", "Ft")

    L = parse(Int, get(ENV, "LVAL", "13"))

    logZ = parse(Bool, get(ENV, "LOGZ", "true"))
    Z = logZ ? ceil(Int, log(1.5, L)) : ceil(Int, L/4)

    p = parse(Float64, get(ENV, "PVAL", "0.011"))

    qrat = parse(Float64, get(ENV, "QRAT", "1"))
    r = parse(Int, get(ENV, "RVAL", "3"))
    synch = parse(Bool, get(ENV, "SYNCH", "true"))

    vary_L = false
    vary_Z = false

    pretty = mode == "hist"
    verbose = true

    out_adj = get(ENV, "OUT_ADJ", "_p$(p)_L$(L)")

    params = parameter_repository(mode, L, Z, p, qrat, r, synch, vary_L, vary_Z, logZ)

    # rest of main unchanged

    params = parameter_repository(mode,L,Z,p,qrat,r,synch,vary_L,vary_Z,logZ)
    Ts = params["Ts"]; samps = params["samps"]; # Ts: total simulation time; samps: number of samples per simulation
    ps = params["ps"]; nps = params["nps"];     # nps: the number of error probabilities being tested; ps: list of error probabilities
    Ls = params["Ls"]; Zs = params["Zs"];       # system size L and Z to be tested
    accu_errors = params["accu_errors"]; accu_errors_vec = params["accu_errors_vec"] # stop simulation if logical error count reaches accu_errors

    data_keys = ["Ft" "trials"] ∪ ["hist" "field_hist" "state_hist"] ∪ ["trels" "trel_stats"] ∪ ["erode_times" "erode_stats"] ∪ ["Ms" "binds" "chis" "anyon_densities"] ∪ ["tpreps" "quenched_anyon_densities" "dectest_times" "tprep_errors"]
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
    println("field update speed = $r")
    println("")

    state = falses(L,L,2); state_correction = falses(L,L,2); fields = zeros(Int,L,L,Z,3,2); new_fields = zeros(Int,L,L,Z,3,2); hist = falses(L,L,Z); hist_correction = falses(L,L,Z,3)
    old_synds = falses(L,L); new_synds = falses(L,L) 

    ### write history of evolution ### 
    if mode == "hist" 
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

            # initialize stuff 
            fields = zeros(Int,thisL,thisL,thisZ,3,2); new_fields = zeros(Int,thisL,thisL,thisZ,3,2)
            state_correction = falses(thisL,thisL,2); hist = falses(thisL,thisL,thisZ); hist_correction = falses(thisL,thisL,thisZ,3); old_synds = falses(thisL,thisL); new_synds = falses(thisL,thisL) 
        
            max_erode_time = thisL^2 
            println("L = $thisL, p = $thisp")
            longest_erosion = 0 
            logical_failures = 0 
            trials = 0 
            while logical_failures < accu_errors # sample until a certain number of logical errors are created
                state, old_synds = init_2d(thisp,thisL,"rand","periodic") # random initial state with noise of strength p (assume p is small enough so that logical state has zero holonomies)
                fields .= 0; hist .= false; new_synds .= false; hist_correction .= false; state_correction .= false # reset everything
                t = 0
                while t < max_erode_time  
                    update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,0,0,synch,pretty)
                    t += 1 
                    if ~any(hist) #stop once the syndrome-history array contains no defects
                        break 
                    end 
                end 
                if t == max_erode_time 
                    println("max erosion time reached! $t") 
                end 
                trials += 1 
                corrected_state = state .⊻ state_correction
                if t < max_erode_time @assert sum(get_synds(corrected_state)) == 0 "syndromes not cleaned up: $(get_synds(corrected_state))" end  # check that syndromes are cleaned up
                # detect_logical_error is called even if the copied decoded state still has nonzero syndrome/anyons
                logical_error = 1-detect_logical_error(state .⊻ state_correction) # 0 only if both cycles have trivial winding 

                logical_failures += logical_error

                if t > longest_erosion longest_erosion = t end

                data["erode_times"][pind] += t 
                data["erode_stats"][pind,2] += t^2 
            end 
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
                max_trel = 0 
                decode_interval = 10 # offline decoding every decode_interval steps

                @showprogress dt=1 desc="sampling..." for samp in 1:samps 
                    # reset everything
                    state .= false; state_correction .= false; old_synds .= false; new_synds .= false 
                    hist .= false; hist_correction .= false; fields .= 0

                    t = 1 
                    while t < maxT
                        update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,p,q,synch,pretty)

                        if t%decode_interval == 0 # offline decoding on a copy at every decode_interval steps 
                            dhist .= hist; dfields .= fields; dstate .= state; dstate_correction .= state_correction; dnew_synds .= new_synds; dold_synds .= old_synds

                            tdec = 0 # time spent decoding
                            while tdec < max_decode_time && any(dhist)
                                update!(dstate,dstate_correction,dold_synds,dnew_synds,dhist,hist_correction,dfields,new_fields,r,0,0,true,pretty)
                                tdec += 1
                            end 
                            corrected_dstate .= dstate_correction .⊻ dstate
                            if any(get_synds(corrected_dstate)) println("decoded state is not logical!, $(tdec/max_decode_time)") end 
                            # detect_logical_error is called even if the copied decoded state still has nonzero syndrome/anyons
                            logical_error = 1-detect_logical_error(corrected_dstate) # 0 only if both cycles have trivial winding
                            if logical_error == 1 break end 
                        end 
                        t += 1 
                    end
                    if t == maxT 
                        println("maxT reached! (sample = $samp)")
                    end
                    mc_data["trels"] += t / samps 
                    if t > max_trel max_trel = t end
                    mc_data["trel_stats"][1] += t / samps
                    mc_data["trel_stats"][2] += t^2 / samps
                end 
                mc_data["trel_stats"][2] = sqrt(mc_data["trel_stats"][2] - mc_data["trel_stats"][1]^2)
                mc_data["trel_stats"][3] = max_trel
            end 

            if mode == "Ft"
                println("T = $T")
                println("accu errors = $acc_err")
                logical_failures = 0 
                trials = 0 
                while logical_failures < acc_err # sample until a certain number of logical errors are created
                    if verbose && trials % 10000 == 0
                        println("trial: ", trials)
                    end
                    hist .= false; fields .= 0; state .= false; state_correction .= false; old_synds .= false; new_synds .= false

                    for _ in 1:T 
                        update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,p,q,synch,pretty)
                    end 
                    cleanup_time = 2T 
                    for _ in 1:cleanup_time # finish up by running some (synchronous) ideal decoding to get rid of anyons that remain at positive RG times 
                        update!(state,state_correction,old_synds,new_synds,hist,hist_correction,fields,new_fields,r,0,0,true,pretty)
                        if ~any(hist) # if we are in a logical state  
                            break 
                        end
                    end 
                    decoded_state = state .⊻ state_correction
                    if ~any(hist) 
                        @assert ~any(get_synds(decoded_state)) "decoded state is not logical!" # check that syndromes are cleaned up
                    else 
                        if verbose println("anyons not cleaned up!") end 
                    end 
                    # detect_logical_error is called even if the copied decoded state still has nonzero syndrome/anyons
                    logical_failure = 1-detect_logical_error(decoded_state) # 0 only if both cycles have trivial winding
                    logical_failures += logical_failure
                    trials += 1
                    if verbose if logical_failure == 1 println("$(logical_failures / acc_err)") end end # progress report 
                end 
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
    fout = "2d_$mode$qadj$padj$Ladj$zadj$sadj$logzadj$out_adj.txt"

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
        ps = [0.01, 0.02, 0.03] 
        nps = length(ps)

        samps = 10
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

    params = Dict{String, Any}()
    params["samps"] = samps_vec; params["Ts"] = Ts; params["mode"] = mode; params["L"] = L; params["Ls"] = Ls; params["nps"] = nps; params["ps"] = ps; params["p"] = p; params["vary_L"] = vary_L; params["Z"] = Z; params["Zs"] = Zs; params["qrat"] = qrat; params["r"] = r; params["synch"] = synch; params["accu_errors"] = accu_errors; params["t_therm"] = t_therm; params["t_data"] = t_data; params["vary_Z"] = vary_Z; params["logZ"] = logZ; params["pc"] = pc; params["accu_errors_vec"] = accu_errors_vec

    return params 
end 

main()