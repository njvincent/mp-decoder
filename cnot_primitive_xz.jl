"""
cnot_primitive_xz.jl

Two-sector (X and Z) toric-code simulator for the local windowed
message-passing decoder, with

- single-qubit depolarizing data noise (X, Y, Z each p/3) every round,
- independent measurement flips (probability q) on every X-type and Z-type
  syndrome bit,
- noisy transversal CNOT rounds: data noise, ideal CNOT, two-qubit
  depolarizing noise (total p2, each non-identity Pauli p2/15), measurement,
- the primitive CNOT decoder rule, mirrored per sector: the target's X decoder
  absorbs the control's X decoder; the control's Z decoder absorbs the
  target's Z decoder.

Design: plans/xz_noise_hub_cnot.md. Implemented behavior:
docs/implementation.md. The decoder kernel is the synchronous kernel of
cnot_primitive_consecutive.jl, split into decode / noise / measure phases.
This file imports no other driver.
"""

using Random
using Dates
using Base.Threads

# =============================================================================
# 1. Lattice helpers
# =============================================================================

function nonzeromin(a, b)
    a == 0 && return b
    b == 0 && return a
    return min(a, b)
end

"""
    get_synds!(synds, state)

Vertex parity of an L x L x 2 edge configuration (periodic), written into
`synds`. Applied to a dual-indexed Z-error array it returns plaquette parities.
"""
function get_synds!(synds, state)
    L = size(state, 1)
    @inbounds for j in 1:L, i in 1:L
        im1 = i == 1 ? L : i - 1
        jm1 = j == 1 ? L : j - 1
        synds[i, j] = state[i, j, 1] ⊻ state[i, j, 2] ⊻ state[im1, j, 1] ⊻ state[i, jm1, 2]
    end
    return synds
end

get_synds(state) = get_synds!(falses(size(state, 1), size(state, 2)), state)

"""
    winding_parities(state) -> (xparity, yparity)

The two cut parities used by `detect_logical_error`; either one being true is
a nontrivial winding (logical failure) of that sector.
"""
function winding_parities(state)
    Lx, Ly, _ = size(state)
    xparity = false
    yparity = false
    for i in 1:Lx
        xparity ⊻= state[i, 1, 2]
    end
    for j in 1:Ly
        yparity ⊻= state[1, j, 1]
    end
    return xparity, yparity
end

"""
    detect_logical_error(state)

Legacy name and semantics: returns `true` when both windings are trivial,
i.e. when the logical test SUCCEEDS.
"""
function detect_logical_error(state)
    xparity, yparity = winding_parities(state)
    return !xparity && !yparity
end

"""
    dual_edge(i, j, a, L) -> (i', j', a')

Dual-lattice index of the Z error on primal edge `(i,j,a)`. Dual vertex
`(i,j)` is the plaquette with lower-left corner `(i,j)`, so

    sigma(i,j,1) = (i,   j-1, 2)
    sigma(i,j,2) = (i-1, j,   1)        (indices mod L)

With this map, `get_synds` of a dual-indexed array gives plaquette parities and
`detect_logical_error` gives the Z-sector winding test.
"""
@inline function dual_edge(i, j, a, L)
    if a == 1
        return i, (j == 1 ? L : j - 1), 2
    else
        return (i == 1 ? L : i - 1), j, 1
    end
end

# =============================================================================
# 2. State
# =============================================================================

"""Physical errors and measurement registers of one sector of one block."""
struct SectorState
    errors::BitArray{3}      # L x L x 2; X sector: primal indexing, Z sector: dual
    frame::BitArray{3}       # accumulated correction (Pauli frame), same indexing
    old_synds::BitMatrix     # previous measured syndrome
    new_synds::BitMatrix     # latest measured syndrome
end

"""Lake buffer of one sector: defects, proposals, and messages."""
struct DecoderLane
    hist::Array{Bool,3}             # L x L x Z buffered syndrome-change defects
    hist_correction::Array{Bool,4}  # L x L x Z x 3 proposed links (x, y, buffer)
    fields::Array{Int,5}            # L x L x Z x 3 x 2 messages; 0 means none
    new_fields::Array{Int,5}        # Jacobi buffer
end

struct Sector
    state::SectorState
    lane::DecoderLane
end

"""One toric-code block: X sector and Z sector."""
struct LogicalBlock
    x::Sector
    z::Sector
end

SectorState(L::Int) = SectorState(falses(L, L, 2), falses(L, L, 2), falses(L, L), falses(L, L))

DecoderLane(L::Int, Z::Int) = DecoderLane(
    fill(false, L, L, Z),
    fill(false, L, L, Z, 3),
    zeros(Int, L, L, Z, 3, 2),
    zeros(Int, L, L, Z, 3, 2),
)

Sector(L::Int, Z::Int) = Sector(SectorState(L), DecoderLane(L, Z))

function LogicalBlock(L::Int, Z::Int)
    L > 1 || error("block size L must be greater than 1")
    Z > 1 || error("buffer depth Z must be at least 2")
    return LogicalBlock(Sector(L, Z), Sector(L, Z))
end

function reset!(s::SectorState)
    fill!(s.errors, false)
    fill!(s.frame, false)
    fill!(s.old_synds, false)
    fill!(s.new_synds, false)
    return nothing
end

function reset!(lane::DecoderLane)
    fill!(lane.hist, false)
    fill!(lane.hist_correction, false)
    fill!(lane.fields, 0)
    fill!(lane.new_fields, 0)
    return nothing
end

reset!(s::Sector) = (reset!(s.state); reset!(s.lane); nothing)
reset!(b::LogicalBlock) = (reset!(b.x); reset!(b.z); nothing)

lane_empty(s::Sector) = !any(s.lane.hist)
lanes_empty(b::LogicalBlock) = lane_empty(b.x) && lane_empty(b.z)

"""Per-worker kernel constants and scratch buffers."""
struct KernelContext
    L::Int
    Z::Int
    r::Int
    im1::Vector{Int}
    ip1::Vector{Int}
    unif_edges::Array{Float64,3}   # L x L x 2 uniform draws
    unif_sites::Matrix{Float64}    # L x L uniform draws
end

function KernelContext(L::Int, Z::Int, r::Int)
    r > 0 || error("RVAL must be positive")
    return KernelContext(
        L, Z, r,
        [mod1(i - 1, L) for i in 1:L],
        [mod1(i + 1, L) for i in 1:L],
        zeros(L, L, 2),
        zeros(L, L),
    )
end

# =============================================================================
# 3. Decoder kernel (synchronous legacy rules, non-allocating)
# =============================================================================
#
# These functions reproduce the synchronous branch of legacy `update!`
# (cnot_primitive_consecutive.jl) bit for bit, including the order of the
# back-wall random draws; test/cnot_primitive_xz_runtests.jl checks this
# against the legacy code. Messages are nonnegative integers and 0 means "no
# message", so "any nonzero message" is the same as "a positive minimum exists".

@inline function _candidate(best, hist, f, ii, jj, kk, a, sind, dist)
    if hist[ii, jj, kk]
        best = min(best, dist)
    end
    v = f[ii, jj, kk, a, sind]
    if v != 0
        best = min(best, v + dist)
    end
    return best
end

@inline _message(best) = best == typemax(Int) ? 0 : best

# Source plane at fixed i = ic; offsets (d1, d2) along (j, k); dist = 1 + |d1| + |d2|.
@inline function _cone_x(hist, f, ic, jm, j, jp, km, k, kp, sind)
    best = typemax(Int)
    best = _candidate(best, hist, f, ic, j,  k,  1, sind, 1)
    best = _candidate(best, hist, f, ic, jm, k,  1, sind, 2)
    best = _candidate(best, hist, f, ic, jp, k,  1, sind, 2)
    best = _candidate(best, hist, f, ic, j,  km, 1, sind, 2)
    best = _candidate(best, hist, f, ic, j,  kp, 1, sind, 2)
    best = _candidate(best, hist, f, ic, jm, km, 1, sind, 3)
    best = _candidate(best, hist, f, ic, jm, kp, 1, sind, 3)
    best = _candidate(best, hist, f, ic, jp, km, 1, sind, 3)
    best = _candidate(best, hist, f, ic, jp, kp, 1, sind, 3)
    return _message(best)
end

# Source plane at fixed j = jc; offsets along (i, k).
@inline function _cone_y(hist, f, jc, im, i, ip, km, k, kp, sind)
    best = typemax(Int)
    best = _candidate(best, hist, f, i,  jc, k,  2, sind, 1)
    best = _candidate(best, hist, f, im, jc, k,  2, sind, 2)
    best = _candidate(best, hist, f, ip, jc, k,  2, sind, 2)
    best = _candidate(best, hist, f, i,  jc, km, 2, sind, 2)
    best = _candidate(best, hist, f, i,  jc, kp, 2, sind, 2)
    best = _candidate(best, hist, f, im, jc, km, 2, sind, 3)
    best = _candidate(best, hist, f, im, jc, kp, 2, sind, 3)
    best = _candidate(best, hist, f, ip, jc, km, 2, sind, 3)
    best = _candidate(best, hist, f, ip, jc, kp, 2, sind, 3)
    return _message(best)
end

# Source plane at fixed k = kc (clamped to 1:Z); offsets along (i, j).
@inline function _cone_z(hist, f, kc, im, i, ip, jm, j, jp, sind)
    best = typemax(Int)
    best = _candidate(best, hist, f, i,  j,  kc, 3, sind, 1)
    best = _candidate(best, hist, f, im, j,  kc, 3, sind, 2)
    best = _candidate(best, hist, f, ip, j,  kc, 3, sind, 2)
    best = _candidate(best, hist, f, i,  jm, kc, 3, sind, 2)
    best = _candidate(best, hist, f, i,  jp, kc, 3, sind, 2)
    best = _candidate(best, hist, f, im, jm, kc, 3, sind, 3)
    best = _candidate(best, hist, f, im, jp, kc, 3, sind, 3)
    best = _candidate(best, hist, f, ip, jm, kc, 3, sind, 3)
    best = _candidate(best, hist, f, ip, jp, kc, 3, sind, 3)
    return _message(best)
end

"""
    sweep_fields!(lane, ctx)

One synchronous (Jacobi) message sweep. For component `(a, sind)` with
`s = +1` (sind 1) or `-1` (sind 2), the new message at `(i,j,k)` is the minimum,
over the 3 x 3 source plane one step away along `-s` on axis `a` (buffer index
clamped to 1:Z), of the distance to an active defect and of a nonzero incoming
message plus that distance; 0 if there is none.
"""
function sweep_fields!(lane::DecoderLane, ctx::KernelContext)
    hist = lane.hist
    f = lane.fields
    nf = lane.new_fields
    L = ctx.L
    Z = ctx.Z
    im1 = ctx.im1
    ip1 = ctx.ip1
    @inbounds for k in 1:Z
        km = k > 1 ? k - 1 : 1
        kp = k < Z ? k + 1 : Z
        for j in 1:L
            jm = im1[j]
            jp = ip1[j]
            for i in 1:L
                im = im1[i]
                ip = ip1[i]
                nf[i, j, k, 1, 1] = _cone_x(hist, f, im, jm, j, jp, km, k, kp, 1)
                nf[i, j, k, 1, 2] = _cone_x(hist, f, ip, jm, j, jp, km, k, kp, 2)
                nf[i, j, k, 2, 1] = _cone_y(hist, f, jm, im, i, ip, km, k, kp, 1)
                nf[i, j, k, 2, 2] = _cone_y(hist, f, jp, im, i, ip, km, k, kp, 2)
                nf[i, j, k, 3, 1] = _cone_z(hist, f, km, im, i, ip, jm, j, jp, 1)
                nf[i, j, k, 3, 2] = _cone_z(hist, f, kp, im, i, ip, jm, j, jp, 2)
            end
        end
    end
    copyto!(f, nf)
    return nothing
end

"""Smallest positive message among axes 1:amax at (i,j,k); 0 if none."""
@inline function _min_positive(f, i, j, k, amax)
    m = 0
    @inbounds for s in 1:2, a in 1:amax
        v = f[i, j, k, a, s]
        if v > 0 && (m == 0 || v < m)
            m = v
        end
    end
    return m
end

"""
    perform_correction!(hist, hist_correction, ctx)

Apply every proposed link to the buffer: a spatial link toggles its two
endpoints in the same slice, a buffer link toggles slices k and k+1.
"""
function perform_correction!(hist, hc, ctx::KernelContext)
    L = ctx.L
    Z = ctx.Z
    ip1 = ctx.ip1
    @inbounds for k in 1:Z, j in 1:L, i in 1:L
        if hc[i, j, k, 1]
            hist[i, j, k] ⊻= true
            hist[ip1[i], j, k] ⊻= true
        end
        if hc[i, j, k, 2]
            hist[i, j, k] ⊻= true
            hist[i, ip1[j], k] ⊻= true
        end
        if k < Z && hc[i, j, k, 3]
            hist[i, j, k] ⊻= true
            hist[i, j, k + 1] ⊻= true
        end
    end
    return nothing
end

"""
    decode_step!(sector, ctx, rng)

Legacy synchronous steps 1-6: `r` message sweeps, one frozen round of feedback
proposals (bulk priority +buffer, -x, -y, +y, +x; back wall spatial only, moving
with probability 0.8), frame update from spatial proposals, history commit.
"""
function decode_step!(sec::Sector, ctx::KernelContext, rng::AbstractRNG)
    hist = sec.lane.hist
    hc = sec.lane.hist_correction
    f = sec.lane.fields
    frame = sec.state.frame
    L = ctx.L
    Z = ctx.Z
    im1 = ctx.im1

    for _ in 1:ctx.r
        sweep_fields!(sec.lane, ctx)
    end

    fill!(hc, false)
    # Loop order (i, j, k) matches legacy so back-wall draws consume the RNG
    # in the same order.
    @inbounds for i in 1:L
        im = im1[i]
        for j in 1:L
            jm = im1[j]
            for k in 1:Z
                hist[i, j, k] || continue
                if k < Z
                    m = _min_positive(f, i, j, k, 3)
                    m == 0 && continue
                    if f[i, j, k, 3, 2] == m
                        hc[i, j, k, 3] = true
                    elseif f[i, j, k, 1, 1] == m
                        hc[im, j, k, 1] = true
                    elseif f[i, j, k, 2, 1] == m
                        hc[i, jm, k, 2] = true
                    elseif f[i, j, k, 2, 2] == m
                        hc[i, j, k, 2] = true
                    elseif f[i, j, k, 1, 2] == m
                        hc[i, j, k, 1] = true
                    end
                else
                    m = _min_positive(f, i, j, k, 2)
                    m == 0 && continue
                    rand(rng) < 0.8 || continue
                    if f[i, j, k, 1, 1] == m
                        hc[im, j, k, 1] = true
                    elseif f[i, j, k, 2, 1] == m
                        hc[i, jm, k, 2] = true
                    elseif f[i, j, k, 2, 2] == m
                        hc[i, j, k, 2] = true
                    elseif f[i, j, k, 1, 2] == m
                        hc[i, j, k, 1] = true
                    end
                end
            end
        end
    end

    @inbounds for a in 1:2, j in 1:L, i in 1:L
        parity = false
        for k in 1:Z
            parity ⊻= hc[i, j, k, a]
        end
        parity && (frame[i, j, a] ⊻= true)
    end
    perform_correction!(hist, hc, ctx)
    return nothing
end

"""
    rg_cycle!(lane, ctx)

Legacy RG cycle: XOR slice Z-1 into the back wall, shift slices 1:Z-2 to
2:Z-1, clear slice 1; merge back-wall spatial messages with nonzero-min, shift
messages likewise (the back wall's buffer-direction messages stay), clear
slice 1.
"""
function rg_cycle!(lane::DecoderLane, ctx::KernelContext)
    hist = lane.hist
    f = lane.fields
    L = ctx.L
    Z = ctx.Z
    @inbounds begin
        for j in 1:L, i in 1:L
            hist[i, j, Z] ⊻= hist[i, j, Z - 1]
        end
        for k in (Z - 1):-1:2, j in 1:L, i in 1:L
            hist[i, j, k] = hist[i, j, k - 1]
        end
        for j in 1:L, i in 1:L
            hist[i, j, 1] = false
        end
        for s in 1:2, a in 1:2, j in 1:L, i in 1:L
            f[i, j, Z, a, s] = nonzeromin(f[i, j, Z - 1, a, s], f[i, j, Z, a, s])
        end
        for s in 1:2, a in 1:3, k in (Z - 1):-1:2, j in 1:L, i in 1:L
            f[i, j, k, a, s] = f[i, j, k - 1, a, s]
        end
        for s in 1:2, a in 1:3, j in 1:L, i in 1:L
            f[i, j, 1, a, s] = 0
        end
    end
    return nothing
end

"""
    measure_and_insert!(sector, q, ctx, rng)

Legacy synchronous steps 8-10: shift the syndrome registers, measure the
sector's physical errors with Bernoulli(q) flips, run the RG cycle, and insert
the syndrome change at buffer slice 1.
"""
function measure_and_insert!(sec::Sector, q, ctx::KernelContext, rng::AbstractRNG)
    st = sec.state
    st.old_synds .= st.new_synds
    get_synds!(st.new_synds, st.errors)
    if q > 0
        rand!(rng, ctx.unif_sites)
        st.new_synds .⊻= ctx.unif_sites .< q
    end
    rg_cycle!(sec.lane, ctx)
    hist = sec.lane.hist
    old = st.old_synds
    new = st.new_synds
    @inbounds for j in 1:ctx.L, i in 1:ctx.L
        hist[i, j, 1] = old[i, j] ⊻ new[i, j]
    end
    return nothing
end

decode_block!(b::LogicalBlock, ctx, rng) = (decode_step!(b.x, ctx, rng); decode_step!(b.z, ctx, rng); nothing)
measure_block!(b::LogicalBlock, q, ctx, rng) = (measure_and_insert!(b.x, q, ctx, rng); measure_and_insert!(b.z, q, ctx, rng); nothing)

# =============================================================================
# 4. Noise channels and the transversal CNOT
# =============================================================================

"""
    apply_x_flip_noise!(sector, p, ctx, rng)

Legacy one-sector data channel (each edge flips with probability p), consuming
the RNG exactly like legacy `update!`. Used by the kernel-equivalence test.
"""
function apply_x_flip_noise!(sec::Sector, p, ctx::KernelContext, rng::AbstractRNG)
    if p > 0
        rand!(rng, ctx.unif_edges)
        sec.state.errors .⊻= ctx.unif_edges .< p
    end
    return nothing
end

"""
    apply_single_qubit_depolarizing!(block, p, ctx, rng)

Each data qubit (edge e) independently suffers X, Y, or Z with probability p/3
each: `x[e] ^= (X or Y)`, `zdual[sigma(e)] ^= (Z or Y)`.
"""
function apply_single_qubit_depolarizing!(b::LogicalBlock, p, ctx::KernelContext, rng::AbstractRNG)
    p > 0 || return nothing
    L = ctx.L
    u = ctx.unif_edges
    rand!(rng, u)
    x = b.x.state.errors
    z = b.z.state.errors
    p_x = p / 3
    p_xy = 2p / 3
    @inbounds for a in 1:2, j in 1:L, i in 1:L
        v = u[i, j, a]
        v < p || continue
        if v < p_x
            x[i, j, a] ⊻= true
        elseif v < p_xy
            x[i, j, a] ⊻= true
            di, dj, da = dual_edge(i, j, a, L)
            z[di, dj, da] ⊻= true
        else
            di, dj, da = dual_edge(i, j, a, L)
            z[di, dj, da] ⊻= true
        end
    end
    return nothing
end

# Pauli code 0..3 = I, X, Y, Z
const PAULI_HAS_X = (false, true, true, false)
const PAULI_HAS_Z = (false, false, true, true)

"""
    two_qubit_pauli(m) -> (control_code, target_code)

Map m in 1:15 to the non-identity two-qubit Pauli (control, target), with
codes 0..3 = I, X, Y, Z.
"""
@inline two_qubit_pauli(m::Int) = (m >> 2, m & 3)

"""
    apply_two_qubit_depolarizing!(control, target, p2, ctx, rng)

For every physical pair (edge e of control, edge e of target), with probability
p2 apply one of the 15 non-identity two-qubit Paulis, chosen uniformly.
"""
function apply_two_qubit_depolarizing!(c::LogicalBlock, t::LogicalBlock, p2, ctx::KernelContext, rng::AbstractRNG)
    p2 > 0 || return nothing
    L = ctx.L
    u = ctx.unif_edges
    rand!(rng, u)
    xc = c.x.state.errors
    zc = c.z.state.errors
    xt = t.x.state.errors
    zt = t.z.state.errors
    @inbounds for a in 1:2, j in 1:L, i in 1:L
        v = u[i, j, a]
        v < p2 || continue
        m = min(floor(Int, 15 * (v / p2)), 14) + 1
        pc, pt = two_qubit_pauli(m)
        di, dj, da = dual_edge(i, j, a, L)
        PAULI_HAS_X[pc + 1] && (xc[i, j, a] ⊻= true)
        PAULI_HAS_Z[pc + 1] && (zc[di, dj, da] ⊻= true)
        PAULI_HAS_X[pt + 1] && (xt[i, j, a] ⊻= true)
        PAULI_HAS_Z[pt + 1] && (zt[di, dj, da] ⊻= true)
    end
    return nothing
end

"""
    apply_transversal_cnot!(control, target)

Ideal transversal CNOT on the physical errors: X errors are copied from
control to target, Z errors from target to control. `sigma` is the same in
every block, so the map is elementwise in both indexings.
"""
function apply_transversal_cnot!(c::LogicalBlock, t::LogicalBlock)
    t.x.state.errors .⊻= c.x.state.errors
    c.z.state.errors .⊻= t.z.state.errors
    return nothing
end

"""
    merge_primitive!(dst, src)

Primitive decoder rule for one sector, without the physical error array (the
transversal CNOT handles that): `dst` absorbs `src`'s frame, syndrome
registers, and buffered defects by XOR, and its messages by nonzero-min.
`src` keeps its own decoder state. Scratch buffers of both are cleared.
"""
function merge_primitive!(dst::Sector, src::Sector)
    dst.state.frame .⊻= src.state.frame
    dst.state.old_synds .⊻= src.state.old_synds
    dst.state.new_synds .⊻= src.state.new_synds
    dst.lane.hist .⊻= src.lane.hist
    dst.lane.fields .= nonzeromin.(dst.lane.fields, src.lane.fields)
    fill!(dst.lane.new_fields, 0)
    fill!(src.lane.new_fields, 0)
    fill!(dst.lane.hist_correction, false)
    fill!(src.lane.hist_correction, false)
    return nothing
end

"""
    apply_cnot_decoder_rule!(control, target, rule)

Decoder update at a CNOT. `:primitive` mirrors the primitive merge per
sector: target X absorbs control X; control Z absorbs target Z.
"""
function apply_cnot_decoder_rule!(c::LogicalBlock, t::LogicalBlock, rule::Symbol)
    rule === :primitive || error("unknown CNOT decoder rule: $rule")
    merge_primitive!(t.x, c.x)
    merge_primitive!(c.z, t.z)
    return nothing
end

# =============================================================================
# 5. Readout
# =============================================================================

"""Readout of one block after cleanup."""
struct BlockReadout
    x_bits::Tuple{Bool,Bool}   # X-sector winding parities
    z_bits::Tuple{Bool,Bool}   # Z-sector winding parities
    cleanup_ok::Bool
end

x_failed(r::BlockReadout) = r.x_bits[1] | r.x_bits[2]
z_failed(r::BlockReadout) = r.z_bits[1] | r.z_bits[2]
block_failed(r::BlockReadout) = x_failed(r) | z_failed(r)

"""
    cleanup_and_readout!(block, cleanup_time, ctx, rng)

Ideal cleanup as in the legacy drivers: up to `cleanup_time` rounds with
p = q = 0 (at least one round if `cleanup_time > 0`), stopping once both
buffers are empty; then the winding tests on `errors xor frame` of each sector.
"""
function cleanup_and_readout!(b::LogicalBlock, cleanup_time::Int, ctx::KernelContext, rng::AbstractRNG)
    for _ in 1:cleanup_time
        decode_block!(b, ctx, rng)
        measure_block!(b, 0.0, ctx, rng)
        lanes_empty(b) && break
    end
    ok = lanes_empty(b)
    decoded_x = b.x.state.errors .⊻ b.x.state.frame
    decoded_z = b.z.state.errors .⊻ b.z.state.frame
    if ok
        @assert !any(get_synds(decoded_x)) "X-sector decoded state is not syndrome-free"
        @assert !any(get_synds(decoded_z)) "Z-sector decoded state is not syndrome-free"
    end
    return BlockReadout(winding_parities(decoded_x), winding_parities(decoded_z), ok)
end

# =============================================================================
# 6. Schedules
# =============================================================================

"""A block's role and lifetime: it lives on rounds [first_round, last_round]."""
struct BlockSpec
    role::Symbol        # :hub (persistent block) or :partner
    index::Int          # partner index j (0 for the hub)
    first_round::Int
    last_round::Int
end

"""A CNOT round at `round` between block ids `control` and `target`."""
struct CnotEvent
    round::Int
    control::Int
    target::Int
end

struct Schedule
    name::String
    blocks::Vector{BlockSpec}
    events::Vector{CnotEvent}
    last_round::Int
    description::Dict{String,Any}
end

"""Single block, `T` ordinary rounds."""
function memory_schedule(T::Int)
    T >= 1 || error("memory schedule needs at least one round")
    return Schedule("memory", [BlockSpec(:hub, 0, 1, T)], CnotEvent[], T,
                    Dict{String,Any}("MEMORY_ROUNDS" => T))
end

"""
    consecutive_schedule(intervals; hub_role=:target)

The `cnot_primitive_consecutive.jl` structure with noisy CNOT rounds. With
`intervals = [I1, ..., I(N+1)]`, the hub and partners 1..N all live on rounds
`1 : sum(intervals) + N`; CNOT j is the round after interval j, i.e. round
`I1 + ... + Ij + j`, between partner j and the hub. The hub is the target
(default) or the control.
"""
function consecutive_schedule(intervals::Vector{Int}; hub_role::Symbol=:target)
    length(intervals) >= 2 || error("a consecutive schedule needs at least two intervals")
    all(>=(0), intervals) || error("intervals must be nonnegative")
    hub_role in (:target, :control) || error("hub_role must be :target or :control")
    N = length(intervals) - 1
    total = sum(intervals) + N
    total >= 1 || error("schedule must contain at least one round")
    blocks = [BlockSpec(:hub, 0, 1, total)]
    append!(blocks, [BlockSpec(:partner, j, 1, total) for j in 1:N])
    events = CnotEvent[]
    tau = 0
    for j in 1:N
        tau += intervals[j] + 1
        partner_id = j + 1
        push!(events, hub_role === :target ? CnotEvent(tau, partner_id, 1) : CnotEvent(tau, 1, partner_id))
    end
    desc = Dict{String,Any}(
        "CNOT_INTERVALS" => intervals,
        "CNOT_COUNT" => N,
        "CNOT_ROUNDS" => [ev.round for ev in events],
        "HUB_ROLE" => String(hub_role),
    )
    return Schedule("consecutive", blocks, events, total, desc)
end

"""Schedule with precomputed per-round bookkeeping and static pool slots."""
struct CompiledSchedule
    schedule::Schedule
    slot::Vector{Int}                       # block id -> pool slot
    nslots::Int
    spawn_at::Vector{Vector{Int}}           # round -> block ids starting
    retire_at::Vector{Vector{Int}}          # round -> block ids ending
    live_at::Vector{Vector{Int}}            # round -> live block ids (ascending)
    events_at::Vector{Vector{CnotEvent}}    # round -> CNOT events
    partner_count::Int
end

function compile_schedule(s::Schedule)
    R = s.last_round
    nb = length(s.blocks)
    for (id, b) in enumerate(s.blocks)
        1 <= b.first_round <= b.last_round <= R ||
            error("block $id has an invalid lifetime [$(b.first_round), $(b.last_round)]")
    end
    spawn_at = [Int[] for _ in 1:R]
    retire_at = [Int[] for _ in 1:R]
    live_at = [Int[] for _ in 1:R]
    for (id, b) in enumerate(s.blocks)
        push!(spawn_at[b.first_round], id)
        push!(retire_at[b.last_round], id)
        for tau in b.first_round:b.last_round
            push!(live_at[tau], id)
        end
    end
    events_at = [CnotEvent[] for _ in 1:R]
    for ev in s.events
        1 <= ev.round <= R || error("CNOT event outside the schedule: $ev")
        ev.control != ev.target || error("CNOT control and target must differ: $ev")
        for id in (ev.control, ev.target)
            1 <= id <= nb || error("CNOT event refers to unknown block: $ev")
            b = s.blocks[id]
            b.first_round <= ev.round <= b.last_round || error("block $id is not live at round $(ev.round)")
        end
        push!(events_at[ev.round], ev)
    end
    for tau in 1:R
        used = Int[]
        for ev in events_at[tau]
            append!(used, (ev.control, ev.target))
        end
        length(unique(used)) == length(used) || error("a block appears in two CNOTs at round $tau")
    end
    # static slot assignment (interval coloring in order of first_round, then id)
    order = sortperm(collect(1:nb); by=id -> (s.blocks[id].first_round, id))
    slot = zeros(Int, nb)
    slot_free_after = Int[]        # slot -> last round of its current occupant
    for id in order
        b = s.blocks[id]
        k = findfirst(<(b.first_round), slot_free_after)
        if k === nothing
            push!(slot_free_after, b.last_round)
            slot[id] = length(slot_free_after)
        else
            slot_free_after[k] = b.last_round
            slot[id] = k
        end
    end
    partner_count = count(b -> b.role === :partner, s.blocks)
    return CompiledSchedule(s, slot, length(slot_free_after), spawn_at, retire_at, live_at, events_at, partner_count)
end

# =============================================================================
# 7. Trials and counters
# =============================================================================

struct SimParams
    p::Float64
    q::Float64
    p2::Float64
    gate_mode::Symbol      # :cnot or :idle
    rule::Symbol           # :primitive
    cleanup_time::Int
end

"""Per-worker failure counts. A trial fails if any block fails."""
mutable struct Counters
    trials::Int
    failures::Int
    cleanup_failures::Int              # trials with any cleanup failure
    hub_failures::Int
    hub_x_failures::Int
    hub_z_failures::Int
    hub_x_bit_failures::Vector{Int}    # per winding parity
    hub_z_bit_failures::Vector{Int}
    hub_cleanup_failures::Int
    partner_failures::Int              # failing partners, summed over trials
    partner_x_failures::Int
    partner_z_failures::Int
    partner_cleanup_failures::Int
    partner_failure_trials::Int        # trials with at least one failing partner
    hub_and_partner_failure_trials::Int
    partner_failures_by_index::Vector{Int}
    partner_x_failures_by_index::Vector{Int}
    partner_z_failures_by_index::Vector{Int}
end

Counters(npartners::Int) = Counters(0, 0, 0, 0, 0, 0, zeros(Int, 2), zeros(Int, 2), 0,
                                    0, 0, 0, 0, 0, 0,
                                    zeros(Int, npartners), zeros(Int, npartners), zeros(Int, npartners))

function add!(acc::Counters, c::Counters)
    for name in fieldnames(Counters)
        v = getfield(c, name)
        if v isa Vector{Int}
            getfield(acc, name) .+= v
        else
            setfield!(acc, name, getfield(acc, name) + v)
        end
    end
    return acc
end

"""
    run_trial!(cs, pool, params, ctx, rng, counters) -> trial_failed::Bool

One trial of a compiled schedule. Round order (plans/xz_noise_hub_cnot.md
Section 3): decode all live blocks; single-qubit noise on all live blocks;
CNOT events (ideal gate + decoder rule unless idle, then two-qubit noise);
measure all live blocks; read out blocks whose last round this is.
"""
function run_trial!(cs::CompiledSchedule, pool::Vector{LogicalBlock}, params::SimParams,
                    ctx::KernelContext, rng::AbstractRNG, c::Counters)
    s = cs.schedule
    trial_failed = false
    cleanup_failed = false
    hub_failed = false
    any_partner_failed = false
    for tau in 1:s.last_round
        for id in cs.spawn_at[tau]
            reset!(pool[cs.slot[id]])
        end
        live = cs.live_at[tau]
        for id in live
            decode_block!(pool[cs.slot[id]], ctx, rng)
        end
        for id in live
            apply_single_qubit_depolarizing!(pool[cs.slot[id]], params.p, ctx, rng)
        end
        for ev in cs.events_at[tau]
            cb = pool[cs.slot[ev.control]]
            tb = pool[cs.slot[ev.target]]
            if params.gate_mode === :cnot
                apply_transversal_cnot!(cb, tb)
                apply_cnot_decoder_rule!(cb, tb, params.rule)
            end
            apply_two_qubit_depolarizing!(cb, tb, params.p2, ctx, rng)
        end
        for id in live
            measure_block!(pool[cs.slot[id]], params.q, ctx, rng)
        end
        for id in cs.retire_at[tau]
            r = cleanup_and_readout!(pool[cs.slot[id]], params.cleanup_time, ctx, rng)
            spec = s.blocks[id]
            failed = block_failed(r)
            trial_failed |= failed
            cleanup_failed |= !r.cleanup_ok
            if spec.role === :hub
                hub_failed |= failed
                c.hub_failures += failed
                c.hub_x_failures += x_failed(r)
                c.hub_z_failures += z_failed(r)
                c.hub_x_bit_failures[1] += r.x_bits[1]
                c.hub_x_bit_failures[2] += r.x_bits[2]
                c.hub_z_bit_failures[1] += r.z_bits[1]
                c.hub_z_bit_failures[2] += r.z_bits[2]
                c.hub_cleanup_failures += !r.cleanup_ok
            else
                any_partner_failed |= failed
                c.partner_failures += failed
                c.partner_x_failures += x_failed(r)
                c.partner_z_failures += z_failed(r)
                c.partner_cleanup_failures += !r.cleanup_ok
                c.partner_failures_by_index[spec.index] += failed
                c.partner_x_failures_by_index[spec.index] += x_failed(r)
                c.partner_z_failures_by_index[spec.index] += z_failed(r)
            end
        end
    end
    c.trials += 1
    c.failures += trial_failed
    c.cleanup_failures += cleanup_failed
    c.partner_failure_trials += any_partner_failed
    c.hub_and_partner_failure_trials += hub_failed & any_partner_failed
    return trial_failed
end

# =============================================================================
# 8. Estimator
# =============================================================================

Base.@kwdef struct XZConfig
    L::Int
    Z::Int
    p::Float64
    qrat::Float64 = 1.0
    p2_ratio::Float64 = 10.0
    r::Int = 3
    gate_mode::Symbol = :cnot
    rule::Symbol = :primitive
    schedule::Schedule
    cleanup_time::Int
    stop_mode::String = "trials"      # "trials", "failures", or "capped"
    acc_errors::Int = 1000
    max_trials::Int = 100_000
    trial_parallel::Bool = true
    seed::UInt64 = rand(RandomDevice(), UInt64)
    verbose::Bool = false
end

q_of(cfg::XZConfig) = cfg.p * cfg.qrat
p2_of(cfg::XZConfig) = cfg.p * cfg.p2_ratio

function validate(cfg::XZConfig)
    cfg.L > 1 || error("L must be greater than 1")
    cfg.Z > 1 || error("Z must be at least 2")
    0 <= cfg.p <= 1 || error("p must lie in [0, 1]")
    cfg.qrat >= 0 && q_of(cfg) <= 1 || error("q = p * QRAT must lie in [0, 1]")
    cfg.p2_ratio >= 0 && p2_of(cfg) <= 1 || error("p2 = p * P2_RATIO must lie in [0, 1]")
    cfg.r > 0 || error("r must be positive")
    cfg.gate_mode in (:cnot, :idle) || error("GATE_MODE must be cnot or idle")
    cfg.rule === :primitive || error("only CNOT_RULE=primitive is implemented")
    cfg.cleanup_time >= 0 || error("cleanup time must be nonnegative")
    cfg.stop_mode in ("trials", "failures", "capped") || error("STOP_MODE must be trials, failures, or capped")
    cfg.stop_mode != "trials" && cfg.acc_errors <= 0 && error("ACC_ERRORS must be positive")
    cfg.stop_mode != "failures" && cfg.max_trials <= 0 && error("MAX_TRIALS must be positive")
    cfg.stop_mode != "trials" && cfg.p == 0 && cfg.qrat * cfg.p == 0 &&
        error("failure-based stopping cannot terminate at zero noise; use STOP_MODE=trials")
    return nothing
end

split_quota(total::Int, workers::Int) = [total ÷ workers + (w <= total % workers ? 1 : 0) for w in 1:workers]

"""
    simulate(cfg) -> Dict{String,Any}

Monte Carlo estimate for one parameter point. Workers run independent trials
with their own `Xoshiro` stream derived from `cfg.seed`.
"""
function simulate(cfg::XZConfig)
    validate(cfg)
    cs = compile_schedule(cfg.schedule)
    params = SimParams(cfg.p, q_of(cfg), p2_of(cfg), cfg.gate_mode, cfg.rule, cfg.cleanup_time)
    budget = cfg.stop_mode == "failures" ? cfg.acc_errors : cfg.max_trials
    workers = cfg.trial_parallel ? max(1, min(nthreads(), budget)) : 1
    fail_quota = split_quota(cfg.stop_mode == "trials" ? 0 : cfg.acc_errors, workers)
    trial_quota = split_quota(cfg.stop_mode == "failures" ? 0 : cfg.max_trials, workers)
    results = Vector{Counters}(undef, workers)
    t0 = time()

    function work(w)
        rng = Xoshiro(hash((cfg.seed, w)))
        ctx = KernelContext(cfg.L, cfg.Z, cfg.r)
        pool = [LogicalBlock(cfg.L, cfg.Z) for _ in 1:cs.nslots]
        c = Counters(cs.partner_count)
        while true
            if cfg.stop_mode == "trials"
                c.trials >= trial_quota[w] && break
            elseif cfg.stop_mode == "failures"
                c.failures >= fail_quota[w] && break
            else
                (c.failures >= fail_quota[w] || c.trials >= trial_quota[w]) && break
            end
            run_trial!(cs, pool, params, ctx, rng, c)
            if cfg.verbose && c.trials % 10_000 == 0
                println("worker $w: trials = $(c.trials), failures = $(c.failures)")
            end
        end
        return c
    end

    if workers == 1
        results[1] = work(1)
    else
        @threads for w in 1:workers
            results[w] = work(w)
        end
    end

    total = Counters(cs.partner_count)
    for c in results
        add!(total, c)
    end
    trials = total.trials
    fail_rate = trials > 0 ? total.failures / trials : NaN
    data = Dict{String,Any}(
        "Ft" => 1 - fail_rate,
        "fail_rate" => fail_rate,
        "trials" => trials,
        "failures" => total.failures,
        "cleanup_failures" => total.cleanup_failures,
        "hub_failures" => total.hub_failures,
        "hub_x_failures" => total.hub_x_failures,
        "hub_z_failures" => total.hub_z_failures,
        "hub_x_bit_failures" => total.hub_x_bit_failures,
        "hub_z_bit_failures" => total.hub_z_bit_failures,
        "hub_cleanup_failures" => total.hub_cleanup_failures,
        "partner_count" => cs.partner_count,
        "partner_trials" => cs.partner_count * trials,
        "partner_failures" => total.partner_failures,
        "partner_x_failures" => total.partner_x_failures,
        "partner_z_failures" => total.partner_z_failures,
        "partner_cleanup_failures" => total.partner_cleanup_failures,
        "partner_failure_trials" => total.partner_failure_trials,
        "hub_and_partner_failure_trials" => total.hub_and_partner_failure_trials,
        "partner_failures_by_index" => total.partner_failures_by_index,
        "partner_x_failures_by_index" => total.partner_x_failures_by_index,
        "partner_z_failures_by_index" => total.partner_z_failures_by_index,
        "wall_seconds" => time() - t0,
        "workers" => workers,
    )
    return data
end

"""Parameter record written under `### params ###`."""
function params_record(cfg::XZConfig)
    s = cfg.schedule
    rec = Dict{String,Any}(
        "driver" => "cnot_primitive_xz.jl",
        "schedule" => s.name,
        "L" => cfg.L,
        "Z" => cfg.Z,
        "p" => cfg.p,
        "q" => q_of(cfg),
        "qrat" => cfg.qrat,
        "p2" => p2_of(cfg),
        "P2_RATIO" => cfg.p2_ratio,
        "r" => cfg.r,
        "GATE_MODE" => String(cfg.gate_mode),
        "CNOT_RULE" => String(cfg.rule),
        "noise_model" => "single-qubit depolarizing p (X,Y,Z each p/3); measurement flips q; two-qubit depolarizing p2 after each CNOT",
        "total_rounds" => s.last_round,
        "CLEANUP_TIME" => cfg.cleanup_time,
        "stop_mode" => cfg.stop_mode,
        "seed" => Int128(cfg.seed),
        "julia_threads" => nthreads(),
        "trial_parallel" => cfg.trial_parallel,
        "synch" => true,
    )
    cfg.stop_mode != "trials" && (rec["acc_errors"] = cfg.acc_errors)
    cfg.stop_mode != "failures" && (rec["max_trials"] = cfg.max_trials)
    for (k, v) in s.description
        rec[k] = v
    end
    return rec
end

function write_result(path::AbstractString, data::Dict, params::Dict)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    open(path, "w") do io
        println(io, "### data ###")
        for key in sort!(collect(keys(data)))
            println(io, "$key = $(repr(data[key]))")
        end
        println(io)
        println(io, "### params ###")
        for key in sort!(collect(keys(params)))
            println(io, "$key = $(repr(params[key]))")
        end
    end
    return path
end

"""Default output filename for a configuration."""
function result_filename(cfg::XZConfig)
    s = cfg.schedule
    sched_adj = if s.name == "memory"
        "_T$(s.last_round)"
    elseif s.name == "consecutive"
        "_seq$(s.description["CNOT_COUNT"])_I$(join(s.description["CNOT_INTERVALS"], "-"))" *
            (s.description["HUB_ROLE"] == "target" ? "" : "_hubcontrol")
    else
        ""
    end
    stop_adj = cfg.stop_mode == "trials" ? "_trials$(cfg.max_trials)" :
               cfg.stop_mode == "failures" ? "_fail$(cfg.acc_errors)" :
               "_fail$(cfg.acc_errors)cap$(cfg.max_trials)"
    return "2d_XZ_$(s.name)_$(cfg.rule)_$(cfg.gate_mode)_p$(round(cfg.p, sigdigits=4))" *
           "_qrat$(cfg.qrat)_P2r$(cfg.p2_ratio)_L$(cfg.L)_Z$(cfg.Z)$(sched_adj)$(stop_adj).txt"
end

# =============================================================================
# 9. Command-line driver
# =============================================================================

function parse_int_list(spec::AbstractString)
    entries = [e for e in split(strip(spec), r"[\s,]+") if !isempty(e)]
    return parse.(Int, entries)
end

function config_from_env(env=ENV)
    L = parse(Int, get(env, "LVAL", "13"))
    logZ = parse(Bool, get(env, "LOGZ", "true"))
    Z = haskey(env, "ZVAL") ? parse(Int, env["ZVAL"]) :
        (logZ ? ceil(Int, log(1.5, L)) : ceil(Int, L / 4))
    parse(Bool, get(env, "SYNCH", "true")) || error("cnot_primitive_xz.jl supports only SYNCH=true")
    T = parse(Int, get(env, "TVAL", string(L)))
    T > 0 || error("TVAL must be positive")
    name = lowercase(strip(get(env, "SCHEDULE", "consecutive")))
    schedule = if name == "memory"
        memory_schedule(parse(Int, get(env, "MEMORY_ROUNDS", string(T))))
    elseif name == "consecutive"
        intervals = parse_int_list(get(env, "CNOT_INTERVALS", "$T,$T"))
        hub_role = Symbol(lowercase(strip(get(env, "HUB_ROLE", "target"))))
        consecutive_schedule(intervals; hub_role=hub_role)
    else
        error("SCHEDULE must be memory or consecutive")
    end
    cleanup_env = lowercase(strip(get(env, "CLEANUP_TIME", "auto")))
    cleanup_time = cleanup_env == "auto" ? 2T : parse(Int, cleanup_env)
    seed = haskey(env, "SEED") ? parse(UInt64, env["SEED"]) : rand(RandomDevice(), UInt64)
    return XZConfig(
        L=L, Z=Z,
        p=parse(Float64, get(env, "PVAL", "0.01")),
        qrat=parse(Float64, get(env, "QRAT", "1")),
        p2_ratio=parse(Float64, get(env, "P2_RATIO", "10")),
        r=parse(Int, get(env, "RVAL", "3")),
        gate_mode=Symbol(lowercase(strip(get(env, "GATE_MODE", "cnot")))),
        rule=Symbol(lowercase(strip(get(env, "CNOT_RULE", "primitive")))),
        schedule=schedule,
        cleanup_time=cleanup_time,
        stop_mode=lowercase(strip(get(env, "STOP_MODE", "trials"))),
        acc_errors=parse(Int, get(env, "ACC_ERRORS", "1000")),
        max_trials=parse(Int, get(env, "MAX_TRIALS", "100000")),
        trial_parallel=parse(Bool, get(env, "TRIAL_PARALLEL", "true")),
        seed=seed,
        verbose=parse(Bool, get(env, "VERBOSE", "false")),
    )
end

function main()
    cfg = config_from_env()
    println("cnot_primitive_xz.jl: schedule=$(cfg.schedule.name), L=$(cfg.L), Z=$(cfg.Z), ",
            "p=$(cfg.p), q=$(q_of(cfg)), p2=$(p2_of(cfg)), gate=$(cfg.gate_mode), ",
            "rounds=$(cfg.schedule.last_round), cleanup=$(cfg.cleanup_time), ",
            "stop=$(cfg.stop_mode), threads=$(nthreads())")
    data = simulate(cfg)
    params = params_record(cfg)
    out = get(ENV, "OUTPUT_FILE", "")
    if isempty(out)
        out = joinpath(get(ENV, "OUTPUT_DIR", "."), result_filename(cfg))
    end
    write_result(out, data, params)
    println("trials = $(data["trials"]), failures = $(data["failures"]), fail_rate = $(data["fail_rate"])")
    println("wrote $out at $(Dates.now())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
