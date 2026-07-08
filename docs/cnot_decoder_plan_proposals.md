# CNOT Decoder Plan Proposals

Last updated: 2026-07-08

This document proposes several next decoder plans for improving CNOT decoding
performance while controlling classical space overhead. The proposals are
grounded in:

- Ethan Lake, "Local active error correction from simulated confinement",
  arXiv:2510.08056v3, https://arxiv.org/abs/2510.08056.
- Dennis, Kitaev, Landahl, Preskill, "Topological quantum memory",
  arXiv:quant-ph/0110143v1, https://arxiv.org/abs/quant-ph/0110143.
- The current repository implementations:
  - `2d_windowed_simulation.jl`
  - `2d_windowed_simulation_thread.jl`
  - `2d_windowed_cnot_primitive.jl`
  - `2d_windowed_cnot_sheetcopy.jl`
  - `implementation.md`

The immediate project target is still the X-sector logical CNOT rule:

```text
control_out = control
target_out  = target xor control
```

The proposals below do not claim to implement a full surface-code computation
decoder, a Z-sector rule, or a CNOT gate fault model. They are intended as
concrete next prototypes to compare against the existing primitive and
sheet-copy CNOT implementations.

## Constraints From References

Lake's decoder succeeds by keeping a local, dynamically updated spacetime
defect buffer of depth `Z`. New syndrome-change events enter at `k=1`, flow
toward the back wall at `k=Z`, and are locally paired by message passing while
moving through this buffer. The analysis relies on enough local RG depth that
clusters of diameter below the relevant `Z` scale are corrected before they
reach the back wall. The important implementation consequence is:

```text
Do not discard or unlabeled-merge active defect history merely because a gate
happened. The live history is part of the decoder state.
```

Dennis-Kitaev-Landahl-Preskill give the toric-code conventions that matter for
this project: Pauli error chains are decoded by choosing recovery chains with
the same boundary, logical failure is homological, faulty measurements turn the
problem into a spacetime syndrome-history problem, and CNOT conjugation in the
X sector propagates control X information to the target.

The current code exposes the tradeoff clearly:

- `primitive_cnot_x_sector!` keeps only two baseline decoder states, but it
  xors control history into target history and combines fields with
  `nonzeromin`. That loses lineage separation and target failures dominate the
  existing scans.
- `apply_cnot_x_sheetcopy!` keeps lineage separation by deep-copying every
  active control sheet to the target. That restores much of the performance,
  but each copied sheet stores a full baseline decoder state, including two
  dense field buffers.

The following plans try to keep the useful property of sheet-copy, namely that
copied control history is not destructively merged with target history, without
letting full sheets accumulate with every CNOT.

## Baseline Accounting

For one baseline block, the dominant live memory is:

```text
fields + new_fields = 12 L^2 Z machine Ints
```

On a 64-bit Julia build this is:

```text
M_field64 = 96 L^2 Z bytes
```

The packed Boolean arrays are smaller:

```text
state + state_correction + old_synds + new_synds = 6 L^2 bits
hist + hist_correction = 4 L^2 Z bits
```

Let `M_block` denote one full baseline decoder block or one current
`DecoderSheet`. Let:

```text
B = number of logical blocks
G = number of CNOT gates executed so far
A(t) = number of live, non-quiescent propagated lineages at time t
K = configured lane cap for bounded-lane plans
```

Current leading overhead:

```text
primitive:     B * M_block
sheet-copy:    S(t) * M_block
```

where `S(t)` is the sheet count. For repeated CNOTs, `S(t)` can grow at least
linearly in copied active sheets and can grow Fibonacci-like under alternating
CNOT directions.

## Proposal 1: Settled-Lineage Sheet Compaction

### Motivation

This is the lowest-risk next plan. It preserves the current sheet-copy update
rule for all active histories, so it should initially match sheet-copy decoding
performance. The improvement is to stop storing a full `DecoderSheet` after its
history has locally drained and only its final decoded contribution remains.

The current sheet-copy prototype keeps the representation lineage-oriented all
the way to readout. That is unnecessary once a sheet is quiescent. At that
point, the sheet no longer needs its own fields, `new_fields`,
`hist_correction`, `old_synds`, `new_synds`, or history. It can be merged into
a compact per-block settled accumulator.

### Data Structures

Add a block-level settled accumulator:

```julia
mutable struct BlockSettledState
    decoded_component::BitArray{3}  # L x L x 2
end
```

Keep active sheets in the existing `DecoderSheet` format:

```julia
mutable struct DecoderSheet
    block::Int
    lineage_id::Int
    parent_lineage_id::Union{Int,Nothing}
    created_by_gate::Union{Int,Nothing}
    hist::BitArray{3}
    fields::Array{FieldInt,5}
    new_fields::Array{FieldInt,5}
    hist_correction::BitArray{4}
    state_component::BitArray{3}
    state_correction::BitArray{3}
    old_synds::BitArray{2}
    new_synds::BitArray{2}
end
```

`FieldInt` can initially remain `Int` to minimize behavior changes. Proposal 5
describes reducing it later.

Maintain:

```text
settled[block] :: BlockSettledState
sheets         :: Vector{DecoderSheet}  # active or not-yet-compacted sheets
```

### Quiescence Invariant

A sheet may be compacted only when all of the following are true:

```text
hist is empty
hist_correction is empty
fields is zero
new_fields is zero
get_synds(state_component xor state_correction) is zero
old_synds == new_synds
```

The `old_synds == new_synds` condition prevents compacting a sheet while it is
about to insert a syndrome-change event on the next RG cycle. For a stricter
first implementation, require both `old_synds` and `new_synds` to equal the
syndrome of `state_component`; if that is too strict for noisy measurement
registers, fall back to `old_synds == new_synds`.

### Compaction Rule

When `sheet_quiescent(sheet)` is true:

```text
settled[sheet.block].decoded_component xor=
    sheet.state_component xor sheet.state_correction

remove sheet from sheets
```

This frees the sheet's field buffers, which dominate memory.

Compaction is an algebraic representation change only. It must not change the
decoded state:

```text
decoded_block(block) =
    settled[block].decoded_component xor
    xor_over_active_sheets_on_block(sheet.state_component xor sheet.state_correction)
```

### CNOT Event Rule

For a CNOT from `c` to `t`, apply two operations:

1. Propagate settled decoded contribution:

```text
settled[t].decoded_component xor= settled[c].decoded_component
```

2. Deep-copy each active control sheet exactly as sheet-copy does today:

```text
for sheet in sheets where sheet.block == c && sheet_active(sheet)
    copied = deepcopy(sheet)
    copied.block = t
    copied.parent_lineage_id = sheet.lineage_id
    copied.lineage_id = fresh_lineage_id()
    copied.created_by_gate = gate_id
    push!(sheets, copied)
end
```

The existing target active sheets are unchanged. No immediate field merge is
performed. No active control sheet is modified.

### Local Decoding

Run the existing `update_sheet!` on every active sheet. After each noisy or
cleanup round:

```text
update all active sheets independently
compact every quiescent sheet
```

The first implementation should keep the current sheet-copy stochastic model:
each sheet receives its own physical and measurement noise sample. A second
implementation should add the shared-physical-noise model described below.

### Readout And Failure Rule

Use:

```text
decoded_state(block) =
    settled[block].decoded_component xor
    xor_over_sheets_on_block(sheet.state_component xor sheet.state_correction)
```

Cleanup success:

```text
all active sheet hists are empty
```

Logical failure remains:

```text
!detect_logical_error(decoded_state(control)) ||
!detect_logical_error(decoded_state(target))
```

Record cleanup failure separately, as the current CNOT scripts do. Also add a
new optional metric that ORs cleanup failure into logical failure so later
analysis can compare both conventions.

### Expected Performance

For one CNOT, performance should match sheet-copy up to representation bugs. It
keeps each active copied history independent until local decoding drains it.

For repeated CNOTs separated by enough decoder rounds, performance should stay
close to sheet-copy while avoiding permanent lineage growth. If gates are
dense enough that active histories do not drain, performance and memory reduce
to ordinary sheet-copy for those live histories.

### Space And Runtime Scaling

Leading memory:

```text
B * 2 L^2 bits for settled decoded components
+ A(t) * M_block
```

where `A(t)` is the number of active, not-yet-compacted sheets. This replaces
growth with total CNOT count by growth with the number of live unsettled
lineages.

Runtime per decoder round:

```text
O(A(t) * baseline_update_time)
```

The CNOT event still copies full active sheets and costs:

```text
O(active_control_sheets * L^2 Z)
```

### Weaknesses

- If CNOTs occur faster than histories drain, the active sheet count can still
  grow quickly.
- This plan improves asymptotic repeated-CNOT overhead only for circuits with
  enough time, cleanup, or scheduling slack between CNOTs.
- It preserves the current sheet-copy caveat that future noise is sampled
  independently on each sheet.

### Implementation Priority

Implement this first. It is the easiest way to separate "does compaction help
space?" from "does a new decoding rule hurt threshold?"

## Proposal 2: Shared-Noise Active Sheets

### Motivation

The current sheet-copy prototype applies independent future physical and
measurement noise to every sheet assigned to the same block. That is a useful
prototype, but it is not the physical block model. After a CNOT has copied
control history into the target, future target noise should be a property of
the target block, not an independent property of each target lineage.

Shared-noise sheets should reduce artificial noise multiplication on blocks
with multiple lineages. That may improve CNOT fidelity while keeping the
lineage separation that primitive loses.

### Data Structures

Keep Proposal 1's settled state and active sheets. Add one primary live-noise
lane per block:

```text
primary_sheet[block] :: DecoderSheet
shadow_sheets        :: Vector{DecoderSheet}
```

Interpretation:

- `primary_sheet[block]` receives future physical and measurement noise for
  that block.
- `shadow_sheets` carry pre-gate propagated history and corrections. They
  receive decoder feedback and RG cycling, but not independent physical noise.

For a first implementation, `primary_sheet` can simply be the original block
sheet. A sheet copied by CNOT becomes a `shadow_sheet`.

### Update Rule

For each block update:

1. Draw one physical noise sample for the block:

```text
noise_state :: L x L x 2 Bool
noise_synd  :: L x L Bool
```

2. Apply `update!` with `p=0,q=0` to all shadow sheets. This lets their
   existing histories drain without injecting new errors into those lineages.

3. Apply the ordinary `update!` to the primary sheet using the drawn noise.
   To avoid duplicating the current `rand` calls inside `update!`, refactor the
   baseline update into:

```text
update_feedback_and_rg!(...)
insert_noise_and_new_syndrome!(..., noise_state, noise_synd)
```

The primary sheet receives the block's new physical and measurement noise.
Shadow sheets do not.

4. Compact quiescent shadow sheets using Proposal 1.

### CNOT Event Rule

For CNOT `c -> t`:

```text
settled[t] xor= settled[c]
copy active control primary/shadow sheets into target shadow_sheets
```

The copied sheets represent pre-gate control history now carried by target.
They should not receive future target noise.

The target primary sheet remains responsible for target's future noise.

### Readout

Same as Proposal 1:

```text
decoded_state(block) =
    settled[block] xor
    primary_sheet[block].decoded_component xor
    xor_over_shadow_sheets_on_block(decoded_component)
```

where `decoded_component = state_component xor state_correction`.

### Expected Performance

This should be at least as physical as the current sheet-copy model. It may
improve target failure rates because copied control lineages no longer receive
extra independent target-like noise. It should still avoid primitive's lossy
history merge.

### Space And Runtime Scaling

Space is the same as Proposal 1:

```text
B primary sheets + A_shadow(t) active shadow sheets + settled accumulators
```

Runtime is also proportional to the number of active sheets, but shadow updates
are cheaper if the refactor lets them skip physical-noise generation and
syndrome measurement noise.

### Weaknesses

- This changes the stochastic model, so compare it separately from Proposal 1.
- It requires a larger `update!` refactor to inject explicit noise samples.
- If shadow sheets do not receive measurement noise, their `old_synds` and
  `new_synds` must still be advanced consistently during RG cycles.

## Proposal 3: Correction-Forwarding Subscriptions

### Motivation

This is the most aggressive space-reduction plan. At an X-sector CNOT, the
target needs the control's pre-gate raw error contribution and the corrections
that will later be inferred for that pre-gate control history. Instead of
copying the control history and fields into a target sheet, keep decoding the
control history in the control block and forward the relevant future correction
edges to the target.

The intended advantage is that the target receives the benefit of the control
decoder's full history and fields without storing another dense field buffer.

### Data Structures

Add a subscription object:

```julia
mutable struct CnotSubscription
    source_block::Int
    target_block::Int
    gate_id::Int
    hist_mask::BitArray{3}       # L x L x Z, source defects present at gate
    active::Bool
end
```

The source block keeps its ordinary baseline arrays:

```text
state, state_correction, old_synds, new_synds,
hist, hist_correction, fields, new_fields
```

The target block keeps its ordinary arrays. No target lineage sheet is created.

### CNOT Event Rule

For `c -> t`:

```text
state_t            xor= state_c
state_correction_t xor= state_correction_c
old_synds_t        xor= old_synds_c
new_synds_t        xor= new_synds_c
push!(subscriptions, CnotSubscription(c, t, gate_id, copy(hist_c), true))
```

Do not xor `hist_c` into `hist_t`.
Do not merge `fields_c` into `fields_t`.
Clear `new_fields_c` and `new_fields_t` only as a stale-scratch precaution.

The `old_synds` and `new_synds` xor keeps the target syndrome registers
consistent with the raw state xor, without inserting the control history into
the target decoder.

### Forwarding Update Rule

Refactor the baseline feedback step so the chosen correction source site is
available. When the source block creates a correction link for an active
history event at `(i,j,k)`, do:

```text
for sub in subscriptions where sub.source_block == source_block && sub.active
    if sub.hist_mask[i,j,k]
        mark the same correction link in sub_mask_correction
        if correction axis is spatial, xor that edge into
            state_correction[target_block]
        end
    end
end
```

After the source block applies `perform_correction!(hist, hist_correction)`,
also apply:

```text
perform_correction!(sub.hist_mask, sub_mask_correction)
```

Vertical RG-time corrections update only the mask. Spatial corrections update
both the mask and the target `state_correction`, because those are physical
target correction edges induced by the propagated control component.

The subscription becomes inactive when:

```text
hist_mask is empty
```

Inactive subscriptions can be dropped.

### Readout

Use the ordinary two-block primitive readout:

```text
decoded_state_c = state_c xor state_correction_c
decoded_state_t = state_t xor state_correction_t
```

There is no final sheet merge.

### Expected Performance

This can outperform primitive because it never destructively merges control
history into target history or merges control fields into target fields. It can
approach sheet-copy only if the source decoder's future corrections for
pre-gate control history are a good proxy for the independent copied-control
sheet that sheet-copy would have placed on the target.

This plan is especially attractive for repeated CNOTs because active
subscriptions store only masks, not dense field buffers.

### Space And Runtime Scaling

Leading memory:

```text
B * M_block
+ S_sub(t) * L^2 Z bits
```

where `S_sub(t)` is the number of active subscriptions.

Runtime overhead per source update:

```text
O(S_sub_for_source * number_of_correction_links)
```

or, in a simple dense implementation:

```text
O(S_sub_for_source * L^2 Z)
```

No additional field update is required for the subscription.

### Known Weakness

The single-mask rule is not an exact colored-history decoder. If a pre-gate
masked defect and a post-gate unmasked defect collide in the source block, the
ordinary source history can annihilate them while the mask still carries
lineage information that the scalar source history no longer sees. This is the
main research risk.

Two safer variants should be considered if the simple subscription plan fails:

1. `source-color-2`: maintain two source histories, pre-gate and post-gate,
   but share one physical state. This costs one extra Boolean history and one
   extra correction mask, still without copying fields.
2. `subscription-with-shadow-fields`: store the masked pre-gate history as a
   shadow sheet with compressed or scratch fields, as in Proposal 5.

### Validation Focus

This plan should be tested before investing in a full refactor:

- zero-noise CNOT must pass for all `L` and timing splits,
- fixed-seed runs with post-gate `p=q=0` should match sheet-copy more closely
  than primitive,
- normal noisy runs should check whether target failures remain primitive-like
  or move toward sheet-copy,
- record how often masked defects collide with unmasked defects.

## Proposal 4: Bounded Labeled Lanes

### Motivation

The sheet-copy representation is exact for lineage separation but unbounded in
the number of active sheets. A bounded-lane decoder stores a fixed number `K`
of independent lineage lanes per logical block. This gives a hard memory cap
and provides a path toward many-CNOT circuits.

This is the most natural long-term computation decoder if Proposal 1 still
leaves too much active-sheet growth and Proposal 3 loses too much threshold.

### Data Structures

For each block:

```text
lane_meta[1:K]             # lineage id, parent id, created_by_gate, active flag
hist[L,L,Z,K]              # Boolean defect history per lane
hist_correction[L,L,Z,3,K] # proposed corrections per lane
state_component[L,L,2,K]   # raw contribution per lane
state_correction[L,L,2,K]  # correction contribution per lane
old_synds[L,L,K]
new_synds[L,L,K]
fields[L,L,Z,3,2,K]
new_fields[L,L,Z,3,2,K]
settled_decoded[L,L,2]
```

The first prototype can literally use one field pair per lane. Later, replace
`fields[...,K]` with top-`m` labeled fields per direction:

```text
field_dist[L,L,Z,3,2,m]
field_lane[L,L,Z,3,2,m]
```

where `m <= K` is usually `2` or `3`.

### Lane Invariants

- A defect in lane `ell` may only follow fields emitted by lane `ell`.
- Corrections from lane `ell` update only `state_correction[:,:,:,ell]`.
- Future physical noise is inserted only into the primary lane for each block,
  unless intentionally reproducing the current sheet-copy stochastic model.
- A lane can be compacted into `settled_decoded` when it satisfies the same
  quiescence rule as Proposal 1.
- The block decoded state is:

```text
settled_decoded xor xor_over_active_lanes(state_component[ell] xor state_correction[ell])
```

### CNOT Event Rule

For `c -> t`:

```text
settled_decoded[t] xor= settled_decoded[c]

for each active lane ell in control:
    allocate free lane ell2 in target
    if no lane is free:
        compact quiescent lanes
    if still no lane is free:
        use overflow policy

    target.lane[ell2] = deep copy of control.lane[ell]
    target.lane_meta[ell2].parent = control.lane_meta[ell].lineage_id
    target.lane_meta[ell2].created_by_gate = gate_id
```

No destructive xor merge is performed between lanes.

### Overflow Policies

The overflow policy is part of the experiment and must be recorded in output
metadata:

```text
reject_gate:
    stop the trial and count an overflow failure.

flush_then_gate:
    run ideal local cleanup for up to tau_flush rounds, compact quiescent
    lanes, then retry allocation.

oldest_primitive_merge:
    merge the two oldest lanes by xor plus nonzeromin and increment a
    lossy_merge counter.

lowest_activity_merge:
    merge the two lanes with the fewest active history defects and increment a
    lossy_merge counter.
```

`reject_gate` is best for measuring required `K`. `flush_then_gate` is best for
gate-scheduled circuits. The merge policies are only fallback heuristics.

### Local Decoding

For the full-field prototype, run baseline `update!` independently per active
lane, with the shared-noise option from Proposal 2. For top-`m` labeled fields:

1. Field update at `(i,j,k,a,s)` collects candidate messages from neighboring
   sites.
2. Keep the smallest nonzero distance per lane.
3. Retain the smallest `m` lane-distance pairs.
4. A defect in lane `ell` only reads candidate fields with `field_lane == ell`.
5. If no same-lane field exists, the defect does not move.

This rule is a bounded-memory approximation to full per-lane fields. It should
be exact when no more than `m` distinct lane messages compete at a local field
entry.

### Expected Performance

For `K` at least the maximum live lineage width, the full-field version should
match compacted sheet-copy. With top-`m` fields, performance should interpolate
between sheet-copy and primitive depending on how often more than `m` labels
compete locally.

This plan should control repeated-CNOT overhead better than sheet-copy:

```text
space <= B * K * M_block + settled accumulators
```

with `K` fixed by the intended circuit schedule rather than total CNOT count.

### Weaknesses

- Full-field bounded lanes still multiply baseline memory by `K`.
- Top-`m` fields are approximate and need careful validation near threshold.
- Overflow behavior introduces a new failure mode or a gate-scheduling
  constraint.

## Proposal 5: Field Compression And Local Fast Annihilation

This proposal is orthogonal to Proposals 1-4. It should be applied to whichever
CNOT strategy survives initial validation.

### Part A: Narrow Field Integer Type

The fields store finite message distances, while `0` means "no message". For
current `L` and `Z`, `Int64` is much larger than needed. Replace:

```julia
Array{Int,5}
```

with:

```julia
Array{UInt16,5}
```

or with a type alias:

```julia
const FieldInt = UInt16
```

Rules:

- `0` remains "no message".
- The update uses a wider temporary integer when adding distances:

```julia
candidate = UInt32(field) + UInt32(dist)
```

- Saturate at `typemax(FieldInt)` if needed, but record an assertion that
  `L + Z + r + cleanup_time` stays below the saturation scale in normal runs.

This reduces dominant field memory by `4x` on a 64-bit Julia build.

### Part B: Recomputed Shadow Fields

For copied shadow lanes, store:

```text
hist
hist_correction
state_component
state_correction
old_synds
new_synds
```

but not persistent `fields` or `new_fields`. Instead, maintain one scratch field
workspace per worker:

```text
shadow_fields[L,L,Z,3,2]
shadow_new_fields[L,L,Z,3,2]
```

Before updating a shadow lane:

```text
shadow_fields .= 0
anyons_source_fields!(shadow.hist, shadow_fields)
repeat r_warm times:
    update_2d_windowed_fields!(shadow_fields, shadow_new_fields, shadow.hist)
choose feedback using shadow_fields
apply correction and RG cycle to shadow.hist
```

Recommended first value:

```text
r_warm = max(r, Z)
```

This trades runtime for space. It will not be bitwise equivalent to sheet-copy,
because persistent message fields are no longer carried between rounds. The
validation question is whether keeping the history but rebuilding the fields is
enough to keep threshold behavior close to sheet-copy.

### Part C: Local Pair Fast Annihilation

Lake notes that an easy performance improvement is to eliminate neighboring
anyon pairs before they send messages. Implement the analogous rule in the
`(x,y,k)` history lattice.

Before the ordinary field update in each synchronous round, run a deterministic
checkerboard pass over active histories. For each lane or sheet:

```text
for parity in 0:1
    for every active defect u=(i,j,k) with checkerboard parity
        inspect candidate neighbors in priority order:
            +z if k < Z
            -x
            -y
            +y
            +x
        if exactly one neighboring active defect v is found and no correction
        touching u or v has already been scheduled in this pass:
            schedule the link u-v as hist_correction
            if link is spatial, xor the corresponding state_correction edge
            toggle u and v in hist
```

On the back wall, omit `+z` and use only spatial neighbors. Keep the existing
back-wall stochasticity for non-fast-path moves.

This rule is local, homogeneous, and should reduce short-distance clutter that
otherwise creates competing messages. It should be tested first on the baseline
memory decoder, then mirrored into the CNOT prototypes.

### Expected Effect

- Narrow fields reduce memory without changing algorithmic behavior.
- Recomputed shadow fields reduce per-shadow memory from dense integer fields
  to packed histories plus one shared scratch field pair.
- Fast annihilation should improve both runtime and threshold trend by removing
  easy local pairs before they screen longer-range messages.

## Recommended Implementation Order

1. Implement Proposal 1, `sheetcopy_compact`, with `FieldInt = Int`.
   This should be a behavior-preserving representation improvement relative to
   current sheet-copy.
2. Add metrics for active sheet count, compacted sheet count, settled-state
   nontrivial winding, and peak memory estimate.
3. Add Proposal 5A, `FieldInt = UInt16`, behind a constant or environment
   switch. Verify no saturation.
4. Implement Proposal 2, shared-noise active sheets, as a separate mode.
5. Prototype Proposal 3 correction-forwarding in a small branch or new file.
   Treat it as a research-risk mode until collision metrics are understood.
6. Implement Proposal 4 only after repeated-CNOT tests show that compaction
   alone is insufficient.
7. Add Proposal 5C fast annihilation after a baseline-only validation pass.

## Validation Experiments

All new modes should compare against:

- baseline memory decoder,
- primitive CNOT,
- sheet-copy CNOT,
- sheet-copy with compaction, once implemented.

### Deterministic Sanity Checks

Run:

```bash
MODE=CNOT_DEBUG LVAL=3 LOGZ=false julia --threads=1 2d_windowed_cnot_primitive.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false julia --threads=1 2d_windowed_cnot_sheetcopy.jl
```

Add equivalent debug modes for each new proposal.

Required assertions:

```text
zero-noise CNOT succeeds
decoded_state is unchanged by compaction
quiescent sheets have zero syndrome after decoded merge
no copied arrays alias mutable parent arrays
settled readout equals explicit un-compacted sheet readout
FieldInt never saturates
```

### Small Noisy Smoke Tests

Use:

```text
L in {5, 7}
p in {0.005, 0.010, 0.015}
qrat = 1
T = L
T_PRE = floor(T/2)
T_POST = ceil(T/2)
CLEANUP_TIME = 2T
fixed_samps = 200 to 1000
```

Record:

```text
CNOT_Ft
logical_failures
control_logical_failures
target_logical_failures
both_logical_failures
cleanup_failures
peak_active_lineages
peak_total_lineages
compacted_lineages
subscription_count or lane_overflow_count, if applicable
estimated_peak_bytes
runtime_seconds
```

### Threshold-Trend Scan

Reuse the current split-timing grid:

```text
L in {5, 9, 13, 19}
p in {0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017}
qrat = 1
r = 3
synch = true
LOGZ = true
T = L
cleanup = 2T
5 repeats
ACC_ERRORS = 1000 per repeat
```

The new mode is promising only if:

```text
CNOT_Ft crossing trend is closer to sheet-copy than primitive
target failures do not dominate as strongly as primitive
cleanup failures remain comparable to sheet-copy
peak memory estimate is below sheet-copy for repeated-CNOT stress tests
```

### Repeated-CNOT Stress Test

Add a new driver with two or three logical blocks and alternating CNOTs:

```text
depth D in {1, 2, 4, 8, 16}
pattern 1: C1 -> C2 repeated
pattern 2: C1 -> C2, C2 -> C1 alternating
pattern 3: C1 -> C2, C1 -> C3 fanout
idle rounds between gates in {0, floor(L/4), floor(L/2), L}
```

This is the decisive test for classical overhead. Record active lineage width
as a function of gate count, not only final failure rate.

## Decision Criteria

Use the following ranking for deciding which plan to implement beyond
prototype:

1. Logical performance:
   - target failure rates near sheet-copy,
   - threshold trend not visibly primitive-like,
   - cleanup failures controlled.
2. Classical space:
   - peak field-buffer count grows with live lineage width, not total CNOT
     count,
   - repeated alternating CNOTs do not produce uncontrolled growth,
   - field integer compression has no saturation events.
3. Runtime:
   - runtime per round scales with active sheets, lanes, or subscriptions as
     predicted,
   - threaded trial-level scaling remains close to current threaded baseline.
4. Implementation risk:
   - no silent change to logical failure criteria,
   - no change to baseline update rules unless deliberately mirrored and
     validated,
   - all output metadata records the CNOT mode and overhead mode.

## Summary Recommendation

Start with settled-lineage sheet compaction plus explicit memory metrics. It is
the least risky way to reduce repeated-CNOT space overhead while preserving the
good sheet-copy threshold behavior. Then add narrow field types, which should
give an immediate constant-factor memory reduction. After those are validated,
test shared-noise sheets and correction-forwarding subscriptions. Use bounded
labeled lanes only if repeated-CNOT stress tests show that compaction still
leaves too much live lineage width.
