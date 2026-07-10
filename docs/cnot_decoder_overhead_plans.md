# CNOT Decoder Plans With Explicit Classical Overhead

Last updated: 2026-07-09

This document proposes X-sector CNOT decoder plans for the current QEC CNOT
decoder project. It is based on the project guide in `agent.md`, the paper
notes in `docs/qec_paper_index.md`,
`docs/lake_2025_simulated_confinement.md`, and
`docs/dklp_2001_topological_quantum_memory.md`, the implementation notes in
`docs/implementation.md`, and the current Julia paths:

- `visualizations/2d_windowed_simulation.jl`
- `visualizations/2d_windowed_simulation_thread.jl`
- `visualizations/2d_windowed_cnot_primitive.jl`
- `visualizations/2d_windowed_cnot_sheetcopy.jl`

The immediate scope is only the X-sector logical CNOT rule:

```text
control_out = control
target_out  = target xor control
```

These plans do not implement a full surface-code computation decoder, a
Z-sector rule, or a CNOT gate fault model. Those are future work and should not
be mixed into the first implementation.

## Baseline Facts From The Current Code

The memory decoder stores one sector of a toric-code block. One baseline block,
or one `DecoderSheet` in the sheet-copy prototype, contains:

```text
state             :: L x L x 2 Bool
state_correction  :: L x L x 2 Bool
old_synds         :: L x L Bool
new_synds         :: L x L Bool
hist              :: L x L x Z Bool
hist_correction   :: L x L x Z x 3 Bool
fields            :: L x L x Z x 3 x 2 Int
new_fields        :: L x L x Z x 3 x 2 Int
```

The dominant memory is the pair of dense field buffers:

```text
fields + new_fields = 12 L^2 Z machine Ints.
```

Let `M_block` mean the memory of one such baseline block or full
`DecoderSheet`, including dense fields, histories, corrections, syndrome
registers, and physical/correction states. On 64-bit Julia the leading field
term is:

```text
M_block = 96 L^2 Z bytes + lower-order packed Bool arrays.
```

Let:

```text
B        = number of logical blocks
L        = code distance / lattice linear size
Z        = decoder history depth
D        = CNOT circuit depth
A(t)     = number of live unresolved lineages at time t
K        = bounded lane cap, if applicable
S_sub(t) = number of active subscriptions, if applicable
M_block  = memory of one baseline decoder block / full DecoderSheet
```

The synchronous baseline update is dominated by `r` field sweeps plus feedback,
correction, noise, syndrome refresh, RG cycling, and event insertion:

```text
T_block_round = Theta(r L^2 Z + L^2 Z)
              = Theta(r L^2 Z) for fixed feedback rules.
```

The primitive CNOT rule in `primitive_cnot_x_sector!` keeps only two baseline
blocks and immediately xors control state, correction, syndrome registers,
history, and nonzero-min fields into the target. Its memory is:

```text
O(B * M_block)
```

but it destructively merges active defect history and target failures dominate
the current scans.

The sheet-copy CNOT rule in `apply_cnot_x_sheetcopy!` deep-copies every active
control sheet to the target and only xors sheets at readout. It preserves
lineage information and performs much closer to the baseline, but its memory is:

```text
O(S(t) * M_block),
```

where `S(t)` is the number of sheets. Under repeated CNOTs, `S(t)` can grow with
the full unresolved CNOT history and can be Fibonacci-like for alternating
CNOT directions.

The goal here is to get closer to:

```text
O(B * M_block + live_extra)
```

where `live_extra` is controlled by current active histories, subscriptions, or
bounded lanes rather than by total past CNOT count.

## Shared Readout Convention

All plans below keep the current logical failure rule unless explicitly marked
as an experimental diagnostic:

```text
decoded_state(block) = physical_component(block) xor correction_component(block)
logical_failure(block) = !detect_logical_error(decoded_state(block))
logical_failure = any block logical_failure
```

Cleanup failure should continue to be recorded separately:

```text
cleanup_failure = unresolved history remains after cleanup
```

For analysis, also record an optional stricter diagnostic:

```text
logical_or_cleanup_failure = logical_failure || cleanup_failure
```

This diagnostic must not silently replace existing scan semantics.

## Plan 1: Conservative Quiescence-Compacted Sheet Copy

### 1. Motivation

This is the conservative plan. It directly addresses primitive CNOT's main
failure mode: destructive merging of active target and copied-control defect
history. It keeps the high-fidelity sheet-copy rule while reducing memory after
lineages become quiescent.

It mainly targets better memory for repeated CNOT runs while preserving
sheet-copy fidelity. Runtime improves only when compaction reduces the number
of active sheets being updated.

This plan does not satisfy the strongest overhead goal in adversarial or dense
CNOT schedules. If active histories do not drain, it degenerates to sheet-copy
with live sheet growth.

### 2. Core Idea

Represent each active propagated history as a full `DecoderSheet`, exactly like
the current sheet-copy prototype. When a sheet is quiescent, fold only its
decoded edge contribution into a compact block accumulator and free the dense
field buffers.

Active histories are never xor-merged. They are decoded independently and only
combined algebraically at readout or quiescent compaction.

### 3. Data Structures

```julia
mutable struct SettledBlockX
    decoded::BitArray{3}  # L x L x 2
end

mutable struct ActiveSheet
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

mutable struct CompactedSheetCopyState
    settled::Vector{SettledBlockX}      # length B
    sheets::Vector{ActiveSheet}         # live, full sheets
    next_lineage_id::Int
end
```

Dense fields:

```text
sheets[*].fields
sheets[*].new_fields
```

Compressed fields:

```text
settled[block].decoded
lineage metadata
```

Physical noise is attached to sheets in the behavior-preserving version because
`update_sheet!` calls the current `update!` for each sheet. A shared-noise
variant is described below, but it should be a separate mode because it changes
the stochastic model.

### 4. CNOT Update Rule

For `c -> t`:

```text
settled[t].decoded xor= settled[c].decoded

for each active sheet s with s.block == c:
    copied = deepcopy(s)
    copied.block = t
    copied.parent_lineage_id = s.lineage_id
    copied.lineage_id = fresh_lineage_id()
    copied.created_by_gate = gate_id
    push!(sheets, copied)
```

Raw state:

```text
control active sheets unchanged
target active sheets unchanged
copied.state_component = copy(parent.state_component)
```

Correction state:

```text
copied.state_correction = copy(parent.state_correction)
```

History and fields:

```text
copied.hist = copy(parent.hist)
copied.fields = copy(parent.fields)
copied.new_fields = copy(parent.new_fields)
copied.hist_correction = copy(parent.hist_correction)
```

Old/new syndromes:

```text
copied.old_synds = copy(parent.old_synds)
copied.new_synds = copy(parent.new_synds)
```

Lineage metadata records the parent and gate id. No active control information
is merged destructively into target histories or fields.

### 5. Local Decoding Update Rule

For each noisy or cleanup round:

```text
for sheet in sheets:
    update_sheet!(sheet, r, p, q, synch, pretty)

for sheet in sheets:
    if sheet_quiescent(sheet):
        settled[sheet.block].decoded xor=
            sheet.state_component xor sheet.state_correction
        remove sheet
```

The strict first quiescence condition should be:

```text
hist is empty
hist_correction is empty
fields is zero
new_fields is zero
get_synds(state_component xor state_correction) is zero
old_synds == new_synds
```

The current RG cycling, feedback, and correction application are unchanged
inside each active sheet. Future physical and measurement noise is independently
sampled per sheet in the behavior-preserving mode.

Shared-noise variant:

```text
primary block sheet receives physical and measurement noise
copied shadow sheets run feedback/RG with p = q = 0 after the gate
```

That variant is more physical but not behavior-preserving, so compare it as a
separate experimental mode.

### 6. Readout And Failure Rule

```julia
function decoded_state(plan, block, L)
    out = copy(plan.settled[block].decoded)
    for s in plan.sheets
        if s.block == block
            out .⊻= s.state_component
            out .⊻= s.state_correction
        end
    end
    return out
end
```

Cleanup failure:

```text
any active sheet has nonempty hist after cleanup.
```

Logical failure criteria are unchanged.

### 7. Classical Overhead Analysis

Leading memory:

```text
O(B * L^2) settled bits
+ O(A(t) * M_block) active sheets
```

If the original physical block sheets are counted as part of `A(t)`, this is:

```text
O(A(t) * M_block + B * L^2 bits).
```

If the implementation always keeps one primary sheet per block, write it as:

```text
O(B * M_block + A_shadow(t) * M_block + B * L^2 bits).
```

Runtime per decoder round:

```text
O((B + A_shadow(t)) * T_block_round)
```

or `O(A(t) * T_block_round)` if all live sheets are counted together.

CNOT-event cost:

```text
O(a_c(t) * M_block)
```

where `a_c(t)` is the number of active unresolved control sheets at the gate.

Worst-case repeated CNOT behavior:

```text
C1 -> C2 repeated:
    If C1 remains with one active lineage, C2 gains one copied lineage per gate.
    Memory O((B + D) * M_block) until lineages compact.

C1 -> C2, C2 -> C1 alternating:
    Active sheet counts follow the sheet-copy recurrence while no lineage
    compacts. Growth can be Fibonacci-like in D.

C1 -> C2, C1 -> C3 fanout:
    If C1 has one active lineage and never receives copies, total copied
    lineages grow O(D). If C1 itself has accumulated incoming lineages, fanout
    copies all of them and can amplify the current live width.
```

This plan has live-lineage scaling, not a hard cap. It avoids growth with total
past CNOT count only when old lineages quiesce and compact.

### 8. Expected Logical Performance

For a single CNOT it should behave closest to sheet-copy because active
lineages are full sheets. It should fix primitive-like target failure asymmetry
by not merging target and copied-control histories.

New failure modes:

```text
incorrect quiescence detection
settled accumulator readout bugs
optional shared-noise model changing current scan comparability
```

### 9. Known Weaknesses And Research Risks

The plan is exact as a representation of current sheet-copy only if compaction
is performed after the sheet is truly quiescent. It is not exact if a sheet is
compacted while stale fields, syndrome registers, or histories can still create
future corrections.

It should be rejected as the final overhead solution if repeated-CNOT tests with
zero or small idle gaps show peak active lineages growing like sheet-copy.

### 10. Implementation Difficulty

Difficulty: small to medium.

Likely code paths:

```text
DecoderSheet and helpers in 2d_windowed_cnot_sheetcopy.jl
sheet_active
apply_cnot_x_sheetcopy!
update_sheets!
merged_decoded_state
estimate_sheetcopy_cnot_Ft metrics
```

It can be implemented as a new mode without changing baseline behavior.

### 11. Validation Plan

Plan-specific deterministic checks:

```text
compacted readout equals explicit un-compacted sheet readout
compaction does not change decoded_state(control) or decoded_state(target)
quiescent compacted sheets have zero syndrome after merge/readout
no copied arrays alias mutable parent arrays
zero-noise CNOT succeeds
```

Noisy tests should compare directly to current sheet-copy at matched seeds or
matched aggregate parameters. Repeated-CNOT tests are decisive because this plan
is only useful if `A(t)` stays much smaller than total copied history.

### 12. Pseudocode

```julia
function cnot!(plan, c, t, gate_id)
    plan.settled[t].decoded .⊻= plan.settled[c].decoded
    parents = [s for s in plan.sheets if s.block == c && active_unsettled(s)]
    for parent in parents
        child = deepcopy(parent)
        child.block = t
        child.parent_lineage_id = parent.lineage_id
        child.lineage_id = fresh!(plan)
        child.created_by_gate = gate_id
        assert_no_mutable_alias(parent, child)
        push!(plan.sheets, child)
    end
end

function round!(plan, r, p, q)
    for s in plan.sheets
        update_sheet!(s, r, p, q, true, false)
    end
    compact_quiescent!(plan)
end
```

## Plan 2: Aggressive Correction-Forwarding Subscriptions

### 1. Motivation

This is the aggressive space-reduction plan. It addresses sheet-copy's main
classical overhead: copying dense `fields` and `new_fields` for every active
control lineage.

It aims for memory close to:

```text
O(B * M_block + S_sub(t) * L^2 Z bits + S_sub(t) * L^2 bits)
```

instead of `O((B + A(t)) * M_block)`. It targets memory first, with possible
runtime benefits because subscriptions do not run field updates.

### 2. Core Idea

At a CNOT, immediately apply the algebraic raw X-sector state transform to the
target, but do not copy the control decoder fields or merge control history into
target history. Instead, keep a subscription to the source's active pre-gate
history. When the source decoder later chooses corrections for that subscribed
history, forward the corresponding spatial correction edges to the target.

The target receives the source's future inferred corrections without owning a
full copied sheet.

This avoids destructive merging in the target. It is not an exact colored
history decoder unless the source can distinguish subscribed pre-gate defects
from later source defects during feedback.

### 3. Data Structures

```julia
mutable struct BlockDecoder
    state::BitArray{3}
    state_correction::BitArray{3}
    old_synds::BitArray{2}
    new_synds::BitArray{2}
    hist::BitArray{3}
    hist_correction::BitArray{4}
    fields::Array{FieldInt,5}
    new_fields::Array{FieldInt,5}
end

mutable struct CnotSubscription
    source_block::Int
    target_block::Int
    gate_id::Int
    lineage_id::Int
    hist_mask::BitArray{3}       # subscribed source defects
    active::Bool
end

mutable struct SubscriptionCnotState
    blocks::Vector{BlockDecoder}         # length B
    subscriptions::Vector{CnotSubscription}
    next_lineage_id::Int
end
```

Dense fields:

```text
blocks[b].fields
blocks[b].new_fields
```

Compressed fields:

```text
subscriptions[*].hist_mask
subscription metadata
```

Physical and measurement noise are attached only to physical blocks, not to
subscriptions. Subscriptions do not sample independent future noise.

If alternating CNOTs are supported, use GF(2) subscription algebra:

```text
block b owns a set of active source masks currently contributing to b.
CNOT c -> t toggles/copies every active source mask represented in c into t.
Identical (source, target, mask epoch) entries cancel mod 2.
```

The simplest two-block one-CNOT prototype can omit this algebra, but the
repeated-CNOT stress tests require it.

### 4. CNOT Update Rule

For `c -> t`:

```text
state_t            xor= state_c
state_correction_t xor= state_correction_c
old_synds_t        xor= old_synds_c
new_synds_t        xor= new_synds_c

push subscription:
    source_block = c
    target_block = t
    hist_mask = copy(hist_c)
```

Do not:

```text
hist_t xor= hist_c
fields_t = nonzeromin(fields_t, fields_c)
```

The target raw state receives the current source component immediately. Future
source corrections for defects in `hist_mask` are forwarded later.

For repeated CNOTs, a source block may contain contributions whose corrections
are owned by other source subscriptions. Then `c -> t` must also toggle those
active owner subscriptions into `t`. Otherwise alternating CNOTs are not
represented correctly.

### 5. Local Decoding Update Rule

Refactor the feedback step so it exposes selected correction links:

```julia
struct CorrectionLink
    i::Int
    j::Int
    k::Int
    axis::Int       # 1, 2, or 3
end
```

During a source block update:

```text
1. run ordinary field updates
2. choose ordinary hist_correction links
3. for each active subscription from this source:
       sub_corr = hist_correction restricted to sub.hist_mask sources
       if sub_corr link is spatial:
           state_correction[target_block] xor= corresponding edge
       perform_correction!(sub.hist_mask, sub_corr)
4. apply ordinary perform_correction! to source hist
5. apply block physical noise and measurement noise
6. advance source hist and every subscription mask consistently through RG
```

The exact restriction rule matters. A conservative implementation forwards only
links whose starting active defect was in `hist_mask` before feedback. Vertical
links update the mask but do not touch target physical corrections.

Future physical/measurement noise is shared by block: subscriptions receive no
new noise. RG cycling and feedback interact with subscriptions by evolving the
mask through the same correction and RG transforms as the source history.

### 6. Readout And Failure Rule

Readout is ordinary block readout:

```text
decoded_state(b) = blocks[b].state xor blocks[b].state_correction
```

There is no final sheet merge.

Cleanup failure:

```text
any block hist is nonempty, or any active subscription hist_mask is nonempty
after cleanup.
```

Logical failure criteria are unchanged.

### 7. Classical Overhead Analysis

Leading memory:

```text
O(B * M_block)
+ O(S_sub(t) * L^2 Z bits)
+ O(S_sub(t) * metadata)
```

If each subscription also stores a temporary correction mask:

```text
+ O(S_sub(t) * L^2 Z bits)
```

CNOT-event cost:

```text
O(L^2 + L^2 Z bits)
```

for state/correction/syndrome xors and a copied history mask. It avoids
`O(M_block)` dense field copies.

Runtime per decoder round:

```text
O(B * T_block_round)
+ O(total active subscription mask update work)
```

Dense first implementation:

```text
O(B * T_block_round + S_sub(t) * L^2 Z)
```

Sparse implementation:

```text
O(B * T_block_round + number_of_forwarded_correction_links
  + active_mask_frontier_updates)
```

Worst-case repeated CNOT behavior:

```text
C1 -> C2 repeated:
    S_sub(t) can grow O(D) until old masks drain, unless identical active
    source-target masks are coalesced mod 2.

C1 -> C2, C2 -> C1 alternating:
    Without GF(2) subscription algebra, this plan is wrong.
    With algebra and mask coalescing, dense field growth is avoided, but
    S_sub(t) can still grow O(D) in unresolved distinct mask epochs.
    It should not be Fibonacci in dense field memory.

C1 -> C2, C1 -> C3 fanout:
    One source mask may have multiple target subscriptions. Memory grows with
    active target subscriptions, O(S_sub), not with copied field buffers.
```

This plan has live-subscription scaling, not a hard cap. It avoids exponential
or Fibonacci-like dense-field growth if subscription references are coalesced,
but it can still accumulate many unresolved masks in a high-depth circuit.

### 8. Expected Logical Performance

It should improve over primitive because control history is not merged into the
target field landscape. It may approach sheet-copy only when the source
decoder's corrections for the subscribed pre-gate history match what a copied
sheet would have done.

It may fix primitive target/control asymmetry if target failures were mainly
caused by destructive target-side field merging.

New failure mode:

```text
masked pre-gate defects and unmasked post-gate defects interact inside the
source block, causing forwarded corrections to differ from sheet-copy.
```

### 9. Known Weaknesses And Research Risks

This plan is not exact as stated. The source block still has one scalar `hist`
and one scalar field landscape. If subscribed and unsubscribed defects
annihilate or redirect each other, the subscription mask follows an approximate
projection of the source correction, not an independently decoded copied
history.

Reject this plan if:

```text
target failures remain primitive-like
masked/unmasked collision rate is high near threshold
alternating CNOT tests disagree with zero-noise linear X-sector algebra
cleanup failures grow faster than sheet-copy
```

A stronger variant is a two-color source history per subscribed epoch, but that
moves toward the bounded-lane plan below.

### 10. Implementation Difficulty

Difficulty: large.

Likely code paths:

```text
update! feedback selection must be factored to expose correction links
perform_correction! needs mask-compatible variants
primitive_cnot_x_sector! logic becomes a new subscription CNOT mode
estimate CNOT drivers need repeated-CNOT schedules and subscription metrics
```

It can be implemented as a new mode without changing baseline behavior, but it
requires substantial refactoring to avoid duplicating update logic incorrectly.

### 11. Validation Plan

Plan-specific deterministic checks:

```text
zero-noise single CNOT succeeds
zero-noise repeated alternating CNOTs implement GF(2) linear algebra exactly
decoded_state is unchanged by inactive subscription deletion
subscription masks drain under ideal cleanup
spatial forwarded corrections match copied-sheet corrections when post-gate
    source noise is disabled
```

Noisy validation should record:

```text
subscription_count
masked_unmasked_collision_count
forwarded_spatial_correction_count
inactive_subscription_count
```

### 12. Pseudocode

```julia
function cnot!(plan, c, t, gate_id)
    plan.blocks[t].state .⊻= plan.blocks[c].state
    plan.blocks[t].state_correction .⊻= plan.blocks[c].state_correction
    plan.blocks[t].old_synds .⊻= plan.blocks[c].old_synds
    plan.blocks[t].new_synds .⊻= plan.blocks[c].new_synds

    sub = CnotSubscription(
        c, t, gate_id, fresh!(plan),
        copy(plan.blocks[c].hist),
        true,
    )
    push!(plan.subscriptions, sub)

    propagate_existing_subscription_refs!(plan, c, t, gate_id)
end

function update_source_block!(plan, b, r, p, q)
    links = choose_feedback_links!(plan.blocks[b], r)
    for sub in active_subscriptions_from(plan, b)
        sub_links = restrict_links_to_mask(links, sub.hist_mask)
        forward_spatial_links!(plan.blocks[sub.target_block], sub_links)
        perform_correction!(sub.hist_mask, sub_links)
    end
    finish_block_update!(plan.blocks[b], links, p, q)
    rg_cycle_subscription_masks!(plan, b)
    drop_empty_subscriptions!(plan)
end
```

## Plan 3: Aggressive Shadow Histories With Scratch Fields

### 1. Motivation

This plan targets the dominant space cost directly: copied lineages need
history and correction state, but they may not need persistent dense field
buffers. It addresses sheet-copy's memory blowup while retaining more of
sheet-copy's independent lineage behavior than subscriptions.

It is mainly for memory. Runtime can increase because fields are recomputed for
shadow histories.

### 2. Core Idea

Keep one full baseline decoder per physical block. Propagated CNOT lineages are
stored as shadow histories with state and correction bits but without persistent
`fields` or `new_fields`. During each update, decode each shadow using a shared
scratch field workspace, then discard the scratch fields.

This avoids destructive merging because each shadow has its own `hist`,
`state_component`, `state_correction`, syndrome registers, and metadata. It
does not keep the long-lived message field state of sheet-copy, so it is an
approximation.

### 3. Data Structures

```julia
mutable struct PrimaryBlock
    sheet::ActiveSheet       # full M_block, receives future physical noise
end

mutable struct ShadowLineage
    block::Int
    lineage_id::Int
    parent_lineage_id::Union{Int,Nothing}
    created_by_gate::Union{Int,Nothing}
    hist::BitArray{3}
    hist_correction::BitArray{4}
    state_component::BitArray{3}
    state_correction::BitArray{3}
    old_synds::BitArray{2}
    new_synds::BitArray{2}
end

mutable struct ScratchFieldWorkspace
    fields::Array{FieldInt,5}
    new_fields::Array{FieldInt,5}
end
```

Dense fields:

```text
primary blocks: B full field pairs
scratch workspace: W field pairs per worker, usually W = 1
```

Compressed fields:

```text
shadow hist, hist_correction, state, correction, syndromes
```

Physical noise is attached to primary physical blocks. Shadow lineages normally
do not receive future physical or measurement noise. To reproduce current
sheet-copy exactly is impossible without giving shadows their own noise and
persistent fields.

### 4. CNOT Update Rule

For `c -> t`:

```text
settled[t] xor= settled[c]

copy primary/control active component into a new target ShadowLineage
copy every active control shadow into a new target ShadowLineage
```

Copied data:

```text
state_component
state_correction
hist
hist_correction
old_synds
new_synds
lineage metadata
```

Not copied:

```text
fields
new_fields
```

No active target history is merged. Control lineages remain unchanged.

### 5. Local Decoding Update Rule

Primary blocks:

```text
ordinary update! with p, q
```

Shadow lineages:

```text
scratch.fields .= 0
scratch.new_fields .= 0
anyons_source_fields!(shadow.hist, scratch.fields)
repeat r_warm times:
    update_2d_windowed_fields!(scratch.fields, scratch.new_fields, shadow.hist)
perform the ordinary feedback step using scratch.fields
perform_correction!(shadow.hist, shadow.hist_correction)
advance old/new syndromes consistently
rg_cycle!(shadow.hist, scratch.fields)
shadow.hist[:,:,1] = old_synds xor new_synds
discard scratch fields
```

Recommended first setting:

```text
r_warm = max(r, Z)
```

RG cycling and correction application should match the baseline order. The key
difference is that shadow messages do not persist across rounds.

### 6. Readout And Failure Rule

```text
decoded_state(block) =
    settled[block]
    xor primary[block].state_component
    xor primary[block].state_correction
    xor every shadow on block:
        shadow.state_component xor shadow.state_correction
```

Cleanup failure:

```text
any primary hist or shadow hist remains nonempty after cleanup.
```

Logical failure criteria are unchanged.

### 7. Classical Overhead Analysis

Leading memory:

```text
O(B * M_block)
+ O(A_shadow(t) * L^2 Z bits)
+ O(A_shadow(t) * L^2 bits)
+ O(W * M_field_pair)
```

where:

```text
M_field_pair = 12 L^2 Z sizeof(FieldInt)
W = number of scratch workspaces per worker, usually 1
```

This is much smaller than `A_shadow(t) * M_block` because shadow lineages do not
store dense persistent fields.

Runtime per decoder round:

```text
O(B * T_block_round)
+ O(A_shadow(t) * r_warm * L^2 Z)
```

CNOT-event cost:

```text
O(a_c(t) * L^2 Z bits + a_c(t) * L^2 bits)
```

Worst-case repeated CNOT behavior:

```text
C1 -> C2 repeated:
    Shadow count can grow O(D) until shadows compact, but dense field memory
    stays O(B + W), not O(D).

C1 -> C2, C2 -> C1 alternating:
    Shadow count can still be Fibonacci-like if every active shadow is copied
    and none compact. The Fibonacci growth is in packed histories, not dense
    field buffers. Runtime can still become too large.

C1 -> C2, C1 -> C3 fanout:
    Shadow histories grow with active fanout width. Dense memory stays bounded
    by primary blocks plus scratch fields.
```

This plan does not have a hard cap. It meets the "avoid dense sheet-copy
growth" goal, but not the strongest "no unbounded live lineage growth" goal.

### 8. Expected Logical Performance

It should be between sheet-copy and primitive. It keeps independent histories
and avoids target merging, so target failures should improve relative to
primitive. It may underperform sheet-copy because persistent fields carry
message inertia that recomputed scratch fields discard.

New failure mode:

```text
shadow histories may move differently from full sheet-copy because fields are
rebuilt from current hist rather than evolved through RG time.
```

### 9. Known Weaknesses And Research Risks

This is not exact. If the field buffers contain useful long-range or back-wall
information not recoverable from the current shadow history in `r_warm` sweeps,
logical performance may become primitive-like.

Reject it if target failures or cleanup failures move close to primitive in the
threshold-trend tests.

### 10. Implementation Difficulty

Difficulty: medium to large.

Likely code paths:

```text
factor update! into field update, feedback, noise/syndrome insertion, RG cycle
new shadow update routine using scratch fields
sheet-copy CNOT driver as the starting point
metrics for scratch field counts and shadow counts
```

It can be implemented as a new mode without changing baseline behavior.

### 11. Validation Plan

Plan-specific deterministic checks:

```text
zero-noise CNOT succeeds
decoded_state is unchanged by shadow compaction
scratch fields are not aliased across shadows
quiescent shadows have zero syndrome after merge/readout
FieldInt never saturates if compressed fields are used
```

The key noisy diagnostic is whether `r_warm` can be small enough to save space
without losing sheet-copy-like fidelity.

### 12. Pseudocode

```julia
function update_shadow!(shadow, scratch, r_warm)
    scratch.fields .= 0
    scratch.new_fields .= 0
    anyons_source_fields!(shadow.hist, scratch.fields)
    for _ in 1:r_warm
        update_2d_windowed_fields!(
            scratch.fields, scratch.new_fields, shadow.hist)
    end
    choose_feedback_using_fields!(
        shadow.hist, shadow.hist_correction, shadow.state_correction,
        scratch.fields)
    perform_correction!(shadow.hist, shadow.hist_correction)
    shadow.old_synds .= shadow.new_synds
    shadow.new_synds .= get_synds(shadow.state_component)
    rg_cycle!(shadow.hist, scratch.fields)
    shadow.hist[:,:,1] .= shadow.old_synds .⊻ shadow.new_synds
end
```

## Plan 4: Bounded Labeled Lanes With Cap K

### 1. Motivation

This is the bounded-overhead plan. It addresses the main failure of sheet-copy,
compacted sheet-copy, and scratch shadows: none has a hard cap under dense
alternating CNOTs.

It targets predictable memory and runtime first. Fidelity depends on whether
`K` is large enough for the live lineage width and on the overflow policy.

### 2. Core Idea

Each physical block has at most `K` active X-sector lineage lanes. A lane is a
labelled copy of the relevant active history. CNOT copies active control lanes
into free target lanes. If no lane is free, the decoder uses an explicit
overflow policy.

No active defect history is destructively merged unless the chosen overflow
policy is a lossy merge, and lossy merges are counted.

### 3. Data Structures

Full-field bounded lane version:

```julia
mutable struct LaneMeta
    active::Bool
    lineage_id::Int
    parent_lineage_id::Union{Int,Nothing}
    created_by_gate::Union{Int,Nothing}
    primary::Bool
end

mutable struct LaneBlockFull
    settled::BitArray{3}                 # L x L x 2
    meta::Vector{LaneMeta}               # length K
    hist::BitArray{4}                    # L x L x Z x K
    hist_correction::BitArray{5}         # L x L x Z x 3 x K
    state_component::BitArray{4}         # L x L x 2 x K
    state_correction::BitArray{4}        # L x L x 2 x K
    old_synds::BitArray{3}               # L x L x K
    new_synds::BitArray{3}               # L x L x K
    fields::Array{FieldInt,6}            # L x L x Z x 3 x 2 x K
    new_fields::Array{FieldInt,6}
end
```

Space-reduced bounded lane version:

```julia
mutable struct LaneBlockCompressed
    settled::BitArray{3}
    meta::Vector{LaneMeta}               # length K
    lane_histories_and_states            # packed Bool arrays as above
    field_dist::Array{FieldInt,6}        # L x L x Z x 3 x 2 x m
    field_lane::Array{UInt16,6}          # matching lane labels
    scratch::ScratchFieldWorkspace
end
```

Dense fields:

```text
full-field: K field pairs per block
compressed: m labeled field slots per direction plus scratch
```

Compressed fields:

```text
lane histories, states, corrections, metadata
settled decoded accumulator
```

Physical noise should be attached to the physical block primary lane. Shadow
lanes receive no independent future noise in the shared-noise mode.

### 4. CNOT Update Rule

For `c -> t`:

```text
target.settled xor= control.settled

for each active lane ell in control:
    ell2 = allocate_free_lane(target)
    if no free lane:
        compact quiescent lanes
    if still no free lane:
        apply overflow policy
    else:
        target.lane[ell2] = deep copy of control.lane[ell]
        target.meta[ell2].parent_lineage_id = control.meta[ell].lineage_id
        target.meta[ell2].lineage_id = fresh_lineage_id()
        target.meta[ell2].created_by_gate = gate_id
```

Raw state, correction state, history, old/new syndromes, and metadata are
copied lane-to-lane. In full-field mode, fields are copied too. In compressed
mode, fields are reconstructed or relabeled.

Overflow policies:

```text
reject_gate:
    stop the trial and count lane_overflow_count.

flush_then_gate:
    run ideal cleanup up to tau_flush, compact quiescent lanes, retry.

oldest_primitive_merge:
    merge two oldest lanes with xor and nonzeromin; increment lossy_merge_count.

lowest_activity_merge:
    merge two lanes with fewest active defects; increment lossy_merge_count.
```

Only the first two policies preserve a non-lossy interpretation. The merge
policies are explicit approximations.

### 5. Local Decoding Update Rule

Full-field mode:

```text
for each active lane:
    run baseline update! on that lane's arrays
    compact lane if quiescent
```

Compressed top-`m` labelled field mode:

```text
1. For each field entry, collect candidate messages from neighboring entries.
2. Keep the smallest nonzero distance per lane label.
3. Retain only the smallest m lane-distance pairs.
4. A defect in lane ell may only follow a message whose label is ell.
5. If no same-lane message is present, the defect does not move this round.
```

RG cycling, feedback, and correction application are lane-local. Spatial
corrections affect only `state_correction[:,:,:,ell]` until readout.

### 6. Readout And Failure Rule

```text
decoded_state(block) =
    settled
    xor over active lanes:
        state_component[:,:,:,ell] xor state_correction[:,:,:,ell]
```

Cleanup failure:

```text
any active lane hist remains nonempty after cleanup
or lane_overflow_count > 0 if reject_gate is treated as cleanup/gate failure.
```

Logical failure criteria are unchanged. If lossy merge policies are enabled,
record logical failures separately by `lossy_merge_count`.

### 7. Classical Overhead Analysis

Full-field lanes:

```text
memory = O(B * K * M_block)
runtime per round = O(B * K * T_block_round)
CNOT cost = O(active_control_lanes * M_block)
```

This has a hard cap but can be too expensive if `K` is large.

Compressed labelled lanes:

```text
memory =
    O(B * M_primary)
  + O(B * K * (L^2 Z bits + L^2 bits))
  + O(B * m * L^2 Z * (sizeof(FieldInt) + sizeof(label)))
```

For fixed `K` and `m`, this is:

```text
O(B * M_block + B * K * L^2 Z bits)
```

with a hard cap.

Runtime per round:

```text
full-field: O(B * active_lanes * T_block_round)
top-m:      O(B * L^2 Z * local_candidate_work * m)
            + O(active_defects)
```

CNOT-event cost:

```text
full-field: O(active_control_lanes * M_block)
compressed: O(active_control_lanes * L^2 Z bits)
```

Worst-case repeated CNOT behavior:

```text
C1 -> C2 repeated:
    Active target lanes fill up to K. After that, overflow policy determines
    behavior. Memory is capped at O(B*K).

C1 -> C2, C2 -> C1 alternating:
    No uncontrolled growth. Lanes fill rapidly; overflow count measures whether
    K is sufficient. With reject_gate or flush_then_gate, no silent lossy merge.

C1 -> C2, C1 -> C3 fanout:
    Targets fill independently up to K. Control width determines allocation
    pressure, but memory remains capped.
```

This is the only plan here with a hard cap independent of `D`.

### 8. Expected Logical Performance

If `K >= max_t A_block(t)` and full fields are used, this should match
compacted sheet-copy up to implementation details. With compressed top-`m`
fields, it should interpolate between sheet-copy and primitive depending on
how often local field entries need more than `m` labels.

It should fix primitive target/control asymmetry when no overflow or lossy
merge occurs.

New failure modes:

```text
lane overflow
lossy merge changing homology class
top-m field label eviction suppressing a needed same-lane message
```

### 9. Known Weaknesses And Research Risks

`K` is a scheduling assumption. If real circuits require lineage width greater
than `K`, the decoder must either reject/flush gates or become approximate.

Full-field lanes may meet the hard-cap requirement while failing practical
memory targets. Top-`m` fields are approximate and need rejection criteria:

```text
high lane_overflow_count
high lossy_merge_count
target failures primitive-like
cleanup failures increasing with D
```

### 10. Implementation Difficulty

Difficulty: large.

Likely code paths:

```text
new lane-based CNOT mode, probably copied from sheet-copy driver
lane allocation and compaction helpers
update! factorization or lane wrappers
field representation if top-m labelled fields are used
repeated-CNOT scheduler and metrics
```

It can be implemented as a new mode without changing baseline behavior, but it
is a significant new representation.

### 11. Validation Plan

Plan-specific deterministic checks:

```text
zero-noise CNOT succeeds for K >= required live width
K overflow is reported, never silent
decoded_state is unchanged by lane compaction
no lane arrays alias parent lanes after CNOT copy
quiescent lanes have zero syndrome after merge/readout
FieldInt and lane labels never saturate
top-m with m >= K matches full-field lanes on deterministic cases
```

Repeated-CNOT stress tests should be run first with `reject_gate` to measure
the required `K`, then with `flush_then_gate`, and only then with lossy merge
policies if needed.

### 12. Pseudocode

```julia
function cnot!(plan, c, t, gate_id)
    block_t(plan, t).settled .⊻= block_t(plan, c).settled
    for ell in active_lanes(block_t(plan, c))
        ell2 = allocate_lane!(block_t(plan, t))
        if ell2 === nothing
            compact_quiescent_lanes!(block_t(plan, t))
            ell2 = allocate_lane!(block_t(plan, t))
        end
        if ell2 === nothing
            handle_overflow!(plan, c, t, gate_id)
        else
            copy_lane!(block_t(plan, t), ell2, block_t(plan, c), ell)
            set_parent_metadata!(block_t(plan, t), ell2, ell, gate_id)
        end
    end
end

function round!(plan, r, p, q)
    for b in 1:B
        for ell in active_lanes(block_t(plan, b))
            update_lane!(block_t(plan, b), ell, r, p_for_lane(ell), q_for_lane(ell))
        end
        compact_quiescent_lanes!(block_t(plan, b))
    end
end
```

## Plan 5: Orthogonal FieldInt Compression

### 1. Motivation

This is not a CNOT representation by itself. It reduces the dominant dense
field memory for every plan that keeps field buffers.

It is for memory, with little or no intended logical-performance change.

### 2. Core Idea

Replace `Array{Int,5}` field buffers with a narrower field integer type:

```julia
const FieldInt = UInt16
```

Use a wider temporary for additions:

```julia
candidate = UInt32(field) + UInt32(dist)
```

Keep `0` as "no message". Saturation must be asserted and counted.

### 3. Data Structures

```julia
const FieldInt = UInt16

fields::Array{FieldInt,5}
new_fields::Array{FieldInt,5}
```

The dominant memory changes from:

```text
12 L^2 Z * 8 bytes
```

to:

```text
12 L^2 Z * 2 bytes
```

on current 64-bit Julia builds.

### 4. CNOT Update Rule

No algorithmic change. `nonzeromin` must preserve `FieldInt` and the same
zero-sentinel semantics.

### 5. Local Decoding Update Rule

No intended logical change. Field update uses widened temporary arithmetic and
converts back to `FieldInt` after checking:

```text
candidate <= typemax(FieldInt)
```

### 6. Readout And Failure Rule

Unchanged.

### 7. Classical Overhead Analysis

For any plan with `N_field_pairs(t)` full field pairs:

```text
field_memory_Int64  = N_field_pairs(t) * 96 L^2 Z bytes
field_memory_UInt16 = N_field_pairs(t) * 24 L^2 Z bytes
```

This is a constant-factor reduction, not a fix for unbounded lineage growth.

### 8. Expected Logical Performance

Identical if no saturation occurs.

### 9. Known Weaknesses And Research Risks

If messages can exceed `typemax(UInt16)`, saturation silently changes the
decoder unless it is trapped. The first implementation should assert no
saturation in all validation runs.

### 10. Implementation Difficulty

Difficulty: small to medium.

Likely code paths:

```text
field allocations in all Julia files
onesite_field_update
update_2d_windowed_fields!
nonzeromin
debug serialization of fields
```

It should be gated by a constant or mode so baseline comparisons remain clear.

### 11. Validation Plan

Run all deterministic and noisy tests with:

```text
FieldInt = Int
FieldInt = UInt16
```

and compare aggregate outputs. Required assertion:

```text
FieldInt never saturates
```

### 12. Pseudocode

```julia
const FieldInt = UInt16

function checked_field(x::UInt32)
    @assert x <= typemax(FieldInt)
    return FieldInt(x)
end

function nonzeromin(a::FieldInt, b::FieldInt)
    a == 0 && return b
    b == 0 && return a
    return min(a, b)
end
```

## Common Validation Plan

All new modes should run the same validation grid against primitive and current
sheet-copy. Use separate output directories and record the mode name, CNOT
schedule, `K`, `m`, `r_warm`, overflow policy, field type, and shared-noise
flag.

### Deterministic Sanity Checks

Required:

```text
zero-noise CNOT succeeds
decoded_state is unchanged by compaction or representation changes
no copied arrays alias mutable parent arrays
quiescent histories have zero syndrome after merge/readout
FieldInt never saturates, if compressed fields are used
```

Additional checks:

```text
single-CNOT zero-noise truth table for X-sector block parities
alternating zero-noise CNOTs match GF(2) linear algebra
settled/shadow/lane/subscription deletion does not change decoded_state
cleanup failure is recorded separately from logical failure
```

### Noisy Smoke Tests

```text
L in {5, 7}
p in {0.005, 0.010, 0.015}
qrat = 1
T = L
cleanup = 2T
fixed_samps = 200 to 1000
```

### Threshold-Trend Tests

```text
L in {5, 9, 13, 19}
p in {0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017}
qrat = 1
r = 3
synch = true
T = L
cleanup = 2T
ACC_ERRORS = 1000
```

Use the current split timing unless a schedule test explicitly varies it:

```text
T_PRE = floor(T/2)
T_POST = T - T_PRE
```

### Repeated-CNOT Stress Tests

```text
depth D in {1, 2, 4, 8, 16}
patterns:
    C1 -> C2 repeated
    C1 -> C2, C2 -> C1 alternating
    C1 -> C2, C1 -> C3 fanout
idle rounds between gates in {0, floor(L/4), floor(L/2), L}
```

### Required Metrics

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
subscription_count or lane_overflow_count
lossy_merge_count, if applicable
estimated_peak_bytes
peak_field_buffer_count
runtime_seconds
```

Plan-specific metrics:

```text
Plan 1:
    peak_active_sheets
    compacted_sheet_count
    settled_nonzero_count

Plan 2:
    peak_subscription_count
    masked_unmasked_collision_count
    forwarded_spatial_correction_count
    canceled_subscription_count

Plan 3:
    peak_shadow_count
    peak_field_buffer_count
    r_warm
    scratch_rebuild_count

Plan 4:
    K
    m, if top-m fields are used
    lane_overflow_count
    flush_round_count
    lossy_merge_count
    field_label_eviction_count

Plan 5:
    field_type
    field_saturation_count
```

Estimated peak bytes should use a simple accounting model:

```text
estimated_peak_bytes =
    full_field_pair_count * 12 L^2 Z sizeof(FieldInt)
  + full_sheet_bool_bits / 8
  + shadow_or_subscription_bits / 8
  + metadata_bytes
```

## Decision Criteria

Rank plans by:

```text
logical performance near sheet-copy
target failures not primitive-like
cleanup failures controlled
peak memory below sheet-copy in repeated-CNOT tests
no uncontrolled growth under alternating CNOTs
runtime scaling as predicted
minimal silent changes to baseline decoder behavior
```

Reject or downgrade a plan if:

```text
target_logical_failures remain close to primitive
cleanup_failures grow with D faster than sheet-copy
estimated_peak_bytes follows sheet-copy in alternating CNOT tests
lossy_merge_count is nonzero in nominal runs
lane_overflow_count is high for intended K
FieldInt saturation occurs
runtime_seconds does not match the predicted scaling
```

## Recommendation Table

| Plan | Role | Memory scaling | Hard cap? | Expected fidelity | Main risk | Difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Quiescence-compacted sheet copy | Conservative first implementation | `O(B*M_block + A_shadow(t)*M_block)` | No | Closest to sheet-copy | Alternating dense CNOTs can still grow like sheet-copy until histories drain | Small/medium |
| 2. Correction-forwarding subscriptions | Aggressive space reduction | `O(B*M_block + S_sub(t)*L^2 Z bits)` | No | Between primitive and sheet-copy, uncertain | Not exact when subscribed and unsubscribed source defects interact | Large |
| 3. Shadow histories with scratch fields | Aggressive practical memory reduction | `O(B*M_block + A_shadow(t)*bits + W*M_field)` | No | Between sheet-copy and primitive | Rebuilt fields may lose useful message history | Medium/large |
| 4. Bounded labelled lanes | Bounded-overhead plan | Full: `O(B*K*M_block)`, compressed: `O(B*M_block + B*K*bits + B*m*fields)` | Yes | Sheet-copy if no overflow and full fields | Choosing `K`, overflow policy, top-`m` approximation | Large |
| 5. FieldInt compression | Orthogonal constant-factor reduction | Multiplies field memory by about `1/4` for `UInt16` | No | Identical if no saturation | Silent saturation | Small/medium |

## Critical Takeaway

The compacted sheet-copy/shared-noise direction is the right first engineering
step, but it does not meet the strongest overhead goal. Its memory scales with
live unresolved full sheets:

```text
O(B * M_block + A_shadow(t) * M_block)
```

If a circuit keeps creating CNOTs before histories quiesce, especially under
`C1 -> C2, C2 -> C1` alternation, `A_shadow(t)` can still grow like the
uncompacted sheet-copy recurrence. Shared noise fixes a physical-model issue
and may improve fidelity, but it does not by itself bound lineage memory.

The stronger alternatives are:

```text
subscriptions, which replace copied dense fields with active masks, and
bounded lanes, which impose an explicit cap K and make overflow visible.
```

The bounded-lane plan is the only proposal here with a hard cap independent of
`D`. The subscription and scratch-shadow plans aim for much lower live extra
memory but still need stress tests to prove they avoid hidden growth.

## Implementation Order

1. Add memory accounting and repeated-CNOT stress-test infrastructure to the
   current sheet-copy driver before changing algorithms.
2. Implement Plan 1 as `sheetcopy_compact` with the current independent
   per-sheet noise model.
3. Add Plan 5 `FieldInt` compression behind a mode or constant and verify no
   saturation.
4. Add Plan 1 shared-noise as a separate mode, not as a silent replacement for
   sheet-copy.
5. Prototype Plan 3 scratch shadows if full-sheet active memory is too high in
   repeated-CNOT tests but fidelity remains sheet-copy-like.
6. Prototype Plan 2 subscriptions only after update feedback can expose
   correction links cleanly.
7. Implement Plan 4 bounded lanes once stress tests quantify the required
   live-lineage width and a target `K` can be justified.

## What To Implement First

Implement `sheetcopy_compact` plus explicit memory metrics first. It is the
smallest change that preserves sheet-copy lineage separation and will provide
the measurement baseline needed to judge every more aggressive plan. The first
pass should not change noise semantics; shared-noise should be a separate mode
after compaction is validated.
