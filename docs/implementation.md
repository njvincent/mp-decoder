# mp-decoder Implementation Documentation

Last updated: 2026-07-17

This document describes the implemented Julia behavior, with emphasis on state
ownership, update order, and classical overhead. The Markdown paper notes
explain the physical motivation; the Julia source remains authoritative for
the actual simulator.

## 1. Orientation

### 1.1 Implementation hierarchy

The repository has two decoder-code lineages:

1. Legacy baseline kernel

   1. Serial memory driver: `2d_windowed_simulation.jl`

      - Reference one-block `update!` and baseline Monte Carlo modes.

   2. Trial-parallel memory driver: `2d_windowed_simulation_thread.jl`

      - Decoder helpers from `circle_distance` through `get_decoding_time` are
        byte-identical to the serial file.
      - Parallelism is across independent trials, not within a decoder round.

   3. CNOT extensions that retain the same legacy `update!`

      1. Primitive merge: `2d_windowed_cnot_primitive.jl`

         - Keeps two baseline decoder states and merges control state directly
           into the target at the gate.

      2. Legacy sheet-copy: `2d_windowed_cnot_sheetcopy.jl`

         - Keeps one complete baseline decoder state per lineage sheet and
           calls `update!` once per sheet.

      3. Physical snapshot: `2d_windowed_cnot_snapshot.jl`

         - Keeps exactly one physical error/syndrome channel per observable
           block and separate pre-/post-gate decoder histories.
         - Routes recovery contributions with `applies_to::BitVector` rather
           than deep-copying physical sheets.

      4. Two-pass causal junction: `2d_windowed_cnot_twopass.jl`

         - Keeps two physical blocks, a continuous control decoder, and three
           labeled target-output history streams connected at the CNOT.
         - Uses deterministic two-stage direction selection/routing and a
           primitive history only for finite back-wall retirement.
         - Owns the required synchronous baseline kernel copied from the
           primitive lineage; it imports no other CNOT driver.

      5. Moving Y-junction: `2d_windowed_cnot_yjunction.jl`

         - Keeps two pre-gate target branches only above a moving CNOT
           interface and one unlabeled post-gate target lane below it.
         - Propagates messages bidirectionally across the branched local graph,
           while every defect remains in exactly one lane.
         - XOR-collapses the pre-gate branches into the ordinary target back
           wall after one buffer depth, returning to two decoder lanes.

2. Block-level fork

   1. Block CNOT driver: `2d_windowed_cnot_block.jl`

      - Keeps one observable decoder per physical block.
      - Preserves the synchronous feedback order but intentionally changes the
        asynchronous physical-round schedule.

   2. Block regression suite: `test/runtests.jl`

      - Includes only the import-guarded block driver.
      - Tests CNOT algebra, observable noise/measurement channels, ancestry
        invariance, combined decoding, and cleanup.

The CNOT drivers are separate algorithms, not selectable front ends for one
shared implementation. They also duplicate the baseline non-CNOT modes, so a
change in one file does not propagate automatically to the others.

## 2. Common Model, Notation, and Cost Unit

### 2.1 Physical and syndrome arrays

All implementations track one toric-code error sector. The CNOT prototypes
implement only the X-sector propagation

~~~text
control_out = control
target_out  = target xor control
~~~

They use an ideal instantaneous gate with no gate-fault channel. They do not
implement the complementary Z-sector rule, circuit-level syndrome extraction,
hook errors, or a full fault-tolerant logical CNOT.

An edge configuration has shape

~~~text
L x L x 2
~~~

where orientation 1 is the edge from `(i,j)` toward `(i+1,j)`, and
orientation 2 is the edge toward `(i,j+1)`. Main simulation paths use periodic
spatial boundaries. The syndrome is

~~~text
synds[i,j] =
    state[i,j,1] xor state[i,j,2] xor
    state[i-1,j,1] xor state[i,j-1,2]
~~~

with spatial indices wrapped by `mod1`.

The physical-error array and accumulated recovery array are kept separate.
The baseline names them `state` and `state_correction`; the block model
names them `errors` and `pauli_frame`. Decoder feedback changes the
recovery array and the buffered history, not the physical-error array. Readout
uses their XOR.

### 2.2 Buffer and scan parameters

`hist[i,j,k]` is a rolling buffer of syndrome-change events, not the current
syndrome:

~~~text
new front event = old_synds xor new_synds
~~~

The coordinate `k=1:Z` is a decoder buffer/RG coordinate. It is not physical
time and is not Pauli Z. The default depth is

~~~text
Z = ceil(Int, log(1.5, L))   when LOGZ=true
Z = ceil(Int, L/4)           when LOGZ=false
~~~

Common defaults are `r=3`, `q=p`, `SYNCH=true`, and `LOGZ=true`.

### 2.3 Cost notation

Let

~~~text
N = L^2
w = sizeof(Int)
~~~

and define one baseline block working state as

~~~text
M_block =
    12 N Z machine Ints
    + (6 N + 4 N Z) packed Boolean bits
    + Julia array/container overhead
~~~

The two field buffers supply the `12NZ` integer words and dominate memory.
On a 64-bit build, those buffers alone use `96NZ` bytes. Packed
`BitArray` storage is rounded to machine-word boundaries, so the bit formula
is a content count rather than an exact `Base.summarysize` result.

One synchronous block round, or one legacy asynchronous block round in
expectation, costs

~~~text
U_block = Theta((r + 1) N Z)
~~~

because field propagation and feedback both scan `Theta(NZ)` sites. The
current `onesite_field_update` also creates a fresh `3 x 2` integer array
for every updated buffer site, producing `Theta(rNZ)` small transient
allocations per global field-sweep sequence.

These units describe the live decoder working set. History/demo modes store
whole time series and add `Theta(TNZ)` output storage. Threaded execution
multiplies trial-local working state by the number of active workers.

## 3. Baseline Memory Decoder

This section is canonical for the serial baseline, threaded baseline,
primitive block arrays, and legacy sheet-copy sheets. The block file reuses
the same message and feedback rules but changes physical-round acquisition as
described in Section 4.4.2.

### 3.1 Per-block state

| Array | Shape and element type | Meaning |
| --- | --- | --- |
| `state` | `L x L x 2 BitArray` | Accumulated physical X errors. |
| `state_correction` | `L x L x 2 BitArray` | Accumulated spatial recovery chain/Pauli frame. |
| `old_synds` | `L x L BitArray` | Previous measured syndrome register. |
| `new_synds` | `L x L BitArray` | Latest measured syndrome register. |
| `hist` | `L x L x Z BitArray` | Buffered syndrome-change defects. |
| `hist_correction` | `L x L x Z x 3 BitArray` | Proposed recovery links in x, y, and buffer directions. |
| `fields` | `L x L x Z x 3 x 2 Array{Int}` | Current six directed distance messages. |
| `new_fields` | same as `fields` | Jacobi/synchronous field-update buffer and asynchronous column buffer. |

For `fields[i,j,k,a,s]`, `a=1,2,3` denotes x, y, and buffer directions.
Zero means “no message,” not distance zero. In the feedback convention:

| Field component | Selected move |
| --- | --- |
| `[1,1]`, `[1,2]` | `-x`, `+x` |
| `[2,1]`, `[2,2]` | `-y`, `+y` |
| `[3,2]` | toward larger `k` |

`nonzeromin(a,b)` is therefore required whenever two message values are
compared componentwise and one may be absent.

### 3.2 Update rules

#### 3.2.1 Message update

`onesite_field_update(i,j,k,fields,hist)` computes six outgoing messages.
For each axis and direction, it examines the nine sites in the neighboring
`3 x 3` plane one step away along that axis. The candidate value is the
minimum of:

- the 1-norm distance to an active history event; and
- a nonzero incoming message plus that distance.

If no source or incoming message exists, the result is zero. Spatial indices
are periodic; buffer-neighbor indices are clamped to `1:Z`.

`update_2d_windowed_fields!` computes every site into `new_fields` before
copying to `fields`. The asynchronous helper updates only one spatial
processor column `(i,j,:)`.

#### 3.2.2 Synchronous round

For `synch=true`, `update!` performs this ordered sequence:

1. Run `r` global message sweeps. With `pretty=true`, run `r-1` now and
   reserve one sweep for the end of the round.
2. Clear `hist_correction`.
3. For every active `hist[i,j,k]`, select the smallest positive local
   message. In the bulk, ties are resolved in this order:

   ~~~text
   +buffer, -x, -y, +y, +x
   ~~~

4. On the back wall `k=Z`, permit only spatial moves. A selected defect moves
   with probability `0.8`; the tie order is:

   ~~~text
   -x, -y, +y, +x
   ~~~

5. XOR the parity of all spatial `hist_correction` links in a buffer column
   into `state_correction`.
6. Apply every proposed link to `hist`. A spatial link toggles its two
   spatial endpoints; a buffer link toggles `k` and `k+1`.
7. Toggle every physical edge independently with probability `p`.
8. Set `old_synds = new_synds`, calculate `get_synds(state)`, and XOR an
   independent Bernoulli(`q`) measurement mask into `new_synds`.
9. Run the RG cycle:

   - XOR slice `Z-1` into the back-wall history;
   - shift intermediate history toward larger `k`;
   - clear history slice 1;
   - merge the old and incoming back-wall spatial fields with
     `nonzeromin`;
   - shift intermediate fields and clear field slice 1.

10. Insert `old_synds xor new_synds` into `hist[:,:,1]`.
11. With `pretty=true`, seed source-adjacent fields and run the reserved
    message sweep.

#### 3.2.3 Legacy asynchronous round

The serial baseline, threaded baseline, primitive, and sheet-copy files use the
same legacy asynchronous branch. It runs `(r+1)N` random-column microsteps.
Each microstep is:

- a field-column update with probability `r/(r+1)`; or
- a feedback/noise/syndrome/RG-column update with probability `1/(r+1)`.

Spatial corrections are applied immediately. Buffer-direction proposals use a
temporary `Z`-bit vector and are then applied to the selected column.

The physical stochastic meaning differs from the synchronous branch:

- a feedback microstep toggles one uniformly random edge with probability
  `p`, rather than sampling every edge once;
- only the selected syndrome site is refreshed with measurement probability
  `q`;
- the selected history/field column is cycled, so columns may be updated
  repeatedly or not at all during one outer call.

Consequently, `p` and `q` are not the same per-round Bernoulli channels in
the two update modes. Main performance scans use `SYNCH=true`.

### 3.3 Cleanup, readout, and logical test

The decoded edge configuration is

~~~text
decoded_state = state xor state_correction
~~~

`detect_logical_error(decoded_state)` is named misleadingly: it returns
`true` when both torus winding parities are trivial, hence when the logical
test succeeds. Callers negate it or subtract it from one to obtain failure.

`get_decoding_time` copies the physical, syndrome, and history state, creates
fresh decoder fields/corrections, and repeatedly calls `update!` with
`p=q=0`. Fixed-time modes similarly run ideal cleanup after the noisy
interval.

When cleanup empties `hist`, the drivers assert that the decoded state is
syndrome-free. Several paths still evaluate winding even if cleanup times out
with nonempty history. That implemented failure convention must not be changed
silently.

### 3.4 Execution variants and overhead

#### 3.4.1 Serial baseline

`2d_windowed_simulation.jl` owns one baseline block state per active
trajectory. Its main modes are:

| Mode | Behavior |
| --- | --- |
| `hist` | Save one trajectory’s history, decoded state, and fields. |
| `erode` | Decode random initial states with no later noise. |
| `quench` | Track preparation and offline-decoding time from a random state. |
| `trel` | Evolve online and periodically decode a copy to estimate memory lifetime. |
| `Ft` | Run `T` noisy rounds, then up to `2T` ideal cleanup rounds. |
| `stats` | Estimate steady-state buffered-defect density. |

For ordinary online or `Ft` evolution, persistent decoder storage is
`M_block` and round work is `U_block`. The `trel` path additionally keeps
an offline snapshot. Its dominant integer storage is three field arrays
(`fields`, `new_fields`, and copied `dfields`), or `18NZ` machine
integers, rather than the ordinary `12NZ`.

#### 3.4.2 Threaded baseline

`2d_windowed_simulation_thread.jl` changes trial orchestration, not the local
decoder. It parallelizes independent trials for `erode`, `trel`, and
`Ft`; it does not parallelize a field sweep within one trial.

If `W` trial workers are active, the leading working-set and work scaling are:

~~~text
memory:  Theta(W M_block)
wall time: ideally about serial trial work / W
~~~

`trel` gives every worker its own online state and offline snapshot. The
current `compute` closures also retain coordinator/scratch arrays, so
`W M_block` is the worker-dependent leading term, not an exact resident-byte
formula.

Random trajectories are not bitwise stable across thread counts because trial
assignment and random-number consumption change. `TRIAL_PARALLEL=false`
forces one trial worker even when Julia has multiple threads.

## 4. CNOT Decoder Family

### 4.1 Common experiment lifecycle

The CNOT fixed-time drivers start with one control and one target logical block
and run:

~~~text
T_PRE noisy rounds on both outputs
one ideal X-sector CNOT event
T_POST noisy rounds on both outputs
up to CLEANUP_TIME synchronous ideal rounds
logical readout of control and target
~~~

By default:

~~~text
T_PRE        = floor(T/2)
T_POST       = T - T_PRE
CLEANUP_TIME = 2T
T            = L
~~~

For odd `T`, the extra noisy round is post-gate. The primitive driver also
accepts paired `CNOT_T_PRE` and `CNOT_T_POST` overrides; the sheet-copy and
block drivers use the split above. All accept a `CLEANUP_TIME` override.

The CNOT fidelity estimators either run exactly `SAMPS` trials or accumulate
`ACC_ERRORS` failed trials. Trial-level threading is enabled by default.
Each trial reports control, target, joint, and cleanup counts.

For all current CNOT models:

~~~text
logical_failure =
    control_logical_failure || target_logical_failure
~~~

`cleanup_failures` is a separate statistic and is not ORed into
`logical_failure`.

### 4.2 Primitive CNOT

#### 4.2.1 State ownership

`2d_windowed_cnot_primitive.jl` stores two independent baseline states:

~~~text
control:
    state_c, state_correction_c,
    old_synds_c, new_synds_c,
    hist_c, hist_correction_c,
    fields_c, new_fields_c

target:
    state_t, state_correction_t,
    old_synds_t, new_synds_t,
    hist_t, hist_correction_t,
    fields_t, new_fields_t
~~~

There are no structs, lineage objects, or per-gate persistent records.
`update_two_blocks!` simply calls the legacy baseline `update!` once on the
control and once on the target.

#### 4.2.2 Gate and round updates

`primitive_cnot_x_sector!` leaves the control physical state, frame, syndrome
registers, history, and current fields unchanged. It mutates the target:

~~~text
state_t            xor= state_c
state_correction_t xor= state_correction_c
old_synds_t        xor= old_synds_c
new_synds_t        xor= new_synds_c
hist_t             xor= hist_c
fields_t            = nonzeromin(fields_t, fields_c)

new_fields_c        = 0
new_fields_t        = 0
~~~

After the gate helper returns, the trial/demo driver clears both
`hist_correction` scratch arrays. Normal two-block updates then resume.

The target retains only one Boolean history and one componentwise-minimized
message landscape. It no longer records which residual defects or messages
came from the pre-gate target versus the control. This provenance loss is an
implemented fact; whether it fully explains the observed threshold loss
remains a working hypothesis.

#### 4.2.3 Readout and overhead

Readout is local:

~~~text
decoded_control = state_c xor state_correction_c
decoded_target  = state_t xor state_correction_t
~~~

For one trial worker:

~~~text
persistent decoder state = 2 M_block
leading field storage     = 24 N Z machine Ints
round work                = 2 U_block
gate work                 = Theta(NZ)
~~~

The gate adds no persistent storage. Repeated CNOTs between a fixed set of
blocks therefore do not increase memory with gate count. Each gate still
performs full-array XOR, field merge, and buffer clears. With `W` trial
workers, the leading CNOT working set is `2W M_block`, plus small decoded-state
and orchestration buffers.

### 4.3 Legacy sheet-copy CNOT

#### 4.3.1 State ownership

`2d_windowed_cnot_sheetcopy.jl` stores one complete baseline decoder per
lineage:

~~~text
mutable struct DecoderSheet
    block
    lineage_id
    parent_lineage_id
    created_by_gate
    hist
    fields
    new_fields
    hist_correction
    state_component
    state_correction
    old_synds
    new_synds
end
~~~

`state_component` is that sheet’s contribution to its assigned output
block. Every other array has the same role and shape as its baseline
counterpart. `initial_sheet_set` creates exactly two sheets: control lineage
1 and target lineage 2.

A sheet is “active” if any physical, frame, syndrome, history, correction, or
field array is nonzero. Allocated inactive sheets are not deleted and are still
updated every round.

#### 4.3.2 Gate and round updates

At a control-to-target CNOT, `apply_cnot_x_sheetcopy!`:

1. finds every active sheet currently assigned to the control;
2. deep-copies the full `DecoderSheet`;
3. assigns the copy to the target;
4. gives it a fresh lineage ID and records its parent and gate;
5. appends it without changing existing control or target sheets.

Mutable-array alias checks verify that parent and child do not share storage.
No histories or fields are merged at the gate.

`update_sheets!` then calls the complete legacy `update!` on every allocated
sheet. This has a crucial stochastic consequence: after a CNOT creates two
target sheets, both target sheets independently sample physical noise,
measurement noise, syndromes, histories, fields, and corrections. The model
therefore exposes hidden component syndromes and gives one physical output
multiple post-gate noise channels. It is retained as a legacy algorithmic
comparison, not as the same physical model as the block implementation.

#### 4.3.3 Readout and overhead

Only final readout combines lineages:

~~~text
decoded_state(block) =
    xor over all sheets assigned to block of
        (sheet.state_component xor sheet.state_correction)
~~~

Cleanup succeeds only when every sheet history is empty. Logical testing is
then performed on the two merged output states.

Let `S` be the total allocated sheet count, `S_c` the allocated control-sheet
count, and `S_c_active` the number of active control sheets at a gate. Then:

~~~text
persistent decoder state = S M_block + O(S) lineage metadata
round work                = S U_block
gate memory increase      = S_c_active M_block
gate work                 = Theta(S + (S_c + S_c_active) NZ) in the worst case
~~~

The gate-time expression includes the sheet-list scan, activity scans over
control sheets, and deep copies of active control states. Its coarse upper
bound is `O(SNZ)`. Readout costs `Theta(SN)`; cleanup checks can scan
`Theta(SNZ)` bits.

For one noisy CNOT, the usual counts are:

~~~text
before gate: 2 sheets
after gate:  3 sheets
target:      2 full decoder sheets
~~~

At zero noise, the empty control sheet can be inactive, in which case no copy
is made and the count remains two.

If all sheets are active, a gate updates lineage counts as

~~~text
s_target <- s_target + s_control
s_control unchanged
~~~

Repeated gates in one direction therefore grow linearly when the control count
is fixed. Alternating gate direction produces the Fibonacci recurrence and
exponential-in-gate-count lineage growth. Both memory and per-round decoder
work follow the total allocated sheet count. With `W` trial workers, the
leading working set is `W S M_block`.

The saved sheet-count metrics are:

- `sheetcopy_final_sheet_count_mean`;
- `sheetcopy_final_active_sheet_count_mean`;
- `sheetcopy_max_sheet_count`;
- `sheetcopy_max_active_sheet_count`;
- first-trial total and active count traces.

#### 4.3.4 Physical snapshot CNOT

`2d_windowed_cnot_snapshot.jl` is a separate synchronous prototype derived
from the legacy sheet-copy driver. It implements one ideal X-sector CNOT
between two blocks and rejects a second CNOT or `SYNCH=false`.

Physical and decoder state are separated:

~~~text
PhysicalBlock:
    errors, old_synds, new_synds, saved_correction,
    noise_rounds, measurement_rounds

DecoderHistory:
    history_id, live_block, applies_to,
    hist, fields, new_fields, hist_correction, correction
~~~

Only the two `PhysicalBlock`s receive physical noise, measurement noise, and
syndrome calculation. A live history is attached to one physical block. An
old history has `live_block=nothing` and advances only its stored defects,
fields, and corrections with an empty new front slice.

At the CNOT `c -> t`:

~~~text
errors[t]           xor= errors[c]
saved_correction[t] xor= saved_correction[c]
for every existing history h:
    h.applies_to[t] xor= h.applies_to[c]
~~~

The two existing live histories then become old histories without copying
their arrays. Fresh empty live control and target histories are allocated with
unit `applies_to` vectors. The control syndrome baseline remains its last
measured value. The target baseline becomes the XOR of the last measured
control and target syndromes. No noise or measurement occurs at the gate.

After the gate there are at most four histories before deletion:

~~~text
pre-control:  applies_to = [control,target], old
pre-target:   applies_to = [target],         old
post-control: applies_to = [control],        live
post-target:  applies_to = [target],         live
~~~

An empty old history is folded into `saved_correction` for every selected
output and deleted. Readout XORs the physical error array, saved correction,
and the correction from every remaining history whose `applies_to` bit selects
that block. Cleanup runs live histories with `p=q=0`, old histories with the
history-only rule, and succeeds only when every remaining history is empty.

For one worker, the physical state remains fixed at two blocks. Decoder work
is at most four history updates per round immediately after the gate and falls
as old histories empty. The dominant field storage is at most `4 * 12NZ`
machine integers before deletion. This version has no threshold evidence and
does not yet provide a repeated-gate driver; `applies_to` is present to support
that later extension without a deep-copied lineage tree.

### 4.4 Two-pass causal-junction CNOT

#### 4.4.1 State ownership

`2d_windowed_cnot_twopass.jl` implements one synchronous ideal X-sector CNOT.
It is a standalone derivative of the primitive CNOT lineage and embeds only
the required synchronous baseline memory-decoder helpers. Its CNOT state is
separate:

~~~text
TwopassPhysicalBlock:
    errors, frame, old_synds, new_synds,
    noise_rounds, measurement_rounds

TwopassCNOTState:
    two physical blocks
    one continuous control TwopassHistory
    one TwopassTargetHistory
    cnot_applied
    fixed spatial and temporal decoding weights

TwopassRoundMasks:
    one data mask and one measurement mask per physical block
~~~

The target history owns three fixed Boolean histories and independent field
banks:

~~~text
PRE_CONTROL = stored control observations before the gate
PRE_TARGET  = target observations before the gate
POST_TARGET = observable target observations after the gate
~~~

These are observable spacetime-segment labels, not hidden physical fault
labels. The target also owns a primitive retirement history, two branch-message
buffers, proposal arrays, a frozen `L x L x 3` retirement mask, and the moving
junction depth. Every target spatial proposal updates the one physical target
frame. Different labeled histories never annihilate directly.

#### 4.4.2 Gate and physical channels

Before the gate, the control and target use the ordinary baseline `update!`;
the target data is stored in `PRE_TARGET`. At the gate:

~~~text
errors[target]     xor= errors[control]
frame[target]      xor= frame[control]
old_synds[target]  xor= old_synds[control]
new_synds[target]  xor= new_synds[control]
~~~

The continuous control history is unchanged. Its history and current fields
are copied once into `PRE_CONTROL`, the existing `PRE_TARGET` history is
retained, and `POST_TARGET` is cleared. Scratch fields, proposals, junction
messages, and the retirement history are cleared. The arrays do not alias the
control decoder. No noise, measurement, or syndrome event occurs at the gate.

After the gate, `update_twopass_round!` pre-samples one Bernoulli-`p` edge mask
and one Bernoulli-`q` measurement mask for each physical block, then calls the
ordinary control update once and the custom target update once. The masks are
applied at the inherited physical-noise stage. Callers may instead pass an
explicit `TwopassRoundMasks` and separate noise/decoder RNGs; this is the paired
primitive-comparison interface. If `decoder_rng` is omitted, the round derives
a distinct `Xoshiro` from `noise_rng` only after sampling the four masks.
Decoder back-wall draws therefore cannot change the physical noise tape.
Only the observed target syndrome change is inserted into `POST_TARGET`;
`PRE_CONTROL`, `PRE_TARGET`, and the retirement history receive empty fronts.
Post-CNOT control noise is not supplied to target inference.

#### 4.4.3 Two-pass labeled feedback

The target update is synchronous and ordered as follows:

1. Run `r` inherited field sweeps independently on each labeled history and
   the retirement history. Junction messages are computed from the same
   pre-sweep lane fields, preserving global synchronous propagation.
2. First pass: store one preferred direction for every active labeled defect
   from the frozen histories.
3. Second pass: seed a scratch three-stream field bank from the persistent
   pass-one messages, run `r` synchronous sweeps on the frozen current labeled
   histories, route only the axis selected in pass one, and record the link
   without changing a history. Seeding preserves Lake-like message reach beyond
   one round's `r` sweeps. Junction branches are selected before ordinary
   proposals, and a directly selected pre-gate endpoint is skipped as a source.
4. Select primitive-retirement proposals. From the same frozen histories,
   select a labeled back-wall retirement only when no same-stream or junction
   proposal touches the defect. Incidence includes incoming edges.
5. Apply every labeled, junction, primitive, and retirement proposal in one
   XOR commit. Spatial proposal parity is XORed into the target frame;
   temporal, junction, and retirement operations change only histories.
6. Apply the pre-sampled target physical and measurement masks once, cycle
   every target history once, insert the new event only into `POST_TARGET`, and
   advance the junction depth.

Zero message means no candidate: the defect makes no corrective move and ages
normally. For positive equal minima, the immediate priority is

~~~text
temporal, -x, -y, +y, +x
~~~

There is no reciprocal-match requirement, confidence delay, or retirement for
an ordinary tie. A defect proposes at most one adjacent edge per round. Labeled
legal back-wall motion is deterministic; a wall defect with no legal incident
proposal retires. The primitive retirement history retains the inherited
raw-distance feedback and `0.8` stochastic back-wall rule.

The stored nominal weights are

~~~text
w_p = log((1-p)/p)
w_q = log((1-q)/q)
~~~

with infinite weight at probability zero. They remain fixed during noiseless
cleanup. The implementation multiplies an inherited integer directional
message by the corresponding axis weight at selection time. It does not
replace the inherited message kernel with exact edge-by-edge anisotropic
propagation.

Spatial and temporal movement stays within one stream except at the moving
CNOT junction, where a newer `POST_TARGET` defect may enter either
`PRE_CONTROL` or `PRE_TARGET`. A direct `PRE_CONTROL <-> PRE_TARGET` link is
forbidden. Equal branch costs select `PRE_CONTROL` before `PRE_TARGET`. Once a
post-gate defect crosses, it belongs to only the selected pre-gate stream.

#### 4.4.4 Retirement, cleanup, readout, and interface

The primitive retirement history is not an ordinary ambiguity fallback. For a
frozen labeled back-wall defect `H[i,j,Z,s]`, retirement is selected iff no
selected same-stream or junction edge is incident on that site. The XOR commit
clears the labeled bit and toggles the same coordinate in the retirement
history; it is never copied. Consequently, an occupied endpoint touched by an
incoming lane or capped-junction proposal resolves before retirement, a newly
arriving back-wall defect remains labeled until the next round, and two labels
retiring at one coordinate cancel in the primitive history. The retirement
history receives no direct syndrome stream.

Readout is local:

~~~text
decoded_control = errors[control] xor frame[control]
decoded_target  = errors[target]  xor frame[target]
~~~

Cleanup uses physical `p=q=0` and succeeds only when the control history, all
three labeled target histories, and the retirement history are empty. Cleanup
failure is recorded separately from logical failure. The logical trial fails
if either decoded output has nontrivial winding.

The public estimator is `estimate_twopass_cnot_Ft`, with the same timing and
fixed-sample/accumulate-failures arguments as the primitive estimator. The
guarded driver accepts `MODE=CNOT_Ft` and `MODE=CNOT_DEBUG`; it rejects
`SYNCH=false`, pretty updates, and a second CNOT. `OUTPUT_FILE` is optional; if
absent, `CNOT_Ft` prints results without writing a file.

#### 4.4.5 Overhead and implemented limits

The arrays are allocated at initialization and do not grow with elapsed time.
The control, three persistent target histories, and retirement history use five
field pairs, or `60NZ` machine integers. The reusable three-stream second-pass
field pair adds `36NZ`; persistent and second-pass junction buffers add `8NZ`.
Leading integer storage is therefore `104NZ` per trial, plus Boolean histories,
proposals, the `3N`-bit retirement mask, physical arrays, and container
overhead. Post-gate round work remains
`Theta(rNZ)` with a larger constant from the persistent and fresh routing
sweeps.

The driver implements only two blocks and one gate, so constant overhead under
repeated CNOTs is not established. There is no threshold scan, visualization,
asynchronous path, Z sector, gate noise, or general circuit interface.

### 4.5 Moving Y-junction CNOT

#### 4.5.1 State and gate

`2d_windowed_cnot_yjunction.jl` is a standalone synchronous derivative of the
primitive kernel. It separates two observable physical blocks from decoder
evidence:

~~~text
YPhysicalBlock:
    errors, frame, old_synds, new_synds,
    noise_rounds, measurement_rounds

DecoderLane:
    hist, fields, new_fields, proposals
~~~

Before the CNOT, control and target each own one ordinary `DecoderLane`. At the
gate, the target physical arrays transform as

~~~text
errors[target]     xor= errors[control]
frame[target]      xor= frame[control]
old_synds[target]  xor= old_synds[control]
new_synds[target]  xor= new_synds[control]
~~~

The target lane becomes `pre_target`; a non-aliased snapshot of the control
history and current field becomes `pre_control`; and a fresh empty
`post_target` lane is allocated. Scratch fields and proposals start empty. The
continuous control decoder is unchanged, and the gate creates no noise,
measurement, correction, or artificial history event.

All target lanes update one shared observable target frame. The copied control
lane contains no separate physical state or correction chain. Exactly one data
mask and one measurement mask are sampled per physical block per round through
the `YJunctionRoundMasks` interface.

#### 4.5.2 Y-graph field and feedback rules

At junction depth `g`, the post lane owns `k <= g`, while both pre-gate lanes
independently own `k > g`. The junction starts at `g=0` and advances once per
physical or cleanup round. Invalid lane slices are zero.

Every field sweep is globally Jacobi over this branched topology. The inherited
`3 x 3` candidate plane and 1-norm distance are unchanged. A post-side cone
crossing from `g` to `g+1` evaluates both pre lanes and retains their smallest
positive candidate. A pre-side cone crossing toward `g` evaluates the single
post lane, so post messages advertise into both branches. The rule applies to
all six field components. Only the merged distance is stored below the
junction; zero still means no message.

The two temporal branch costs at the interface are retained from the same
frozen field state as the merged post message. If a post defect at `k=g`
selects buffer motion, it crosses to exactly one pre endpoint: the smaller
positive cost wins, with control first on an equal positive tie. A defect is
never copied. Post defects below the interface move within the post lane;
pre-gate defects retain ordinary one-way aging toward larger `k`.

All ordinary and junction proposals are selected from frozen histories and
committed atomically. The XOR parity of spatial proposals from every target
lane updates the one target frame. The primitive priorities and spatial-only
`0.8` stochastic back-wall rule are retained. After feedback, the one observed
target syndrome change is inserted only into `post_target`.

#### 4.5.3 Collapse, readout, and interface

After each round, all three lanes cycle toward the back wall and `g` advances.
When `g=Z`, the unresolved pre-gate evidence is destructively collapsed:

~~~text
post.hist[:,:,Z] xor=
    pre_control.hist[:,:,Z] xor pre_target.hist[:,:,Z]

post.fields[:,:,Z,spatial,:] =
    nonzeromin(post, pre_control, pre_target)
~~~

Post temporal back-wall fields and scratch are cleared. The state replaces the
transition object with `post_target`, releasing both pre lanes. Subsequent
target rounds use the ordinary baseline lane update. This is the only
provenance-destroying merge and is a heuristic for residual clusters that
survive the complete buffer.

Readout is local:

~~~text
decoded_control = errors[control] xor frame[control]
decoded_target  = errors[target]  xor frame[target]
~~~

Cleanup continues with `p=q=0` until histories are empty and the junction has
collapsed, subject to the configured cap. Nonempty-history cleanup failures
and collapse failures are reported separately. The guarded driver accepts
`MODE=CNOT_Ft` and `MODE=CNOT_DEBUG`, exposes
`estimate_yjunction_cnot_Ft`, and rejects asynchronous, pretty, or second-gate
use. The scan scripts export an absolute `OUTPUT_FILE` for every array task;
the driver creates its parent directory and writes the standard `### data ###`
and `### params ###` result sections. Fixed-sample runs use `SAMPS` (with
`CNOT_SAMPS` retained as a legacy fallback); without `OUTPUT_FILE`, a direct
driver invocation prints the result without creating a file.

#### 4.5.4 Overhead and implemented limits

Before the gate and after collapse, the control and target use two field pairs,
or `24NZ` machine integers. During the transition, continuous control,
pre-control, pre-target, and post-target use four pairs, or `48NZ` integers.
At depth `g`, field work visits `3Z-g` slice-equivalents per spatial site and
falls back to two ordinary block updates after collapse. The gate costs
`Theta(NZ)`; collapse touches `Theta(N)` wall entries. The transition lasts
exactly `Z` post-gate or cleanup rounds.

The driver implements two blocks and one ideal synchronous X-sector gate. It
has no repeated-gate schedule, asynchronous update, visualization, Z sector,
gate noise, or matched finite-size scan.



## 5. Comparison and Evidence

### 5.1 Overhead and information flow

The following table compares one trial worker. Multiply trial-local state by
the active worker count for threaded scans.

| Model | Full decoder states | Post-gate target information | Persistent state | Round work | Repeated-gate growth |
| --- | ---: | --- | --- | --- | --- |
| Baseline | 1 | Not applicable | `M_block` | `U_block` | None |
| Primitive | 2 | One merged history; fields compressed with `nonzeromin` | `2 M_block` | `2 U_block` | No gate-count growth |
| Sheet-copy | `S` | Separate hidden target and copied-control decoder sheets | `S M_block + O(S)` | `S U_block` | Full-state lineage recurrence; alternating gates can grow as Fibonacci |
| Snapshot | 2 physical blocks, up to 4 histories | Separate pre-/post-gate histories; only observable blocks are measured | Up to about `4 M_block` before old-history deletion | Up to `4 U_block`, falling as old histories empty | Repeated driver not implemented; `applies_to` avoids deep-copy routing |
| Two-pass | 2 physical blocks, 1 control history, 3 target streams, 1 retirement history | Separate causal streams joined by label-aware temporal edges | `104NZ` leading integer words plus Boolean state | `Theta(rNZ)` with persistent and fresh route sweeps | Repeated driver not implemented; fixed one-gate allocation |
| Y-junction | 2 physical blocks; 1 control lane and transient 3-lane target | Two pre-gate branches meet one unlabeled post-gate field on a full local Y graph | Peak `48NZ` integer words, returning to `24NZ` after `Z` rounds | At most `3U_block`, falling to `2U_block` | Repeated driver not implemented; one-gate branches are released at collapse |
| Block | 2 | One observable XOR-combined history; target fields rebuilt | `2 M_block + O(A)` metadata | `2 U_block`, independent of `A` | Decoder state fixed; stored metadata can follow the same Fibonacci recurrence |

The central tradeoff is therefore:

- primitive has fixed low overhead but irreversibly compresses pre-gate
  histories and messages;
- sheet-copy preserves component histories but pays one decoder and one later
  stochastic channel per sheet;
- snapshot preserves separate pre-/post-gate histories while using only two
  physical channels, but can temporarily require four decoder histories and
  forbids matching between histories;
- two-pass uses the same two physical channels, retains separate target causal
  streams, and permits only the two target-output CNOT-junction paths; its
  performance and larger constant field cost are not yet characterized;
- Y-junction uses the same two physical channels, permits full bidirectional
  message cones across a moving interface, and deliberately loses the oldest
  branch labels at the back wall; its performance is not yet characterized;
- block restores one physical/measurement channel per output and fixed decoder
  state, but uses a field reset whose decoding performance is not yet
  established.

### 5.2 Empirical status

These are summaries of saved finite-size scans, not theoretical threshold
claims. Comparisons require matched `L`, `p`, `q`, `Z`, `r`, noisy
time, CNOT timing, and cleanup.

#### 5.2.1 Baseline memory

`results/baseline/ft/thread_test/summary.csv` uses
`q=p`, `r=3`, synchronous updates, logarithmic `Z`, `T=L`, and
`2T` cleanup. Representative fidelities are:

~~~text
p=0.015: L=5 0.9577, L=9 0.9556, L=13 0.9575, L=19 0.9539
p=0.016: L=5 0.9524, L=9 0.9435, L=13 0.9373, L=19 0.9313
p=0.017: L=5 0.9390, L=9 0.9217, L=13 0.9133, L=19 0.8837
~~~

The finite-size crossing trend is near `p=0.015-0.016` for this protocol.

#### 5.2.2 Primitive CNOT

`results/cnot_primitive/full_scan/T∕2_CNOT_T∕2_2T` uses matched split
timing and otherwise comparable decoder parameters. Representative
`CNOT_Ft` values are:

~~~text
p=0.012: L=5 0.9294, L=9 0.9338, L=13 0.9446, L=19 0.9559
p=0.014: L=5 0.8894, L=9 0.8718, L=13 0.8717, L=19 0.8605
p=0.015: L=5 0.8683, L=9 0.8291, L=13 0.8149, L=19 0.7804
~~~

The crossing-like trend is lower than the baseline. Target failures dominate
the saved split-timing data, which is consistent with—but does not prove—the
history-compression hypothesis.

#### 5.2.3 Legacy sheet-copy CNOT

`results/cnot_sheetcopy/full_scan/T∕2_CNOT_T∕2_2T` contains:

~~~text
p=0.014: L=5 0.9147, L=9 0.9143, L=13 0.9236, L=19 0.9333
p=0.015: L=5 0.8944, L=9 0.8814, L=13 0.8890, L=19 0.8850
p=0.016: L=5 0.8721, L=9 0.8459, L=13 0.8340, L=19 0.8127
~~~

The trend is closer to the baseline than primitive CNOT. These results were
generated by the independently noisy, independently measured sheet model.
They are valid only as results for
`2d_windowed_cnot_sheetcopy.jl`; they are not physical-performance evidence
for the block implementation.

Representative saved sheet counts are:

~~~text
p=0.009, L=5:  final mean 2.744, active mean 2.677
p=0.011, L=9:  final mean 3.000, active mean 3.000
p=0.015, L=13: final mean 3.000, active mean 3.000
p=0.020, L=19: final mean 3.000, active mean 3.000
~~~

#### 5.2.4 Block CNOT

No threshold or matched full-scan result is established for the block-level
field-reset rule. New scans are required before claiming baseline-like or
sheet-copy-like performance.

#### 5.2.5 Snapshot CNOT

Only zero-noise and small deterministic/noisy regression runs are established
for `2d_windowed_cnot_snapshot.jl`. No threshold or matched full-scan result is
available. Legacy sheet-copy scan values cannot be transferred because the
snapshot model has one physical and measurement channel per output rather than
one channel per hidden sheet.

#### 5.2.6 Two-pass CNOT

The deterministic suite, bounds-checked suite, zero-noise debug path, and a
small noisy fixed-sample run pass. Explicit paired mask tapes for
measurement-only (`p=0`), storage-only (`q=0`), and balanced (`p=q`) noise give
exact pre-gate parity with the copied primitive kernel and exact physical-error
and syndrome parity after the gate. Decoder frames may differ after the gate by
design. No matched finite-size scan or threshold estimate exists. Existing
saved two-pass scans predate the atomic no-continuation retirement rule and are
not validation of the current implementation. The remaining comparison is
paired logical trials against primitive followed by the matched
`L=5,7,9,13,19` protocol. Improved storage-error attribution without loss of
the primitive crossing remains a working hypothesis, not an established
result.

#### 5.2.7 Moving Y-junction CNOT

The focused deterministic and bounds-checked suites pass. Explicit paired
masks give exact persistent-state parity with the copied primitive kernel
before the gate and exact physical-error and syndrome-register parity after
the gate; decoder frames may differ by design. Reversed lane/site iteration
gives the same globally Jacobi field state and branch costs. The zero-noise
estimator, small noisy fixed-sample path, and two-thread fixed-sample accounting
also pass. No matched finite-size scan or threshold estimate exists. The
primary empirical question is whether retaining two pre-gate fields until the
moving interface reaches the back wall improves target failures over primitive
without introducing excess cleanup failures. Sheet-copy remains an
algorithmic oracle rather than a matched physical model.

## 6. Validation and Operational Caveats

### 6.1 Validation and diagnostics

The CNOT drivers expose `MODE=CNOT_DEBUG`:

~~~bash
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 julia --threads=1 2d_windowed_cnot_primitive.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 julia --threads=1 2d_windowed_cnot_sheetcopy.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 julia --threads=1 2d_windowed_cnot_block.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 CLEANUP_TIME=4 SYNCH=true \
    TRIAL_PARALLEL=false julia --threads=1 2d_windowed_cnot_snapshot.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 CLEANUP_TIME=4 SYNCH=true \
    TRIAL_PARALLEL=false julia --threads=1 2d_windowed_cnot_twopass.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false ZVAL=2 TVAL=2 CLEANUP_TIME=4 SYNCH=true \
    TRIAL_PARALLEL=false julia --threads=1 2d_windowed_cnot_yjunction.jl
~~~

The sheet-copy and block commands complete successfully. Sheet-copy debug mode
checks baseline single-sheet equivalence, deep-copy ownership, lineage IDs,
merged readout, and zero-noise behavior.

Snapshot debug mode checks physical CNOT algebra, the XOR target syndrome
baseline, live/old history ownership, `applies_to`, one physical update per
block, deletion/readout invariance, and zero-noise cleanup. Its separate test
suite also runs a small noisy sample:

~~~bash
julia --startup-file=no test/snapshot_runtests.jl
~~~

Two-pass debug mode checks the ideal physical-error map, non-aliasing of copied
pre-control evidence, and zero-noise cleanup. Its focused suite covers field
priority, standalone loading without snapshot symbols, stored weights,
no-candidate aging, second-pass recomputation, globally synchronous junction
messages, forced single data and measurement faults on all observable streams,
labeled routing, junction branch priority, physical-channel ownership,
one-edge locality, frozen proposal generation, XOR frame parity, reversed
stream/site/edge-order equivalence, atomic back-wall retirement and parity collisions,
paired primitive mask traces in all three noise regimes, and exact threaded
fixed-sample accounting:

~~~bash
julia --startup-file=no test/twopass_runtests.jl
julia --startup-file=no --check-bounds=yes test/twopass_runtests.jl
~~~

The Y-junction suite checks persistent pre-gate parity with the primitive
kernel, paired physical/syndrome parity through the gate, gate algebra and
non-aliasing, all-component bidirectional field cones, global Jacobi
propagation, reversed lane/site-order equivalence, zero-aware branch minima,
single-owned crossings, control-first ties, atomic shared-frame feedback, one
physical channel per block, back-wall parity/field collapse, lane release,
post-collapse ordinary updates, metric accounting, and small zero/noisy
estimators:

~~~bash
julia --startup-file=no test/yjunction_runtests.jl
julia --startup-file=no --check-bounds=yes test/yjunction_runtests.jl
~~~

Primitive debug mode is intended to check XOR propagation, `nonzeromin`,
field-buffer clearing, and zero-noise success. Its current sanity function uses
`any(new_fields_c)` and `any(new_fields_t)` on integer arrays, however, which
raises `TypeError: non-boolean (Int64) used in boolean context` on the current
Julia version before the mode completes. This is a defect in the diagnostic
assertion; it is not a passing primitive regression hook.

The block regression suite checks:

- ideal error/frame propagation and control preservation;
- XOR of aligned binary decoder state and target field reset;
- no continuous post-gate propagation;
- exactly one physical and measurement mask per block round;
- cancellation of hidden contributions in the observable target;
- a case where combined decoding differs from separate component decoding;
- ancestry-count invariance in synchronous and asynchronous updates;
- noise-free two-block cleanup and readout.

Run:

~~~bash
julia --startup-file=no test/runtests.jl
julia --startup-file=no --check-bounds=yes test/runtests.jl
~~~

The block CNOT output reports ancestry counts, constant physical-block counts,
and separate decoder/physical byte measurements. The sheet-copy output reports
total and active sheet counts. Those diagnostics measure different ownership
models and should not be compared as if both were physical-block counts.

### 6.2 Implemented limitations and pitfalls

- `detect_logical_error` returns logical success.
- `hist` stores syndrome changes, not current syndrome values.
- Zero-valued fields mean “absent.” Primitive uses `nonzeromin`; block resets
  target fields instead.
- Primitive clears both `new_fields` arrays at the gate and both
  `hist_correction` arrays in the driver.
- Primitive `CNOT_DEBUG` currently aborts on an integer-array `any` assertion;
  there is no clean end-to-end primitive debug hook until that assertion is
  fixed.
- The serial, threaded, and primitive drivers call `Alert.alert` unconditionally
  at normal exit. The notification backend can make an otherwise completed
  headless run exit with an external-command error. Sheet-copy and block use
  the opt-in, exception-catching `safe_alert` path instead.
- Sheet-copy deep-copies only active control sheets, never prunes allocated
  sheets, and applies later noise once per sheet.
- Snapshot CNOT supports one ideal X-sector gate, two blocks, and synchronous
  updates only. It has no `CNOT_DEMO` mode or threshold scan evidence yet.
- Snapshot old histories receive an empty front slice and cannot match defects
  with other old or live histories; the performance cost is unknown.
- Snapshot `applies_to` implements future correction routing, but the current
  driver rejects a second CNOT.
- Two-pass supports one ideal synchronous X-sector CNOT and two blocks. It has
  no asynchronous, repeated-gate, visualization, or threshold-scan result.
- Two-pass ordinary bulk ties move immediately by fixed priority; labeled
  defects enter the primitive retirement history only at the finite back wall
  and only when no legal same-stream or junction proposal touches them.
- Two-pass physical masks are sampled before decoder work and applied later at
  the inherited noise stage. This changes RNG consumption relative to old
  saved scans while preserving the Bernoulli channel.
- Two-pass applies `w_p` or `w_q` to inherited integer directional distances;
  this is not exact edge-weighted anisotropic message propagation.
- Two-pass embeds a synchronous-only copy of the primitive/baseline memory
  kernel. Later kernel fixes must therefore be reviewed explicitly here too.
- Y-junction supports one ideal synchronous X-sector CNOT and two blocks. It
  has no asynchronous, repeated-gate, visualization, or threshold-scan result.
- Y-junction message values may fan out into both pre branches, but defects,
  physical errors, correction links, and noise masks remain single-owned.
- Y-junction back-wall XOR-collapse deliberately discards pre-control versus
  pre-target provenance for residual defects after one buffer depth. Its
  logical-performance cost is unknown.
- Block ancestry is dynamically irrelevant but not memory-free. Its byte
  diagnostics exclude ancestry.
- Block asynchronous dynamics are not the legacy baseline asynchronous
  dynamics.
- Block `CNOT_STYLE=sheetcopy` is accepted only as a legacy alias and still
  selects the block algorithm. The actual sheet model lives in
  `2d_windowed_cnot_sheetcopy.jl`.
- CNOT cleanup failure is recorded separately from logical failure.
- The block prototype requires exactly two physical blocks and aligned
  pre-gate buffer coordinates.
- Saved sheet-copy scans do not validate the block field-reset policy.
- Saved two-pass scans from before the atomic retirement change do not validate
  the current two-pass rule.
- Result directories with different `T_PRE`, `T_POST`, or cleanup schedules
  are not directly comparable.
- Some result paths use the Unicode division slash in `T∕2`; quote those
  paths in shell commands.
- Core decoder code is duplicated. Any behavioral fix must be reviewed
  deliberately across the serial baseline, threaded baseline, primitive,
  sheet-copy, snapshot, two-pass, Y-junction, and block files.
