# mp-decoder Implementation Documentation

Last updated: 2026-07-15

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

All three CNOT drivers start with one control and one target logical block and
run:

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

For all three models:

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
| Block | 2 | One observable XOR-combined history; target fields rebuilt | `2 M_block + O(A)` metadata | `2 U_block`, independent of `A` | Decoder state fixed; stored metadata can follow the same Fibonacci recurrence |

The central tradeoff is therefore:

- primitive has fixed low overhead but irreversibly compresses pre-gate
  histories and messages;
- sheet-copy preserves component histories but pays one decoder and one later
  stochastic channel per sheet;
- snapshot preserves separate pre-/post-gate histories while using only two
  physical channels, but can temporarily require four decoder histories and
  forbids matching between histories;
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

## 6. Validation and Operational Caveats

### 6.1 Validation and diagnostics

The CNOT drivers expose `MODE=CNOT_DEBUG`:

~~~bash
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 julia --threads=1 2d_windowed_cnot_primitive.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 julia --threads=1 2d_windowed_cnot_sheetcopy.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 julia --threads=1 2d_windowed_cnot_block.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 CLEANUP_TIME=4 SYNCH=true \
    TRIAL_PARALLEL=false julia --threads=1 2d_windowed_cnot_snapshot.jl
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
- Result directories with different `T_PRE`, `T_POST`, or cleanup schedules
  are not directly comparable.
- Some result paths use the Unicode division slash in `T∕2`; quote those
  paths in shell commands.
- Core decoder code is duplicated. Any behavioral fix must be reviewed
  deliberately across the serial baseline, threaded baseline, primitive,
  sheet-copy, snapshot, and block files.
