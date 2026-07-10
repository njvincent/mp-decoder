# mp-decoder Implementation Documentation

Last updated: 2026-07-08

This document summarizes the existing Julia implementations in this repository
so an agent can understand the data structures, update rules, performance
behavior, and classical overhead without rereading every file from scratch. It
is still not a license to edit decoder logic blindly: before changing an update
rule, re-check the relevant Julia function and the reference papers listed in
`agent.md`.

## Implementation Map

- `2d_windowed_simulation.jl`
  - Serial baseline 2D windowed message-passing decoder for one toric-code error
    sector.
  - Contains the reference `update!` rule and all baseline Monte Carlo modes.
- `2d_windowed_simulation_thread.jl`
  - Threaded baseline. The local decoder rule is intended to be equivalent to
    `2d_windowed_simulation.jl`.
  - Adds trial-level parallelism for independent Monte Carlo samples.
- `2d_windowed_cnot_primitive.jl`
  - First CNOT prototype.
  - Duplicates the baseline decoder and adds a two-block X-sector CNOT
    bookkeeping map that immediately merges control information into the
    target arrays.
- `2d_windowed_cnot_sheetcopy.jl`
  - Second CNOT prototype.
  - Duplicates the baseline decoder and adds explicit decoder-sheet lineages.
    A CNOT deep-copies active control sheets to the target and defers algebraic
    merging until final readout.
- `2d_windowed_history_visualizer.py`
  - Visualizes saved baseline or primitive CNOT history data.
- `2d_cnot_sheetcopy_visualizer.js`
  - Browser-side visualization support for sheet-copy CNOT demos.
- `jobs/`
  - Slurm scan scripts for baseline, primitive CNOT, and sheet-copy CNOT runs.
- `results/`
  - Saved scan outputs. Treat these as data, not scratch files.

## Conventions

The current implementations track one error sector. The CNOT prototypes are
explicitly X-sector demonstrations. They do not implement a full surface-code
CNOT, a Z-sector propagation rule, labeled defects, or a CNOT gate fault model.

The physical toric-code state is a Boolean array:

```text
state :: L x L x 2
```

The last index is the edge orientation:

- `o = 1`: x-directed edge from `(i,j)` to `(i+1,j)`.
- `o = 2`: y-directed edge from `(i,j)` to `(i,j+1)`.

Spatial boundary conditions are periodic in the main scans. The helper
`init_2d` contains a finite-boundary branch, but the Monte Carlo paths use
`"periodic"`.

The syndrome array is:

```text
synds :: L x L
synds[i,j] = state[i,j,1] xor state[i,j,2] xor
             state[i-1,j,1] xor state[i,j-1,2]
```

All spatial indices are wrapped with `mod1`. RG-time indices are clamped to
`1:Z` when looking up neighbors for field updates.

Important naming caveat: `detect_logical_error(state)` returns `true` when the
decoded state has trivial winding in both cycles, so it is logically successful.
Most callers convert this to a failure indicator with
`1 - detect_logical_error(...)` or `!detect_logical_error(...)`.

Default RG depth is:

```text
Z = ceil(Int, log(1.5, L))   if LOGZ=true
Z = ceil(Int, L/4)           if LOGZ=false
```

The common scan defaults are `QRAT=1`, `RVAL=3`, `SYNCH=true`, `LOGZ=true`.
Measurement noise is set as `q = qrat * p`.

## Baseline Memory Decoder

### Live State

One baseline decoder block has these live arrays:

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

`hist` is not the current syndrome. It is the rolling RG-time history of
syndrome-changing events:

```text
hist[:,:,1] = old_synds xor new_synds
```

after the RG cycle makes room at the front of the window.

`fields` stores integer distance messages. The fourth index is the axis:

- `a = 1`: x direction.
- `a = 2`: y direction.
- `a = 3`: RG-time direction.

The fifth index is direction. In the feedback rule:

- `fields[...,1,1]` means the preferred x move is toward `i-1`.
- `fields[...,1,2]` means the preferred x move is toward `i+1`.
- `fields[...,2,1]` means the preferred y move is toward `j-1`.
- `fields[...,2,2]` means the preferred y move is toward `j+1`.
- `fields[...,3,2]` is used for motion toward `k+1`.

Zero means "no message". This is why CNOT code uses `nonzeromin` instead of a
plain minimum when merging field arrays.

### Field Update Rule

`onesite_field_update(i,j,k,fields,hist)` computes all six outgoing field
values for one space-time point. For each axis `a` and direction `s`, it scans
the nine neighboring sites one step away along axis `a` and with offsets
`-1:1` along the other two axes. It sets the new field to the minimum of:

- the 1-norm distance to a neighboring active history event, and
- a neighboring nonzero field plus the same 1-norm distance.

If no neighboring source or field exists, the new field is zero.

The synchronous field update `update_2d_windowed_fields!` computes this for
every `(i,j,k)` into `new_fields`, then copies `new_fields` into `fields`.

The asynchronous field update `update_2d_windowed_fields_column!` only updates
one spatial processor column `(i,j,:)`.

### Synchronous Decoder Step

`update!(..., r, p, q, synch=true, pretty)` performs one online decoding step.
The synchronous branch is the reference rule:

1. Update all fields `r` times. If `pretty=true`, update them `r-1` times here
   and do one extra source-smoothing update after the new syndrome is inserted.
2. Clear `hist_correction`.
3. For every active history event `hist[i,j,k]`, inspect local fields and choose
   a correction direction with smallest positive field value.
4. In the bulk (`k < Z`), the tie priority is:

```text
+z, -x, -y, +y, +x
```

5. On the back wall (`k == Z`), only spatial moves are allowed. A move is
   attempted with probability `0.8`; this stochasticity is intended to break
   locked cycles. The tie priority is:

```text
-x, -y, +y, +x
```

6. Convert spatial history corrections into physical state corrections:

```text
state_correction[i,j,a] xor= xor_reduce(hist_correction[i,j,:,a])
```

for `a = 1,2`.

7. Apply all history corrections with `perform_correction!`. A correction link
   toggles the two endpoint history sites. Axis 3 toggles `(i,j,k)` and
   `(i,j,k+1)`.
8. Apply physical noise to every edge independently with probability `p`.
9. Shift syndrome registers:

```text
old_synds = new_synds
new_synds = get_synds(state)
new_synds xor= measurement_noise(q)
```

10. Run `rg_cycle!`:

- `hist[:,:,Z] xor= hist[:,:,Z-1]` splats the old front of the back wall.
- `hist[:,:,2:end-1] = hist[:,:,1:end-2]` shifts history toward larger `k`.
- `hist[:,:,1] = false` clears the front slice.
- For fields, the back-wall spatial messages are preserved by
  `nonzeromin(fields[:,:,Z-1,1:2,:], fields[:,:,Z,1:2,:])`.
- Middle field slices shift forward in RG time.
- `fields[:,:,1,:,:] = 0` clears the front slice.

11. Insert the new syndrome-change events:

```text
hist[:,:,1] = old_synds xor new_synds
```

12. If `pretty=true`, source fields around active anyons with
`anyons_source_fields!` and do one final field update for smoother animations.

### Asynchronous Decoder Step

The asynchronous branch of `update!` runs `(r+1) * L^2` microsteps. Each
microstep chooses a random spatial column `(i,j)`.

- With probability `r/(r+1)`, it updates fields only in that column.
- With probability `1/(r+1)`, it performs feedback, noise, single-site syndrome
  refresh, and RG cycling for that column.

Spatial corrections are applied immediately to `state_correction` and `hist`.
Vertical RG-time corrections are accumulated in a temporary `vertical_correction`
vector and then applied with `perform_correction_column!`.

Physical noise in this branch toggles one random edge with probability `p` per
feedback microstep, rather than applying independent noise to every edge in a
global sweep. This branch is present but the main scans use `SYNCH=true`.

### Offline Decoding

`get_decoding_time(state,old_synds,new_synds,hist,r,synch)` decodes a copy of a
history with `p=q=0` until either:

- `hist` is empty, or
- `t > L^2`.

It returns the number of ideal decoding steps used. It constructs fresh fields
and corrections for the copied run.

### Failure Criteria

The decoded state is always:

```text
decoded_state = state xor state_correction
```

When cleanup succeeds and `hist` is empty, callers assert that
`get_synds(decoded_state)` is zero. Logical success is then
`detect_logical_error(decoded_state) == true`.

Some Monte Carlo paths still call `detect_logical_error` even when cleanup left
nonzero history; this behavior is intentional in the current code and should
not be changed silently.

### Baseline Modes

`MODE` selects the simulation:

- `hist`: save a single evolution history for visualization.
- `erode`: offline decode random initial states with no further noise.
- `quench`: track preparation/decoding time from random initial state.
- `trel`: estimate relaxation time or memory lifetime. The code periodically
  decodes a copy and stops a sample on logical failure.
- `Ft`: fixed-time online decoding fidelity. Default `T=L`; after noisy
  evolution, run `2T` ideal cleanup steps.
- `stats`: thermalize and estimate steady-state anyon density.

For `Ft`, the recorded value is:

```text
Ft = 1 - logical_failures / trials
```

The default `ACC_ERRORS` is 1000 in the threaded baseline and CNOT scan scripts.

## Threaded Baseline

`2d_windowed_simulation_thread.jl` keeps the local decoder functions equivalent
to the serial baseline and adds `using Base.Threads`.

Parallelism is trial-level, not intra-trial. Each worker allocates its own
state, history, field buffers, and correction buffers. This applies to:

- `erode`: split the target logical-failure count across workers.
- `trel`: split samples by worker stride.
- `Ft`: split the target logical-failure count across workers.

Consequences:

- The local update rule should match the serial implementation.
- Random trajectories are not bitwise reproducible across different thread
  counts because worker scheduling and RNG consumption differ.
- Resident memory scales approximately with the number of active workers times
  the per-trial decoder memory.

## Primitive CNOT Prototype

### Scope

`2d_windowed_cnot_primitive.jl` implements an X-sector two-block CNOT
bookkeeping experiment. It is not a full computation decoder.

The tracked rule is:

```text
control_out = control
target_out  = target xor control
```

Only control-to-target X-sector propagation is represented.

### Live State

Primitive CNOT uses two independent copies of the baseline arrays:

```text
control: state_c, state_correction_c, old_synds_c, new_synds_c,
         hist_c, hist_correction_c, fields_c, new_fields_c

target:  state_t, state_correction_t, old_synds_t, new_synds_t,
         hist_t, hist_correction_t, fields_t, new_fields_t
```

There is no lineage structure and no per-gate sheet storage.

`update_two_blocks!` simply applies the ordinary baseline `update!` to control
and target separately with the same `(r,p,q,synch,pretty)` arguments.

### Timing

`split_cnot_timing(T)` returns:

```text
T_PRE       = floor(T/2)
T_POST      = T - T_PRE
CLEANUP_TIME = 2T
```

For odd `T`, the extra noisy round is after the CNOT. In `main`, `T` defaults
to `L` for `MODE=CNOT_Ft`. The primitive script also supports explicit
`CNOT_T_PRE` and `CNOT_T_POST`, but they must be set together and must sum to
`TVAL`.

Existing result directories include older timing choices such as `T,0,2T` and
`T,T,2T`. Do not compare these to the current split-timing default unless the
timing difference is part of the comparison.

### CNOT Event Rule

`primitive_cnot_x_sector!` mutates only the target block, except for clearing
both `new_fields` buffers:

```text
state_t            xor= state_c
state_correction_t xor= state_correction_c
old_synds_t        xor= old_synds_c
new_synds_t        xor= new_synds_c
hist_t             xor= hist_c
fields_t            = nonzeromin(fields_t, fields_c)
new_fields_c        = 0
new_fields_t        = 0
```

The control block's state, correction, syndrome registers, history, and fields
are otherwise unchanged. The driver clears both `hist_correction` buffers after
the CNOT.

The lossy part is the immediate merge into the target. After the gate, target
history and fields no longer know which defects came from target history and
which came from copied control history. The field merge keeps only nearest
nonzero messages by component. It does not keep two independent histories or
two independent field landscapes.

### Trial Failure Rule

`estimate_primitive_cnot_Ft` runs:

1. `T_PRE` noisy two-block updates.
2. One primitive CNOT event.
3. `T_POST` noisy two-block updates.
4. Up to `CLEANUP_TIME` ideal synchronous two-block updates.

It then computes:

```text
decoded_state_c = state_c xor state_correction_c
decoded_state_t = state_t xor state_correction_t
control_logical_failure = !detect_logical_error(decoded_state_c)
target_logical_failure  = !detect_logical_error(decoded_state_t)
logical_failure = control_logical_failure || target_logical_failure
```

`cleanup_failures` is recorded separately as nonempty control or target history
after cleanup. It is not explicitly ORed into `logical_failure` in the current
implementation.

Recorded CNOT metrics include:

- `CNOT_Ft`
- `CNOT_fail_rate`
- `trials`
- `logical_failures`
- `control_logical_failures`
- `target_logical_failures`
- `both_logical_failures`
- `cleanup_failures`

### Classical Overhead

For two logical blocks, primitive CNOT stores exactly two baseline decoder
states. The CNOT event itself allocates no persistent lineage state.

Ignoring thread-level replication, the leading field-buffer memory is:

```text
2 blocks * 12 L^2 Z Ints
```

plus bit-packed state, syndrome, history, and correction arrays. The gate costs
`O(L^2 Z)` time for the history and field merge. Each decoder round costs two
baseline updates.

This is the main advantage of the primitive prototype: it keeps classical space
overhead close to baseline for the number of active logical blocks. The
drawback is decoding performance.

## Sheet-Copy CNOT Prototype

### Scope

`2d_windowed_cnot_sheetcopy.jl` implements an X-sector CNOT by tracking
independent decoder-sheet lineages. It preserves the full copied control
history and fields by deep-copying active control sheets to the target at the
CNOT event.

The key invariant is:

```text
Sheets do not exchange hist, field, correction, or syndrome data during local
decoding. They are merged only by xor at final readout.
```

### DecoderSheet

The core data type is:

```text
mutable struct DecoderSheet
    block::Int
    lineage_id::Int
    parent_lineage_id::Union{Int,Nothing}
    created_by_gate::Union{Int,Nothing}
    hist::BitArray{3}
    fields::Array{Int,5}
    new_fields::Array{Int,5}
    hist_correction::BitArray{4}
    state_component::BitArray{3}
    state_correction::BitArray{3}
    old_synds::BitArray{2}
    new_synds::BitArray{2}
end
```

`block` is `CONTROL_BLOCK = 1` or `TARGET_BLOCK = 2`.

`state_component` is the sheet's contribution to the physical state. This is
the sheet-copy counterpart of baseline `state`.

`lineage_id` is unique. `parent_lineage_id` and `created_by_gate` record where a
copied target sheet came from.

`initial_sheet_set(L,Z)` creates exactly two sheets:

- control sheet, lineage 1,
- target sheet, lineage 2.

`sheet_active(sheet)` returns true if any of the sheet's arrays contain
nonzero/nonfalse data. This includes physical state, corrections, syndrome
registers, history, and fields. A completely empty control sheet is not copied
at a CNOT.

### Sheet-Copy CNOT Event Rule

`apply_cnot_x_sheetcopy!(sheets, control_block, target_block, gate_id,
next_lineage_id)`:

1. Finds all active sheets currently assigned to the control block.
2. Deep-copies each active control sheet.
3. Assigns the copy to the target block.
4. Sets:

```text
copied.parent_lineage_id = parent.lineage_id
copied.lineage_id        = fresh_lineage_id
copied.created_by_gate   = gate_id
```

5. Appends the copied sheet to `sheets`.

The parent control sheet is unchanged. Existing target sheets are unchanged.
There is no immediate xor merge and no `nonzeromin` field merge at the gate.

The code asserts that mutable arrays are not aliased between parent and copy
when `check_aliasing=true`.

### Sheet Updates

`update_sheet!` calls the baseline `update!` on one sheet's arrays:

```text
update!(sheet.state_component,
        sheet.state_correction,
        sheet.old_synds,
        sheet.new_synds,
        sheet.hist,
        sheet.hist_correction,
        sheet.fields,
        sheet.new_fields,
        r,p,q,synch,pretty)
```

`update_sheets!` loops over all sheets and updates them independently.

Precise implementation caveat: after a CNOT has created multiple sheets on the
same output block, the current code applies independent physical and measurement
noise to every sheet, because each sheet receives its own `update!` call with
the same `p` and `q`. This is the implemented stochastic model. Do not assume it
shares one physical-noise sample across all sheets assigned to the same block.

### Readout Merge

`merged_decoded_state(sheets, block, L)` computes:

```text
decoded_state = zero
for sheet in sheets assigned to block:
    decoded_state xor= sheet.state_component
    decoded_state xor= sheet.state_correction
```

This is the only algebraic merge between sheets in the sheet-copy prototype.

Cleanup success is:

```text
all_sheet_hists_empty(sheets)
```

If cleanup succeeds, the code asserts that the merged decoded control and target
states are syndrome-free. Logical failure is then checked on the merged decoded
states, with the same "either block fails" rule as primitive CNOT.

Like primitive CNOT, `cleanup_failures` is recorded separately and is not
explicitly ORed into `logical_failure`.

### Sheet-Count Metrics

`estimate_sheetcopy_cnot_Ft` records:

- `sheetcopy_final_sheet_count_mean`
- `sheetcopy_final_active_sheet_count_mean`
- `sheetcopy_max_sheet_count`
- `sheetcopy_max_active_sheet_count`
- `sheetcopy_first_trial_sheet_count_trace`
- `sheetcopy_first_trial_active_sheet_count_trace`

The trace note is:

```text
init, after each pre update, after CNOT, after each post update,
after each cleanup update
```

For a single noisy CNOT between two initially empty blocks, the maximum sheet
count is usually 3 once the control sheet is active: original control sheet,
original target sheet, and copied control-to-target sheet. In zero noise, the
control sheet can remain inactive at the CNOT and no copy is made, so sheet
count remains 2.

### Classical Overhead

A sheet stores essentially one full baseline decoder state. If there are `S`
live sheets, leading memory is:

```text
S * 12 L^2 Z Ints
```

plus bit-packed state, syndrome, history, and correction arrays for each sheet.

For one active CNOT from control to target:

- before gate: 2 sheets total,
- after gate: 3 sheets total,
- target block representation doubles from 1 sheet to 2 sheets.

For repeated CNOTs, sheet count follows the linear update:

```text
n_target <- n_target + n_control
n_control unchanged
```

for each CNOT. Therefore overhead can grow quickly under repeated gates. In the
worst case, alternating CNOTs can create Fibonacci-like growth in lineage count.
At minimum, the overhead is linear in the number of copied active sheets, not
just in the number of logical blocks.

Runtime per decoder round is also proportional to the number of sheets because
`update_sheets!` serially calls the full baseline `update!` once per sheet
inside each trial.

## Space Overhead Summary

Per baseline single-block decoder, the dominant live arrays are the two integer
field buffers:

```text
fields + new_fields = 12 L^2 Z machine Ints
```

The main Boolean arrays total:

```text
state + state_correction + old_synds + new_synds = 6 L^2 bits
hist + hist_correction = 4 L^2 Z bits
```

Julia `BitArray` packs these Boolean arrays, while `Array{Int}` uses machine
integers. On a typical 64-bit Julia build, field buffers dominate memory.

For a single Monte Carlo worker:

- baseline one block: `1 * baseline_block_state`,
- primitive two-block CNOT: `2 * baseline_block_state`,
- sheet-copy one-CNOT run: `S * baseline_block_state`, usually `S=2` or `S=3`
  for the current two-block one-gate protocol.

Threaded scans multiply this by the number of active trial workers, because
each worker allocates independent arrays.

## Decoding Performance Notes

These notes summarize existing scan outputs. They are not theoretical threshold
claims.

### Baseline Memory

The compact baseline `Ft` table in
`results/baseline/ft/thread_test/summary.csv` uses:

```text
qrat=1, r=3, synch=true, logZ=true, T=L, cleanup=2T
```

Representative fidelities:

```text
p=0.015: L=5 0.9577, L=9 0.9556, L=13 0.9575, L=19 0.9539
p=0.016: L=5 0.9524, L=9 0.9435, L=13 0.9373, L=19 0.9313
p=0.017: L=5 0.9390, L=9 0.9217, L=13 0.9133, L=19 0.8837
```

The crossing trend is around `p = 0.015-0.016` for this finite-size scan and
noise model.

The baseline `trel` thread-test summary shows lifetimes growing strongly with
`L` below this region and shrinking or flattening above it. For example:

```text
p=0.012: L=5 179.8, L=9 319.1, L=13 603.1, L=19 1168.2
p=0.018: L=5 49.9,  L=9 51.5,  L=13 55.0,  L=19 53.5
```

### Primitive CNOT

The current split-timing primitive scan
`results/cnot_primitive/full_scan/T∕2_CNOT_T∕2_2T` uses:

```text
qrat=1, r=3, synch=true, logZ=true, T=L,
T_PRE=floor(T/2), T_POST=ceil(T/2), cleanup=2T,
5 repeats, ACC_ERRORS=1000 per repeat
```

Representative averaged `CNOT_Ft` values:

```text
p=0.012: L=5 0.9294, L=9 0.9338, L=13 0.9446, L=19 0.9559
p=0.014: L=5 0.8894, L=9 0.8718, L=13 0.8717, L=19 0.8605
p=0.015: L=5 0.8683, L=9 0.8291, L=13 0.8149, L=19 0.7804
```

The threshold-like crossing is visibly below the baseline memory scan. Target
failures dominate in the primitive data, consistent with the target carrying a
lossy merge of control and target histories. For example, in the split-timing
scan at `p=0.015`, aggregated over five repeats:

```text
L=13: control logical failures 1132, target logical failures 4251
L=19: control logical failures  955, target logical failures 4406
```

### Sheet-Copy CNOT

The current sheet-copy full scan
`results/cnot_sheetcopy/full_scan/T∕2_CNOT_T∕2_2T` uses the same split timing
and noise parameters as the primitive split scan. Representative averaged
`CNOT_Ft` values:

```text
p=0.014: L=5 0.9147, L=9 0.9143, L=13 0.9236, L=19 0.9333
p=0.015: L=5 0.8944, L=9 0.8814, L=13 0.8890, L=19 0.8850
p=0.016: L=5 0.8721, L=9 0.8459, L=13 0.8340, L=19 0.8127
```

This is closer to the baseline crossing trend than primitive CNOT, though it is
not identical because the CNOT experiment has two blocks, one gate, and
sheet-specific noise evolution.

The cost is visible in sheet-count metrics. In the same sheet-copy split scan:

```text
p=0.009, L=5:  final sheet count mean 2.744, active mean 2.677
p=0.011, L=9:  final sheet count mean 3.000, active mean 3.000
p=0.015, L=13: final sheet count mean 3.000, active mean 3.000
p=0.020, L=19: final sheet count mean 3.000, active mean 3.000
```

So for the one-CNOT protocol, sheet-copy usually pays for three full sheets
instead of two once the control sheet is active.

## Validation Hooks

Both CNOT files support:

```text
MODE=CNOT_DEBUG
```

Primitive sanity checks verify:

- `nonzeromin`,
- primitive target xor behavior,
- field nonzero-min behavior,
- `new_fields` clearing,
- zero-noise CNOT success.

Sheet-copy sanity checks verify:

- `nonzeromin`,
- one sheet follows the baseline decoder when no CNOT is applied,
- copied sheets are deep copies with no mutable aliasing,
- lineage ids remain unique,
- merged target readout contains both target and copied control components,
- zero-noise CNOT success.

For implementation changes, run a small deterministic/debug case before any
large scan. Useful environment patterns:

```bash
MODE=CNOT_DEBUG LVAL=3 LOGZ=false julia --threads=1 2d_windowed_cnot_primitive.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false julia --threads=1 2d_windowed_cnot_sheetcopy.jl
MODE=CNOT_Ft LVAL=5 PVAL=0.0 QRAT=0 SAMPS=1 julia --threads=1 2d_windowed_cnot_sheetcopy.jl
```

## Common Pitfalls

- `detect_logical_error` returns logical success, not failure.
- `hist` stores syndrome changes, not the current syndrome.
- `fields == 0` means no message; use `nonzeromin` when merging fields.
- Primitive CNOT clears `new_fields` after the gate. Do not remove this without
  checking stale-buffer effects.
- Sheet-copy CNOT deep-copies only active control sheets.
- Sheet-copy readout merges sheets only at final decoded-state construction.
- Sheet-copy repeated-CNOT overhead can grow much faster than the number of
  logical blocks.
- Current CNOT logical failure does not explicitly include cleanup failure,
  though cleanup failure is recorded.
- Current CNOT prototypes are X-sector only.
- The CNOT files duplicate baseline decoder code. A decoder-rule bug fix may
  need to be mirrored in the baseline, primitive, and sheet-copy files.
- Do not compare scan outputs from different timing directories without
  accounting for `T_PRE`, `T_POST`, and `CLEANUP_TIME`.
- Some timing-directory names use the Unicode division slash in `T∕2`; quote
  these paths in shell commands.
