# Block-Level CNOT Refactor Report

Date: 2026-07-15

## Scope and Changed Files

The refactor changes the CNOT implementation from independently evolving
decoder sheets to two physical decoder blocks. It retains the X sector, ideal
instantaneous CNOT, toric-code geometry, synchronous and asynchronous local
message passing, Monte Carlo stopping modes, logical-failure criterion, and
per-block failure statistics.

The block implementation changes are in:

- `2d_windowed_cnot_block.jl`: block state, round updates, CNOT boundary,
  cleanup, readout, Monte Carlo driver, diagnostics, debug mode, and demo data;
- `test/runtests.jl`: deterministic and fixed-seed regression tests;
- `2d_cnot_block_visualizer.js`: two-block visualization and ancestry
  display;
- `docs/implementation.md`: authoritative description of the implemented
  architecture and limitations.

The modified Julia implementation was saved under the new `_block.jl` name.
`2d_windowed_cnot_sheetcopy.jl` and its paired
`2d_cnot_sheetcopy_visualizer.js` were restored to their pre-refactor versions
so the legacy sheet-copy comparison remains available. The baseline-memory and
primitive-CNOT Julia files were not modified.

## Problems Removed

The former `DecoderSheet` representation treated the old-target and
copied-control contributions as independently evolving decoder systems. After
the CNOT, `update_sheets!` called the complete noisy `update!` once per sheet.
The represented target therefore received multiple independent Bernoulli
physical-noise masks and multiple independent measurement-noise masks per
round. Its effective noise rate depended on the number of target lineages.

Each target sheet also kept private syndrome registers, history, message
fields, correction proposals, and Pauli-frame correction. The hidden
old-target and copied-control contributions were decoded separately and XORed
only at readout. This exposed a decomposition that cannot be measured on the
physical combined target: cancelling defects could still drive separate
corrections, and the combined target correction could differ from the
correction selected from its observable syndrome.

## Major Code Changes

### 1. Sheet API Replaced by a Block API

The following sheet-level types and entry points were removed from
`2d_windowed_cnot_block.jl` rather than retained as alternate execution paths.
They remain present only in the restored legacy sheet-copy implementation:

- `DecoderSheet` and `make_decoder_sheet`;
- `initial_sheet_set`;
- `apply_cnot_x_sheetcopy!`;
- `update_sheet!` and `update_sheets!`;
- `merged_decoded_state`;
- sheet-activity, mutable-alias, and sheet-count helpers;
- `estimate_sheetcopy_cnot_Ft`.

Their block-level replacements are:

| New code | Responsibility |
| --- | --- |
| `PhysicalBlockState` | Owns one `L x L x 2` physical X-error array and one tracked Pauli-frame array. |
| `BlockDecoderState` | Owns one observable decoder state: `old_synds`, `new_synds`, `hist`, `fields`, `new_fields`, and `hist_correction`. |
| `AncestryRecord` | Stores identifiers only: lineage ID, origin block, parent lineage, and creating gate. |
| `DecoderBlock` | Groups one block ID, one physical state, one decoder state, and optional ancestry records. |
| `make_decoder_block` / `initial_block_set` | Construct one block or the exactly two blocks used by this prototype. |
| `decoder_block` | Requires exactly one block with a requested ID. |
| `update_block!` / `update_blocks!` | Advance one physical block or both physical blocks. |
| `decoded_state` | Returns `physical.errors xor physical.pauli_frame` for one block. |
| `estimate_block_cnot_Ft` | Runs the two-block CNOT Monte Carlo experiment. |

`initial_block_set` always creates one control and one target. A CNOT does not
append a physical or decoder object. `apply_cnot_x_blocks!` rejects identical
control/target IDs and rejects block sets whose length is not two.

Ancestry is deliberately outside both physical and decoder state. Update,
measurement, correction, cleanup, logical testing, and random-number sampling
never inspect ancestry. A CNOT may append ancestry records for diagnostics, but
record count cannot change the simulated dynamics.

### 2. One Observable Round Helper Added

`sample_bernoulli_mask`, `resolve_round_mask`, and
`advance_observable_round!` centralize the stochastic part of a physical
round. For each call on one block, `advance_observable_round!` executes:

```text
noise_mask       <- one L x L x 2 Bernoulli(p) sample
measurement_mask <- one L x L Bernoulli(q) sample
errors           <- errors xor noise_mask
old_synds        <- new_synds
new_synds        <- syndrome(errors) xor measurement_mask
cycle hist and fields by one window layer
hist[:, :, 1]    <- old_synds xor new_synds
```

Probabilities are checked to lie in `[0,1]`. Optional Boolean noise and
measurement masks are shape-checked and can replace random sampling. This mask
injection is used by deterministic tests and exposes the masks returned from a
round without introducing a second update path.

`update!`, `update_block!`, and `update_blocks!` now accept an explicit random
number generator, defaulting to Julia's task-local generator. Decoder schedule
choices, stochastic back-wall moves, physical masks, and measurement masks all
use that generator, which makes fixed-seed block-level regressions possible.

`update_block!` passes exactly one physical state and one decoder state to
`update!`. `update_blocks!` calls it once for the control and once for the
target and optionally accepts dictionaries of injected masks keyed by block
ID. There is no loop over ancestry records.

### 3. Synchronous and Asynchronous Scheduling Updated

The synchronous decoder retains its prior causal order within a call:

1. update message fields;
2. select and apply history corrections, accumulating spatial corrections in
   the Pauli frame;
3. call `advance_observable_round!` once to add physical noise, acquire the
   block syndrome, cycle the window, and insert the new front layer;
4. perform the optional `pretty` field refresh used by animations.

For `p=0` or `q=0`, the corresponding sampled mask is identically zero, so
cleanup cannot acquire stochastic errors.

The asynchronous path still performs `(r+1)L^2` randomly selected local
field/feedback attempts and updates corrections locally. Its physical
acquisition schedule was intentionally changed: per-feedback single-edge noise
attempts and selected-column measurement/history cycling were removed, and
`advance_observable_round!` is called once after the local microsteps. The
asynchronous correction scratch buffer is also cleared at the start of the
round. Thus `p` and `q` describe a physical round independently of how many
feedback events were selected. Because `update!` is shared inside this file,
this asynchronous timing correction also applies to its non-CNOT modes; the
separate baseline-memory files remain unchanged.

### 4. CNOT Boundary Made Block-Level

`apply_cnot_x_blocks!` performs the ideal X-sector propagation once at the
gate event:

```text
T.errors       <- T.errors xor C.errors
T.pauli_frame  <- T.pauli_frame xor C.pauli_frame
T.old_synds    <- T.old_synds xor C.old_synds
T.new_synds    <- T.new_synds xor C.new_synds
T.hist         <- T.hist xor C.hist
```

Every control array remains unchanged. The first two assignments implement
the physical X-error and tracked-frame propagation rules. The syndrome
registers and residual history are aligned binary stabilizer quantities, so
their target values follow the same XOR algebra. The current driver advances
both blocks through the same number of outer rounds before the gate; no
unaligned-history merge is supported.

`fields` and `new_fields` are integer-valued nonlinear message caches, not
binary observables. They are therefore zeroed on the target instead of being
XORed, concatenated, minimized, or retained per lineage. The target
`hist_correction` scratch array is cleared because its proposals have already
been consumed. Subsequent field updates rebuild one message landscape from the
combined target history.

For each control ancestry record, the function adds a target metadata record
with a fresh lineage ID, preserved origin, parent lineage ID, and gate ID. It
copies no physical or decoder arrays. Post-gate control errors remain local to
the control because propagation occurs only when this function is called
again; there is no shared array alias or continuously updated copied state.

No measurement is synthesized at the instantaneous gate. The next post-CNOT
round measures the already-combined target state through its single target
decoder stream.

### 5. Pre-CNOT, Post-CNOT, Cleanup, and Readout Loops Rewritten

Each Monte Carlo trial now follows one block-level lifecycle:

```text
create control block and target block
repeat T_PRE:  update both blocks with p and q
apply one block-level CNOT
repeat T_POST: update both blocks with p and q
repeat cleanup: update both blocks synchronously with p=q=0
read out and test the two decoded block states
```

Cleanup stops early only when both block histories are empty. On successful
cleanup, the code asserts that both decoded states are syndrome-free. Cleanup
failure remains a separately reported statistic; the existing logical-failure
criterion remains the OR of control and target logical failures.

Readout no longer collects or XORs sheets. It constructs exactly:

```text
decoded_control = C.errors xor C.pauli_frame
decoded_target  = T.errors xor T.pauli_frame
```

The same existing toric-code logical-error test is then applied independently
to those two arrays.

### 6. Monte Carlo Interface and Output Updated

`estimate_block_cnot_Ft` preserves fixed-sample mode, accumulate-until-failure
mode, trial-level threading, and these meaningful output keys:

- `CNOT_Ft`, `CNOT_fail_rate`, `trials`, and `logical_failures`;
- `control_logical_failures`, `target_logical_failures`, and
  `both_logical_failures`;
- `cleanup_failures`.

The sheet-dependent outputs were removed:

- `sheetcopy_final_sheet_count_mean`;
- `sheetcopy_final_active_sheet_count_mean`;
- `sheetcopy_max_sheet_count` and `sheetcopy_max_active_sheet_count`;
- the sheet-count traces and trace note.

They were replaced with diagnostics whose ownership is explicit:

- `ancestry_final_record_count_mean`, `ancestry_max_record_count`, and
  `ancestry_first_trial_record_count_trace` describe metadata only;
- `physical_block_count` and `physical_block_count_trace` verify that the
  simulator remains at two blocks;
- `block_decoder_memory_bytes` measures the two `BlockDecoderState` objects;
- `block_physical_memory_bytes` measures the two `PhysicalBlockState` objects;
- `block_diagnostic_trace_note`, `cnot_state_model`, and
  `cnot_boundary_field_policy` document trace timing and semantics in each
  output file.

The memory-byte diagnostics intentionally exclude ancestry metadata; ancestry
growth is reported by record count instead. Saved pre-refactor sheet-copy scans
remain historical data for the removed stochastic model and are not comparable
performance results for this implementation.

`CNOT_STYLE` now defaults to `"block"`. The legacy value `"sheetcopy"` is
accepted only as a command-line/environment alias and is normalized to
`"block"`; it does not select the old algorithm. New CNOT result filenames use
the block style, reducing the risk of mixing new runs with legacy scans.

### 7. Debug, Demo, and Import Behavior Updated

`CNOT_DEBUG` now checks block CNOT algebra, field reset, ancestry metadata,
absence of continuous propagation, one injected measurement stream, zero-noise
Monte Carlo behavior, and the invariant of two physical blocks.

The demo generator now serializes `window.BLOCK_CNOT_DEMO` frames containing:

- one control block and one combined target block;
- physical errors, Pauli frame, decoded state, syndrome, history, and field
  summaries for each block;
- physical-block and ancestry-record counts;
- ancestry records labeled by their current and origin blocks.

`2d_cnot_block_visualizer.js` uses two physical-block panels. It does not
search for a hidden copied-control sheet or construct a target only for
display, and it requires the new `BLOCK_CNOT_DEMO` payload. The restored
`2d_cnot_sheetcopy_visualizer.js` continues to consume the legacy
`SHEETCOPY_CNOT_DEMO` payload.

The Julia entry point is guarded by:

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

This permits `test/runtests.jl` to include the simulator and call its functions
without launching a Monte Carlo job or writing an output file.

### 8. Regression Suite Added

`test/runtests.jl` adds focused coverage for:

- ideal CNOT propagation of errors and the Pauli frame, preservation of the
  control, XOR of aligned binary decoder state, and reset of nonlinear target
  fields;
- no propagation of an error added to the control after a CNOT until a second
  CNOT is explicitly applied;
- one target physical-noise channel using both a deterministic mask and a
  2,500-sample fixed-seed Bernoulli regression;
- one target measurement mask and the absence of syndrome fields in
  `AncestryRecord`;
- exact cancellation of identical hidden contributions in the combined target
  state, syndrome, and history;
- a four-defect configuration for which the combined decoder selects a
  different correction from the XOR of two hypothetical component decoders;
- ancestry-count invariance of sampled masks, syndrome state, histories,
  fields, corrections, cleanup, and decoded output in both synchronous and
  asynchronous modes;
- two-block, `p=q=0` cleanup with no injected noise or measurement masks.

## Assumptions and Remaining Limitations

The implementation remains an X-sector-only, ideal-instantaneous-CNOT
prototype on periodic toric-code geometry. It does not implement the Z sector,
a noisy CNOT channel, circuit-level syndrome extraction, hook errors, more than
two physical blocks, or unaligned gate histories. The physical and decoder
state size is independent of ancestry count, although optional ancestry
metadata can still grow across repeated gate records.

The unresolved conceptual choice is optimal initialization of nonlinear target
message fields at a gate. Resetting them is conservative and prevents an
unjustified oracle merge while retaining the observable binary history, but it
discards pre-gate field convergence and may reduce decoding performance. A
better initialization requires an explicit gate-aware derivation and new
performance scans; no unproved merge rule is implemented here.

## Validation Results

- `julia --startup-file=no test/runtests.jl`: 244/244 assertions passed.
- `julia --startup-file=no --check-bounds=yes test/runtests.jl`: 244/244
  assertions passed.
- `MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2`: passed.
- Synchronous smoke, `L=5`, `Z=4`, `T=4`, `p=q=0.02`, 20 trials: completed
  with two physical blocks and zero cleanup failures.
- Asynchronous smoke with the same parameters: completed with two physical
  blocks and zero cleanup failures.
- Two-thread fixed-sample smoke, `L=5`, `T=2`, four trials: completed.
- `CNOT_DEMO` generation smoke: completed with the new two-block payload.
- Restored `2d_windowed_cnot_sheetcopy.jl` `CNOT_DEBUG` smoke: passed with the
  legacy `sheetcopy` style.
- `git diff --check`: passed.
