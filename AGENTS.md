# mp-decoder Agent Guide

Last updated: 2026-07-08

## Scope

This file is for agents working on `projects/mp-decoder`. Treat paths in this
document as relative to the repository root unless otherwise stated.

The project goal is to generalize the local/windowed decoder from quantum memory
to quantum computation, starting with logical CNOT-style dynamics for the toric
code.

## Required References

- Paper notes index: `docs/qec_paper_index.md`
  - Read this first for the cross-paper map, shared glossary, and guidance on
    which reference note to consult for a given decoder task.
- Lake paper notes: `docs/lake_2025_simulated_confinement.md`
  - Future-agent notes for the local/windowed message-passing decoder:
    notation, claims, buffer/back-wall logic, implementation guidance, and
    project-specific cautions.
- DKLP paper notes: `docs/dklp_2001_topological_quantum_memory.md`
  - Future-agent notes for toric-code stabilizers, spacetime syndrome history,
    error/recovery chains, matching baselines, finite windows, and CNOT
    history motivation.
- Main project paper: `papers/2510.08056v3.pdf`
  - This is the Ethan Lake quantum-memory decoder reference for this project.
  - Use the notes above for orientation, then re-check the relevant paper
    sections before changing decoder logic or performance assumptions.
- Toric-code background: `papers/0110143v1.pdf`
  - Use this for general toric-code definitions, logical operators, and
    syndrome/error conventions after consulting the DKLP note.

Do not rely only on comments or prior summaries when modifying core update
rules. Use the Markdown notes as navigation aids, then re-check the relevant paper
sections and the existing implementation.

- Existing implementation documentation: `docs/implementation.md`
  - Read this first for the current data structures, update rules, CNOT
    prototypes, performance notes, and classical-overhead accounting.
  - This is an orientation document, not a replacement for re-checking Julia
    code before changing core decoder behavior.

## Current Code Map

- `projects/mp-decoder/2d_windowed_simulation.jl`
  - Baseline 2D windowed message-passing decoder for quantum memory.
  - This is the main reference implementation for decoder behavior.
- `projects/mp-decoder/2d_windowed_simulation_thread.jl`
  - Parallelized version of the baseline memory simulation.
  - Preserve equivalence with `2d_windowed_simulation.jl` unless intentionally
    changing behavior.
- `projects/mp-decoder/2d_windowed_cnot_primitive.jl`
  - First CNOT extension attempt.
  - Primitive rule: pass control information to the target by XORing the
    control into the target decoder state/history-like data.
- `projects/mp-decoder/2d_windowed_cnot_sheetcopy.jl`
  - Second CNOT extension attempt.
  - Sheet-copy rule: keep an independent copy of the control sheet and merge it
    with the target only at final readout.
- `projects/mp-decoder/2d_windowed_cnot_snapshot.jl`
  - Physical snapshot CNOT prototype.
  - Keeps two observable physical blocks, separate pre-/post-gate decoder
    histories, and `applies_to` correction-routing vectors.
  - Currently supports one synchronous ideal X-sector CNOT.
- `projects/mp-decoder/2d_windowed_history_visualizer.py`
  - History visualization support.
- `projects/mp-decoder/2d_cnot_sheetcopy_visualizer.js`
  - Sheet-copy CNOT visualization support.
- `projects/mp-decoder/notebooks/`
  - Analysis notebooks for baseline, primitive CNOT, and sheet-copy CNOT runs.
- `projects/mp-decoder/jobs/`
  - Batch scripts for scans and repeated runs.
- `projects/mp-decoder/results/`
  - Existing simulation outputs. Treat these as data unless the user explicitly
    asks to regenerate, move, or delete them.

## Progress So Far

1. Baseline quantum-memory decoder exists in `2d_windowed_simulation.jl`.
2. A threaded/parallel baseline exists in `2d_windowed_simulation_thread.jl`.
3. Primitive CNOT prototype exists in `2d_windowed_cnot_primitive.jl`.
   - Advantage: same nominal classical overhead as the baseline.
   - Drawback: lower threshold and worse performance, likely because the target
     does not retain the full useful history information after the CNOT.
4. Sheet-copy CNOT prototype exists in `2d_windowed_cnot_sheetcopy.jl`.
   - Advantage: threshold/performance is comparable to the baseline.
   - Drawback: each CNOT copies sheets and can double classical overhead, so it
     does not scale for circuits with many CNOT gates.
5. Physical snapshot prototype exists in `2d_windowed_cnot_snapshot.jl`.
   - Separates the two noisy/measured physical outputs from old decoder
     histories and deletes old histories after their defects are cleared.
   - Includes `applies_to::BitVector` for future repeated-gate correction
     routing, but the current driver implements only one CNOT.
   - Zero-noise and small noisy regression tests pass; no threshold scan is
     established.
6. Current research task: evaluate and extend a CNOT/computation decoder that keeps
   enough history for good threshold behavior without the sheet-copy overhead
   blowup.

## Working Hypotheses To Preserve

- Primitive and sheet-copy are prototypes, not final designs.
- The right next design should be judged by both decoding performance and
  classical overhead.
- CNOT logic must be explicit about which error sector is implemented. The
  current CNOT prototypes primarily track the X-sector rule where control
  information propagates to the target.
- A proposal that only improves threshold by keeping all copied history is not
  enough unless it also controls memory/runtime overhead across repeated CNOTs.
- A proposal that only matches baseline overhead is not enough unless it keeps
  enough history to avoid the primitive-plan threshold loss.

## General Rules For Agents

- Work inside `projects/mp-decoder` for project code and outputs.
- The user will mostly use Codex for simulation code and new decoder-plan
  proposals. Treat both as first-class project outputs.
- Read the relevant Julia file before editing it. These files contain duplicated
  baseline logic, so small fixes may need to be mirrored deliberately.
- Keep baseline, primitive CNOT, and sheet-copy CNOT behavior comparable. When
  changing one path, state whether the change is algorithmic or only a
  refactor/performance change.
- Do not silently change simulation definitions such as failure criteria,
  logical-error detection, boundary conditions, time ordering, or CNOT timing.
- Do not overwrite existing scan results without explicit permission. Put new
  experiments under a clearly named subdirectory in `results/`.
- Prefer small deterministic/debug runs before large scans. Use small `L`,
  small acceptance counts, and fixed parameters to validate update rules.
- When evaluating a new plan, compare against:
  - baseline memory decoder,
  - primitive CNOT,
  - sheet-copy CNOT.
- Track at least these metrics for new proposals:
  - logical failure rate,
  - threshold trend or crossing behavior,
  - memory overhead per logical block,
  - overhead growth under repeated CNOTs,
  - runtime and thread scaling.
- If a change affects threaded code, verify that serial and threaded behavior
  agree on representative small cases.
- Be careful with Unicode in existing filenames, especially paths containing
  `T/2` rendered with the Unicode division slash. Quote paths in shell commands.
- Leave `.DS_Store` and unrelated dirty files alone.

## Plan Proposal Documentation

When proposing a new decoder plan, create or update a written design document in
Markdown before or alongside implementation. Use `plans/` for these documents
unless the user requests another location.

A useful plan document should be detailed and rigorous enough that another agent
can implement the proposal without reconstructing the argument from chat logs.
It should include:

- motivation and failure mode being addressed,
- precise update rules, including time ordering and CNOT event behavior,
- data structures and invariants for blocks, sheets, histories, fields,
  corrections, and readout state,
- how local decoding proceeds before, during, and after gates,
- final merge/readout rule and logical-failure criterion,
- expected decoding-performance behavior, especially threshold comparison to
  baseline, primitive CNOT, and sheet-copy CNOT,
- classical memory overhead, runtime overhead, and how these scale with number
  of logical blocks, circuit depth, and number of CNOTs,
- minimal validation experiments and scan parameters needed to test the plan,
- known weaknesses, assumptions, and open questions.

Do not describe a new plan only at a high level. The core of each proposal must
be the concrete update rule and the data structure needed to execute it.

## Implementation Documentation

`docs/implementation.md` is the authoritative description of the current implementation.

Whenever a coding task changes the architecture, data structures, physical/error model, decoder semantics, interfaces, execution flow, performance characteristics, supported behavior, limitations, or implementation status, update `docs/implementation.md` in the same task.

The documentation must describe the final implemented behavior, not merely the intended plan. Remove or revise statements that are no longer accurate. Clearly distinguish:

* currently implemented behavior;
* known limitations or approximations;
* proposed but unimplemented work.

Do not update `docs/implementation.md` for formatting changes, comments, tests, or internal refactors that do not affect documented behavior. Before completing a coding task, explicitly check whether the implementation documentation needs revision.


## Response and Documentation Standards

Responses and project documentation should be comprehensive in substance but
precise, concise, and non-repetitive in presentation.

* Aim to give a complete, well-synthesized answer in one pass. Before writing,
  identify the full set of relevant issues, dependencies, assumptions, and
  tradeoffs so that the answer does not require repeated clarification or
  rediscovery of the same points.
* Organize explanations in causal and implementation order rather than in the
  order ideas were discovered. For decoder proposals, normally use:

  1. problem or failure mode,
  2. design objective,
  3. proposed state representation,
  4. update rules and time ordering,
  5. CNOT behavior,
  6. readout and failure criterion,
  7. complexity and scaling,
  8. validation plan,
  9. limitations and unresolved questions.
* State each substantive point once. Do not restate the same conclusion in
  multiple sections using slightly different wording. Later sections should
  refer back to the established definition or invariant instead of explaining
  it again.
* Distinguish clearly between:

  * facts established by the papers or code,
  * conclusions inferred from those facts,
  * working hypotheses,
  * proposed design choices,
  * unresolved questions requiring experiments or further derivation.
* Do not present speculation as an established explanation. When the cause of an
  observed threshold or performance difference is uncertain, state the evidence
  and the remaining uncertainty explicitly.
* Prefer concrete rules, invariants, equations, pseudocode, data layouts, and
  scaling expressions over broad qualitative descriptions.
* Define terminology once and use it consistently. Do not introduce multiple
  names for the same sheet, history object, label, field, correction, or CNOT
  event unless the distinction is necessary.
* Use concise sentences and compact paragraphs. Remove conversational filler,
  repeated motivation, generic background, and conclusions that merely repeat
  the preceding section.
* Include all material details needed for implementation or evaluation, but
  omit details that do not affect decoder behavior, correctness, complexity, or
  experimental interpretation.
* When several alternatives exist, compare them using the same criteria in one
  structured comparison rather than discussing each alternative independently
  and repeating the criteria.
* When revising an existing document, consolidate duplicated discussion and
  maintain one canonical description of each rule or design choice.
* Do not end with a generic summary that repeats the document. End with only the
  concrete decision, implementation steps, validation requirements, or open
  questions that remain actionable.

### Standards for Technical Explanations

For questions about code, papers, or decoder behavior:

* Begin with the direct answer or central mechanism.
* Explain the minimum necessary background before using it.
* Follow the actual causal chain or execution order.
* Connect mathematical statements to the corresponding code state or physical
  interpretation where relevant.
* Use examples only when they resolve a specific ambiguity; do not repeat the
  full explanation through multiple analogous examples.
* Explicitly identify any mismatch between the user's proposed interpretation,
  the paper, and the implementation.

### Standards for Design Proposals

A proposed decoder plan must form one internally consistent design, not a list
of loosely connected ideas.

* Specify one canonical state model and one canonical update sequence.
* Show how the proposed mechanism addresses the identified failure mode.
* Account for all stored information and all operations contributing to memory
  and runtime overhead.
* Give scaling in terms of relevant quantities such as lattice size, window
  depth, number of logical blocks, circuit depth, number of CNOT events, and
  number of active dependencies.
* Explain what information is preserved, compressed, merged, or discarded and
  why discarding it should not invalidate decoding.
* Separate required components from optional optimizations.
* Avoid presenting minor variations of the same mechanism as independent plans.
  Combine them into one plan with clearly labeled implementation choices.
* State the strongest expected advantage, the principal risk, and the experiment
  that would most directly falsify the proposal.

## Suggested Next Work

1. Read `docs/qec_paper_index.md` and
   `docs/lake_2025_simulated_confinement.md`, then re-read
   `papers/2510.08056v3.pdf` around the memory decoder assumptions and identify
   exactly which history information primitive CNOT loses.
2. Make a rigorous Markdown design note for a third CNOT plan before coding
   it.
3. Define the intended data structure invariants:
   - what each sheet/block stores,
   - what is copied or compressed at a CNOT,
   - what is merged immediately,
   - what is deferred until readout,
   - how old history is discarded or summarized.
4. Implement the smallest possible prototype by copying the current CNOT driver
   pattern only where necessary.
5. Validate on zero-noise and small noisy cases before running threshold scans.
6. Add analysis notebook or script output that directly compares primitive,
   sheet-copy, and the new plan at the same `L`, `p`, `T`, and cleanup schedule.

## Open Design Target

Find a CNOT decoder strategy with:

- threshold close to the memory baseline or sheet-copy result,
- classical overhead close to baseline or at least sublinear in the number of
  CNOTs,
- local update rules compatible with the windowed message-passing decoder,
- clear handling of history information at CNOT events,
- a path toward extending beyond a single X-sector demonstration.
