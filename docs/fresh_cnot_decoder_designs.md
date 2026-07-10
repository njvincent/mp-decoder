# Fresh design-space investigation for an X-sector CNOT decoder

## Scope, source basis, and headline result

This document addresses only the ideal X-sector logical rule

```text
control_out = control
target_out  = target xor control
```

for the existing toric-code, windowed message-passing decoder. It does not design a Z-sector decoder, lattice surgery, a complete computation decoder, or a physical CNOT fault model. Those are future projects.

The investigation uses only `agent.md`, `implementation.md`, the four Julia implementations named in the task, the three paper notes cited by `agent.md`, and the two cited papers. The important paper locations are Lake, *Local active error correction from simulated confinement*, especially PDF pp. 3-4, 7-8, 14-15, and 22, and Dennis-Kitaev-Landahl-Preskill (DKLP), *Topological quantum memory*, especially PDF pp. 12, 21-23, and 31. Two equation-number mismatches in the local DKLP note should be corrected when citing the paper itself: the homology-class posterior is Eq. (10), and Pauli propagation through CNOT is Eq. (105). The computation-specific combined-chain rule used here is Eq. (72).

The main conclusion is conditional:

> Exact memory independent of total CNOT depth requires either a factorized continuation computation or terminal reduction of old unresolved states. Under the online resident-state model analyzed here—without an external replay tape and with independently reachable, continuation-distinguishable sheet states—the literal independently randomized sheet-copy process cannot guarantee `O(B * M_block)`. The cited papers do not prove a more general impossibility theorem.

The reason is not the linear CNOT rule. The CNOT dependency map is compressible. The obstruction is the nonlinear, stateful decoder update: min-plus message propagation, minimum selection, fixed tie priorities, and stochastic back-wall motion do not commute with XOR. An exact design must either retain every distinguishable live nonlinear state, replay it, or replace the decoder with a joint inference rule whose own live separator width is bounded.

The most useful near-term design is the **incidence-vector epoch transducer**. It stores every unique live decoder payload once and represents its current set of destination blocks by a bit vector. It removes path enumeration, samples noise once per physical block, and returns close to baseline memory after old epochs drain. It does not promise an unconditional `D`-independent worst-case bound.

The strongest depth-overhead research direction is the **rolling CNOT factor graph with overlapping-window elimination**. It replaces path count by a fixed live window, but generic exact inference is exponential in the **full spatial-spacetime** induced width, not just CNOT causal width, and a DKLP-like accurate half-window can require `Z_FG >> L`. It is therefore a very-small-system research prototype, not yet a practical bounded-space result in `L`.

## 1. Baseline semantics and cost model

### 1.1 State at a completed decoder-round boundary

The existing synchronous decoder is a deterministic/stochastic transducer once its random masks are supplied. At a completed round boundary, the allocated arrays divide as follows.

| object | shape | role | must persist for exact continuation? |
| --- | ---: | --- | --- |
| `state` | `L x L x 2` bits | raw physical X chain; future ideal syndrome is its boundary | yes, or an exact boundary-plus-homology quotient |
| `state_correction` | `L x L x 2` bits | accumulated recovery/Pauli-frame chain | yes for readout, or an exact quotient |
| `old_synds` | `L x L` bits | temporary previous-measurement register | no; the next update overwrites it from `new_synds` before use |
| `new_synds` | `L x L` bits | latest measured syndrome contribution | yes |
| `hist` | `L x L x Z` bits | unresolved syndrome-change events | yes |
| `hist_correction` | `L x L x Z x 3` bits | current feedback proposal | no; cleared/recomputed each synchronous step |
| `fields` | `L x L x Z x 3 x 2` integers | live six-direction min-plus message state | yes |
| `new_fields` | same as `fields` | destination scratch for one field sweep | no; overwrite-only scratch |

The minimum full-array Markov state is therefore

```julia
(state, state_correction, new_synds, hist, fields)
```

plus RNG/update-phase state and a reusable workspace. `new_fields`, `hist_correction`, and `old_synds` can be pooled per serial updater or worker rather than per lineage.

The synchronous update order is significant:

```text
r field sweeps
clear and compute feedback proposals
accumulate spatial feedback into state_correction
toggle feedback endpoints in hist
sample one data-noise mask
compute a new measured syndrome, with one measurement-noise mask
cycle hist and fields toward the back wall
insert old_syndrome xor new_syndrome at k = 1
```

Bulk feedback priority is `+z, -x, -y, +y, +x`. Back-wall feedback is spatial only, uses priority `-x, -y, +y, +x`, and is accepted with probability `0.8`. The RG cycle XORs the penultimate defect slice into the persistent back wall, preserves the nonzero minimum of old and incoming spatial back-wall fields, shifts the middle slices, and clears the front slice.

The current `hist` does **not** determine the current `fields`. A message can remain in flight after its source moved or annihilated, and spatial back-wall fields persist. Reconstructing fields exactly requires a checkpoint plus every intervening defect configuration, correction decision, RG shift, CNOT operation, and back-wall random choice. Recomputing a static distance transform from current defects changes the decoder.

### 1.2 A logically exact quotient of the edge arrays

For the present X-sector simulation, online feedback never reads the detailed edge pattern in `state_correction`; syndrome measurement reads `state` only through `get_synds(state)`, and terminal logical readout uses two cut parities. Consequently, an edge chain can be represented, up to stabilizer equivalence, by

```julia
mutable struct ChainQuotient
    boundary::BitMatrix          # L x L syndrome boundary
    winding::BitVector           # exactly two bits: x and y
end

function xor_quotient!(dst::ChainQuotient, src::ChainQuotient)
    dst.boundary .⊻= src.boundary
    dst.winding .⊻= src.winding
    return dst
end
```

Each sampled edge toggle updates two boundary bits and, if it crosses a chosen cut, one winding bit. Each spatial recovery link does the same to the correction quotient. A deterministic spanning-tree solver can reconstruct a canonical edge array with the stored boundary and winding. The reconstructed array need not be bitwise equal to the original; their difference is a stabilizer boundary. It gives the same syndrome and logical-failure result.

This quotient is exact for the observable semantics of the current simulations. It would not be sufficient if a later physical model made noise rates depend on the microscopic edge configuration or required bitwise recovery-chain output.

### 1.3 Baseline memory and runtime

Let `s_I = sizeof(Int)`, currently 8 bytes. One allocated baseline block has

```text
fields + new_fields = 12 L^2 Z machine integers
all BitArrays        = 4 L^2 Z + 6 L^2 bits
```

so, ignoring headers and chunk rounding,

```text
M_block = 12 s_I L^2 Z + (4 L^2 Z + 6 L^2)/8 bytes
        = 96.5 L^2 Z + 0.75 L^2 bytes       when s_I = 8.
```

With `Z = ceil(log_{1.5} L)`, this is `Theta(L^2 log L)` bytes. With `Z = ceil(L/4)`, it is `Theta(L^3)` bytes.

A persistent full-field lineage with pooled scratch needs only one dense field buffer. With the chain quotient above, its leading payload is

```text
M_epoch = 6 s_I L^2 Z + (L^2 Z + 3 L^2 + O(1))/8 bytes
        = 48.125 L^2 Z + 0.375 L^2 + O(1) bytes.
```

One shared updater workspace adds approximately another `48 L^2 Z` bytes plus packed correction scratch. This constant-factor saving does not solve growth in the number of distinct live states.

One field-site update inspects 54 constant-size candidates. With fixed `r`, one synchronous decoder round costs

```text
U_block = Theta((r + 1) L^2 Z).
```

Cleanup of `C` ideal rounds costs `Theta(C (r+1) L^2 Z)` per independently updated state. The usual `C=2L` cap gives `Theta((r+1)L^3 Z)` per state in the worst completed trial.

All memory bounds below are per live Monte Carlo trial/worker. Trial-level threading multiplies resident memory by the number of simultaneously active workers and does not change the within-trial CNOT-depth scaling.

Let `W` be the machine-word bit width and `s_B=ceil(B/W)`. In word-count formulas, `B/W` is shorthand for `s_B`, and `B^2/W` means `B*s_B`; this keeps the mandatory one-word minimum for small `B`. Information-level bit bounds explicitly labeled “bits” do not include per-vector word rounding or headers.

## 2. What must survive a CNOT

This section answers the information questions before introducing algorithms.

### 2.1 Information from the control that can affect the target

At the physical-chain level, an ideal X-sector CNOT propagates the control's pre-gate X-chain into the target. DKLP Eq. (72) expresses the target decoding problem as the mod-2 combination of control-before, target-before, and target-after syndrome/error chains. Therefore the target can depend on:

1. the unresolved pre-CNOT control defect boundary;
2. the control recovery/Pauli-frame contribution already accumulated;
3. the control chain's two homology parities;
4. the control's latest syndrome contribution needed to close measurement-error world lines consistently;
5. for exact continuation of this particular Lake-like automaton, the control's live `hist`, message field, back-wall content, update phase, and future random-input identity;
6. a causal support label saying which current blocks receive that contribution after subsequent CNOTs.

Post-gate control noise does not propagate backward through that already completed CNOT. It must enter a new control-local causal epoch.

### 2.2 Logically necessary versus cached state

In DKLP's stated overlapping-recovery procedure, the relative boundary is sufficient to construct the current minimum-weight `E'` and to carry `E'_keep`, given fixed window geometry, edge weights, boundary conditions, and a deterministic tie rule. Old chain interiors can be erased after the chosen `E'_old` projection and homology have been committed. This is not a universal sufficient statistic for the optimal summed homology posterior or for exact infinite-history decoding.

For exact continuation of the current message-passing automaton, more is operationally necessary:

| category | contents |
| --- | --- |
| logically necessary evidence | unresolved defect boundary, current syndrome contribution, physical/correction boundary and winding, causal block support |
| live computational state | exact `fields`, including back-wall messages; update phase and stochastic-choice stream |
| reconstructible only by replay | message fields and correction trajectory from an earlier complete checkpoint plus all later inputs/choices |
| disposable scratch | `old_synds` at an inter-round boundary, `hist_correction`, `new_fields` |
| disposable after terminal commitment | contractible closed-chain interior; a frozen component's fields after its final boundary/homology is folded into the block frame |

The key distinction is that “cached” does not mean “safe to drop.” `fields` is cached computation, but its exact value changes the next correction.

### 2.3 What must remain distinct

Two propagated histories must remain distinct whenever any of the following differ:

- unresolved `hist`;
- message field or back-wall state;
- current measured-syndrome contribution;
- physical or accumulated-correction boundary/homology;
- current destination support or future CNOT routing;
- future injected noise/measurement input;
- future back-wall/tie-breaking random choices.

Packing two objects in the same sparse container is harmless if labels remain. XORing their defects and keeping a componentwise minimum field is not: coincident defects can cancel while ghost messages remain, and noncoincident defects begin interacting through a field belonging to neither independent decoder.

### 2.4 Exact merge criterion

The exact criterion is continuation equivalence. Let `Q(x,u)` be the terminal decoded contribution obtained from live state `x` under every admissible future input sequence `u` (noise masks, CNOT schedule, and stochastic choices). States `x` and `y` are exactly mergeable only if replacing them by one state preserves every block output for every `u` allowed by the promised semantics.

Practical sufficient conditions are:

1. **Bytewise/full-state equality plus coupled future inputs.** Equal `hist`, `fields`, syndrome, chain quotients, phase, support, and random stream remain equal.
2. **A proved automaton congruence.** Equality of a smaller canonical signature is sufficient only after proving that all transitions and outputs respect it.
3. **Terminal XOR.** Frozen syndrome-free contributions can be reduced to winding parities and XORed into each supported block. Equal terminal contributions cancel by parity.
4. **Exact structural sharing.** Identical immutable prefixes may share storage while retaining separate logical identities; copy-on-write occurs at divergence.
5. **Identical finite-window summaries.** For the specified DKLP-style minimum-weight overlapping procedure, equal relative boundary, fixed weights/geometry/tie context, and committed frame are mergeable within that procedure's semantics.

Collision-checked hashing can find condition 1. Hash equality alone is insufficient. Equal defect bitmaps with different fields are not mergeable. Disjoint current causal cones are not mergeable if later fields can meet; they may merely be stored sparsely in one labeled container.

### 2.5 Can future corrections omit the full field?

There are three honest answers:

- **Current automaton, no replay:** no.
- **Current automaton, exact checkpoint/replay:** yes, by paying replay time and retaining all causal inputs/choices since the checkpoint.
- **A specified global/windowed minimum-weight decoder:** yes; the carried relative boundary can be sufficient for DKLP's overlapping procedure under its fixed geometry/weights/tie rule, but this is a different decoder rather than compression of a live Lake sheet.

### 2.6 Literal sheet-copy noise versus block noise

The current sheet-copy implementation calls the complete noisy `update!` independently on every sheet. If `n` sheets occupy a block, even the raw parity of independent data-noise bits has effective probability

```text
p_eff(n) = (1 - (1 - 2p)^n)/2,
```

which approaches `1/2` as `n` grows. This is exact implemented behavior, but it is not a one-physical-block noise model.

The first, deliberately simplified benchmark for new designs is **shared-recovery block-noise epoch semantics**:

1. sample one data mask and one measurement mask per physical block per round;
2. route that new block-local information into one unit-support native decoder epoch;
3. evolve distinct source epochs separately and without duplicating the physical noise;
4. evolve one multi-destination epoch only once, with one correction state and one back-wall random stream shared by all blocks in its support.

Item 4 means descendants of one source do **not** decode independently as literal sheets do. Splitting on destination-local feedback/randomness would create divergent epochs and can restore growth in `A(t)`. This family is therefore adjacent to shared-recovery/correction-forwarding approaches, even though the support algebra eliminates explicit causal paths.

Routing hidden sampled `eta_p` and `eta_q` to a chosen native epoch is also a simulator-oracle decomposition: an online decoder observes only the total measured syndrome, not a label saying which hidden fault belongs to which epoch. Every epoch proposal below must therefore report two modes:

- **oracle-component benchmark:** hidden masks/components and virtual per-component measurement registers are routed as above; after a CNOT their XOR need not equal the actual block's previous measurement register;
- **total-syndrome-only mode:** decoder code receives only the block's total `old_meas xor new_meas` and cannot inspect `eta_p`, `eta_q`, raw component boundaries, or fault labels.

How total observed events are assigned or jointly decoded is part of the heuristic. This factorization is not a theorem in either paper. Whenever “exact relative to sheet-copy semantics” appears, it means the literal implemented per-sheet process unless explicitly qualified.

### 2.7 Depth independence and lower bounds

The raw CNOT dependency graph is linear. If `R` maps origin block contributions to current blocks, a CNOT `c -> t` is the row operation

```text
R[t, :] xor= R[c, :].
```

A dense `B x B` bit matrix stores the current transform in `Theta(B^2)` bits, and one row update costs `Theta(B/W)` word operations for word width `W`. Gate history need not be stored. Since

```text
|GL(B,2)| = product_{i=0}^{B-1} (2^B - 2^i),
```

an arbitrary invertible current dependency transform itself requires `B^2 + O(1)` bits in the worst case if the gate list is not retained and replayed. This is a width cost, not a depth cost.

The following are original conditional counting arguments, not lower bounds proved by DKLP or Lake. They assume an online resident-state representation without an externally retained circuit/noise tape; checkpoint replay changes space into time.

- If `A(t)` live histories contain independent continuation-distinguishable binary obligations, resident exact state needs `Omega(A(t))` bits.
- If each such obligation can independently realize an arbitrary support subset of `B` blocks, an unfactored support representation needs `Omega(A(t)B)` bits; correlated circuit structure can reduce it.
- If independently reachable per-lineage defect patterns over `Theta(L^2Z)` positions remain continuation-distinguishable, the literal independent-noise model needs `Omega(A(t)L^2Z)` bits up to parity constraints. This reachability/distinguishability premise must be tested, not assumed for every state.
- `Theta(A(t)L^2Z log V)` is the cost of the explicit dense integer-field representation with value range `V`; it is an upper/storage cost, not a proved information lower bound on all exact encodings.

Therefore this analysis does not justify an exact arbitrary-schedule `O(B*M_block)` promise under the current online rules unless one proves a bound on `A(t)`, a smaller transition congruence, or a factorized/joint decoder. An external replay tape can reduce resident state while increasing retained external history/runtime and must be costed separately. Lake's back wall has no deterministic drain deadline; `O(L)` idle time is a useful typical-case condition below threshold, not an exact guarantee.

The appropriate exact target is

```text
O(B * M_block + live_extra(A(t), active defects, causal width)),
```

with garbage collection returning toward `O(B*M_block)` after histories genuinely drain. A stronger unconditional bound must expose either a schedule restriction, an inference-width restriction, or an approximation/overflow event.

## 3. Classification of exploratory directions

| direction | classification | reason |
| --- | --- | --- |
| sparse causal DAG of raw XOR dependencies | exact | CNOT propagation is linear; the DAG must still retain live nonlinear decoder payloads or deferred computations |
| immutable structural sharing | exact | safe until descendants mutate differently; alone it does not bound the number of divergent states |
| `B x B` circuit dependency matrix | exact for physical/syndrome propagation | insufficient by itself for nonlinear independent decoder updates |
| correction obligations or relative boundary only | exact for a DKLP finite-window transition; approximate for a live Lake sheet | the current message field is absent from the summary |
| reconstruct fields from current sparse defects | not compatible with the current decoder | finite-speed and back-wall fields contain additional causal state |
| reconstruct fields from checkpoint plus replay | exact | requires all later inputs/choices and trades memory for runtime |
| lazy/on-demand fields | exact only when implemented as causal replay; otherwise approximate | an equilibrium field is not the current field |
| deferred linear CNOT transformation | exact for gate propagation | decoder feedback between gate layers remains nonlinear |
| persistent functional decoder states | exact | useful for fanout and no-update gate bursts; dense mutation destroys sharing |
| exact garbage collection | exact under terminal conditions | requires frozen input, empty unresolved history, and committed boundary/homology; `hist==0` alone is unsafe for a reusable native state |
| canonicalization/hashing | exact only with full equality or a proved congruence | hash collisions and unequal fields invalidate weaker tests |
| sparse union-find/event graph | approximate relative to the current decoder; exact only under a separately stated graph-decoder objective | it changes the local message-passing rule |
| junction tree/dynamic programming | exact under bounded induced width | time and memory are exponential in live separator width |
| tensor network or decision diagram | exact if untruncated | bond/node count can grow exponentially; truncation is approximate |
| one shared physical-noise state | exact for the proposed block-noise semantics | deliberately not equivalent to the current per-sheet-noise implementation |
| fixed cap `K` | approximate unless execution stops or the gate is deferred | every lossy action must produce an overflow/approximation certificate |
| circuit scheduling with quiescent barriers | exact under the stated schedule and no-noise barrier assumptions | it restricts when gates may occur |
| overhead controlled by treewidth | exact only under bounded **full spacetime** induced width | small CNOT-skeleton width alone does not control spatial inference cost |

## 4. Common invariants for all physically normalized candidates

Unless a strategy explicitly says it reproduces literal sheet-copy noise, it must obey these invariants:

```text
N1  Exactly one data-noise mask eta_p[b] is sampled per physical block b and round.
N2  Exactly one measurement-noise mask eta_q[b] is sampled per block and round.
N3  A CNOT never duplicates either future mask.
N4  Scratch arrays are not lineage state.
N5  Cleanup failure and logical failure are recorded separately.
N6  Back-wall random decisions use injected/counter-based RNG keys for deterministic comparisons.
N7  Physical/correction chain quotients are updated on every edge toggle and CNOT.
N8  A representation may reconstruct a canonical decoded_state, but equality tests use syndrome and winding equivalence unless bitwise preservation is promised.
```

For a block `b`, logical readout is always based on

```text
decoded_quotient[b] = physical_quotient[b] xor correction_quotient[b]
```

or its exactly equivalent aggregation over causal components. Logical success still means zero syndrome after successful cleanup and trivial winding in both torus cycles. Cleanup failure is not silently ORed into logical failure.

## 5. Candidate family: incidence-vector epoch transducers

### 5.1 Core idea and exactness status

Store one decoder state for each distinct source epoch. Attach a `B`-bit incidence vector saying which current logical blocks receive its terminal contribution. A CNOT changes incidence bits; it does not copy the payload. New post-gate noise enters a fresh unit-support epoch, because it must not propagate backward through an earlier gate. All destinations of one epoch deliberately share one field, correction trajectory, and back-wall random stream.

This is not full-sheet copying with cheaper allocation. Descendant paths are algebraically factored into one support vector, and all post-gate physical noise is separated from propagated decoder dynamics.

**Exactness classification: heuristic.** The representation and CNOT support update are exact relative to the shared-recovery epoch semantics in Section 2.6. They match an explicit descendant construction only when every descendant of one source is forced to use coupled feedback/randomness and receives no destination-local noise. Literal sheet-copy gives descendants independent random decisions and noise, so this is not exact relative to that file. The oracle-component and total-syndrome-only routing modes are separate experiments.

### 5.2 Core representation

```julia
const SupportWord = UInt64

# Reuse ChainQuotient and xor_quotient! from Section 1.2.

mutable struct DecoderWorkspace
    new_fields::Array{Int,5}             # L x L x Z x 3 x 2
    hist_correction::BitArray{4}         # L x L x Z x 3
end

mutable struct DecoderEpoch
    id::UInt64
    support::Vector{SupportWord}        # ceil(B/64) words
    raw::ChainQuotient                  # physical contribution
    corr::ChainQuotient                 # accumulated recovery contribution
    measured_component::BitMatrix       # L x L, latest syndrome contribution
    hist::BitArray{3}                   # L x L x Z
    fields::Array{Int,5}                # L x L x Z x 3 x 2
    hist_weight::Int                    # maintained exactly
    decoded_boundary_weight::Int        # count(raw.boundary xor corr.boundary)
    measurement_mismatch_weight::Int    # count(measured_component xor raw.boundary)
    frozen_input::Bool                  # no later noise may enter
    rng_key::UInt128                    # counter-based back-wall stream
    feedback_round::UInt64              # increments after every epoch update
end

mutable struct EpochDecoder
    epochs::Vector{DecoderEpoch}
    native_epoch::Vector{UInt64}        # one unit-support receiver per block
    settled_winding::BitMatrix          # B x 2
    scratch_new_fields::Array{Int,5}    # pooled for serial updates
    scratch_hist_correction::BitArray{4}
    data_noise_counter::Vector{UInt64}
    measurement_noise_counter::Vector{UInt64}
end
```

Dense objects are one `hist` and one `fields` array per live epoch plus one shared workspace. Support and scalar metadata are small. `raw` and `corr` are dense bit boundaries but not dense edge chains. `fields` is mutable and unshared after states diverge. A newly created zero epoch is a lazy sentinel and allocates dense arrays only on its first nonzero event.

The incidence vector is the lineage: `support[b]=1` means that epoch contributes to block `b`. Immediate parent/gate lists are unnecessary for decoding; an optional audit DAG can be retained only for debugging and garbage-collected with the epoch.

Discarded objects are `old_synds`, per-epoch `new_fields`, and per-epoch `hist_correction`. Frozen terminal epochs are reduced to two winding bits per supported block. Contractible closed-chain interiors are not retained.

### 5.3 CNOT update

For `c -> t`, the support transformation is

```text
for every epoch e:
    e.support[t] xor= e.support[c]
settled_winding[t, :] xor= settled_winding[c, :]
```

Any epoch that was accepting future control noise is frozen before this transformation. A new lazy unit-support control epoch is installed for post-gate control noise. A target-native unit epoch can remain native because target-local post-gate noise should remain target-local.

Field-by-field behavior is:

| item | CNOT action |
| --- | --- |
| physical state | no payload copy; transform each raw contribution's support, equivalent to `state_t xor= state_c` |
| state correction | no payload copy; the same support transform propagates the Pauli-frame contribution |
| old syndrome | absent; it is scratch |
| new syndrome | each epoch's `measured_component` travels with the epoch; it is not XOR-merged with another epoch |
| defect history | unchanged payload, transformed destination support |
| history correction | pooled scratch is cleared by the next feedback phase, not at the gate |
| message fields | unchanged payload; never pointwise merged |
| new message fields | pooled scratch is overwrite-only in the next synchronous sweep; no gate-time clear is needed |
| lineage/dependency | toggle target support bit wherever control support bit is one |
| settled information | XOR the two control winding bits into target winding bits |

```julia
function cnot!(d::EpochDecoder, c::Int, t::Int)
    freeze_native!(d, c)
    for e in d.epochs
        if supportbit(e, c)
            toggle_supportbit!(e, t)
        end
    end
    d.settled_winding[t,1] ⊻= d.settled_winding[c,1]
    d.settled_winding[t,2] ⊻= d.settled_winding[c,2]
    d.native_epoch[c] = new_lazy_unit_epoch!(d, c)
end
```

If several CNOTs occur with no intervening decoder/noise round, `freeze_native!` is performed only once and the subsequent gate burst changes support bits only. This avoids creating empty epochs proportional to gate count.

### 5.4 Normal decoder update

The monolithic baseline `update!` must be split into deterministic epoch evolution and physical block sampling.

```julia
function round!(d, p, q, r)
    # Evolve each source epoch once; all blocks in its support share this result.
    for e in d.epochs
        r_field_sweeps!(e.hist, e.fields, d.scratch_new_fields, r)
        feedback!(e.hist, e.fields, e.corr,
                  d.scratch_hist_correction,
                  keyed_round(e.rng_key, e.feedback_round))
        e.feedback_round += 1
    end

    # Sample physical and measurement noise exactly once per block.
    eta_p = [next_edge_mask!(d, p, :data, b) for b in 1:B]
    eta_q = [next_syndrome_mask!(d, q, :measurement, b) for b in 1:B]

    # Route each block's new information to its unit-support native epoch.
    for b in 1:B
        e = ensure_unit_native!(d, b)
        xor_edge_mask_into_epoch_raw!(e, eta_p[b]) # updates both exact weights
        next_component = e.raw.boundary .⊻ eta_q[b]
        delta = e.measured_component .⊻ next_component
        rg_cycle!(e.hist, e.fields)
        e.hist[:,:,1] .= delta
        e.hist_weight = count(e.hist)
        set_measured_component!(e, next_component) # updates mismatch weight
    end

    # Frozen propagated epochs receive no new block noise.
    for e in frozen_epochs(d)
        next_component = e.raw.boundary
        delta = e.measured_component .⊻ next_component
        rg_cycle!(e.hist, e.fields)
        e.hist[:,:,1] .= delta
        e.hist_weight = count(e.hist)
        set_measured_component!(e, next_component)
    end

    collect_terminal_epochs!(d)
end
```

The pseudocode above is specifically the **oracle-component benchmark**: decoder bookkeeping reads hidden component boundaries and routes hidden masks. Its `measured_component` values are virtual sheet-like registers. Immediately after `c -> t`, their target XOR is the pre-gate virtual value `y_t xor y_c`, whereas the actual target's last hardware measurement is still `y_t`; if the old control measurement contained noise, these are different. Oracle deltas therefore define an intentional hidden-component transition semantics and are not claimed to decompose the one observable block delta. `next_edge_mask!` and `next_syndrome_mask!` domain-separate the keys, increment their respective block counter exactly once, and return the mask. `feedback!` consumes the supplied key `(rng_key,feedback_round,i,j,k)` without mutating the counter; the caller increments `feedback_round` exactly once after the complete epoch feedback phase, as shown. `xor_edge_mask_into_epoch_raw!`, feedback correction toggles, and `set_measured_component!` update `decoded_boundary_weight` and `measurement_mismatch_weight` bit-by-bit; a debug mode recomputes both dense counts after every subphase.

The implementable **total-syndrome-only** variant uses a different state split:

```julia
mutable struct TotalEpoch
    id::UInt64
    support::Vector{UInt64}
    corr::ChainQuotient            # no hidden raw component
    hist::BitArray{3}
    fields::Array{Int,5}
    hist_weight::Int
    frozen_input::Bool
    rng_key::UInt128
    feedback_round::UInt64
end

mutable struct TotalSyndromeEpochDecoder
    last_measured::Vector{BitMatrix}      # only observed noisy syndrome
    settled_corr::Vector{ChainQuotient}   # may have nonzero boundary
    epochs::Vector{TotalEpoch}
    native_epoch::Vector{UInt64}
    workspace::DecoderWorkspace
end

mutable struct TotalEpochTrial
    physical::Vector{ChainQuotient}       # simulator/driver only
    decoder::TotalSyndromeEpochDecoder    # cannot access hidden faults/state
end
```

The driver alone samples one block mask, updates `physical[b]`, and returns the one observed `next_measured[b]`. Decoder code sees only `delta_total = last_measured[b] xor next_measured[b]`. It routes that **entire** observed event to the unit-support native epoch; frozen epochs receive an all-zero front slice. This is a concrete assignment heuristic, not hidden fault classification.

```julia
function total_round!(d, observed_next, r)
    for e in d.epochs
        r_field_sweeps!(e.hist, e.fields, d.workspace.new_fields, r)
        feedback!(e.hist, e.fields, e.corr,
                  d.workspace.hist_correction,
                  keyed_round(e.rng_key, e.feedback_round))
        e.feedback_round += 1
        rg_cycle!(e.hist, e.fields)
        e.hist[:,:,1] .= false
    end
    for b in 1:B
        delta_total = d.last_measured[b] .⊻ observed_next[b]
        e = ensure_unit_native!(d, b)
        e.hist[:,:,1] .= delta_total
        e.hist_weight = count(e.hist)
        d.last_measured[b] .= observed_next[b]
    end
    for e in frozen_epochs(d)
        e.hist_weight = count(e.hist)
        if e.hist_weight == 0
            for b in support_indices(e)
                xor_quotient!(d.settled_corr[b], e.corr)
            end
            release!(e) # full correction boundary is retained, not only winding
        end
    end
end
```

At `c -> t`, the trial driver performs `physical[t] xor= physical[c]`; the decoder performs `settled_corr[t] xor= settled_corr[c]` and the same support transform on live correction epochs. It freezes/replaces the control native. It leaves `last_measured[t]` as the actual pre-gate observation; the next post-gate measurement supplies the target event. Thus old measurement noise is not propagated as physical state.

Total-mode readout is

```text
decoded[b] = physical[b] xor settled_corr[b]
             xor XOR_{e: support_e[b]=1} e.corr.
```

This mode never stores a hidden raw contribution or `measured_component` per epoch. A frozen empty-history epoch can be collected because its **full correction quotient**, including any nonzero boundary, is folded into `settled_corr`; it does not use the oracle terminal-winding test. The rule is implementable from total syndromes but may assign a post-gate event to the wrong causal history and may perform worse.

Both modes are synchronous-first; the asynchronous branch should wait for reference tests.

In oracle mode, feedback changes only an epoch's correction quotient/history and raw state is a sum of epoch quotients. In total mode, raw physical state is block-level and only corrections are epoch-factored. In both, the baseline convention measures raw state rather than `state xor state_correction`, so correction timing does not alter later observed physical syndrome.

An **oracle** epoch may be collected when it is frozen, `hist_weight==0`, `measurement_mismatch_weight==0`, and `decoded_boundary_weight==0`. The last two exact counters are updated on every raw, correction, and measured-syndrome toggle; without them checks cost `Theta(L^2)`. XOR its decoded winding into every supported block. A **total-mode** epoch instead uses the full-correction fold in the pseudocode above. In either mode, nonzero stale fields can be released only because the epoch is frozen and empty; a reusable native cannot be reset this way.

Before final cleanup, perform one mode-specific closure. In oracle mode, for **every** live epoch insert `measured_component xor raw.boundary` as that virtual register's final front event; this samples no extra noise but is **not** a decomposition of one physical block measurement. In total-syndrome mode, take one perfect (`q=0`) measurement of each unchanged physical block, route the resulting block-level `last_measured xor next_measured` event through the current native, and give other epochs a zero front. Then mark every native epoch frozen **without** creating a replacement. Subsequent cleanup rounds have `p=q=0` and zero new front input; continue all epochs synchronously and collect terminal objects. At the cap, cleanup fails if any epoch remains nonterminal, any relevant mismatch/defect history remains, or the reconstructed decoded block has nonzero syndrome; checking `hist` alone is insufficient. This explicit closure and freeze is what makes native epochs eligible for terminal collection.

### 5.5 Readout

In oracle mode, for each block `b`, XOR the raw and correction quotients of every live epoch whose support contains `b`, then XOR `settled_winding[b,:]` into the two winding bits. Total-syndrome mode instead uses the block-level `physical`, full `settled_corr`, and active correction quotients defined above. Reconstruct a canonical edge array only if the caller needs `decoded_state` as an array.

```julia
function decoded_quotient(d, b)
    out = zero_chain_quotient(L)
    out.winding .⊻= d.settled_winding[b,:]
    for e in d.epochs
        if supportbit(e,b)
            xor_quotient!(out, e.raw)
            xor_quotient!(out, e.corr)
        end
    end
    return out
end
```

Thus `decoded_state(control)` is unchanged by the CNOT support transform, while `decoded_state(target)` receives exactly the control contribution. Logical failure remains the OR of control and target winding failures after successful cleanup. Cleanup failure stays separate.

### 5.6 Classical overhead

Let `A(t)` be the number of allocated nonzero live epochs, including at most one materialized native epoch per block. With pooled workspace, oracle and total modes differ by permanent block-level state:

```text
memory oracle = Theta(A(t) M_epoch + A(t)s_B W/8 + M_workspace + B) bytes
memory total  = Theta(A(t) M_total_epoch + A(t)s_B W/8 + B L^2/8
                      + M_workspace) bytes,
                M_total_epoch <= M_epoch up to constants

both are O(B M_block + (A(t)-B)_+ M_epoch
           + A(t)s_B W/8 + B L^2/8 + M_workspace).
```

The `Theta(BL^2)`-bit total-mode term comprises block physical quotients, `last_measured`, and full `settled_corr`; it remains even if all native epochs are lazy and `A=0`. After all frozen epochs drain, `A(t)<=B` and either mode returns to `O(B*M_block)` (in fact about half the dense field storage if updates are serial). There is no path-count factor.

```text
runtime per round = Theta(A(t)(r+1)L^2Z + B L^2)
one CNOT, oracle  = Theta(A(t)) bit tests/toggles
one CNOT, total   = Theta(A(t) + L^2) including physical/settled quotient XORs
gate-only burst   = Theta(D A(t)) oracle;
                    Theta(D(A(t)+L^2)) total mode; O(A(t)) peak state
cleanup C rounds  = Theta(C A_peak (r+1)L^2Z)
                    plus O(A_peak B/W) oracle terminal folds or
                    O(A_peak B L^2) total-mode terminal folds
GC oracle         = O(1) checks + O(B/W) winding-support fold
GC total          = Theta(|support| L^2), worst Theta(BL^2),
                    to XOR the full correction quotient into settled blocks
buffer recycling  = Theta(L^2Z) only when memory is eagerly cleared
```

Worst-case `A(t)` is not bounded by `D`-independent constants. If `D` counts individual CNOT events, at most one new native epoch is materialized per outgoing event, and no terminal histories drain, then `A(t)<=A(0)+D`. The convenient `A<=B+D` tables assume the burst begins with only `B` native epochs and empty lazy natives are coalesced rather than accumulated. This is linear rather than Fibonacci/exponential, but still a live-width dependence.

### 5.7 Required repeated-CNOT cases

| schedule | behavior |
| --- | --- |
| `C1 -> C2` repeated | Support toggling automatically represents cancellation from repeated ideal CNOTs. With no intervening noise, `A=B`; with a noisy round after each gate and no drain, `A<=B+D`. No copied-sheet linear factor. |
| `C1 -> C2, C2 -> C1` alternating | Supports undergo the correct two row operations. `A<=B+D` in the no-drain/noisy-round worst case, not `Theta(phi^D)`. |
| `C1 -> C2, C1 -> C3` fanout | One control payload has a three-block support rather than three payload copies. New post-fanout control noise creates one unit epoch. |
| long chain `C1 -> C2 -> ...` | A gate-only chain keeps `A=B`; support metadata is at most `Theta(B^2)` bits. With one noisy round per link, at most one new epoch per outgoing link, still `Theta(B)` for one length-`B` chain. |
| dense CNOTs, no idle rounds | If no decoder/noise update occurs between gates, compose support updates with `A=B`; peak memory is independent of `D`. If each gate is followed by a noisy decoder round, `A` can grow as `B+D`. |
| CNOTs separated by `O(L)` idle rounds | Below threshold, most frozen epochs should drain and memory should return near `B*M_block`; this is typical, not deterministic because back-wall states can persist. |

Dependence is on `B`, `A(t)`, `L`, and `Z`. Fanout affects support density and CNOT work but not payload count directly. Total historical depth disappears after GC. Circuit treewidth is not used. No configured cap is required.

### 5.8 Logical-performance expectation and failure modes

This design keeps **different source epochs** from destructively XORing their fields, but it does not preserve independent decoding of multiple destinations of one source. Those descendants share feedback and correction forwarding. Whether that is enough to recover the target-performance advantage is the central hypothesis; it may fall between primitive and literal sheet-copy rather than approach sheet-copy. Target failures may remain dominant because the target has a larger causal support and because coupled descendant corrections can be wrong for one destination.

Two unproven points are the shared-recovery rule and the noise routing. Oracle mode assigns a future virtual target register transition to hidden component streams and assigns the physical measurement-error mask to a native epoch; it is not an observable decomposition. Total-syndrome-only mode lacks even that hidden label. Even when the XOR algebra is consistent, both the shared correction stream and event assignment can underuse or misassign correlations between propagated and new target defects.

Primary failure modes are:

- accidental injection of the same physical noise into multiple epochs;
- failure to create a fresh unit control epoch after an outgoing CNOT;
- stale pooled `new_fields` or correction scratch;
- mutating a shared support vector or dense payload through aliasing;
- collecting an epoch that can receive future defects;
- treating `hist==0` as reusable-field quiescence;
- inconsistent component syndrome closure at a gate boundary;
- support-bit mistakes under alternating CNOTs;
- destination lineages that require different back-wall choices but are forced to share one;
- oracle leakage from hidden `eta_p`, `eta_q`, or raw component boundaries;
- `A(t)` growing linearly without idle drainage in dense noisy circuits.

Reject this family if deterministic no-noise/support tests fail, if it does not materially close the target-failure gap to literal sheet-copy at `L=5,7`, or if target performance falls back to the primitive trend despite exact separation of fields. Also reject the claimed overhead benefit if `A(t)` continues to grow after `L` ideal idle rounds in almost every subthreshold sample.

### 5.9 Implementation difficulty and validation

**Difficulty: large.** The baseline decoder may remain unchanged as a reference mode, but a separate mode must introduce `ChainQuotient`, `DecoderEpoch`, `EpochDecoder`, support-bit operations, terminal folding, and pooled workspaces. The duplicated `update!` logic must be refactored into field propagation, feedback, noise sampling, syndrome-event insertion, and RG cycling. Relevant existing entry points are `onesite_field_update`, `update_2d_windowed_fields!`, `perform_correction!`, `rg_cycle!`, `primitive_cnot_x_sector!`, and the sheet update/readout driver.

Deterministic comparisons are possible only with injected data masks, measurement masks, and back-wall random bits. In addition to the common validation suite in Section 11, this family needs:

- exact `hist`/`fields`/syndrome-register equivalence to baseline and syndrome-plus-winding equivalence of the canonical chain quotient (not bitwise edge equality);
- support-vector equivalence to explicit **coupled-randomness** descendants through every repeated-CNOT pattern;
- exact cancellation after applying the same CNOT twice with no intervening noise;
- an assertion that every sampled physical/measurement mask has exactly one epoch consumer;
- a witness comparison against two explicit descendants with independent back-wall coins, documenting the intentional semantic difference;
- a black-box total-syndrome-only test that makes hidden masks/component boundaries inaccessible;
- GC equivalence before/after folding under later noisy native rounds and later CNOTs, not only CNOT-only continuations.

## 6. Candidate family: sparse causal replay engine

### 6.1 Core idea and exactness status

This family keeps the epoch/support algebra from Section 5 but replaces dense per-epoch history and message arrays by exact sparse wavefront state, immutable event logs, and bounded replay intervals. It attacks the dominant `A(t)L^2Z` storage rather than only descendant copying.

**Exactness classification: heuristic.** The sparse engine is representation-exact relative to the shared-recovery epoch semantics when it preserves every nonzero message, uses exact `Int` values, and replays every input/random choice. It inherits both the coupled-destination rule and the oracle-versus-total-syndrome routing split from Section 5. Dropping small fields, clipping messages, or reconstructing from current defects would be an additional approximation.

### 6.2 Core representation

Coordinates `(i,j,k,a,s)` use a packed 64-bit key below. A proved-range 32-bit specialization is optional, not part of the reference semantics.

```julia
const CoordKey = UInt64

struct FieldEntry
    key::CoordKey               # packed (i,j,k,a,s)
    value::Int                  # exact current value
end

struct LogHandle
    slot::Int64                 # 0 is null
    generation::UInt64          # rejects stale recycled handles
end

struct RoundInput
    parent::LogHandle
    round::Int
    data_events::Vector{CoordKey} # sparse edge toggles
    meas_events::Vector{CoordKey} # sparse site toggles
    backwall_key::UInt128       # reproducible choices
end

mutable struct LogArena
    nodes::Vector{Union{Nothing,RoundInput}}
    generations::Vector{UInt64}
    refcounts::Vector{Int32}
    free_slots::Vector{Int64}
end

abstract type EpochPayload end

mutable struct SparsePayload <: EpochPayload
    hist::Vector{CoordKey}      # sorted unique (i,j,k) keys
    fields::Vector{FieldEntry}  # sorted unique nonzero keys
end

mutable struct DensePayload <: EpochPayload
    hist::BitArray{3}
    fields::Array{Int,5}
end

# Snapshot arrays are deep-owned and write-protected by the replay engine.
# They are never passed to an in-place decoder routine.
abstract type EpochCheckpoint end

struct OracleCheckpoint <: EpochCheckpoint
    raw::ChainQuotient
    corr::ChainQuotient
    measured_component::Vector{CoordKey} # sorted true-site keys
    payload::Union{SparsePayload,DensePayload}
    decoded_boundary_weight::Int
    measurement_mismatch_weight::Int
    hist_weight::Int
    frozen_input::Bool
    feedback_round::UInt64
end

struct TotalCheckpoint <: EpochCheckpoint
    corr::ChainQuotient
    payload::Union{SparsePayload,DensePayload}
    hist_weight::Int
    frozen_input::Bool
    feedback_round::UInt64
end

mutable struct SparseOracleEpoch
    id::UInt64
    support::Vector{UInt64}
    raw::ChainQuotient
    corr::ChainQuotient
    measured_component::Vector{CoordKey} # sorted true-site keys
    live_payload::Union{SparsePayload,DensePayload}
    checkpoint::OracleCheckpoint
    decoded_boundary_weight::Int
    measurement_mismatch_weight::Int
    hist_weight::Int
    checkpoint_round::Int
    log_tail::LogHandle
    frozen_input::Bool
    rng_key::UInt128
    feedback_round::UInt64
end


mutable struct SparseTotalEpoch
    id::UInt64
    support::Vector{UInt64}
    corr::ChainQuotient                  # no hidden raw component
    live_payload::Union{SparsePayload,DensePayload}
    checkpoint::TotalCheckpoint
    hist_weight::Int
    checkpoint_round::Int
    log_tail::LogHandle
    frozen_input::Bool
    rng_key::UInt128
    feedback_round::UInt64
end


mutable struct SparseTotalSyndromeEpochDecoder
    last_measured::Vector{BitMatrix}
    settled_corr::Vector{ChainQuotient}
    epochs::Vector{SparseTotalEpoch}
    native_epoch::Vector{UInt64}
    workspace::DecoderWorkspace
end
```

The indexed event-log nodes are immutable and structurally shared; their event vectors are uniquely owned and never mutated after insertion. Arena storage itself uses reference counts plus a free list. Releasing a prefix sets its slots to `nothing`; reuse increments `generation`, so stale handles fail rather than alias. Thus allocated arena capacity is bounded by peak simultaneously reachable nodes rather than cumulative `D`. A periodic stop-the-world rebuild may shrink capacity to the reachable set and rewrites all live handles; reference counting alone without slot reuse/rebuild would not satisfy the depth claim. `SparseOracleEpoch` implements the oracle mode; `SparseTotalEpoch` and its block-level wrapper implement the total-syndrome-only mode without adding hidden `raw` or component-syndrome state. The oracle wrapper mirrors `EpochDecoder` and is omitted only to avoid repeating its native-index, settled-winding, workspace, and counter fields.

The checkpoint and the live cache are distinct owned states. Every `K` decoder rounds, `freeze_copy(live_payload)` and deep copies of the mode-specific quotient/register scalars replace `checkpoint`; `log_tail` is reset after unreachable prefixes are reference-counted away. The live payload remains separately mutable. Replay begins from a mutable clone of the write-protected checkpoint and applies the retained log. Copy-on-write can avoid the initial duplicate only if every in-place history/field operation first proves unique ownership. Merely storing `checkpoint_round` without a checkpoint snapshot is not a valid replay design. `K` is a replay/checkpoint interval, not a cap on histories and not an approximation parameter.

`SparsePayload` is the normal live representation. The quotient boundaries remain dense in both the live state and checkpoint. A mode switch replaces `live_payload` by `DensePayload` when sparse entry bytes exceed dense cost, and converts back below a lower hysteresis threshold. Dispatch, rather than mutating a `Vector` field into an array, makes this adaptive conversion implementable and exact. A checkpoint records whichever representation was active, but its arrays remain read-only.

### 6.3 Exact sparse update, CNOT, and normal rounds

A zero output field can become nonzero only if its reverse cone contains a current defect or nonzero predecessor field. Build the next candidate set by enumerating reverse cones of all current defects and fields, sort/deduplicate it, and evaluate the unchanged `onesite_field_update` formula only there. Current nonzero entries whose next value is zero are omitted.

```julia
function sparse_field_sweep!(e, p::SparsePayload)
    candidates = reverse_cone_candidates(p.hist, p.fields)
    sort_unique!(candidates)
    next = FieldEntry[]
    for key in candidates
        value = exact_field_value(key, p.hist, p.fields)
        value == 0 || push!(next, FieldEntry(key,value))
    end
    p.fields = next
end
```

Feedback iterates active `hist` entries but performs the same positive-min test, priorities, correction parity, and keyed `0.8` back-wall decision. RG cycling remaps sparse `k` coordinates, XOR-cancels coincident back-wall defects, applies exact `nonzeromin` to coincident spatial fields, drops the front, and inserts new events. No equilibrium recomputation occurs.

Replay starts from a complete sparse or dense checkpoint and executes the same sweeps and recorded inputs. A lazy query may replay only the backward causal cone needed for one field value, but it must memoize all dependencies and produce the same value as full replay. Local replay is an optimization, not a different rule.

The CNOT field-by-field action is:

| item | sparse CNOT action |
| --- | --- |
| physical state/correction | oracle retains `raw/corr` per epoch; total mode retains only epoch `corr` plus block-level physical state; transform support in either mode |
| old/new syndrome | old is scratch; oracle component syndrome stays distinct; total mode stores only block-level `last_measured` |
| defect/history correction | sparse defects stay distinct; correction scratch is pooled |
| fields/new fields | current sparse field map remains intact; next-field scratch is temporary |
| lineage | toggle target support wherever control support is one; no log/map copy |
| settled information | oracle applies `settled_winding[t,:] xor=settled_winding[c,:]`; total mode applies `settled_corr[t] xor=settled_corr[c]` |

Thus it has the same shared descendant correction stream as Section 5; sparsity does not restore destination independence.

Normal rounds:

1. run `r` exact sparse sweeps and sparse feedback for each epoch;
2. sample one data and measurement mask per block;
3. append one `RoundInput` for **every** live epoch: native epochs receive their routed edge/site events, while frozen epochs receive empty event vectors but retain that round's exact back-wall key (or a specified counter-derived equivalent);
4. update chain quotients and component syndromes;
5. perform the sparse RG cycle and insert the new event set;
6. checkpoint any epoch whose replay distance reaches `K`;
7. fold terminal frozen epochs exactly as in Section 5.

Future noise is never sampled per bookkeeping lineage. Oracle mode routes hidden events into `SparseOracleEpoch`; total-syndrome-only mode logs only the observed block event into the unit-support `SparseTotalEpoch` and gives frozen total epochs an empty front slice. Cleanup first performs the mode-specific perfect-measurement closure from Section 5, then runs the sparse engine with empty physical/measurement inputs but logged/counter-derived back-wall choices. Oracle readout XORs `raw` and `corr` quotients for every support bit and adds settled winding. Total readout uses block physical state XOR block `settled_corr` XOR supported live `corr` quotients. Either can reconstruct a canonical array if requested. Because the same support row operation is used, `decoded_state(control)` is unchanged and `decoded_state(target)` is its pre-gate target quotient XOR its pre-gate control quotient. Logical failure is unchanged and cleanup failure remains separate.

### 6.4 Classical overhead

Let

```text
H(t) = total active sparse hist entries
F(t) = total nonzero field entries
Q(t) = total sparse measured-component entries
H_0(t), F_0(t), Q_0(t) = corresponding entries in retained checkpoints
J_K  = retained sparse edge/site input entries since checkpoints
R_K  = retained per-round log nodes/keys, with R_K <= A(t)K
S_payload(t) = actual payload locations copied in one checkpoint batch;
               sparse epochs contribute their H/F entries, while each dense
               epoch contributes Theta(L^2Z) locations even when values are zero
V    = largest exact field value represented.
```

An information-level bound is

```text
memory = O(A(t) L^2
           + [H(t)+H_0(t)] log(L^2Z)
           + [Q(t)+Q_0(t)] log(L^2)
           + [F(t)+F_0(t)] [log(L^2Z) + log V]
           + J_K log(L^2Z)
           + R_K log(R_K+1)
           + A(t)B
           + workspace) bits.
```

The preceding formula is an information-level packed bound, not the allocation cost of the concrete structs. The concrete log uses one `UInt64` word per event plus a constant `c_log` words per `RoundInput` node, as well as vector headers/capacity. The `A L^2` term includes the live and checkpoint chain quotients and oracle component register up to constant factors. Estimated bytes must use `Base.summarysize` in experiments. In the worst case both the live cache and checkpoint are dense, so the **payload** portion is a constant-factor `Theta(A*M_epoch)` (approximately two persistent epoch payloads per live epoch before copy-on-write savings), and the concrete total worst case is

```text
Theta(A M_epoch + A s_B W/8
      + [B K L^2 + c_log A K] W/8) bytes,
```

up to headers/capacity, because at most `B` native streams contain dense edge/site events during each of the `K` retained rounds while all `A` streams contain log nodes. A separate bit-packed log implementation could approach the earlier logarithmic bound, but is not the reference layout. The shorter statement `Theta(A M_epoch)` is valid only when `K=O(Z)`, `A>=B`, and these log/support terms are dominated; it is not unconditional. Delayed sparse/dense conversion can add more constant-factor capacity. With independent subthreshold clusters, the hoped-for regime is `H+H_0,F+F_0 << A L^2Z`.

If `C_F` is the number of distinct candidate field keys in a sweep,

```text
round runtime = O(A + r C_F[log C_F + log(H+1) + log(F+1)]
                  + H[log H + log(F+1)] + B L^2
                  + [A L^2 + S_payload + Q + J_K]/K),
                C_F <= 54(F+H) before deduplication
one CNOT oracle = Theta(A(t)) support operations
one CNOT total  = Theta(A(t) + L^2)
checkpoint batch = Theta(A L^2 + S_payload + Q), with
                   S_payload <= Theta(A L^2Z); amortize over K rounds
log-prefix release = Theta(R_K + J_K) per checkpoint batch,
                     amortized Theta((R_K+J_K)/K); the leading O(A)
                     round term absorbs R_K/K <= A
replay        = O(K * local dense-or-sparse update cost)
cleanup C rounds = O(C times the non-checkpoint round terms
                     + C[A L^2+S_payload+Q+J_K]/K
                     + terminal-fold costs below)
GC oracle     = O(B/W) support fold with exact mismatch counters
GC total      = Theta(|support|L^2), worst Theta(BL^2),
                plus reference-count release
arena shrink/rebuild = Theta(A + R_K + J_K) when requested, including
                       rewriting all live tail/checkpoint handles; ordinary
                       free-list recycling is linear in released nodes/events.
```

Worst-case decoder evolution is `Theta(A(r+1)L^2Z)` per round after dense fallback, plus the displayed checkpoint/log and physical-block terms. Over a `C`-round dense cleanup, terminal folding can add `O(A_peak B L^2)` in total mode or `O(A_peak B/W)` in oracle mode. The checkpoint interval bounds replay depth, not total live causal width.

### 6.5 Required repeated-CNOT cases

| schedule | behavior |
| --- | --- |
| repeated `C1 -> C2` | Same `A<=B+D` no-drain bound as Section 5. Support toggles do not duplicate logs; the stated epoch semantics creates no descendant fork to share. Sparse size follows active defects/messages, not copied sheets. |
| alternating two-block CNOTs | No Fibonacci path materialization. Worst live epochs remain `B+D`; sparse fields may nevertheless become dense near persistent back-wall defects. |
| one-to-two-target fanout | One event/log payload has a multi-block support. No duplicate sparse wavefront is created. |
| long chain | A gate-only chain uses `A=B` and `Theta(B^2)` support bits; noisy rounds add at most one epoch per outgoing link. |
| dense CNOTs with no idle rounds | CNOT-only bursts add almost no event data. With noisy rounds between gates, `J_K`, `A`, and `F` can grow until checkpoints/GC; no `D`-independent worst-case bound. |
| CNOTs separated by `O(L)` idle rounds | Expected best case: clusters erode, sparse maps shrink, old logs are released, and memory approaches `O(BL^2 + live sparse fields)`; persistent back-wall fields are the adversarial case. |

The representation depends on active defects, nonzero fields, `A(t)`, support density/fanout, and `K`. It does not intrinsically depend on total past depth after logs are checkpointed and terminal epochs are collected. It does not use circuit treewidth.

### 6.6 Performance, failure modes, implementation, and validation

When representation-exact, logical performance must match the dense shared-recovery epoch transducer for identical inputs and keyed randomness. It loses no additional decoder information. It does not approach independent-sheet behavior by virtue of sparsity; target performance has the same coupled-destination and oracle-routing risks as Section 5, and target failures may remain dominant.

Failure modes are missing reverse-cone candidates, incorrect sparse zero deletion, duplicate-key/lineage collisions, stale memoized fields, mutable aliasing of shared logs, iteration-order tie changes, `Int` clipping, missing frozen-epoch random choices, inconsistent component syndromes, incorrect correction/support forwarding, oracle leakage, reference cycles, and sparse metadata overhead exceeding dense storage. There is no lossy merge in exact sparse mode and no configured overflow; dense/alternating circuits can still grow `A(t)` and force dense fallback.

Reject the sparse engine immediately if any deterministic round differs from dense epoch execution. Reject it as an overhead strategy if, at `p<=0.015`, median `F/(6A L^2Z)` exceeds the empirically measured sparse/dense break-even or runtime exceeds dense execution by more than a predeclared factor (suggested initial cutoff: `3x`) without a compensating peak-memory reduction of at least `2x`.

**Difficulty: research-scale.** It requires replacements or wrappers for `onesite_field_update`, both field sweep functions, feedback iteration, `rg_cycle!`, checkpoint/replay, adaptive dense conversion, and deterministic RNG injection. The baseline remains an unchanged oracle mode.

In addition to Section 11, validate:

- every sparse sweep against a dense sweep on exhaustive `L=3`, small-`Z` states;
- sparse RG/back-wall collision cases constructed by hand;
- replay from every checkpoint position `0:K` against uninterrupted execution;
- replay frozen epochs across stochastic back-wall rounds with empty physical inputs;
- sparse-to-dense-to-sparse round trips;
- immutable-prefix reference counts and post-GC decoded equivalence;
- free-list slot reuse, generation-mismatch rejection, and periodic arena rebuild with no growth under fixed `R_K` as `D` increases;
- identical memory traces under different map insertion orders.

## 7. Candidate family: rolling CNOT factor graph with overlapping-window elimination

### 7.1 Core idea and exactness status

Abandon independent Lake sheets as the object of inference. Build one spacetime factor graph for all blocks in a finite overlapping window. A candidate horizontal link represents a data/recovery-chain segment, a vertical link represents a measurement-chain segment, observed syndrome changes impose XOR-boundary factors, and a CNOT interface imposes the control-before/target-before/target-after chain relation. Solve the complete window, commit only the old part of the selected chain, and carry the relative boundary of the recent part into the next window.

This is a proposed factor-graph construction motivated by, and algebraically consistent with, DKLP Eq. (72). Equation (72) itself does not provide a factor graph, an elimination order, or a width bound.

**Exactness classification: exact under restricted CNOT schedules.** Exact mode is exact only for the fully specified finite-window **rational-weight minimum-cost chain** objective below, with a deterministic tie rule and full induced width `w_full<=K`. It is not the optimal DKLP homology-class posterior, which requires summing probabilities over all chains in each class, and it is not equivalent to infinite-history recovery. It is not sheet-copy or Lake semantics. If `w_full>K`, exact mode terminates that trial with a bounded width certificate; a separately selected loopy-BP policy is approximate and emits a fallback certificate. No result is labeled exact after a width overflow.

### 7.2 Core representation and exact objective

```julia
@enum VarKind latent_edge_state data_fault measurement_fault

struct BinaryVar
    id::Int32
    kind::VarKind
    block::UInt16
    i::UInt16
    j::UInt16
    time::Int32
    orientation::UInt8
end

struct UnaryWeight
    var::Int32
    cost_if_one::BigInt          # scaled exact integer weight w_p or w_q
end

struct XorFactor
    vars::Vector{Int32}
    rhs::Bool                    # observed defect/boundary parity
end

mutable struct GateTransform
    time::Int32
    rows::Dict{UInt16,BitVector} # changed rows only; absent row is identity
end

mutable struct EliminationBag
    vars::Vector{Int32}
    cost::Vector{BigInt}         # exact min-sum table
    feasible::BitVector          # false represents +infinity
    argmin_backptr::Vector{Int32}     # reconstruct selected chain this window
end

struct WidthCertificate
    time::Int32
    width_cap::Int32
    observed_width::Int32
    variable_count::Int64
    factor_count::Int64
    gate_incidence_count::Int64
    elimination_order_hash::UInt128
    graph_observation_hash::UInt128
    largest_bag_hash::UInt128
    gate_layer_hash::UInt128
    policy::Symbol               # :stop or :bp
    bp_iterations::Int32
    bp_residual::Float64         # diagnostic only, never used by exact mode
end

mutable struct RollingCNOTGraph
    variables::Vector{BinaryVar}
    weights::Vector{UnaryWeight}
    xor_factors::Vector{XorFactor}
    gate_layers::Vector{GateTransform}
    carried_relative_boundary::Vector{BitMatrix}
    carried_homology::BitMatrix           # B x 2
    measured_syndrome::Vector{BitMatrix}
    committed_corr::Vector{ChainQuotient}
    half_window::Int                     # Z_FG
    time::Int32
    width_cap::Int
    overflow_policy::Symbol              # :stop or :bp
    width_overflow_count::Int
    approximation_certificate_count::Int
    recent_width_certificates::Vector{WidthCertificate} # fixed-size ring
    width_certificate_capacity::Int
    width_certificate_cursor::Int
    width_certificate_sink::Function       # streaming; no resident growth in D
    trial_status::Symbol                    # :exact, :bp_approx, :width_stopped
end

mutable struct FactorGraphTrial
    physical::Vector{ChainQuotient}       # simulator/driver only
    decoder::RollingCNOTGraph             # receives observations only
end
```

The `2Z_FG` spacetime coverage is dense in block/space/time, while factor adjacency and changed gate rows are sparse until a dense circuit makes them dense. Observations and closed-slice factors are immutable and may share canonical coordinate/row objects; the current gate layer is mutable only until it closes. Exact bag tables/backpointers are dense in their separator assignments and are never shared mutably. There are no Lake message fields to retain or reconstruct. The chosen recovery is reconstructed from backpointers, then old factors/tables and committed chain interiors are discarded. Gate factors, carried boundary/homology, and time labels carry all causal dependence.

Use the following explicit convention. Measurement slices are indexed by `tau`. For physical edge `e=(i,j,o)` and stabilizer vertex `v`:

```text
a[b,e,tau]   latent cumulative physical X-error state just before measurement tau
x[b,e,tau]   data-fault bit between slices tau-1 and tau
mu[b,v,tau]  measurement-flip bit at slice tau
y[b,v,tau]   observed measured syndrome bit (constant, not a latent variable).
```

For every non-gate transition and every edge, add

```text
a[b,e,tau] xor a[b,e,tau-1] xor x[b,e,tau] = 0.
```

For every measurement, add the observed-syndrome factor

```text
mu[b,v,tau] xor y[b,v,tau]
xor XOR_{e incident on v} a[b,e,tau] = 0.
```

Only `x` and `mu` receive unary costs. Exact mode takes explicit nonnegative rational decoder weights, clears their common denominator, and stores the resulting coprime integer weights `w_p,w_q`; for the required `qrat=1` tests use `w_p=w_q=1`, which exactly minimizes fault count. A probability-calibrated implementation may supply a preregistered rational approximation to the log-odds ratio, but exactness then refers to that rational-weight decoder, not to real-arithmetic Bernoulli MAP. Thus noisy measured bits never appear in a CNOT propagation factor. The equivalent candidate spacetime chain consists of selected `x` and `mu` variables; `a` supplies an implementation-friendly state-space factorization.

For fixed observations, geometry, boundary conditions, integer weights `w_p,w_q`, and tie order, exact mode minimizes

```text
w_p * sum_data_faults x_e
+ w_q * sum_measurement_faults mu_v
```

subject to all transition, observation, and CNOT XOR factors. Two homology bits per affected block are introduced as explicit sector/separator variables during elimination so a cycle eliminated inside a bag is not lost; they count toward `w_full`. The compiler deterministically fixes one elimination order (including deterministic tie-breaking in its width heuristic). For equal table costs, elimination and backtracking prefer bit `0` before bit `1` in that compiled order. This order-dependent rule—not an uncosted global lexicographic self-reduction—is part of the decoder semantics. A `Float64` fast mode is exact only relative to its explicitly fixed floating-point comparison semantics and must not be reported as mathematical MAP. If the target changes to maximum-probability homology class, use a separately specified sector-sum algorithm; ordinary floating log-sum-exp is approximate.

At the left edge of each overlapping window, constrain the boundary of latent `a[b,:,0]` to the carried relative boundary and carry its two homology-sector bits. Materialize the deterministic canonical representative of that boundary/homology as zero-cost, cut-time-tagged `CarryLink` edges. Contractible gauge choices are resolved by the compiled elimination/backtracking tie rule. `CarryLink`s participate in the next augmented explanation graph and are eligible for later commitment; merely constraining the new boundary and then forgetting the representative would lose the old recovery obligation. This boundary factor and its canonical links are how `E_keep`, rather than hidden physical state, initializes the next inference window.

Call this family's half-window depth `Z_FG` to distinguish it from the Lake buffer `Z`. The temporal graph contains `2Z_FG` measurement rounds. After solving it, form an **augmented explanation hypergraph**: selected `x`/`mu` and canonical `CarryLink` edges are incidence edges, and transition/CNOT parity factors are explicit factor nodes connecting the latent-state incidences through which parity flow continues. Its terminals are the observed-boundary monopoles. Do not assume every connected component has exactly two terminals. A component enters `E_old` iff every one of its terminals lies in `0 <= time < Z_FG`; a terminal-free cycle enters `E_old` only if all of its selected links and factor incidences lie in that old half. Components with any recent terminal, and cycles crossing the cut, enter `E_keep`. For an ordinary two-terminal string this reduces to DKLP's both-endpoints-old rule; it is not a blind time cut. `Pi(E_old)` updates `committed_corr`; the cut-relative boundary and homology of only the **surviving, uncommitted** `E_keep` are re-summarized. A committed carry link is never carried again. This augmented-connectivity convention is part of the proposed finite-window decoder and must be tested against direct factor assignments. For `p` comparable to `q`, DKLP's accuracy argument requires `Z_FG >> L`, not the baseline `Theta(log L)` buffer.

### 7.3 CNOT update

The simulation order places a CNOT between two completed measurement slices, with no data or measurement fault at the ideal gate instant. Let `a^-` and `a^+` be the latent edge-state variables immediately before and after that interface. For **every physical edge coordinate** `e=(i,j,o)`, emit exactly

```text
a[c,e,+] xor a[c,e,-] = 0
a[t,e,+] xor a[t,e,-] xor a[c,e,-] = 0.
```

For every uninvolved block `b`, identity is represented by aliasing `a[b,e,+]` to `a[b,e,-]`; an explicit equality factor is allowed in the deterministic reference but should not allocate a new variable in the optimized graph. The unchanged control row may be aliased in the same way. The next ordinary transition introduces its own `x[b,e,tau]` fault variable. Observation variables `y` and measurement faults `mu` never enter these gate equations. Taking the boundary over incident edges gives the control-before/target-before/target-after chain relation in Eq. (72).

For a gate-only burst, do not emit factors after each event. Compose its row operations into `R`, create one pre/post latent state per block and edge, and emit

```text
a[b,e,+] xor XOR_{j: R[b,j]=1} a[j,e,-] = 0
```

for every `b,e` when the layer closes. Introducing intermediate latent states and the two single-CNOT equations after each event is the deterministic reference and must give the same factor relation.

| item | CNOT action |
| --- | --- |
| physical state | `physical[t] xor= physical[c]` |
| state correction | propagate only already committed recovery: `committed_corr[t] xor= committed_corr[c]` |
| old syndrome | pre-gate `y` stays a constant observation factor; it is not overwritten or propagated |
| new syndrome | no post-gate value exists until the next actual measurement |
| defect history | append a symbolic gate layer and its XOR interface |
| history correction | none yet; the window solve selects candidate links |
| message/new fields | absent; exact elimination tables or approximate BP messages replace them |
| lineage/dependency | mutate the current gate layer by `R[t,:] xor= R[c,:]` |
| settled information | already committed target correction receives already committed control; a recovery inferred later is time-tagged and transported through every gate after its own time before commitment |

```julia
function cnot!(trial::FactorGraphTrial, c, t)
    g = trial.decoder
    xor_quotient!(trial.physical[t], trial.physical[c])
    xor_quotient!(g.committed_corr[t], g.committed_corr[c])
    layer = gate_layer_at!(g, g.time)
    row!(layer, t) .⊻= row_or_identity(layer, c)
    # close_gate_layer! later emits the composed per-edge equations above
end
```

Immediate physical work is `Theta(L^2)` plus `Theta(B/W)` for the bit row. A gate layer stores only changed rows, so its first gate does not allocate an identity `B x B` matrix. Compilation always traverses dictionary keys in sorted block order so hash iteration cannot change variable or tie order. If `g_tau` rows differ from identity at slice `tau`, keeping the layer costs `Theta(g_tau B/W)` words and expanding it costs `Theta(g_tau L^2)` variables/factors before counting row weight; a single CNOT has `g_tau<=1`, while a dense composed layer can have `g_tau=Theta(B)`. The complete incidence cost is `I_gate` in Section 7.6.

### 7.4 Normal update, commitment, and cleanup

Each noisy round samples one data mask and one measurement mask per physical block, updates the simulation's physical quotient, records the observed noisy syndrome `y` (and its change for metrics), and appends one new graph slice. The inference graph never receives hidden masks. There are no Lake RG shifts, local defect feedback, or physical corrections between window solves.

```julia
function round!(trial::FactorGraphTrial, p, q)
    g = trial.decoder
    close_gate_layer_if_open!(g) # emit composed pre/post factors exactly once
    for b in 1:B
        eta_p = sample_edge_mask_once(p, b, g.time)
        eta_q = sample_measurement_mask_once(q, b, g.time)
        xor_edge_mask_into_quotient!(trial.physical[b], eta_p)
        next_meas = trial.physical[b].boundary .⊻ eta_q
        observed = g.measured_syndrome[b] .⊻ next_meas
        append_measurement_slice!(g, b, next_meas, p, q)
        record_observed_change!(g, b, observed)
        g.measured_syndrome[b] .= next_meas
    end
    if live_round_count(g) == 2 * g.half_window
        result = solve_with_width_policy!(g)
        result.status == :stopped && return
        selected = result.selected
        E_old, E_keep = partition_by_endpoint_vintage(selected, g.half_window)
        commit_time_tagged_projection!(g, E_old)
        g.carried_relative_boundary = relative_boundary(E_keep)
        g.carried_homology = homology_summary(E_keep)
        discard_old_half_and_solver_tables!(g)
    end
    g.time += 1
end
```

`solve_with_width_policy!` first constructs the proposed elimination order and its exact `w_full` without allocating oversized tables. At overflow it builds and streams the bounded `WidthCertificate`; `:stop` sets `trial_status=:width_stopped` and returns no selected chain, while `:bp` sets `:bp_approx`, runs the configured finite iteration count, and records its final residual. A sink failure stops the trial. The policy is fixed before the run; there is no silent fallback.

`exact_min_weight_solve` stores conditional argmin backpointers until the whole current-window minimizing assignment is known; it does not commit an eliminated link while its argmin still depends on a live separator. Recovery links retain their block and measurement-slice labels. Before committing a link selected at time `tau`, compute the suffix CNOT transform `R(time_now <- tau)` and send that recovery quotient to every current block in the corresponding origin column. This is mandatory: merely XORing `committed_corr[t]` at the earlier gate cannot propagate a control recovery that is inferred only after that gate.

```julia
function commit_time_tagged_projection!(g, E_old)
    suffix = suffix_gate_transforms(g.gate_layers)
    for (tau, origin, edge) in horizontal_and_carry_links(E_old)
        destinations = columnbits(suffix[tau], origin)
        for b in eachsetbit(destinations)
            toggle_edge_in_quotient!(g.committed_corr[b], edge)
        end
    end
end
```

The optimized implementation aggregates equal `(tau,origin)` links—including canonical carry links—into quotients before transport, but must be syndrome-and-winding equivalent to the loop above under the fixed canonical edge convention. `E_keep` is expressed at the new window's cut slice after applying all gate factors up to that cut, and only its uncommitted remainder is canonicalized into the next window's carry links. After backtracking, the finite-window procedure commits `E_old` by this transported projection and deletes all solver tables. An approximate mode may run `r` min-sum BP sweeps, but `r` is not a parameter of exact elimination.

Cleanup first calls `close_gate_layer_if_open!`, so a gate-only burst is compiled even when no later noisy round occurs. It then sets `p=q=0`, appends exact measurements, and solves the final partial/full window. At finalization, transport and commit **all** remaining selected horizontal recovery links, canonical carry links, and their homology through later gates, then clear the carried objects. Cleanup succeeds only when the carried relative boundary is empty, carried homology has been committed and reset, and no selected live projection or factor obligation remains; zero boundary alone is insufficient. Continue ideal slices until this condition holds or the cleanup cap is reached. Cleanup failure remains separate.

### 7.5 Readout

After final solve/cleanup,

```text
decoded_state[b] = canonical(physical[b] xor committed_corr[b]
                             xor selected_live_projection_at_readout[b]).
```

The selected live projection—including any uncommitted canonical carry links—is reconstructed from exact backpointers and transported through every later gate between each selected link's time and the readout slice, using the same suffix-transform routine as commitment. Homology is not stored as an extra `O(B)`-bit annotation on every table assignment: the two winding bits per affected block are explicit binary separator variables and are therefore already included in `vars` and `w_full`. On successful cleanup the selected projection has zero open boundary. The CNOT compiler makes control output unchanged and target output receive the control chain. Logical failure is the winding of this decoded quotient; cleanup failure is not included automatically.

### 7.6 Classical overhead and the two width notions

Define

```text
w_c(t)    = induced width of the logical-block/CNOT dependency skeleton
w_full(t) = induced width of the complete L x L x 2Z_FG factor graph,
            including every live homology-sector separator variable.
```

Exact tables use `w_full`. Thus the `2^w_full` table count already pays for homology sectors; implementations must not attach an uncounted `B`-bit homology vector to each assignment. Even one block can have width growing with `L` and `Z_FG`; `w_c<=2` does not make the solver cheap. Define

```text
N        = Theta(BL^2Z_FG) ordinary variables/factors in the dense window
G        = sum_tau g_tau <= 2BZ_FG changed logical rows in live gate layers
I_gate   = Theta(L^2 sum_{tau,b changed} [1 + weight(R_tau[b,:])])
         <= O(G B L^2) explicit gate-factor incidences
J_old    = number of selected horizontal or canonical carry links transported
           at one commitment
         <= O(BL^2Z_FG)
b_cost   = O(log N + log(max(w_p,w_q)+1)) bits per exact accumulated cost
T_tr     = O(Z_FG B^3/W + J_old B)
```

`T_tr` is a conservative suffix-transport bound: materialize one `B x B` suffix map per measurement slice by dense GF(2) composition, then scan the appropriate column for each selected link. A streamed/sparse implementation may be faster, but its measured cost must replace rather than omit this term. Let `N_bag` be the number of elimination bags.

```text
memory exact = O(BL^2 + G B/W + B^2Z_FG/W + N + I_gate
                 + N_bag 2^w_full ceil(b_cost/W)) words
window-solve runtime = O(N + I_gate + N_bag 2^w_full C_arith(b_cost) + T_tr)
amortized runtime/round = O([N + I_gate + N_bag 2^w_full C_arith(b_cost)
                              + T_tr]/Z_FG + BL^2)
one CNOT immediate = O(L^2 + B/W)
isolated identity-layer CNOT close = O(L^2) constant-arity incidences
general layer close = Theta(I_gate_layer); one dense changed row alone can
                      require Theta(BL^2) incidences
one CNOT eventual solve effect = as large as the entire window-solve bound
                                if it raises width across many bags
cleanup = O(N_cleanup + I_gate_cleanup
            + N_bag_cleanup 2^w_full_peak C_arith(b_cost)
            + T_tr_cleanup)
GC/window compaction = O(N + I_gate + B^2Z_FG/W + N_bag 2^w_full)
bounded width-certificate ring = O(R_width) words.
external width-certificate stream = O(N_width_cert) words total.
```

`C_arith(b_cost)` is the bit-complexity of exact integer add/compare; it is `O(1)` only when a proved fixed-width integer bound applies. The explicit `XorFactor.vars` representation pays `I_gate`. A lazy structured gate factor may store only `G B/W` row bits, but exact elimination must still account for its induced scopes, and BP must either materialize or stream one message per incidence; it cannot claim `O(N)` independently of row weight.

For bounded `w_full<=K`, peak memory is

```text
O(BL^2Z_FG + B^2Z_FG/W + I_gate
  + BL^2Z_FG 2^K ceil(b_cost/W)),
```

independent of completed `D` because only `2Z_FG` rounds remain. But DKLP-like accuracy with `Z_FG>>L` already gives at least `Omega(BL^3)` raw-window scaling, before exponential separator tables. A specialized polynomial matching solver might remove the spatial-treewidth exponential for ordinary within-block graphs; whether CNOT correlation factors preserve such a decomposition is an open research question and must not be assumed.

Loopy BP stores `O(N+I_gate)` variables, factors, and incidence messages and has `O(r(N+I_gate))` sweep time when each XOR factor uses linear-time prefix/suffix parity updates; it is heuristic. On `w_full>K`, `:stop` increments `width_overflow_count`, stores a bounded hash/count certificate, and terminates the trial without readout. `:bp` additionally increments `approximation_certificate_count`, records iteration count/residual, discards exact tables, and produces an explicitly approximate result. The fixed resident certificate ring is streamed or overwritten, so resident diagnostics do not accumulate with `D`; a retained external stream costs `Theta(N_width_cert)` words and must be counted or aggregated separately.

### 7.7 Required repeated-CNOT cases

| schedule | causal-width and depth behavior |
| --- | --- |
| repeated `C1 -> C2` | `w_c<=2`, not `w_full<=2`. Gate-only repetitions in one measurement slice compose; completed gate slices leave memory after the overlapping window advances. |
| alternating `C1 -> C2, C2 -> C1` | Still `w_c<=2` and no Fibonacci paths, while full spatial width remains `L,Z_FG` dependent. |
| fanout `C1 -> C2, C1 -> C3` | The block skeleton is a small-width star with a good order; coupling it to spatial recovery can still enlarge `w_full`. |
| long chain | The block skeleton is path-like with constant `w_c`; raw graph memory is `Theta(BL^2Z_FG)` and full solver width need not be constant. |
| dense CNOTs with no idle rounds | Gates within one measurement slice compose into one layer. Dense gate slices separated by noise can make `w_c=Theta(B)` and further enlarge `w_full`. |
| CNOTs separated by `O(L)` idle rounds | If `Z_FG>>L`, several gates may still share one global window. Only exact window advancement removes their factors; idle time reduces CNOT-induced width but not spatial width. |

Depth control is exact for a fixed window: peak memory depends on `B,L,Z_FG,w_full,K`, active graph sparsity, fanout, and live gate slices, not all past `D`. The price is poor scaling in `L` and possible `2^w_full` behavior.

### 7.8 Performance, failure modes, implementation, and validation

Joint minimum-weight inference can use control-target correlations and might improve target performance beyond independent histories. It may also perform worse than Lake/sheet-copy because a finite global window, tie rule, or approximate BP chooses poor homology. Target failures need not remain dominant, but the target carries the denser factor neighborhood.

Failure modes include a wrong CNOT interface, propagation of noisy measurements as physical state, inconsistent pre/post-gate registers, mutable aliasing between pre/post, carried, or committed buffers, omission of homology separator variables, undefined connectivity at a CNOT hyperfactor, committing before backtracking, failing to transport a recovery inferred after a gate, confusing a minimum-cost chain with a maximum-probability homology class, inadequate `Z_FG`, dense gate-incidence blowup, full-treewidth explosion, exact-integer cost growth, BP nonconvergence, an incomplete width certificate, archive failure, and silent fallback. Committed control correction must not also remain in an unresolved factor.

Reject exact mode if exhaustive rational-weight costs/assignments disagree, if transport differs from explicit propagation, if `I_gate` or `w_full` is impractical even at target `L`, or if required `Z_FG>>L` eliminates the desired space advantage. Reject BP if its certificates correlate strongly with failure or its threshold trend is worse than primitive.

**Difficulty: research-scale.** This is a new decoder mode with graph compilation, CNOT parity factors, exact integer min-sum tables, homology bookkeeping, exact backtracking, overlapping-window commitment, recovery transport, and width instrumentation. The baseline remains unchanged. A deterministic very-small-`L` prototype is mandatory before performance engineering.

In addition to Section 11, validate:

- exhaustive minimum-weight chains for `L=3`, one/two rounds and every homology sector;
- the CNOT interface compiler against Eq. (72) hand cases;
- gate-layer flush exactly once before the next measurement and at finalization of a gate-only burst;
- exact elimination/backtracking against brute force using the same compiled elimination/backtracking tie order; alternative orders must agree on optimum cost, while tied decoded chains may differ and are reported as such;
- committed `E_old` plus carried boundary against an explicit overlapping-window reference;
- a two-window witness in which a first-window carry link closes and commits in the second window, with no lost or double-committed boundary/homology;
- augmented explanation components with zero, two, and four terminals, plus components crossing a CNOT factor and terminal-free cycles;
- a delayed-recovery witness in which a pre-gate control fault is inferred only after the gate and its recovery must reach the target;
- time-tagged suffix transport and live readout projection against explicit one-gate-at-a-time propagation for every stress pattern;
- separate `w_c(t)` and `w_full(t)` traces and full-width certificates;
- exact integer-cost tables against exhaustive rational-weight enumeration and deliberate width overflow under both `:stop` and certified `:bp`;
- measured `I_gate`, `J_old`, suffix-map bytes, and `T_tr` against the derived bounds;
- BP versus exact whenever `w_full<=K`;
- gate-layer composition equivalence to one-at-a-time factors.

## 8. Candidate family: symbolic min-plus decision-diagram ensemble

### 8.1 Core idea and exactness status

Represent the entire materialized sheet ensemble as functions from a lineage key to each Boolean or integer state entry. Reduced algebraic decision diagrams (ADDs) hash-cons identical subfunctions. A CNOT clone adds a lineage-key branch that initially points to the same immutable roots; subsequent baseline operations are applied symbolically to all lineages at once.

This is neither a sheet array nor a fixed number of lanes. It is a symbolic program for the sheet ensemble. It can exploit enormous sharing during gate-only bursts and exposes whether the nonlinear decoder actually has a compact quotient.

**Exactness classification: exact relative to sheet-copy semantics.** With untruncated canonical diagrams, exact integer leaves, and explicit per-lineage random masks/choices, it reproduces the literal sheet-copy transition and readout for a fixed random input table. Its probability distribution is exact if those table entries are independent. A counter-based pseudorandom generator is practical but should be described as the simulator's RNG semantics. A block-shared-noise variant is possible, but then the target semantics changes to Section 2.6.

### 8.2 Core representation

```julia
struct DDNode{T}
    variable::UInt64             # lineage/path decision variable; no 65,535 cap
    low::Int64                   # arena node id
    high::Int64
    leaf::T                      # used only for terminal nodes
end

mutable struct HashConsArena{T}
    nodes::Vector{DDNode{T}}
    unique::Dict{Tuple,Int64}
    apply_cache::Dict{Tuple,Int64}
end

struct DDArray{T,N}
    roots::Array{Int64,N}        # dense spatial roots, symbolic lineage values
end

mutable struct SymbolicSheets
    bool_arena::HashConsArena{Bool}
    int_arena::HashConsArena{Int}
    lineage_set::Int64           # id in bool_arena
    block_label::Int64           # id in int_arena
    state::DDArray{Bool,3}       # L x L x 2 roots
    corr::DDArray{Bool,3}
    old_syndrome::DDArray{Bool,2}
    syndrome::DDArray{Bool,2}    # current new_synds contribution
    hist::DDArray{Bool,3}        # L x L x Z roots
    fields::DDArray{Int,5}       # L x L x Z x 3 x 2 roots
    new_fields::DDArray{Int,5}
    hist_correction::DDArray{Bool,4}
    settled_parity::BitMatrix    # optional canonical terminal reduction
end
```

Spatial root arrays are dense and mutable, but all diagram nodes are immutable and shared. Every Boolean root id is owned by `bool_arena`; every integer root id is owned by `int_arena`. Although `old_synds`, `hist_correction`, and `new_fields` are semantically redundant at a completed-round boundary, literal `sheet_active` tests them and literal `deepcopy` copies them. Exact-to-file mode therefore represents all three. A second, explicitly named cleaned-semantics mode may omit/pool them, but it is not the literal reference. Path variables themselves are the canonical lineage identity. The file's numeric parent/gate IDs do not affect activity, transition, or readout; a small-test audit table may enumerate them externally, but no unbounded parent list is part of the compressed decoder.

An integer field root maps every live lineage key to its exact field value at one spacetime/direction cell. A Boolean history root does the same for a defect. Reduced nodes merge exactly identical lineage subfunctions; no approximate bond truncation is allowed in exact mode.

### 8.3 CNOT update

Literal sheet-copy clones only active control sheets. Symbolically compute

```text
active(lineage) = OR over all eight literal mutable sheet arrays being nonzero
control_active  = lineage_set AND (block_label == c) AND active.
```

Introduce a fresh path variable for `gate_id`. The low branch denotes the existing parent; the high branch denotes a renamed clone whose block label is `t`. Every persistent state root on the high branch initially points to the parent's same immutable subdiagram.

| item | CNOT action |
| --- | --- |
| physical state | high-branch roots share the control state subfunction |
| state correction | same immutable branch sharing |
| old syndrome | copied by root sharing because literal `sheet_active` and `deepcopy` include it |
| new syndrome | copied by sharing its root |
| defect history | copied by sharing roots |
| history correction | copied by root sharing in exact-to-file mode, then cleared when the next update does so |
| message fields | copied by root sharing |
| new fields | copied by root sharing in exact-to-file mode because it participates in `sheet_active` |
| lineage/dependency | new decision variable, parent relation, and target block-label branch |
| settled information | literal mode retains every sheet; parity reduction is allowed only at terminal readout, not for continued evolution |

```julia
function symbolic_cnot!(s, c, t, gate_id)
    active = symbolic_sheet_active(s)
    selected = s.lineage_set & (s.block_label .== c) & active
    v = fresh_lineage_variable!(s, gate_id)
    s.lineage_set = disjoint_clone_union(s.lineage_set, selected, v)
    s.block_label = clone_with_target_label(s.block_label, selected, v, t)
    wrap_all_persistent_roots_with_shared_clone!(s, selected, v)
end
```

Computing `active` touches all root families and costs at least proportional to the relevant diagram operations; the clone itself is pointer-level. No mutable aliasing exists because nodes are immutable.

### 8.4 Normal update, noise, and cleanup

Every scalar Boolean/integer operation in baseline `update!` becomes a canonical DD `Apply` or `ITE` operation:

1. six-direction min-plus field propagation combines neighboring integer roots;
2. feedback builds symbolic predicates for positive minimum and tie priority;
3. symbolic correction roots XOR history endpoints and correction chains;
4. a per-lineage, per-cell random terminal function supplies independent data and measurement bits, reproducing current sheet behavior intentionally;
5. old/current syndrome roots produce the new front history root;
6. RG root arrays shift, back-wall roots use symbolic XOR/nonzero-min, and the front clears.

Back-wall `0.8` choices are Boolean DD functions keyed by `(lineage,round,i,j,k)`. A deterministic input-table mode must be used for comparisons with explicit sheets.

Feedback, correction application, and noise therefore remain independent for each symbolic lineage. No physical-block noise sharing is claimed in this exact mode. Cleanup applies the same symbolic transition with zero data/measurement noise but retains per-lineage back-wall random choices. Literal sheets remain eligible for future independent noise, so `hist==0` does not permit mid-circuit lineage deletion. Hash-cons GC removes arena nodes unreachable from current roots, not logical lineages still present in `lineage_set`.

### 8.5 Readout

For each block and each physical edge, restrict the lineage set to that block and parity-quantify the lineage variables:

```text
decoded_edge[b,i,j,o]
    = XOR_{lineage: block(lineage)=b}
        (state[lineage,i,j,o] xor corr[lineage,i,j,o]).
```

Parity quantification over a reduced DD gives a Boolean root/leaf for the sampled trial without enumerating lineages when the diagram stays compact. The resulting dense array is passed to the unchanged syndrome and logical-winding checks. All histories must be empty lineage-wise for cleanup success. Cleanup failure remains separate.

The parent control branch remains in the control restriction, while the cloned branch is added only to the target restriction; hence control readout is unchanged and target readout is target XOR control, exactly as in the explicit sheet reference.

### 8.6 Classical overhead

Let `chi_state(t)` count nodes reachable from the spatial state-array roots and let `P(t)` count nodes reachable from lineage-set/block-label roots as diagnostics; these sets may overlap and are not added for memory accounting. Let `R_union(t)` be the cardinality of the union reachable from **all** spatial, lineage/block, and retained random-function roots. Let `chi_alloc_total(t)` count all allocated arena slots and unique-table buckets exactly once, including unreachable nodes awaiting rebuild. `chi_peak` is the maximum input/result/intermediate diagram size during one operation, while `C_apply` and `C_rng_table` count memo/random lookup entries outside the node arenas. Finally let

```text
R_root = Theta(L^2Z)
```

be the number of spatial roots across persistent arrays. Then

```text
retained memory = Theta(R_root + chi_alloc_total(t) + C_apply + C_rng_table)
peak memory = Theta(R_root + chi_alloc_total_peak
                    + C_apply_peak + C_rng_table_peak).
```

After each completed subphase, or whenever `chi_alloc_total>2R_union`, trace the union of all `R_root`, lineage/block, and retained RNG roots, rebuild arenas/unique tables, and rewrite root IDs. The rebuild atomically clears all Apply caches and random-root lookup tables; counter keys regenerate random functions, so no stale node ID survives. Afterwards `chi_alloc_total=Theta(R_union)`. A non-rebuilding arena can retain `Omega(D)` dead nodes. Here `D_live` is the number of successful clone variables not eliminated by terminal parity reduction. `P(t)` is not free even though it is not double-counted: literal live lineage identity generally gives `P(t)=Omega(D_live)` and can be `Theta(S(t))`, so exact DD semantics do not meet the depth-independent target when lineages never become terminal.

Each individual binary `Apply` uses a complete operation-local memo table that is not evicted until that Apply finishes; this is what justifies the product bound below. A separate cross-operation cache has fixed cap `K_cache` and may be cleared after each subphase. Evicting the operation-local memo can cause repeated subproblems and invalidates the stated bound. Random-function lookup tables are subphase scoped or included in `C_rng_table`; an explicit worst-case table has `C_rng_table_peak=O(SL^2Z)` entries, while counter generation can reduce retained entries but not the symbolic branch complexity.

A binary `Apply` has worst-case time proportional to the product of its input diagram sizes. Conservative explicit bounds are

```text
runtime per round = O((r+1)L^2Z * chi_peak^2)
one CNOT = O(L^2Z * chi_peak^2) worst case for active Apply/ITE/wrappers
readout = O(B L^2 * chi_peak^2) for restriction/parity quantification
cleanup C rounds = O(C(r+1)L^2Z * chi_peak^2)
arena tracing/rebuild GC = O(R_root + chi_alloc_total
                             + C_apply + C_rng_table).
```

Costs can be much lower when roots share ordering/subgraphs. The bounds deliberately use peak intermediate size rather than retained `chi_state(t)`.

There is no uniform compression theorem. Independent post-copy noise can make every lineage state different. In adversarial orderings this can force `R_union=Omega(S(t)L^2Z)` for `S(t)` explicit sheets; a naive explicit trie gives the generic upper bound `O(S(t)D_live L^2Z)`. Thus alternating CNOTs can recover at least explicit-sheet-scale exponential memory and symbolic `Apply` may be worse. A gate-only circuit with no intervening updates can instead keep state roots compact, but lineage metadata is still generally `Omega(D_live)` and may be about `O(B^2+D)` in favorable orderings because literal multiplicities must remain distinguishable.

### 8.7 Required repeated-CNOT cases

| schedule | expected diagram behavior |
| --- | --- |
| repeated `C1 -> C2` | Explicit sheet count is linear. With no update, shared roots make diagram growth mostly lineage metadata; independent noisy rounds tend to make `R_union=Omega(D)`. |
| alternating two-block CNOTs | Gate-only paths may reduce through identical roots and parity. Independent noise after each gate destroys equivalence; worst case follows `Theta(phi^D)`. |
| fanout | The same control subdiagram is shared by multiple clone branches. Noise on descendants determines whether sharing survives. |
| long chain | Initially excellent prefix sharing; after block-specific independent sheet noise, diagram width can grow toward the explicit `Theta(B^2)` sheet count. |
| dense CNOTs with no idle rounds | Best case and principal research motivation: many clone operations, few mutations, compact shared roots, no dependence on enumerated paths. |
| CNOTs separated by `O(L)` idle rounds | Under literal per-sheet noise this is often the worst case for compression: descendants acquire independent random histories before the next gate. Terminal parity reduction may recover space only after cleanup. |

Overhead depends on DD width/node count, which in turn depends on causal-treewidth-like variable ordering, fanout, and independent random entropy. It may depend exponentially on `D`. There is no fixed cap unless truncation is added, in which case the method becomes approximate.

### 8.8 Performance, failure modes, implementation, and validation

Untruncated symbolic execution has exactly the same logical performance and target/control imbalance as literal sheet-copy for the same random table. It can beat the primitive decoder because control and target message fields never undergo the primitive's destructive XOR/nonzero-min merge. This is valuable as an exact compression experiment but preserves the current nonphysical multiple-noise behavior. The block-shared-noise variant would need separate validation and would converge conceptually toward the epoch transducer.

Failure modes are catastrophic DD width growth, bad lineage-variable order, expensive `Apply` cross-products, hidden mutable root/arena aliasing, lineage-key collision, failure to clone exactly the active subset, incorrect parity quantification, stale or unbounded apply caches, inconsistent syndrome roots, random functions correlated across lineages, integer leaf overflow, and accidental truncation. There is no destructive history merge or configured overflow in exact mode; alternating/dense noisy circuits instead cause explicit diagram blowup. Hash-cons keys must compare complete node tuples.

Reject this direction if `R_union` tracks more than a fixed substantial fraction of explicit sheet payload at `D<=8` in the alternating noisy test, or if one symbolic round is more expensive than explicit sheets before memory savings appear. Its most informative outcome may be a negative compression result.

**Difficulty: research-scale.** Nearly every baseline scalar operation needs a symbolic analogue. New canonical arenas, variable ordering, parity quantification, random-input functions, and memory tracing are required. The baseline and explicit sheet-copy implementations should remain unchanged as deterministic oracles. This must be a separate small prototype before any production integration.

In addition to Section 11, validate:

- all Boolean and min-plus `Apply` primitives exhaustively;
- symbolic versus explicit sheets after every subphase, not only readout;
- block-label roots and optional enumerated parent/gate audit metadata against the explicit lineage table after every gate;
- literal active-sheet predicate equivalence for sheets active solely through each of `old_synds`, `new_fields`, and `hist_correction`;
- parity readout on hand-built duplicate/canceling lineages;
- variable-order invariance of decoded output;
- exact node-count and peak-byte traces for every repeated-CNOT pattern;
- cache-cap/eviction invariance and accounting of unique/apply/random tables;
- arena rebuild invariance, stale-ID rejection, cache clearing/remapping, and a flat `chi_alloc_total/R_union` envelope across repeated allocate/GC cycles;
- operation-local Apply memo retention sufficient for the claimed product bound;
- independent random-table consumption per literal sheet.

## 9. Candidate family: quiescent circuit barriers with streamed GF(2) gates

### 9.1 Core idea and exactness status

Restrict the schedule so no unresolved decoder history crosses a CNOT layer. Before a gate layer, finish ideal decoding, reduce every block to a syndrome-free logical frame, reset the dynamic decoder state, then stream all CNOT row operations through the frame. Resume noisy decoding only after the layer.

This is a scheduling solution, not lineage compression. It tests whether bounded live causal width can be guaranteed architecturally.

**Exactness classification: exact under restricted CNOT schedules.** It is exact for the explicitly defined **terminal-reset barrier decoder** if (i) cleanup is noiseless, (ii) every involved `hist` empties and decoded boundary is zero, (iii) the barrier intentionally canonicalizes residual fields to zero after folding logical output, (iv) no noise occurs inside the gate-only layer, and (v) the CNOT is ideal. The field reset is a decoder transition, not exact continuation of baseline: baseline fields can remain nonzero after `hist` empties and affect later defects.

### 9.2 Core representation

```julia
mutable struct BarrierBlock
    baseline::BaselineArrays             # ordinary one-block decoder
    logical_frame::BitVector              # length 2, syndrome-free winding
    cleanup_failed::Bool
end

mutable struct BarrierCircuit
    blocks::Vector{BarrierBlock}          # length B
    pending_transform::Vector{BitVector}  # B row-packed rows; audit only
    in_gate_layer::Bool
    gate_count::Int
    overflow_count::Int                  # failed barriers/refused gates
    trial_terminated::Bool               # partial failed barrier is never resumed
end
```

The baseline state is dense, mutable, and not shared. `pending_transform` is audit-only and can be omitted if gates are immediately applied to realized winding columns; applying it later would double-propagate the frame. There are no lineage objects or copied fields. At a successful barrier, physical/correction chain interiors, syndrome registers, defect history, and fields are discarded after syndrome and winding commitment. Field discard is the named terminal-reset rule.

### 9.3 Barrier and CNOT update

```julia
function enter_barrier!(d; cleanup_cap)
    for block in d.blocks
        prepare_perfect_measurement_closure!(block.baseline) # one q=0 observation
        for step in 1:cleanup_cap
            baseline_round!(block.baseline; p=0, q=0, synch=true)
            cleanup_terminal(block.baseline) && break
        end
        decoded = quotient(block.baseline.state)
        xor_quotient!(decoded, quotient(block.baseline.state_correction))
        if !cleanup_terminal(block.baseline) || !iszero(decoded.boundary)
            block.cleanup_failed = true
            d.overflow_count += 1
            d.trial_terminated = true
            return false
        end
        block.logical_frame .⊻= decoded.winding
        apply_recovery_and_canonicalize_physical!(block.baseline)
        terminal_reset_all_arrays!(block.baseline) # includes defined fields->0
    end
    d.in_gate_layer = true
    return true
end

function cnot!(d, c, t)
    @assert d.in_gate_layer
    d.blocks[t].logical_frame .⊻= d.blocks[c].logical_frame
    d.pending_transform[t] .⊻= d.pending_transform[c] # audit only, row packed
    d.gate_count += 1
end
```

Field-by-field action after a successful barrier:

| item | action |
| --- | --- |
| physical state | apply the accumulated recovery in the noiseless barrier, reduce the resulting closed chain to its logical winding, then reset to a canonical zero representative |
| state correction | physically applied/folded into the winding frame, then reset |
| old/new syndrome | reset to zero after terminal consistency check |
| defect history | must be empty; then released/reset |
| history correction | scratch cleared |
| fields/new fields | explicitly canonicalized to zero by the terminal-reset transition; not baseline continuation |
| lineage/dependency | optional row update in `pending_transform` |
| settled information | `logical_frame[t] xor= logical_frame[c]` |

Here `cleanup_terminal` requires empty defect history **and** consistency of the last syndrome register with the final perfect measurement; an empty `hist` before that closing measurement is insufficient. `apply_recovery_and_canonicalize_physical!` is part of the terminal-reset barrier semantics: it applies the accumulated X recovery during the assumed noiseless pause and replaces any contractible closed remainder by a stabilizer-equivalent representative. It is not merely deletion of a still-syndromic physical state. Control is unchanged; target receives control winding exactly. A failed barrier prevents the gate in exact mode and increments `cleanup_failures`/`overflow_count`; it is not silently approximated. Because blocks are processed in place, failure terminates the trial immediately and no partially reset state may be resumed or read out. A retry-capable implementation must instead stage all `B` results and commit atomically, paying up to another `Theta(B M_block)` transient bytes.

### 9.4 Normal update and readout

Outside a gate layer, every block uses the unchanged baseline synchronous decoder. Data and measurement noise are sampled once by that one block update. Feedback, correction, RG cycling, and back-wall behavior are baseline between barriers. A gate layer contains no noisy or RG rounds. Exiting a layer resets the audit matrix to identity without applying it and resumes clean arrays.

Readout combines the current baseline decoded quotient with `logical_frame`:

```text
decoded[b] = canonical(quotient(state[b] xor state_correction[b])
                       xor logical_frame[b]).
```

The logical-failure rule is unchanged. Barrier cleanup failure and final cleanup failure are reported separately from winding failure.

### 9.5 Classical overhead

```text
memory = B * M_block + B^2/8 + O(B) bytes
round runtime = Theta(B(r+1)L^2Z)
one streamed CNOT = Theta(1) for realized frames
                  + Theta(B/W) if pending_transform is retained
barrier cleanup actual = Theta(U_block * sum_{b=1}^B C_b),
                         where C_b<=C_cap is that block's ideal-round count
barrier cleanup worst = Theta(B C_cap U_block)
barrier cleanup typical = Theta(B E[C_b] U_block) for representative/iid blocks
GC/reset = Theta(B L^2Z) for ordinary dense clearing;
           O(B) only with a specified lazy-zero sentinel/generation-stamp buffer
```

Peak memory is independent of `D`. Cumulative runtime must still be at least `Omega(D)` to process `D` gates. If every CNOT needs its own barrier with `C=Theta(L)`, gate overhead becomes `Theta(D B(r+1)L^3Z)`, which is likely unacceptable.

### 9.6 Required repeated-CNOT cases

| schedule | behavior |
| --- | --- |
| repeated `C1 -> C2` | All gates in one quiescent layer stream in `O(D)` time and fixed memory; even repetitions cancel in the GF(2) frame. A barrier per gate adds `D` cleanups. |
| alternating two-block gates | Fixed memory and correct streamed row operations; no unresolved decoder dynamics may occur between them. |
| fanout | Constant-size frame updates per target; optional dependency matrix is `Theta(B^2)` bits. |
| long chain | `Theta(B)` gate work and `B*M_block` memory; the frame propagates down the chain exactly. |
| dense CNOTs with no idle rounds | Supported only as a noiseless gate-only layer entered from a successful barrier. Dense noisy interleaving is outside the exact contract. |
| CNOTs separated by `O(L)` idle rounds | If those rounds are specified as ideal cleanup, barriers likely succeed; if they contain physical noise, terminality is not guaranteed and the schedule restriction is not met. |

Overhead depends on `B`, `L`, `Z`, barrier frequency, and cleanup time, not on live causal width or completed `D`. Fanout/treewidth do not affect memory beyond the optional `B^2` transform. The restriction, rather than a cap, supplies boundedness.

### 9.7 Performance, failure modes, implementation, and validation

When barriers succeed, pre-gate histories are decoded before propagation, so target degradation from destructive merging should disappear. Between barriers the rule is baseline; the canonical field reset changes future feedback and must be measured. Performance may resemble independent blocks plus propagated frame failures, but this is a hypothesis. Target failures should not be intrinsically dominant beyond inherited control error if reset is benign.

The central weakness is physical/causal realism. A decoder cannot instantaneously obtain future perfect syndrome information; noiseless cleanup is a scheduled resource. With ongoing noise a barrier may never be quiescent. Other failure modes are destructive reset before history empties, stale fields surviving a promised reset, gate-matrix/lineage mistakes, folding nonzero syndrome, incorrect correction forwarding, mutable aliasing across blocks, a noisy round inside a layer, dense alternating schedules outside the contract, and confusing cleanup with logical failure. Approximation/lossy merging are not used: a failed barrier refuses the gate and records overflow.

Reject this family as a computation architecture if barrier latency or failure probability eliminates the threshold advantage, if the intended circuit cannot supply noiseless/paused intervals, or if `O(L)` cleanup is required before most gates. It remains useful as a controlled upper bound on what schedule-enforced causal width can achieve.

**Difficulty: medium.** It can be a wrapper mode around unchanged baseline blocks. New pieces are the barrier controller, chain quotient/canonicalization, logical-frame propagation, and result metrics. No baseline field function needs modification. Deterministic reference comparisons are straightforward with fixed masks and a fixed cleanup random stream.

In addition to Section 11, validate:

- barrier folding versus continuing the same block to terminal readout;
- refusal to execute a gate after cleanup failure;
- failure on a later block terminates the trial and forbids readout/resume of earlier partially reset blocks;
- arbitrary gate-layer matrices against direct bit-vector multiplication;
- deterministic equality to an explicit terminal-reset reference for every later noise trace (not equality to stale-field baseline continuation);
- direct comparison of intentional terminal reset against baseline stale-field continuation;
- latency and success distributions for cleanup caps `2L` and `L^2`.

## 10. Candidate family: certified capped epoch ensemble

### 10.1 Core idea and exactness status

Place a hard global capacity of `K*B` **materialized** epoch payloads on the incidence-vector design. Run exactly until capacity would be exceeded. First attempt only proven exact canonical merges. If none exists, either stop with an overflow record or invoke one explicitly named lossy projection and issue a machine-readable approximation certificate.

This family is intentionally the bounded labeled-state baseline. Incidence epochs and sparse replay are also acknowledged as adjacent to the listed shared-recovery/forwarding mechanisms. The independently distinct half of the design set is the factor graph, symbolic decision diagram, and quiescent barrier families.

**Exactness classification: approximate with an explicit approximation.** Before the first lossy projection it is exact relative to the Section 5 epoch semantics. `:stop` is exact but incomplete. `:project` produces a result for the original schedule but is approximate and increments both `lossy_merge_count` and `approximation_certificate_count`. No gate or noisy round is partially applied before its capacity action is chosen.

### 10.2 Core representation

```julia
struct ApproximationCertificate
    transaction_id::UInt64
    record_phase::Symbol            # :prepared; optional :committed marker follows
    trial::Int
    gate_id::Int
    round::Int
    reason::Symbol                 # :capacity, :width, :cleanup_timeout
    policy::Symbol                 # :stop or :project
    decoder_mode::Symbol           # :oracle or :total_syndrome
    epoch_ids::Vector{UInt64}
    supports::Vector{BitVector}
    active_defect_counts::Vector{Int32}
    backwall_defect_counts::Vector{Int32}
    state_hashes::Vector{UInt128}
    current_quotient_hashes::Vector{UInt128}
    victim_ids::Vector{UInt64}      # empty for :stop/no eligible victim
    projection_rules::Vector{Symbol}
    projection_versions::Vector{UInt32}
    discarded_defect_hashes::Vector{UInt128} # hashes, not copied defect sets
    closure_or_fold_hashes::Vector{UInt128}
    pre_aggregate_hash::UInt128
    expected_post_aggregate_hash::UInt128
end

struct LazyNative
    block::Int32                   # zero-payload sentinel; one per block
end

struct ProjectionPlan
    victim_id::UInt64
    projection_rule::Symbol
    projection_version::UInt32
    closure_or_fold_hash::UInt128
    expected_post_aggregate_hash::UInt128
end

struct StagedProjectionBatch{F,E}
    replacement_settled::F       # oracle BitMatrix or total Vector{ChainQuotient}
    retained_epochs::E           # newly allocated vector of retained pool-slot ids
    replacement_live_slots::BitVector
    replacement_free_slots::Vector{Int32}
    new_materialized_count::Int
    lossy_increment::Int
    certificate_increment::Int
    expected_post_aggregate_hash::UInt128
end

mutable struct CappedEpochDecoder{D,P}
    core::D                         # mode metadata; live epochs are pool-slot ids
    capacity::Int                  # K * B
    overflow_policy::Symbol        # :stop or :project, fixed for the run
    payload_pool::Vector{P}        # exactly capacity preallocated mode payloads
    live_slots::BitVector
    free_slots::Vector{Int32}
    recent_certificates::Vector{ApproximationCertificate} # fixed ring R_cert
    certificate_ring_capacity::Int
    certificate_cursor::Int
    certificate_sink::Function    # stream each record; no unbounded resident log
    overflow_count::Int
    lossy_merge_count::Int
    approximation_certificate_count::Int
    lazy_native::Vector{LazyNative} # fixed O(B), not in materialized capacity
    materialized_count::Int
    trial_terminated::Bool
end
```

Each live epoch is a slot containing the mode-appropriate dense `DecoderEpoch` or `TotalEpoch` from Section 5 with pooled scratch. The pool is allocated once with exactly `K*B` payload slots and never grows; `core.epochs` and native references are slot IDs. Projection/GC marks victim slots free, and later materialization reinitializes those slots using generation stamps or complete overwrite, so unreachable Julia allocations cannot cause a transient breach of the resident cap. `materialized_count=count(live_slots)`; free pool capacity remains included in resident memory. A collision-checked canonical table indexes the complete persistent state plus support and future-input class. Under shared-recovery semantics, two frozen byte-identical epochs with identical support and RNG continuation contribute twice to an XOR and may be removed as an exact pair; no unspecified multiplicity object is used. Hash equality alone never authorizes removal.

Every certificate owns deep copies of its support bit vectors and scalar/hash arrays; it never aliases live epoch metadata. Its bounded ring may overwrite old records only after the sink has accepted them.

The specified lossy projection is deliberately simple and auditable, but it is mode-specific. In oracle mode, choose the oldest frozen epoch with the fewest active defects, form its current decoded quotient `q=e.raw xor e.corr`, close `boundary(q)` by a fixed spanning-tree chain, fold the winding of the resulting closed chain into every supported block, and remove the epoch. This deliberately declares its open syndrome obligation resolved and ignores delayed messages. In total-syndrome mode, choose by the same rule, XOR the epoch's **full current** `e.corr` quotient into `settled_corr[b]` for every supported block, then discard its history and fields. The total-mode fold preserves instantaneous readout exactly but discards all later correction that the unresolved state would have generated. Neither operation is exact cleanup. To keep certificates bounded, they record counts plus hashes of the discarded history/state, not the full defect coordinates; a full replay witness is an optional separately costed archive.

### 10.3 CNOT and normal update

CNOT transformation, physical/correction quotients, component syndrome, fields, and support follow Section 5 exactly. The CNOT itself adds no materialized payload: it freezes any materialized outgoing native, performs the support/settled/physical transforms, and installs the block's fixed `LazyNative` sentinel. A gate-only burst therefore never consumes capacity. Capacity is checked only before a later observed/noise event would materialize one or more lazy natives.

```julia
function reserve_materializations!(d, needed_blocks, context)
    needed = count(b -> native_is_lazy(d, b), needed_blocks)
    d.materialized_count + needed <= d.capacity && return true
    exact_canonicalize!(d.core)
    d.materialized_count + needed <= d.capacity && return true

    d.overflow_count += 1
    if overflow_policy(d) == :stop
        cert = make_stop_certificate(d, context)
        stream_and_store_recent!(d, cert)
        d.trial_terminated = true
        return false
    end

    # Transactional preflight: choose every victim and predicted fold first.
    plans = plan_distinct_frozen_projections(d,
                 d.materialized_count + needed - d.capacity)
    if plans === nothing
        cert = make_no_victim_certificate(d, context)
        stream_and_store_recent!(d, cert)
        d.trial_terminated = true
        return false
    end
    staged = stage_projection_batch(d, plans) # allocates replacement frames/vector
    aggregate_hash(staged) == staged.expected_post_aggregate_hash ||
        return terminate_without_mutation!(d)
    cert = make_prepared_certificate(d, plans, staged, context)
    stream_or_fail_without_mutation!(d, cert) ||
        return terminate_without_mutation!(d)
    commit_staged_noalloc!(d, staged) # pointer/scalar swaps only; updates counters
    try_stream_committed_marker!(d, cert.transaction_id) # nonthrowing; flags I/O
    return true
end
```

The driver first uses pure counter-keyed `peek_edge_mask`/`peek_measurement_mask` calls, which neither mutate block state nor advance RNG counters. From those masks/observations it determines which lazy natives need nonzero materialization and calls `reserve_materializations!` for the whole round. Only a successful commit applies the masks/events, allocates natives, and advances each block's data and measurement counter exactly once. A failed preflight therefore leaves the future random stream unchanged. The preliminary exact canonicalization may change object identities, but it is a proved representation-preserving transition. If there is no eligible frozen victim—possible for a very small cap—the trial stops rather than projecting a live input receiver. Cached state/quotient hashes are recomputed while the ordinary dense update already touches each payload; hashes accelerate auditing but never authorize an exact merge.

For projection, all replacement settled frames, the retained-epoch ownership vector, new materialized count, counter deltas, and post-hash are allocated and computed in `staged` without touching live objects. The post-hash is verified **before** the `:prepared` certificate is durably streamed. `commit_staged_noalloc!` then performs only prevalidated pointer/scalar swaps; it cannot fail halfway through a quotient fold because folds were performed in the staged copies. An optional constant-size `:committed` marker distinguishes process failure after preparation. This staged boundary—not mutation followed by hash checking—is the transaction guarantee.

`capped_cnot!` has no fallible capacity action: all support vectors are preallocated, the old native is frozen, and the fixed lazy sentinel replaces it. In total mode, the trial wrapper applies the block physical quotient only in the same nonthrowing commit phase. Thus a refused materialization cannot leave a half-applied gate or a block without a native receiver.

Field-by-field, no arrays are merged in the exact portion: mode-appropriate raw/correction and syndrome registers, dense history, and dense message fields stay separate; old syndrome and correction/new-field scratch are omitted/pooled; only support and settled information transform as in Section 5. Under `:project`, oracle mode performs the deterministic boundary closure and winding fold above; total mode folds the full current correction quotient into `settled_corr`. Both discard unresolved history/fields, and the bounded certificate makes the lost future continuation explicit through counts and hashes.

| item | exact capped CNOT action; projection action if capacity overflows |
| --- | --- |
| physical state | oracle toggles support of `raw`; total driver performs one block `physical[t] xor=physical[c]`; oracle projection closes the selected decoded component, total projection does not inspect physical state |
| state correction | toggle live-epoch support and propagate settled winding/full correction by mode; total projection folds the selected full `corr` into `settled_corr` |
| old syndrome | absent/pooled scratch and never a lineage key |
| new syndrome | oracle component register remains with its epoch; total mode leaves block `last_measured` pre-gate until the next observation |
| defect history | remains distinct before overflow; the selected projected epoch's unresolved history is counted/hashed and discarded |
| history correction | pooled scratch, cleared by the next feedback phase; never projected as state |
| message fields | remain distinct before overflow; projected fields are hashed/counted and discarded |
| new message fields | pooled overwrite-only synchronous scratch; no gate-time clear |
| lineage/dependency metadata | `support[t] xor=support[c]`; exact pair removal requires full equality, not label collision |
| settled/compressed information | oracle receives a closed winding; total mode receives the entire current correction quotient; certificate ring receives the loss witness |

Normal noisy rounds use the same shared-recovery oracle or total-syndrome-only mode as Section 5; the cap does not improve that heuristic. Each live epoch runs its `r` message sweeps, feedback, correction application, and RG/history cycle. The driver samples exactly one data and measurement mask per physical block; oracle and total modes route the resulting information exactly as specified in Section 5. GC first tries only the mode-specific exact terminal rule. Cleanup performs the final perfect-measurement closure and then zero-physical/measurement-noise rounds, while retaining the specified back-wall random stream. `:stop` trials report overflow without a manufactured logical result; `:project` trials report logical and certified-approximation outcomes.

An overflow with `:project` increments `lossy_merge_count` by the number of victims and `approximation_certificate_count` by one batch record only after the certificate has been durably streamed and the predicted post-hash has matched. Every victim has its own rule/version and hashes inside that record. `:stop`, no-victim, and sink-failure outcomes increment overflow but produce no manufactured logical result. There is no hidden deferred-gate queue.

### 10.4 Readout and overhead

Readout is identical to Section 5 for retained epochs plus settled frames. Before projection, the support transform gives unchanged control and target XOR control exactly under the capped epoch semantics. In oracle `:project` mode the deterministic closure forces the removed component's open boundary to zero and may choose the wrong homology. In total `:project` mode the full-current-correction fold leaves `decoded_state` unchanged at the projection instant, but subsequent evolution may be wrong because the removed state can no longer respond. Logical failure and cleanup failure remain separate; overflow and approximation are additional orthogonal outcomes.

```text
N_epoch = Theta(L^2Z) scalar/packed-word locations scanned in one dense epoch
certificate_size = O(KB + KB^2/W) machine words in the worst case
resident memory = O(K B M_epoch
                    + [K B^2/W + B L^2/W
                       + R_cert(KB + KB^2/W)] W/8
                    + M_workspace) bytes
round runtime = O(K B (r+1)L^2Z + B L^2)
one CNOT oracle = O(KB) support checks
one CNOT total = O(KB + L^2)
exact canonicalization expected = O(KB [N_epoch + B/W])
                                  with well-distributed hashes
exact canonicalization deterministic worst = O((KB)^2 [N_epoch + B/W])
certificate construction with cached state hashes = O(KB + KB^2/W)
certificate construction without cached hashes = O(KB [N_epoch + B/W])
projection batch with v<=B victims:
    victim selection = O(KB)
    oracle closures/folds = O(v[L^2 + B/W])
    total full-correction folds = O(sum_i |support_i| L^2),
                                  worst O(v B L^2)
    plus one certificate construction/stream
transaction staging runtime/peak extra:
    oracle = O(KB + B/W) operations/words
    total = O(KB + B L^2) operations and O(KB + B L^2/W) words
cleanup evolution = O(C K B (r+1)L^2Z)
cleanup terminal folds, oracle = O(K B^2/W)
cleanup terminal folds, total = O(K B^2 L^2)
GC scan/reclamation = O(KB + KB L^2Z) with eager buffer clearing,
    plus oracle O(KB^2/W) or total O(KB^2L^2) terminal folds
external certificate output = O(N_cert * (KB + KB^2/W)) words total.
```

The ordinary round bound excludes overflow work; add the displayed projection-batch, staging, and certificate costs on a certified round. The resident formula already contains a `Theta(BL^2/W)` total-mode frame term; staging can double that term transiently but does not change its asymptotic order. Cached audit hashes add no new asymptotic round cost because every dense payload is already scanned by field/history updates, but they are never trusted for exact equality. A lazy buffer pool can make reclamation `O(KB)` metadata while retaining its already counted capacity; eager zeroing costs `O(KB L^2Z)`.

Resident decoder memory is independent of `D` for fixed `K,B,L,Z,R_cert`. The streamed audit file can grow with the number `N_cert` of overflows and is classical output storage, so total decoder-plus-archive storage is not depth-independent unless certificates are aggregated. Bounded resident space is purchased by stopping or by certified approximation.

### 10.5 Required repeated-CNOT cases

| schedule | behavior |
| --- | --- |
| repeated `C1 -> C2` | With a noisy round after each gate and no drain, capacity is reached after roughly `KB-B` new epochs. CNOT-only repetitions do not consume payload capacity. |
| alternating two-block gates | Incidence factoring prevents Fibonacci copies, but one noisy post-gate native epoch per outgoing block still causes linear pressure and eventual overflow. |
| fanout | Multi-target support costs bits, not payloads. Repeated noisy fanout consumes new control-native epochs and can overflow. |
| long chain | One gate-only chain fits with `A=B`; noisy rounds between links use `O(B)` additional epochs and may exceed small `K`. |
| dense CNOTs with no idle rounds | Gate-only layer fits; noisy interleaving gives the highest overflow rate. Memory remains capped and every forced action is counted. |
| CNOTs separated by `O(L)` idle rounds | Expected low overflow below threshold as epochs drain. The measured overflow curve versus idle time is the primary test for choosing `K`. |

Overhead depends explicitly on configured `K`, `B`, and support metadata. Runtime before overflow depends on `A<=KB`; memory never depends on past `D`. Circuit treewidth is not used.

### 10.6 Performance, failure modes, implementation, and validation

Before overflow, performance matches the uncapped epoch transducer. Increasing `K` should converge monotonically in the sense that the first overflow is delayed, but logical failure rates need not be monotone after different projected histories are chosen. Target failures are expected to dominate approximation damage because propagated supports concentrate there.

Failure modes are silent overflow, `materialized_count` disagreeing with buffer ownership, a partial preflight/commit, projecting an active native when no frozen victim exists, destructive pair removal after a hash/lineage collision, mutable aliasing in the bounded pool, stale fields or syndrome registers, incorrect correction/support forwarding, a certificate that omits its declared witness, predicted post-hash mismatch, pathological homology projection, stale native references, archive I/O failure, and biased reporting. Approximation is explicit but can dominate target failures. Dense/alternating schedules can certify nearly every sample despite bounded resident memory.

Reject a chosen `K` if `overflow_count/trials` exceeds a preregistered tolerance in the intended schedule (start with `1%`), if certified trials dominate the logical estimate, or if increasing `K` changes the inferred threshold trend beyond statistical error. Always report unconditional rates, exact-no-overflow conditional rates, and projected rates separately.

**Difficulty: large.** It adds a capacity manager, exact pair canonicalization, bounded certificate ring/stream, projection routine, and reporting on top of the epoch engine. Baseline remains unchanged; this is a separate mode.

In addition to Section 11, validate:

- `length(payload_pool)==K*B` never changes, `materialized_count==count(live_slots)<=K*B`, and fixed lazy sentinels/free slots are counted separately;
- a `D=32` gate-only burst leaves `materialized_count` unchanged;
- failed/no-victim/sink-failure reservation leaves physical/event state and future outputs unchanged; exact canonicalization may change epoch identities/support-object layout but must pass the exact-representation invariance test;
- failed preflight leaves data/measurement counters unchanged, while successful commit advances each exactly once;
- exact duplicate merge versus two explicit coupled epochs;
- every lossy removal has exactly one victim entry in its batch certificate; `lossy_merge_count` advances by victim count and `approximation_certificate_count` by batch count;
- planned rule/version, victim, closure/fold hash, and predicted post-hash are verified on staged state and streamed before a no-allocation commit, then asserted again after the pointer swap;
- `:stop` and `:project` never share result files accidentally;
- certificate fields/hashes are verified online while the discarded payload still exists; exact replay is not claimed without a separately costed full archive;
- resident certificate-ring bytes remain flat in `D` and stream I/O failures are surfaced;
- curves versus `K in {1,2,4,8}` expose convergence or lack of it.

### 10.7 Cross-strategy failure-mode audit

This matrix makes explicit which named hazards apply; it supplements, rather than replaces, each rejection rule.

| strategy | destructive merge / lineage collision | stale fields / syndrome inconsistency | correction forwarding / aliasing | approximation / overflow | alternating and dense circuits |
| --- | --- | --- | --- | --- | --- |
| incidence epochs | distinct source fields are not XOR-merged, but destinations of one source are intentionally coupled; support-id collisions are fatal | pooled scratch, mismatch counters, and component/total syndrome routing are high risk | support-bit forwarding is the core heuristic; dense payloads must never alias | no configured approximation; `A(t)` may grow without bound | no Fibonacci paths, but `A=A(0)+O(D)` and shared correction bias can accumulate |
| sparse replay | no lossy merge when exact; packed-key collisions are fatal | missing sparse zeros/wavefronts or replay inputs creates stale state | immutable log aliasing is safe only with immutable nodes; support errors remain | no cap; dense fallback is exact, not overflow | `A,F,J_K` can grow and force dense storage |
| factor graph | no sheet merge; a wrong CNOT XOR factor is the analogous destructive error | pre/post measured nodes and carried boundary can become inconsistent | committed control correction must not also remain unresolved | `w_full>K` must stop or produce a certified BP fallback | dense gates enlarge `w_c` and `w_full`; fixed window bounds depth but not solve cost |
| symbolic DD | canonical sharing is exact; lineage/hash collisions or wrong active subset are fatal | every literal scratch/syndrome root must be represented and cache-correct | immutable nodes prevent array aliasing; arena/root ownership must be correct | untruncated mode has no overflow; node explosion is explicit | independently noisy alternating gates can recover `phi^D` nodes |
| quiescent barriers | no lineage merge; terminal field reset is an intentional semantic change | gate forbidden unless history empty and decoded syndrome zero; reset must clear fields | realized frame updates once; audit matrix is never reapplied; blocks cannot alias | failed barrier records overflow and refuses gate, with no lossy fallback | only noiseless gate layers are supported; dense noisy circuits violate the contract |
| capped epochs | exact until projection; hash pair cancellation requires byte equality | projection deliberately discards history/fields and can corrupt syndrome inference | inherits epoch forwarding and pool-aliasing risks | explicit lossy/overflow certificate is mandatory; stream storage separately | resident memory stays bounded while certificate/error rates may approach one per gate |

## 11. Validation and measurement program

Every candidate must implement this common program in addition to its strategy-specific tests. A candidate is not ready for threshold scans until all applicable deterministic tests pass.

### 11.1 Deterministic reference harness

Refactor the simulation harness so a trial accepts explicit arrays/streams for:

```julia
data_mask[round, block, i, j, orientation]
measurement_mask[round, block, i, j]
backwall_uniform[round, decoder_object, i, j, k]
```

The harness counts consumers of every mask. For block-noise candidates, each `(round,block)` data and measurement mask must have exactly one physical consumer. For literal symbolic sheet-copy, a separate explicitly indexed per-lineage table is intentional and labeled accordingly.

The reference output after every subphase should include raw/correction syndrome and winding, current measured syndrome, `hist`, `fields` or the representation's exact equivalent, live dependency metadata, and RNG counters. This makes a mismatch local rather than terminal-only.

For any mode in which every attempted trial returns a logical result, retain the implementation convention

```text
CNOT_Ft = 1 - logical_failures / trials.
```

Cleanup failures are still a separate count and are not silently added to `logical_failures`. For `:stop`/width-stopped modes, no logical result is manufactured: report attempted trials, completion rate, and conditional `CNOT_Ft_exact` or `CNOT_Ft_projected` with its denominator explicitly, while unconditional `CNOT_Ft` is `N/A`.

### 11.2 Required deterministic tests

Run at `L=3` and `L=5` with small fixed `Z`, fixed masks, and fixed back-wall choices:

1. **Zero-noise CNOT succeeds.** Begin with all-zero state, run pre/post rounds and cleanup, and require zero logical and cleanup failure.
2. **Control output is unchanged.** Seed a set of contractible and noncontractible control chains; a CNOT must not change the control decoded quotient.
3. **Target output is target XOR control.** Test all four pairs of two winding bits and several nontrivial syndrome boundaries.
4. **Readout is invariant under exact representation changes.** Dense versus quotient, dense versus sparse, explicit versus symbolic, or before versus exact GC must have equal syndrome and winding; require bitwise equality only where promised.
5. **No unintended mutable aliasing.** Mutate every array/entry in a child or pooled workspace and prove unrelated payloads/parents do not change.
6. **Garbage collection does not alter decoded state.** Apply later noisy native rounds and later CNOTs before/after collection; use CNOT-only continuation only where the semantics forbids future noise on the collected object.
7. **Quiescent compressed objects equal explicit objects.** Terminal winding/boundary folding must match a continued explicit ideal decode.
8. **Noise is sampled once per physical block.** Count samples and consumers. Literal per-sheet symbolic mode is the only named exception.
9. **Cleanup failure is recorded separately.** Force a short cleanup cap and verify that winding failure, cleanup failure, overflow, and approximation are distinct fields.
10. **Old-syndrome scratch elimination.** Removing `old_synds` at an inter-round gate must reproduce the explicit register path.
11. **Pooled scratch invariance.** Fill `new_fields` and `hist_correction` with adversarial stale values before every use and obtain the same result after the required clear/overwrite.
12. **Message-field necessity witness.** Construct equal-`hist`, unequal-`fields` states whose next feedback differs; any exact canonicalization must keep them distinct.
13. **Alternating-gate algebra.** Compare support/dependency transformations to direct multiplication by the exact GF(2) CNOT matrices for every prefix through `D=32`.
14. **Failure-function polarity.** Verify that `detect_logical_error=true` is success, not failure.
15. **Shared versus independent descendant witness.** Feed two explicit descendants different back-wall coins and show the epoch mode's coupled result is intentionally different, not mislabeled exact sheet behavior.
16. **Total-syndrome black box.** In non-oracle modes, enforce an API boundary that makes hidden `eta_p`, `eta_q`, component raw boundaries, and fault labels inaccessible to decoder code.
17. **Factor-width separation.** Report `w_c` and `w_full` on the same graph and verify that resource formulas use only `w_full` in exact table exponents.
18. **Online certificate verification.** Validate certificate hashes/counts while the discarded payload exists; do not claim later replay from a summary-only record.
19. **Bounded instrumentation.** Metrics, traces, and certificate summaries stream or use bounded online aggregates so measurement code itself does not accumulate `O(D)` resident memory.

### 11.3 Small noisy tests

For every candidate and for baseline, primitive, and literal sheet-copy references, run matched fixed-sample tests:

```text
L in {5, 7}
p in {0.005, 0.010, 0.015}
qrat = 1
T = L
cleanup = 2T
fixed_samps = 200 to 1000
r = 3
synch = true
```

For baseline-like modes use the implementation default `Z=ceil(log_{1.5}L)` unless `Z` is an explicit scan variable. The factor-graph family instead reports half-window `Z_FG`; test `Z_FG in {L,2L,4L}`. These finite choices probe convergence but do not satisfy `Z_FG>>L` by declaration alone.

For exact factor-graph elimination, record `r = N/A`; `r=3` applies only to its explicitly approximate BP mode. Do not imply that exact results used three local sweeps.

For the one-CNOT small and threshold scans, place the gate at

```text
T_PRE  = floor(T/2)
T_POST = T - T_PRE
```

so the extra noisy round for odd `T` is after the CNOT. Run exactly `T_PRE` noisy rounds, the gate, `T_POST` noisy rounds, then at most `2T` ideal cleanup rounds.

Use the same pre-sampled block masks across block-noise candidates. Literal sheet-copy and its symbolic exact representation use their separate per-sheet input table and are reported as a different stochastic model. Report Wilson or exact binomial intervals for logical rates and bootstrap intervals for peak memory/runtime summaries.

The first questions are not threshold claims:

- Does the target/control failure imbalance improve over primitive?
- Does source-history separation with shared descendant recovery improve target performance over destructive primitive merging?
- Does sparse execution exactly match dense epoch execution?
- Does factor-graph exact mode agree with brute force on embedded small cases?
- How often do width/cap certificates occur?

### 11.4 Threshold-trend tests

Only candidates that pass deterministic and small-noisy gates proceed to:

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

Use identical timing splits and cleanup definitions across comparable modes. Plot total, control, target, and both-block logical failure rates. Do not combine cleanup or approximation failures with logical failure; show them as separate panels/columns. A crossing trend is evidence for prioritization, not proof of a true threshold.

### 11.5 Repeated-CNOT stress tests

Run

```text
D in {1, 2, 4, 8, 16, 32}

patterns:
    C1 -> C2 repeated
    C1 -> C2, C2 -> C1 alternating
    C1 -> C2, C1 -> C3 fanout
    C1 -> C2 -> C3 -> C4 chain

idle rounds:
    0
    floor(L/4)
    floor(L/2)
    L
```

For the primary memory-versus-depth comparison, `D` counts **individual CNOT events**, not motif repetitions:

```text
repeated:   event k is C1 -> C2
alternating:event k is C1 -> C2 for odd k, C2 -> C1 for even k
fanout:     event k is C1 -> C2 for odd k, C1 -> C3 for even k
chain:      cycle events C1 -> C2, C2 -> C3, C3 -> C4 until D events
```

Record motif/layer depth as an additional column when gates are declared parallel; do not relabel it as the primary `D`. For `idle>0`, place exactly that many noisy decoder rounds immediately **after each individual event** and before the next event. For `idle=0`, use the two explicitly separated subcases below. This convention makes all four patterns comparable at equal gate count.

Define precisely whether `idle=0` includes a noisy decoder round between gates. Run two subcases:

- **gate-only burst:** no noise or decoder transition between adjacent gates;
- **dense noisy:** one noisy decoder round after each gate.

This distinction is essential: support/DAG methods compress the former exactly, while the latter creates genuinely new live epochs.

For each `(strategy,L,p,D,pattern,idle)` record at least:

```text
CNOT_Ft
logical_failures
control_logical_failures
target_logical_failures
both_logical_failures
cleanup_failures
peak_active_histories
peak_live_dependencies
peak_total_metadata_objects
peak_dense_field_buffers
peak_sparse_entries
estimated_peak_bytes
runtime_seconds
garbage-collected objects
lossy_merge_count
overflow_count
approximation-certificate count
```

Also record strategy-specific state:

```text
epoch support density
H(t), F(t), J_K
factor-graph N(t), w_c(t), w_full(t), separator bytes
decision-diagram chi_state(t), P(t), R_union(t), chi_alloc_total(t), root count, Apply-cache hit rate
barrier cleanup rounds and gate refusals
certificate reason/policy counts
```

Compute `estimated_peak_bytes` from actual Julia object sizes, including array headers, sparse container capacity, arenas, cached tables, and pooled workspaces. Do not estimate it only by sheet count.

For every repeated-CNOT pattern, tabulate and plot peak bytes against `D` on linear and log-y axes, with separate curves for each idle interval. Add `peak/D`, `peak/phi^D`, and `peak/(B*M_block)` diagnostic ratios where meaningful. Plot runtime against both `D` and the strategy's live-width variable (`A`, `F`, `w_full`, or `chi_peak`) to test the derived scaling rather than only an empirical depth curve.

### 11.6 Predeclared rejection and success criteria

All exact representations have zero tolerance for deterministic mismatch. For statistical performance, predeclare margins before large scans. A reasonable initial gate is:

- target logical failure at `L=7,p=0.010` must be materially below primitive and statistically compatible with, or plausibly trending toward, the appropriate independent-history reference;
- block-noise candidates must never consume more than one physical mask per block-round;
- after `L` ideal idle rounds at subthreshold `p`, median live extra state should decrease rather than remain monotone in completed `D`;
- claimed bounded modes must demonstrate a flat peak-memory curve in `D`, with every overflow/approximation shown;
- a method whose runtime scaling contradicts its stated live-width bound is rejected even if `D<=8` looks favorable.

## 12. Comparative analysis

### 12.1 Strategy table

| strategy | core representation | exact or approximate | logical-performance expectation | memory scaling | runtime scaling | worst-case CNOT-depth behavior | implementation difficulty | main research risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incidence-vector epoch transducer | one source-epoch Markov payload plus a `B`-bit support, shared correction/RNG across destinations | heuristic; representation-exact for shared-recovery oracle semantics | unknown between primitive and independent sheets; target may suffer coupled descendant corrections | oracle bytes `O(A M_epoch+A s_B W/8+workspace)`; total adds `O(BL^2/8)`; near `O(BM_block)` after drain | round `Theta(A(r+1)L^2Z+BL^2)`; CNOT oracle `Theta(A)`, total `Theta(A+L^2)` | `A=A(0)+O(D)` possible with noisy no-drain gates; no path/Fibonacci enumeration | large | virtual oracle registers and shared descendant feedback may be invalid physically |
| sparse causal replay engine | adaptive sparse/dense live cache, distinct write-protected checkpoint, reusable generational log arena, oracle and total epoch types | heuristic overall; exact representation of shared-recovery epoch semantics | identical to dense epoch engine, including its semantic risks | concrete dense worst `Theta(A M_epoch + [A s_B+BK L^2+c_log AK]W/8)` bytes plus headers; packed sparse bound in Sec. 6.4 | round sparse work plus `O(A+[AL^2+S_payload+Q+J_K]/K)`; total CNOT adds `Theta(L^2)` | no path growth, but `A,F,J_K` can grow with unresolved noisy depth | research-scale | fields/checkpoints may be dense; replay may cost more than it saves |
| rolling CNOT factor graph | `2Z_FG`-round spacetime parity graph, changed-row gate layers, explicit homology variables and time-tagged recovery transport | exact for a rational-weight finite-window min-cost chain when `w_full<=K`; BP fallback certified approximate | joint inference may reduce target failures, but differs from Lake/sheet-copy | `O(N+I_gate+B^2Z_FG/W+N_bag 2^w_full ceil(b_cost/W))` words; accuracy may need `Z_FG>>L` | solve `O(N+I_gate+N_bag 2^w_full C_arith(b_cost)+T_tr)`; one CNOT `O(L^2+B/W)` immediate | fixed-window peak independent of completed `D`; incidence/full width may scale badly in `L,Z_FG,B` | research-scale | dense gate incidence, transport, `w_full`, and `Z_FG>>L` can erase the advantage |
| symbolic min-plus decision diagrams | DD/ADD functions from lineage key to every literal sheet entry | exact relative to literal sheet-copy when untruncated | exactly literal sheet-copy, including per-lineage noise | peak `Theta(L^2Z+chi_alloc_total_peak+C_apply_peak+C_rng_table_peak)`; after rebuild `chi_alloc_total=Theta(R_union)` | round/CNOT worst `O((r+1)L^2Z chi_peak^2)`; readout `O(BL^2chi_peak^2)`; rebuild scans roots/arena/caches | even gate-only metadata is generally `Omega(D_live)`; noisy alternating can recover explicit-sheet exponential scale | research-scale | random entropy, literal lineage metadata, and intermediates destroy compression |
| quiescent circuit barriers | `B` baseline decoders, terminal-reset winding frames, audit-only `B x B` map | exact for the restricted terminal-reset barrier decoder | no unresolved merge; reset may change later performance | `B M_block + O(B^2)` | round `Theta(BU_block)`; barrier `Theta(CBU_block)`; gate `O(1)` or `O(B/W)` | resident memory independent of `D`; runtime can be `Theta(DCBU_block)` | medium | noiseless barriers and canonical field reset may be unusable |
| certified capped epoch ensemble | at most `KB` materialized shared-recovery epochs, fixed lazy natives, and a bounded certificate ring/stream | approximate with explicit mode-specific projection; stop is exact but incomplete | epoch behavior until overflow; target likely bears projection errors | resident bytes `O(KB M_epoch + [KB^2/W+BL^2/W+R_cert*cert_size]W/8)`; archive `Theta(N_cert*cert_size)` words | round `O(KB U_block+BL^2)` plus certified overflow work; CNOT oracle `O(KB)`, total `O(KB+L^2)` | resident bounded; archive and approximation count can grow with `D` | large | a flat resident curve may hide an unacceptable certificate/archive rate |

Here `U_block=Theta((r+1)L^2Z)`, `M_epoch` is the one-field persistent payload, `C_F` is sparse candidate-field count, `w_full` is complete spacetime induced width, `chi_peak` is peak intermediate DD size, and `S` is explicit literal sheet count.

### 12.2 Independence from the excluded mechanism list

Three of the six families are unambiguously outside all five excluded mechanism classes, satisfying the “at least half” requirement:

- factor-graph elimination replaces local sheets with a global finite-window chain-inference problem;
- decision diagrams replace per-sheet arrays and per-sheet update calls by canonical Boolean/integer functions and symbolic `Apply`; although their untruncated semantics deliberately match the literal reference, this is a whole-program symbolic execution method, not copied or compacted sheets;
- quiescent barriers obtain boundedness by forbidding any unresolved history from crossing a gate and physically applying/canonicalizing recovery at a scheduled terminal interface; they do not retain then compact settled lineages.

Incidence-vector epochs are deliberately classified as adjacent to shared-noise/correction-forwarding mechanisms because one multi-destination payload shares a correction stream. Sparse replay is a new exact storage/execution technique but inherits that semantic core. The capped ensemble is intentionally the bounded-labeled-state approximation baseline. No novelty claim relies on renaming those three adjacent mechanisms.

### 12.3 Rankings

Rank 1 is strongest. Ties indicate genuinely different risk profiles.

#### Logical fidelity to the desired one-noise-per-block CNOT decoder

1. Rolling CNOT factor graph in exact bounded-width mode.
2. Quiescent circuit barriers under their terminal-reset schedule assumptions.
3. Incidence-vector epoch transducer in total-syndrome-only mode.
4. Sparse causal replay engine (same decoder semantics as the epoch engine, with more implementation risk).
5. Certified capped epoch ensemble.
6. Symbolic decision diagrams in literal mode, because exact reproduction includes the undesirable per-lineage noise model.

#### Space efficiency in the expected subthreshold sparse regime

1. Sparse causal replay engine.
2. Quiescent circuit barriers.
3. Incidence-vector epoch transducer.
4. Certified capped ensemble for moderate `K`.
5. Symbolic decision diagrams, whose compression is least predictable.
6. Exact rolling factor graph, because `Z_FG>>L` and `2^w_full` dominate despite good depth behavior.

#### Worst-case boundedness in total CNOT depth

1. Quiescent circuit barriers (exact but restricted).
2. Certified capped ensemble (unconditional space cap but approximate/incomplete on overflow).
3. Rolling factor graph with a hard width cap and explicit stop/fallback.
4. Incidence-vector epoch transducer.
5. Sparse causal replay engine.
6. Symbolic decision diagrams under literal independent sheet noise.

#### Implementation feasibility

1. Quiescent circuit barriers.
2. Incidence-vector epoch transducer.
3. Certified capped ensemble after the epoch core exists.
4. Sparse causal replay engine.
5. Rolling CNOT factor graph.
6. Symbolic min-plus decision diagrams.

#### Novelty

1. Symbolic min-plus decision-diagram ensemble.
2. Rolling CNOT factor graph with causal-treewidth accounting.
3. Sparse causal replay engine.
4. Incidence-vector epoch transducer.
5. Quiescent circuit barriers.
6. Certified capped epoch ensemble.

#### Value as a publishable research direction

1. Rolling CNOT factor graph: it supplies a falsifiable `w_c` versus `w_full` hypothesis and connects CNOT decoding to exact/approximate graphical inference.
2. Incidence-vector epoch transducer plus total-syndrome comparison: it directly tests whether path enumeration can be removed without unacceptable shared-recovery bias.
3. Sparse causal replay engine: publishable if measured field sparsity and exact replay yield a real memory advantage.
4. Symbolic decision diagrams: high novelty and a useful negative result if compression fails for information-theoretic reasons.
5. Certified capped ensemble: valuable if certificates predict error and allow controlled resource-performance curves.
6. Quiescent barriers: useful engineering control, but the restrictive resource assumption limits novelty.

## 13. Final recommendation

### Most promising low-risk implementation

The **quiescent circuit barrier** is the lowest-risk complete implementation. Its contract is narrow but testable: a gate either sees a terminal-reset state or is refused, its resident-memory proof is `B*M_block+O(B^2)`, and it never merges unresolved fields. It is valuable because it establishes a bounded-space, target-performance control and measures the actual latency price of enforcing zero live causal width—not merely because its wrapper is easier to code.

For the unrestricted-schedule research question, the lowest-risk **informative** experiment remains the paired oracle/total-syndrome incidence-vector epoch transducer. Its worst-case statement is honest—`O(A(t)M_epoch + A(t)s_B W/8 + BL^2/8 + M_workspace)` bytes, with the `BL^2` term needed in total mode—and a negative result on target performance should stop the sparse/capped descendants before further investment.

### Most promising strong-overhead direction

The only immediately defensible strong resident-memory implementation is the **quiescent circuit barrier**, with `B*M_block+O(B^2)` memory independent of `D`, but only under its noiseless terminal-reset schedule. For arbitrary gate timing, the most promising strong-**depth** research track is the rolling factor graph: a fixed `2Z_FG` window removes completed depth, while `w_full`, `2^w_full`, and `Z_FG>>L` expose why it may fail in system size. Its first prototype must remain at very small `L` and one/two interfaces.

### Most intellectually novel proposal

The **symbolic min-plus decision-diagram ensemble** is the most novel. It asks whether the nonlinear decoder ensemble itself has a compact canonical functional representation. It is also a clean way to demonstrate when independent random entropy makes exact sheet semantics incompressible.

### Most likely to fail

The decision-diagram ensemble is most likely to fail as a practical decoder. Independent per-lineage noise and min/argmin feedback tend to separate previously shared branches, and exact `Apply` operations can turn modest diagram width into quadratic work. Its failure would still be informative, but it should not absorb production implementation effort.

### Execution order

1. Refactor noise and scratch phases behind a deterministic harness without changing the baseline reference.
2. Implement the quiescent-barrier wrapper first among candidates and use it to establish the terminal-reset performance/latency control.
3. Implement chain quotients and the paired oracle/total-syndrome incidence-vector epoch mode; this is the first unrestricted-schedule candidate.
4. Add exact terminal GC and all repeated-CNOT metrics.
5. Measure field sparsity before deciding whether to build the sparse replay engine.
6. Prototype the factor graph on exhaustive small cases in parallel with epoch scans.
7. Build only a small decision-diagram prototype (`L=3`, then `L=5`, `D<=8`). Do not integrate it into the production simulation unless node-count scaling is decisively favorable.
8. Add the capped/certificate mode only after uncapped `A(t)` distributions are measured; choose `K` from data rather than convenience.

The barrier is a control, not a substitute for the unresolved-history investigation; the epoch experiment must follow even if the restricted barrier performs well.

## 14. Future work kept outside the immediate scope

Only after one X-sector strategy passes the validation program should the project consider the reverse-direction Z-sector propagation, simultaneous sectors, circuit-level CNOT faults, planar boundaries, lattice-surgery alternatives, or a full computation scheduler. None of the scaling or performance claims here should be extended to those problems without new derivations.

## References used

- `agent.md`.
- `implementation.md`.
- `2d_windowed_simulation.jl`.
- `2d_windowed_simulation_thread.jl`.
- `2d_windowed_cnot_primitive.jl`.
- `2d_windowed_cnot_sheetcopy.jl`.
- `docs/qec_paper_index.md`.
- `docs/lake_2025_simulated_confinement.md` and `2510.08056v3.pdf`, Ethan Lake, *Local active error correction from simulated confinement*.
- `docs/dklp_2001_topological_quantum_memory.md` and `0110143v1.pdf`, Dennis, Kitaev, Landahl, and Preskill, *Topological quantum memory*.
