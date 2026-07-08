# Reference Notes: Local Active Error Correction from Simulated Confinement

## Metadata

**Title:** Local active error correction from simulated confinement

**Authors:** Ethan Lake

**arXiv identifier:** arXiv:2510.08056v3

**Why this paper matters for this project:** This is the main reference for the local/windowed message-passing decoder implemented in this repository. It explains why the decoder stores syndrome-change events rather than only current syndromes, why an extra classical buffer dimension is used, why copying full syndrome histories creates large overhead, and why a local real-time decoder needs a spacetime view of defects.

**Main objects:** Code: topological stabilizer codes with Abelian anyons, especially the two-dimensional toric/surface code and the one-dimensional repetition code. Decoder: local cellular-automaton message passing with per-stabilizer processors, integer messages, local feedback, buffer depth $Z$, and a back wall. Syndrome/defect model: a defect is a spacetime event where a measured syndrome bit changes between rounds. Noise model: analytic results use $p$-bounded stochastic noise; numerics use independent Pauli/depolarizing data noise and measurement-bit noise, often with equal strength.

## Executive Summary

Lake's paper builds a fully local active decoder by simulating a confining interaction between spacetime defects. In the surface-code setting, a stabilizer measurement produces a syndrome bit at each stabilizer location and each round. With faulty repeated measurements, the object to correct is not just the current anyon configuration, but the set of syndrome-change events through time. These events are the defects in the paper, following the spacetime picture of Dennis, Kitaev, Landahl, and Preskill.

The decoder places a small classical processor $\mathsf{P}_{\mathbf r}$ at each stabilizer location $\mathbf r$. Each processor stores local syndrome information and message variables. A message is an integer-valued signal emitted by a defect and propagated locally; defects move in the direction of the strongest or nearest received signal, which approximates attraction toward nearby defects. Figure 1 shows this basic message-passing architecture.

The central problem is that a message-passing decoder with no buffer, $Z=0$, only simulates confinement in space. Under active noise, spatial confinement alone is not enough. Noise creates many small anyon pairs whose messages screen the interaction between well-separated anyons. The result is a pseudothreshold rather than a true threshold: the memory time grows rapidly at small systems and small $p$, but it saturates as $L\to\infty$ for fixed $p>0$. This is summarized in Section III, Figure 3, Equation (17), and Theorem 2; the bounded-memory scaling is stated in Equation (23).

Lake's fix is to store a dynamically updated buffer of syndrome-change history in a small extra classical dimension of depth $Z$. New defects enter at $z=0$, drift toward larger $z$, run $(d+1)$-dimensional message passing while in the buffer, and accumulate on the back wall at $z=Z$. Small spacetime clusters should annihilate before reaching the back wall. Only large clusters survive to the back wall, where a $d$-dimensional message-passing decoder handles the renormalized residual noise. The architecture is shown in Figure 2, with the key update rules in Section II.1, Equations (6)--(8).

The main theorem is a threshold theorem for modified message-passing dynamics. If $v>2$ and the buffer depth satisfies $Z \ge a(\log L/\log(p_c/p))^{1/\zeta}$, then for $p<p_c$ the memory lifetime obeys $\tau_{\rm mem}=(p_c/p)^{O(L^\beta)}$; see Section IV.1, Theorem 3, and Equation (44). The proof combines cluster erosion before the back wall (Lemma 1), a back-wall clump-size argument (Lemma 2), and rare-cluster bounds for $p$-bounded noise.

For this project, the implementation lesson is direct: the array called `hist` should be interpreted as a rolling buffer of defect events, not as the current syndrome. The extra dimension $k=1,\dots,Z$ is a classical RG/buffer coordinate, not physical time itself. CNOT decoding schemes must preserve enough event history, source-block lineage, and correction provenance to distinguish defects created before, during, and after a gate. Primitive merging can destroy this information; sheet-copying preserves it but can create large classical overhead.

## Notation

| Symbol | Meaning |
| --- | --- |
| $L$ | Linear size of the code block. |
| $d$ | Spatial dimension of the code lattice; the main toric-code case has $d=2$. |
| $\mathbf r$ | Physical stabilizer location in the $d$-dimensional code lattice. |
| $z$ or $x^{d+1}$ | Extra classical buffer coordinate, $1\le z\le Z$. |
| $\mathbf x=(\mathbf r,z)$ | Site of the classical control lattice. |
| $Z$ | Buffer depth in the extra classical dimension. Not the Pauli $Z$. |
| $\mathsf P_{\mathbf r}$ | Local classical processor associated with stabilizer location $\mathbf r$. |
| $\sigma_{\mathbf r}(t)$ | Measured syndrome bit at physical stabilizer location $\mathbf r$ and round $t$. |
| $s_{\mathbf x}$ | Internal defect variable in the buffer; $s_{\mathbf x}=-1$ denotes a defect at $\mathbf x$. |
| $m_{\mathbf x}^{\pm a}$ | Integer message at $\mathbf x$ propagating along $\pm\hat{\mathbf a}$. |
| $m_{\rm max}$ | Maximum message value; the paper sets $m_{\rm max}=L$ for analysis. |
| $v$ | Message velocity, i.e. number or range of message-propagation steps per decoder round. |
| $p$ | Noise strength in a $p$-bounded noise model, or the common noise rate in many numerics. |
| $q$ | Measurement-error probability when separated from data-error probability. |
| $p_c$ | Threshold noise strength. |
| $\tau_{\rm mem}$ | Memory time: expected runtime before logical failure, as defined in Definition 3 and Equation (14). |
| $p_{\rm log}$ | Logical error rate after running for time $L$, defined in Equation (16). |
| $\mathsf N_k$ | Level-$k$ residual noise set after clustered points at lower scales have been removed. |
| $p_k$ | Level-$k$ error rate; rare-cluster decay is bounded in Equation (13). |
| $k_Z$ | Largest cluster level expected to erode before reaching the back wall. |
| $\mathsf{bw}(\mathcal C_\alpha,t)$ | Back-wall defects at time $t$ descended from noise cluster $\mathcal C_\alpha$. |

## Conceptual Flow by Section

### Section I: Introduction and Summary

The paper sets three design criteria for a self-organized real-time decoder: locality, asynchronicity, and homogeneity. Locality means quantum operations, classical processing, and noise all occur on a common timescale. Asynchronicity rules out relying on a global clock. Homogeneity rules out hard-coded spatial or temporal schedules.

The key idea is simulated confinement. In code-capacity or offline decoding, moving anyons toward nearest neighbors can approximate a minimum-weight matching. For active fault-tolerant decoding, the relevant objects are spacetime defects, so the decoder must simulate confinement in spacetime, not only in space. Figure 1 gives the message-passing picture. Figure 2 introduces the buffer of depth $Z$ and the back wall.

Important references: Section I.1.1 explains the dynamic buffer as an alternative to storing full syndrome histories. Section I.1.2 explains pseudothreshold behavior. Section I.1.3 states the informal threshold result as Theorem 1, with the lifetime scaling in Equation (4). Table I compares windowed MWPM, Harrington-style local decoders, field-based decoders, and Lake's construction.

### Section II: Preliminaries

Section II.1 defines the architecture. The control lattice has shape $L^d\times Z$. Defects are inserted when the current measured syndrome differs from the previous measured syndrome. Equation (6) shifts defects and messages toward the back wall. Equation (7) defines the local cone $\mathsf C_{\mathbf x}^{\pm a}$ used for message propagation. Equation (8) updates messages by a local minimum over neighboring candidate messages plus distance.

Section II.2 defines $p$-bounded noise and the clustering hierarchy. The analytic proof does not require independent identically distributed noise. Equation (13) bounds level-$k$ error rates as $(p/p_c)^{2^k}$ once $p$ is small enough. This is the source of the renormalized-noise argument.

Section II.3 defines memory time and logical error rate. Equation (14) defines $\tau_{\rm mem}$ through failure of a noiseless offline decoder applied after noisy online dynamics. Equation (16) defines $p_{\rm log}$ after time $L$.

### Section III: Message Passing at $Z=0$ and Field-Based Decoders

This section is the warning for this project. With $Z=0$, message passing only uses the current anyon pattern and does not buffer spacetime defect history. The paper argues and numerically supports the pseudothreshold scaling

$$
\tau_{\rm mem}\sim \exp(b\min(L,c/p^{1/d}))
$$

in Equation (17). Figure 3 illustrates screening: small noise-induced pairs sit between a large pair and screen the confining signal. Figure 4 gives numerical pseudothreshold evidence for the $1d$ repetition code and $2d$ toric code.

Theorem 2 proves a no-threshold result for $Z=0$ message passing under a particular $p$-bounded error model; Equation (23) bounds $\lim_{L\to\infty}\tau_{\rm mem}$. Appendix A strengthens this as Theorem 5. Section III.2 and Theorem 6 make a similar no-threshold point for power-law or field-based decoders.

### Section IV: Decoding Performance

Section IV.1 proves the threshold theorem. The formal theorem is Theorem 3. Its assumptions include modified feedback rules, $v>2$, $p$-bounded noise with $p<p_c$, and buffer depth at least

$$
Z_{\rm max}=a\left(\frac{\log L}{\log(p_c/p)}\right)^{1/\zeta}.
$$

The theorem's lifetime conclusion is Equation (44): $\tau_{\rm mem}=(p_c/p)^{O(L^\beta)}$.

Lemma 1 says low-level spacetime clusters erode before reaching the back wall. Lemma 2 says creating a large back-wall clump requires at least one large noise cluster unless many clusters participate. Corollary 1 says the architecture extends to arbitrary Abelian anyon models when defect motion is charge-independent.

Section IV.2 discusses desynchronization. A non-Markovian asynchronous update schedule has bounded waiting times, Equation (58), and a threshold if the velocity condition is strengthened to $v>2(1+\epsilon)/(1-\epsilon)$. Equations (59)--(61) bound message propagation and defect motion. Poissonian desynchronization is used in numerics but is not covered by the same clean proof.

Section IV.3 introduces gerrymandering. Theorem 4 says local message-passing decoders have suboptimal asymptotic suppression under independent noise: local decoding can fail on sparse Cantor-like error patterns. This should temper comparisons with global matching.

Section IV.4 reports numerics. Figure 7 shows logical-error scaling for $Z=\lceil\log_{3/2}L\rceil$. For equal data and measurement noise, the observed threshold is about $7.5\%$ for the $1d$ repetition code and about $1.5\%$ for the $2d$ toric code. Table II reports approximate thresholds with perfect/noisy measurements and synchronous/Poissonian asynchronous updates.

Sections IV.5 and IV.6 discuss phase transitions and initialization. Figure 10 suggests initialization time scaling roughly as $O(L)$ for the tested subthreshold cases.

### Section V: Discussion and Outlook

The discussion emphasizes speed and local implementation. It also gives useful overhead estimates: with messages propagating up to half the system size, memory per stabilizer is roughly $\lceil 6Z\log_2(L/2)\rceil$ bits for messages, plus syndrome, feedback, and comparison bits. For $L=21$ and $Z=5$, the paper estimates about $10^5$ total classical bits. The paper explicitly notes that the construction still relies on reliable noiseless classical bits and does not solve the problem of a fully self-correcting decoder with noisy classical memory.

## Major Results

### Result 1: Message Passing Simulates Local Confinement

**Plain-language statement.** Defects emit local messages. Defects move toward smaller-valued incoming messages, which approximates moving toward nearby defects. This gives a local proxy for a minimum-weight pairing without global matching.

**Formal statement.** The architecture is defined in Section II.1. Defect/message drift is Equation (6), the message update cone is Equation (7), and the message update rule is Equation (8). Figure 1 shows the spatial message-passing mechanism; Figure 2 shows the buffered spacetime version.

**Assumptions.** Processors have reliable local memory. The main exposition uses synchronized discrete updates; Section IV.2 discusses asynchronous replacements. Messages are integer-valued and stored up to a cutoff, taken as $m_{\rm max}=L$ for analysis.

**Why it matters for decoding.** The rule is a local approximation to matching. It tries to pair defects by local causal propagation rather than by global graph construction.

**Implementation relevance.** Store per buffer site an active defect bit and six message fields in the $2d$ toric-code case: $\pm x$, $\pm y$, and $\pm z$. In this repository, this corresponds to `hist[i,j,k]` for active defect events and `fields[i,j,k,a,s]` for distance messages. A zero or sentinel value for "no message" must be handled consistently; the current code uses zero and therefore needs `nonzeromin` logic when merging fields.

### Result 2: $Z=0$ Active Message Passing Has Only a Pseudothreshold

**Plain-language statement.** A local decoder that only sees the current spatial syndrome can appear good at small sizes, but active noise eventually screens the attractive interaction. The memory time stops improving with $L$ at fixed $p$.

**Formal statement.** Section III gives the scaling in Equation (17). Theorem 2 states that for $Z=0$ there exists a $p$-bounded error model with bounded $\lim_{L\to\infty}\tau_{\rm mem}$; the explicit upper bound is Equation (23). Appendix A states the stronger no-threshold theorem as Theorem 5.

**Assumptions.** The rigorous no-go is for a chosen adversarial $p$-bounded noise model and even holds without measurement errors. The numerical pseudothreshold evidence uses independent noise.

**Why it matters for decoding.** It explains why retaining spacetime defect history is not optional for active fault tolerance. Measurement errors are not the only problem; transient data noise alone can screen a purely spatial interaction.

**Implementation relevance.** A decoder state that only contains current stabilizer values, or a CNOT merge that destroys history provenance, risks recreating the $Z=0$ failure mode. Do not evaluate a CNOT plan only on small $L$ crossings; pseudothreshold behavior can mimic a true threshold.

### Result 3: A Polylogarithmic Buffer Restores a True Threshold

**Plain-language statement.** A rolling buffer of syndrome-change events lets small spacetime clusters correct themselves before reaching the back wall. The back wall only sees a renormalized, much sparser defect process.

**Formal statement.** Theorem 3 in Section IV.1 proves a threshold for modified message-passing dynamics. If $v>2$, $p<p_c$, and $Z\ge a(\log L/\log(p_c/p))^{1/\zeta}$, then $\tau_{\rm mem}=(p_c/p)^{O(L^\beta)}$ as in Equation (44).

**Assumptions.** The theorem uses $p$-bounded stochastic noise, noiseless classical bits, modified feedback rules, and an offline cleanup decoder. It proves the theorem for the analytically convenient variant, not every engineering variant.

**Why it matters for decoding.** This is the paper's main reason to store a buffer of defect history rather than a global history. It gives a local, homogeneous alternative to windowed global matching.

**Implementation relevance.** The code's RG cycle should shift existing history toward larger $k$, insert new events at $k=1$, and preserve/splice back-wall spatial fields carefully. The back wall is not a garbage collector; it is a residual decoder for events that survive the buffer.

### Result 4: Locality, Homogeneity, and Asynchronicity Are Distinct Constraints

**Plain-language statement.** A decoder may be local in space yet still rely on a global clock or a hard-coded hierarchy. Lake's design aims to avoid those resources.

**Formal statement.** The desiderata are listed in Section I. Section IV.2 gives an asynchronous non-Markovian update scheme and a proposition showing the threshold proof survives if $v>2(1+\epsilon)/(1-\epsilon)$. Equations (58)--(61) bound update intervals, message travel time, update counts, and defect motion.

**Assumptions.** The rigorous asynchronous proof uses bounded waiting times and cooldown variables. Poissonian asynchronous updates are studied numerically but not proved by the same theorem.

**Why it matters for decoding.** A hardware-realistic local decoder should not require a global schedule. But desynchronization can lower thresholds and can change scaling.

**Implementation relevance.** The repository's synchronous and asynchronous branches are not interchangeable. If changing asynchronous update logic, track whether message updates, defect feedback, noise, syndrome refresh, and RG cycling still obey the intended causal ordering.

### Result 5: Numerical Thresholds Are Good but Not Global-Matching Optimal

**Plain-language statement.** The simple local decoder reaches a numerically observed surface-code phenomenological threshold near $1.5\%$ under equal data and measurement noise, about half the quoted global MWPM spacetime-history value.

**Formal statement.** Section IV.4, Figure 7, and Table II report the estimates. The table gives $2d$ thresholds of about $3.5\%$ with perfect measurements and about $1.5\%$ with noisy measurements for synchronous updates; Poissonian asynchronous values are lower.

**Assumptions.** These are Monte Carlo estimates for the implemented architecture, with $Z=\lceil\log_{3/2}L\rceil$ and message velocity $v=3$. They are not analytic threshold values.

**Why it matters for decoding.** It gives a realistic performance target for this repository's baseline memory decoder and a benchmark for CNOT-decoder variants.

**Implementation relevance.** Compare CNOT prototypes to the baseline at matched $L$, $p$, $q$, $Z$, runtime, and cleanup rules. A threshold close to sheet-copy but with lower lineage overhead is the current project target.

## Implementation Notes for Future Agents

### Suggested Data Structures

- Per physical edge: current data error state and accumulated physical correction.
- Per stabilizer location: previous measured syndrome, current measured syndrome, and measurement-noise sample for the current round.
- Per buffer site $(i,j,k)$: defect bit representing a syndrome-change event.
- Per buffer site and direction: integer message values for $\pm x$, $\pm y$, and $\pm z$.
- Per buffer site and direction: proposed history correction bits before applying feedback.
- Per logical block or sheet: separate state, syndrome registers, history, fields, and correction buffers.
- For CNOT variants: source block, target block, gate time, lineage or provenance id, error sector, and whether the object is a live sheet, copied sheet, merged summary, or readout-only contribution.

### Quantities to Store Per Stabilizer or Defect

- Current measured syndrome and previous measured syndrome. The defect event is their XOR or product difference.
- Buffer depth coordinate $k$ for every stored event.
- Message values and direction preferences at each event.
- Whether a spatial feedback move toggles a physical edge or a $z$-move only changes buffer history.
- Back-wall membership. Back-wall defects need spatial message passing only; they should not continue moving beyond $Z$.
- Origin metadata in CNOT experiments: original block, current block, source block, gate id, copied/merged state, and error sector.

### What Should Be Updated Each Round

1. Propagate messages locally for the configured velocity or number of field substeps.
2. Choose local feedback from the smallest valid positive message.
3. Convert spatial history corrections into physical corrections.
4. Apply history corrections in the buffer.
5. Apply physical noise and measurement noise.
6. Shift syndrome registers.
7. Run the RG/buffer cycle toward the back wall.
8. Insert new defects at the front slice from syndrome changes.
9. Preserve back-wall spatial messages according to the local rule; do not erase them accidentally.

### What Not to Conflate

- Do not conflate a current stabilizer violation with a defect event. A defect is a change between consecutive syndrome measurements.
- Do not conflate physical time with the buffer coordinate $z$. The buffer coordinate is a rolling classical RG/history coordinate.
- Do not conflate Pauli $Z$ with buffer depth $Z$.
- Do not conflate local message-passing corrections with global MWPM. Message passing is a local proxy, not an exact matching solver.
- Do not conflate defects from different CNOT causal histories. Merging field arrays can erase information about whether a defect came from the target block, copied control block, or post-gate target noise.
- Do not conflate cleanup failure with logical failure unless the experiment explicitly defines them the same way.

### Common Implementation Mistakes

- Inserting the current syndrome into `hist` instead of inserting the syndrome change.
- Clearing or overwriting back-wall fields during the RG cycle.
- Treating zero-valued fields as smaller than real messages when zero means "no message" in the code.
- Applying a physical correction for a vertical buffer move.
- Updating `new_fields` stale buffers across CNOT or merge events.
- Copying history without copying the associated fields and correction state needed to interpret it.
- Merging histories from two blocks without recording origin, which can make later defects indistinguishable.
- Treating the $Z=\lceil\log_{3/2}L\rceil$ numeric setting as if it were the exact analytic $Z_{\rm max}$ requirement.

### Theoretical Assumptions Not Yet Implemented

- The proof uses modified feedback rules; the repository implementation may use the simpler numerical rules.
- The analytic theorem assumes noiseless reliable classical bits.
- The rigorous asynchronous threshold uses bounded non-Markovian waiting times plus cooldown variables; Poissonian asynchronous code paths are numerically motivated.
- The paper analyzes memory, not a complete fault-tolerant CNOT decoder with lineage compression.
- The proof is sector-agnostic for Abelian anyons, but current CNOT prototypes are X-sector demonstrations.

## Connections to Our Project

Lake's decoder is the conceptual source of the baseline local decoder. In repository terms, `hist[:,:,1] = old_synds xor new_synds` is the DKLP/Lake defect event. The array dimension `k=1:Z` is the extra classical buffer. The field arrays are the integer messages. The back wall `k==Z` holds residual large-scale events.

The local decoder is close to Lake's architecture but not automatically covered by every theorem. The code uses finite arrays, implementation-specific tie-breaking, stochastic back-wall motion, and practical choices such as `Z=ceil(log(1.5,L))`. Treat the paper as the conceptual and asymptotic guide, then verify the exact Julia rule before changing behavior.

Copying or buffering a full syndrome history can cause large classical overhead because a full spacetime history for a distance-$L$ surface code has $O(L^3)$ syndrome bits over an $O(L)$ reliable window. Lake's dynamic buffer reduces the intended memory to roughly $O(L^2\operatorname{polylog}L)$ for memory decoding. The sheet-copy CNOT prototype preserves performance by copying full active decoder sheets, but repeated CNOTs can create many lineage sheets and defeat the overhead goal.

For CNOT decoding, the information that must not be lost is more detailed than a Boolean defect array. A future design must preserve enough information to distinguish defect origin, current block, source block, gate time, error sector, and whether a defect existed before the CNOT or was created by post-CNOT noise. Primitive CNOT merging loses much of this provenance. Sheet-copy preserves it by duplication, but the price is overhead growth.

The project target is therefore a compressed-lineage or local-summary CNOT decoder: preserve the spacetime/causal information that Lake's buffer needs, without copying every full active sheet indefinitely.

## Do Not Overclaim

- The analytic threshold is proved for a modified message-passing decoder under $p$-bounded noise, not necessarily for every line of the current Julia implementation.
- The observed $2d$ threshold near $1.5\%$ is numerical, not a theorem.
- The $Z=0$ no-go uses a constructed $p$-bounded noise model for the rigorous theorem; the independent-noise pseudothreshold evidence is numerical and heuristic.
- The paper does not provide a full CNOT or computation decoder for this repository.
- The construction still requires noiseless classical bits; it is not a no-classical-overhead or noisy-classical-memory solution.
- The asynchronous proof is not the same as the Poissonian asynchronous numerics.
- The paper argues that $O(\log L)$ may be enough numerically, but the theorem uses $O(\operatorname{polylog}L)$.
- The local decoder is not claimed to match global MWPM. Gerrymandering means its subthreshold scaling is asymptotically worse than optimal global decoding.
