# Reference Notes: Topological Quantum Memory

## Metadata

**Title:** Topological quantum memory

**Authors:** Eric Dennis, Alexei Kitaev, Andrew Landahl, John Preskill

**arXiv identifier:** arXiv:quant-ph/0110143v1

**Why this paper matters for this project:** This paper gives the surface/toric-code and spacetime syndrome-history framework that underlies the local decoder and the CNOT decoder work. It defines stabilizer defects, error chains, recovery chains, measurement-error spacetime histories, and the minimum-weight-chain viewpoint that Lake's local message passing approximates.

**Main objects:** Code: toric and planar surface codes with qubits on links and stabilizer checks on sites and plaquettes. Decoder: global classical recovery by most-probable homology class or minimal-weight chain, plus finite-time overlapping recovery. Syndrome/defect model: stabilizer outcomes identify defects; repeated faulty measurements create a spacetime syndrome chain whose boundary equals the error-chain boundary. Noise model: independent data errors with probability $p$ and measurement-bit errors with probability $q$, later related to circuit-level storage, CNOT, preparation, and measurement faults.

## Executive Summary

DKLP is the canonical source for the spacetime view of surface-code decoding. The paper starts with toric and planar surface codes: qubits live on lattice links, site checks are products of $X$ around a vertex, and plaquette checks are products of $Z$ around a face. A Pauli error chain has boundary at violated checks. The same syndrome can come from many chains, and recovery succeeds when the recovery chain and actual error chain differ by a homologically trivial cycle. This is developed in Section III and Figures 1--5, especially Figure 4.

Faulty measurement changes the decoding problem. A single measured syndrome snapshot is not reliable because measurement errors create ghost defects and can hide genuine defects; see Section III.C and Figure 7. DKLP therefore repeats syndrome measurement and represents the whole history as a spacetime lattice. Data errors are horizontal links. Measurement errors are vertical links. The measured syndrome history is a chain $S$, the actual errors form a chain $E$, and they have the same boundary, $\partial S=\partial E$. This is the key implementation idea for this project: a defect is a syndrome-change event in spacetime, not just a current nontrivial stabilizer value. Figures 8 and 9 are the important visual references.

The optimal global decoder uses the measured syndrome to infer the most likely homology class of the unknown error chain. Equation (9) gives the posterior probability of a homology class conditioned on $S$. Equation (10) states the threshold criterion in terms of vanishing probability that an alternative chain differs from the actual chain by a homologically nontrivial cycle.

For practical decoding, DKLP studies a minimum-weight chain $E_{\rm min}$ with the same boundary as the syndrome chain. The weight is anisotropic when data and measurement error probabilities differ: horizontal links get weight $\log((1-p)/p)$ and vertical links get weight $\log((1-q)/q)$; see Section V.A and Equation (45). This is the spacetime version of minimum-weight matching. The paper proves a conservative lower bound: for $p=q$, recovery succeeds when $p,q<0.0114$; see Equation (63). This is a bound, not the actual best threshold.

The paper also explains finite-memory recovery. A decoder need not keep an infinite history; Section VI and Figure 13 describe an overlapping recovery method that stores a finite window, acts only on sufficiently old/trustworthy events, and carries forward recent boundary data. For $p\simeq q$, a window time $T\gg L$ is sufficient in their bound. This motivates both Lake's buffer idea and this repository's concern with history depth and classical overhead.

For CNOT work, DKLP matters because it shows that gates couple syndrome histories across blocks. Section VI.C discusses how transversal CNOT propagates errors between blocks and forces the decoder to use syndrome data from the causal past of the interacting blocks. Equation (104) gives the Pauli propagation identities for CNOT. This is exactly why primitive CNOT merging can lose information, and why sheet-copying works but has overhead growth.

## Notation

| Symbol | Meaning |
| --- | --- |
| $L$ | Linear size and distance scale of the square toric/planar code. |
| $\ell$ | Lattice link; a physical qubit is associated with each link. |
| $s$ | Site or vertex of the lattice. |
| $P$ | Plaquette of the lattice. |
| $X_s$ | Site check, product of $X$ operators on links incident on $s$. |
| $Z_P$ | Plaquette check, product of $Z$ operators on links in $P$. |
| $\partial$ | Boundary operator on $\mathbb Z_2$ chains. |
| $E$ | Actual error chain in space or spacetime. |
| $E'$ | Hypothetical error chain consistent with the same observed syndrome boundary. |
| $E_{\rm min}$ | Minimum-weight chain with the same boundary as the syndrome chain. |
| $S$ | Syndrome chain: vertical links where measured syndrome is nontrivial in the spacetime history. |
| $C,C',D$ | Cycles used to compare actual and hypothetical chains or homology classes. |
| $h$ | Homology class of a cycle. |
| $p$ | Data-qubit error probability per qubit per time step. |
| $q$ | Syndrome-measurement error probability per stabilizer measurement. |
| $H,V$ | Number of horizontal data-error links and vertical measurement-error links in a spacetime chain. |
| $T$ | Number of syndrome-measurement rounds in a finite recovery window. |
| $\Pi(E')$ | Projection of a spacetime chain onto the final time slice. |
| $p_{\rm single}$ | Effective probability of an isolated data error per round in the circuit-level analysis. |
| $q_{\rm single}$ | Effective probability of an isolated syndrome bit error per round. |
| $p_{\rm hook},q_{\rm hook}$ | Effective probabilities for correlated hook errors from syndrome-extraction circuits. |

## Conceptual Flow by Section

### Section I: Introduction

The introduction motivates surface codes as local-gate-friendly fault-tolerant memories. The key assumptions for the main threshold estimate are local quantum gates, rapid measurements, and fast reliable classical processing. The introduction states the main storage lower bound: under their assumptions, if data phase errors, data bit-flip errors, and syndrome bit errors are each below about $1.14\%$, recovery succeeds in the large-block limit.

### Section II: Fault Tolerance and Quantum Architecture

This section lists architectural assumptions: constant error rate, weakly correlated errors, parallel operation, reusable memory, fast measurements, fast accurate classical processing, no leakage, and locality considerations. The paper's main two-dimensional recovery protocol keeps quantum operations local but allows nonlocal classical computation.

### Section III: Surface Codes

Section III.A defines toric codes. For an $L\times L$ torus, qubits live on $2L^2$ links. The site stabilizer is $X_s=\bigotimes_{\ell\ni s}X_\ell$, and the plaquette stabilizer is $Z_P=\bigotimes_{\ell\in P}Z_\ell$. Figure 1 shows the checks. Figures 2 and 3 show homologically trivial and nontrivial cycles and logical operators.

The syndrome is a set of sites or plaquettes with check value $-1$, interpreted as defects. Figure 4 shows the ambiguity: two chains with the same boundary produce the same syndrome. Recovery succeeds if the actual error chain plus recovery chain is homologically trivial.

Section III.B defines planar codes and rough/smooth boundaries. Figure 5 is the reference for planar boundaries, relative homology, and single defects terminating at compatible boundaries.

Section III.C explains why faulty measurements require repeated syndrome history. Figure 6 shows correlated defect pairs at low error rate. Figure 7 shows genuine defects, ghost defects, and missed defects.

### Section IV: The Statistical Physics of Error Recovery

Section IV.A defines the phenomenological noise model: data errors on horizontal links with probability $p$ and measurement errors on vertical links with probability $q$.

Section IV.B introduces the spacetime lattice. For toric-code phase-error recovery, data errors are horizontal links and syndrome measurement outcomes are vertical links. Figures 8 and 9 show the syndrome history and error history in the repetition-code case, but the same logic applies to surface-code spacetime decoding.

Section IV.C is the core. The measured syndrome chain is $S$, the actual error chain is $E$, and the true defect worldline chain is $S+E$. Since worldlines do not end, $\partial(S+E)=0$, so $\partial S=\partial E$. Equation (9) gives the posterior probability of a homology class. Equation (10) gives the threshold criterion: homologically nontrivial alternatives must have vanishing total conditional probability as $L\to\infty$.

Sections IV.D--IV.F relate this to statistical mechanics. Perfect measurement gives a two-dimensional random-bond Ising model. Faulty measurement gives a three-dimensional $\mathbb Z_2$ lattice gauge theory with quenched disorder. Equation (40) quotes the numerical random-bond Ising value $p_c=0.1094\pm0.0002$ for the perfect-measurement case.

### Section V: Chains of Minimal Weight

Section V.A defines the minimal-weight recovery chain. If a chain has $H$ horizontal and $V$ vertical links, the minimum-weight objective is Equation (45):

$$
H\log\left(\frac{1-p}{p}\right)+V\log\left(\frac{1-q}{q}\right).
$$

This is the global spacetime matching baseline for local decoders.

Section V.B derives chain-probability bounds. Figure 12 shows $E$ and $E_{\rm min}$ whose sum includes a nontrivial cycle. Equations (47)--(54) bound the probability of a damaging self-avoiding polygon. Equation (63) gives the conservative sufficient condition $p,q<0.0114$ for $p=q$.

### Section VI: Error Correction for a Finite Time Interval

Section VI explains why infinite syndrome storage is not necessary. In a finite interval, $S+E$ may have open paths ending on the final time slice. The decoder constructs a chain with the same relative boundary, projects it onto the final slice, and corrects the projected data errors.

Figure 13 shows overlapping recovery. Old monopoles are corrected and erased from the record, while recent boundary data is retained. For comparable $p$ and $q$, the paper argues that it suffices to take $T\gg L$.

Section VI.C discusses computation. When a transversal CNOT is applied, errors from one block propagate to the other, so the history for a block after the gate may depend on syndrome histories from the other block before the gate. This is the conceptual reason CNOT decoding needs lineage or causal-history tracking.

### Section VII: Quantum Circuits for Syndrome Measurement

Section VII studies a syndrome-extraction circuit using one ancilla per check. Figure 14 gives check-measurement circuits. Figure 15 gives the ordering of CNOTs around a data qubit.

Equation (72) estimates isolated syndrome measurement error:

$$
q_{\rm single}=p_p+4p_{\rm CNOT}+6p_s+p_m+\hbox{higher order}.
$$

The section also identifies vertical and horizontal hook errors. Equation (89) gives a conservative gate-accuracy sufficient condition involving $q_{\rm hook}=3p_{\rm CNOT}+2p_s$.

### Sections VIII--IX: Measurement, Encoding, and Fault-Tolerant Computation

These sections cover encoded measurement, code growth, and logical gates. For this project, the most useful single reference is Equation (104), the CNOT Pauli propagation rule:

$$
XI\mapsto XX,\quad IX\mapsto IX,\quad ZI\mapsto ZI,\quad IZ\mapsto ZZ.
$$

This identifies which error sector propagates in which direction during a CNOT.

### Section X: A Local Algorithm in Four Dimensions

Section X gives a robust local recovery procedure that avoids measurement and fast classical computation, but requires four spatial dimensions for full locality. This is conceptually interesting but not directly applicable to the repository's two-dimensional toric-code decoder.

### Section XI: Conclusions

The conclusion reiterates the computational model: local quantum gates, fast measurements, and perfect fast classical processing. The accuracy threshold is interpreted as a phase transition in a three-dimensional lattice gauge theory with quenched randomness.

## Major Results

### Result 1: Surface-Code Stabilizers and Homological Logical Operators

**Plain-language statement.** Surface-code checks are local products of Pauli operators. Errors create pairs of defects at chain boundaries. Logical errors correspond to homologically nontrivial cycles.

**Formal statement.** Section III.A defines $X_s$ and $Z_P$. Figure 1 shows check operators. Figures 2 and 3 show homologically trivial cycles, nontrivial cycles, and logical operators. Figure 4 shows syndrome ambiguity.

**Assumptions.** The basic toric-code discussion uses periodic boundary conditions. Planar-code variants with rough and smooth edges are in Section III.B and Figure 5.

**Why it matters for decoding.** A decoder does not need to identify the exact error chain. It must choose a recovery chain in the correct homology class.

**Implementation relevance.** In the repository's one-sector toric-code implementation, syndrome calculation should be exactly a boundary map from edge errors to vertex or plaquette defects. Logical failure must be tested by nontrivial winding after applying correction, not merely by nonempty syndrome.

### Result 2: Faulty Measurement Turns Decoding into a Spacetime Problem

**Plain-language statement.** Measurement errors create false syndrome events. Repeating measurements turns decoding into a three-dimensional chain problem with time as the extra dimension.

**Formal statement.** Section IV.B defines horizontal links as data errors and vertical links as measurement errors. Figure 8 shows syndrome history; Figure 9 shows the error history and syndrome history with the same boundary. Section IV.C states $\partial S=\partial E$.

**Assumptions.** The phenomenological model assumes independent data errors with probability $p$ and independent measurement errors with probability $q$.

**Why it matters for decoding.** The decoder's input should be syndrome changes over time, not just a single syndrome snapshot.

**Implementation relevance.** The array `hist` in this project should store $old\_synds \oplus new\_synds$. A current syndrome bit alone cannot identify whether a defect was created by data noise, measurement noise, or previous recovery dynamics.

### Result 3: Optimal Recovery Is a Homology-Class Inference Problem

**Plain-language statement.** Given the measured syndrome history, all chains with the same boundary are possible. The best decoder chooses the most likely homology class.

**Formal statement.** Equation (9) gives

$$
{\rm prob}(h|S)=
\frac{\sum_{C'\in h}{\rm prob}(S+C')}
{\sum_{C'}{\rm prob}(S+C')}.
$$

Equation (10) gives the threshold criterion in terms of the vanishing probability of homologically nontrivial alternatives.

**Assumptions.** This is an ideal global classical inference view. It assumes the decoder can process the recorded syndrome history.

**Why it matters for decoding.** MWPM and minimum-weight-chain decoders are approximations or special cases of this homology inference.

**Implementation relevance.** Local message passing should be understood as a local approximation to homology-class inference. When CNOTs mix histories, preserving homology-relevant causal information matters more than preserving a superficial current state.

### Result 4: Minimum-Weight Spacetime Chains Give a Global Baseline

**Plain-language statement.** A practical global decoder can choose the most likely chain by minimizing weighted spacetime length.

**Formal statement.** Section V.A defines $E_{\rm min}$ with $\partial E_{\rm min}=\partial S$ and objective Equation (45). The failure-probability analysis uses Figure 12 and Equations (53)--(54). Equation (63) gives the sufficient bound $p,q<0.0114$ for $p=q$.

**Assumptions.** The lower bound is conservative. The algorithm assumes fast, reliable global classical computation.

**Why it matters for decoding.** This is the nonlocal baseline that Lake's local message passing tries to approximate without storing/processing the full history globally.

**Implementation relevance.** For validation, compare local decoder failures to a spacetime matching or windowed matching mental model. The local decoder's history buffer is a compressed, local alternative to a global graph over an $L\times L\times T$ syndrome history.

### Result 5: Finite Overlapping Recovery Controls Classical Memory

**Plain-language statement.** A decoder can use a finite time window if it treats recent syndrome information cautiously and only finalizes older, more reliable pairings.

**Formal statement.** Section VI and Figure 13 define overlapping recovery. The method decomposes a minimum chain into $E'_{\rm old}$ and $E'_{\rm keep}$, corrects only the old part, and carries the remaining boundary data forward.

**Assumptions.** The argument relies on exponential decay of correlations below threshold and still uses global minimum-weight chains within windows.

**Why it matters for decoding.** It explains why $O(L)$ syndrome windows are enough for global schemes and why a full infinite history is unnecessary.

**Implementation relevance.** Lake's buffer is a local, self-organized analogue of finite-window recovery. CNOT sheet-copying is closer to preserving complete windows; a better project design should preserve only the needed old/recent boundary information and causal lineage.

### Result 6: Syndrome Circuits Create Hook Errors

**Plain-language statement.** Local syndrome extraction can create correlated two-qubit errors, but careful ordering controls their orientation and impact.

**Formal statement.** Section VII, Figures 14--15, Equation (72), and Equation (89) give the circuit-level estimates and a conservative sufficient gate-accuracy condition.

**Assumptions.** The analysis is pessimistic and first-order in elementary error probabilities.

**Why it matters for decoding.** Circuit-level noise is not exactly the independent phenomenological model. Hook orientation and correlation matter.

**Implementation relevance.** Current CNOT prototypes do not implement a circuit-level syndrome-extraction or gate-fault model. Do not infer circuit-level thresholds from phenomenological simulations without adding correlated hook and CNOT fault events.

## Implementation Notes for Future Agents

### Suggested Data Structures

- Physical edge state: Boolean error state per edge orientation.
- Stabilizer syndrome: Boolean current and previous measurement per site or plaquette.
- Defect event: spacetime coordinate, computed from a syndrome change.
- Error/recovery chain representation: sets of spacetime links, with horizontal links for data changes and vertical links for measurement faults or temporal syndrome changes.
- Window state: finite list or buffer of defect events plus unresolved boundary data.
- Logical/homology tracker: winding parity or relative-boundary class after correction.
- CNOT lineage record: block id, source/target role, error sector, gate time, and causal predecessor sheets or summaries.

### Quantities to Store Per Site, Stabilizer, or Defect

- The latest two syndrome measurements at each stabilizer.
- The defect event time or buffer depth.
- Whether a defect is associated with an $X$-sector or $Z$-sector recovery problem.
- In planar codes, whether a boundary can absorb that defect type.
- In CNOT code, whether the defect's causal history crosses a CNOT and from which block.
- Pending recovery-chain or correction contribution, not just the visible defect bit.

### What Should Be Updated Each Round

1. Apply or sample data noise.
2. Measure stabilizers and sample measurement noise.
3. Compare new and old syndrome to create spacetime defect events.
4. Advance the decoder's finite history/window.
5. Pair, move, or locally process defects according to the decoder rule.
6. Apply physical corrections associated with horizontal recovery links.
7. Preserve unresolved recent boundary data for future rounds.

### What Not to Conflate

- Do not conflate the measured syndrome chain $S$ with the actual error chain $E$.
- Do not conflate a ghost defect with a true data-error endpoint.
- Do not conflate recovery success with empty syndrome at one time slice; homology class matters.
- Do not conflate finite-window decoding with deleting old history arbitrarily. Old information is acted on; recent unresolved boundaries are retained.
- Do not conflate CNOT-propagated errors with independent post-CNOT target errors. Their causal histories differ.

### Common Implementation Mistakes

- Storing only current syndrome and losing syndrome-change events.
- Treating measurement errors as data-edge errors instead of vertical spacetime links.
- Forgetting that planar boundaries absorb only compatible defect types.
- Applying a correction chain with the right boundary but wrong homology class.
- Merging CNOT histories from two blocks without preserving source-block provenance.
- Comparing a local decoder directly to DKLP's threshold bounds without accounting for global classical computation assumptions.

### Theoretical Assumptions Not Yet Implemented

- DKLP's main threshold estimate assumes fast perfect classical processing.
- The repository's local decoder is not a global minimum-weight-chain decoder.
- The current code does not implement both sectors of a full surface-code CNOT.
- The current code does not include syndrome-circuit hook errors or CNOT gate faults.
- DKLP's finite-time overlapping recovery is a global method; Lake's buffer is a different local mechanism inspired by the same spacetime history issue.

## Connections to Our Project

DKLP explains why `hist` must contain syndrome changes. In the paper's notation, the measured syndrome chain $S$ and the actual error chain $E$ have the same boundary. In code, the new defect event is the difference between two consecutive syndrome measurements. This is the object that enters Lake's buffer and the local message-passing fields.

DKLP's minimum-weight chain is the global/nonlocal baseline. Lake's message-passing decoder is a local self-organized alternative that tries to produce a reasonable recovery chain without constructing the whole spacetime matching graph.

For CNOT, DKLP's Section VI.C is the key warning. A CNOT propagates errors between blocks, so a block's post-gate syndrome history may depend on another block's pre-gate history. Primitive CNOT merging can lose which defects came from which causal past. Sheet-copy CNOT preserves the distinction by duplicating decoder sheets, but this can grow classical overhead under repeated CNOTs.

Future CNOT decoder designs should keep enough information to distinguish defect origin, time, source block, target block, CNOT gate id, and error sector. A design that stores only a merged Boolean history is unlikely to have enough information to reproduce the homology-class inference that DKLP motivates.

## Do Not Overclaim

- The $1.14\%$ value in Equation (63) is a rigorous lower bound from a conservative counting argument, not the exact phenomenological threshold.
- The perfect-measurement value $p_c\simeq 10.94\%$ in Equation (40) comes from numerical/statistical-physics input for a different setting.
- The global minimal-chain decoder assumes fast reliable classical computation and is not local in the sense Lake targets.
- The finite-time overlapping method reduces memory to a finite window but still uses global chain computation within windows.
- The circuit-level estimates are pessimistic first-order sufficient conditions, not optimized hardware thresholds.
- The four-dimensional local algorithm in Section X does not directly apply to this project's two-dimensional toric-code implementation.
- DKLP does not solve the low-overhead repeated-CNOT lineage problem in this repository; it explains why the problem exists.
