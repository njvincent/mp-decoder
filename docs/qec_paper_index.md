# Cross-Paper Index for QEC Decoder Work

## Purpose

This index relates two reference notes:

- `docs/dklp_2001_topological_quantum_memory.md`
- `docs/lake_2025_simulated_confinement.md`

DKLP gives the spacetime error-chain and syndrome-history framework. Lake builds a local active decoder that tries to simulate confinement between spacetime defects without global matching or a full stored history. Together they explain why this project tracks a rolling history buffer and why CNOT decoder designs must preserve defect provenance across gates.

## How the Papers Relate

| Concept | DKLP | Lake |
| --- | --- | --- |
| Surface/toric code | Defines qubits on links, site checks $X_s$, plaquette checks $Z_P$, logical cycles, planar boundaries; see DKLP Section III and Figures 1--5. | Assumes topological stabilizer codes with Abelian anyons, especially $2d$ toric/surface code and $1d$ repetition code. |
| Syndrome bit | A stabilizer measurement outcome. Faulty measurements mean a single snapshot is unreliable. | The decoder stores the latest syndrome value at each processor and compares consecutive values. |
| Defect event | In spacetime, a defect is a boundary point of the syndrome/error chains; see DKLP Figures 8--9. | A defect is explicitly a syndrome-changing event inserted into the buffer at $z=0$. |
| Error chain | Actual data and measurement faults, with horizontal links for data errors and vertical links for measurement errors. | Noise clusters generate buffered defect events; small clusters should erode before reaching the back wall. |
| Recovery chain | A hypothesized chain with the same syndrome boundary. Recovery succeeds if actual plus recovery chain is homologically trivial. | Local feedback moves defects and applies physical corrections, creating a local recovery chain without global matching. |
| Global matching/windowed decoding | Minimum-weight spacetime chains; finite overlapping windows can reduce infinite history to $T\gg L$ windows. | Nonlocal or high-memory baseline. Windowed MWPM needs $O(L)$ temporal depth per processor for maximal suppression. |
| Message-passing confinement | Not present as an algorithm; DKLP supplies the target spacetime problem. | Core algorithm: defects emit integer messages and move toward local message minima. |
| Buffer/back wall | Finite-time recovery keeps old/recent history carefully but does not use Lake's $z$ dimension. | The $z$ buffer is a dynamic local history/RG dimension; the back wall handles residual large-scale defects. |
| Classical overhead | Full histories and global chain computation are allowed in the main model; finite windows reduce but do not localize all processing. | The decoder targets local processing with $O(\operatorname{polylog}L)$ reliable classical bits per qubit analytically, likely $O(\log L)$ numerically. |
| CNOT histories | Section VI.C explains that CNOTs make a block's history depend on another block's causal past; Equation (104) gives Pauli propagation. | Lake does not solve CNOT decoding, but his buffer tells us what history information a local decoder needs to preserve. |

## Concept Mapping

| Project phrase | Paper meaning |
| --- | --- |
| Syndrome bit | A measured stabilizer value at a site/plaquette and a round. Store at least current and previous values. |
| Defect event | A syndrome-change event: in code, usually `old_synds xor new_synds`; in DKLP, a boundary point in the spacetime chain picture. |
| Error chain | The unknown set of data and measurement faults. In spacetime, data errors are horizontal links and measurement faults are vertical links. |
| Syndrome history | The measured chain $S$ through time. DKLP's global decoder uses it directly; Lake stores a local rolling buffer. |
| Recovery chain | The correction selected by the decoder. In global DKLP it is $E_{\rm min}$ or a most-likely homology class; in Lake it is produced by defect motion and local feedback. |
| Global matching | Nonlocal minimum-weight chain/matching over the syndrome history. Strong baseline, but needs large memory and fast global classical computation. |
| Windowed decoding | A finite-history version of global matching. It reduces infinite history but still uses windows of temporal depth $O(L)$ for maximal suppression. |
| Message passing | Lake's local alternative: integer messages propagate between defects and approximate attraction. |
| Simulated confinement | The intended effect of message passing: defects move toward partners as if confined, approximating minimal recovery chains. |
| Buffer depth $Z$ | Lake's extra classical dimension. It is not Pauli $Z$ and not physical time; it is a rolling local history/RG coordinate. |
| Back wall | The $z=Z$ surface where surviving large-scale defects accumulate and run a lower-dimensional message-passing decoder. |
| Pseudothreshold | Lake's warning for $Z=0$: finite-size scaling can look threshold-like even though $\tau_{\rm mem}$ saturates as $L\to\infty$. |
| True threshold | For fixed $p<p_c$, logical memory improves without bound as $L$ grows. DKLP establishes this for global recovery; Lake proves it for modified local message passing with sufficient $Z$. |
| CNOT source history | The causal syndrome/error history that propagates through a CNOT from one logical block to another. DKLP motivates why it must be tracked; current prototypes explore primitive merge versus sheet-copy preservation. |

## Which File Should an Agent Read First?

1. Read this index first for the cross-paper map and project vocabulary.
2. Read `implementation.md` before touching code; it documents the actual arrays and current prototype behavior.
3. Read `lake_2025_simulated_confinement.md` before changing local decoder update rules, buffer behavior, message fields, back-wall logic, asynchronous logic, or classical-overhead assumptions.
4. Read `dklp_2001_topological_quantum_memory.md` before changing syndrome definitions, logical-error criteria, spacetime history handling, matching baselines, finite windows, or CNOT history logic.
5. Read the relevant Julia file last and verify exact implementation details before editing; the repo contains duplicated decoder logic across baseline, threaded, primitive CNOT, and sheet-copy CNOT files.

## Project-Specific Guidance

- The local memory decoder is Lake-like: `hist` is a rolling buffer of DKLP-style syndrome-change events, and `fields` are Lake-style messages.
- Primitive CNOT is cheap because it merges arrays, but merging can destroy DKLP/Lake provenance: defect origin, time, source block, and causal path through the CNOT.
- Sheet-copy CNOT is accurate because it preserves independent histories, but it pays by copying entire decoder sheets. Under repeated CNOTs this can grow too quickly.
- The next useful CNOT design should preserve homology-relevant history in compressed form: enough to distinguish causal origins, not so much that every active control history is deep-copied forever.
- Always distinguish the error sector. The current CNOT prototypes mainly demonstrate the X-sector propagation rule. A full surface-code computation decoder must also handle the complementary sector.

## Glossary of Shared Terms

| Term | Definition |
| --- | --- |
| Anyon | A nontrivial stabilizer excitation in the topological-code picture. In faulty measurement settings, current anyons are not enough; use spacetime defects. |
| Defect | A syndrome-change event or chain boundary in spacetime. In code, it should be derived from consecutive syndrome measurements. |
| Ghost defect | A defect-like measured event caused by a measurement error rather than a data error. DKLP Figure 7. |
| Error chain | The unknown physical fault chain whose boundary matches the observed syndrome-chain boundary. |
| Recovery chain | The decoder's chosen correction chain. Correct recovery requires the combined actual-plus-recovery chain to be homologically trivial. |
| Homology class | The equivalence class of cycles modulo boundaries. Logical failure occurs when the combined chain is homologically nontrivial. |
| Minimum-weight chain | DKLP's global chain estimate minimizing anisotropic spacetime weight, Equation (45). |
| Message field | Lake's integer local signal used to approximate nearest-defect attraction. |
| Screening | Lake's mechanism for pseudothresholds: small noise pairs interrupt or overwhelm the signal between far-separated defects. |
| Buffer | Lake's dynamic local history storage of depth $Z$. |
| Back wall | The final buffer slice where residual large-scale events are decoded. |
| Lineage | Project term for a decoder sheet or summary's causal origin through CNOT gates. Not a DKLP/Lake term, but required by their spacetime-history logic. |

## Do Not Overclaim Across Papers

- DKLP proves and estimates thresholds for global classical recovery under strong classical-processing assumptions. Lake proves a local threshold for a modified message-passing decoder with reliable classical memory. These are different computational models.
- Lake's $1.5\%$ surface-code value is numerical and for phenomenological noise. It is not DKLP's conservative $1.14\%$ lower bound and not global MWPM's higher threshold.
- Neither note proves that the current CNOT prototypes have a true threshold. The notes explain what information such a decoder must preserve.
- Neither paper removes the cost of reliable classical storage. Lake reduces the asymptotic memory for local active memory decoding, but the repeated-CNOT lineage problem remains a project problem.
