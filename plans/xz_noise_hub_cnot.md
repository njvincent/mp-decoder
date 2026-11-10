# Plan: X/Z Depolarizing Noise, Noisy CNOT Rounds, and the Hub CNOT Circuit

Status (2026-11-10): Stage 1 is implemented and scanned
(`results/cnot_primitive_xz/stage1/`). Stage 2's hub schedule, frame
snapshots, local runner, and Slurm script are implemented for one design: a
fixed gap `k` at every size and `N = c d` CNOTs. Its first local scan (`k = 1`,
`N = L, 2L`, `results/cnot_primitive_xz/hub/k_1/`) is analyzed in
`notebooks/cnot_primitive_xz_hub_fixed_gap_analysis.ipynb`. The implementation
is documented in `docs/implementation.md` Section 4.6, and the results in
Sections 5.2.8-5.2.9.

Stage 1 adds both error sectors, depolarizing data noise, and noisy CNOT
rounds. Stage 2 adds the hub circuit and the target's per-gate error rate,
threshold, and suppression factor, plus a rate for every block. A Y-junction
gate rule is a later stage (Section 12).

## 1. Decisions fixed with the user

| Item | Decision |
| --- | --- |
| Data noise per round | Single-qubit depolarizing with total probability `p`: X, Y, Z each with `p/3`, on every data qubit of every live block. |
| Measurement noise | Every X-type and Z-type syndrome bit flips independently with `q = QRAT * p` (default `QRAT = 1`), as now. |
| Gate noise | After each transversal CNOT, every physical pair gets a two-qubit depolarizing channel of total `p2 = 10p`: each of the 15 non-identity Paulis with `10p/15`. The ratio is exposed as `P2_RATIO` (default 10). |
| CNOT round | single-qubit noise -> ideal CNOT -> two-qubit noise -> noisy syndrome -> decoder update. The CNOT round is one round. |
| Decoder rule at a gate | Primitive merge, mirrored for the Z sector (Section 4). Y-junction comes in the next stage. |
| Hub direction | A is always the target. |
| Partner lifetime | Each partner gets `d` noisy rounds, its CNOT round, then `d` noisy rounds, cleanup, and readout. Its failures count. |
| Metric | The final error rate is characterized by the target A alone: A's final readout, either sector. Lake-style fixed-aspect threshold, plus Google-style per-gate rate and `Lambda`. Every block also gets its own rate (Section 8). |
| Gate density | Fixed: the gap `k` (ordinary hub rounds between CNOTs) is the same for every `L`, and the number of CNOTs scales with the distance, `N = c d` with `d = L` (`c = 1, 2`). The density `1/(k+1)` and the aspect ratio are then the same at every `L`. The first scan uses `k = 1`; other `k` come later. |
| Sizes | `L in {5, 7, 9, 11, 13}`; `L = 19` only if the budget allows later. |
| Local runs | Run on this machine, about 10^4 trials per point. Stage 1 used 12 threads; later local runs use at most 8 so the machine stays usable. Slurm scripts are written for Stage 2. |
| Kernel | The new driver gets a non-allocating kernel, checked bit-for-bit against legacy `update!` after the correctness tests pass. |

## 2. Problem

Established by the code (`docs/implementation.md` Sections 2.1, 4.2.4):

- All drivers track one sector: X errors on edges, vertex checks,
  Bernoulli(`p`) flips, Bernoulli(`q`) measurement flips.
- CNOTs are ideal and instantaneous between two rounds. There is no gate noise.
- In `cnot_primitive_consecutive.jl`, every control lives for the whole
  `(N+1)T`-round experiment, and control `j` idles `jT` rounds before its gate.
  The saved count scan shows idle control memory taking over the signal. At
  `L=9, p=0.01`, the any-control failure rate rises from 1.4% (`N=1`) to 14.5%
  (`N=4`), against 3.8% to 16.1% for the target.

Goal: a physically complete Pauli model (both sectors, Y correlations, gate
noise), plus a schedule whose per-gate cost is not mixed with idle memory.

Out of scope here: circuit-level syndrome extraction (hook errors),
cross-sector or cross-block correlated decoding, the asynchronous update
branch, and the Y-junction rule.

## 3. Physical model

### 3.1 Two sectors on the torus

Qubits sit on edges `e=(i,j,a)` of an `L x L` periodic lattice. Edge `a=1`
joins `(i,j)` to `(i+1,j)`; edge `a=2` joins `(i,j)` to `(i,j+1)`.

- X sector: X errors are stored in primal indexing `x[i,j,a]`. The syndrome is
  the existing vertex parity `get_synds`, and the logical test is the existing
  `detect_logical_error`.
- Z sector: Z errors flip plaquette checks. They are stored on the dual lattice
  (dual vertex `(i,j)` is the plaquette with lower-left corner `(i,j)`) through
  the edge bijection

  ~~~text
  sigma(i,j,1) = (i,   j-1, 2)      indices mod L
  sigma(i,j,2) = (i-1, j,   1)
  ~~~

  Then `get_synds(zdual)[i,j]` is the parity of plaquette `(i,j)`, and
  `detect_logical_error(zdual)` is the Z-sector winding test. The decoder
  kernel therefore runs on the Z sector unchanged.
- `sigma` matters only when one physical Pauli touches both sectors (Y, and
  two-qubit Paulis). The transversal CNOT pairs edge `e` of C with edge `e` of
  T, and `sigma` is the same in every block, so the gate acts elementwise on
  both the `x` and the `zdual` arrays.

### 3.2 Ordinary round (each live block)

This keeps the synchronous order of `docs/implementation.md` Section 3.2.2,
split into three phases:

1. decode, per sector: `r` message sweeps, feedback proposals, frame update,
   history commit (legacy steps 1-6, same tie order and 0.8 back-wall draw);
2. data noise: draw one Pauli per edge `e`, with X/Y/Z each at `p/3`;
   `x[e] xor= (X or Y)` and `zdual[sigma(e)] xor= (Z or Y)`;
3. measure, per sector: `old = new`, `new = get_synds xor Bernoulli(q)`, RG
   cycle, insert `old xor new` at `k=1` (legacy steps 8-10).

Per-sector marginals: the data flip rate is `2p/3` per round (X and Z flips
correlated, `P(both) = p/3`), and the measurement flip rate is `q`. The old
model flipped X with probability `p`, so at the same `p` and `q` the per-sector
data noise is now 2/3 of before. Thresholds on the `p` axis are therefore not
directly comparable with the existing X-only scans.

### 3.3 CNOT round (control C, target T)

1. decode both blocks and both sectors on their own pre-gate buffers;
2. data noise on both blocks (this noise is copied by the gate);
3. ideal transversal CNOT on the physical arrays:
   `x_T xor= x_C`, `zdual_C xor= zdual_T`;
4. decoder rule at the gate (Section 4);
5. two-qubit noise: for each pair (`e` in C, `e` in T), with probability `p2`,
   apply one of the 15 non-identity Paulis drawn uniformly, mapping its X/Z
   components into both blocks through `sigma`;
6. measure both blocks and both sectors. This is the first post-gate syndrome.

The user's ordering puts the decoder update after the syndrome. The code keeps
it at the start of the next round (step 1), which gives the same sequence
except for the trivial first round and the final cleanup.

Gate-noise marginals per sector: flip C only, flip T only, and flip both each
occur with probability `4p2/15` (= `8p/3` at `P2_RATIO=10`). The per-qubit
X flip probability is `16p/3`.

Difference from all current drivers: there, noise after the gate is post-gate.
Here each gate adds one round, and that round's data noise comes before the
gate and is copied by it.

`GATE_MODE=idle` skips steps 3-4. `P2_RATIO` controls step 5 independently.
The four combinations give the ablations: pure memory round, gate noise
without propagation, ideal gate, full gate.

## 4. Decoder rule: mirrored primitive merge

Define `merge_primitive!(dst, src)` as the existing primitive map (Section 4.2.2
of the implementation doc) without the physical error array, which step 3
already handled:

~~~text
dst.frame     xor= src.frame
dst.old_synds xor= src.old_synds
dst.new_synds xor= src.new_synds
dst.hist      xor= src.hist
dst.fields     =   nonzeromin(dst.fields, src.fields)
dst.new_fields = 0;  src.new_fields = 0;  both hist_correction = 0
~~~

A CNOT `C -> T` calls `merge_primitive!(T.x, C.x)` (X errors flow from control
to target) and `merge_primitive!(C.z, T.z)` (Z errors flow from target to
control).

Consequences, from the code and 2408.01393 Appendix B:

- The first post-gate event in the absorbing decoder is
  `S_dst(t) xor S_src(t) xor S_dst(t+1)`. That is the hybrid-frame detector.
  Later events are ordinary static-frame changes.
- The absorbing decoder must also decode the source's buffered pre-gate
  defects, which the source decoder keeps decoding too. This is the primitive
  rule's provenance loss, now present in both sectors. In the hub circuit, A's
  X decoder absorbs every partner, and every partner's Z decoder absorbs A's.

Extensibility: the engine calls the rule only at gate time, and per-round
decoding goes through one dispatch point (`decode_step!` on a lane). The
Y-junction stage replaces `merge_primitive!` on the absorbing side and adds a
junction lane type.

## 5. State and code structure

New standalone driver: `cnot_primitive_xz.jl`. It is synchronous only and
rejects `SYNCH=false`, because the asynchronous branch samples single-edge
noise, which is incompatible with joint X/Z sampling. Its kernel is copied from
`cnot_primitive_consecutive.jl` and it imports no other driver, matching the
two-pass and Y-junction convention. Physical state and decoder state are kept
separate, matching the Y-junction vocabulary, so the next stage can attach
extra lanes to one physical block:

~~~text
SectorState    errors (L,L,2; x primal / z dual), frame (L,L,2),
               old_synds (L,L), new_synds (L,L)
DecoderLane    hist (L,L,Z), hist_correction (L,L,Z,3),
               fields, new_fields (L,L,Z,3,2)
LogicalBlock   id, x::(SectorState, DecoderLane), z::(SectorState, DecoderLane),
               first_round, last_round
CnotEvent      round, control_id, target_id
~~~

Core functions:

- `decode_step!(state, lane, r, rng)`
- `measure_and_insert!(state, lane, q, rng)`
- `apply_single_qubit_depolarizing!(block, p, rng)`
- `apply_transversal_cnot!(C, T)`
- `merge_primitive!`
- `apply_two_qubit_depolarizing!(C, T, p2, rng)`
- `run_round!(live_blocks, events, params, rng)`
- `cleanup_and_readout!(block, ...)`
- schedule builders (Section 6)
- `estimate_xz_Ft`, which keeps the consecutive driver's `STOP_MODE`, quota,
  and output conventions.

Every stochastic call takes an explicit per-worker RNG (`Xoshiro`, seed from
`SEED` or drawn and recorded in `### params ###`).

Memory: each block holds `2 M_block` (implementation doc, Section 2.3). Blocks
come from a reusable pool sized to the largest live set; for the hub that is at
most `1 + ceil((2d+1)/(k+1))` blocks.

## 6. Schedules

Rounds are numbered `tau = 1, 2, ...`. A block lives on
`[first_round, last_round]` and is read out right after `last_round`.

1. `memory`: one block on `[1, T_mem]`. This is the reference under the new
   noise model.
2. `consecutive` (Stage 1 continuity): T and C1..CN live on `[1, sum(I) + N]`.
   CNOT `j` happens at round `I1 + ... + Ij + j`. This is the existing
   `CNOT_INTERVALS` structure plus one CNOT round per gate.
3. `hub` (Stage 2), with parameters `d = TVAL` (default `L`), `N = CNOT_COUNT`,
   `k = GAP_ROUNDS >= 0`:

   ~~~text
   tau_j = d + 1 + (j-1)(k+1)        CNOT round with partner j, j = 1..N
   A    lives on [1, tau_N + d]
   P_j  lives on [tau_j - d, tau_j + d]       (P_1 = B starts with A)
   event j = (tau_j, control = P_j, target = A)
   ~~~

   Every partner gets `d` ordinary rounds, its CNOT round, and `d` ordinary
   rounds (`2d+1` in total). A gets `k` ordinary rounds between CNOT rounds and
   `d` after the last one. Scans fix `k` across sizes and set `N = c d`.
   `N=1` is the single-gadget experiment of 2408.01393 and is identical to
   `consecutive` with intervals `[d,d]`; this is a test of the schedule
   engine.

Early partner readout is exact. After its gate, a partner never interacts with
A again: physically there are no further gates, and under the primitive rule
the two decoders are independent from then on.

## 7. Readout and recorded counts

The failure convention is unchanged. Cleanup runs with `p = q = 0` on both
sectors until both buffers are empty or `CLEANUP_TIME` (default `2d`) rounds
have passed; then the winding tests run. A block fails if X winding or Z
winding is nontrivial. The driver also counts failed trials (any block); the
headline uses A alone (Section 8). Cleanup failure is recorded separately and
not ORed in (implementation doc, Sections 3.3, 4.1).

Recorded per run:

- trials and failed trials;
- hub X and Z failures, per winding bit;
- partner X and Z failure totals, and partner failures by index `j`;
- with frame snapshots, own faults per block and sector (partners by index);
- hub, partner, and snapshot cleanup failures;
- the resolved `tau_j` list and lifetimes, the stopping rule (`STOP_COUNT`),
  plus all parameters, under `### params ###`.

The file format stays `### data ###` / `### params ###`.

## 8. Metrics: per-gate error, suppression factor, threshold

The four references use different conventions:

| Reference | Reported quantity | Fixed prep/readout cost | Threshold method |
| --- | --- | --- | --- |
| 2408.01393 | Failure probability of one gadget: `d` rounds, tCNOT, `d` rounds, both blocks | Included | Finite-size crossing |
| 2403.03272 | Per-round rate from one 32-layer circuit, Markov inversion; fit `P_L ~ alpha^((d+1)/2)` | Made noiseless | Largest `p` at which `P_L` still falls with `d` |
| 2408.13687 (Google) | Per-cycle `eps_d`: slope of `ln(1-2p_L)` vs cycle count; `Lambda` from regressing `ln eps_d` on `(d+1)/2` | Absorbed by the intercept | `Lambda > 1` |
| Lake 2510.08056 | `p_log` = failure after `T = L` rounds | Included | Finite-size crossing at fixed `T/L` |

They differ in the unit (gadget, round, cycle, `d` rounds) and in whether the
fixed prep/readout cost is removed, so they are not interchangeable.

Propagation. Logical errors move through the CNOTs. With A as target, a
partner's pre-gate X logical error reaches A, and A's Z logical errors reach
every later partner. A block's final readout therefore contains other blocks'
faults: at `L = 13`, `p = 0.5%`, `k = 1`, `N = 13`, the partners' final Z
failures grow with the index from 29 to 339 per 5332 trials. Frame snapshots
at each CNOT (`docs/implementation.md` Section 4.6.5) give each block's own
faults, which remove that inheritance.

Metrics, all from A's final readout unless stated:

1. Threshold. With a fixed `k` and `N = c d`, the circuit has a fixed gate
   density and aspect ratio, so A's final failure probability
   `P_A(N = c d)` (either sector, `hub_failures`) is the Lake-style
   observable. Its finite-size crossing at each `c` is the threshold. `P_A`
   includes the partners' X errors copied onto A, i.e. it is A's logical
   error after the circuit. The per-sector crossings (X absorbs the partners'
   X history; Z is the source of the partners' inherited Z) show which sector
   limits A.
2. Per-gate rate, Google style. Fit
   `P_A(N) = P_max [1 - B (1 - eps_A / P_max)^N]` with `P_max = 15/16` (per
   sector `3/4`). `B` absorbs A's first and last `d` rounds. With only
   `N = L, 2L` the fit is exact, and it loses precision where `N = 2L`
   saturates. Once it confirms `B ≈ 1`, the headline per-gate rate is the
   single-point value at `N = L`.
3. Suppression factor: `Lambda_A` from regressing `ln eps_A` on `(L+1)/2`,
   with the pairwise ratios `eps_L / eps_(L+2)`. The crossing of `eps_A` and
   `Lambda_A = 1` lie above the threshold at finite `L`: since
   `P_A ≈ cL eps_A`, `P_A` is flat where `Lambda_A ≈ (L+2)/L`.
4. Each block. A partner has one gate, so its final failure probability `P_j`
   is a per-gate number; it grows with `j` because it carries A's Z errors
   from before gate `j`. Its own X and Z faults are the per-gate rate charged
   to that partner. A's final failures equal its own faults (final/own 1.003
   in X at `k = 1`).

Statistics: `STOP_COUNT=hub` makes `ACC_ERRORS` count A's failures (A has
about 0.7 of the failed trials).

Memory reference: `memory` at `T in {d, 2d, 4d}` gives the per-round rate and
Lake's `p_log(T=d)` (Stage 1, threshold 1.77%).

Steady state: the working hypothesis was that A's buffer reaches steady state
after about `ceil(Z/(k+1))` gates. At `k = 1`, `B` is 1 within errors over
`N = L, 2L`, and the partners' own Z faults level off after 4-5 gates.

## 9. Validation (`test/cnot_primitive_xz_runtests.jl`)

1. Kernel equivalence, bit-exact. Use an X-only flip-noise helper that
   consumes the RNG exactly like legacy `update!`: back-wall `rand()`, then
   `rand(L,L,2)` if `p>0`, then `rand(L,L)` if `q>0`. Then decode, noise, and
   measure must reproduce legacy `update!` arrays for `L in {3,5,9}` over 200
   rounds with the same seed. In Julia 1.12, `Random.seed!(s)` and
   `Xoshiro(s)` give the same stream.
2. Dual lattice:
   - one Z error on edge `e` flips exactly the two plaquettes sharing `e`;
   - Z on the star of a vertex gives no syndrome and no failure;
   - Z on every edge crossed by a dual cycle gives no syndrome and a Z-sector
     failure.
3. Samplers (at least 10^6 draws):
   - single-qubit: `P(x) = P(z) = 2p/3` and `P(x and z) = p/3`;
   - two-qubit: all 15 Paulis equally likely and never `II`; per sector, the
     (C only, T only, both) probabilities are each `4p2/15`.
4. Gate algebra and merge: the physical map matches Section 3.3 step 3. The X
   result of `merge_primitive!` equals legacy `primitive_cnot_x_sector!` on
   random arrays, and the Z result is its mirror.
5. Single injected faults (`L=5`, `LOGZ=false`, no background noise): a
   pre-gate data error on C or T in each sector, a measurement error in the
   last pre-gate round, and one two-qubit fault. Each must give the expected
   defects in the expected decoders, and cleanup must leave syndrome-free
   decoded states with no logical failure.
6. Zero noise: no failures for any schedule; failure-stopping is rejected at
   `p = q = 0`.
7. Schedules:
   - hub `tau_j` values and lifetimes (partners live `2d+1` rounds);
   - the largest live set;
   - at most one event per block per round;
   - hub `N=1` and consecutive `[d,d]` give identical results for the same
     seed.
8. Statistical checks:
   - reversing the gate direction swaps the X and Z failure statistics
     within error bars;
   - fixed-trial accounting gives the same totals with one thread and with
     several.

## 10. Cost

A hub trial costs about

~~~text
R = [2d + 1 + (N-1)(k+1)]  +  N (2d+1)  +  ~10 (N+1)     block-rounds, x 2 sectors
     hub A                    partners      cleanup
~~~

Partners dominate, at about `2d+1` rounds per gate, so with `N = c d` a trial
costs `O(c d^2)` block-rounds of `O(d^2)` work each. Measured in the `k = 1`
local scan at `L = 13` (frame tracking on): 0.23 s per trial and thread at
`N = L`, 0.46 s at `N = 2L`. 10^5 trials of both take about 1.2 h per `p` on
16 threads. At `L = 19`, scaling by the kernel time (2.4x) and `d^2` (2.1x)
gives about 6 h per `p` on 16 threads, less where capped stopping ends early.

## 11. Implementation steps

Stage 1:

1. Driver skeleton, kernel split, RNG plumbing; test 1.
2. Two sectors, dual map, samplers, CNOT round, mirrored merge; tests 2-5.
3. Engine, `memory` and `consecutive` schedules, estimator, environment
   parsing and output; tests 6-7.
4. Runs at `L in {5,7,9,11,13}`:
   - `memory` at `T in {d,2d,4d}`;
   - the `N=1` gadget with ablations (`P2_RATIO in {0,10}`,
     `GATE_MODE in {cnot,idle}`) at a few `p`;
   - a coarse memory threshold under the new noise model.
5. Documentation: a new `docs/implementation.md` section and a comparison
   table entry; refresh the `AGENTS.md` code map. Also fix the existing
   references to `2d_windowed_cnot_block.jl` in `implementation.md` and
   `test/runtests.jl`; that file is not in the working tree and has no git
   history.

Stage 2:

6. Hub schedule and the hub/partner counters, frame snapshots, and
   `STOP_COUNT`; tests 7-9.
7. `jobs/cnot_primitive_xz_hub_scan.sh` (Slurm) and
   `jobs/cnot_primitive_xz_hub_local.jl`: arrays over `(p, L, c)` at one `k`,
   writing to `results/cnot_primitive_xz/hub/k_<k>/N<c>L/`.
8. `notebooks/analysis_utils.py`: loader, target and per-block rates, decay
   fit, `Lambda` regression, reuse of the finite-size fit; the analysis
   notebook `notebooks/cnot_primitive_xz_hub_fixed_gap_analysis.ipynb`.
9. Scans:
   - done (local): `k = 1`, `N = L, 2L`, `L = 5-13`, `p = 0.3-0.9%`;
   - next, on the cluster: `k = 1` with `STOP_COUNT=hub`, more statistics,
     and `L = 15-19`, to test whether A's crossing (0.53%) moves toward its
     X sector's (0.52%);
   - then other gaps (`k = 0, 2, 3, 4`), each with a coarse pilot to locate
     its threshold before the `p` grid is set, and the idle control
     (`GATE_MODE=idle`) at `k = 1`, which separates the gate-noise budget from
     the merges.
10. Documentation: results in `docs/implementation.md` Section 5.2.9.

## 12. Limitations and open questions

- The model is phenomenological plus gate noise. Syndrome extraction is not
  simulated at circuit level: no hook errors, and gate noise never touches
  measurements.
- The two sector decoders are independent, so they ignore Y correlations and
  the cross-block correlations of two-qubit faults. This is like MWPM with Y
  errors decomposed. Correlated rules belong to later stages.
- Stage-1 result (`docs/implementation.md` Section 5.2.8): the single noisy
  gadget crosses near `p = 0.97%`, against `1.77%` for memory. The two-qubit
  channel accounts for most of the drop, and the primitive merge for about
  0.07%. The merge also doubles the failure probability of the absorbing
  sectors at fixed `p`. Smaller `k` stacks gate noise on A and is expected to
  push the crossing lower.
- Stage-2 result (`docs/implementation.md` Section 5.2.9), `k = 1`,
  `N = L, 2L`:
  - A's fixed-aspect threshold is 0.533(5)% at `N = L` and 0.532(5)% at
    `N = 2L` (0.53-0.54% over fit choices), 0.30 of the memory threshold.
  - A's X sector crosses lowest (0.51-0.52%), so the asymptotic threshold may
    be nearer 0.52%.
  - `Lambda_A` = 1.59, 1.34, 1.19 at `p` = 0.4, 0.5, 0.6%; `Lambda_A = 1`
    near 0.83%, above the threshold because of the per-gate normalization
    (Section 8 item 3).
- `Lambda` comes from `L = 5-13` only. Lake's gerrymandering theorem implies
  suppression slower than `(L+1)/2` asymptotically for this decoder.
- Next stage (Y-junction): with `k < Z`, a new gate reaches A before the
  previous junction collapses (`Z` rounds), so the hub needs overlapping
  junctions on one target. That is a design problem for that stage's plan.
- At `k = 1` A is stationary after a warm-up of 4-5 gates (Section 8,
  steady state). A longer circuit (`N = 4L`) is not needed for stationarity.
- Still open: `L >= 15`, the other gaps `k`, and the two-qubit noise ratio
  (`P2_RATIO = 10` makes gate noise 80% of A's data noise at `k = 1`).
