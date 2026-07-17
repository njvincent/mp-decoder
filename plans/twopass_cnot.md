# Two-Pass Causal-Junction CNOT Decoder

Status: implemented by `2d_windowed_cnot_twopass.jl` for one synchronous,
ideal, X-sector CNOT between two periodic toric-code blocks.

## 1. Motivation and objective

The legacy sheet-copy decoder preserves pre-CNOT control and target histories,
but gives each hidden sheet an independent post-CNOT physical and measurement
channel. The snapshot decoder restores two observable physical channels but
places pre- and post-CNOT defects in mutually invisible histories. The primitive
decoder keeps cross-gate attraction but XOR-compresses the two input histories.

The two-pass decoder keeps exactly two physical blocks and preserves three
separate target-output syndrome streams across a causal CNOT junction. It uses
the inherited Lake local fields and deterministic feedback order while allowing
only physically admissible cross-stream temporal motion.

The implementation is a standalone derivative of the primitive CNOT lineage.
It owns only the required synchronous baseline memory-decoder helpers and does
not include the primitive, snapshot, or block driver.

## 2. Scope and invariants

The implemented physical map is the ideal X-sector rule

~~~text
control_out = control
target_out  = target xor control
~~~

The prototype supports exactly two aligned periodic blocks, one CNOT, and
`SYNCH=true`. It has no gate-fault channel, Z-sector propagation, repeated-gate
driver, or circuit-level syndrome extraction.

There are always two physical blocks. Each owns one error array, one accumulated
correction frame, and old/new measured-syndrome registers. After the gate only
these blocks receive `p` data noise and `q` measurement noise.

The target-output decoder has fixed stream indices:

~~~text
PRE_CONTROL = 1
PRE_TARGET  = 2
POST_TARGET = 3
~~~

Each stream owns an `L x L x Z` Boolean history and independent current/scratch
message fields. Labels denote observable history segments, not simulator-only
fault origins. Different streams never annihilate directly. A separate primitive
history receives labeled defects only when they reach the finite back wall;
ordinary ties never enter it in the bulk.

## 3. Gate rule

Immediately after the last pre-gate round:

~~~text
errors[target]              xor= errors[control]
correction_frame[target]    xor= correction_frame[control]
old_syndrome[target]        xor= old_syndrome[control]
new_syndrome[target]        xor= new_syndrome[control]
~~~

The live control history remains continuous. Its history and current fields are
deep-copied once into `PRE_CONTROL`; the old target history becomes
`PRE_TARGET`; and `POST_TARGET` starts empty. Scratch fields and proposals are
cleared. No noise, measurement, or syndrome event occurs at the gate.

The copy is stored past measurement evidence for target-output inference. It is
not a physical sheet and never receives an independent noise channel.

## 4. Post-gate target round

The target round preserves the inherited ordering:

1. Run `r` synchronous local field sweeps independently in every labeled stream
   and in the primitive retirement history. Update the two junction messages
   from the same pre-sweep state.
2. First pass: for every defect in a frozen history snapshot, select its minimum
   weighted direction. Zero means no candidate. Ties use
   `temporal, -x, -y, +y, +x`.
3. Second pass: seed a scratch field bank from the persistent pass-one messages,
   run `r` synchronous sweeps on the frozen current stream-aware graph, then
   route only the axis selected in pass one. This preserves Lake-like
   long-range propagation while recomputing before the move. Proposals are
   recorded without mutating histories. Junction branches are selected first;
   a directly selected pre-gate endpoint is skipped as an ordinary source.
4. Apply all labeled and primitive proposals synchronously. Every source defect
   proposes at most one adjacent edge. Spatial proposal parity is XORed into
   the one target frame;
   temporal proposals change only decoder histories.
5. XOR every labeled defect still on the back wall into the primitive
   retirement history, never copying it.
6. Sample one target data-noise mask and one target measurement-noise mask.
7. Cycle every labeled and primitive history once. Insert the observed target
   syndrome change only into `POST_TARGET`.
8. Advance the moving CNOT-junction depth, capped at the last bulk slice.

The control uses the ordinary baseline `update!` throughout. Thus post-CNOT
control noise is never supplied to target inference.

### 4.1 Direction cost and priority

Let

~~~text
w_p = log((1-p)/p)
w_q = log((1-q)/q)
~~~

with infinite weight at probability zero. The decoder stores the nominal
experiment weights, so noiseless cleanup does not freeze existing defects.
Inherited integer message distance is multiplied by `w_p` for spatial
directions and `w_q` for the temporal direction. This is a directional
anisotropic weighting of the inherited field, not a new edge-by-edge weighted
message kernel.

No candidate means no proposal; the defect ages normally. Equal positive costs
move immediately according to the fixed priority above. Labeled back-wall
motion is deterministic. After retirement, the primitive history retains the
inherited raw-distance priority and `0.8` stochastic back-wall escape rule.

### 4.2 Stream-aware spacetime graph

Spatial and temporal motion normally stays within one stream. The only
cross-stream edges are at the moving CNOT junction:

~~~text
POST_TARGET -> PRE_CONTROL
POST_TARGET -> PRE_TARGET
~~~

The post-gate defect is always the newer endpoint and moves toward the selected
pre-gate stream. `PRE_CONTROL <-> PRE_TARGET` is forbidden. Equal junction
branch costs select `PRE_CONTROL` before `PRE_TARGET`. After crossing, the
defect belongs to exactly that selected stream. All fields are recomputed before
the next one-edge move.

### 4.3 Back-wall retirement

After labeled proposals are applied, every labeled defect remaining at `k=Z`
is removed from its stream and XORed into the primitive retirement history at
the same coordinate. Valid same-stream or capped-junction matches can therefore
resolve before retirement. This finite-buffer rule is not a response to a
directional tie. The primitive history uses the inherited unrestricted decoder
feedback and receives no direct physical syndrome stream.

## 5. Readout and failure criterion

Readout is

~~~text
decoded_control = errors[control] xor correction_frame[control]
decoded_target  = errors[target]  xor correction_frame[target]
~~~

Cleanup runs the same decoder with physical `p=q=0` while retaining the nominal
direction weights. Cleanup succeeds when the continuous control history, all
three target streams, and the primitive retirement history are empty. Cleanup
failure remains separate from logical failure. A trial fails logically if the
control or target decoded state has nontrivial torus winding, using the inherited
`detect_logical_error` convention in which `true` means logical success.

## 6. Cost and scaling

The physical state remains two blocks. The control, three persistent target
streams, primitive retirement history, and three-stream second-pass scratch
bank use `104 L^2 Z` leading integer words including both junction buffers.
Memory and post-gate work remain `Theta(L^2 Z)` and `Theta(r L^2 Z)` with fixed
constants independent of elapsed time. Repeated gates are not implemented, so
no claim is made about the construction for general circuits.

## 7. Validation and performance hypothesis

`test/twopass_runtests.jl` checks gate algebra and non-aliasing, deterministic
standalone loading without snapshot symbols, direction priority, stored cleanup
weights, no-candidate aging, same-stream routing, genuine second-pass
recomputation, globally synchronous junction messages, real cross-gate
measurement-defect closure, allowed and forbidden junction paths, branch
priority, one-edge motion, physical-channel ownership, back-wall retirement,
threaded fixed-sample accounting, rejection of unsupported modes, and small
zero/noisy end-to-end runs.

The minimum performance study uses paired noise masks against the primitive
decoder for `p=0`, `q=0`, and `p=q`, followed by matched scans at
`L=5,7,9,13,19`, `T=L`, the same `Z`, `r`, CNOT timing, and cleanup. The working
hypothesis is primitive-like measurement performance with improved storage-error
attribution. No threshold or noninferiority claim is established until those
scans are run. Sheet-copy results remain an algorithmic oracle rather than a
physical-model benchmark.

## 8. Known limitations

- The inherited fields contain unweighted integer 1-norm messages; applying
  `w_p` or `w_q` at direction selection is not exact anisotropic belief
  propagation.
- Stream labels are causal-segment hypotheses. Equal junction explanations are
  resolved by a fixed algorithmic priority rather than inferred physical origin.
- Primitive retirement makes the finite back wall a deliberate information-
  compression boundary.
- Only the X sector, one ideal CNOT, synchronous updates, and two blocks are
  implemented.
- Threshold, runtime constants, and repeated-CNOT behavior remain unmeasured.
