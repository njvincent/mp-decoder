# Moving Y-Junction CNOT Decoder

## 1. Motivation and status

The primitive CNOT immediately XORs the complete pre-CNOT control evidence into
the target and componentwise-minimizes their messages. This has constant
gate-count overhead, but it removes the distinction between the two pre-CNOT
causal lanes everywhere in the buffer. Literal sheet-copy retains that
distinction but creates a full independently updated target sheet for every
propagated lineage.

The moving Y-junction decoder retains two independent pre-CNOT lanes only on
the older side of the CNOT event and uses one observable post-CNOT target lane
on the newer side. The interface advances with the Lake buffer. At the finite
back wall, unresolved pre-CNOT defects are deliberately XOR-merged and their
messages are reduced by nonzero minimum. This is an implemented heuristic,
not an exact compression of sheet-copy or a result proved by Lake or DKLP.

The implementation is `2d_windowed_cnot_yjunction.jl`.  It supports one ideal
synchronous X-sector CNOT between two toric-code blocks.  Its focused tests are
in `test/yjunction_runtests.jl`.

## 2. State and invariants

Each observable block owns one physical state:

~~~text
PhysicalBlockState:
    block_id, errors, correction_frame,
    previous_syndrome, current_syndrome,
    data_noise_rounds, measurement_noise_rounds
~~~

A decoder lane contains only evidence and message-passing state:

~~~text
DecoderLane:
    defects, messages, next_messages, correction_links

YJunctionState:
    physical_blocks, control_decoder, target_decoder
~~~

Before the gate, control and target each have one ordinary lane.  During the
transition the target owns:

~~~text
TargetJunctionDecoder:
    pre_cnot_control_lane
    pre_cnot_target_lane
    post_cnot_target_lane
    interface_depth
    branch_crossing_costs
    branch_choices
~~~

Implementation functions use role-first families. Ordinary-lane stages contain
`decoder_lane`; active-CNOT stages contain `target_junction`. Within either
family, `compute` is pure, `update` advances a stage, `select` fills frozen
choices, and `commit` applies those choices. Lifecycle and query functions use
`age`/`clear`/`collapse` and `is`/`has`/`count`, respectively.

At `interface_depth = g`, `POST_CNOT_TARGET_LANE_ID` owns slices `k <= g`;
both pre-CNOT lanes own independent copies of slices `k > g`. Every unowned
defect and message slice is zero. The interface depth is zero immediately
after the CNOT and advances once per physical or cleanup round.

The principal invariants are:

- there is one physical and measurement channel per observable block;
- every defect bit belongs to exactly one target lane;
- messages may fan out across the Y graph, but defects, errors, noise, and
  corrections are never copied;
- all target spatial correction links update the same target correction frame;
- the two pre-CNOT lanes do not directly XOR before the back-wall collapse;
- after collapse, the target state owns one ordinary `DecoderLane` and no
  references to the two pre-CNOT lanes.

## 3. Gate rule

At the ideal control-to-target X-sector CNOT:

~~~text
target.errors              xor= control.errors
target.correction_frame    xor= control.correction_frame
target.previous_syndrome   xor= control.previous_syndrome
target.current_syndrome    xor= control.current_syndrome
~~~

The existing target lane becomes `pre_cnot_target_lane` without copying. A
non-aliased snapshot of `control_decoder.defects` and
`control_decoder.messages` becomes `pre_cnot_control_lane`; its next-message
buffer and correction links start empty. A fresh empty lane becomes
`post_cnot_target_lane`. The continuous `control_decoder` is unchanged. No
noise, measurement, correction, or defect insertion occurs at the CNOT.

Corrections already selected before the gate are present in the propagated
target correction frame. Corrections selected later while resolving the copied
control evidence act on the target correction frame only; `control_decoder`
chooses its own later recovery.

## 4. Y-graph message and correction-link update

Each of the `r` message sweeps is globally Jacobi.  Every valid destination is
computed from the same frozen three-lane message state before any lane is
committed.  The inherited `3 x 3` candidate plane, 1-norm distances, and zero
sentinel are unchanged.

For a destination in `POST_CNOT_TARGET_LANE_ID`, an update cone that crosses
from `k=g` to `k=g+1` evaluates candidates in both pre-CNOT lanes and stores
their smallest positive value. For a destination in either pre-CNOT lane, a
cone crossing from `k=g+1` to `k=g` evaluates `post_cnot_target_lane`. This
rule applies to all six message components. Consequently post-CNOT messages
advertise into both branches, while branch messages reduce to one unlabeled
value in the post-CNOT lane. Message direction is retained; propagation does
not reverse a `+a` message into a `-a` message.

For the post-side temporal component at the junction, the implementation also
stores the two branch-specific costs computed from the same frozen message
state.
These costs exist only to route a temporal defect crossing; below the junction
the message stores no source label.

Correction links are selected from frozen defects with the primitive bulk
priority:

~~~text
+buffer, -x, -y, +y, +x
~~~

A post-CNOT defect below the junction moves within its lane. If a post-CNOT
defect at `k=g` selects the buffer direction, it crosses to exactly one
pre-CNOT endpoint: the smaller positive branch cost wins, with control first
on an equal positive tie. Pre-CNOT defects retain ordinary one-way aging
toward larger `k`; they never move backward into `post_cnot_target_lane`. All
same-lane correction links and junction branch choices are committed
atomically. The XOR parity of every selected spatial edge is applied to the
one target correction frame. The inherited back wall remains spatial-only and
accepts a selected move with probability `0.8`.

After feedback, exactly one target data mask and one target measurement mask
are applied.  The resulting observed syndrome change is inserted only into
`post_cnot_target_lane` after all three lanes are cycled.

## 5. Aging and back-wall collapse

Each round shifts all three lanes' defects and messages toward larger `k`, then
increments `g`.  Invalid slices are cleared.  When `g` reaches `Z`, all
remaining pre-CNOT defects are on the back wall and the target is collapsed:

~~~text
post_cnot_target_lane.defects[:,:,Z] xor=
    pre_cnot_control_lane.defects[:,:,Z] xor
    pre_cnot_target_lane.defects[:,:,Z]

post_cnot_target_lane.messages[:,:,Z,spatial,:] = nonzero_minimum(
    post_cnot_target_lane,
    pre_cnot_control_lane,
    pre_cnot_target_lane,
)
~~~

The post-CNOT back-wall temporal components are cleared. Scratch buffers and
correction links are cleared, `target_decoder` is replaced by
`post_cnot_target_lane`, and the transition object becomes unreachable. Later
rounds call the ordinary baseline update.

This collapse is the only destructive provenance merge.  Coincident defects
cancel by GF(2) parity; surviving defects subsequently attract according to one
unlabeled message. The approximation can lose a useful pre-CNOT-control versus
pre-CNOT-target pairing preference for residual clusters that survive the entire
buffer.

## 6. Cleanup, readout, and cost

Cleanup runs the same physical schedule with `p=q=0` until all decoder defects
are empty and the target has collapsed, subject to the configured cap. A
nonempty decoder and an uncollapsed junction are reported separately. Logical
readout is local:

~~~text
decoded_control = control.errors xor control.correction_frame
decoded_target  = target.errors  xor target.correction_frame
~~~

Let `N=L^2`. Before the gate and after collapse there are two full
message-buffer pairs, or `24NZ` machine integers. During the transition there
are four pairs:
continuous control, copied pre-control, moved pre-target, and post-target, for
`48NZ` machine integers. The transition lasts exactly `Z` post-CNOT or cleanup
rounds.

At depth `g`, the message kernel visits `Z + 2(Z-g) + g = 3Z-g` slice-equivalents
per spatial site.  After collapse it returns to two ordinary block updates.
The gate copies `Theta(NZ)` control defects/messages and allocates an empty
post-CNOT lane. Collapse touches `Theta(N)` back-wall entries and releases the two
pre-CNOT lanes.

## 7. Validation and limits

The focused suite checks primitive pre-CNOT parity under explicit physical
masks, paired physical-error and syndrome parity through the gate, gate
algebra, ownership, all-component cone crossing, zero-aware minima,
bidirectional messages, global Jacobi propagation, reversed lane/site-order
equivalence, single-owned branch crossings, control-first ties, shared-frame
parity, atomic defect updates, one physical channel per block, collapse
parity, message merge, lane release, post-collapse ordinary updates, exact
threaded fixed-sample accounting, zero-noise success, and small noisy samples.

Performance must be evaluated at matched `q=p`, `r=3`, logarithmic `Z`, `T=L`,
split gate timing, and `2T` cleanup.  The primary comparison is target failure
against primitive.  Baseline is the desired threshold reference; sheet-copy is
only an algorithmic oracle because it supplies independent post-CNOT noise to
hidden sheets.  The hypothesis is rejected if target failures remain
primitive-like, cleanup failures materially increase, or the target does not
return to one lane after one buffer depth.

The current implementation has no repeated CNOTs, asynchronous path,
visualization, Z sector, CNOT gate faults, circuit-level syndrome extraction,
or threshold result.
