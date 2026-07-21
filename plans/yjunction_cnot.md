# Moving Y-Junction CNOT Decoder

## 1. Motivation and status

The primitive CNOT immediately XORs the complete pre-gate control history into
the target and componentwise-minimizes their fields.  This has constant
gate-count overhead, but it removes the distinction between the two pre-gate
causal histories everywhere in the buffer.  Literal sheet-copy retains that
distinction but creates a full independently updated target sheet for every
propagated lineage.

The moving Y-junction decoder retains two independent pre-gate histories only
on the older side of the CNOT event and uses one observable post-gate target
history on the newer side.  The interface advances with the Lake buffer.  At
the finite back wall, unresolved pre-gate evidence is deliberately XOR-merged
and its field is reduced by nonzero minimum.  This is an implemented heuristic,
not an exact compression of sheet-copy or a result proved by Lake or DKLP.

The implementation is `2d_windowed_cnot_yjunction.jl`.  It supports one ideal
synchronous X-sector CNOT between two toric-code blocks.  Its focused tests are
in `test/yjunction_runtests.jl`.

## 2. State and invariants

Each observable block owns one physical state:

~~~text
YPhysicalBlock:
    errors, frame, old_synds, new_synds,
    noise_rounds, measurement_rounds
~~~

A decoder lane contains only evidence and message-passing state:

~~~text
DecoderLane:
    hist, fields, new_fields, proposals
~~~

Before the gate, control and target each have one ordinary lane.  During the
transition the target owns:

~~~text
TargetYJunction:
    pre_control
    pre_target
    post_target
    junction_depth
    branch_temporal_costs
    junction_proposals
~~~

At junction depth `g`, `POST_TARGET` owns slices `k <= g`; both pre-gate lanes
own independent copies of slices `k > g`.  Every invalid history and field
slice is zero.  `g=0` immediately after the gate, so the post lane is initially
empty.  The depth advances once per physical or cleanup round.

The principal invariants are:

- there is one physical and measurement channel per observable block;
- every defect bit belongs to exactly one target lane;
- messages may fan out across the Y graph, but defects, errors, noise, and
  corrections are never copied;
- all target spatial proposals update the same target frame;
- the two pre-gate histories do not directly XOR before the back-wall collapse;
- after collapse, the target state owns one ordinary `DecoderLane` and no
  references to the two pre-gate lanes.

## 3. Gate rule

At the ideal control-to-target X-sector CNOT:

~~~text
target.errors     xor= control.errors
target.frame      xor= control.frame
target.old_synds  xor= control.old_synds
target.new_synds  xor= control.new_synds
~~~

The existing target lane becomes `pre_target` without copying.  A non-aliased
snapshot of the control lane's `hist` and current `fields` becomes
`pre_control`; its scratch and proposal arrays start empty.  A fresh empty lane
becomes `post_target`.  The continuous control decoder is unchanged.  No noise,
measurement, correction, or history insertion occurs at the gate.

Corrections already selected before the gate are present in the propagated
target frame.  Corrections selected later while resolving the copied control
evidence act on the target frame only; the continuous control decoder chooses
its own later recovery.

## 4. Y-graph field and feedback update

Each of the `r` message sweeps is globally Jacobi.  Every valid destination is
computed from the same frozen three-lane field state before any lane is
committed.  The inherited `3 x 3` candidate plane, 1-norm distances, and zero
sentinel are unchanged.

For a destination in `POST_TARGET`, an update cone that crosses from `k=g` to
`k=g+1` evaluates candidates in both pre-gate lanes and stores their smallest
positive value.  For a destination in either pre-gate lane, a cone crossing
from `k=g+1` to `k=g` evaluates the single post lane.  This rule applies to all
six message components.  Consequently post messages advertise into both
branches, while branch messages reduce to one unlabeled value in the post
lane.  Message direction is retained; propagation does not reverse a `+a`
message into a `-a` message.

For the post-side temporal component at the junction, the implementation also
stores the two branch-specific costs computed from the same frozen field state.
These costs exist only to route a temporal defect crossing; below the junction
the field stores no source label.

Feedback is selected from a frozen history with the primitive bulk priority:

~~~text
+buffer, -x, -y, +y, +x
~~~

A post defect below the junction moves within the post lane.  If a post defect
at `k=g` selects the buffer direction, it crosses to exactly one pre-gate
endpoint: the smaller positive branch cost wins, with control first on an equal
positive tie.  Pre-gate defects retain ordinary one-way aging toward larger
`k`; they never move backward into the post lane.  All same-lane and junction
proposals are committed atomically.  The XOR parity of every selected spatial
edge is applied to the one target frame.  The inherited back wall remains
spatial-only and accepts a selected move with probability `0.8`.

After feedback, exactly one target data mask and one target measurement mask
are applied.  The resulting observed syndrome change is inserted only into
`POST_TARGET` after all three lanes are cycled.

## 5. Aging and back-wall collapse

Each round shifts all three lane histories and fields toward larger `k`, then
increments `g`.  Invalid slices are cleared.  When `g` reaches `Z`, all
remaining pre-gate history is on the back wall and the target is collapsed:

~~~text
post.hist[:,:,Z] xor=
    pre_control.hist[:,:,Z] xor pre_target.hist[:,:,Z]

post.fields[:,:,Z,spatial,:] =
    nonzeromin(post, pre_control, pre_target)
~~~

The post back-wall temporal components are cleared.  Scratch and proposals are
cleared, the target decoder is replaced by the post lane, and the transition
object becomes unreachable.  Later rounds call the ordinary baseline update.

This collapse is the only destructive provenance merge.  Coincident defects
cancel by GF(2) parity; surviving defects subsequently attract according to one
unlabeled field.  The approximation can lose a useful pre-control versus
pre-target pairing preference for residual clusters that survive the entire
buffer.

## 6. Cleanup, readout, and cost

Cleanup runs the same physical schedule with `p=q=0` until all histories are
empty and the target has collapsed, subject to the configured cap.  A nonempty
history and an uncollapsed junction are reported separately.  Logical readout
is local:

~~~text
decoded_control = control.errors xor control.frame
decoded_target  = target.errors  xor target.frame
~~~

Let `N=L^2`.  Before the gate and after collapse there are two full field pairs,
or `24NZ` machine integers.  During the transition there are four pairs:
continuous control, copied pre-control, moved pre-target, and post-target, for
`48NZ` machine integers.  The transition lasts exactly `Z` post-gate or cleanup
rounds.

At depth `g`, the field kernel visits `Z + 2(Z-g) + g = 3Z-g` slice-equivalents
per spatial site.  After collapse it returns to two ordinary block updates.
The gate copies `Theta(NZ)` control history/field state and allocates an empty
post lane.  Collapse touches `Theta(N)` back-wall entries and releases the two
pre-gate lanes.

## 7. Validation and limits

The focused suite checks primitive pre-gate parity under explicit physical
masks, paired physical-error and syndrome parity through the gate, gate
algebra, ownership, all-component cone crossing, zero-aware minima,
bidirectional messages, global Jacobi propagation, reversed lane/site-order
equivalence, single-owned branch crossings, control-first ties, shared-frame
parity, atomic history updates, one physical channel per block, collapse
parity, field merge, lane release, post-collapse ordinary updates, exact
threaded fixed-sample accounting, zero-noise success, and small noisy samples.

Performance must be evaluated at matched `q=p`, `r=3`, logarithmic `Z`, `T=L`,
split gate timing, and `2T` cleanup.  The primary comparison is target failure
against primitive.  Baseline is the desired threshold reference; sheet-copy is
only an algorithmic oracle because it supplies independent post-gate noise to
hidden sheets.  The hypothesis is rejected if target failures remain
primitive-like, cleanup failures materially increase, or the target does not
return to one lane after one buffer depth.

The current implementation has no repeated CNOTs, asynchronous path,
visualization, Z sector, CNOT gate faults, circuit-level syndrome extraction,
or threshold result.
