# Physical Snapshot CNOT Decoder

Status: implemented in `2d_windowed_cnot_snapshot.jl` for one synchronous,
ideal, X-sector CNOT between two toric-code blocks.

## 1. Problem and design goal

The legacy sheet-copy model keeps separate control and target histories, but it
also gives every copied sheet its own later physical noise and syndrome
measurements. After a physical CNOT there are only two observable outputs: the
control and the new target. The new target, not its hidden control and target
components, must receive the post-gate noise and measurement channel.

The snapshot design therefore separates:

- physical blocks, which own errors and measured syndromes; and
- decoder histories, which own buffered syndrome-change events, message
  fields, and recovery contributions.

This follows the DKLP spacetime-history rule that pre-gate control history can
contribute to both CNOT outputs, while using Lake's rolling buffer separately
for each stored history. The papers do not prove a threshold for this split
history rule; its performance must be measured.

## 2. State and invariants

### 2.1 Physical blocks

There are always two `PhysicalBlock` objects in this prototype. Each contains:

~~~text
errors               L x L x 2 physical X-error array
old_synds             L x L previous measured syndrome
new_synds             L x L latest measured syndrome
saved_correction      L x L x 2 correction from deleted histories
noise_rounds          physical-channel update counter
measurement_rounds    measurement-channel update counter
~~~

Only a `PhysicalBlock` may receive `p` noise, `q` measurement noise, or a call
to `get_synds` during the snapshot experiment.

### 2.2 Decoder histories

Each `DecoderHistory` contains:

~~~text
history_id
live_block            block id for a live history; nothing for an old history
applies_to            BitVector over physical output blocks
hist                  L x L x Z buffered syndrome-change events
fields, new_fields    integer message arrays
hist_correction       proposed local recovery links
correction            L x L x 2 accumulated spatial recovery
~~~

A live history receives syndrome-change events from exactly one physical
block. An old history receives no new event. `applies_to[b]` says whether that
history's correction is XORed into output block `b`.

Before the CNOT there are two live histories:

~~~text
control history: applies_to = [1,0]
target history:  applies_to = [0,1]
~~~

## 3. Round and gate rules

### 3.1 Live-history round

A live history uses the inherited synchronous baseline `update!`:

1. propagate fields and choose feedback;
2. update its correction and buffered history;
3. update its one physical block with one `p` channel;
4. measure that block with one `q` channel;
5. cycle the buffer and insert `old_synds xor new_synds`.

The driver calls this exactly once per physical block per round.

### 3.2 Old-history round

`update_history_only!` performs:

1. field propagation and feedback;
2. correction accumulation and history correction;
3. buffer/field cycling;
4. insertion of an empty front slice.

It has no physical-state argument, does not sample `p` or `q`, and does not
calculate a syndrome. It only finishes decoding data observed before the gate.

### 3.3 Ideal X-sector CNOT

For control `c` and target `t`, after the final pre-gate round:

~~~text
errors[t]           xor= errors[c]
saved_correction[t] xor= saved_correction[c]
~~~

Every existing history is routed through the same binary CNOT map:

~~~text
history.applies_to[t] xor= history.applies_to[c]
~~~

The current control and target histories become old histories. Two new empty
live histories are created:

~~~text
new control history: applies_to = [1,0]
new target history:  applies_to = [0,1]
~~~

No decoder history is deep-copied. The target syndrome baseline is

~~~text
measured_target_at_gate =
    last_control_measurement xor last_target_measurement
~~~

The control baseline remains its last control measurement. No physical noise
or extra measurement occurs at the gate. The derived target baseline has
pre-gate measurement-error probability `2q(1-q)` when the two input
measurement errors are independent.

## 4. Deletion, cleanup, and readout

After each round, an old history with empty `hist` is deleted. Before deletion,
its `correction` is XORed into `saved_correction` for every block selected by
`applies_to`. Message fields can then be freed because an empty old history
will never receive another defect.

Cleanup runs live histories with `p=q=0` and old histories with the
history-only rule. Cleanup succeeds when every remaining history is empty.
Cleanup failure remains separate from logical failure.

Readout for block `b` is

~~~text
decoded[b] = errors[b] xor saved_correction[b]
for each remaining history h:
    if h.applies_to[b]:
        decoded[b] xor= h.correction
~~~

For the single CNOT this gives

~~~text
decoded control = physical control
                  xor pre-gate control correction
                  xor post-gate control correction

decoded target  = physical target
                  xor pre-gate control correction
                  xor pre-gate target correction
                  xor post-gate target correction
~~~

## 5. Cost and future repeated gates

The single-CNOT prototype has two physical blocks and at most four decoder
histories immediately after the gate. A history has the same dominant
`12NZ` integer-field storage as a baseline decoder history. Old histories are
removed as soon as they become empty, so post-gate work falls back toward two
live decoder updates.

`applies_to` is included for future repeated gates. The same routing rule
handles repeated direction, alternating direction, and fanout at the level of
X-sector correction ownership. A future repeated-gate driver must open fresh
live histories at every gate and apply the physical/saved-correction CNOT map.
The present driver rejects a second gate. History count would initially grow
linearly with unresolved recent gates, not by deep-copying a lineage tree, but
high gate rates could still leave many active histories.

## 6. Validation and current limits

Run:

~~~bash
julia --startup-file=no test/snapshot_runtests.jl
MODE=CNOT_DEBUG LVAL=3 LOGZ=false TVAL=2 CLEANUP_TIME=4 \
    SYNCH=true TRIAL_PARALLEL=false \
    julia --startup-file=no 2d_windowed_cnot_snapshot.jl
~~~

The tests cover physical CNOT algebra, syndrome baselines, `applies_to`, one
physical update per block, old-history isolation, deletion/readout invariance,
zero-noise success, and small noisy trials.

Current limits:

- one CNOT and exactly two blocks;
- X sector only;
- ideal gate with no gate-fault channel;
- synchronous updates only;
- periodic aligned lattices and the inherited buffer/message rules;
- no snapshot visualization yet;
- no threshold evidence yet;
- old and new histories cannot match defects with one another, so the target
  threshold may be lower than the legacy sheet-copy result.
