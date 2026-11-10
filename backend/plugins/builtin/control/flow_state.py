"""Built-in node — flow.state (Loop state) — category: Control.

Persistent state variable for use inside a DO Loop. Solves the "how do
I feed iteration k's output back as iteration k+1's input?" problem —
the standard DAG has no cycles, so a normal wire can't loop back. This
node lets you break that cycle: wire `initial` from OUTSIDE the DO Loop
for iteration 1, and wire `value_in` from an INSIDE-DO node's output
for the feedback. Iteration k+1 reads that captured value from
`value_out`.

The engine handles this specially in `_stream_do_loop`:
  * The `value_in` edge is filtered out of the intra-DO topo sort so
    the DAG stays acyclic within one iteration.
  * `value_out` is populated *before* children run each iteration.
  * `value_in` is captured *after* children run each iteration, ready
    for the next round.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port

register(NodeSpec(
    id='flow.state',
    label='Loop state',
    category='Control',
    inputs=[
        Port(name='initial',  kind='image'),           # required, from outside DO
        Port(name='value_in', kind='image', optional=True),  # feedback, inside DO
    ],
    outputs=[Port(name='value_out', kind='image')],
    params=[],
    doc=(
        'Iterative state variable for use inside a DO Loop.\n\n'
        'Iteration 1: value_out = the value of the `initial` upstream '
        '(usually wired from outside the DO container).\n\n'
        'Iteration k > 1: value_out = whatever value_in was capturing at '
        'the end of iteration k − 1.\n\n'
        'Wire a node inside the DO Loop (e.g. the current-iteration '
        'aligned average) into value_in to feed it forward.'
    ),
    fn=None,   # engine special-cases this in `_stream_do_loop`
    manual=None,
))
