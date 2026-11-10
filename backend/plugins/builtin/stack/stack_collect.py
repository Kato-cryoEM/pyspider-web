"""Built-in node — stack.collect (Collect frames inside DO) — category: Stack.

Accumulates one image per DO iteration into a stack. The optional `index`
wireable param lets the student write directly: `output_stack[index] = image`
by wiring `flow.do.i` in, which reads like an imperative for-loop.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port

register(NodeSpec(
    id='stack.collect',
    label='Collect frames (inside loop)',
    category='Stack',
    inputs=[Port(name='image', kind='image')],
    outputs=[Port(name='stack', kind='stack')],
    params=[
        Param(name='index', kind='int', default=-1, wireable=True,
              placeholder='optional: wire flow.do.i for stack[i] = image'),
    ],
    doc=(
        'Accumulator for a DO loop. Put this inside a DO container and wire '
        'an image in — each iteration appends that image onto an internal '
        'stack. The optional `index` input lets you write to specific slots: '
        'wire `flow.do.i` into it and the behaviour becomes '
        '`output_stack[i] = image`, with gaps filled by zeros if some '
        'iterations are skipped. Leave `index` unwired for simple append-order '
        'collection. Outside a DO container it passes the single image '
        'through as a length-1 stack.'
    ),
    fn=None,
    manual=None,
))
