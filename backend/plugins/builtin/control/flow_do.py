"""Built-in node — flow.do (DO · Loop) — category: Control.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port

register(NodeSpec(id='flow.do', label='Loop container', category='Control', inputs=[], outputs=[Port(name='i', kind='scalar')], params=[Param(name='start', kind='int', default=1, min=-1000000, max=1000000, wireable=False), Param(name='end', kind='int', default=10, min=-1000000, max=1000000, wireable=True), Param(name='step', kind='int', default=1, min=-1000000, max=1000000, wireable=False)], doc='Iterate everything dropped inside the red frame. The loop index `i` is emitted from the right edge as a scalar and can be wired into any parameter of a child node. Semantics: `i = start, start+step, start+2*step, …, ≤ end`  (SPIDER DO convention, inclusive of `end`). After the loop, each child node holds its final-iteration outputs so you can inspect them the usual way.', fn=None, manual='DO'))
