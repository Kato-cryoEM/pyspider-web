"""Progress + log reporting from node functions to the streaming engine.

Long-running nodes (SGD, BP CG, CTFFIND, iterative alignment, …) call
``report_progress(fraction, log)`` freely; the engine wires those calls to
the SSE stream so the UI can render a live progress bar and a log tail
per node.

Design: a ``ContextVar`` holds a per-thread emit callback. The engine
sets it before invoking ``spec.fn(**kwargs)`` in a worker thread and
drains the queue back to the caller as ``node_progress`` events.

A node function that doesn't care about progress simply doesn't call
``report_progress`` — the ContextVar defaults to a no-op emitter.
"""

from __future__ import annotations

import contextvars
from typing import Callable, Optional

Emitter = Callable[[Optional[float], Optional[str]], None]


def _noop(_progress: Optional[float] = None, _log: Optional[str] = None) -> None:
    return None


_progress_ctx: contextvars.ContextVar[Emitter] = contextvars.ContextVar(
    "pyspider_progress_emitter", default=_noop,
)


def report_progress(progress: Optional[float] = None,
                    log: Optional[str] = None) -> None:
    """Emit a progress event from inside a node's ``_fn_*`` implementation.

    - ``progress`` (0.0 – 1.0) — fractional completion; ``None`` keeps the
      current bar (log-only update).
    - ``log`` — a short line to append to the node's live log panel.

    Both arguments are optional; passing ``report_progress()`` is a no-op.
    """
    if progress is None and (log is None or not log):
        return
    _progress_ctx.get()(progress, log)


def set_emitter(fn: Emitter) -> contextvars.Token:
    """Install a progress emitter for the current context. Returns a token
    the engine uses to restore the previous emitter on teardown."""
    return _progress_ctx.set(fn)


def reset_emitter(token: contextvars.Token) -> None:
    _progress_ctx.reset(token)
