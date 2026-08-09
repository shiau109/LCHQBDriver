"""Silence one benign Windows asyncio shutdown message coming out of qblox_instruments.

Every ``scqo run`` on Windows ends with a traceback printed AFTER the results and after
``saved: <path>``::

    Error on reading from the event loop self pipe
    loop: <ProactorEventLoop running=True closed=False debug=False>
      ...
    OSError: [WinError 87]

Nothing is lost — the dataset is already on disk — but it reads like a crash.

Cause: ``qblox_instruments/ieee488_2/transport.py`` builds a bare
``asyncio.new_event_loop()`` per transport whenever no loop is already running (a
``ProactorEventLoop`` here) and never closes it. At interpreter shutdown the loop's
self-pipe socket is torn down while ``_loop_self_reading`` still has a ``recv`` pending, and
asyncio reports that through ``call_exception_handler`` rather than raising. Slot
connections are created with ``loop_from=self._transport``, so there is exactly ONE loop per
Cluster to deal with.

Why an exception handler rather than closing the loop: closing it at ``atexit`` would
genuinely prevent the message (``ProactorEventLoop.close()`` cancels the self-pipe future),
but ``atexit`` runs LIFO and qcodes registers its own instrument-closing hooks — closing the
transport's loop first would break those. A handler has no ordering hazard and cannot touch
data on its way out.

The suppression is deliberately narrow: that one message, an ``OSError``, and winerror 87.
A self-pipe that fails any other way still surfaces.

Stdlib only, so this stays importable without ``qblox_scheduler``.
"""

from __future__ import annotations

from typing import Any

#: asyncio/proactor_events.py ``_loop_self_reading`` uses exactly this text.
_SELF_PIPE_MESSAGE = "Error on reading from the event loop self pipe"

#: ERROR_INVALID_PARAMETER, out of ``CreateIoCompletionPort`` on an already-dead handle.
_WINERROR_INVALID_PARAMETER = 87

#: Set on a loop we have already wrapped. ``acquire()`` runs once per experiment, so
#: without this the handlers would nest one deeper every run.
_MARKER = "_scqo_qblox_self_pipe_handler"


def _install(loop: Any) -> bool:
    """Wrap *loop*'s exception handler. Returns True if this call installed it."""
    if getattr(loop, _MARKER, False):
        return False
    previous = loop.get_exception_handler()

    def handler(
        loop_: Any,
        context: dict,
        _previous: Any = previous,
        # Bound as defaults, not looked up: this runs during interpreter finalization,
        # where this module's globals (and eventually __builtins__) are torn down. A
        # handler that raises gets "Unhandled error in exception handler" logged on top
        # of the original — i.e. MORE noise, the opposite of the point.
        _oserror: Any = OSError,
        _isinstance: Any = isinstance,
        _getattr: Any = getattr,
        _message: str = _SELF_PIPE_MESSAGE,
        _winerror: int = _WINERROR_INVALID_PARAMETER,
    ) -> None:
        exception = context.get("exception")
        if (
            context.get("message") == _message
            and _isinstance(exception, _oserror)
            and _getattr(exception, "winerror", None) == _winerror
        ):
            return
        if _previous is None:
            loop_.default_exception_handler(context)
        else:
            _previous(loop_, context)

    loop.set_exception_handler(handler)
    setattr(loop, _MARKER, True)
    return True


def silence_proactor_self_pipe_noise(hw_agent: Any) -> int:
    """Install the handler on every connected cluster's transport loop.

    Idempotent, and never raises: this is cosmetic, so a shape change in
    ``qblox_scheduler``/``qblox_instruments`` must degrade to "the noise comes back",
    never to a failed measurement. Returns how many loops were newly wrapped (for tests).
    """
    installed = 0
    try:
        clusters = getattr(hw_agent, "_clusters", None) or {}
        for component in clusters.values():
            try:
                loop = component.instrument._transport._loop
            except Exception:  # noqa: BLE001 - a component without a live transport
                continue
            if loop is not None and _install(loop):
                installed += 1
    except Exception:  # noqa: BLE001 - see docstring
        pass
    return installed
