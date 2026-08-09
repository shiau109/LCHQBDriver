"""``scqo_qblox/backend/_asyncio_noise.py`` — the Windows shutdown-noise suppressor.

Every ``scqo run`` on Windows printed a traceback after ``saved: <path>``: qblox's transport
leaves a bare ``ProactorEventLoop`` open, and at interpreter shutdown its self-pipe read
fails with ``OSError: [WinError 87]``, which asyncio reports through
``call_exception_handler`` rather than raising.

The suppression is deliberately narrow, so most of what is worth testing is what it does
NOT swallow. Pure stdlib — no qblox, no hardware, and nothing Windows-specific (the
``winerror`` attribute is set by hand, so this runs the same on any platform).
"""

from __future__ import annotations

import asyncio

import pytest

from scqo_qblox.backend._asyncio_noise import (
    _SELF_PIPE_MESSAGE,
    silence_proactor_self_pipe_noise,
)


class _Cluster:
    def __init__(self, loop):
        self._transport = type("T", (), {"_loop": loop})()


class _Component:
    def __init__(self, loop):
        self.instrument = _Cluster(loop)


class _Agent:
    """The shape ``silence_proactor_self_pipe_noise`` reads off a HardwareAgent."""

    def __init__(self, *loops):
        self._clusters = {f"cluster_{i}": _Component(loop) for i, loop in enumerate(loops)}


@pytest.fixture
def loop():
    made = asyncio.new_event_loop()
    yield made
    made.close()


@pytest.fixture
def seen(loop):
    """Install a recording handler first, so we can see what gets delegated through."""
    captured: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: captured.append(context))
    return captured


def _win87(message: str = _SELF_PIPE_MESSAGE) -> dict:
    error = OSError("the real one carries a localized message")
    error.winerror = 87
    return {"message": message, "exception": error, "loop": None}


def test_the_shutdown_message_is_swallowed(loop, seen):
    assert silence_proactor_self_pipe_noise(_Agent(loop)) == 1

    loop.call_exception_handler(_win87())

    assert seen == []


def test_the_same_message_with_a_different_error_still_surfaces(loop, seen):
    """Narrow on purpose: a self-pipe that fails some other way is a real problem."""
    silence_proactor_self_pipe_noise(_Agent(loop))

    other = OSError("broken pipe")
    other.winerror = 109
    loop.call_exception_handler({"message": _SELF_PIPE_MESSAGE, "exception": other})
    loop.call_exception_handler({"message": _SELF_PIPE_MESSAGE, "exception": RuntimeError()})
    loop.call_exception_handler({"message": _SELF_PIPE_MESSAGE})  # no exception at all

    assert len(seen) == 3


def test_unrelated_failures_are_delegated_untouched(loop, seen):
    silence_proactor_self_pipe_noise(_Agent(loop))

    loop.call_exception_handler(_win87(message="Task exception was never retrieved"))
    loop.call_exception_handler({"message": "Future exception was never retrieved"})

    assert [c["message"] for c in seen] == [
        "Task exception was never retrieved",
        "Future exception was never retrieved",
    ]


def test_installing_twice_does_not_stack_handlers(loop, seen):
    """``acquire()`` runs once per experiment; without the marker the handlers would nest
    one deeper every run for the whole session."""
    assert silence_proactor_self_pipe_noise(_Agent(loop)) == 1
    first = loop.get_exception_handler()

    assert silence_proactor_self_pipe_noise(_Agent(loop)) == 0
    assert loop.get_exception_handler() is first

    loop.call_exception_handler(_win87())
    assert seen == []


def test_every_cluster_gets_its_own_loop_wrapped():
    loops = [asyncio.new_event_loop() for _ in range(3)]
    try:
        assert silence_proactor_self_pipe_noise(_Agent(*loops)) == 3
        for one in loops:
            captured: list[dict] = []
            # the wrapper is already in place, so anything it passes on lands here
            previous = one.get_exception_handler()
            assert previous is not None
    finally:
        for one in loops:
            one.close()


def test_a_broken_agent_shape_never_raises():
    """Cosmetic code must degrade to 'the noise comes back', never to a failed run."""

    class Exploding:
        @property
        def _clusters(self):
            raise RuntimeError("qblox changed shape")

    class NoTransport:
        def __init__(self):
            self._clusters = {"c": type("X", (), {"instrument": object()})()}

    assert silence_proactor_self_pipe_noise(Exploding()) == 0
    assert silence_proactor_self_pipe_noise(NoTransport()) == 0
    assert silence_proactor_self_pipe_noise(object()) == 0
    assert silence_proactor_self_pipe_noise(None) == 0


# --------------------------------------------------------------------------- wiring
def _experiment(tmp_path, roster):
    from conftest import make_backend, make_experiment
    from scqo.experiments import get

    backend = make_backend(tmp_path, roster)
    cls = get("resonator_spectroscopy")
    exp = make_experiment(cls, backend, roster, cls.Parameters(targets=["q1"], num_points=5))
    exp.sweep_axes = exp.define_sweep()
    return backend, exp


@pytest.mark.parametrize("run_fails", [False, True])
def test_acquire_installs_the_handler_around_every_run(tmp_path, roster, monkeypatch, run_fails):
    """Pins the WIRING, not just the helper. The clusters only exist once ``run()`` has
    connected them, so the call sits in a ``finally`` — a run that RAISES has still opened
    the loops that would otherwise print at shutdown, so it needs the handler just as much.

    ``run()`` is stubbed because it would otherwise dial the real cluster in hw_config.
    """
    pytest.importorskip("qblox_scheduler")
    import scqo_qblox.backend._asyncio_noise as noise
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    backend, exp = _experiment(tmp_path, roster)
    seen_agents = []
    monkeypatch.setattr(noise, "silence_proactor_self_pipe_noise", seen_agents.append)

    sentinel = object()
    if run_fails:
        def fake_run(schedule, timeout=None):
            raise RuntimeError("cluster went away")
    else:
        def fake_run(schedule, timeout=None):
            return sentinel

    monkeypatch.setattr(backend._hw_agent, "run", fake_run)
    monkeypatch.setattr(QbloxBackend, "_to_canonical", staticmethod(lambda raw, experiment: raw))

    if run_fails:
        with pytest.raises(RuntimeError, match="cluster went away"):
            backend.acquire(exp)
    else:
        assert backend.acquire(exp) is sentinel

    assert seen_agents == [backend._hw_agent]
