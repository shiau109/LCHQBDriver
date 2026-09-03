"""``QbloxBackend.release_instruments`` — hand the cluster back before a prompt.

This backend is the reason the neutral ``scqo.Backend`` hook exists:
``HardwareAgent.run()`` connects the clusters and the vendor exposes no
disconnect, so a process that has acquired once holds four sockets per cluster
until it exits. The CLI calls this before blocking on the accept prompt.

Two things are worth pinning and neither needs hardware: that a backend which
never dialled does NOT dial in order to hang up (``get_clusters()`` would, which
is exactly why the implementation reads ``_clusters`` directly), and that a
close which never returns costs a warning rather than the caller's thread.
"""

from __future__ import annotations

import threading

import pytest

from conftest import make_backend  # noqa: E402


class _Cluster:
    def __init__(self, block: threading.Event | None = None, raises: bool = False):
        self.closed = False
        self._block = block
        self._raises = raises

    def close(self) -> None:
        if self._block is not None:
            self._block.wait(30)
        if self._raises:
            raise RuntimeError("slot 3 did not answer")
        self.closed = True


class _Component:
    def __init__(self, name: str, cluster: _Cluster):
        self.name = f"ic_{name}"
        self.cluster = cluster


class _Coordinator:
    def __init__(self):
        self.removed: list[str] = []

    def remove_component(self, name: str) -> None:
        self.removed.append(name)


class _Agent:
    """The shape ``release_instruments`` reads off a HardwareAgent."""

    def __init__(self, clusters: dict):
        self._clusters = clusters
        self.instrument_coordinator = _Coordinator()
        self.get_clusters_calls = 0

    def get_clusters(self) -> dict:  # the accessor that would CONNECT
        self.get_clusters_calls += 1
        raise AssertionError("release_instruments must never call get_clusters()")


def _with_agent(backend, clusters: dict) -> _Agent:
    agent = _Agent(clusters)
    backend._hw_agent = agent
    return agent


def test_nothing_connected_releases_nothing_and_dials_nothing(tmp_path, roster):
    """The common case: `scqo accept` on a session that never acquired."""
    backend = make_backend(tmp_path, roster)
    agent = _with_agent(backend, {})

    assert backend.release_instruments() == []
    assert agent.get_clusters_calls == 0  # never connect in order to disconnect


def test_a_connected_cluster_is_deregistered_closed_and_forgotten(tmp_path, roster):
    backend = make_backend(tmp_path, roster)
    cluster = _Cluster()
    agent = _with_agent(backend, {"cluster0": _Component("cluster0", cluster)})

    assert backend.release_instruments() == ["cluster0"]
    assert cluster.closed
    assert agent.instrument_coordinator.removed == ["ic_cluster0"]
    # Cleared, so a later acquire() rebuilds rather than using dead objects.
    assert agent._clusters == {}


def test_a_close_that_never_returns_warns_instead_of_hanging(tmp_path, roster):
    """The failure this design exists to bound: ``Cluster.close()`` can block
    forever, and the caller is on its way to ask a human a question."""
    from scqo_qblox.backend import qblox_backend

    backend = make_backend(tmp_path, roster)
    never = threading.Event()
    agent = _with_agent(backend, {"cluster0": _Component("cluster0", _Cluster(never))})
    monkey = qblox_backend._CLUSTER_CLOSE_TIMEOUT_S
    qblox_backend._CLUSTER_CLOSE_TIMEOUT_S = 0.1
    try:
        with pytest.warns(UserWarning, match="did not return within"):
            released = backend.release_instruments()
    finally:
        qblox_backend._CLUSTER_CLOSE_TIMEOUT_S = monkey
        never.set()  # let the daemon thread finish so pytest exits clean

    assert released == []  # not claimed as released: the sockets are still open
    assert agent._clusters == {}  # but deregistered, so a rerun rebuilds


def test_a_raising_close_warns_and_does_not_propagate(tmp_path, roster):
    backend = make_backend(tmp_path, roster)
    _with_agent(backend, {"cluster0": _Component("cluster0", _Cluster(raises=True))})

    with pytest.warns(UserWarning, match="Cluster.close.. raised"):
        assert backend.release_instruments() == ["cluster0"]
