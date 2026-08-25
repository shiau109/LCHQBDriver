"""``QbloxBackend.preview``: compiled-schedule artifacts, fully offline.

The contract (scqo ``Backend.preview`` hook): the caller has set
``exp.sweep_axes``; the backend compiles exactly as ``HardwareAgent.run``
would and renders ``pulse_diagram.html`` + ``timing_table.html`` — both need
the compiled timeline, both are mandatory. No gate-level circuit diagram by
design (the vendor renderer cannot draw hardware-loop swept schedules — see
the preview() docstring). No cluster call is allowed anywhere on this path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import make_backend, make_experiment  # noqa: E402

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402


def _ramsey(tmp_path, roster):
    cls = get("qubit_ramsey")
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_points=5,
                                         num_averages=2))
    exp.sweep_axes = exp.define_sweep()  # the Session's job before the hook
    return backend, exp


def test_preview_writes_both_artifacts_offline(tmp_path, roster):
    backend, exp = _ramsey(tmp_path, roster)
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir)
    assert [f.name for f in files] == ["pulse_diagram.html",
                                       "timing_table.html"]
    for f in files:
        assert f.parent == out_dir
        assert f.is_file() and f.stat().st_size > 0


def test_preview_never_touches_the_network(tmp_path, roster, monkeypatch):
    """acquire() syncs attenuator ceilings off the cluster before probing;
    preview must not (offline by contract, and _sync_att_limits can also
    write att_limits.json)."""
    backend, exp = _ramsey(tmp_path, roster)
    monkeypatch.setattr(
        backend, "_sync_att_limits",
        lambda: pytest.fail("preview called _sync_att_limits (network)"))
    files = backend.preview(exp, tmp_path / "prev")
    assert files  # rendered without any cluster interaction


def test_preview_overwrites_a_pinned_out_dir(tmp_path, roster):
    """`scqo run ... --preview --out DIR` reuses one folder: a second render
    into the same directory must succeed in place, not refuse."""
    backend, exp = _ramsey(tmp_path, roster)
    out_dir = tmp_path / "pinned"
    first = backend.preview(exp, out_dir)
    second = backend.preview(exp, out_dir)
    assert [f.name for f in first] == [f.name for f in second]
    assert all(f.is_file() for f in second)


def test_backend_specific_options_are_gated(tmp_path, roster):
    """--simulate-ns is a QM thing: here it must refuse by name, while
    --no-simulate (a request for offline-ness this preview already has) is
    accepted as a harmless no-op."""
    backend, exp = _ramsey(tmp_path, roster)
    with pytest.raises(ValueError, match="no simulator"):
        backend.preview(exp, tmp_path / "prev", simulate_ns=5000)
    assert not (tmp_path / "prev").exists()
    files = backend.preview(exp, tmp_path / "prev", no_simulate=True)
    assert [f.name for f in files] == ["pulse_diagram.html",
                                       "timing_table.html"]


def test_preview_refuses_an_oversized_render(tmp_path, roster):
    """The vendor diagram unrolls every loop before sampling (points x
    repetitions of deep-copied operations — profiled 2026-08-09), so a
    lab-sized schedule is refused by name, with the remedy, BEFORE the
    compile. 51 x 400 is the real chipA standing default that hung."""
    cls = get("qubit_ramsey")
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_points=51,
                                         num_averages=400))
    exp.sweep_axes = exp.define_sweep()
    out_dir = tmp_path / "prev"
    with pytest.raises(ValueError, match="rendered shots") as excinfo:
        backend.preview(exp, out_dir)
    assert "--set num_averages=2" in str(excinfo.value)  # names the remedy
    assert not out_dir.exists()


def test_preview_refuses_self_acquiring_probes(tmp_path, roster):
    """Self-acquiring probes step hardware parameters across sub-bands in Python
    and have no single schedule to preview."""
    cls = get("broadband_resonator_spectroscopy")
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"]))
    exp.sweep_axes = exp.define_sweep()
    out_dir = tmp_path / "prev"
    with pytest.raises(ValueError, match="cannot be previewed"):
        backend.preview(exp, out_dir)
    assert not out_dir.exists()
