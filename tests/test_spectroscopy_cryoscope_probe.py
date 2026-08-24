"""Qblox spectroscopy-cryoscope probe: the pi-area square tone, the refusals,
and the built schedule's structure (arch-centered NCO sweep, wait unrolling,
flux step ordering). Compilation is covered by ``test_probe_surface.py``.
"""

import numpy as np
import pytest

from conftest import make_backend, make_experiment

from scqo.experiments import get

import scqo_qblox.experiments  # noqa: F401  (registers the qblox probes)
from scqo_qblox.experiments.qubit_spectroscopy_cryoscope import (
    check_drive_amp,
    square_pi_amp,
    validate_inputs,
    x180_area_amp_ns,
)

SMALL = dict(num_drive_freq_points=5, num_wait_points=5, max_wait_ns=200,
             drive_len_ns=64, num_averages=2)


def _experiment(tmp_path, roster, **params):
    cls = get("qubit_spectroscopy_cryoscope")
    exp = make_experiment(cls, make_backend(tmp_path, roster), roster,
                          cls.Parameters(targets=["q1"], **{**SMALL, **params}))
    exp.sweep_axes = exp.define_sweep()
    return exp


def _walk(schedule, out=None, seen=None):
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(schedule) in seen:
        return out
    seen.add(id(schedule))
    ops = getattr(schedule, "operations", None) or {}
    for sch in getattr(schedule, "schedulables", {}).values():
        op = ops.get(sch.get("operation_id"))
        if op is None:
            continue
        out.append((type(op).__name__, op))
        if hasattr(op, "body"):
            _walk(op.body, out, seen)
        elif hasattr(op, "operations"):
            _walk(op, out, seen)
    return out


# ------------------------------------------------------------------- pure math

def test_x180_area_matches_the_rendered_drag_envelope():
    """The area helper integrates exactly what rxy_drag_pulse renders: a
    Gaussian at sigma = duration/8 with the average of its first and last
    samples subtracted (the factory's offset handling). Independent formula
    here, vendor sampling in the helper."""
    amp, dur_s = 0.21, 200e-9
    n = 200
    t = np.arange(n) * 1e-9
    mu, sigma = dur_s / 2, dur_s / 8
    g = amp * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    expected = float(np.sum(g - (g[0] + g[-1]) / 2))
    assert x180_area_amp_ns(amp, dur_s) == pytest.approx(expected, rel=1e-6)


def test_square_pi_amp_holds_the_area():
    assert square_pi_amp(52.0, 400.0, 1.0) == pytest.approx(0.13)
    # amp_factor scales linearly; a longer pulse is proportionally weaker
    assert square_pi_amp(52.0, 400.0, 0.5) == pytest.approx(0.065)
    assert square_pi_amp(52.0, 800.0, 1.0) == pytest.approx(0.065)


def test_refusals_by_name():
    with pytest.raises(ValueError, match="one at a time"):
        validate_inputs(["q1", "q2"], "square")
    with pytest.raises(NotImplementedError, match="cosine"):
        validate_inputs(["q1"], "cosine")
    with pytest.raises(NotImplementedError, match="gaussian"):
        validate_inputs(["q1"], "gaussian")
    with pytest.raises(ValueError, match="qubit_power_rabi"):
        check_drive_amp("q1", float("nan"), 0.1, 400.0)
    with pytest.raises(ValueError, match="drive_len_ns"):
        check_drive_amp("q1", 0.5, 3.25, 16.0)


def test_shape_refusal_fires_from_probe(tmp_path, roster):
    exp = _experiment(tmp_path, roster, drive_shape="cosine")
    with pytest.raises(NotImplementedError, match="drive_shape='square'"):
        exp.probe()


# ------------------------------------------------------------ built schedule

@pytest.fixture()
def walked(tmp_path, roster):
    exp = _experiment(tmp_path, roster)
    return exp, _walk(exp.probe())


def test_nco_sweep_is_endpoint_form_around_the_parked_center(walked):
    """The FREQUENCY domain runs center+det[0] .. center+det[-1] where the
    center is drive_freq_hz PLUS the arch-predicted parked offset — the base
    class's resolved_center_offset_hz, the same value the estimator adds back."""
    exp, ops = walked
    det = exp.sweep_axes["detuning_hz"]
    center = (float(exp.device.channel("q1", "drive").drive_freq_hz)
              + exp.resolved_center_offset_hz("q1"))
    domains = []
    for name, op in ops:
        for domain in (getattr(op, "domain", None) or {}).values():
            if getattr(domain.dtype, "value", domain.dtype) == "frequency":
                domains.append(domain)
    assert len(domains) == 1
    assert domains[0].start == pytest.approx(center + det[0])
    assert domains[0].stop == pytest.approx(center + det[-1])
    assert domains[0].num == det.size


def test_wait_blocks_follow_the_snapped_deduped_axis_in_order(walked):
    """The wait axis is consumed VERBATIM from sweep_axes (log-spaced,
    16 ns-floored, 4 ns-snapped, deduped) and unrolled ascending — the flat bin
    order the canonical decode expects."""
    exp, ops = walked
    waits = [float(v) for v in exp.sweep_axes["wait_time_ns"]]
    measures = [op for name, op in ops if name == "Measure"]
    assert [op.data["gate_info"]["coords"]["wait_q1"] for op in measures] == waits
    assert all(w >= 16 and w % 4 == 0 for w in waits)


def test_flux_step_holds_through_wait_drive_and_tail(walked):
    """Per wait block: flux ON -> wait -> pi-area square tone on the drive
    port-clock -> 100 ns tail -> flux back to idle -> Measure (readout at the
    calibrated point, the QM mirror)."""
    exp, ops = walked
    names = [name for name, _ in ops]
    first = names.index("Reset")
    block = ops[first:first + 8]
    kinds = [name for name, _ in block]
    assert kinds[:2] == ["Reset", "VoltageOffset"]          # step ON after reset
    square = next(op for name, op in block if name == "SquarePulse")
    info = square.data["pulse_info"]
    assert info["clock"] == "q1.01"                         # the drive tone
    assert round(square.duration * 1e9) == SMALL["drive_len_ns"]
    assert kinds.index("Measure") > kinds.index("SquarePulse")
    # the tone holds the x180 area at drive_len_ns
    view = exp.device.channel("q1", "drive")
    expected_amp = square_pi_amp(
        x180_area_amp_ns(float(view.pi_amp), float(view.pi_duration_s)),
        float(exp.params.drive_len_ns), 1.0)
    assert info["amplitude"] == pytest.approx(expected_amp)
