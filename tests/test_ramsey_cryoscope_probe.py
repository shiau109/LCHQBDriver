"""Qblox ramsey-cryoscope probe: the composed 4k+r duration scheme, the
realtime PHASE frame loop, and the refusals. Compilation itself is covered by
``test_probe_surface.py::test_probe_compiles``; these tests pin the BUILT
schedule's structure — the parts a clean compile does not prove.
"""

import numpy as np
import pytest

from conftest import make_backend, make_experiment

from scqo.experiments import get

import scqo_qblox.experiments  # noqa: F401  (registers the qblox probes)
from scqo_qblox.experiments.qubit_ramsey_cryoscope import (
    MAX_DURATION_NS,
    duration_split,
    validate_inputs,
)

MAX_NS = 32
N_FRAMES = 4


def _experiment(tmp_path, roster, **params):
    cls = get("qubit_ramsey_cryoscope")
    exp = make_experiment(
        cls, make_backend(tmp_path, roster), roster,
        cls.Parameters(targets=["q1"], max_duration_ns=MAX_NS,
                       num_frames=N_FRAMES, num_averages=2, **params))
    exp.sweep_axes = exp.define_sweep()
    return exp


def _walk(schedule, out=None, seen=None):
    """Every operation in the tree, in SCHEDULABLE order (the test_active_reset
    walker: resolve schedulables, record before descending into loop bodies)."""
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

def test_duration_split_composes_every_nanosecond():
    """n = 4k + r with r < 4 — and the pad keeps the free-evolution window
    constant with every same-sequencer gap a multiple of 4 (the compiler's
    wait floor; the reason the composed scheme exists)."""
    for max_ns in (32, 240, MAX_DURATION_NS):
        for n in range(1, max_ns + 1):
            k4, r = duration_split(n)
            assert k4 + r == n and 0 <= r < 4 and k4 % 4 == 0
            pad = max_ns + 4 - n
            assert pad >= 4                       # never a sub-floor wait
            assert (k4 + r + pad) == max_ns + 4   # constant window


def test_refusals_by_name():
    durations = np.arange(1, MAX_NS + 1, dtype=float)
    with pytest.raises(ValueError, match="one at a time"):
        validate_inputs(["q1", "q2"], durations, MAX_NS)
    with pytest.raises(ValueError, match=f"{MAX_DURATION_NS} ns ceiling"):
        validate_inputs(["q1"], np.arange(1, 601, dtype=float), 600)


# ------------------------------------------------------------ built schedule

@pytest.fixture()
def walked(tmp_path, roster):
    exp = _experiment(tmp_path, roster)
    return exp, _walk(exp.probe())


def test_frame_loop_is_a_positive_endpoint_exclusive_phase_domain(walked):
    """The frame axis rides a realtime PHASE loop in DEGREES: 0 .. 360*(nf-1)/nf
    (endpoint-exclusive, like the turns axis), POSITIVE — phase tomography, not
    qubit_ramsey's negative virtual-detuning ramp."""
    _exp, ops = walked
    domains = []
    for name, op in ops:
        for domain in (getattr(op, "domain", None) or {}).values():
            if getattr(domain.dtype, "value", domain.dtype) == "phase":
                domains.append(domain)
    assert len(domains) == MAX_NS  # one frame loop per unrolled duration block
    for domain in domains:
        assert domain.start == pytest.approx(0.0)
        assert domain.stop == pytest.approx(360.0 * (N_FRAMES - 1) / N_FRAMES)
        assert domain.num == N_FRAMES


def test_frame_shift_goes_through_shift_clock_phase(walked):
    """The variable phase must ride ShiftClockPhase (expression-safe), never an
    Rxy phi — a variable pulse phase does not compile in a realtime loop, and
    the drive clock is the qubit's own."""
    _exp, ops = walked
    shifts = [op for name, op in ops if name == "ShiftClockPhase"]
    resets = [op for name, op in ops if name == "ResetClockPhase"]
    assert len(shifts) == MAX_NS and len(resets) == MAX_NS
    for op in shifts + resets:
        assert op.data["pulse_info"]["clock"] == "q1.01"


def test_composed_flux_pulse_splits_offset_segment_and_remainder(walked):
    """Durations decompose as the scheme promises: an offset pair for the 4k
    segment (n >= 4), a 1/2/3-sample SquarePulse for the remainder — so only
    THREE remainder waveforms ever exist, and n = 4k plays no waveform at all."""
    exp, ops = walked
    flux_squares = [
        op for name, op in ops if name == "SquarePulse"
        and op.data["pulse_info"]["port"].endswith(":fl")]
    seen_ns = sorted({round(op.duration * 1e9) for op in flux_squares})
    assert seen_ns == [1, 2, 3]
    # one remainder pulse per duration with r > 0
    expected = sum(1 for n in range(1, MAX_NS + 1) if n % 4)
    assert len(flux_squares) == expected
    # every remainder rides ON the standing bias: amplitude is the EXCURSION
    from scqo.experiments._capabilities import flux_anchor_v
    from scqo_qblox.experiments._flux_limits import flux_rail_v, to_dac_fraction

    port = flux_squares[0].data["pulse_info"]["port"]
    rail = flux_rail_v(exp, port, name="test")
    exc = to_dac_fraction(float(exp.params.flux_pulse_amp_v), rail)
    for op in flux_squares:
        assert op.data["pulse_info"]["amplitude"] == pytest.approx(exc)
    # and the offset segments express idle vs idle+amp ABSOLUTE levels
    idle = flux_anchor_v(exp, "q1")
    offsets = {round(op.data["pulse_info"]["offset_path_I"], 9)
               for name, op in ops if name == "VoltageOffset"}
    assert round(to_dac_fraction(idle + exp.params.flux_pulse_amp_v, rail), 9) in offsets
    assert round(to_dac_fraction(idle, rail), 9) in offsets
    assert 0.0 in offsets  # the end-of-schedule safety park


def test_one_measure_per_duration_block_on_the_probe_channel(walked):
    _exp, ops = walked
    measures = [op for name, op in ops if name == "Measure"]
    assert len(measures) == MAX_NS  # frame axis rides the realtime loop
    durations = [op.data["gate_info"]["coords"]["duration_q1"] for op in measures]
    assert durations == [float(n) for n in range(1, MAX_NS + 1)]  # ascending
