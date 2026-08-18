"""``qubit_spectroscopy_overlap``: the drive and the readout tone really are
concurrent, and the ADC really does open after both.

That claim is the entire reason the experiment exists, and NOTHING else in this
suite can see it. `test_probe_surface` proves the schedule compiles — a
drive-then-readout schedule compiles just as happily. So this file walks the
COMPILED tree with absolute times and asserts the three instants against each
other. Compiled, not built: the readout pulse is emitted as a long-square
optimization (a ``VoltageOffset`` pair plus a 4 ns tail, not one ``SquarePulse``),
and only the compiler knows that, so a built-tree assertion would be checking a
shape the instrument never sees.

Reading the numbers below: ``acq_delay`` is measured from the readout pulse
ONSET and already carries the element's time-of-flight, so an acquisition that
starts at ``TOF + acq_start_ns`` in emit time is integrating the tone from
``acq_start_ns`` in ARRIVAL time. TOF is not part of the lead — it is the cable,
and the probe adds to it rather than replacing it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402

DRIVE_PORT = "q1:mw"
READOUT_PORT = "q1:res"


def _events(schedule, base=0.0, out=None):
    """Every operation in the compiled tree as ``(abs_time_s, name, op)``.

    Absolute time has to ACCUMULATE down the nesting: each schedulable's
    ``abs_time`` is relative to its own (sub)schedule, and this probe's ops sit
    four levels deep inside two loop bodies plus the Measure's subschedule.
    Resolve schedulables rather than iterating ``operations`` — that dict is
    hash-keyed, so the two ``VoltageOffset(0, 0)``-shaped ops would collapse.
    """
    out = [] if out is None else out
    ops = getattr(schedule, "operations", None) or {}
    for sch in getattr(schedule, "schedulables", {}).values():
        op = ops.get(sch.get("operation_id"))
        if op is None:
            continue
        t = base + float(sch.get("abs_time") or 0.0)
        out.append((t, type(op).__name__, op))
        if hasattr(op, "body"):
            _events(op.body, t, out)
        elif hasattr(op, "operations"):
            _events(op, t, out)
    return out


def _pulses(events, port):
    """``(start_s, end_s, info)`` for every emitted pulse on one port."""
    found = []
    for t, _name, op in events:
        info = (getattr(op, "data", {}) or {}).get("pulse_info") or []
        if isinstance(info, dict):
            info = [info]
        for p in info:
            if isinstance(p, dict) and p.get("port") == port:
                start = t + float(p.get("t0") or 0.0)
                found.append((start, start + float(p.get("duration") or 0.0), p))
    return sorted(found)


def _acquisition(events):
    """``(start_s, end_s, info)`` of the single acquisition."""
    found = []
    for t, _name, op in events:
        info = (getattr(op, "data", {}) or {}).get("acquisition_info") or []
        if isinstance(info, dict):
            info = [info]
        for a in info:
            if isinstance(a, dict):
                start = t + float(a.get("t0") or 0.0)
                found.append((start, start + float(a.get("duration") or 0.0), a))
    assert len(found) == 1, f"expected exactly one acquisition, got {len(found)}"
    return found[0]


def _run(tmp_path, roster, **params):
    """Compile the probe and return ``(events, tof_s, drive_amp, readout_s)``."""
    cls = get("qubit_spectroscopy_overlap")
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        cls, backend, roster,
        cls.Parameters(targets=["q1"], num_drive_freq_points=5, num_averages=2, **params),
    )
    # the two-tone probes play the drive chain's residual (spec_amp), which the
    # fixture leaves unseeded; the core run() solves it before probing
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    drive_amp = float(exp.device.channel("q1", "drive").drive_amp)
    readout_s = float(exp.device.channel("q1", "readout").readout_duration_s)
    element = backend.device.component("q1_ro")._element
    tof = element.measure.acq_delay
    tof_s = float(tof() if callable(tof) else tof)
    return _events(compile_probe(backend, exp)), tof_s, drive_amp, readout_s


# --------------------------------------------------------------- the co-start

def test_the_two_tones_start_at_the_same_instant(tmp_path, roster):
    """THE claim. A zero-duration ``VoltageOffset`` latched immediately before
    the ``Measure`` puts the drive up on the same clock edge the readout pulse
    begins on — no ``ref_op``, no arithmetic, just ASAP chaining off an op with
    no length. Equal to the nanosecond, not merely close."""
    events, _tof, drive_amp, _ro = _run(tmp_path, roster, acq_start_ns=400.0)

    drive_on = [p for p in _pulses(events, DRIVE_PORT)
                if p[2].get("offset_path_I") == pytest.approx(drive_amp)]
    assert len(drive_on) == 1, "expected one saturation VoltageOffset(drive_amp)"
    readout = _pulses(events, READOUT_PORT)
    assert readout, "expected an emitted readout tone"

    assert drive_on[0][0] * 1e9 == pytest.approx(readout[0][0] * 1e9, abs=1e-3)


def test_the_drive_is_bounded_by_drive_len_ns(tmp_path, roster):
    """The drive is CW on this backend, so its LENGTH is the gap between the two
    offsets. The sibling probe leaves it latched across the whole loop and
    ignores ``drive_len_ns`` entirely; here it is a real window."""
    events, _tof, drive_amp, _ro = _run(
        tmp_path, roster, acq_start_ns=400.0, drive_len_ns=600.0)

    offsets = _pulses(events, DRIVE_PORT)
    on = [p for p in offsets if p[2].get("offset_path_I") == pytest.approx(drive_amp)]
    off = [p for p in offsets if p[2].get("offset_path_I") == 0]
    assert on and off, "the saturation drive must be turned back OFF"
    assert (off[0][0] - on[0][0]) * 1e9 == pytest.approx(600.0, abs=1e-3)


# ------------------------------------------------------- the ADC opens later

@pytest.mark.parametrize("acq_start_ns", [0.0, 400.0])
def test_the_tone_is_lengthened_and_the_adc_delayed_by_the_same_lead(
        tmp_path, roster, acq_start_ns):
    """``acq_start_ns=0`` is today's timing (the ADC opens with the pulse, cable
    delay aside) and must stay exactly that; a positive lead must move the tone
    END and the ADC by the same amount, or the integration window would run off
    the end of the pulse."""
    events, tof_s, _amp, readout_s = _run(
        tmp_path, roster, acq_start_ns=acq_start_ns)

    readout = _pulses(events, READOUT_PORT)
    tone_start, tone_end = readout[0][0], max(end for _s, end, _p in readout)
    assert (tone_end - tone_start) * 1e9 == pytest.approx(
        acq_start_ns + readout_s * 1e9, abs=1e-3), "tone not lengthened by the lead"

    acq_start, acq_end, _info = _acquisition(events)
    assert (acq_start - tone_start) * 1e9 == pytest.approx(
        tof_s * 1e9 + acq_start_ns, abs=1e-3), "acq_delay is not TOF + the lead"
    # the window still lands inside the tone, in the ARRIVAL frame (TOF removed)
    assert acq_end - tof_s <= tone_end + 1e-12


def test_the_adc_opens_after_both_tones_are_already_on(tmp_path, roster):
    """The request in one assertion: with a lead, the acquisition starts strictly
    after BOTH pulses have started — and by the lead, not by an accident of the
    cable delay (which is why TOF is subtracted before comparing)."""
    events, tof_s, drive_amp, _ro = _run(
        tmp_path, roster, acq_start_ns=400.0, drive_len_ns=600.0)

    drive_on = [p[0] for p in _pulses(events, DRIVE_PORT)
                if p[2].get("offset_path_I") == pytest.approx(drive_amp)][0]
    tone_on = _pulses(events, READOUT_PORT)[0][0]
    acq_on = _acquisition(events)[0] - tof_s  # arrival frame

    assert acq_on > drive_on and acq_on > tone_on
    assert (acq_on - max(drive_on, tone_on)) * 1e9 == pytest.approx(400.0, abs=1e-3)


def test_the_next_point_starts_after_the_acquisition_finishes(tmp_path, roster):
    """The trailing re-anchor. Without it the ASAP chain would continue from the
    zero-duration drive-OFF — which for a short ``drive_len_ns`` sits well before
    the readout ends — and each sweep point would begin inside the previous
    measurement. Silent on hardware; visible only here."""
    events, _tof, _amp, _ro = _run(
        tmp_path, roster, acq_start_ns=400.0, drive_len_ns=100.0)

    _acq_start, acq_end, _info = _acquisition(events)
    trailing = [t for t, name, _op in events if name == "IdlePulse" and t > 0]
    assert trailing, "expected the trailing re-anchor IdlePulse"
    assert max(trailing) >= acq_end - 1e-12
