"""The Ramsey artificial detuning actually reaches the instrument, with the right sign.

Two independent silent failures live here, and both shipped:

1. SPELLING. `X90(qubit, phase=...)` compiles clean and DISCARDS the phase — `phase` is
   not an Rxy factory kwarg, so it lands in `gate_info["device_overrides"]` and
   `circuit_to_device` drops it. Until 2026-08-01 every Qblox Ramsey requested a
   detuning the hardware never applied. `Rxy(theta=90, phi=...)` is the only door: `phi`
   IS a gate_info factory kwarg.
2. SIGN. A programmed phase is a carrier-phase ADVANCE, so it runs opposite to the free
   precession of a qubit above its drive. Only a NEGATIVE ramp yields a fringe at
   (applied + err), which is what scqo's shared `estimate()` subtracts `applied` from.
   A positive ramp gives |applied - err|, and the accepted updates then walk the drive to
   the absorbing point err = 2*applied, where `detuning_error_hz` reads exactly 0.0 and
   the fit looks perfect. There is no error anywhere in that failure — only a wrong qubit
   frequency — which is why it is pinned here.

Both assertions are on the COMPILED tree, and both pin VALUES: a test asserting merely
"phase != 0" would pass on a constant phase, which is the same silent-no-op class as
defect 1.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

from scqo_qblox.experiments.qubit_ramsey import QbloxQubitRamsey  # noqa: E402

#: a calibrated discriminator, needed only by the active-reset shape
ROTATION_RAD = -0.38962897776554817
THRESHOLD = -3.25e-4
DEPLETION_S = 795.77e-9


def _leaves(node):
    """Every leaf operation of a compiled schedule, in play order.

    Descends BOTH `.operations` and `LoopOperation.body`. Two traps here: the whole
    swept body lives inside the loop, so an `.operations`-only walker sees an empty tree
    and passes vacuously; and `.operations` is HASH-KEYED, so two identical pulses in one
    iteration collapse to a single entry unless the loop body is walked directly.
    """
    ops = getattr(node, "operations", None)
    if ops:
        for op in ops.values():
            yield from _leaves(op)
        return
    body = getattr(node, "body", None)
    if body is not None:
        yield from _leaves(body)
        return
    yield node


def _pi_half_phases(compiled, port="q1:mw") -> list[float]:
    """Phases of the pi/2 DRIVE pulses, in play order.

    Filtered by waveform AND amplitude on purpose: under `reset_method="active"` the same
    port also carries the ConditionalReset's X180 (twice the amplitude) and a
    zero-waveform 4 ns op whose phase is None, and an unfiltered elementwise assertion
    both misaligns and raises on `abs(None)`.
    """
    entries = []
    for op in _leaves(compiled):
        info = (getattr(op, "data", {}) or {}).get("pulse_info") or {}
        for e in (info if isinstance(info, list) else [info]):
            if not isinstance(e, dict) or e.get("port") != port:
                continue
            if e.get("wf_func") != "qblox_scheduler.waveforms.drag":
                continue
            if e.get("phase") is None or e.get("amplitude") is None:
                continue
            entries.append((float(e["amplitude"]), float(e["phase"])))
    if not entries:
        return []
    pi_half_amp = min(amp for amp, _ in entries)  # the pi is twice the pi/2
    return [phase for amp, phase in entries
            if amp == pytest.approx(pi_half_amp, rel=1e-9)]


def _compiled(tmp_path, roster, **params):
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(QbloxQubitRamsey, backend, roster,
                          QbloxQubitRamsey.Parameters(targets=["q1"], **params))
    if params.get("reset_method") == "active":
        readout = exp.device.channel("q1", "readout")
        readout.readout_rotation_rad = ROTATION_RAD
        readout.readout_threshold = THRESHOLD
        readout.readout_depletion_s = DEPLETION_S
    return exp, compile_probe(backend, exp)


def _expected(exp, detuning_hz) -> list[float]:
    """The phase pairs the schedule must carry: the first pi/2 at 0, the second ramped
    backwards by 360 * detuning * tau. Derived from the axis the probe actually built —
    never a pasted literal, since the window spans the full 16..4000 ns default even at
    a small num_points."""
    taus_s = np.asarray(exp.sweep_axes["idle_time_ns"], dtype=float) * 1e-9
    out: list[float] = []
    for tau in taus_s:
        out += [0.0, -360.0 * detuning_hz * tau]
    return out


class TestRamseyDetuningReachesTheInstrument:

    @pytest.mark.parametrize("detuning_hz", [1e6, 2e6])
    def test_second_pi_half_carries_the_negative_ramp(self, tmp_path, roster, detuning_hz):
        """THE sign assertion, elementwise against the probe's own time axis."""
        exp, compiled = _compiled(tmp_path, roster, num_points=5, num_averages=2,
                                  frequency_detuning_hz=detuning_hz)
        assert _pi_half_phases(compiled) == pytest.approx(_expected(exp, detuning_hz))

    def test_the_ramp_slope_tracks_frequency_detuning_hz(self, tmp_path, roster):
        """Pinning the SLOPE separates the three states this line has been in:
        correct (-360*f), wrong sign (+360*f) and absent (0)."""
        for detuning_hz in (1e6, 4e6):
            exp, compiled = _compiled(tmp_path, roster, num_points=5, num_averages=2,
                                      frequency_detuning_hz=detuning_hz)
            taus_s = np.asarray(exp.sweep_axes["idle_time_ns"], dtype=float) * 1e-9
            ramped = np.asarray(_pi_half_phases(compiled)[1::2], dtype=float)
            slope = np.polyfit(taus_s, ramped, 1)[0]
            assert slope == pytest.approx(-360.0 * detuning_hz, rel=1e-6)

    def test_large_angles_are_not_wrapped(self, tmp_path, roster):
        """At the production window the ramp runs to many turns; the compiler must carry
        the unwrapped angle (it folds into the AWG gains later, not here)."""
        exp, compiled = _compiled(tmp_path, roster, num_points=101, num_averages=2,
                                  frequency_detuning_hz=1e6)
        phases = _pi_half_phases(compiled)
        assert len(phases) == 2 * 101
        assert min(phases) < -1000.0
        assert phases == pytest.approx(_expected(exp, 1e6))

    def test_ramp_survives_active_reset(self, tmp_path, roster):
        """The reset adds its own pulses to the same port; the ramp must be unaffected."""
        exp, compiled = _compiled(tmp_path, roster, num_points=5, num_averages=2,
                                  frequency_detuning_hz=2e6, reset_method="active")
        assert _pi_half_phases(compiled) == pytest.approx(_expected(exp, 2e6))

    def test_the_two_pulses_are_the_same_pulse_but_for_phase(self, tmp_path, roster):
        """Rxy(theta=90) must be X90 (amplitude = amp180 * theta/180): the fix may change
        the phase and nothing else, or it is a pulse-calibration change in disguise."""
        _exp, compiled = _compiled(tmp_path, roster, num_points=3, num_averages=2,
                                   frequency_detuning_hz=2e6)
        drive = []
        for op in _leaves(compiled):
            info = (getattr(op, "data", {}) or {}).get("pulse_info") or {}
            for e in (info if isinstance(info, list) else [info]):
                if isinstance(e, dict) and e.get("port") == "q1:mw" \
                        and e.get("wf_func") == "qblox_scheduler.waveforms.drag":
                    drive.append(e)
        assert len(drive) >= 2
        first, second = drive[0], drive[1]
        for key in ("wf_func", "amplitude", "duration", "beta", "clock", "port"):
            assert first.get(key) == second.get(key), key
        assert first["phase"] != second["phase"]

    def test_the_device_override_spelling_is_a_no_op(self, tmp_path, roster):
        """The anti-vacuity guard, and the reason the gate had to change.

        Rebuild the same body with the OLD `X90(qubit, phase=...)` spelling and show the
        compiled phases are all zero. Without this, a refactor back to X90(phase=) would
        reintroduce the original bug with only the assertions above standing in the way,
        and nothing would record WHY the spelling matters.
        """
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X90
        from qblox_scheduler.operations.loop_domains import DType, linspace
        from qblox_scheduler.backends.graph_compilation import SerialCompiler

        backend = make_backend(tmp_path, roster)
        schedule = Schedule("ramsey_old_spelling")
        sub = Schedule("ramsey_q1")
        with sub.loop(linspace(16e-9, 4000e-9, 5, dtype=DType.TIME)) as tau:
            sub.add(X90("q1"))
            sub.add(IdlePulse(tau))
            sub.add(X90("q1", phase=360.0 * 2e6 * tau))   # the historical no-op
            sub.add(Measure("q1", coords={"tau_q1": tau}, acq_channel="S_21_q1"))
            sub.add(IdlePulse(4e-9))
        schedule.add(sub)

        qd = backend._hw_agent.quantum_device
        qd.hardware_config = backend._hw_agent.hardware_configuration
        compiled = SerialCompiler().compile(schedule=schedule,
                                            config=qd.generate_compilation_config())
        phases = _pi_half_phases(compiled)
        assert phases, "guard: the walker must actually reach the drive pulses"
        assert phases == pytest.approx([0.0] * len(phases)), (
            "X90(..., phase=...) unexpectedly reached the compiler — if the vendor "
            "started forwarding it, revisit the probe's Rxy spelling")
