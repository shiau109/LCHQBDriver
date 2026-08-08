"""The Qblox CONTINUOUS parity monitor: what its schedule must and must NOT
contain. (The discrete sibling's schedule is covered by
``test_parity_switch_discrete.py``, which reuses this file's helpers.)

This probe is defined as much by its absences as by its pulses. Between shots
it waits ONLY for resonator depletion — no ``Reset``, no ``ConditionalReset``,
no thermalization — because the shot cadence IS the telegraph timebase the
switching rate is measured against. A reset silently added back would not fail
anything: the schedule would compile, the data would look like a plausible
telegraph, and every reported rate would be wrong by the ratio of the periods.
So the absence is asserted on the COMPILED tree.

The other silent failure is the shot period itself. The neutral layer converts
a per-shot switching probability into Hz with it, so a period that disagrees
with what was scheduled scales every rate linearly. It is asserted against the
schedule's own operations rather than recomputed from the same formula.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

from lchqb.experiments.qubit_parity_switch_continuous import (  # noqa: E402
    QbloxQubitParitySwitchContinuous,
)

#: 250 kHz splitting -> idle = 1 / (2 x 250 kHz) = 2000 ns, already on grid.
SPLITTING_HZ = 250e3
IDLE_NS = 2000.0
#: an arbitrary governed wait (the factor-5-era proposal for a 1 MHz linewidth,
#: 5 / (2 pi x 1e6); the default depletion_factor is 10 now, but this is a
#: fixture value, not a formula under test).
DEPLETION_S = 795.77e-9

ROTATION_RAD = -0.38962897776554817
THRESHOLD = -3.25e-4


def _experiment(tmp_path, roster, *, cls=QbloxQubitParitySwitchContinuous,
                splitting=True, depletion=True, discriminator=False, **params):
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        cls, backend, roster,
        cls.Parameters(targets=["q1"], num_shots=100, **params),
    )
    readout = exp.device.channel("q1", "readout")
    readout.pos_g_i, readout.pos_g_q = 0.0, 0.0
    readout.pos_e_i, readout.pos_e_q = 4.0, 0.0
    if depletion:
        readout.readout_depletion_s = DEPLETION_S
    if splitting:
        exp.device.channel("q1", "drive").parity_delta_f_hz = SPLITTING_HZ
    if discriminator:
        readout.readout_rotation_rad = ROTATION_RAD
        readout.readout_threshold = THRESHOLD
    return backend, exp


def _leaves(node):
    """Every leaf operation of a (compiled) schedule, in play order.

    Descends BOTH nestings: sub-schedules (``.operations``) and loop bodies
    (``.body``). The loop body is the one that matters here — the whole shot
    sequence lives inside it, so a walker that only follows ``.operations``
    inspects an empty tree and passes every absence assertion vacuously.
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


def _drive_phases(compiled, port="q1:mw") -> list[float]:
    """The phase of every microwave pulse played on the drive port, in order."""
    out = []
    for op in _leaves(compiled):
        info = (getattr(op, "data", {}) or {}).get("pulse_info") or {}
        for entry in (info if isinstance(info, list) else [info]):
            if isinstance(entry, dict) and entry.get("port") == port \
                    and entry.get("wf_func") and "phase" in entry:
                out.append(float(entry["phase"]))
    return out


def test_schedule_contains_no_reset_of_any_kind(tmp_path, roster):
    """THE defining property. `test_active_reset` covers probes that build a
    reset correctly; nothing else covers a probe that must build none."""
    backend, exp = _experiment(tmp_path, roster)
    names = " ".join(str(op) for op in _leaves(compile_probe(backend, exp)))
    assert "Reset(" not in names
    assert "ConditionalReset" not in names
    # guard: the walker really did reach the loop body it is asserting over
    assert "Y90" in names, names


def test_the_two_pulses_are_90_degrees_apart_on_hardware(tmp_path, roster):
    """The physics, asserted where it actually lands: the COMPILED pulse
    phases. The 90-degree-shifted second pulse measures sin of the parity phase
    (odd in parity); a same-axis pair would measure the even cos and see no
    parity at all — and it would look identical in the schedule tree.

    Worth checking at this level rather than trusting the gate name: on this
    backend `X90(q, phase=...)` silently DROPS the phase (it is not an Rxy
    factory kwarg), which is exactly how qubit_ramsey's virtual detuning became
    a no-op. Y90 carries its phase through the factory instead.
    """
    backend, exp = _experiment(tmp_path, roster)
    phases = _drive_phases(compile_probe(backend, exp))
    assert len(phases) == 2, phases
    assert phases[0] == pytest.approx(90.0)  # y90 first
    assert phases[1] == pytest.approx(0.0)   # x90 second


def test_shot_period_matches_the_scheduled_operations(tmp_path, roster):
    """The telegraph timebase. Asserted as a bound against the parts the probe
    controls, not recomputed from its own formula."""
    backend, exp = _experiment(tmp_path, roster)
    compile_probe(backend, exp)
    period = exp.probe_shot_period_s["q1"]
    floor = IDLE_NS * 1e-9 + DEPLETION_S
    assert period > floor  # + two pi/2 pulses + the readout
    assert period < floor + 50e-6  # but not absurdly more


def test_idle_comes_from_the_stored_splitting(tmp_path, roster):
    backend, exp = _experiment(tmp_path, roster)
    exp.sweep_axes = exp.define_sweep()
    assert exp.resolved_idle_ns("q1") == pytest.approx(IDLE_NS)


def test_idle_override_wins(tmp_path, roster):
    backend, exp = _experiment(tmp_path, roster, splitting=False,
                               idle_time_ns=1000.0)
    exp.sweep_axes = exp.define_sweep()
    assert exp.resolved_idle_ns("q1") == pytest.approx(1000.0)


def test_refuses_without_the_stored_splitting(tmp_path, roster):
    backend, exp = _experiment(tmp_path, roster, splitting=False)
    with pytest.raises(ValueError, match="parity_delta_f_hz"):
        exp.define_sweep()


def test_refuses_without_a_governed_depletion_wait(tmp_path, roster):
    """Unlike other probes, an ungoverned wait here does not merely risk a Stark
    shift — it makes the shot cadence, and therefore the rate, a guess."""
    backend, exp = _experiment(tmp_path, roster, depletion=False)
    with pytest.raises(ValueError, match="readout_depletion_s"):
        exp.define_sweep()


def test_state_mode_refuses_an_uncalibrated_discriminator(tmp_path, roster):
    backend, exp = _experiment(tmp_path, roster, use_state_discrimination=True)
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="discriminator"):
        exp.probe()


def test_state_mode_compiles_with_a_discriminator(tmp_path, roster):
    backend, exp = _experiment(tmp_path, roster, discriminator=True,
                               use_state_discrimination=True)
    compiled = compile_probe(backend, exp)
    assert compiled.operations


def test_too_many_shots_refused_by_name(tmp_path, roster):
    backend, exp = _experiment(tmp_path, roster)
    # an explicit num_shots override bypasses max_num_shots, so it reaches the
    # probe and the SEQUENCER limit is what must catch it
    exp.params = exp.params.model_copy(update={"num_shots": 4_000_000})
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="acquisition-bin"):
        exp.probe()


class TestAcquisitionTimeout:
    """The wall-clock deadline on a run: computed per experiment by
    ``_run_timeout_s`` (issue #24 — a fixed number was outgrown twice), with
    ``_RUN_TIMEOUT_S`` surviving as the proven-safe FLOOR. This experiment is
    the floor's binding case: parameterized by RECORD TIME, one long
    uninterrupted train of single shots, bounded by the acquisition-bin ceiling.
    """

    def test_floor_covers_the_longest_bin_limited_record(self):
        """3e6 acquisition bins at chipA's ~48.5 us shot (idle_multiple=3) is
        ~145 s of sequencer time — the most this experiment can ever request.
        The floor must clear that even when the estimate is unavailable."""
        from lchqb.backend.qblox_backend import _RUN_TIMEOUT_S

        longest_run_s = 3_000_000 * 48.5e-6
        assert _RUN_TIMEOUT_S > longest_run_s
        # ...and the old 120 s did NOT, which is what killed a 30 s-record run
        assert 120 < longest_run_s

    def test_floor_is_a_whole_number_of_minutes(self):
        """The seconds only look like seconds: the instrument coordinator hands
        the cluster ``timeout_sec // 60`` MINUTES, floor division. A value that
        is not a multiple of 60 silently loses the remainder, and anything under
        60 truncates to a zero-minute deadline."""
        from lchqb.backend.qblox_backend import _RUN_TIMEOUT_S

        assert _RUN_TIMEOUT_S % 60 == 0
        assert _RUN_TIMEOUT_S >= 300  # never below the proven-safe 5 minutes

    def test_parity_period_scales_the_deadline(self):
        """The parity monitors publish their exact shot period during probe();
        shots x the slowest period, doubled and rounded up to a minute."""
        from types import SimpleNamespace

        from lchqb.backend.qblox_backend import _run_timeout_s

        exp = SimpleNamespace(
            probe_shot_period_s={"q1": 100e-6, "q2": 80e-6},
            resolved_num_shots=lambda: 3_000_000,
            params=SimpleNamespace(targets=["q1", "q2"]),
        )
        # 3e6 x 100us = 300 s of sequencer time -> x2 = 600 s
        assert _run_timeout_s(exp) == 600

    def test_generic_sweep_scales_the_deadline(self):
        """The issue #24 shape: num_averages x sweep points x thermalization has
        no bin ceiling and outgrew the old fixed 300 s (e.g. 100 points x 4000
        averages x 1.86 ms is ~744 s of wall clock)."""
        import numpy as np
        from types import SimpleNamespace

        from lchqb.backend.qblox_backend import _run_timeout_s

        chan = SimpleNamespace(thermalization_time_s=1.86e-3)
        exp = SimpleNamespace(
            params=SimpleNamespace(targets=["q1"], num_averages=4000),
            device=SimpleNamespace(channel=lambda t, kind: chan),
            sweep_axes={"detuning_hz": np.arange(100)},
        )
        t = _run_timeout_s(exp)
        # 4000 x 100 x (1.86 ms + 1 ms margin) x 2 safety = 2288 s, up-rounded
        assert t >= 2 * 4000 * 100 * 1.86e-3
        assert t % 60 == 0

    def test_unestimable_falls_back_to_the_floor(self):
        """No shot periods, no repetition count -> the floor, never a crash: a
        wrong timeout estimate must never be the thing that kills a run."""
        from types import SimpleNamespace

        from lchqb.backend.qblox_backend import _RUN_TIMEOUT_S, _run_timeout_s

        assert _run_timeout_s(SimpleNamespace(params=SimpleNamespace())) == _RUN_TIMEOUT_S
        assert _run_timeout_s(SimpleNamespace(params=None)) == _RUN_TIMEOUT_S

    def test_short_estimates_stay_on_the_floor(self):
        """A quick experiment must not LOWER the deadline below the floor."""
        from types import SimpleNamespace

        from lchqb.backend.qblox_backend import _RUN_TIMEOUT_S, _run_timeout_s

        chan = SimpleNamespace(thermalization_time_s=200e-6)
        exp = SimpleNamespace(
            params=SimpleNamespace(targets=["q1"], num_averages=100),
            device=SimpleNamespace(channel=lambda t, kind: chan),
            sweep_axes={"amp_prefactor": [0.9, 1.0, 1.1]},
        )
        assert _run_timeout_s(exp) == _RUN_TIMEOUT_S

    def test_acquire_wires_the_computed_deadline(self, tmp_path, roster, monkeypatch):
        """acquire() hands the computed value to HardwareAgent.run AND mirrors
        it onto the instrument coordinator's own timeout parameter — the second
        gate ``retrieve_acquisition()`` re-waits against (vendor default 60 s)."""
        pytest.importorskip("qblox_scheduler")
        from conftest import make_backend, make_experiment
        from scqo.experiments import get

        from lchqb.backend.qblox_backend import QbloxBackend

        backend = make_backend(tmp_path, roster)
        cls = get("resonator_spectroscopy")
        exp = make_experiment(cls, backend, roster,
                              cls.Parameters(targets=["q1"], num_points=5))
        exp.sweep_axes = exp.define_sweep()

        seen = {}

        def fake_run(schedule, timeout=None):
            seen["timeout"] = timeout
            return object()

        monkeypatch.setattr(backend._hw_agent, "run", fake_run)
        monkeypatch.setattr(QbloxBackend, "_to_canonical",
                            staticmethod(lambda raw, experiment: raw))
        backend.acquire(exp)
        assert seen["timeout"] >= 300 and seen["timeout"] % 60 == 0
        assert backend._hw_agent.instrument_coordinator.timeout() == seen["timeout"]
