"""The Qblox DISCRETE parity monitor: two measurements per cycle, one channel.

The discrete variant's defining properties, asserted on the COMPILED tree
(reusing the continuous test module's helpers):

* still no reset of any kind — M1's projection IS the initialization;
* exactly TWO acquisitions per cycle, BOTH on ``S_21_<q>`` (a second acq
  channel would compile clean and be silently dropped by ``_to_canonical``);
* the reference protocol's pulse order (x90 first, y90 second), pinned on the
  compiled PHASES — on this backend ``X90(q, phase=...)`` silently drops a
  phase kwarg, so gate names prove nothing;
* the pad math: ``cycle_period_ns`` is the telegraph timebase, hit exactly,
  refused by name when shorter than the sequence;
* the bin guard counts TWO bins per cycle against the 3e6 sequencer limit.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe  # noqa: E402

from test_parity_switch_continuous import (  # noqa: E402
    DEPLETION_S,
    IDLE_NS,
    _drive_phases,
    _experiment,
    _leaves,
)

from scqo_qblox.experiments.qubit_parity_switch_discrete import (  # noqa: E402
    QbloxQubitParitySwitchDiscrete,
)


def _discrete(tmp_path, roster, **kwargs):
    return _experiment(tmp_path, roster, cls=QbloxQubitParitySwitchDiscrete,
                       **kwargs)


def test_schedule_contains_no_reset_of_any_kind(tmp_path, roster):
    """M1 replaces the reset; nothing may add one back."""
    backend, exp = _discrete(tmp_path, roster)
    names = " ".join(str(op) for op in _leaves(compile_probe(backend, exp)))
    assert "Reset(" not in names
    assert "ConditionalReset" not in names
    # guard: the walker really did reach the loop body it is asserting over
    assert "X90" in names, names


def test_two_measures_per_cycle_on_one_channel(tmp_path, roster):
    """Both measurements are labeled bins on the SAME S_21 channel. A second
    acq channel would compile clean and then be silently dropped by
    _to_canonical (it reads only S_21_{name}) — the known active-reset gap this
    probe must not walk into. Asserted on the schedule's gate_info (the
    test_active_reset pattern); compile-cleanliness is covered by
    test_probe_compiles and the state-mode test below."""
    backend, exp = _discrete(tmp_path, roster)
    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()
    gates = [(getattr(op, "data", {}) or {}).get("gate_info") or {}
             for op in _leaves(schedule)]
    measures = [g for g in gates if g.get("operation_type") == "measure"]
    assert len(measures) == 2, measures             # per loop iteration
    channels = {m.get("acq_channel_override") for m in measures}
    assert channels == {"S_21_q1"}, channels


def test_the_two_pulses_are_90_degrees_apart_on_hardware(tmp_path, roster):
    """x90 FIRST here (the reference protocol's order; the continuous variant
    plays y90 first) — still a 90-degree-shifted pair, so still sin, and the
    swap only flips the telegraph's sign, invisible to the PSD. Asserted on
    the compiled phases, never gate names."""
    backend, exp = _discrete(tmp_path, roster)
    phases = _drive_phases(compile_probe(backend, exp))
    assert len(phases) == 2, phases
    assert phases[0] == pytest.approx(0.0)   # x90 first
    assert phases[1] == pytest.approx(90.0)  # y90 second


def test_shot_period_matches_the_scheduled_operations(tmp_path, roster):
    """The telegraph timebase, with TWO readouts in the floor."""
    backend, exp = _discrete(tmp_path, roster)
    compile_probe(backend, exp)
    period = exp.probe_shot_period_s["q1"]
    floor = IDLE_NS * 1e-9 + DEPLETION_S
    assert period > floor  # + two pi/2 pulses + TWO readouts
    # the discrete minimal cycle carries one more readout than continuous
    cont_dir = tmp_path / "cont"
    cont_dir.mkdir()
    cont_backend, cont = _experiment(cont_dir, roster)
    compile_probe(cont_backend, cont)
    readout_gap = period - cont.probe_shot_period_s["q1"]
    assert readout_gap > 0
    assert readout_gap < 50e-6


def test_cycle_period_pads_tau_wait(tmp_path, roster):
    """cycle_period_ns IS the reported period: the pad absorbs the difference
    exactly (1 ns grid, no floor on this backend)."""
    backend, exp = _discrete(tmp_path, roster, cycle_period_ns=50000.0)
    compile_probe(backend, exp)
    assert exp.probe_shot_period_s["q1"] == pytest.approx(50e-6)


def test_cycle_period_shorter_than_sequence_refused_by_name(tmp_path, roster):
    backend, exp = _discrete(tmp_path, roster, cycle_period_ns=3000.0)
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="cycle_period_ns"):
        exp.probe()


def test_bin_ceiling_counts_two_bins_per_cycle(tmp_path, roster):
    """1.6e6 cycles would clear the continuous guard (1.6e6 < 3e6 bins) but
    needs 3.2e6 bins here — two per cycle — so it must refuse, and say why."""
    backend, exp = _discrete(tmp_path, roster)
    exp.params = exp.params.model_copy(update={"num_shots": 1_600_000})
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="TWO per cycle"):
        exp.probe()


def test_idle_comes_from_the_stored_splitting(tmp_path, roster):
    """The inherited derivation: same 1/(2 x parity_delta_f_hz) as continuous."""
    backend, exp = _discrete(tmp_path, roster)
    exp.sweep_axes = exp.define_sweep()
    assert exp.resolved_idle_ns("q1") == pytest.approx(IDLE_NS)


def test_refuses_without_the_stored_splitting(tmp_path, roster):
    backend, exp = _discrete(tmp_path, roster, splitting=False)
    with pytest.raises(ValueError, match="parity_delta_f_hz"):
        exp.define_sweep()


def test_refuses_without_a_governed_depletion_wait(tmp_path, roster):
    backend, exp = _discrete(tmp_path, roster, depletion=False)
    with pytest.raises(ValueError, match="readout_depletion_s"):
        exp.define_sweep()


def test_state_mode_compiles_with_a_discriminator(tmp_path, roster):
    backend, exp = _discrete(tmp_path, roster, discriminator=True,
                             use_state_discrimination=True)
    compiled = compile_probe(backend, exp)
    assert compiled.operations
