"""Qblox Deterministic Benchmarking acquisition probe.

Repeats a specified target gate (x180, y180, x90, y90, -x90, -y90) N times
across an array of repetition counts N and amplitude scaling factors to observe
gate rotation error accumulation.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitDeterministicBenchmarking``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitDeterministicBenchmarking

from ._amp_limits import check_amp_window
from ._reset import add_reset
from ._state import measure_kwargs


def _gate_to_angles(gate: str) -> tuple[float, float]:
    """Map target gate name to (theta_deg, phi_deg) for Rxy."""
    g = str(gate).strip().lower()
    if g in ("x180", "x", "pi", "x_pi"):
        return 180.0, 0.0
    if g in ("y180", "y", "y_pi"):
        return 180.0, 90.0
    if g in ("x90", "pi_half", "x_half"):
        return 90.0, 0.0
    if g in ("y90", "y_half"):
        return 90.0, 90.0
    if g in ("-x90", "minus_x90", "-x_half"):
        return 90.0, 180.0
    if g in ("-y90", "minus_y90", "-y_half"):
        return 90.0, 270.0
    raise ValueError(
        f"unknown target_gate {gate!r}; expected x180, y180, x90, y90, -x90, -y90"
    )


@register
class QbloxQubitDeterministicBenchmarking(QubitDeterministicBenchmarking):
    """Build a multiplexed Deterministic Benchmarking Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        amp_factors = self.sweep_axes["amp_prefactor"]
        repetitions = [int(r) for r in self.sweep_axes["repetitions"]]
        reps = int(self.params.num_averages)
        target_gate = getattr(self.params, "target_gate", "x180")
        theta, phi = _gate_to_angles(target_gate)

        schedule = Schedule("deterministic_benchmarking_multiplexed")
        for qubit_name in self.params.targets:
            pi_amp = float(self.device.channel(qubit_name, "drive").pi_amp)
            amp_abs = check_amp_window(
                amp_factors, pi_amp, target=qubit_name, field="pi_amp"
            )
            acq = measure_kwargs(self, qubit_name)
            sub = Schedule(f"deterministic_benchmarking_{qubit_name}")

            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(
                    linspace(
                        float(amp_abs[0]),
                        float(amp_abs[-1]),
                        amp_abs.size,
                        dtype=DType.AMPLITUDE,
                    )
                ) as amp:
                    for rep in repetitions:
                        add_reset(sub, self, qubit_name)
                        for _ in range(rep):
                            sub.add(
                                Rxy(
                                    theta=theta,
                                    phi=phi,
                                    qubit=qubit_name,
                                    amp180=amp,
                                )
                            )
                        sub.add(
                            Measure(
                                qubit_name,
                                coords={
                                    f"amp_{qubit_name}": amp,
                                    f"rep_{qubit_name}": rep,
                                },
                                acq_channel=f"S_21_{qubit_name}",
                                **acq,
                            )
                        )
                        sub.add(IdlePulse(4e-9))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)

        return schedule
