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

    def _amp_field(self) -> str:
        """The knob this run benchmarks, refusing the ones Qblox cannot offer.

        Qblox DERIVES X90 from ``rxy.amp180`` (amp180*theta/180), so there is no
        independent pi/2 amplitude to sweep OR to write back -- ``pi_amp_x90``
        is Unrealized here (see backend/fieldmap.py). Without this the sweep
        would run against the pi amplitude and the write-back would land on a
        knob nothing played.
        """
        field = self.amp_reference_field()
        if field != "pi_amp":
            raise NotImplementedError(
                f"{self.name}: target_gate={self.params.target_gate!r} calibrates "
                f"{field}, which is Unrealized on the Qblox backend -- X90 is "
                f"DERIVED from rxy.amp180 here, so there is no independent pi/2 "
                f"amplitude to sweep or write. Benchmark a pi gate "
                f"(target_gate=x180 or y180), or run this on the QM backend")
        return field

    def define_sweep(self):
        # refuse BEFORE the neutral layer reaches for the unrealized anchor,
        # so the operator gets the reason instead of "has no value yet"
        self._amp_field()
        return super().define_sweep()

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        amp_factors = self.sweep_axes["amp_prefactor"]
        repetitions = [int(r) for r in self.sweep_axes["repetitions"]]
        reps = int(self.params.num_averages)
        target_gate = getattr(self.params, "target_gate", "x180")
        theta, phi = _gate_to_angles(target_gate)

        # The knob the benchmarked gate calibrates -- scqo decides in ONE place
        # (amp_reference_field -> amp_knob(target_gate)) so the amplitude we
        # PLAY, the absolute axis estimate() attaches and the knob update()
        # writes can never disagree. Hard-coding pi_amp here is exactly the
        # x180/x90 mismatch issue #24 reported on the QM side.
        amp_field = self._amp_field()

        schedule = Schedule("deterministic_benchmarking_multiplexed")
        for qubit_name in self.params.targets:
            base_amp = float(getattr(self.device.channel(qubit_name, "drive"),
                                     amp_field))
            amp_abs = check_amp_window(
                amp_factors, base_amp, target=qubit_name, field=amp_field
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
