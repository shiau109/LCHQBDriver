"""Qblox chain-stepped absolute punchout — supplies only ``probe()``.

Parameters, the per-point chain-stepping run lifecycle (python loop: re-solve
output_att + pulse_amp per power point, one 1D acquisition each,
boundary-recorded set/revert), the punchout analysis and the
readout_power_dbm/readout_freq_hz proposals are inherited from
``scqo.experiments.ResonatorSpectroscopyPowerChain``.

By the time ``probe()`` runs, the core loop has already written THIS point's
``output_att`` into the hardware config (re-applied at every run's prepare) and
the residual into ``measure.pulse_amp`` (~0.5 full scale) — so the probe is the
plain 1D resonator-spectroscopy schedule measured at the element's CURRENT
amplitude, with the 10 µs IdlePulse between measures as the depletion wait.
``self.sweep_axes`` holds only the detuning axis during each per-point call
(the run loop swaps it in). The power axis is the core's uniform dB grid —
achieved exactly per point by the chain solve, no hardware amplitude loop at all.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ResonatorSpectroscopyPowerChain


@register
class QbloxResonatorSpectroscopyPowerChain(ResonatorSpectroscopyPowerChain):
    """Per-point 1D resonator-spectroscopy Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes["detuning_hz"]
        span = float(detuning[-1] - detuning[0])
        num = self.params.num_freq_points
        reps = self.params.num_averages

        schedule = Schedule("resonator_spectroscopy_power_chain_point")
        for qubit_name in self.params.targets:
            center = self.device.channel(qubit_name, "readout").readout_freq_hz
            sub = Schedule(f"punchout_point_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(
                    linspace(center - span / 2, center + span / 2, num, dtype=DType.FREQUENCY)
                ) as freq:
                    # measured at the element's CURRENT pulse_amp (the chain solve
                    # set it); the IdlePulse is the resonator depletion wait
                    sub.add(
                        Measure(
                            qubit_name,
                            freq=freq,
                            coords={f"frequency_{qubit_name}": freq},
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    sub.add(IdlePulse(10e-6))
            schedule.add(sub)
        return schedule
