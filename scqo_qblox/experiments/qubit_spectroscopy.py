"""Qblox qubit spectroscopy — supplies only ``probe()``.

Two-tone: a continuous weak microwave drive (voltage offset on the drive port, cal05
reference pattern) while the drive-clock NCO steps through the sweep; measure after
the qubit reaches its driven steady state. Parameters, peak fitting and the
drive_freq_hz writeback are inherited from ``scqo.experiments.QubitSpectroscopy``.

Drive power contract: the core ``run()`` already solved the drive chain for
``drive_power_dbm`` (recorded set -> acquire -> revert), parking the residual on
``element.spec.spec_amp`` — so the probe plays each qubit's OWN solved amplitude
(``view.drive_amp``) as the VoltageOffset value.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitSpectroscopy

from ._reset import add_reset
from ._vendor import vendor_element


@register
class QbloxQubitSpectroscopy(QubitSpectroscopy):
    """Build a multiplexed two-tone spectroscopy Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            VoltageOffset,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes["detuning_hz"]
        reps = self.params.num_averages

        schedule = Schedule("qubit_spectroscopy_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "drive")
            element = vendor_element(self, qubit_name, "drive")  # ports: vendor-only
            center = float(view.drive_freq_hz)
            drive_amp = float(view.drive_amp)  # run() parked the solved residual here
            drive_clock = f"{qubit_name}.01"
            mw_port = element.ports.microwave
            mw_port = mw_port() if callable(mw_port) else mw_port
            sub = Schedule(f"qubit_spec_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                # endpoint form, never center +/- span/2: the scqo window is an
                # explicit [start, end] relative to drive_freq_hz and may be
                # ASYMMETRIC — re-deriving a symmetric span would silently
                # re-center it (the flux_pulse probe set this pattern)
                with sub.loop(
                    linspace(
                        center + float(detuning[0]),
                        center + float(detuning[-1]),
                        detuning.size,
                        dtype=DType.FREQUENCY,
                    )
                ) as freq:
                    # continuous weak drive on the microwave port (cal05 pattern)
                    sub.add(VoltageOffset(drive_amp, 0, port=mw_port, clock=drive_clock))
                    sub.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                    # let the qubit reach the driven steady state before measuring.
                    # This Reset is the DRIVEN DWELL, not a state reset — the CW
                    # drive above is still on — which is why _reset.py refuses
                    # reset_method='active' here (no supports_active_reset).
                    add_reset(sub, self, qubit_name)
                    sub.add(
                        Measure(
                            qubit_name,
                            coords={f"frequency_{qubit_name}": freq},
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    sub.add(IdlePulse(4e-9))
            # drive OFF before the next qubit / end of schedule
            sub.add(VoltageOffset(0, 0, port=mw_port, clock=drive_clock))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
