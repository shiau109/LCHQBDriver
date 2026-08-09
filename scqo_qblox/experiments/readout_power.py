"""Qblox per-shot readout-amplitude scan — supplies only ``probe()``.

Per amplitude prefactor the readout pulse amplitude is set via
``Measure(pulse_amp=...)``, prefactor x the CURRENT ``readout_amp`` (read the same
way resonator_spectroscopy_power_amp's punchout probe reads it). Each prefactor runs
two sequential prepared-state blocks (|0>: Reset -> Measure, |1>: Reset -> X ->
Measure) with the shot loop variable CAPTURED into labeled coords — the
single_shot_readout per-shot mechanism, so the cluster appends one I/Q point per
shot instead of averaging. Flat bin order is amp-major, then state, then shot,
matching the canonical sweep axes (``amp_prefactor``, ``prepared_state``,
``shot_idx``). No reps loop / no ``num_averages`` by design.

Method note: QBLOX_training's cal17 calibrates readout amplitude via the AC-Stark
shift of the qubit frequency — a different (averaged) method deliberately NOT
followed here; this probe is the per-shot fidelity method matching the QM backend.
Parameters and the fidelity-vs-amplitude fit are inherited from
``scqo.experiments.ReadoutPower``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ReadoutPower

from ._amp_limits import check_amp_window
from ._reset import add_reset


@register
class QbloxReadoutPower(ReadoutPower):
    """Build a multiplexed per-shot readout-amplitude Schedule for a Qblox cluster."""

    # No supports_active_reset: this probe SWEEPS the readout amplitude, so the
    # discriminator single_shot_readout solved at the nominal power is wrong at
    # almost every point. _reset.py refuses reset_method='active' by name.

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        prefactors = self.sweep_axes["amp_prefactor"]
        num_shots = int(self.params.num_shots)

        schedule = Schedule("readout_power_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "readout")
            # the prefactor scales the CURRENT readout amplitude (punchout pattern),
            # validated against the DAC here so an over-range window is refused by
            # name instead of dying inside the compiler on `awg_gain`
            amp_abs = check_amp_window(prefactors, view.readout_amp,
                                       target=qubit_name, field="readout_amp")
            amp_lo, amp_hi = float(amp_abs[0]), float(amp_abs[-1])
            sub = Schedule(f"readout_power_{qubit_name}")
            with sub.loop(
                linspace(amp_lo, amp_hi, prefactors.size, dtype=DType.AMPLITUDE)
            ) as amp:
                # prepared_state 0: Reset -> Measure, one labeled bin per shot
                with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                    add_reset(sub, self, qubit_name)
                    sub.add(
                        Measure(
                            qubit_name,
                            pulse_amp=amp,
                            coords={
                                f"amp_{qubit_name}": amp,
                                f"state_{qubit_name}": 0,
                                f"shot_{qubit_name}": shot,
                            },
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                # prepared_state 1: Reset -> X -> Measure, one labeled bin per shot
                with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                    add_reset(sub, self, qubit_name)
                    sub.add(X(qubit=qubit_name))
                    sub.add(
                        Measure(
                            qubit_name,
                            pulse_amp=amp,
                            coords={
                                f"amp_{qubit_name}": amp,
                                f"state_{qubit_name}": 1,
                                f"shot_{qubit_name}": shot,
                            },
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
