"""Qblox readout-frequency scan — supplies only ``probe()``.

Per readout detuning the readout frequency is set via ``Measure(freq=...)``
(detuning relative to the CURRENT ``readout_freq_hz``). State preparation follows the
cal13 dispersive-shift reference (measure |0>, then Reset -> X -> measure |1>),
arranged as two sequential prepared-state blocks per frequency.

BOTH readout modes come off the same schedule, and the ONE difference is whether
the repetition loop's variable is CAPTURED into the bin coords:

* ``readout_mode="shot"`` — ``coords={... "shot_<q>": shot}`` labels every
  repetition, so the cluster appends one I/Q point per shot instead of averaging
  (the single_shot_readout mechanism). Flat bin order is frequency-major, then
  state, then shot: ``detuning_hz``, ``prepared_state``, ``shot_idx``.
* ``readout_mode="average"`` — the same loop runs unlabeled, so every repetition
  lands in the SAME bin and the cluster averages it (the resonator_spectroscopy
  idiom). No ``shot_idx`` at all, which is the form scqo's contract accepts as
  its alt set.

Same goal as cal13's averaged chi scan; in shot mode it is measured by the
criterion that matters directly, assignment fidelity. Parameters and the analysis
are inherited from ``scqo.experiments.ReadoutFrequency``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ReadoutFrequency

from ._reset import add_reset


@register
class QbloxReadoutFrequency(ReadoutFrequency):
    """Build a multiplexed readout-frequency Schedule for a Qblox cluster."""

    # No supports_active_reset: this probe SWEEPS the readout frequency, so the
    # discriminator single_shot_readout solved at the nominal tone is wrong at
    # almost every point. _reset.py refuses reset_method='active' by name.

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes["detuning_hz"]
        num_shots = int(self.params.num_shots)
        per_shot = self.params.readout_mode == "shot"

        def shot_coord(qubit: str, shot) -> dict:
            """The shot label, and ONLY in shot mode: an unlabeled repetition
            loop is what makes the cluster average into one bin."""
            return {f"shot_{qubit}": shot} if per_shot else {}

        schedule = Schedule("readout_frequency_multiplexed")
        for qubit_name in self.params.targets:
            # detuning is relative to the CURRENT readout_freq_hz (on q<n>_ro)
            center = self.device.channel(qubit_name, "readout").readout_freq_hz
            sub = Schedule(f"readout_frequency_{qubit_name}")
            with sub.loop(
                linspace(
                    center + float(detuning[0]),
                    center + float(detuning[-1]),
                    detuning.size,
                    dtype=DType.FREQUENCY,
                )
            ) as freq:
                # prepared_state 0: Reset -> Measure, one bin per shot (shot
                # mode) or one averaged bin (average mode)
                with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                    add_reset(sub, self, qubit_name)
                    sub.add(
                        Measure(
                            qubit_name,
                            freq=freq,
                            coords={
                                f"frequency_{qubit_name}": freq,
                                f"state_{qubit_name}": 0,
                                **shot_coord(qubit_name, shot),
                            },
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    # NOT optional, and NOT the depletion wait (Reset already is
                    # one): Measure(freq=...) appends a zero-duration clock
                    # RESTORE, whose latched UpdateParameters may not land on
                    # the loop's ControlFlowReturn. 4 ns = the compiler's
                    # MIN_TIME_BETWEEN_OPERATIONS, so this costs nothing per
                    # shot. Without it, chipA 2026-07-26: "Parameter operation
                    # ... cannot be scheduled exactly before ... ControlFlowReturn".
                    sub.add(IdlePulse(4e-9))
                # prepared_state 1: Reset -> X -> Measure, same binning rule
                with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                    add_reset(sub, self, qubit_name)
                    sub.add(X(qubit=qubit_name))
                    sub.add(
                        Measure(
                            qubit_name,
                            freq=freq,
                            coords={
                                f"frequency_{qubit_name}": freq,
                                f"state_{qubit_name}": 1,
                                **shot_coord(qubit_name, shot),
                            },
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    sub.add(IdlePulse(4e-9))  # same clock-restore reason as above
            # and once more OUTSIDE the loops — a different check: a parameter
            # op may not sit at the very end of a TimeableSchedule either
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
