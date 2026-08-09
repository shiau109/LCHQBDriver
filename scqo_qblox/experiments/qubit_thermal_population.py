"""Qblox residual-population measurement (|g>-only IQ cloud) — supplies only ``probe()``.

The single_shot_readout probe with the |1> block removed: reset, measure, one
labeled bin per shot, no X gate anywhere. Same per-shot mechanism (the loop
variable is CAPTURED into the acquisition ``coords``, so every shot is a distinct
bin and the cluster appends instead of averaging), and the ``prepared_state``
coord is still written — as the constant 0 — so the flat bin order matches the
canonical (``prepared_state``, ``shot_idx``) axes ``_to_canonical`` reshapes by.

Parameters, the pinned-center fit against the readout channel's stored blob
centers, and the ``n_th`` writeback are inherited from
``scqo.experiments.QubitThermalPopulation``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitThermalPopulation

from ._reset import add_reset


@register
class QbloxQubitThermalPopulation(QubitThermalPopulation):
    """Build a multiplexed per-shot |g>-only Schedule for a Qblox cluster."""

    # No supports_active_reset: a conditional pi drives the qubit toward |g>,
    # which is precisely the population being counted. The neutral Parameters
    # already refuse reset_method='active' at construction; this is the second
    # lock, so the refusal holds even if that one is ever relaxed.

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure
        from qblox_scheduler.operations.loop_domains import DType, arange

        num_shots = int(self.params.num_shots)

        schedule = Schedule("thermal_population_multiplexed")
        for qubit_name in self.params.targets:
            sub = Schedule(f"thermal_{qubit_name}")
            # prepared_state 0 only: Reset -> Measure, one labeled bin per shot
            with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                add_reset(sub, self, qubit_name)
                sub.add(
                    Measure(
                        qubit_name,
                        coords={f"state_{qubit_name}": 0, f"shot_{qubit_name}": shot},
                        acq_channel=f"S_21_{qubit_name}",
                    )
                )
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
