"""Qblox qubit relaxation (T1) — supplies only ``probe()``.

cal14 reference pattern: Reset -> X (pi pulse) -> Measure delayed by the swept wait
(``rel_time=tau``). Parameters, the exponential fit and reporting are inherited from
``scqo.experiments.QubitRelaxation``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitRelaxation

from scqo_qblox.experiments._reset import add_reset
from scqo_qblox.experiments._state import measure_kwargs


@register
class QbloxQubitRelaxation(QubitRelaxation):
    """Build a multiplexed T1 Schedule for a Qblox cluster."""

    #: readout is held at the calibrated point for the whole run and the Reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        wait_ns = self.sweep_axes["wait_time_ns"]
        reps = self.params.num_averages

        schedule = Schedule("qubit_relaxation_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            sub = Schedule(f"t1_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(
                    linspace(wait_ns[0] * 1e-9, wait_ns[-1] * 1e-9, wait_ns.size, dtype=DType.TIME)
                ) as tau:
                    add_reset(sub, self, qubit_name)
                    sub.add(X(qubit=qubit_name))
                    sub.add(
                        Measure(
                            qubit_name,
                            coords={f"tau_{qubit_name}": tau},
                            acq_channel=f"S_21_{qubit_name}",
                            **acq,
                        ),
                        rel_time=tau,  # the decay wait: measure tau after the pi pulse
                    )
                    sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
