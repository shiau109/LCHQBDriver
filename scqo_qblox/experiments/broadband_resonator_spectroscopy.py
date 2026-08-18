"""Qblox broadband resonator spectroscopy — supplies only ``probe()``.

Parameters, fitting, simulation are inherited from
``scqo.experiments.BroadbandResonatorSpectroscopy``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scqo import register
from scqo.experiments import BroadbandResonatorSpectroscopy


@register
class QbloxBroadbandResonatorSpectroscopy(BroadbandResonatorSpectroscopy):
    """Build a wideband resonator-spectroscopy Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        start = float(self.params.start_freq_hz)
        stop = float(self.params.stop_freq_hz)
        bw = float(self.params.bandwidth_per_lo_hz)
        pts_per_lo = int(self.params.num_points_per_lo)
        gap = float(self.params.lo_gap_hz)
        reps = self.params.num_averages

        lo_step = bw
        n_segments = max(1, int(np.ceil((stop - start) / lo_step)))
        lo_centers = [start + (i + 0.5) * lo_step for i in range(n_segments)]

        schedule = Schedule("broadband_resonator_spectroscopy")
        # Measure once across the feedline using the primary target
        primary_qubit = self.params.targets[0]
        sub = Schedule(f"broadband_res_spec_{primary_qubit}")
        with sub.loop(arange(0, reps, 1, DType.NUMBER)):
            for lo in lo_centers:
                sub_min = max(start, lo - bw / 2.0)
                sub_max = min(stop, lo + bw / 2.0)
                if sub_max <= sub_min:
                    continue
                if gap > 0 and sub_min < lo < sub_max:
                    n_half = max(2, pts_per_lo // 2)
                    with sub.loop(
                        linspace(sub_min, lo - gap / 2.0, n_half, dtype=DType.FREQUENCY)
                    ) as freq:
                        sub.add(
                            Measure(
                                primary_qubit,
                                freq=freq,
                                coords={f"frequency_{primary_qubit}": freq},
                                acq_channel=f"S_21_{primary_qubit}",
                            )
                        )
                        sub.add(IdlePulse(10e-6))
                    with sub.loop(
                        linspace(lo + gap / 2.0, sub_max, n_half, dtype=DType.FREQUENCY)
                    ) as freq:
                        sub.add(
                            Measure(
                                primary_qubit,
                                freq=freq,
                                coords={f"frequency_{primary_qubit}": freq},
                                acq_channel=f"S_21_{primary_qubit}",
                            )
                        )
                        sub.add(IdlePulse(10e-6))
                else:
                    with sub.loop(
                        linspace(sub_min, sub_max, pts_per_lo, dtype=DType.FREQUENCY)
                    ) as freq:
                        sub.add(
                            Measure(
                                primary_qubit,
                                freq=freq,
                                coords={f"frequency_{primary_qubit}": freq},
                                acq_channel=f"S_21_{primary_qubit}",
                            )
                        )
                        sub.add(IdlePulse(10e-6))
        schedule.add(sub)
        return schedule
