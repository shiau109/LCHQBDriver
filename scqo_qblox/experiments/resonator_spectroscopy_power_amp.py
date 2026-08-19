"""Qblox resonator spectroscopy vs ABSOLUTE power via the AMPLITUDE sweep — supplies
only ``probe()`` (the power is swept by changing the digital amplitude; the core
``run()`` already solved the chain for the window top ``max_power_dbm``, so the
element's CURRENT ``readout_amp`` IS the top amplitude and the sweep descends from
it).

cal03 reference pattern for the readout pulse amplitude (``Measure(pulse_amp=...)``),
but the amplitude axis is **Python-unrolled**: the scheduler's loop domains are
LINEAR-only, and a uniform-dBm power grid needs GEOMETRIC amplitudes — so instead of
one hardware amplitude loop the schedule contains one block per power point, each
carrying its exact amplitude as a literal ``pulse_amp`` + ``amp_<q>`` coord (the
cluster bins by coord VALUE, so literals work like loop variables — the same
mechanism readout_power.py uses for its literal ``state`` coord). Axis identical to
QM's, uniform in dBm, endpoints exact; still ONE compile+run cycle (the sequencer
program grows ~N_power×, a hardware prove-out item at large point counts).

Loop order (2026-07-14, user-decided; the QM probe matches): amplitude (outer,
unrolled ascending — only upward power jumps between slow outer blocks, ring-down
safe) -> averages (middle) -> frequency (INNER = fastest) — each power point repeats
a fast frequency sweep ``num_averages`` times; same-coords bins average on the
cluster across the middle repetition loop. The acquired axis order is
(power, detuning). The between-readout IdlePulse (ring-down wait) is the governed
depletion wait: the per-run ``readout_depletion_ns`` override, else the readout
channel's calibrated ``readout_depletion_s`` knob, else the probe's built-in 4 ns
— which is likely far too short for a real resonator (the plain res-spec idles
10 µs), so run ``resonator_spectroscopy`` first on a cold chip.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import ResonatorSpectroscopyPowerAmp
from scqo.experiments._depletion import depletion_wait_ns

from scqo_qblox.backend.qblox_backend import snap_ns

#: Between-readout idle when NOTHING governs the wait — no per-run
#: readout_depletion_ns and no calibrated readout_depletion_s knob. The probe's
#: historical value, deliberately unchanged: this is a BRING-UP sweep, often the
#: first thing run on a cold chip, so it must not require a calibration it is
#: being run to inform. (Active reset takes the opposite policy and refuses —
#: see experiments/_reset.py for why the shared helper returns None rather than
#: raising.)
_DEFAULT_IDLE_S = 4e-9


@register
class QbloxResonatorSpectroscopyPowerAmp(ResonatorSpectroscopyPowerAmp):
    """Build a multiplexed punchout Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes["detuning_hz"]
        # geometric prefactors realizing the core's UNIFORM dBm grid (top exactly 1.0)
        power_dbm = np.asarray(self.sweep_axes["power_dbm"])
        amps_rel = 10.0 ** ((power_dbm - self.params.max_power_dbm) / 20.0)
        reps = self.params.num_averages
        # Per-run override, else the calibrated readout_depletion_s knob, else
        # the probe's own fallback — resolved by scqo's ONE precedence helper so
        # this and active reset cannot disagree about what the wait is.
        # snapped onto the scheduler's 1 ns grid (and / 1e9, never * 1e-9, so the
        # float equals the parsed literal): a free user float in ns would
        # otherwise compile into an off-grid IdlePulse and fail with a raw
        # vendor traceback pointing at an absolute timestamp.
        relax_ns = depletion_wait_ns(self, self.params.targets[0])
        idle = snap_ns(relax_ns / 1e9, "readout_depletion_ns") if relax_ns \
            else _DEFAULT_IDLE_S

        schedule = Schedule("resonator_spectroscopy_power_amp_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "readout")
            center = view.readout_freq_hz
            # run() solved the chain for max_power_dbm, so readout_amp IS the top
            # amplitude (<= 0.5 by the setter policy — DAC-safe by construction)
            amps = amps_rel * float(view.readout_amp)
            sub = Schedule(f"punchout_{qubit_name}")
            # amp OUTER (unrolled, ascending) -> averages MIDDLE -> freq INNER
            # (fastest): the flat bin order is (power, detuning), matching the
            # canonical (power_dbm, detuning_hz); repetitions of one power point
            # average into the same coords-keyed bins on the cluster. The resonator
            # only jumps power between the slow outer blocks.
            for amp_val in amps:
                amp_val = float(amp_val)
                with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                    # endpoint form, never center +/- span/2: the scqo window is
                    # an explicit [start, end] relative to readout_freq_hz and
                    # may be ASYMMETRIC — a punchout walks the dip DOWN toward
                    # the bare resonator, so re-deriving a symmetric span would
                    # silently re-center exactly the sweep that needs it
                    with sub.loop(
                        linspace(
                            center + float(detuning[0]),
                            center + float(detuning[-1]),
                            detuning.size,
                            dtype=DType.FREQUENCY,
                        )
                    ) as freq:
                        sub.add(
                            Measure(
                                qubit_name,
                                freq=freq,
                                pulse_amp=amp_val,
                                coords={f"frequency_{qubit_name}": freq, f"amp_{qubit_name}": amp_val},
                                acq_channel=f"S_21_{qubit_name}",
                            )
                        )
                        sub.add(IdlePulse(idle))
            schedule.add(sub)
        return schedule
