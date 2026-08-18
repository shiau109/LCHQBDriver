"""Qblox concurrent-tone qubit spectroscopy — supplies only ``probe()``.

The two-tone sequence of :mod:`.qubit_spectroscopy`, rearranged so the weak
saturation drive and the readout tone go up TOGETHER and the acquisition opens
only after both have been on for ``acq_start_ns``. Parameters, peak fitting and
the ``drive_freq_hz`` writeback are inherited from
``scqo.experiments.QubitSpectroscopyOverlap``; the window arithmetic is scqo's
shared ``_overlap.overlap_windows``, so this probe and the QM one cannot drift
apart on what the same Parameters mean.

Three things carry the timing, and each is load-bearing:

1. **The co-start needs no reference.** ``VoltageOffset`` is a zero-duration
   latched parameter op, so under the default ASAP chaining the ``Measure``
   added straight after it begins at the same instant the drive latches. That
   is the cal05 idiom of the sibling probe, just tightened into a guarantee.
2. **The ADC lead is two ``Measure`` device overrides.** ``pulse_duration``
   grows by ``acq_start_ns`` and ``acq_delay`` becomes ``TOF + acq_start_ns``;
   ``_dispersive_measurement`` anchors the readout ``SquarePulse`` at ``t0=0``
   and ``SSBIntegrationComplex`` at ``t0=acq_delay``, both ``ref_pt="start"``,
   which is exactly "tones first, ADC later". These are per-run STIMULUS, not
   state: the ``readout_duration_s`` knob is never written.
3. **The trailing idle re-anchors the ASAP chain.** Without it the chain would
   continue from the zero-duration drive-OFF, which for a ``drive_len_ns``
   shorter than the tone sits BEFORE the readout pulse ends — and the next loop
   iteration would start inside the previous measurement.

Drive power contract: unchanged from the sibling probe — the core ``run()``
already solved the drive chain for ``drive_power_dbm`` (recorded set -> acquire
-> revert), parking the residual on ``element.spec.spec_amp``, so the probe
plays each qubit's OWN solved amplitude (``view.drive_amp``).
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitSpectroscopyOverlap
from scqo.experiments._overlap import overlap_windows

from ._reset import add_reset
from ._vendor import vendor_element


def _acq_delay_s(element: Any) -> float:
    """The element's standing time-of-flight, seconds.

    Vendor-only by design (``fieldmap.VENDOR_ONLY["readout_acq_delay"]``): it
    aligns the instrument's receive path with its own transmit path, so the
    probe ADDS its lead to it and never replaces it. Tolerates both scheduler
    API generations (legacy QCoDeS callables, pydantic plain attributes).
    """
    attr = element.measure.acq_delay
    return float(attr() if callable(attr) else attr)


@register
class QbloxQubitSpectroscopyOverlap(QubitSpectroscopyOverlap):
    """Build a multiplexed concurrent-tone spectroscopy Schedule for a Qblox
    cluster."""

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

        schedule = Schedule("qubit_spectroscopy_overlap_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "drive")
            element = vendor_element(self, qubit_name, "drive")  # ports: vendor-only
            readout_element = vendor_element(self, qubit_name, "readout")
            windows = overlap_windows(self, qubit_name)  # the ONE timing authority
            center = view.drive_freq_hz
            drive_amp = float(view.drive_amp)  # run() parked the solved residual here
            drive_clock = f"{qubit_name}.01"
            microwave_port = element.ports.microwave
            # / 1e9, never * 1e-9: the probes' float-exactness rule (the scheduler
            # refuses anything off its 1 ns grid).
            tone_len_s = round(windows.tone_len_ns) / 1e9
            drive_len_s = round(windows.drive_len_ns) / 1e9
            acq_delay_s = _acq_delay_s(readout_element) + round(windows.acq_start_ns) / 1e9

            meas_label = f"meas_{qubit_name}"
            sub = Schedule(f"qubit_spec_overlap_{qubit_name}")
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
                    # retune the drive NCO while the drive is still OFF, then let
                    # the reset run — unlike the sibling probe this Reset is a
                    # genuine state reset, not a driven dwell: the qubit reaches
                    # its driven steady state DURING the tone, not before it.
                    sub.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                    add_reset(sub, self, qubit_name)
                    # --- t = 0: both tones up together (see module docstring, 1)
                    sub.add(
                        VoltageOffset(drive_amp, 0, port=microwave_port, clock=drive_clock)
                    )
                    sub.add(
                        Measure(
                            qubit_name,
                            coords={f"frequency_{qubit_name}": freq},
                            acq_channel=f"S_21_{qubit_name}",
                            pulse_duration=tone_len_s,  # lengthened for the lead
                            acq_delay=acq_delay_s,  # TOF + the lead (see 2)
                        ),
                        label=meas_label,
                    )
                    # drive OFF drive_len_ns after the shared onset. ref_op takes
                    # the LABEL, not the return of sub.add: inside a loop body the
                    # schedulable a caller gets back is not the anchor to lean on.
                    sub.add(
                        VoltageOffset(0, 0, port=microwave_port, clock=drive_clock),
                        ref_op=meas_label,
                        ref_pt="start",
                        rel_time=drive_len_s,
                    )
                    # re-anchor the ASAP chain past the readout pulse (see 3)
                    sub.add(IdlePulse(4e-9), ref_op=meas_label, ref_pt="end")
            schedule.add(sub)
        return schedule
