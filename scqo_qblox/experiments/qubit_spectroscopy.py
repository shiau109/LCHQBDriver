"""Qblox qubit spectroscopy — supplies only ``probe()``.

Two-tone: a weak saturation drive on the qubit's microwave port while the
drive-clock NCO steps through the sweep. The drive is a FINITE ``SquarePulse``
that ENDS at an anchor and starts ``drive_len_ns`` earlier; ``readout_overlap``
picks the anchor:

    readout_overlap = False              readout_overlap = True
    [==== drive ====]                       [======== drive ========]
                    [## readout ##]   [## readout tone ############]
                    ^ anchor                                anchor ^
                                            [acq_start_ns][== ADC ==]

``False`` is the default and is what the experiment assumes: the drive is over
before the readout tone starts, so the line is measured with no readout photons
present (T1 outlasts the readout). ``True`` ends the drive with the tone instead,
so the ADC window is covered by a live drive and the line comes back AC-Stark
shifted — see the experiment description before trusting its writeback.

WHY A ``SquarePulse`` AND NOT A LATCHED ``VoltageOffset``. It used to be an
offset, latched across the WHOLE sweep and zeroed once per qubit at the end — so
the drive was still on during every ``Measure``, and this backend was measuring a
different experiment from QM's, which has always played a finite saturation
pulse. Nothing is lost by switching: the vendor's own
``compile_long_pulses_to_awg_offsets`` pass rewrites any square pulse of
``PULSE_STITCHING_DURATION`` (100 ns) or longer into exactly that offset pair
plus a 4 ns tail, so a 20 us drive costs no waveform memory and never approaches
``MAX_SAMPLE_SIZE_WAVEFORMS``. What is gained is that the pulse has a LENGTH the
scheduler honours, which is the whole point.

Timing is NOT computed here: ``scqo.experiments._overlap.overlap_windows``
resolves the tone length, the ADC lead and the two start leads, so this probe and
the QM one cannot drift apart on what the same Parameters mean. In overlap mode
the element that starts SECOND is anchored with a NON-NEGATIVE ``rel_time`` —
subschedules never get ``_normalize_absolute_timing``, so a negative one is not
an option, which is why the shared helper hands out two leads instead of one
signed offset.

Drive power contract: the core ``run()`` already solved the drive chain for
``drive_power_dbm`` (recorded set -> acquire -> revert), parking the residual on
``element.spec.spec_amp`` — so the probe plays each qubit's OWN solved amplitude
(``view.drive_amp``) as the pulse amplitude.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitSpectroscopy
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
class QbloxQubitSpectroscopy(QubitSpectroscopy):
    """Build a multiplexed two-tone spectroscopy Schedule for a Qblox cluster."""

    #: Both sequences play a FINITE saturation pulse that has ended before the
    #: next point's reset, so the reset is a genuine state reset — it used to be
    #: a driven dwell under a live CW drive, which is why this was denied. The
    #: readout condition is frozen for the whole run (only the DRIVE frequency
    #: sweeps), which is the other half of the rule in _reset.py.
    #: NOT yet validated on the instrument — see the hardware ledger.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            SquarePulse,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes["detuning_hz"]
        reps = self.params.num_averages
        overlap = self.params.readout_overlap

        schedule = Schedule("qubit_spectroscopy_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "drive")
            element = vendor_element(self, qubit_name, "drive")  # ports: vendor-only
            center = float(view.drive_freq_hz)
            drive_amp = float(view.drive_amp)  # run() parked the solved residual here
            drive_clock = f"{qubit_name}.01"
            mw_port = element.ports.microwave
            mw_port = mw_port() if callable(mw_port) else mw_port
            # / 1e9, never * 1e-9: the probes' float-exactness rule (the scheduler
            # refuses anything off its 1 ns grid).
            if overlap:
                readout_element = vendor_element(self, qubit_name, "readout")
                windows = overlap_windows(self, qubit_name)  # the ONE timing authority
                drive_len_s = round(windows.drive_len_ns) / 1e9
                tone_len_s = round(windows.tone_len_ns) / 1e9
                drive_lead_s = round(windows.drive_lead_ns) / 1e9
                readout_lead_s = round(windows.readout_lead_ns) / 1e9
                acq_delay_s = _acq_delay_s(readout_element) + round(windows.acq_start_ns) / 1e9
            else:
                drive_len_s = round(float(self.params.drive_len_ns)) / 1e9

            drive_label = f"drive_{qubit_name}"
            meas_label = f"meas_{qubit_name}"
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
                    # retune the drive NCO while the drive is still off, then let
                    # the reset run: a real state reset in both modes, since the
                    # previous point's drive pulse ended before it.
                    sub.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                    add_reset(sub, self, qubit_name)

                    drive = SquarePulse(
                        drive_amp, drive_len_s, port=mw_port, clock=drive_clock
                    )
                    measure = Measure(
                        qubit_name,
                        coords={f"frequency_{qubit_name}": freq},
                        acq_channel=f"S_21_{qubit_name}",
                        **({"pulse_duration": tone_len_s, "acq_delay": acq_delay_s}
                           if overlap else {}),
                    )

                    if not overlap:
                        # ASAP chaining IS the anchor: the pulse has a length, so
                        # the Measure lands exactly at its end. This is QM's
                        # align() written out.
                        sub.add(drive)
                        sub.add(measure)
                        sub.add(IdlePulse(4e-9))
                    else:
                        # Both END together, so whichever starts FIRST goes on the
                        # ASAP chain and the other hangs off it by its lead.
                        # ref_op takes the LABEL, not the return of sub.add: inside
                        # a loop body the schedulable a caller gets back is not the
                        # anchor to lean on.
                        if windows.drive_lead_ns > 0:
                            sub.add(drive, label=drive_label)
                            sub.add(measure, label=meas_label, ref_op=drive_label,
                                    ref_pt="start", rel_time=drive_lead_s)
                        else:
                            sub.add(measure, label=meas_label)
                            sub.add(drive, label=drive_label, ref_op=meas_label,
                                    ref_pt="start", rel_time=readout_lead_s)
                        # re-anchor the ASAP chain past the readout pulse: without
                        # it the chain would continue from whichever op happens to
                        # be last and the next sweep point could start inside the
                        # previous measurement.
                        sub.add(IdlePulse(4e-9), ref_op=meas_label, ref_pt="end")
            schedule.add(sub)
        return schedule
