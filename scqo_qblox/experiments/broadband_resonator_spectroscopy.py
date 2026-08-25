"""Qblox broadband resonator spectroscopy — supplies only ``probe()``.

Parameters, fitting, simulation are inherited from
``scqo.experiments.BroadbandResonatorSpectroscopy``.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import BroadbandResonatorSpectroscopy

from ._broadband import read_field, stitch_subbands, write_field


@register
class QbloxBroadbandResonatorSpectroscopy(BroadbandResonatorSpectroscopy):
    """Build and execute wideband resonator spectroscopy across stepped LO sub-bands on Qblox."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "broadband spectroscopy steps LO frequencies across sub-bands"

    @staticmethod
    def build_sub_schedule(
        primary_qubit: str, f_start: float, f_stop: float, n_pts: int, num_averages: int
    ) -> Any:
        """Build a single-segment 1D frequency sweep schedule."""
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        schedule = Schedule(f"broadband_res_spec_{primary_qubit}")
        with schedule.loop(arange(0, num_averages, 1, DType.NUMBER)):
            with schedule.loop(
                linspace(f_start, f_stop, n_pts, dtype=DType.FREQUENCY)
            ) as freq:
                schedule.add(
                    Measure(
                        primary_qubit,
                        freq=freq,
                        coords={f"frequency_{primary_qubit}": freq},
                        acq_channel=f"S_21_{primary_qubit}",
                    )
                )
                schedule.add(IdlePulse(10e-6))
        return schedule

    def probe(self) -> xr.Dataset:
        from scqo_qblox.backend.qblox_backend import chunk_timeout_s

        primary_target = self.params.targets[0]
        chan_name = self.device.roster.default_channel(primary_target, "readout")
        view = self.backend.device.component(chan_name)
        port_clock = view._port_clock()

        hw_agent = self.backend._hw_agent
        hw_cfg = hw_agent.hardware_configuration
        opts = hw_cfg.hardware_options
        mf = getattr(opts, "modulation_frequencies", None)
        if mf is None:
            opts.modulation_frequencies = {}
            mf = opts.modulation_frequencies

        mf_entry = mf.get(port_clock)
        if isinstance(mf_entry, dict):
            orig_lo = mf_entry.get("lo_freq")
        else:
            orig_lo = getattr(mf_entry, "lo_freq", None)

        # Save original readout clock frequencies for all qubits
        orig_readout_clocks: dict[str, float] = {}
        for q_name in self.device.roster.modes():
            try:
                ch_name = self.device.roster.default_channel(q_name, "readout")
            except Exception:
                # roster.modes() spans EVERY mode, couplers included, and those
                # have no readout channel -- skipping them is this guard's whole
                # job. It does NOT cover the reads below.
                continue
            el = self.backend.device.component(ch_name)._element
            val = read_field(el.clock_freqs, "readout")
            if val is not None:
                orig_readout_clocks[q_name] = float(val)

        start = float(self.params.start_freq_hz)
        stop = float(self.params.stop_freq_hz)
        bw = float(self.params.bandwidth_per_lo_hz)
        pts_per_lo = int(self.params.num_points_per_lo)
        gap = float(self.params.lo_gap_hz)
        reps = int(self.params.num_averages)

        # Single-sideband IF offset avoiding LO leakage: IF in [min_if, max_if]
        min_if = max(20.0e6, gap / 2.0)
        max_if = min(400.0e6, min_if + bw)
        span_per_lo = max_if - min_if
        # Overlap margin between adjacent sub-bands for phase & gain alignment
        overlap = min(30.0e6, span_per_lo * 0.15) if (stop - start) > span_per_lo else 0.0

        slices: list[tuple[float, float, float]] = []
        curr_f = start
        while curr_f < stop:
            next_f = min(stop, curr_f + span_per_lo)
            lo = curr_f - min_if
            slices.append((curr_f, next_f, lo))
            if next_f >= stop:
                break
            curr_f = next_f - overlap

        subband_data: list[tuple[np.ndarray, np.ndarray]] = []

        try:
            for f_a, f_b, lo in slices:
                slice_span = f_b - f_a
                if slice_span <= 0:
                    continue

                n_pts = max(2, int(round(pts_per_lo * (slice_span / span_per_lo))))
                rf_seg = np.linspace(f_a, f_b, n_pts)

                # Update LO frequency in hardware config for this sub-band
                if isinstance(mf_entry, dict):
                    mf_entry["lo_freq"] = lo
                elif hasattr(mf_entry, "lo_freq"):
                    mf_entry.lo_freq = lo
                elif isinstance(mf, dict):
                    try:
                        from qblox_scheduler.backends.types.qblox import ModulationFrequencies

                        mf[port_clock] = ModulationFrequencies(lo_freq=lo)
                    except Exception:
                        mf[port_clock] = {"lo_freq": lo}

                # Synchronize base readout clock frequency to f_a (initial NCO = f_a - lo = min_if)
                # NOT best-effort: a swallowed failure here measures the
                # whole sub-band at the PREVIOUS LO's frequency and labels it
                # with this one's.
                for q_name in orig_readout_clocks:
                    ch_name = self.device.roster.default_channel(q_name, "readout")
                    el = self.backend.device.component(ch_name)._element
                    write_field(el.clock_freqs, "readout", f_a)

                sub = self.build_sub_schedule(primary_target, f_a, f_b, n_pts, reps)
                qd = hw_agent.quantum_device
                qd.hardware_config = hw_cfg

                timeout_s = chunk_timeout_s(self, shots=reps, points=n_pts)
                hw_agent.instrument_coordinator.timeout(timeout_s)
                raw_sub = hw_agent.run(sub, timeout=timeout_s)

                key = f"S_21_{primary_target}"
                if key in raw_sub.data_vars:
                    vals = np.asarray(raw_sub[key].values).squeeze()
                    subband_data.append((rf_seg, vals))
        finally:
            # Restore original LO setting
            if isinstance(mf_entry, dict):
                mf_entry["lo_freq"] = orig_lo
            elif hasattr(mf_entry, "lo_freq"):
                mf_entry.lo_freq = orig_lo
            elif isinstance(mf, dict):
                if orig_lo is not None:
                    try:
                        from qblox_scheduler.backends.types.qblox import ModulationFrequencies

                        mf[port_clock] = ModulationFrequencies(lo_freq=orig_lo)
                    except Exception:
                        mf[port_clock] = {"lo_freq": orig_lo}
                else:
                    # There was no entry before this probe injected one; leaving
                    # it behind would pin a synthetic LO override on the port for
                    # the rest of the session.
                    mf.pop(port_clock, None)

            # Restore original readout clock frequencies
            for q_name, orig_clk in orig_readout_clocks.items():
                try:
                    ch_name = self.device.roster.default_channel(q_name, "readout")
                    el = self.backend.device.component(ch_name)._element
                    write_field(el.clock_freqs, "readout", orig_clk)
                except Exception:
                    pass

        final_freqs, final_z = stitch_subbands(subband_data)

        # Every resonator hangs off the SAME feedline, so one transmission
        # trace IS each target's data -- the neutral simulate() broadcasts
        # identically ("Broadcast identical feedline transmission to all
        # targets") and the estimator's job is to find N dips in the one trace.
        # Unlike the qubit variant, where each target owns its own drive line,
        # this replication is the physics, not a stand-in for an unmeasured
        # target.
        targets = list(self.params.targets)
        n_targets = len(targets)

        i_2d = np.tile(np.real(final_z), (n_targets, 1))
        q_2d = np.tile(np.imag(final_z), (n_targets, 1))

        return xr.Dataset(
            data_vars={
                "I": (("target", "frequency_hz"), i_2d),
                "Q": (("target", "frequency_hz"), q_2d),
            },
            coords={
                "target": targets,
                "frequency_hz": final_freqs,
            },
        )
