"""Qblox broadband qubit spectroscopy — supplies only ``probe()``.

Parameters, fitting, simulation are inherited from
``scqo.experiments.BroadbandQubitSpectroscopy``.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import BroadbandQubitSpectroscopy

from ._reset import add_reset
from ._vendor import vendor_element


def _read(container: Any, param: str) -> Any:
    getter = getattr(container, param, None)
    return getter() if callable(getter) else getter


def _write(container: Any, param: str, value: Any) -> None:
    setter = getattr(container, param, None)
    if callable(setter):
        setter(value)
    else:
        setattr(container, param, value)


def _stitch_subbands(
    subbands: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch overlapping sub-band spectra by aligning complex phase/gain and cross-fading."""
    if not subbands:
        raise RuntimeError("no frequency sub-bands were measured")
    if len(subbands) == 1:
        return subbands[0]

    acc_f, acc_z = subbands[0]

    for f_curr, z_curr in subbands[1:]:
        f_ov_min = max(acc_f[0], f_curr[0])
        f_ov_max = min(acc_f[-1], f_curr[-1])

        if f_ov_max > f_ov_min + 1e5:
            # Overlap frequency points on the current slice's grid
            ov_mask = (f_curr >= f_ov_min) & (f_curr <= f_ov_max)
            f_ov = f_curr[ov_mask]
            if len(f_ov) < 2:
                f_ov = np.linspace(f_ov_min, f_ov_max, 21)

            z_prev_ov = np.interp(f_ov, acc_f, acc_z.real) + 1j * np.interp(
                f_ov, acc_f, acc_z.imag
            )
            z_curr_ov = np.interp(f_ov, f_curr, z_curr.real) + 1j * np.interp(
                f_ov, f_curr, z_curr.imag
            )

            # Compute complex least-squares ratio to align current slice to previous
            denom = np.sum(np.abs(z_curr_ov) ** 2)
            if denom > 1e-15:
                alpha = np.sum(z_prev_ov * np.conj(z_curr_ov)) / denom
                if not np.isfinite(alpha) or np.abs(alpha) < 1e-6 or np.abs(alpha) > 1e6:
                    alpha = 1.0
            else:
                alpha = 1.0

            z_curr_aligned = z_curr * alpha

            # Smooth cross-fade blending in the overlap region
            w = (f_ov - f_ov_min) / (f_ov_max - f_ov_min)
            z_curr_ov_aligned = (
                z_curr_aligned[ov_mask]
                if len(f_ov) == np.sum(ov_mask)
                else (
                    np.interp(f_ov, f_curr, z_curr_aligned.real)
                    + 1j * np.interp(f_ov, f_curr, z_curr_aligned.imag)
                )
            )
            z_blended = (1.0 - w) * z_prev_ov + w * z_curr_ov_aligned

            m_prev = acc_f < f_ov_min
            m_curr = f_curr > f_ov_max

            acc_f = np.concatenate([acc_f[m_prev], f_ov, f_curr[m_curr]])
            acc_z = np.concatenate([acc_z[m_prev], z_blended, z_curr_aligned[m_curr]])
        else:
            acc_f = np.concatenate([acc_f, f_curr])
            acc_z = np.concatenate([acc_z, z_curr])

    order = np.argsort(acc_f)
    unique_idx = np.unique(acc_f[order], return_index=True)[1]
    sorted_idx = order[unique_idx]
    return acc_f[sorted_idx], acc_z[sorted_idx]


@register
class QbloxBroadbandQubitSpectroscopy(BroadbandQubitSpectroscopy):
    """Build and execute wideband two-tone qubit spectroscopy across stepped drive LOs on Qblox."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "broadband qubit spectroscopy steps drive LO frequencies across sub-bands"

    def build_sub_schedule(
        self,
        target: str,
        f_start: float,
        f_stop: float,
        n_pts: int,
        reps: int,
    ) -> Any:
        """Build a single-segment 1D frequency sweep schedule for qubit spectroscopy."""
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            VoltageOffset,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        view = self.device.channel(target, "drive")
        element = vendor_element(self, target, "drive")
        drive_amp = float(view.drive_amp)
        drive_clock = f"{target}.01"
        mw_port = element.ports.microwave
        mw_port = mw_port() if callable(mw_port) else mw_port

        schedule = Schedule(f"broadband_qubit_spec_{target}")
        with schedule.loop(arange(0, reps, 1, DType.NUMBER)):
            with schedule.loop(
                linspace(f_start, f_stop, n_pts, dtype=DType.FREQUENCY)
            ) as freq:
                # Continuous weak drive on the microwave port
                schedule.add(VoltageOffset(drive_amp, 0, port=mw_port, clock=drive_clock))
                schedule.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                add_reset(schedule, self, target)
                schedule.add(
                    Measure(
                        target,
                        coords={f"frequency_{target}": freq},
                        acq_channel=f"S_21_{target}",
                    )
                )
                schedule.add(IdlePulse(4e-9))
        # Drive OFF before end of schedule
        schedule.add(VoltageOffset(0, 0, port=mw_port, clock=drive_clock))
        schedule.add(IdlePulse(4e-9))
        return schedule

    def probe(self) -> xr.Dataset:
        primary_target = self.params.targets[0]
        chan_name = self.device.roster.default_channel(primary_target, "drive")
        view = self.backend.device.component(chan_name)
        drive_port_clock = getattr(view, "_drive_port_clock", getattr(view, "_port_clock", None))()

        hw_agent = self.backend._hw_agent
        hw_cfg = hw_agent.hardware_configuration
        opts = hw_cfg.hardware_options
        mf = getattr(opts, "modulation_frequencies", None)
        if mf is None:
            opts.modulation_frequencies = {}
            mf = opts.modulation_frequencies

        mf_entry = mf.get(drive_port_clock)
        if isinstance(mf_entry, dict):
            orig_lo = mf_entry.get("lo_freq")
        else:
            orig_lo = getattr(mf_entry, "lo_freq", None)

        # Save original drive clock frequencies for all targets
        orig_drive_clocks: dict[str, float] = {}
        for q_name in self.params.targets:
            try:
                ch_name = self.device.roster.default_channel(q_name, "drive")
                el = self.backend.device.component(ch_name)._element
                val = _read(el.clock_freqs, "f01")
                if val is not None:
                    orig_drive_clocks[q_name] = float(val)
            except Exception:
                continue

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

                # Update drive LO frequency in hardware config for this sub-band
                if isinstance(mf_entry, dict):
                    mf_entry["lo_freq"] = lo
                elif hasattr(mf_entry, "lo_freq"):
                    mf_entry.lo_freq = lo
                elif isinstance(mf, dict):
                    try:
                        from qblox_scheduler.backends.types.qblox import ModulationFrequencies

                        mf[drive_port_clock] = ModulationFrequencies(lo_freq=lo)
                    except Exception:
                        mf[drive_port_clock] = {"lo_freq": lo}

                # Synchronize base drive clock frequency to f_a (initial NCO = f_a - lo = min_if)
                for q_name in orig_drive_clocks:
                    try:
                        ch_name = self.device.roster.default_channel(q_name, "drive")
                        el = self.backend.device.component(ch_name)._element
                        _write(el.clock_freqs, "f01", f_a)
                    except Exception:
                        pass

                sub = self.build_sub_schedule(primary_target, f_a, f_b, n_pts, reps)
                qd = hw_agent.quantum_device
                qd.hardware_config = hw_cfg

                timeout_s = max(60, int(np.ceil(reps * n_pts * 20e-6 * 2)))
                hw_agent.instrument_coordinator.timeout(timeout_s)
                raw_sub = hw_agent.run(sub, timeout=timeout_s)

                key = f"S_21_{primary_target}"
                if key in raw_sub.data_vars:
                    vals = np.asarray(raw_sub[key].values).squeeze()
                    subband_data.append((rf_seg, vals))
        finally:
            # Restore original drive LO setting
            if isinstance(mf_entry, dict):
                mf_entry["lo_freq"] = orig_lo
            elif hasattr(mf_entry, "lo_freq"):
                mf_entry.lo_freq = orig_lo
            elif isinstance(mf, dict) and orig_lo is not None:
                try:
                    from qblox_scheduler.backends.types.qblox import ModulationFrequencies

                    mf[drive_port_clock] = ModulationFrequencies(lo_freq=orig_lo)
                except Exception:
                    mf[drive_port_clock] = {"lo_freq": orig_lo}

            # Restore original drive clock frequencies
            for q_name, orig_clk in orig_drive_clocks.items():
                try:
                    ch_name = self.device.roster.default_channel(q_name, "drive")
                    el = self.backend.device.component(ch_name)._element
                    _write(el.clock_freqs, "f01", orig_clk)
                except Exception:
                    pass

        final_freqs, final_z = _stitch_subbands(subband_data)

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
