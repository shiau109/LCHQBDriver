"""Qblox DRAG Equator (3-Line) calibration acquisition probe.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitDragEquator``.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import QubitDragEquator

from ._reset import add_reset
from ._state import measure_kwargs


def _read(container: Any, param: str) -> Any:
    getter = getattr(container, param, None)
    return getter() if callable(getter) else getter


def _write(container: Any, param: str, value: Any) -> None:
    setter = getattr(container, param, None)
    if callable(setter):
        setter(value)
    else:
        setattr(container, param, value)


@register
class QbloxQubitDragEquator(QubitDragEquator):
    """Build and execute the DRAG equator calibration across swept beta on Qblox."""

    probe_self_acquires = (
        "it sweeps DRAG beta by stepping element.rxy.beta and acquiring across schedules"
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange

        qubits = list(self.params.targets)
        beta_array = np.asarray(self.sweep_axes["beta"], dtype=float)
        reps = int(self.params.num_averages)
        target_gate = getattr(self.params, "target_gate", "x180")
        op_name = "x90" if str(target_gate).strip().lower() == "x90" else "x180"

        n_qubits = len(qubits)
        n_seq = 2
        n_beta = len(beta_array)

        i_data = np.zeros((n_qubits, n_seq, n_beta), dtype=float)
        q_data = np.zeros((n_qubits, n_seq, n_beta), dtype=float)

        orig_betas: dict[str, float] = {}
        for q_name in qubits:
            channel = self.device.channel(q_name, "drive")
            orig_betas[q_name] = float(channel.drag_beta or 0.0)

        hw_agent = self.backend._hw_agent

        try:
            for i_b, beta_val in enumerate(beta_array):
                # Update DRAG beta on each qubit channel before building the schedule
                for q_name in qubits:
                    self.device.channel(q_name, "drive").drag_beta = float(beta_val)

                schedule = Schedule(f"drag_equator_beta_{i_b}")
                for q_name in qubits:
                    acq = measure_kwargs(self, q_name)
                    sub = Schedule(f"drag_equator_{q_name}")
                    with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                        # Seq 0: Rx(pi) - Ry(pi/2)  (or two x90 - Ry(pi/2))
                        add_reset(sub, self, q_name)
                        if op_name == "x90":
                            sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                            sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                            sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                        else:
                            sub.add(Rxy(theta=180.0, phi=0.0, qubit=q_name))
                            sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                        sub.add(
                            Measure(
                                q_name,
                                coords={f"seq_{q_name}": 0},
                                acq_channel=f"S_21_{q_name}",
                                **acq,
                            )
                        )
                        sub.add(IdlePulse(4e-9))

                        # Seq 1: Ry(pi) - Rx(pi/2)  (or two y90 - Rx(pi/2))
                        add_reset(sub, self, q_name)
                        if op_name == "x90":
                            sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                            sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                            sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                        else:
                            sub.add(Rxy(theta=180.0, phi=90.0, qubit=q_name))
                            sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                        sub.add(
                            Measure(
                                q_name,
                                coords={f"seq_{q_name}": 1},
                                acq_channel=f"S_21_{q_name}",
                                **acq,
                            )
                        )
                        sub.add(IdlePulse(4e-9))
                    sub.add(IdlePulse(4e-9))
                    schedule.add(sub)

                timeout_s = max(300, int(reps * 2 * 500e-6 * 3.0 + 60))
                hw_agent.instrument_coordinator.timeout(timeout_s)
                raw = hw_agent.run(schedule, timeout=timeout_s)

                for k, q_name in enumerate(qubits):
                    key = f"S_21_{q_name}"
                    if key not in raw.data_vars:
                        raise KeyError(f"acquisition channel {key!r} not in raw dataset")
                    values = np.asarray(raw[key].values).squeeze()
                    # values has 2 points corresponding to seq 0 and seq 1
                    i_data[k, 0, i_b] = float(np.real(values[0]))
                    i_data[k, 1, i_b] = float(np.real(values[1]))
                    q_data[k, 0, i_b] = float(np.imag(values[0]))
                    q_data[k, 1, i_b] = float(np.imag(values[1]))
        finally:
            for q_name, orig_beta in orig_betas.items():
                self.device.channel(q_name, "drive").drag_beta = orig_beta

        coords = {
            "target": qubits,
            "seq_idx": np.array([0, 1]),
            "beta": beta_array,
        }
        dims = ("target", "seq_idx", "beta")
        return xr.Dataset(
            {"I": (dims, i_data), "Q": (dims, q_data)},
            coords=coords,
        )
