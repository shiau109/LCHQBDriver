"""Qblox single-shot readout (IQ blobs) — supplies only ``probe()``.

cal16 (MultiplexedSSRO) reference pattern, per-shot mechanics: the shot loop
variable is CAPTURED and written into the acquisition ``coords`` — every
(state, shot) pair is then a distinct labeled bin, so the cluster appends one
I/Q point per shot instead of averaging (same mechanism the averaged probes use
in reverse: there the reps loop is *unlabeled*, so identical coords average).
No ``num_averages`` exists here by design.

Deviation from cal16: the two prepared states are sequential blocks (all |0>
shots, then all |1> shots) instead of interleaved per shot, so the flat bin
order is state-major and matches the canonical sweep-axes order
(``prepared_state``, ``shot_idx``) that ``_to_canonical`` reshapes by.
Parameters and the two-Gaussian fidelity fit are inherited from
``scqo.experiments.SingleShotReadout``; this driver additionally supplies the
``update()`` half that PROPOSES the readout discriminator (see below).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scqo import register
from scqo.experiments import SingleShotReadout
from scqo.result import Outcome

from ._reset import add_reset


@register
class QbloxSingleShotReadout(SingleShotReadout):
    """Build a multiplexed per-shot SSRO Schedule for a Qblox cluster."""

    # No supports_active_reset: this experiment IS the discriminator's
    # calibration, so resetting with it would be circular — and a conditional
    # pi driven by the very threshold being measured biases the |1> blob.
    # _reset.py refuses reset_method='active' by name.

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X
        from qblox_scheduler.operations.loop_domains import DType, arange

        num_shots = int(self.params.num_shots)

        schedule = Schedule("single_shot_readout_multiplexed")
        for qubit_name in self.params.targets:
            sub = Schedule(f"ssro_{qubit_name}")
            # prepared_state 0: Reset -> Measure, one labeled bin per shot
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
            # prepared_state 1: Reset -> X -> Measure, one labeled bin per shot
            with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                add_reset(sub, self, qubit_name)
                sub.add(X(qubit=qubit_name))
                sub.add(
                    Measure(
                        qubit_name,
                        coords={f"state_{qubit_name}": 1, f"shot_{qubit_name}": shot},
                        acq_channel=f"S_21_{qubit_name}",
                    )
                )
                sub.add(IdlePulse(4e-9))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule

    def update(self) -> None:
        """The inherited fidelity + blob-center MONITOR writes, plus the Qblox
        readout DISCRIMINATOR as governed suggestions.

        For every SUCCESSFUL target this solves the demod rotation and the g/e cut
        from the measured blobs (``scqat.tools.compute_ge_discriminator`` — the same
        solve the QM backend uses) and PROPOSES them as the neutral
        ``readout_rotation_rad`` / ``readout_threshold`` on the target's READOUT
        channel: pending suggestions, decided with ``scqo accept``, pushed to
        ``element.measure.acq_rotation`` / ``acq_threshold`` on accept. That is what
        arms ``use_state_discrimination`` on the four coherent-drive probes — until
        those two are accepted, they refuse (experiments/_state.py).

        The run itself never touches the element, so the figure always shows the
        data in the frame it was measured in.

        The rotation is proposed ABSOLUTELY here, and that is the one place this
        deliberately differs from the QM shell — do not "fix" it back to QM's
        ``current - delta``. Accumulate only when the vendor's rotation knob feeds
        back into the acquired data. QM's ``integration_weights_angle`` is folded
        into the integration weights, so it rotates EVERY acquisition and the next
        run's blobs come back already rotated, driving the delta to zero. Qblox's
        ``acq_rotation`` is read only on the ``ThresholdedAcquisition`` path, so the
        ``SSBIntegrationComplex`` blobs this experiment measures are never rotated:
        the solve always sees the RAW frame and its answer is already the absolute
        rotation. Accumulating there has no fixed point — chipA 2026-07-26 walked
        0 -> -0.388 -> -0.788 -> -1.177 rad over three accepted runs while the
        measured blob angle sat at -0.39 the whole time.

        ``readout_rus_threshold`` is deliberately NOT proposed: repeat-until-success
        has no Qblox counterpart and the knob is Unrealized here, so writing it
        would raise.
        """
        super().update()  # fidelity_g/e + pos_* monitors (through self.device)

        if self.result is None or self.dataset is None:
            return
        from scqat.tools import compute_ge_discriminator

        for qubit in self.params.targets:
            if self.result.outcomes.get(qubit) is not Outcome.SUCCESSFUL:
                continue
            fit = self.result.fit[qubit]
            mean_g = (fit["mean_g_i"], fit["mean_g_q"])
            mean_e = (fit["mean_e_i"], fit["mean_e_q"])
            if not np.all(np.isfinite([*mean_g, *mean_e])):
                print(f"[single_shot_readout] {qubit}: degenerate blobs; no discriminator proposal")
                continue

            sq = self.dataset.sel(target=qubit)
            shots_g = (sq["I"].sel(prepared_state=0).values, sq["Q"].sel(prepared_state=0).values)
            shots_e = (sq["I"].sel(prepared_state=1).values, sq["Q"].sel(prepared_state=1).values)
            d = compute_ge_discriminator(mean_g, mean_e, shots_g, shots_e)

            view = self.device.channel(qubit, "readout")
            current = float(view.readout_rotation_rad)  # for the log line only
            # ABSOLUTE, not current - delta: the blobs above were acquired in the
            # raw frame regardless of what acq_rotation holds (see the docstring).
            new_rotation = -d["delta_angle_rad"]
            view.readout_rotation_rad = new_rotation
            view.readout_threshold = d["ge_threshold"]
            print(
                f"[single_shot_readout] {qubit}: discriminator PROPOSED "
                f"(readout_rotation_rad {current:.4f} -> {new_rotation:.4f} rad, "
                f"readout_threshold {d['ge_threshold']:.4g}) — review with scqo accept, "
                f"then re-run to confirm"
            )
