"""The ``use_state_discrimination`` branch, written once for every carrier probe.

A discriminating probe differs from an averaging one by exactly one keyword:
``Measure(..., acq_protocol="ThresholdedAcquisition")``. Everything else is
already on the ELEMENT — the compiler pulls ``acq_rotation`` / ``acq_threshold``
out of the ``measure`` factory kwargs (``transmon_element.py``), and those two are
realized scqo knobs on the readout channel (``backend/fieldmap.py``). So the probe
never passes the calibration; it only asks for the protocol, and the governed
device state supplies the numbers.

``BinMode`` is deliberately NOT passed: ``AVERAGE_APPEND`` is already the
``ThresholdedAcquisition`` default, and it is what turns the probes' unlabeled
reps loop into an averaged population in [0, 1] — the same shape the neutral
``STATE_ALT`` contract and scqat's estimators expect. (``BinMode.AVERAGE`` is
deprecated in qblox-scheduler 1.0.0b6 and silently remapped with a FutureWarning.)

This module also owns :func:`require_discriminator`, the "are the two knobs
calibrated" refusal — shared with ``_reset.py``, because active reset thresholds
against exactly the same pair. It lives HERE rather than in a third module: the
discriminator is this file's subject, and the error text is asserted by name in
the suite.

The decode half lives in ``backend/qblox_backend.py::_to_canonical``, which emits
``state`` instead of ``I``/``Q`` when the raw acquisition says it was thresholded.
The two halves are inseparable: a thresholded acquisition returns a REAL float on
the same ``S_21_<qubit>`` variable, so a probe change alone would put the
population in ``I``, leave ``Q`` at zero, and still satisfy the ``("I","Q")``
contract — a wrong fit with no error anywhere.
"""

from __future__ import annotations

from typing import Any

#: The vendor's uncalibrated value for BOTH discriminator knobs. `acq_rotation`
#: and `acq_threshold` are plain `Parameter(initial_value=0.0)` with no validator,
#: so 0.0 is "never calibrated" — a real solved threshold is never exactly zero.
_UNCALIBRATED = 0.0


def require_discriminator(experiment: Any, target: str, because: str) -> None:
    """Refuse, by name, when ``target`` has no calibrated discriminator.

    TWO callers, one message: :func:`measure_kwargs` below (thresholding the
    DATA) and ``_reset.py`` (thresholding the RESET). Both compare shots against
    ``acq_threshold`` in the frame ``acq_rotation`` defines, so both are equally
    dead without them — and active reset does NOT ride on
    ``use_state_discrimination``, so the check cannot live behind that flag.

    ``because`` names the setting that demanded the discriminator, so the error
    points at the parameter the caller actually set.
    """
    view = experiment.device.channel(target, "readout")
    if view.readout_threshold == _UNCALIBRATED and view.readout_rotation_rad == _UNCALIBRATED:
        # Fail here, by name, rather than let the cluster discriminate every shot
        # against zero and return a meaningless population. QM has no equivalent
        # guard — there an uncalibrated threshold is None and surfaces as a
        # TypeError deep inside the QUA DSL, naming nothing.
        raise ValueError(
            f"{target}: {because} but the readout discriminator "
            f"is uncalibrated (readout_rotation_rad and readout_threshold are both "
            f"at the vendor default 0.0). Run `scqo run single_shot_readout`, review "
            f"its proposals with `scqo accept`, then re-run this experiment — or set "
            f"them directly with `scqo set {target}.readout_threshold=...`"
        )


def measure_kwargs(experiment: Any, target: str) -> dict:
    """The ``Measure`` keyword(s) this run's readout mode needs, for one target.

    ``{}`` in the ordinary averaged-I/Q mode; the thresholded protocol when
    ``use_state_discrimination`` is set — after checking the target actually has a
    discriminator to threshold against.
    """
    if not getattr(experiment.params, "use_state_discrimination", False):
        return {}

    require_discriminator(experiment, target, "use_state_discrimination=True")
    return {"acq_protocol": "ThresholdedAcquisition"}
