"""Neutral flux-distortion facts -> Qblox QCM hardware distortion-correction config.

Pure config-value helper (no push, no qblox_scheduler import): turn the scqo
flux-channel facts ``distortion_amp[]`` (relative amplitudes ``A_i``) +
``distortion_tau_s[]`` (seconds) into the plain dict a QCM baseband output's
``QbloxHardwareDistortionCorrection`` accepts. Qblox uses the SAME forward
convention as our model (amplitude = exponential overshoot, negative = undershoot;
``time_constant`` in seconds), so there is NO unit or sign change — only the
4-stage exp bank limit + bounds, applied via scqat's ``partition_exp_stages``.

The caller keys the returned dict into ``hw_config.json``'s
``hardware_options.distortion_corrections["<flux_port>-cl0.baseband"]`` (e.g.
``"q1:fl-cl0.baseband"`` — flux operations play on the baseband IDENTITY clock;
there is no ``<q>.flux`` clock), as exactly ONE correction object — a real
(flux) output refuses a list, which is the complex-channel shape. The one-step
door is ``python -m scqo_qblox.backend.apply_distortion``. Only a DECLARED
block persists across runs: the agent's ``hardware_configuration`` is authoritative
and recompiled every upload, so a live qcodes ``.set()`` is overwritten.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

#: QCM baseband exponential-overshoot bank: 4 stages, amplitude in [-1, 1],
#: time_constant >= 6 ns (qblox_scheduler ExpCoeffs). The FIR is a 32-tap sibling.
QBLOX_MAX_EXP_STAGES = 4
QBLOX_EXP_AMP_RANGE = (-1.0, 1.0)
QBLOX_EXP_TAU_MIN_S = 6e-9


def to_qblox_distortion(
    amps: Sequence[float],
    taus_s: Sequence[float],
    *,
    fir_coeffs: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Assemble a ``QbloxHardwareDistortionCorrection``-shaped dict from the facts.

    Returns ``{"exp0_coeffs": {"time_constant": tau_s, "amplitude": A}, ...,
    ["fir_coeffs": [...]], "overflow": [(A, tau), ...]}`` — up to 4 exp stages (the
    most significant by ``|A|``, within the amplitude/tau bounds), amplitudes and
    time constants passed through unchanged (same forward convention, seconds).
    ``overflow`` lists taps that exceed the 4-stage bank or the bounds and need the
    wideband/FIR path (deferred) — they are NEVER silently dropped. ``fir_coeffs``,
    if given, is passed through (Qblox validates that it sums to 1).
    """
    if len(amps) != len(taus_s):
        raise ValueError(
            f"amps ({len(amps)}) and taus_s ({len(taus_s)}) must be equal length")
    from scqat.tools.flux_predistortion import partition_exp_stages

    part = partition_exp_stages(
        amps, taus_s,
        max_stages=QBLOX_MAX_EXP_STAGES,
        amp_range=QBLOX_EXP_AMP_RANGE,
        tau_min_s=QBLOX_EXP_TAU_MIN_S,
    )
    out: dict[str, Any] = {}
    for i, (a, tau) in enumerate(part["kept"]):
        out[f"exp{i}_coeffs"] = {"time_constant": float(tau), "amplitude": float(a)}
    if fir_coeffs is not None:
        out["fir_coeffs"] = [float(c) for c in fir_coeffs]
    out["overflow"] = part["overflow"]
    return out
