"""Shared machinery for the two broadband sweeps.

Both broadband probes walk the requested RF range in sub-bands, re-parking the
LO between them, and both then have to make ONE trace out of the pieces. The
stitching is identical for a drive line and for the feedline, so it lives here
rather than twice -- the two copies this replaces were byte-for-byte the same,
and a divergence between them would stay invisible until two runs of "the same"
experiment disagreed.

The vendor accessors live here too: a ``qblox_scheduler`` submodule field is
sometimes a QCoDeS ``Parameter`` (callable) and sometimes a plain attribute
depending on the element class, so every probe that reaches into ``clock_freqs``
or ``rxy`` needs the same two-line dance.

NOTE the QM half does NOT rescale its sub-bands -- it concatenates and sorts.
Keeping the gain/phase alignment here means the two backends hand their
estimators differently-normalised traces for the same experiment. That is a
physics call still to settle; until it is, this is the one place to change.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def read_field(container: Any, param: str) -> Any:
    getter = getattr(container, param, None)
    return getter() if callable(getter) else getter


def write_field(container: Any, param: str, value: Any) -> None:
    setter = getattr(container, param, None)
    if callable(setter):
        setter(value)
    else:
        setattr(container, param, value)


def stitch_subbands(
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
