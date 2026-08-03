"""The Qblox flux-distortion config-value wrapper (facts -> exp-stage dict).

Pure unit tests: same forward convention, so amplitude/time_constant pass through
unchanged; only the 4-stage bank + bounds are enforced. The final test optionally
constructs a real ``QbloxHardwareDistortionCorrection`` to prove the dict validates.
"""

import pytest

from lchqb.backend._distortion import (
    QBLOX_MAX_EXP_STAGES,
    to_qblox_distortion,
)


def test_maps_stages_seconds_and_amp_verbatim():
    out = to_qblox_distortion([0.05, -0.03], [100e-9, 3000e-9])
    # ranked by |A|: 0.05 first
    assert out["exp0_coeffs"] == {"time_constant": 100e-9, "amplitude": 0.05}
    assert out["exp1_coeffs"] == {"time_constant": 3000e-9, "amplitude": -0.03}
    assert out["overflow"] == []


def test_length_mismatch_refused():
    with pytest.raises(ValueError, match="equal length"):
        to_qblox_distortion([0.05], [1e-9, 2e-9])


def test_more_than_four_stages_overflow():
    amps = [0.06, 0.05, 0.04, 0.03, 0.02]  # 5 stages, all in-bounds
    taus = [1e-6] * 5
    out = to_qblox_distortion(amps, taus)
    exp_keys = [k for k in out if k.startswith("exp") and k.endswith("_coeffs")]
    assert len(exp_keys) == QBLOX_MAX_EXP_STAGES == 4
    assert len(out["overflow"]) == 1
    assert out["overflow"][0][0] == 0.02  # the smallest |A| overflowed


def test_out_of_bounds_overflow():
    # 1.5 amplitude is outside [-1, 1]; 2e-9 tau is below the 6 ns floor.
    out = to_qblox_distortion([1.5, 0.05, 0.04], [1e-6, 2e-9, 3e-6])
    assert (1.5, 1e-6) in out["overflow"]
    assert (0.05, 2e-9) in out["overflow"]
    assert out["exp0_coeffs"] == {"time_constant": 3e-6, "amplitude": 0.04}


def test_fir_coeffs_passed_through():
    out = to_qblox_distortion([0.05], [100e-9], fir_coeffs=[0.5, 0.5])
    assert out["fir_coeffs"] == [0.5, 0.5]


def test_dict_validates_as_qblox_correction():
    """The emitted dict constructs a real QbloxHardwareDistortionCorrection."""
    pytest.importorskip("qblox_scheduler")
    from qblox_scheduler.backends.types.qblox.calibration import (
        QbloxHardwareDistortionCorrection,
    )

    out = to_qblox_distortion([0.05, -0.03], [100e-9, 3000e-9])
    payload = {k: v for k, v in out.items() if k != "overflow"}
    corr = QbloxHardwareDistortionCorrection(**payload)
    assert corr.exp0_coeffs.amplitude == pytest.approx(0.05)
    assert corr.exp0_coeffs.time_constant == pytest.approx(100e-9)
    assert corr.exp1_coeffs.amplitude == pytest.approx(-0.03)
