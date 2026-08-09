"""Hardware-config serialization: explicit nulls must never be written or trusted.

pydantic's default ``model_dump_json`` writes every unset Optional as ``null``, and
the qblox compiler treats an explicitly-null channel description as "user set None"
(it lands in ``model_fields_set`` on reload) and crashes sequencer compilation with
``ValidationError: ... hardware_description ... input_value=None`` — whereas an
ABSENT key gets a default ChannelDescription. Found on chipA 2026-07-16: the first
accepted calibration called save(), which re-wrote hw_config.json with nulls and
broke every later run. Offline: no cluster is contacted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import HW_MIN, make_backend, make_experiment  # noqa: E402

from scqo_qblox.backend.qblox_backend import QbloxBackend  # noqa: E402


def _make_backend(tmp_path: Path, roster, poison: bool) -> QbloxBackend:
    hw = json.loads(HW_MIN.read_text(encoding="utf-8"))
    if poison:  # exactly the disease a pre-fix save() wrote to chipA's file
        hw["hardware_description"]["cluster_A"]["modules"]["8"].update(
            {"complex_output_0": None, "complex_input_0": None,
             "digital_output_0": None, "digital_output_1": None})
    return make_backend(tmp_path, roster, hw_config=hw)


def _null_set_fields(backend: QbloxBackend) -> list[str]:
    """Fields that are SET-to-None on any module description (the crash trigger)."""
    hw = backend._hw_agent.hardware_configuration
    bad = []
    for cluster in hw.hardware_description.values():
        for slot, module in (getattr(cluster, "modules", None) or {}).items():
            bad += [f"{slot}.{f}" for f in module.model_fields_set
                    if getattr(module, f) is None]
    return bad


def _compile_res_spec(backend: QbloxBackend, roster):
    """Build the real resonator-spectroscopy probe Schedule and compile it offline —
    the same SerialCompiler call HardwareAgent.run makes after connecting."""
    import scqo_qblox.experiments  # noqa: F401  (registers the qblox probe classes)
    from qblox_scheduler.backends.graph_compilation import SerialCompiler
    from scqo.experiments import get

    exp_cls = get("resonator_spectroscopy")
    exp = make_experiment(
        exp_cls, backend, roster,
        exp_cls.Parameters(targets=["q1"], num_points=11, num_averages=2))
    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()
    qd = backend._hw_agent.quantum_device
    qd.hardware_config = backend._hw_agent.hardware_configuration  # the runtime truth
    return SerialCompiler().compile(schedule=schedule, config=qd.generate_compilation_config())


def test_save_writes_no_null_channel_descriptions(tmp_path, roster):
    backend = _make_backend(tmp_path, roster, poison=False)
    backend.device.save()
    written = json.loads((tmp_path / "hw_config.json").read_text(encoding="utf-8"))
    module = written["hardware_description"]["cluster_A"]["modules"]["8"]
    assert not [k for k, v in module.items() if v is None]  # no nulls, ever
    # ...and the dut-embedded copy carries the SAME clean block (to_json would
    # otherwise re-embed nulls — save() rewrites it from the clean dump)
    dut = json.loads((tmp_path / "dut_config.json").read_text(encoding="utf-8"))
    emb = dut["hardware_config"]["hardware_description"]["cluster_A"]["modules"]["8"]
    assert not [k for k, v in emb.items() if v is None]


def test_null_poisoned_config_loads_normalized_and_compiles(tmp_path, roster):
    """The chipA 2026-07-16 pin: a null-poisoned hw config must (a) normalize at
    construction (nulls leave model_fields_set) and (b) COMPILE a real probe
    schedule — the exact step that crashed."""
    backend = _make_backend(tmp_path, roster, poison=True)
    assert _null_set_fields(backend) == []  # (a) normalized on load
    assert _compile_res_spec(backend, roster) is not None  # (b) compiles offline

    # and the next save() self-heals the file on disk
    backend.device.save()
    written = json.loads((tmp_path / "hw_config.json").read_text(encoding="utf-8"))
    module = written["hardware_description"]["cluster_A"]["modules"]["8"]
    assert "complex_output_0" not in module  # the poison is gone from disk


def test_clean_config_still_compiles(tmp_path, roster):
    backend = _make_backend(tmp_path, roster, poison=False)
    assert _compile_res_spec(backend, roster) is not None
