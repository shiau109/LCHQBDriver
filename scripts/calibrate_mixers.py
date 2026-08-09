r"""Automatic Mixer Calibration (AMC) for the RF modules named in a Qblox config folder.

Suppresses LO leakage (w_LO) and the image sideband (w_LO - w_NCO) on QCM-RF / QRM-RF
outputs using the module's *built-in* calibration circuitry. No spectrum analyzer, no
output cabling, no external instrument.

    python scripts/calibrate_mixers.py D:\qpu_data_dev\chipA\cd1\qblox\backend_config
    python scripts/calibrate_mixers.py <config_dir> --dry-run       # plan only, no cluster
    python scripts/calibrate_mixers.py <config_dir> --dummy         # dummy cluster smoke test
    python scripts/calibrate_mixers.py <config_dir> --slot 4        # QCM-RF (drive) only
    python scripts/calibrate_mixers.py <config_dir> --force         # ignore the cache
    python scripts/calibrate_mixers.py <config_dir> --diagnose      # why is the cal a no-op?

What it reads
-------------
``<config_dir>/hw_config.json`` is authoritative:

* ``hardware_description.<cluster>.modules``  -> slot -> module type (+ the cluster ``ip``)
* ``connectivity.graph``                      -> port -> (slot, complex_output_<k>)
* ``hardware_options.modulation_frequencies`` -> per port-clock ``lo_freq`` / ``interm_freq``
* ``hardware_options.output_att``             -> per port-clock output attenuation

``<config_dir>/dut_config.json`` supplies the absolute clock frequency whenever the hardware
options give an ``lo_freq`` but no ``interm_freq``::

    NCO = clock_freq - lo_freq          # qblox_scheduler's own convention

Clock names follow the transmon element convention: ``q1.01`` -> ``clock_freqs.f01``,
``q1.12`` -> ``.f12``, ``q1.ro`` -> ``.readout``.

Call order (per Qblox tutorial 110_automatic_mixer_calibration)
---------------------------------------------------------------
1. ``module.disconnect_outputs()``, then ``sequencer.connect_out<k>("IQ")``
2. set ``out<k>_lo_freq`` / ``out0_in0_lo_freq``, ``sequencer.nco_freq``, ``mod_en_awg(True)``
3. play a steady tone at ~30 % of full IF amplitude (``--amp``, default 0.3) and CONFIRM the
   sequencer reached ``RUNNING`` -- see "verifying the tone" below
4. ``module.out<k>_lo_cal()``   (QCM-RF)  /  ``module.out0_in0_lo_cal()``  (QRM-RF)
5. ``module.sequencer<n>.sideband_cal()``
6. re-arm and restart whatever was running before

The silent-failure problem this routine is built around
-------------------------------------------------------
``sideband_cal()`` frequently does NOTHING and still reports success: it returns with
``mixer_corr_gain_ratio``/``mixer_corr_phase_offset_degree`` untouched on the vendor
defaults. On chipA (2026-07-27) roughly half to three-quarters of calls did this, and the
same sequencer both worked and did not minutes apart. When it lands the value is solid
(slot 4 ratio 1.0330 +/- 0.0003 over five runs), so the TRIGGERING is unreliable, not the
measurement. ``--attempts`` (default 12) is the knob that makes this dependable; the
attempt count is recorded so a rising trend is visible.

You cannot lean on the LO cal to tell you: LO leakage is DC mixer feedthrough, so
``out<k>_lo_cal()`` produces perfectly plausible ``out<k>_offset_path0/1`` values *with no
tone playing at all*. Plausible LO numbers say nothing about the sideband. Four guards:

* the sequencer must report ``RUNNING`` before either calibration is called;
* a step whose values are still on the vendor defaults afterwards (LO 0/0 mV, sideband
  ratio 1.0 / phase 0.0) is a ``no-op``: never cached, and the CLI exits non-zero. The
  comparison is TOLERANCE-based, not ``==`` -- a failed cal can null out to 1.000031 and
  sail past an equality test;
* the sideband cal is RETRIED (``--attempts``) until the values move, re-arming and
  resetting to the defaults each attempt. The attempt count is recorded in the cache's
  ``history`` -- if it creeps up, that is the number to take to Qblox support;
* the sideband cal runs BEFORE the LO cal, and its result is re-read afterwards to prove
  the LO cal did not undo it.

Output attenuation during calibration
-------------------------------------
The attenuator sits AFTER the mixer (DAC -> mixer -> variable attenuator -> output switch), so
an operating value like the 42 dB on a readout line plausibly puts the image below the AMC
detector. Calibration therefore runs at ``--cal-att`` (default 0 dB) and restores whatever the
module had. (Honest caveat: this looked decisive in one A/B, but later runs showed the cal is
non-deterministic, so 0 dB is the defensible default rather than a proven fix. ``keep`` leaves
attenuation alone.) It is safe because the RF output switch stays OPEN throughout
(``--switch-on`` to close it): the detector is internal, Qblox opens the switch during the
calibration anyway, and suppression is >60 dB, so nothing reaches the fridge. The Qblox
tutorial closes the switch only so the spurs are visible on an analyzer.

Scope and invalidation
----------------------
* LO cal is **per output**, keyed by LO frequency -> lands in ``out<k>_offset_path0/1``
  (module-level state, independent of which sequencer plays).
* Sideband cal is **per sequencer**, keyed by NCO frequency -> lands in
  ``mixer_corr_gain_ratio`` / ``mixer_corr_phase_offset_degree``. Changing the IF
  invalidates it.

Results are cached in ``<config_dir>/mixer_cal.json`` keyed by ``(slot, output, lo_freq)``
and ``(slot, sequencer, nco_freq)``. The cache is validated **against the live hardware
values**, not against a timestamp: a cluster reboot (which clears the corrections) or any
retune shows up as a mismatch and the step re-runs. A stored entry sitting on the vendor
defaults is treated as a miss, so one silent failure cannot poison the file. ``--force``
skips the check entirely. Every run also appends a full before/after record to the file's
``history`` list for cooldown-to-cooldown drift tracking.

Side effects -- schedule this as an explicit calibration step, never mid-experiment
------------------------------------------------------------------------------------
* The output switches are turned OFF during calibration and back ON afterwards.
* Calibration interrupts **all** sequencers in the same module. This script snapshots the
  running sequencers per module and re-arms/restarts them afterwards. Other modules are
  unaffected. Work is serialized per module.
* This connects to the cluster directly through ``qblox_instruments``. Do not run it while a
  ``scqo`` session / ``HardwareAgent`` holds the same cluster.

Expected performance
--------------------
~35 dBc suppression of both LO and image at 30 % IF amplitude. That is the AMC design
floor, not a bug. If the spur budget needs more than that -- an image landing on a
neighbouring qubit or a readout resonator -- calibrate manually against an external
analyzer or a qubit-based probe using ``out<k>_offset_path0/1`` (LO) and
``sequencer.mixer_corr_gain_ratio`` / ``mixer_corr_phase_offset_degree`` (sideband).

Hardware preconditions (they affect the achievable suppression)
---------------------------------------------------------------
* All modules screwed in at the top AND bottom (grounding).
* Every empty cluster slot filled with a metal flow blocker, also screwed in top and bottom.

Requires ``qblox-instruments >= 0.14`` with matching Cluster firmware (this repo runs 1.3.0)
and applies to RF modules only -- baseband QCM/QRM with external mixers must be calibrated
the old way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # runnable without pip-installing scqo_qblox
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Reference conditions from the Qblox AMC tutorial. `--amp` is the PER-PATH
# waveform amplitude, exactly the tutorial's knob: a DC waveform of `amp` is
# played on both AWG paths, so the complex envelope the modulator sees has
# magnitude amp*sqrt(2). 0.3 is the amplitude the 35 dBc figure is quoted at.
# Calibration is amplitude dependent -- calibrate near your operating amplitude.
DEFAULT_IF_AMP = 0.3
TONE_SAMPLES = 1200  # one loop iteration of the steady tone

RF_MODULE_TYPES = {"QCM_RF", "QRM_RF"}

# Marker bit that closes the RF output switch, mirrored from qblox_scheduler's
# `default_markers` (backends/qblox/instrument_compilers.py). A raw Q1ASM upload
# gets no `set_mrk` from the compiler, so we drive it with the marker override.
_OUTPUT_MARKER = {
    ("QCM_RF", 0): 0b0001,
    ("QCM_RF", 1): 0b0010,
    ("QRM_RF", 0): 0b0010,
}

# Sequencer states that mean "this was doing something before we interrupted it".
_ACTIVE_STATES = ("RUNNING", "ARMED", "Q1_STOPPED")

# How long to wait for a started sequencer to report RUNNING before giving up.
_RUNNING_TIMEOUT_S = 0.5
_RUNNING_POLL_S = 0.02

# `sideband_cal()` is NON-DETERMINISTIC on this firmware: the same code, same conditions,
# same sequencer returns a correction sometimes and the untouched defaults other times
# (chipA 2026-07-27: 4 successes in ~11 calls across both modules). The firmware reports
# success either way. So retry against the no-op check until it takes, and let the tone
# settle first. Attempt counts are recorded in mixer_cal.json -- if they creep up, that is
# the number to take to Qblox support.
_SETTLE_BEFORE_CAL_S = 0.0
_CAL_ATTEMPTS = 12

# Vendor defaults. A calibration that leaves these untouched did not do anything.
_LO_DEFAULTS = {"offset_path0": 0.0, "offset_path1": 0.0}
_SIDEBAND_DEFAULTS = {"gain_ratio": 1.0, "phase_offset_degree": 0.0}

# How close to the defaults still counts as "it did nothing". Not exact equality: a failed
# calibration can null out to 1.000031 / -0.0000 (chipA slot 8, 2026-07-27) and sail past an
# == check. These thresholds are ~2 orders of magnitude below any genuine correction -- the
# real ones here are ratio ~0.03-0.07 and phase ~7-23 deg, and a mixer good enough to need
# 0.01% of gain correction would already be 40 dB better than the AMC can measure.
_NOOP_TOL = {
    "offset_path0": 1e-4,
    "offset_path1": 1e-4,
    "gain_ratio": 1e-4,
    "phase_offset_degree": 1e-3,
}

# Tolerance for "the hardware still holds the value we cached".
_OFFSET_TOL_MV = 1e-6
_RATIO_TOL = 1e-9
_PHASE_TOL_DEG = 1e-9
_FREQ_TOL_HZ = 1.0


# ---------------------------------------------------------------------------
# Plan (pure -- no hardware, no vendor imports)
# ---------------------------------------------------------------------------
@dataclass
class SidebandTarget:
    """One port-clock: a sequencer that must be sideband-calibrated at its NCO."""

    portclock: str
    port: str
    clock: str
    nco_freq: float
    sequencer: int

    @property
    def cache_key(self) -> str:  # filled in by OutputGroup.keys(); slot lives there
        return f"{self.sequencer}/{self.nco_freq:.2f}"


@dataclass
class OutputGroup:
    """One physical RF output: one LO cal, plus one sideband cal per sequencer on it."""

    slot: int
    module_type: str
    output: int
    lo_freq: float
    output_att: int | None
    targets: list[SidebandTarget] = field(default_factory=list)

    @property
    def lo_cache_key(self) -> str:
        return f"slot{self.slot}/out{self.output}/lo{self.lo_freq:.0f}"

    def sideband_cache_key(self, target: SidebandTarget) -> str:
        return f"slot{self.slot}/seq{target.sequencer}/nco{target.nco_freq:.2f}"


@dataclass
class Plan:
    cluster_name: str
    ip: str | None
    modules: dict[int, str]  # slot -> instrument_type (every module, RF or not)
    groups: list[OutputGroup]


_CLOCK_SUFFIX_TO_FIELD = {"01": "f01", "12": "f12", "ro": "readout"}


def load_configs(config_dir: str | Path) -> tuple[dict, dict]:
    """Load ``hw_config.json`` + ``dut_config.json`` from *config_dir*.

    Both files carry literal ``NaN`` (qblox's loader expects it); Python's ``json``
    accepts that natively, so no custom parser is needed.
    """
    folder = Path(config_dir)
    hw_path, dut_path = folder / "hw_config.json", folder / "dut_config.json"
    if not hw_path.is_file():
        raise SystemExit(f"no hw_config.json in {folder}")
    hw = json.loads(hw_path.read_text(encoding="utf-8"))
    dut = json.loads(dut_path.read_text(encoding="utf-8")) if dut_path.is_file() else {}
    return hw, dut


def clock_frequency(dut: dict, clock: str) -> float | None:
    """Absolute frequency of a qblox_scheduler clock name (``q1.01``, ``q1.ro``, ...)."""
    element, _, suffix = clock.rpartition(".")
    field_name = _CLOCK_SUFFIX_TO_FIELD.get(suffix)
    if not element or field_name is None:
        return None
    freqs = dut.get("elements", {}).get(element, {}).get("clock_freqs", {})
    value = freqs.get(field_name)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _parse_connectivity(hw: dict, cluster_name: str) -> dict[str, tuple[int, int]]:
    """port -> (slot, complex-output index), from the connectivity graph."""
    prefix = f"{cluster_name}.module"
    port_to_output: dict[str, tuple[int, int]] = {}
    for edge in hw.get("connectivity", {}).get("graph", []):
        if len(edge) != 2:
            continue
        for node, other in ((edge[0], edge[1]), (edge[1], edge[0])):
            if not (isinstance(node, str) and node.startswith(prefix)):
                continue
            _, _, tail = node.partition(prefix)
            slot_txt, _, channel = tail.partition(".")
            if not channel.startswith("complex_output_"):
                continue  # inputs / digital / real channels have no mixer to calibrate
            try:
                slot, output = int(slot_txt), int(channel.rsplit("_", 1)[1])
            except ValueError:
                continue
            port_to_output[other] = (slot, output)
    return port_to_output


def build_plan(
    hw: dict,
    dut: dict,
    *,
    slots: list[int] | None = None,
    portclocks: list[str] | None = None,
    sequencer_map: dict[str, int] | None = None,
) -> Plan:
    """Turn the two config files into an ordered, validated calibration plan.

    Sequencer indices mirror qblox_scheduler's allocation (``_sequencer_to_portclock``):
    the lowest free sequencer on the module, walking port-clocks in config order. With one
    port-clock per module -- chipA today -- that is sequencer 0, exactly what the compiler
    uses. On a multiplexed module the mirror is best-effort; override it with
    ``--sequencer <portclock>=<index>`` if the compiler disagrees.
    """
    descriptions = hw.get("hardware_description", {})
    clusters = {
        name: body
        for name, body in descriptions.items()
        if isinstance(body, dict) and body.get("instrument_type") == "Cluster"
    }
    if len(clusters) != 1:
        raise SystemExit(
            f"expected exactly one Cluster in hardware_description, found {sorted(clusters)}"
        )
    cluster_name, cluster = next(iter(clusters.items()))
    modules = {
        int(slot): body.get("instrument_type", "")
        for slot, body in (cluster.get("modules") or {}).items()
    }

    port_to_output = _parse_connectivity(hw, cluster_name)
    options = hw.get("hardware_options", {})
    modulation = options.get("modulation_frequencies", {}) or {}
    attenuations = options.get("output_att", {}) or {}
    sequencer_map = sequencer_map or {}

    groups: dict[tuple[int, int], OutputGroup] = {}
    next_sequencer: dict[int, int] = {}
    skipped: list[str] = []

    for portclock, entry in modulation.items():
        port, _, clock = portclock.partition("-")
        if not clock:
            skipped.append(f"{portclock}: not a '<port>-<clock>' key")
            continue
        if portclocks and portclock not in portclocks:
            continue
        if port not in port_to_output:
            skipped.append(f"{portclock}: port '{port}' drives no complex output")
            continue
        slot, output = port_to_output[port]
        module_type = modules.get(slot, "")
        if module_type not in RF_MODULE_TYPES:
            skipped.append(f"{portclock}: slot {slot} is {module_type or '?'}, not an RF module")
            continue
        if slots and slot not in slots:
            continue

        lo_freq = entry.get("lo_freq")
        interm = entry.get("interm_freq")
        if lo_freq is None:
            skipped.append(f"{portclock}: no lo_freq (external LO or downconverter?)")
            continue
        lo_freq = float(lo_freq)
        if interm is not None:
            nco = float(interm)
        else:
            absolute = clock_frequency(dut, clock)
            if absolute is None:
                skipped.append(f"{portclock}: no interm_freq and no clock '{clock}' in dut_config")
                continue
            nco = absolute - lo_freq

        if not 2e9 <= lo_freq <= 18e9:
            raise SystemExit(f"{portclock}: lo_freq {lo_freq:.6g} Hz is outside the 2-18 GHz range")
        if abs(nco) > 500e6:
            raise SystemExit(
                f"{portclock}: NCO {nco / 1e6:.3f} MHz is outside the +/-500 MHz range "
                f"(lo_freq={lo_freq / 1e9:.6f} GHz vs clock '{clock}')"
            )
        if module_type == "QRM_RF" and output != 0:
            raise SystemExit(f"{portclock}: QRM-RF in slot {slot} has only complex_output_0")

        group = groups.get((slot, output))
        if group is None:
            att = attenuations.get(portclock)
            group = OutputGroup(
                slot=slot,
                module_type=module_type,
                output=output,
                lo_freq=lo_freq,
                output_att=int(att) if att is not None else None,
            )
            groups[(slot, output)] = group
        elif abs(group.lo_freq - lo_freq) > _FREQ_TOL_HZ:
            raise SystemExit(
                f"slot {slot} output {output}: port-clocks disagree about the LO "
                f"({group.lo_freq:.6g} vs {lo_freq:.6g} Hz) -- one output has one LO"
            )

        if portclock in sequencer_map:
            index = sequencer_map[portclock]
        else:
            index = next_sequencer.get(slot, 0)
            next_sequencer[slot] = index + 1
        group.targets.append(
            SidebandTarget(
                portclock=portclock, port=port, clock=clock, nco_freq=nco, sequencer=index
            )
        )

    for line in skipped:
        print(f"  skip  {line}")
    ordered = sorted(groups.values(), key=lambda g: (g.slot, g.output))
    return Plan(cluster_name=cluster_name, ip=cluster.get("ip"), modules=modules, groups=ordered)


def describe_plan(plan: Plan) -> str:
    lines = [f"cluster '{plan.cluster_name}' @ {plan.ip or '(no ip)'}"]
    for slot in sorted(plan.modules):
        lines.append(f"  slot {slot}: {plan.modules[slot]}")
    if not plan.groups:
        lines.append("  nothing to calibrate")
    for group in plan.groups:
        att = f", out{group.output}_att={group.output_att} dB" if group.output_att is not None else ""
        lines.append(
            f"  slot {group.slot} {group.module_type} out{group.output}: "
            f"LO {group.lo_freq / 1e9:.6f} GHz{att}"
        )
        for target in group.targets:
            lines.append(
                f"      seq{target.sequencer}  {target.portclock:<20} "
                f"NCO {target.nco_freq / 1e6:+.4f} MHz  "
                f"-> RF {(group.lo_freq + target.nco_freq) / 1e9:.6f} GHz"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache (validated against live hardware, not against a timestamp)
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {"lo": {}, "sideband": {}, "history": []}
    cache = json.loads(path.read_text(encoding="utf-8"))
    for key in ("lo", "sideband"):
        cache.setdefault(key, {})
    cache.setdefault("history", [])
    return cache


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")


def _close_enough(live: dict[str, float], cached: dict[str, Any], tolerances: dict[str, float]) -> bool:
    for name, tol in tolerances.items():
        got, want = live.get(name), cached.get(name)
        if got is None or want is None:
            return False
        if not math.isfinite(got) or abs(float(got) - float(want)) > tol:
            return False
    return True


def _is_default(values: dict[str, Any], defaults: dict[str, float]) -> bool:
    """True when every calibrated field is still (within noise) on its vendor default.

    Used both ways: a fresh result that is all-defaults is a NO-OP (the calibration
    ran but found nothing), and a CACHED entry that is all-defaults is a recorded
    no-op, which must not be served as a hit -- otherwise one bad run poisons the
    cache and every later run reports 'cached' for a calibration that never happened.
    """
    for name, default in defaults.items():
        value = values.get(name)
        if value is None or abs(float(value) - default) > _NOOP_TOL[name]:
            return False
    return True


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def _tone_sequence(amp: float) -> dict:
    """Steady DC tone on both AWG paths, looping forever (Qblox tutorial 110)."""
    return {
        "waveforms": {"dc": {"data": [amp] * TONE_SAMPLES, "index": 0}},
        "weights": {},
        "acquisitions": {},
        "program": f"      wait_sync 4\nloop: play    0,0,{TONE_SAMPLES}\n      jmp     @loop\n",
    }


def _lo_api(module: Any, output: int) -> tuple[str, str, str]:
    """(lo_freq param, cal-type-default param, lo_cal method) for this module kind.

    QRM-RF shares one LO between out0 and in0 -> ``out0_in0_*``.
    QCM-RF has an independent LO per output   -> ``out<k>_*``.
    """
    if module.is_qrm_type:
        return "out0_in0_lo_freq", "out0_in0_lo_freq_cal_type_default", "out0_in0_lo_cal"
    return f"out{output}_lo_freq", f"out{output}_lo_freq_cal_type_default", f"out{output}_lo_cal"


def _lo_enable_param(module: Any, output: int) -> str:
    return "out0_in0_lo_en" if module.is_qrm_type else f"out{output}_lo_en"


def _read_lo_state(module: Any, output: int) -> dict[str, float]:
    return {
        "offset_path0": float(module.parameters[f"out{output}_offset_path0"]()),
        "offset_path1": float(module.parameters[f"out{output}_offset_path1"]()),
    }


def _read_sideband_state(sequencer: Any) -> dict[str, float]:
    return {
        "gain_ratio": float(sequencer.mixer_corr_gain_ratio()),
        "phase_offset_degree": float(sequencer.mixer_corr_phase_offset_degree()),
    }


def _running_sequencers(module: Any) -> list[int]:
    active = []
    for index in range(len(module.sequencers)):
        try:
            status = module.get_sequencer_status(index)
        except Exception:  # noqa: BLE001 - a status we cannot read is not restartable
            continue
        state = str(getattr(status, "state", status))
        if any(name in state for name in _ACTIVE_STATES):
            active.append(index)
    return active


def _wait_running(module: Any, index: int) -> str | None:
    """Poll until the sequencer reports RUNNING. Returns the state, or None on a dummy.

    Without this the whole routine fails SILENTLY: LO leakage is DC mixer feedthrough
    and calibrates fine with no tone at all, so plausible ``out{k}_offset_path*`` values
    prove nothing about whether the AWG is actually playing. The sideband cal, which
    needs a real tone to find an image of, then returns the untouched defaults and the
    firmware reports success.
    """
    if module.is_dummy:
        return None  # the dummy transport always reports STOPPED
    deadline = time.monotonic() + _RUNNING_TIMEOUT_S
    state = ""
    while True:
        status = module.get_sequencer_status(index)
        state = str(getattr(status, "state", status))
        if "RUNNING" in state:
            return state
        if time.monotonic() >= deadline:
            return state
        time.sleep(_RUNNING_POLL_S)


def _require_running(module: Any, index: int) -> None:
    state = _wait_running(module, index)
    if state is None or "RUNNING" in state:
        return
    try:
        status = module.get_sequencer_status(index)
        detail = (
            f"state={getattr(status, 'state', status)} "
            f"err={getattr(status, 'err_flags', '?')} "
            f"warn={getattr(status, 'warn_flags', '?')} "
            f"log={getattr(status, 'log', '?')}"
        )
    except Exception as err:  # noqa: BLE001 - diagnostics must not mask the real failure
        detail = f"(status unreadable: {err})"
    try:
        assembler = module.get_assembler_log()
    except Exception:  # noqa: BLE001
        assembler = "(unavailable)"
    raise SystemExit(
        f"slot {module.slot_idx} seq{index} never reached RUNNING within "
        f"{_RUNNING_TIMEOUT_S:.1f} s -- no tone, so the sideband calibration would "
        f"silently do nothing.\n  {detail}\n  assembler log: {assembler}"
    )


def _play_tone(
    module: Any,
    sequencer: Any,
    index: int,
    *,
    marker: int,
    clear_flags: bool = False,
    require_running: bool = True,
) -> str | None:
    """Start the steady tone on one sequencer and confirm it is really playing.

    ``clear_flags`` is OFF by default because on chipA it BREAKS the calibration.
    `--diagnose` on the cluster (2026-07-27, slot 4 QCM-RF) ran identical conditions with
    and without it: without, ``sideband_cal()`` returned ratio 1.0330 / phase -6.96 deg;
    with, it returned the untouched defaults. It was added on the strength of the Qblox
    FAQ noting stale flags can stop a sequencer running -- but the sequencer reaches
    RUNNING here either way, so the FAQ's problem is not the one we have. Kept as an
    option for the diagnose matrix.
    """
    if clear_flags:
        module.clear_sequencer_flags(index)
    sequencer.marker_ovr_en(True)
    sequencer.marker_ovr_value(marker)
    module.arm_sequencer(index)
    module.start_sequencer(index)
    if require_running:
        _require_running(module, index)
        return None
    return _wait_running(module, index)


def _stop_tone(module: Any, sequencer: Any, index: int) -> None:
    module.stop_sequencer(index)
    sequencer.marker_ovr_en(False)


def _arm_tone(module: Any, group: OutputGroup, target: SidebandTarget, amp: float) -> Any:
    """Configure one sequencer for the tone (channel map, NCO, waveform). No playing."""
    sequencer = module.sequencers[target.sequencer]
    sequencer.nco_freq_cal_type_default("off")
    sequencer.parameters[f"connect_out{group.output}"]("IQ")
    if "connect_acq" in sequencer.parameters:
        # QRM-RF only. The Qblox tutorial connects a QRM-RF sequencer with
        # connect_sequencer("io0") -- output AND acquisition -- where a QCM-RF gets
        # "out0". A QRM measures with its own ADC, so the sideband detector needs the
        # receive path: with the output alone, slot 8 no-op'd 5/5 while the QCM-RF in
        # slot 4 calibrated first try on the same run (chipA 2026-07-27).
        sequencer.connect_acq("in0")
    sequencer.mod_en_awg(True)
    sequencer.nco_freq(target.nco_freq)
    sequencer.sync_en(True)
    sequencer.sequence(_tone_sequence(amp))
    return sequencer


def _cached_hit(cache_section: dict, key: str, live: dict, tolerances: dict, defaults: dict,
                *, force: bool) -> bool:
    """Is *live* still the calibration we recorded under *key*?

    A stored entry sitting on the vendor defaults is a recorded NO-OP, never a hit --
    that is what heals a cache file poisoned by an earlier silent failure.
    """
    if force:
        return False
    entry = cache_section.get(key)
    if entry is None or _is_default(entry, defaults):
        return False
    return _close_enough(live, entry, tolerances)


def calibrate_group(
    module: Any,
    group: OutputGroup,
    *,
    amp: float,
    cache: dict,
    force: bool,
    do_lo: bool,
    do_sideband: bool,
    cal_att: int | None = 0,
    switch_on: bool = False,
    attempts: int = _CAL_ATTEMPTS,
) -> list[dict]:
    """Calibrate one physical output. Returns one record per step performed/skipped.

    ``cal_att`` is the output attenuation held DURING the calibration (``None`` leaves it
    alone) -- the attenuator sits after the mixer, so a high operating value can bury the
    image below the AMC detector while leaving LO leakage perfectly detectable. Whatever
    the module had is restored afterwards.

    ``switch_on`` closes the RF output switch while the tone plays. Off by default: the
    detector is internal (Qblox opens the switch during the calibration anyway), the Qblox
    tutorial only closes it so the spurs are visible on an analyzer, and with the switch
    open nothing reaches the fridge no matter how low ``cal_att`` is.
    """
    records: list[dict] = []
    lo_param, lo_cal_default, lo_cal_method = _lo_api(module, group.output)
    att_param = f"out{group.output}_att"
    if module.is_dummy:
        attempts = 1  # a dummy can never move the values; retrying is pure wall-clock

    # -- 2. LO / NCO. cal_type_default OFF: the explicit calls below are the only
    #       calibration that runs, so nothing fires behind our back.
    module.parameters[lo_cal_default]("off")
    module.parameters[_lo_enable_param(module, group.output)](True)
    module.parameters[lo_param](group.lo_freq)

    saved_att = module.parameters[att_param]()
    if cal_att is not None and cal_att != saved_att:
        module.parameters[att_param](cal_att)
        print(
            f"    slot {group.slot} out{group.output} attenuation {saved_att} -> {cal_att} dB "
            f"for the calibration (restored after)"
        )

    marker = _OUTPUT_MARKER[(group.module_type, group.output)] if switch_on else 0
    primary_target = group.targets[0]
    primary = module.sequencers[primary_target.sequencer]

    def run_lo() -> None:
        """-- 3./4. LO leakage, per output, keyed by LO frequency."""
        before = _read_lo_state(module, group.output)
        label = f"slot {group.slot} out{group.output} LO "
        if _cached_hit(cache["lo"], group.lo_cache_key, before,
                       {"offset_path0": _OFFSET_TOL_MV, "offset_path1": _OFFSET_TOL_MV},
                       _LO_DEFAULTS, force=force):
            print(
                f"    {label} cached (path0={before['offset_path0']:+.4f} mV, "
                f"path1={before['offset_path1']:+.4f} mV)"
            )
            records.append({"step": "lo", "key": group.lo_cache_key, "status": "cached", **before})
            return
        _arm_tone(module, group, primary_target, amp)
        _play_tone(module, primary, primary_target.sequencer, marker=marker)
        getattr(module, lo_cal_method)()
        _stop_tone(module, primary, primary_target.sequencer)
        after = _read_lo_state(module, group.output)
        entry = {"slot": group.slot, "output": group.output, "lo_freq": group.lo_freq, **after}
        no_op = _is_default(after, _LO_DEFAULTS)
        if no_op:
            print(
                f"    {label} WARNING: still at the vendor defaults (path0=0, path1=0) -- "
                f"the calibration did nothing. NOT cached; run --diagnose to find out why."
            )
        else:
            cache["lo"][group.lo_cache_key] = entry
            print(
                f"    {label} path0 {before['offset_path0']:+.4f} -> "
                f"{after['offset_path0']:+.4f} mV, path1 "
                f"{before['offset_path1']:+.4f} -> {after['offset_path1']:+.4f} mV"
            )
        records.append(
            {
                "step": "lo",
                "key": group.lo_cache_key,
                "status": "no-op" if no_op else "calibrated",
                "before": before,
                "after": after,
                **entry,
            }
        )

    def run_sideband() -> None:
        """-- 5. Image sideband, per sequencer, keyed by NCO frequency."""
        for target in group.targets:
            sequencer = module.sequencers[target.sequencer]
            key = group.sideband_cache_key(target)
            before = _read_sideband_state(sequencer)
            label = f"slot {group.slot} seq{target.sequencer} {target.portclock} sideband"
            if _cached_hit(cache["sideband"], key, before,
                           {"gain_ratio": _RATIO_TOL, "phase_offset_degree": _PHASE_TOL_DEG},
                           _SIDEBAND_DEFAULTS, force=force):
                print(
                    f"    {label} cached (ratio={before['gain_ratio']:.6f}, "
                    f"phase={before['phase_offset_degree']:+.4f} deg)"
                )
                records.append({"step": "sideband", "key": key, "status": "cached", **before})
                continue
            for attempt in range(1, attempts + 1):
                # Re-arm before EVERY attempt: a calibration interrupts the module's
                # sequencers, so "stopped then re-armed" is not the same as a fresh upload.
                _arm_tone(module, group, target, amp)
                # Reset to the vendor defaults so each attempt measures the raw imbalance
                # rather than an already-corrected signal, and so "did it move?" is
                # unambiguous.
                sequencer.mixer_corr_gain_ratio(_SIDEBAND_DEFAULTS["gain_ratio"])
                sequencer.mixer_corr_phase_offset_degree(
                    _SIDEBAND_DEFAULTS["phase_offset_degree"]
                )
                _play_tone(module, sequencer, target.sequencer, marker=marker)
                time.sleep(_SETTLE_BEFORE_CAL_S)
                sequencer.sideband_cal()
                _stop_tone(module, sequencer, target.sequencer)
                after = _read_sideband_state(sequencer)
                if not _is_default(after, _SIDEBAND_DEFAULTS):
                    break
                if attempt < attempts:
                    print(f"    {label} attempt {attempt}/{attempts} did nothing, retrying")
            entry = {
                "slot": group.slot,
                "sequencer": target.sequencer,
                "portclock": target.portclock,
                "nco_freq": target.nco_freq,
                "attempts": attempt,
                **after,
            }
            no_op = _is_default(after, _SIDEBAND_DEFAULTS)
            if no_op:
                print(
                    f"    {label} WARNING: still at the vendor defaults (ratio=1.0, "
                    f"phase=0.0) after {attempt} attempts -- NOT cached. Re-run, or use "
                    f"--diagnose."
                )
            else:
                cache["sideband"][key] = entry
                tries = "" if attempt == 1 else f"  ({attempt} attempts)"
                print(
                    f"    {label} ratio {before['gain_ratio']:.6f} -> "
                    f"{after['gain_ratio']:.6f}, phase "
                    f"{before['phase_offset_degree']:+.4f} -> "
                    f"{after['phase_offset_degree']:+.4f} deg{tries}"
                )
            records.append(
                {
                    "step": "sideband",
                    "key": key,
                    "status": "no-op" if no_op else "calibrated",
                    "before": before,
                    "after": after,
                    **entry,
                }
            )

    def check_sideband_survived_the_lo_cal() -> None:
        """The LO cal runs last because it may disturb the sideband state; verify it did not
        undo a correction we just measured, rather than assuming."""
        calibrated = {
            r["key"] for r in records if r["step"] == "sideband" and r["status"] == "calibrated"
        }
        for target in group.targets:
            key = group.sideband_cache_key(target)
            if key not in calibrated:
                continue  # nothing succeeded here, so there is nothing to lose
            live = _read_sideband_state(module.sequencers[target.sequencer])
            if _is_default(live, _SIDEBAND_DEFAULTS):
                cache["sideband"].pop(key)
                for record in records:
                    if record.get("key") == key:
                        record["status"] = "no-op"
                print(
                    f"    slot {group.slot} seq{target.sequencer} {target.portclock} sideband "
                    f"WARNING: the LO calibration reset it back to the defaults. "
                    f"Un-cached; run with --sideband-only to keep it."
                )

    try:
        # ORDER MATTERS: sideband FIRST, LO second. Running the LO cal on a sequencer
        # first makes the following sideband_cal() return the defaults -- chipA
        # 2026-07-27: sideband-only produced ratio 1.0331, the same code with the LO cal
        # ahead of it produced nothing on slot 4 and nulled slot 8 to 1.000031.
        if do_sideband:
            run_sideband()
        if do_lo:
            run_lo()
            if do_sideband:
                check_sideband_survived_the_lo_cal()
    finally:
        if cal_att is not None and cal_att != saved_att:
            module.parameters[att_param](saved_att)

    return records


def calibrate_cluster(
    cluster: Any,
    plan: Plan,
    *,
    amp: float = DEFAULT_IF_AMP,
    cache: dict | None = None,
    force: bool = False,
    do_lo: bool = True,
    do_sideband: bool = True,
    cal_att: int | None = 0,
    switch_on: bool = False,
    attempts: int = _CAL_ATTEMPTS,
) -> list[dict]:
    """Run the plan against an open ``qblox_instruments.Cluster``.

    Serialized per module: every output of a module is done before moving on, because
    calibration interrupts every sequencer in that module. Sequencers that were running
    are snapshotted and restarted afterwards; sequencers in other modules never stop.
    """
    cache = cache if cache is not None else {"lo": {}, "sideband": {}, "history": []}
    records: list[dict] = []
    by_slot: dict[int, list[OutputGroup]] = {}
    for group in plan.groups:
        by_slot.setdefault(group.slot, []).append(group)

    for slot in sorted(by_slot):
        module = cluster.modules[slot - 1]
        if not module.present():
            raise SystemExit(f"slot {slot} reports no module present")
        if not module.is_rf_type:
            raise SystemExit(f"slot {slot} is not an RF module -- AMC needs QCM-RF / QRM-RF")
        live_type = f"{module.module_type}_RF"
        declared = by_slot[slot][0].module_type
        if live_type != declared:
            raise SystemExit(
                f"slot {slot}: hw_config.json says {declared}, the cluster reports {live_type}"
            )
        was_running = _running_sequencers(module)
        commandeered = {t.sequencer for g in by_slot[slot] for t in g.targets}
        print(f"  slot {slot} {live_type} (running sequencers: {was_running or 'none'})")

        # -- 1. Clear the channel map so no stale sequencer drives this output.
        module.disconnect_outputs()
        if module.is_qrm_type:
            module.disconnect_inputs()  # the QRM-RF acquisition path is remapped below
        try:
            for group in by_slot[slot]:
                records += calibrate_group(
                    module,
                    group,
                    amp=amp,
                    cache=cache,
                    force=force,
                    do_lo=do_lo,
                    do_sideband=do_sideband,
                    cal_att=cal_att,
                    switch_on=switch_on,
                    attempts=attempts,
                )
        finally:
            # -- 6. Re-arm/restart what we interrupted, so nothing is silently left
            #       stopped. NOT the sequencers we borrowed for the tone: their program
            #       is now the calibration tone, and restarting that would play a steady
            #       carrier into the fridge forever. They need a fresh upload, which the
            #       next `scqo run` does anyway (it recompiles from hw_config).
            for index in was_running:
                if index in commandeered:
                    print(
                        f"    NOTE: slot {slot} seq{index} was running but now holds the "
                        f"calibration tone -- left stopped, re-upload before using it"
                    )
                    continue
                try:
                    module.arm_sequencer(index)
                    module.start_sequencer(index)
                except Exception as err:  # noqa: BLE001 - report, do not mask a cal failure
                    print(f"    WARNING: could not restart slot {slot} seq{index}: {err}")
    return records


# ---------------------------------------------------------------------------
# Diagnose -- why did sideband_cal() leave the vendor defaults?
# ---------------------------------------------------------------------------
#: (label, cal_att, switch_on, sync_en, clear_flags). ``cal_att=None`` = the config value.
#: Ordered so the FIRST trial that moves ``mixer_corr_*`` names the cause: A is the
#: original failing behaviour, then one suspect is removed at a time. C is what the
#: production path now does.
#:
#: chipA result, slot 4 QCM-RF, 2026-07-27 -- A no change, B/C/D MOVED (ratio ~1.034,
#: phase ~-7.0 deg), E no change. So the AMC detector sits AFTER the variable attenuator
#: (8 dB was already enough to hide the image), and clear_sequencer_flags breaks the cal.
DIAGNOSE_TRIALS: list[tuple[str, int | None, bool, bool, bool]] = [
    ("A  config att, switch on,  sync_en=True,  flags kept", None, True, True, False),
    ("B  0 dB att,   switch on,  sync_en=True,  flags kept", 0, True, True, False),
    ("C  0 dB att,   switch off, sync_en=True,  flags kept", 0, False, True, False),
    ("D  0 dB att,   switch off, sync_en=False, flags kept", 0, False, False, False),
    ("E  0 dB att,   switch off, sync_en=False, flags cleared", 0, False, False, True),
]


def diagnose(cluster: Any, plan: Plan, *, amp: float = DEFAULT_IF_AMP) -> list[dict]:
    """A/B the suspects for a no-op sideband calibration on ONE port-clock.

    Writes no cache and restores attenuation, markers and the running sequencers. Each
    trial resets ``mixer_corr_*`` to the vendor defaults first, so "moved" is unambiguous.
    """
    group = plan.groups[0]
    target = group.targets[0]
    module = cluster.modules[group.slot - 1]
    att_param = f"out{group.output}_att"
    lo_param, lo_cal_default, _ = _lo_api(module, group.output)
    on_marker = _OUTPUT_MARKER[(group.module_type, group.output)]

    print(
        f"\ndiagnosing slot {group.slot} {group.module_type} out{group.output} "
        f"seq{target.sequencer} {target.portclock}\n"
        f"  LO {group.lo_freq / 1e9:.6f} GHz, NCO {target.nco_freq / 1e6:+.4f} MHz, "
        f"tone amp {amp} per path\n"
    )

    was_running = _running_sequencers(module)
    saved_att = module.parameters[att_param]()
    module.disconnect_outputs()
    if module.is_qrm_type:
        module.disconnect_inputs()
    module.parameters[lo_cal_default]("off")
    module.parameters[_lo_enable_param(module, group.output)](True)
    module.parameters[lo_param](group.lo_freq)

    results: list[dict] = []
    try:
        for label, cal_att, switch_on, sync_en, clear_flags in DIAGNOSE_TRIALS:
            # `None` means the trial reproduces the ORIGINAL failing condition, which used
            # the config's operating attenuation -- fall back to the live value only when
            # the config does not name one.
            att = (group.output_att if group.output_att is not None else saved_att) \
                if cal_att is None else cal_att
            module.parameters[att_param](att)
            sequencer = _arm_tone(module, group, target, amp)
            sequencer.sync_en(sync_en)
            # Known starting point, so "did it move?" needs no interpretation.
            sequencer.mixer_corr_gain_ratio(_SIDEBAND_DEFAULTS["gain_ratio"])
            sequencer.mixer_corr_phase_offset_degree(_SIDEBAND_DEFAULTS["phase_offset_degree"])

            state = _play_tone(
                module,
                sequencer,
                target.sequencer,
                marker=on_marker if switch_on else 0,
                clear_flags=clear_flags,
                require_running=False,
            )
            sequencer.sideband_cal()
            _stop_tone(module, sequencer, target.sequencer)
            after = _read_sideband_state(sequencer)
            moved = not _is_default(after, _SIDEBAND_DEFAULTS)
            results.append({"trial": label, "att": att, "state": state, "moved": moved, **after})
            print(
                f"  {label}\n"
                f"      sequencer {state or 'n/a (dummy)'} | "
                f"ratio {after['gain_ratio']:.6f}, phase "
                f"{after['phase_offset_degree']:+.4f} deg | "
                f"{'MOVED' if moved else 'no change'}"
            )
    finally:
        module.parameters[att_param](saved_att)
        for index in was_running:
            if index == target.sequencer:
                continue  # holds the diagnostic tone now
            try:
                module.arm_sequencer(index)
                module.start_sequencer(index)
            except Exception as err:  # noqa: BLE001
                print(f"  WARNING: could not restart slot {group.slot} seq{index}: {err}")

    winner = next((r for r in results if r["moved"]), None)
    print()
    if winner is None and module.is_dummy:
        print("  VERDICT: dummy cluster -- the cal calls are no-ops, so no trial can move.")
    elif winner is None:
        never_ran = all("RUNNING" not in (r["state"] or "") for r in results)
        print(
            "  VERDICT: no trial produced a correction.\n"
            + (
                "  The sequencer never reached RUNNING -- the tone is not playing at all.\n"
                "  Look at the states above, not at the attenuation.\n"
                if never_ran
                else "  The tone played but the AMC found nothing. Try a larger --amp, or\n"
                "  calibrate manually against an analyzer (see the module docstring).\n"
            )
        )
    else:
        print(f"  VERDICT: {winner['trial'].split()[0]} is the first trial that works.")
        print(f"  -> {winner['trial']}")
        if winner is results[0]:
            print("  Trial A is today's default behaviour, so the cache was masking a fluke.")
    return results


def close_cluster(cluster: Any, timeout_s: float = 15.0) -> None:
    """Close the connection, but never block forever doing it.

    ``Cluster.close()`` ends up in ``run_coroutine_threadsafe(...).result()`` with no
    timeout, so a slot that does not answer hangs the close -- and a hung process keeps its
    four sockets open, which makes the NEXT run contend, which makes it hang too. That
    cascade is what a leaked connection actually costs (2026-07-27: eleven "finished" runs
    still alive holding 44 sockets).

    A WATCHDOG, not a worker thread: ``close()`` must run on the thread that owns the
    transport's loop -- with no loop running ``_run_in_loop`` uses ``run_until_complete``,
    which only works from the owning thread -- so the timer force-exits instead of the
    close being moved off-thread.
    """
    import threading

    def give_up() -> None:
        print(
            f"  WARNING: cluster.close() did not return within {timeout_s:.0f} s -- "
            f"killing the process so the connection is released."
        )
        _hard_exit(2)

    watchdog = threading.Timer(timeout_s, give_up)
    watchdog.daemon = True
    watchdog.start()
    try:
        cluster.close()
    except Exception as err:  # noqa: BLE001 - closing must not mask the run's result
        print(f"  WARNING: cluster.close() raised: {type(err).__name__}: {err}")
    finally:
        watchdog.cancel()


def _hard_exit(code: int) -> None:
    """Terminate now, releasing every socket, instead of trusting interpreter shutdown.

    Two things go wrong at normal shutdown here, and both leak the cluster connection:
    qblox's transport keeps a module-level daemon loop thread whose close can block, and
    on Windows a bare ProactorEventLoop dies noisily (``OSError: [WinError 87]``). During
    the 2026-07-27 session eleven "finished" runs were still alive holding 44 sockets to
    the cluster, and the resulting contention is the best explanation for the calibration's
    apparent flakiness. The OS closes sockets on process death; nothing else here needs a
    destructor to run.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def open_cluster(plan: Plan, *, ip: str | None, dummy: bool, name: str = "mixercal") -> Any:
    """Open (and own) a Cluster connection. Caller must ``close()`` it."""
    from qblox_instruments import Cluster, ClusterType

    from qcodes.instrument import Instrument

    try:
        Instrument.find_instrument(name).close()
    except KeyError:
        pass

    if dummy:
        types = {
            "QCM": ClusterType.CLUSTER_QCM,
            "QCM_RF": ClusterType.CLUSTER_QCM_RF,
            "QRM": ClusterType.CLUSTER_QRM,
            "QRM_RF": ClusterType.CLUSTER_QRM_RF,
            "QTM": ClusterType.CLUSTER_QTM,
            "QDM": ClusterType.CLUSTER_QDM,
            "QRC": ClusterType.CLUSTER_QRC,
            "QSM": ClusterType.CLUSTER_QSM,
        }
        dummy_cfg = {slot: types[kind] for slot, kind in plan.modules.items() if kind in types}
        print(f"  dummy cluster: {[(s, t.value) for s, t in sorted(dummy_cfg.items())]}")
        return Cluster(name, dummy_cfg=dummy_cfg)

    identifier = ip or plan.ip
    if not identifier:
        raise SystemExit("no cluster ip in hw_config.json and none given with --ip")
    print(f"  connecting to {identifier}")
    return Cluster(name, identifier=identifier)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_sequencer_overrides(values: list[str] | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in values or []:
        portclock, _, index = item.partition("=")
        if not index.strip().isdigit():
            raise SystemExit(f"--sequencer expects '<portclock>=<index>', got '{item}'")
        mapping[portclock.strip()] = int(index)
    return mapping


def main(argv: list[str] | None = None) -> int:
    # Line-buffer stdout even when piped. A calibration run talks to hardware for tens of
    # seconds per step, so it is normally watched live; and if the run is interrupted the
    # progress printed so far is the only record of how far it got. Block buffering loses
    # all of it, which is exactly what happens when a wrapper kills the process on a timeout.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # not a real stream (captured in tests)
        pass

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config_dir", help="folder holding hw_config.json + dut_config.json")
    parser.add_argument("--slot", type=int, action="append", help="restrict to this slot (repeatable)")
    parser.add_argument(
        "--port-clock", action="append", help="restrict to this port-clock (repeatable)"
    )
    parser.add_argument(
        "--sequencer",
        action="append",
        metavar="PORTCLOCK=N",
        help="override the assumed sequencer index for a port-clock (repeatable)",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=DEFAULT_IF_AMP,
        help=f"per-path IF tone amplitude (default {DEFAULT_IF_AMP}; the 35 dBc spec is "
        f"quoted here -- calibration is amplitude dependent)",
    )
    parser.add_argument(
        "--cal-att",
        default="0",
        metavar="DB|keep",
        help="output attenuation held DURING the calibration, restored after (default 0 -- "
        "the attenuator sits after the mixer, so an operating value like 42 dB can bury the "
        "image below the AMC detector; 'keep' leaves it alone)",
    )
    parser.add_argument(
        "--switch-on",
        action="store_true",
        help="close the RF output switch while the tone plays (default off: the AMC detector "
        "is internal, so with the switch open nothing reaches the fridge). Only needed when "
        "you want to watch the spurs on an analyzer.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=_CAL_ATTEMPTS,
        help=f"how many times to retry a sideband calibration that does nothing "
        f"(default {_CAL_ATTEMPTS}; the firmware routine only lands some of the time, so "
        f"this is the main reliability knob)",
    )
    parser.add_argument("--lo-only", action="store_true", help="LO leakage only, no sideband cal")
    parser.add_argument("--sideband-only", action="store_true", help="sideband cal only, no LO cal")
    parser.add_argument("--force", action="store_true", help="recalibrate even if the cache matches")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="A/B the suspects for a no-op sideband calibration on ONE port-clock and report "
        "which condition makes mixer_corr_* move; writes no cache",
    )
    parser.add_argument("--dummy", action="store_true", help="run against a dummy cluster (CI)")
    parser.add_argument("--ip", help="override the cluster ip from hw_config.json")
    parser.add_argument(
        "--cache", help="cache/history file (default <config_dir>/mixer_cal.json)"
    )
    parser.add_argument("--no-cache-write", action="store_true", help="do not write the cache file")
    args = parser.parse_args(argv)

    if args.lo_only and args.sideband_only:
        raise SystemExit("--lo-only and --sideband-only are mutually exclusive")
    if args.attempts < 1:
        raise SystemExit(f"--attempts must be >= 1, got {args.attempts}")
    if not 0 < args.amp <= 1.0:
        raise SystemExit(f"--amp must be in (0, 1], got {args.amp}")
    if args.cal_att == "keep":
        cal_att: int | None = None
    elif args.cal_att.isdigit() and int(args.cal_att) % 2 == 0:
        cal_att = int(args.cal_att)
    else:
        raise SystemExit(
            f"--cal-att expects 'keep' or an even dB value (the attenuator steps in 2 dB), "
            f"got '{args.cal_att}'"
        )

    config_dir = Path(args.config_dir)
    hw, dut = load_configs(config_dir)
    plan = build_plan(
        hw,
        dut,
        slots=args.slot,
        portclocks=args.port_clock,
        sequencer_map=_parse_sequencer_overrides(args.sequencer),
    )
    print(describe_plan(plan))
    if not plan.groups:
        raise SystemExit("nothing matched -- check --slot / --port-clock and the connectivity graph")
    if args.dry_run:
        return 0

    print(
        "\nPRECONDITIONS: every module screwed in top AND bottom, every empty slot filled\n"
        "with a screwed-in metal flow blocker, and no scqo session / HardwareAgent holding\n"
        "this cluster. Output switches go OFF during calibration.\n"
    )

    if args.diagnose:
        cluster = open_cluster(plan, ip=args.ip, dummy=args.dummy)
        try:
            diagnose(cluster, plan, amp=args.amp)
        finally:
            close_cluster(cluster)
        if args.dummy:
            print("\nNOTE: dummy cluster -- the cal calls are no-ops, so no trial can move.")
        return 0

    cache_path = Path(args.cache) if args.cache else config_dir / "mixer_cal.json"
    cache = load_cache(cache_path)

    cluster = open_cluster(plan, ip=args.ip, dummy=args.dummy)
    try:
        records = calibrate_cluster(
            cluster,
            plan,
            amp=args.amp,
            cache=cache,
            force=args.force,
            do_lo=not args.sideband_only,
            do_sideband=not args.lo_only,
            cal_att=cal_att,
            switch_on=args.switch_on,
            attempts=args.attempts,
        )
    finally:
        close_cluster(cluster)

    counts = Counter(record["status"] for record in records)
    cache["history"].append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_dir": str(config_dir),
            "cluster": plan.cluster_name,
            "ip": None if args.dummy else (args.ip or plan.ip),
            "dummy": args.dummy,
            "amp": args.amp,
            "cal_att": cal_att,
            "switch_on": args.switch_on,
            "records": records,
        }
    )
    if args.no_cache_write:
        print("\n(cache not written: --no-cache-write)")
    else:
        save_cache(cache_path, cache)
        print(f"\ncache + history -> {cache_path}")

    print(
        f"{counts['calibrated']} calibrated, {counts['cached']} cached, "
        f"{counts['no-op']} no-op"
    )
    if args.dummy:
        print(
            "NOTE: on a dummy cluster the calibration calls are no-ops -- this only\n"
            "      exercises the control flow, the sequencer restart and the cache."
        )
        return 0
    if counts["no-op"]:
        print(
            "\nFAILED: the calibration ran but changed nothing on the steps above.\n"
            "Nothing was cached, so a re-run retries. Start with:\n"
            f"    python {Path(__file__).name} {args.config_dir} --diagnose"
        )
        return 1
    return 0


if __name__ == "__main__":
    # Run the (synchronous) CLI inside a loop on purpose. qblox_instruments picks its
    # event loop in Transport.__init__: with no loop running it makes a bare
    # ProactorEventLoop per transport and drives it with repeated run_until_complete,
    # and on Windows those loops die at interpreter shutdown with
    # "OSError: [WinError 87]" out of _loop_self_reading. With a loop already running it
    # takes the threaded-loop branch instead -- the same one Jupyter gets, which is why
    # the notebook never showed this.
    async def _amain() -> int:
        return main()

    _hard_exit(asyncio.run(_amain()))
