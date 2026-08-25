"""What volts a Qblox flux port may emit — the sibling of scqo-qm's module.

Same two frame-named entry points as the QM backend, so one vocabulary describes
both instruments:

* :func:`check_flux_bias_absolute` — the swept value IS the line voltage
  (``FluxSweepParameters``);
* :func:`check_flux_pulse_relative` — the swept value is an excursion the DAC adds
  to the standing ``idle_flux`` (``FluxPulseSweepParameters``).

Three Qblox-specific facts drive the design, all verified 2026-07-29/30:

**1. The rail is fixed by the MODULE, not switchable.** There is no analogue of the
OPX1000's direct/amplified output mode: a baseband QCM output is 5 Vpp = +/-2.5 V
peak into 50 ohm, full stop, and to change it you change the module or the wiring.
Programmable attenuators exist only on RF modules. The RF modules cannot emit DC
at all, so a flux port on one is a wiring error, not a range problem.

**2. Everything above the hardware is UNITLESS.** ``VoltageOffset``'s operand is
documented as "the unitless amplitude"; ``offset_awg_path{x}`` is
``Numbers(-1.0, 1.0)`` with ``unit=""``; the scheduler's own
``max_awg_output_voltage`` is dead metadata (declared five times, read zero
times). Nothing in the stack converts. So a probe that passes ``flux_bias_v``
straight through emits ``value * rail`` volts, and a requested 0.3 V leaves a QCM
as 0.75 V. :func:`to_dac_fraction` is the missing conversion, kept deliberately
separate from the checkers — a function that validates AND silently rescales is
two jobs, and the rescale is the one you want visible at the call site.

**3. Nothing downstream refuses.** Measured, not assumed: a +/-0.9 domain (=
+/-2.25 V on a QCM) compiles clean, and +/-3.0 dies inside the compiler with
``ufunc 'absolute' did not contain a loop ... StrDType`` — an internal numpy error
naming no port, no volts and no DAC. There is no usable backstop in either
direction, which is why these checks must fire BEFORE compilation.

Offsets and played waveforms SUM at the DAC (the AWG adds ``y = x + O + o``), and
clipping is silent saturation: it flat-tops at the rails, raises only a sticky
``OUTPUT_OVERFLOW`` sequencer flag you have to poll, and never raises in Python.
That is the exact analogue of the OPX1000's idle-bias + pulse-excursion hazard.
"""

from __future__ import annotations

from typing import Any

#: Baseband output full scale (V, peak, into 50 ohm), by module type. The QCM's
#: 5 Vpp spec is PEAK-TO-PEAK; the guard compares a single-ended voltage, so it
#: takes half. Getting that wrong is a silent factor of two in the permissive
#: direction. Sourced from the vendor spec table (qcm.md "Output range | In 50Ohm
#: load | +-2.5V"), the QCoDeS validator (module.py out{i}_offset), and the
#: compiler's StaticAnalogModuleProperties -- three independent agreements.
_BASEBAND_RAIL_V = {"QCM": 2.5, "QRM": 0.5}

#: Fraction of the rail above which we warn. The rail assumes a 50 ohm
#: termination; an unterminated line swings roughly double, landing on the
#: module's +/-5 V ABSOLUTE MAXIMUM rating.
_WARN_FRACTION = 0.9

#: The largest normalized amplitude the sequencer accepts (offset_awg_path{x} is
#: Numbers(-1.0, 1.0)). Reaching it means reaching the rail, by definition.
_MAX_FRACTION = 1.0


def _hw_config(experiment: Any) -> Any:
    """The authoritative hardware configuration this run will compile against."""
    return experiment.backend._hw_agent.hardware_configuration


def _module_types(hw_config: Any) -> dict[tuple[str, int], str]:
    """``(cluster, slot) -> instrument_type`` from the hardware description.

    Tolerates both shapes for the same reason ``_port_outputs`` does: a validated
    model and a raw dict-loaded config both reach this code depending on caller.
    """
    description = hw_config.get("hardware_description") if isinstance(hw_config, dict) \
        else getattr(hw_config, "hardware_description", None)
    types: dict[tuple[str, int], str] = {}
    for cluster, spec in (description or {}).items():
        modules = spec.get("modules") if isinstance(spec, dict) else getattr(spec, "modules", None)
        for slot, module in (modules or {}).items():
            instrument = module.get("instrument_type") if isinstance(module, dict) \
                else getattr(module, "instrument_type", None)
            try:
                types[(str(cluster), int(slot))] = str(instrument)
            except (TypeError, ValueError):
                continue
    return types


def _flux_port_module(hw_config: Any, port: str) -> tuple[str, int, str] | None:
    """``(cluster, slot, instrument_type)`` carrying ``port``, or None if unwired.

    Walks the connectivity graph for a ``real_output_N`` edge — the baseband
    (single-ended) output shape. ``_port_outputs`` in the backend deliberately
    matches only ``complex_output_*``, because it exists to resolve ATTENUATOR
    limits and attenuators are RF-only; flux never appears there.
    """
    connectivity = hw_config.get("connectivity") if isinstance(hw_config, dict) \
        else getattr(hw_config, "connectivity", None)
    graph = connectivity.get("graph") if isinstance(connectivity, dict) \
        else getattr(connectivity, "graph", None)
    edges = getattr(graph, "edges", None)
    if edges is None:
        edges = graph or []

    types = _module_types(hw_config)
    for edge in edges:
        try:
            a, b = edge
        except (TypeError, ValueError):
            continue
        for node, other in ((a, b), (b, a)):
            if str(other) != str(port):
                continue
            parts = str(node).split(".")
            if len(parts) != 3 or not parts[1].startswith("module"):
                continue
            cluster, module, channel = parts
            if not channel.startswith("real_output_"):
                continue
            try:
                slot = int(module[len("module"):])
            except ValueError:
                continue
            return cluster, slot, types.get((cluster, slot), "")
    return None


def flux_rail_v(experiment: Any, port: str, *, name: str) -> float:
    """The full-scale output voltage of the module carrying ``port``.

    Refuses an RF module BY NAME rather than guessing a rail: three of the four
    module types in this lab's fleet are RF-only and cannot emit DC at all, so
    "flux wired to an RF port" is the realistic misconfiguration here, and it is
    a wiring bug that no voltage limit would express.
    """
    resolved = _flux_port_module(_hw_config(experiment), port)
    if resolved is None:
        raise ValueError(
            f"{name}: flux port {port!r} is not wired to any baseband "
            f"(real_output_N) module output in the hardware config, so there is "
            f"no DAC range to check it against. Wire it, or run an experiment "
            f"that does not sweep flux.")
    cluster, slot, instrument = resolved
    where = f"{cluster} slot {slot} ({instrument or 'unknown module'})"
    if instrument not in _BASEBAND_RAIL_V:
        raise ValueError(
            f"{name}: flux port {port!r} resolves to {where}, which cannot emit "
            f"DC at all — the RF modules are up-converted (2 GHz and above) and "
            f"only a baseband module ({'/'.join(sorted(_BASEBAND_RAIL_V))}) can "
            f"carry a flux line. This is a wiring error, not a range one.")
    return _BASEBAND_RAIL_V[instrument]


def to_dac_fraction(volts: float, rail: float) -> float:
    """Volts -> the normalized fraction the sequencer actually consumes.

    THE conversion the Qblox flux path was missing: every operand above the
    hardware is a fraction of full scale, so passing volts through unconverted
    emitted ``value * rail`` — 2.5x the requested number on a QCM. Deliberately
    not folded into the checkers so the rescale is visible where it happens.
    """
    return float(volts) / float(rail)


def _check(name: str, rail: float, peak: float, detail: str) -> float:
    import warnings

    if peak > rail:
        raise ValueError(
            f"{name}: {detail} reaches {peak} V, past the port's {rail} V full "
            f"scale (50 ohm terminated). The DAC saturates SILENTLY — it "
            f"flat-tops at the rail, sets only a sticky OUTPUT_OVERFLOW sequencer "
            f"flag, and raises nothing. Narrow the window.")
    if peak > _WARN_FRACTION * rail:
        warnings.warn(
            f"{name}: {detail} reaches {peak} V, within {100 * (1 - _WARN_FRACTION):.0f}% "
            f"of the port's {rail} V full scale. That rail assumes a 50 ohm "
            f"termination; on an unterminated line the swing is roughly double, "
            f"which is the module's absolute maximum rating.",
            RuntimeWarning, stacklevel=3)
    return rail


def check_flux_bias_absolute(experiment: Any, *, name: str, port: str, bias_v) -> float:
    """Validate an ABSOLUTE flux-bias sweep; return the port's rail.

    ``bias_v`` are the swept values in volts and they ARE the line voltage —
    ``VoltageOffset`` REPLACES the standing bias rather than adding to it, so no
    idle term belongs here.
    """
    rail = flux_rail_v(experiment, port, name=name)
    peak = max(abs(float(v)) for v in bias_v)
    return _check(name, rail, peak, "the flux bias sweep")


def check_flux_pulse_relative(experiment: Any, *, name: str, port: str,
                              idle_v: float, amps_v) -> float:
    """Validate a RELATIVE flux-pulse sweep; return the port's rail.

    ``amps_v`` are excursions from ``idle_v``. The DAC emits the SUM — offsets and
    waveforms add — so a window that is fine on its own still clips once it rides
    on a standing bias.
    """
    rail = flux_rail_v(experiment, port, name=name)
    lo = min(float(v) for v in amps_v)
    hi = max(float(v) for v in amps_v)
    peak = max(abs(idle_v + hi), abs(idle_v + lo))
    return _check(name, rail, peak,
                  f"{idle_v} V idle + the window [{lo}, {hi}] V")
