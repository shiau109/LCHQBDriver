"""Every Qblox probe COMPILES its Schedule against the greenfield device surface.

The cutover re-homed every probe's device READ from one per-qubit view onto the
CHANNEL ENTITY that owns the knob — ``self.device.channel(q, "readout")``
(``q1_ro``), ``...channel(q, "drive")`` (``q1_xy``) — plus the roster-resolved raw
element for the vendor-only bits (ports, the flux sweet spot). A stale field
spelling or a missing entity would otherwise surface only on hardware, so this
walks the WHOLE registered Qblox catalog and exercises each probe once.

It COMPILES rather than merely builds, and the difference is the whole point.
A built Schedule is just a tree of operations; every rule that can reject one
lives in the compiler — the 1 ns time grid, the DAC range, and the latched
parameter alignment that ``readout_frequency`` violated on chipA 2026-07-26 by
ending a loop body with ``Measure(freq=...)`` (the clock restore that factory
appends is a zero-duration parameter op, which may not land on the loop's
``ControlFlowReturn``). Building it caught nothing. Compiling it catches the
whole class, for every probe, including the ones nobody thought to list.

Offline: the real dut fixture + the minimal hw config, no cluster.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import (  # noqa: E402
    ROSTER_TOML,
    compile_probe,
    make_backend,
    make_experiment,
)

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import catalog, get  # noqa: E402

#: every experiment whose registered class comes from THIS driver
QBLOX_PROBES = sorted(
    entry["name"] for entry in catalog()
    if get(entry["name"]).__module__.startswith("scqo_qblox."))

#: keep the schedules small — this test is about the device surface, not physics
#: (the values still clear each Parameters' own minimums: >4 sweep points, >=100
#: shots; the per-shot loops are hardware loops, so the schedule stays tiny).
#: ``max_amp_factor`` is the one value chosen for the COMPILER rather than for
#: size: the fixture's pi_amp x the stock top factor exceeds the DAC's [-1, 1]
#: range, which is an amplitude concern and not what this file is about.
#: Every point-count spelling appears because ``_params`` filters per
#: experiment: ``num_amp_points`` is the amplitude capability's,
#: ``num_drive_freq_points`` / ``num_readout_freq_points`` the two detuning
#: FRAMES' (the frame is in the field name — see scqo's _capabilities/detuning),
#: and ``num_points`` the time sweeps'.
SMALL = {"num_points": 5, "num_amp_points": 5, "num_drive_freq_points": 5,
         "num_readout_freq_points": 5, "num_flux_points": 5,
         "num_power_points": 5, "num_averages": 2,
         "num_shots": 100, "max_amp_factor": 0.5}


def _params(cls):
    fields = set(cls.Parameters.model_fields)
    return cls.Parameters(targets=["q1"],
                          **{k: v for k, v in SMALL.items() if k in fields})


def test_the_whole_driver_catalog_is_covered():
    """The parametrization below is only worth as much as its list."""
    assert len(QBLOX_PROBES) == len(scqo_qblox.experiments.__all__)


@pytest.mark.parametrize("name", QBLOX_PROBES)
def test_probe_compiles(tmp_path, roster, name):
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster, _params(cls))
    # the two-tone probes play the drive chain's residual (spec_amp), which the
    # fixture leaves unseeded (NaN); the core run() solves it before probing
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    # qubit_thermal_population prepares only |g>, so it cannot locate |e> in its
    # own data and refuses to sweep without the stored blob centers an accepted
    # single_shot_readout leaves behind. Monitor writes, harmless to every other
    # probe here.
    readout = exp.device.channel("q1", "readout")
    readout.pos_g_i, readout.pos_g_q = 0.0, 0.0
    readout.pos_e_i, readout.pos_e_q = 4.0, 0.0
    # the parity monitors derive their fixed idle from the drive channel's stored
    # beat splitting and refuse without a GOVERNED depletion wait (their shot
    # cadence is the telegraph timebase). Both harmless to every other probe.
    exp.device.channel("q1", "drive").parity_delta_f_hz = 250e3
    readout.readout_depletion_s = 1e-6

    # compile, don't just build — the compiler is where the instrument's own
    # rules live (see this module's docstring). compile_probe sets sweep_axes
    # and calls probe() the way the Session does.
    compiled = compile_probe(backend, exp)
    assert compiled.operations, f"{name}: empty schedule"


def _acquisition_bins(compiled) -> int:
    """How many acquisition bins the CLUSTER will fill — the compiled truth
    behind 'shot mode appends, average mode averages'. Walks the binned
    acquisition mapping's control-flow tree, whose leaves carry the indices."""
    indices = set()

    def walk(node):
        acq_index = getattr(node, "acq_index", None)
        if acq_index is not None and hasattr(acq_index, "acq_index"):
            indices.update(acq_index.acq_index)
        for child in getattr(node, "children", None) or []:
            walk(child)

    for cluster in compiled.compiled_instructions.values():
        if not isinstance(cluster, dict):
            continue
        for module in cluster.values():
            mapping = module.get("acq_hardware_mapping") if isinstance(module, dict) else None
            for sequencer in (mapping or {}).values():
                for binned in sequencer.binned:
                    walk(binned.tree)
    return len(indices)


@pytest.mark.parametrize("name,points_field", [("readout_power", "num_amp_points"),
                                               ("readout_frequency", "num_readout_freq_points")])
def test_readout_sweeps_bin_per_shot_or_average(tmp_path, roster, name, points_field):
    """The ONE thing that makes average mode average is that the repetition
    loop's variable is not captured into the bin coords. Asserted where it is
    decidable — the COMPILED acquisition mapping: shot mode reserves a bin per
    (sweep point, prepared state, shot), average mode reserves one per (sweep
    point, prepared state) and the cluster averages num_shots repetitions into
    each. A missed label would silently merge every shot into one bin."""
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    params = _params(cls)
    points = getattr(params, points_field)
    swept_bins = points * 2  # x prepared_state

    def bins(mode):
        exp = make_experiment(cls, backend, roster,
                              params.model_copy(update={"readout_mode": mode}))
        return _acquisition_bins(compile_probe(backend, exp))

    assert bins("shot") == swept_bins * params.num_shots
    assert bins("average") == swept_bins


def test_flux_probe_refuses_a_target_with_no_flux_channel(tmp_path):
    """Reaching the flux port through the target's FLUX channel makes wiring the
    guard: a qubit with no flux line refuses in the roster, by name, instead of
    failing on a missing vendor port deep inside the schedule."""
    from scqo.roster import RosterError, parse_components

    from scqo_qblox.experiments.resonator_spectroscopy_flux import (
        QbloxResonatorSpectroscopyFlux,
    )

    # same chip, but q2's flux wire was never installed
    roster = parse_components(ROSTER_TOML.replace('[lines.z2]\nflux = ["q2"]\n', ""))
    assert not roster.channels_of("q2") or "q2_z" not in roster.entities
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        QbloxResonatorSpectroscopyFlux, backend, roster,
        _params(QbloxResonatorSpectroscopyFlux).model_copy(
            update={"targets": ["q2"]}),
    )
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(RosterError, match="no unique flux channel for 'q2'"):
        exp.probe()


#: the readout_detuning carriers — every Qblox probe that builds its OWN
#: readout-frequency NCO sweep. (readout_power sweeps amplitude at a fixed
#: frequency and has no such loop.)
READOUT_DETUNING_PROBES = [
    "readout_frequency",
    "resonator_spectroscopy",
    "resonator_spectroscopy_flux",
    "resonator_spectroscopy_power_amp",
    "resonator_spectroscopy_power_chain",
]

#: one-sided on purpose: 30 MHz of window, all but 5 MHz of it BELOW the
#: current readout frequency. A probe that re-derives `center +/- span/2`
#: reproduces the width and the point count exactly and lands 10 MHz off.
ASYMMETRIC_WINDOW = {"start_readout_detuning_hz": -25e6,
                     "end_readout_detuning_hz": 5e6,
                     "num_readout_freq_points": 5}


def _frequency_domains(operation) -> list:
    """Every FREQUENCY loop domain in a BUILT Schedule, outermost first."""
    found = []

    def walk(op):
        for domain in (getattr(op, "domain", None) or {}).values():
            if getattr(domain.dtype, "value", domain.dtype) == "frequency":
                found.append(domain)
        children = getattr(op, "operations", None)
        if isinstance(children, dict):
            for child in children.values():
                walk(child)
        body = getattr(op, "body", None)
        if body is not None:
            walk(body)

    walk(operation)
    return found


@pytest.mark.parametrize("name", READOUT_DETUNING_PROBES)
def test_asymmetric_readout_window_reaches_the_nco_unshifted(tmp_path, roster, name):
    """A one-sided readout-detuning window must arrive at the NCO sweep as
    given — the endpoint form, never `center +/- span/2`.

    This is the failure mode that produces plausible WRONG numbers instead of
    an error: re-deriving a symmetric span from the detuning axis silently
    re-centers the sweep on readout_freq_hz, keeping the width and the point
    count intact while every acquired frequency — and so every fitted one — is
    attributed to the wrong place. It matters most on exactly the sweeps that
    need the asymmetry: a punchout walks the dip DOWN from the dressed
    resonator toward the bare one, and a flux map walks it down as the qubit
    detunes."""
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster,
                          _params(cls).model_copy(update=ASYMMETRIC_WINDOW))
    exp.sweep_axes = exp.define_sweep()
    center = float(exp.device.channel("q1", "readout").readout_freq_hz)

    domains = _frequency_domains(exp.probe())
    assert domains, f"{name}: no frequency sweep in the built schedule"
    for domain in domains:  # the _amp punchout unrolls one per power block
        assert domain.start == pytest.approx(
            center + ASYMMETRIC_WINDOW["start_readout_detuning_hz"]), name
        assert domain.stop == pytest.approx(
            center + ASYMMETRIC_WINDOW["end_readout_detuning_hz"]), name
        assert domain.num == ASYMMETRIC_WINDOW["num_readout_freq_points"], name
