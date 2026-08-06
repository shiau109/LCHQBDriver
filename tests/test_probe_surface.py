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

import lchqb.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import catalog, get  # noqa: E402

#: every experiment whose registered class comes from THIS driver
QBLOX_PROBES = sorted(
    entry["name"] for entry in catalog()
    if get(entry["name"]).__module__.startswith("lchqb."))

#: keep the schedules small — this test is about the device surface, not physics
#: (the values still clear each Parameters' own minimums: >4 sweep points, >=100
#: shots; the per-shot loops are hardware loops, so the schedule stays tiny).
#: ``max_amp_factor`` is the one value chosen for the COMPILER rather than for
#: size: the fixture's pi_amp x the stock top factor exceeds the DAC's [-1, 1]
#: range, which is an amplitude concern and not what this file is about.
#: Both point-count spellings appear because ``_params`` filters per experiment:
#: ``num_amp_points`` is the amplitude capability's, ``num_points`` the
#: single-axis frequency/time sweeps'.
SMALL = {"num_points": 5, "num_amp_points": 5, "num_freq_points": 5,
         "num_flux_points": 5, "num_power_points": 5, "num_averages": 2,
         "num_shots": 100, "max_amp_factor": 0.5}


def _params(cls):
    fields = set(cls.Parameters.model_fields)
    return cls.Parameters(targets=["q1"],
                          **{k: v for k, v in SMALL.items() if k in fields})


def test_the_whole_driver_catalog_is_covered():
    """The parametrization below is only worth as much as its list."""
    assert len(QBLOX_PROBES) == len(lchqb.experiments.__all__)


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


def test_flux_probe_refuses_a_target_with_no_flux_channel(tmp_path):
    """Reaching the flux port through the target's FLUX channel makes wiring the
    guard: a qubit with no flux line refuses in the roster, by name, instead of
    failing on a missing vendor port deep inside the schedule."""
    from scqo.roster import RosterError, parse_components

    from lchqb.experiments.resonator_spectroscopy_flux import (
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
