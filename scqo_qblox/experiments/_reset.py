"""The ``reset_method`` branch, written once for every probe that resets.

Every Qblox probe adds its reset through :func:`add_reset` instead of building
``Reset(...)`` itself. That is not style: the active path needs a specific
acquisition-channel name, a settle idle and four refusals, and every one of
those is silent-failure-shaped if a probe gets it slightly wrong (see below).
``tests/test_active_reset.py::test_no_probe_constructs_reset_directly`` keeps the
door the only door.

THERMAL is what it always was: ``Reset(target)`` compiles to
``IdlePulse(element.reset.duration)``, and that duration IS the neutral
``thermalization_time_s`` knob, so the wait is governed with no probe-side
argument.

ACTIVE is the vendor's ``ConditionalReset``, which expands to exactly::

    Measure(q, acq_protocol="ThresholdedAcquisition", feedback_trigger_label=q)
    ConditionalOperation(body=X(q), qubit_name=q)   at rel_time=TRIGGER_DELAY

It costs ``integration_time + acq_delay + 364 ns + rxy.duration + 4 ns``
(~18.8 us on chipA) against a 1.86 ms thermal wait — the ~100x that makes long
averaged sweeps affordable.

WHICH PROBES MAY ASK FOR IT. Active reset thresholds against ``acq_threshold``,
which ``single_shot_readout`` solved at ONE readout condition, and it replaces
the reset with a measurement. So it is valid only where the readout condition is
frozen for the whole run and the reset is a genuine state reset — the four
coherent-drive carriers, which are exactly the probes ``_state.py`` already
serves, plus ``qubit_spectroscopy``, whose saturation drive became a finite
``SquarePulse`` that has ended before the next point's reset (it used to latch a
``VoltageOffset`` across the whole sweep, which made that ``Reset`` a driven
dwell rather than a state reset — the reason it was denied) and which sweeps only
the DRIVE frequency. ``readout_frequency`` / ``readout_power`` SWEEP the readout
condition; ``single_shot_readout`` is the calibration itself (and would bias the
|1> blob it measures). Each of those refuses BY NAME — the neutral field's rule
is that an unrealizable method raises rather than silently thermalizing, because
a downgraded run completes, looks plausible, and only the wall clock disagrees.

THREE THINGS HERE ARE LOAD-BEARING, each verified against the compiler:

1. ``acq_channel=f"cond_{target}"``. The conditional measurement writes real
   acquisition bins. Left on the probe's own ``S_21_<q>`` channel the schedule
   STILL COMPILES and the bins silently MERGE — a 3-point sweep comes back with
   6 bins and the run dies afterwards in ``_to_canonical`` as "expected shape
   (3,) ... got (6,)". The helper owns the string so no probe can get it wrong.
   (Do NOT also pass cal19's ``clock=``/``port=``: unnecessary, and hardcoded
   port strings bypass ``_vendor.vendor_element``.)
2. The settle idle AFTER the reset, which is the readout channel's calibrated
   ``readout_depletion_s`` knob (resolved through scqo's ONE precedence helper
   ``_depletion.depletion_wait_ns``), not a number this file picks. The vendor
   gate gives no way to wait BEFORE the conditional pi — ``TRIGGER_DELAY`` is a
   fixed 364 ns hardware latency, about 2.3 photon lifetimes at
   kappa/2pi ~ 1 MHz — so the pi already sees a partly Stark-shifted qubit and
   that is simply the gate's limitation. What we CAN protect is the experiment's
   next pulse, which is what QM's ``reset_qubit_active`` spends its trailing
   ``wait(500)`` on.
3. Targets stay SEQUENTIAL. Only one feedback label may be outstanding at a
   time. Today's per-target sub-schedule pattern (``schedule.add(sub)`` with no
   ``ref_op``) satisfies this by construction, so a future "multiplex the reset
   across targets for speed" refactor must not be made blindly — and the
   toolchain will only partly stop you. Measured: INTERLEAVING the halves
   (acq q1, acq q2, cond q1, cond q2) raises "labels ... were not used yet" at
   compile time, but two whole ``ConditionalReset``s merely overlapping in time
   compile CLEAN, while the vendor's docs warn that triggers inside the same
   364 ns window may not work. The second case is yours to avoid.

COMPILE TIME grows with ``num_averages`` under active reset, and only under it:
``compile_conditional_playback`` unrolls every loop in Python to time-sort the
conditional operations. Measured on the test fixture at 51 points, one target:
thermal is flat at ~0.3 s, active is 0.94 / 3.79 / 14.11 s at 100 / 1000 / 4000
averages. It is spent inside ``acquire()`` before anything reaches the cluster,
so a long production run looks like a hang.

KNOWN GAP: the conditional measurement's own result — the pre-reset excited-state
population, the most useful diagnostic active reset produces — is discarded.
``_to_canonical`` reads only ``S_21_*``, and the neutral contract has no slot for
it.
"""

from __future__ import annotations

from typing import Any

from scqo.experiments._depletion import depletion_wait_ns

from ._state import require_discriminator

#: Class attribute a Qblox probe sets to True to declare active reset VALID for
#: its sequence. Read with getattr(..., False) so the default is DENY: a probe
#: added later that never thought about it fails closed with a named error
#: instead of silently running a wrong sequence.
ACTIVE_RESET_ATTR = "supports_active_reset"

#: Those that opt in, for the refusal message only (the authority is the class
#: attribute — this is a hint, and the census test keeps it honest).
_CARRIERS = ("qubit_relaxation, qubit_ramsey, qubit_echo, qubit_power_rabi, "
             "qubit_spectroscopy")


def check_reset_method(experiment: Any) -> str:
    """This run's reset method, after refusing everything Qblox cannot honour.

    Returns ``"thermal"`` or ``"active"``; raises ``ValueError`` naming the
    experiment and the reason. Cheap and side-effect free, because it is called
    TWICE: once by :func:`add_reset` (so a probe driven directly in a test still
    refuses) and once by ``QbloxBackend.acquire`` before ``probe()`` — the
    backstop that fires even if a probe forgets the helper.
    """
    params = experiment.params
    method = getattr(params, "reset_method", "thermal")
    name = getattr(type(experiment), "name", type(experiment).__name__)

    if method == "thermal":
        return "thermal"
    if method != "active":
        raise ValueError(
            f"{name}: reset_method={method!r} is not realized on the Qblox "
            f"backend (this driver implements 'thermal' and 'active')"
        )

    if not getattr(type(experiment), ACTIVE_RESET_ATTR, False):
        raise ValueError(
            f"{name}: reset_method='active' is not valid for this experiment. "
            f"Active reset thresholds every shot against the readout "
            f"discriminator, so it needs the readout condition held at the point "
            f"single_shot_readout calibrated — which this experiment does not do "
            f"(it sweeps the readout, or it IS the discriminator's calibration, "
            f"or its Reset is a driven dwell rather than a state reset). "
            f"Supported: {_CARRIERS}. Use reset_method='thermal' here."
        )

    if getattr(params, "thermalization_time_ns", None) is not None:
        # Refused, not ignored: the override moves element.reset.duration, which
        # ConditionalReset never reads, so honouring both would silently drop the
        # one the caller typed. Same discipline as _thermalization_override's own
        # "refusing to run with the override silently ignored".
        raise ValueError(
            f"{name}: thermalization_time_ns overrides the THERMAL wait "
            f"(element.reset.duration), which active reset never plays — so the "
            f"two together would silently ignore it. Drop one: keep "
            f"reset_method='active', or set reset_method='thermal' to use the "
            f"override."
        )

    # Device reads last, and the discriminator before the settle: without a
    # discriminator active reset is not a reset at all, whereas a missing settle
    # is a real but narrower defect. Reporting the deeper problem first keeps a
    # cold chip from being told to fix the second thing.
    for target in experiment.params.targets:
        require_discriminator(experiment, target, "reset_method='active'")
    for target in experiment.params.targets:
        if depletion_wait_ns(experiment, target) is None:
            raise ValueError(
                f"{name}: reset_method='active' needs {target}'s "
                f"readout_depletion_s, and it has never been calibrated. That is "
                f"the settle between the conditional pi and this experiment's "
                f"first pulse; without it the readout photons Stark-shift that "
                f"pulse and the fit drifts in a way nothing attributes to the "
                f"reset. Run `scqo run resonator_spectroscopy` (it proposes "
                f"depletion_factor / (2 pi x kappa_tot_hz) from the measured "
                f"linewidth), review with `scqo accept`, then re-run — or set it "
                f"directly with `scqo set {target}.readout_depletion_s=...` (0 "
                f"is legal and means no settle)."
            )
    return "active"


def add_reset(schedule: Any, experiment: Any, target: str) -> None:
    """Add THE reset for one target — every probe's one door.

    Takes the schedule rather than returning an operation because active reset
    is genuinely two or more operations (N conditional resets plus a settle),
    and a probe should not have to know that.
    """
    from qblox_scheduler.operations import ConditionalReset, IdlePulse, Reset

    if check_reset_method(experiment) == "thermal":
        schedule.add(Reset(target))
        return

    for _ in range(int(experiment.params.active_reset_rounds)):
        # A dedicated acq_channel, NOT the probe's S_21_<q> — see (1) above.
        schedule.add(ConditionalReset(qubit_name=target, acq_channel=f"cond_{target}"))
    # Never None here: check_reset_method refused above if it were.
    settle_ns = float(depletion_wait_ns(experiment, target))
    if settle_ns:
        # / 1e9, never * 1e-9: the probes' own float-exactness rule, and the
        # scheduler refuses anything off its 1 ns grid.
        schedule.add(IdlePulse(round(settle_ns) / 1e9))
