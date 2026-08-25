"""Custom Qblox device elements used by this lab's dut configs.

Originally copied from QBLOX_training/docs/applications/superconducting/
custom_elements.py (the read-only reference repo) so scqo-qblox is
self-contained at runtime: ``QuantumDevice.from_json_file`` can only deserialize
a dut config whose ``element_type`` classes are imported (registered) first —
``QbloxBackend`` imports this module before loading. This module now deliberately
EXTENDS the training-repo copy: ``SpectroscopyProperties`` (the ``spec_amp`` slot
behind the neutral ``drive_amp`` / ``drive_power_dbm`` pair),
``DepletionProperties`` (the neutral ``readout_depletion_s``) and
``LCHTransmonElement`` are lab additions with no upstream counterpart; when
syncing, diff only the vendored ``FluxProperties`` / ``PiHalfProperties`` bodies.

NOTE: imports qblox_scheduler — never import this module at package import time
(only inside backend methods), so ``import scqo_qblox`` stays vendor-free.
"""

import math
from typing import Literal

from qblox_scheduler.device_under_test.transmon_element import BasicTransmonElement
from qblox_scheduler.structure.model import Numbers, Parameter, SchedulerSubmodule


class FluxProperties(SchedulerSubmodule):
    """Submodule containing flux-specific parameters for Qubits AND Couplers."""

    sweet_spot: float = Parameter(
        label="Flux Sweet Spot", unit="V", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )
    flux_period: float = Parameter(
        label="Voltage per Flux Quantum (V_Phi0)", unit="V", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )
    asymmetry: float = Parameter(
        label="Junction Asymmetry (d)", unit="", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )
    crosstalk_vector: dict = Parameter(
        label="Crosstalk Vector",
        docstring="Maps the names of aggressor lines to their crosstalk coefficient (M_ij).",
        initial_value={},
    )


class PiHalfProperties(SchedulerSubmodule):
    """Submodule dedicated explicitly to Pi/2 pulse parameters."""

    amp90: float = Parameter(
        label="Pi/2 Pulse Amplitude", unit="V", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )


class SpectroscopyProperties(SchedulerSubmodule):
    """Submodule for the saturation (spec) drive — the qubit_spectroscopy stimulus.

    ``spec_amp`` is the stored saturation-drive amplitude (fraction of full
    scale, the CW ``VoltageOffset`` value the probe plays): the neutral
    ``drive_amp``, and the residual the ``drive_power_dbm`` chain solve writes."""

    spec_amp: float = Parameter(
        label="Saturation (spec) Drive Amplitude", unit="", initial_value=math.nan,
        vals=Numbers(allow_nan=True)
    )


class DepletionProperties(SchedulerSubmodule):
    """Submodule for the resonator photon-depletion wait.

    ``duration`` is the neutral ``readout_depletion_s`` knob: how long to wait
    after a readout for the photons to leave the resonator before the next pulse
    sees a Stark-shifted qubit. The scheduler has no slot of its own for this
    (QM does — ``resonator.depletion_time``), so the lab element carries it,
    exactly as it carries ``spec_amp``.

    NaN = never calibrated. Run ``scqo run resonator_spectroscopy``, which
    proposes ``depletion_factor / (2 pi x kappa_tot_hz)`` from the measured
    linewidth. NaN is the honest default rather than 0 because "no wait" and
    "nobody has measured this resonator" are different claims, and active reset
    refuses the second one."""

    duration: float = Parameter(
        label="Resonator Depletion Wait", unit="s", initial_value=math.nan,
        vals=Numbers(allow_nan=True)
    )


class LCHTransmonElement(BasicTransmonElement):
    """The lab's base transmon element: BasicTransmonElement + the ``spec``
    (saturation-drive) and ``depletion`` (post-readout wait) submodules.
    Fixed-frequency chips use this type directly; flux-tunable ones use the
    subclass below."""

    element_type: Literal["LCHTransmonElement"] = "LCHTransmonElement"

    spec: SpectroscopyProperties
    depletion: DepletionProperties


class FluxTunableTransmonElement(LCHTransmonElement):
    """A custom device element for all flux-tunable elements (Qubits or Couplers).

    Inherits all standard properties from BasicTransmonElement, plus the lab's
    ``spec`` submodule via LCHTransmonElement.
    """

    element_type: Literal["FluxTunableTransmonElement"] = "FluxTunableTransmonElement"

    flux_params: FluxProperties
    pi_half: PiHalfProperties

    @property
    def sensitive_point(self) -> float:
        if math.isnan(self.flux_params.sweet_spot) or math.isnan(self.flux_params.flux_period):
            return math.nan
        return self.flux_params.sweet_spot + (self.flux_params.flux_period / 4.0)
