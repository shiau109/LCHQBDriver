"""Importing this package registers every Qblox experiment into the scqo catalog.

Add a line here for each new experiment module so its ``@register`` runs.
"""

from . import qubit_echo  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_continuous  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_discrete  # noqa: F401  (import side effect: @register)
from . import qubit_power_rabi  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey  # noqa: F401  (import side effect: @register)
from . import qubit_relaxation  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_overlap  # noqa: F401  (import side effect: @register)
from . import qubit_thermal_population  # noqa: F401  (import side effect: @register)
from . import readout_frequency  # noqa: F401  (import side effect: @register)
from . import readout_power  # noqa: F401  (import side effect: @register)
from . import broadband_resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_flux  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_chain  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_amp  # noqa: F401  (import side effect: @register)
from . import single_shot_readout  # noqa: F401  (import side effect: @register)

__all__ = [
    "broadband_resonator_spectroscopy",
    "resonator_spectroscopy",
    "qubit_spectroscopy",
    "qubit_spectroscopy_flux_pulse",
    "qubit_spectroscopy_overlap",
    "qubit_ramsey",
    "qubit_power_rabi",
    "resonator_spectroscopy_flux",
    "resonator_spectroscopy_power_amp",
    "resonator_spectroscopy_power_chain",
    "readout_power",
    "readout_frequency",
    "qubit_relaxation",
    "qubit_echo",
    "qubit_thermal_population",
    "qubit_parity_switch_continuous",
    "qubit_parity_switch_discrete",
    "single_shot_readout",
]
