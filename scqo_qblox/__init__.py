"""scqo-qblox — Qblox backend for the scqo experiment API.

Importing this package is lightweight and does NOT import qblox_scheduler; the vendor
library is imported lazily inside the backend / each experiment's probe(). Import
``scqo_qblox.experiments`` to populate the scqo experiment registry.
"""

__all__ = ["backend", "experiments"]
