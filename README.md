# scqo-qblox

Qblox backend for the vendor-neutral `scqo` experiment API. It implements the Qblox-specific
half of each experiment (`probe()`) plus the backend/device adapter over `qblox_scheduler`
(`Schedule`, `HardwareAgent`, `QuantumDevice`). Independent of the Quantum Machines stack — the
only shared code is `scqo`.

See [CLAUDE.md](CLAUDE.md) for the full architecture, conventions, and operating rules.
