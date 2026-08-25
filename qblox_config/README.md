# qblox_config

Analogous to scqo-qm's `quam_config/`. Holds the Qblox device-model definition and
the scripts that generate the initial serialized config into `../qblox_state/`.

To fill in:
- `custom_elements.py` — `FluxTunableTransmonElement` (extend `qblox_scheduler`'s
  `BasicTransmonElement`), mirroring `QBLOX_training/.../custom_elements.py`.
- `generate_config.py` — build initial `hw_config.json` + `dut_config.json` for the lab cluster.

Not code yet — placeholder so the package layout is established.
