# AGENTS.md — scqo-qblox

**You are reading the contributor brief.** If you are working in the maintainer's lab
tree, read [CLAUDE.md](CLAUDE.md) instead — it is the far fuller document (every hardware
invariant lives there) and its rules assume a shared, live checkout.

## What this repo is

The **Qblox backend** for [SCQO](https://github.com/shiau109/SCQO), the vendor-neutral
experiment API. It implements the Qblox half of each experiment — `probe()` — plus the
backend/device adapter over `qblox_scheduler` (`Schedule`, `HardwareAgent`,
`QuantumDevice`). Its QM sibling is
[scqo-qm](https://github.com/shiau109/scqo-qm); never import from it.

## Three design rules (do not break these)

1. **Independent of Quantum Machines.** Never import `qm`, `quam`, `quam_builder`,
   `qualibrate` or `qualibration_libs`. The only shared code is `scqo`, which is itself
   vendor-free. That independence is what lets both drivers coexist.
2. **The common API lives in `scqo`, not here.** Parameters, Result, `estimate`,
   `simulate`, `update`, the registry and `Session` all come from SCQO. This repo adds
   only `probe()` and the device adapter.
3. **One entry point.** `scqo run <name>` — no wrappers, no launcher stubs.

## Setup — a standalone fork of this repo cannot install

`pyproject.toml` resolves scqo as `{ path = "../SCQO", editable = true }`. You need SCQO
(and scqat) cloned as **siblings** under one parent, under their own names:

```
<parent>/
  SCQO/
  scqat/
  scqo-qblox/
```

```bash
cd <parent>
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e "./SCQO[viewer]" -e ./scqat -e ./scqo-qblox pytest
uv pip install --python .venv/bin/python "qblox-scheduler==1.0.0b6"
```

Windows: `.venv\Scripts\python.exe`. The `qblox-scheduler` pin is not cosmetic — PyPI's
only non-prerelease is an empty 0.0.0 placeholder that fails to build, and b4 vs b6
*disagree about whether a schedule is legal* (a probe once compiled clean offline and
died on hardware). Keep every environment on the same version.

## Adding an experiment

1. Subclass the backend-free experiment from `scqo.experiments.<name>`.
2. Implement **only** `probe()`, importing the vendor lib *inside* the method so
   `import scqo_qblox` stays light and the simulated path needs no Qblox.
3. Read device state through the **channel** that owns the knob —
   `self.device.channel(target, "readout").readout_freq_hz` — never
   `backend.device.component(<qubit>)`. Vendor-only bits (ports, the flux sweet spot)
   come from `_vendor.vendor_element(...)`.
4. `@register` it and add its import line to `scqo_qblox/experiments/__init__.py`, keeping
   `__all__` in step. `tests/test_experiment_registration.py` refuses a module missing
   its line.

Not every scqo experiment is realized here — several are QM-only. That is legitimate; a
backend that cannot realize something must **refuse it by name**, never silently
downgrade.

## Testing

```bash
uv run pytest tests/ -q
```

Plain `uv run` is correct here (`scqo` is a hard dependency, so uv's sync keeps it). The
suite is small enough that the **full run is the targeted run** — run it before every
commit. While iterating on a probe, loop on
`uv run pytest tests/test_probe_surface.py tests/test_time_grid.py -q` and pick the glue
test back up before you commit.

`test_probe_surface.py` is the one that matters most: it **compiles** every registered
probe's Schedule. Building a schedule proves nothing — the time grid, the DAC range and
the latched-parameter alignment all live in the compiler.

**Always report the exact command you ran.**

## What you can and cannot verify

You **can** run the offline suite, and `python scripts/check_real_config.py <folder>`
against your own lab's `dut_config*.json` + `hw_config*.json`. It runs the whole pipeline
with simulated data over your real device tree, on a temporary copy — your originals are
never opened for writing.

You **cannot** validate against the maintainer's cluster. Every PR records `offline`,
`hardware <chip> <date>`, or `unverified`.

## Hardware claims in CLAUDE.md are one lab's truth

`CLAUDE.md` documents specific module inventories and attenuator ceilings measured on
one chip. **The attenuator ceiling in particular is per module output and only the
instrument knows it** — it is read from a live SCPI query, not a constant. Do not
hardcode a limit you read in the docs; a hardcoded 0–60 already killed a run on a 30 dB
module.

## Branch and PR

1. `feature/<slug>`, never `main`.
2. **Same branch name in every repo you touch.**
3. Drivers merge **last**: `scqat → SCQO → drivers`. If your change needs a matching
   SCQO change, say so — and note whether the pair is **lockstep** (an old driver
   against a new SCQO producing wrong numbers rather than an error). That distinction
   goes in the release notes.

Full detail: [SCQO's CONTRIBUTING.md](https://github.com/shiau109/SCQO/blob/main/CONTRIBUTING.md).
