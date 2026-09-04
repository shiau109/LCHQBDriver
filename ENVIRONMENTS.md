# Environments

Which Python environment to use lives in one place for all four repos, because the rule is a
**combo** property — the vendor-version half of it is literally cross-repo, and separate copies
disagree within one cycle:

**https://github.com/shiau109/SCQO/blob/main/ENVIRONMENTS.md**

The two lines for this repo, so you cannot get them wrong by not clicking:

```bash
uv run pytest tests/ -q                                              # scqo-qblox/.venv
<parent>/.venv-qblox/Scripts/python.exe -m pytest tests/ -q          # the lab env, TOO
```

Plain `uv run` is correct here — `scqo` is a hard dependency, so uv's sync keeps it, and
`uv.lock` carries `qblox-scheduler==1.0.0b6` plus `scqat` editable from `../scqat`.

**The second run is not a formality.** It is the vendor-version rule: on 2026-07-26 the two
environments held `qblox-scheduler` b4 and b6, they *disagreed about whether a schedule is
legal*, and `readout_frequency` compiled clean offline and died on the hardware. Compiling in one
environment proves nothing about the other.
