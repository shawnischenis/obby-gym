# Roblox Obby RL

Research scaffold for training Roblox obstacle-course agents in Python and exporting their policies to Luau.

The active roadmap is [ROBLOX_OBBY_RL_PLAN.md](ROBLOX_OBBY_RL_PLAN.md). M0 establishes reproducible configuration, protocol contracts, a Rojo project, and local validation. The earlier generic MCP build-plan experiment is preserved under `legacy/mcp_build_skeleton/`; it is not part of the active runtime.

M1 implementation status and the remaining live-playtest gates are tracked in [docs/M1_STATUS.md](docs/M1_STATUS.md).

## Live Studio bridge

Build and install the local Studio plugin:

```bash
rojo build plugin.project.json -o /tmp/ObbyRLBridge.rbxm
cp /tmp/ObbyRLBridge.rbxm "$HOME/Documents/Roblox/Plugins/ObbyRLBridge.rbxm"
```

Restart Studio, allow the plugin to access `127.0.0.1` if prompted, open the built place, and press Play. Then run `python scripts/run_live_env.py`. The broker binds only to loopback.

## M0 quick start

Requirements: Python 3.11, Roblox Studio, and Rojo 7.5.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/validate_m0.py
pytest
rojo build -o ObbyRL.rbxlx
```

If `roblox.yml` needs refreshing after an engine API update, run `selene generate-roblox-std`. It is checked in so ordinary lint runs are deterministic and offline.

`configs/m0.json` is the canonical environment contract. The JSON Schemas under `schemas/` define configuration and Studio↔Python messages. Generated policies and experiment runs belong in `artifacts/` and `runs/` and are ignored by Git except for their README files.

## Layout

- `src/`: Rojo-managed Roblox code.
- `python/obby_rl/`: Python package and protocol/config utilities.
- `configs/`: versioned experiment and environment configuration.
- `schemas/`: machine-readable protocol contracts.
- `tests/`: Python and Luau contract tests.
- `scripts/`: local validation and future orchestration entry points.
- `legacy/`: archived pre-project MCP object-generation skeleton.

## Reproducibility identity

Every run must record:

- Git commit (and whether the tree was dirty).
- Configuration SHA-256.
- Observation/action/protocol schema versions.
- Generator version and train/validation/test seed partitions.
- Python/package versions and Roblox Studio version when available.

The M0 validator prints the configuration hash used as the initial experiment identity.
`python scripts/show_provenance.py` prints the metadata block that future training runs must save with their outputs.
