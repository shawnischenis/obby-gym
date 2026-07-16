# M3 status — fixed-course PPO baseline

M3 is in progress.

## Implemented

- Fixed-course Gymnasium wrapper that forces course seed `0` on every episode reset.
- Python-side 400-step episode time limit, corresponding to 20 seconds at 20 decisions/second.
- Stable-Baselines3 PPO entry point with configurable rollout length, batch size, network architecture, checkpoint interval, and master seed.
- Per-run resolved configuration, Monitor CSV output, periodic checkpoints, and final model serialization.
- Unit tests for fixed-seed resets and episode truncation.

## Default baseline

- Policy: `MlpPolicy`, two hidden layers of 128 units.
- Budget: 100,000 environment steps.
- PPO rollout: 256 steps; batch size 64; 10 epochs.
- Training course: procedural generator seed `0` for every episode.
- Evaluation: 20 deterministic episodes on seed `0`, then the untouched validation partition beginning at seed `1,000,000`.

## Live smoke gate

The second 128-step smoke invocation completed successfully against course seed `0`:

- Two 64-step rollout iterations at 7 environment steps/second.
- Ten PPO optimization epochs after the first rollout.
- Approximate KL `0.0084`, clip fraction `0.0375`, and entropy loss `-5.67`.
- Final artifact saved as `runs/m3-fixed-smoke-2/final_model.zip` (504,162 bytes).
- Reloaded model produced a finite four-component deterministic action for a test observation.

The smoke budget is intentionally too small to assess policy quality. The negative explained variance (`-8.28`) is not treated as a learning result.

## Remaining M3 work

1. Add periodic fixed-seed and held-out-seed evaluation with completion-rate reporting.
2. Save complete provenance and explicit run state (`running`, `complete`, or `failed`) with each experiment.
3. Run the declared 100,000-step fixed-course budget and evaluate the frozen checkpoint.
