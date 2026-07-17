# M3 eight-lane Roblox vector environment

The vector path runs eight independent humanoid/course pairs inside one Roblox Studio server. Python sends one action array with shape `(8, 4)`; Roblox applies every action before crossing a shared three-heartbeat barrier and returns observations with shape `(8, 22)`.

## Isolation and synchronization

- Each course is generated under `workspace.ObbyRLVectorLanes.Lane_XX`.
- Lane origins are separated by 400 studs on the X axis, clearing the 300-stud kill planes.
- Every lane owns a cloned server-controlled humanoid, checkpoint state, previous action, reward potential, and jump cooldown.
- Forward and strafe inputs are refreshed for all agents on the same heartbeats. Jump and yaw are one-shot per batch action.
- Completed or time-limited lanes are independently rebuilt and settled. Other lanes remain stopped at their recorded observations during that reset exchange.
- The original player character is parked outside the simulation and serves only as the rig template.

## Python interfaces

- `StudioHTTPTransport.vector_reset`, `vector_step`, and `vector_reset_lanes` implement the batch protocol.
- `RobloxObbyBatch` validates batch shapes and converts continuous jump signals independently per lane.
- `RobloxBatchedVecEnv` implements the Stable-Baselines3 `VecEnv` contract, including automatic lane reset, `terminal_observation`, and `TimeLimit.truncated`.
- `scripts/train_vector_ppo.py` is the vector PPO entry point.

## Validation and use

```bash
.venv/bin/python scripts/smoke_vector_env.py --num-envs 8 --steps 40 --curriculum-stage 1
.venv/bin/python scripts/train_vector_ppo.py --num-envs 8 --timesteps 2048 --n-steps 32 --curriculum-stage 1 --run-name m3-vector-stage1-smoke
```

The first live batch smoke produced 272 transitions in 4.835 seconds, or 56.3 aggregate transitions/second, with ordered lane IDs 1 through 8 and observation shape `(8, 22)`. It stopped after 34 batches because the scripted forward action completed Stage 1, not because of a transport failure.

PPO's `n_steps` is per lane. With eight lanes, `--n-steps 32` yields a 256-transition rollout. Total timesteps and checkpoint intervals remain aggregate transition counts.
