#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from obby_rl.config import ROOT, load_json
from obby_rl.provenance import collect_m3_provenance
from obby_rl.run_state import finish_run, start_run, write_json
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxBatchedVecEnv, RobloxObbyBatch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO over batched Roblox lanes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "m3_fixed_ppo.json")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 5))
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_json(args.config)
    num_envs = int(args.num_envs)
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    timesteps = int(args.timesteps or config["total_timesteps"])
    n_steps = int(args.n_steps or config["ppo"]["n_steps"])
    rollout_size = n_steps * num_envs
    batch_size = min(int(config["ppo"]["batch_size"]), rollout_size)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"m3-vector-{num_envs}x-{timestamp}"
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config,
        "num_envs": num_envs,
        "curriculum_stage": int(args.curriculum_stage or config["curriculum_stage"]),
        "initial_model": str(args.init_model) if args.init_model else None,
        "total_timesteps": timesteps,
        "ppo": {**config["ppo"], "n_steps": n_steps, "batch_size": batch_size},
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "provenance.json", collect_m3_provenance(resolved))
    state_path = run_dir / "run_state.json"
    state = start_run(state_path, target_timesteps=timesteps)
    env: VecMonitor | None = None
    try:
        transport = StudioHTTPTransport(
            timeout=120, curriculum_stage=int(resolved["curriculum_stage"])
        )
        mapping = config["action_mapping"]
        batch = RobloxObbyBatch(
            transport,
            num_envs,
            jump_threshold=float(mapping["jump_threshold"]),
            jump_cooldown_steps=int(mapping["jump_cooldown_steps"]),
            yaw_scale=float(mapping["yaw_scale"]),
        )
        vector_env = RobloxBatchedVecEnv(
            batch,
            course_seed=int(config["course_seed"]),
            max_episode_steps=int(config["episode_max_steps"]),
        )
        env = VecMonitor(vector_env, filename=str(run_dir / "monitor.csv"))
        checkpoint = CheckpointCallback(
            save_freq=max(1, int(config["checkpoint_every_steps"]) // num_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_vector",
        )
        ppo = config["ppo"]
        if args.init_model:
            model = PPO.load(args.init_model, env=env, device="cpu")
            model.verbose = 1
        else:
            model = PPO(
                str(ppo["policy"]),
                env,
                learning_rate=float(ppo["learning_rate"]),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=int(ppo["n_epochs"]),
                gamma=float(ppo["gamma"]),
                gae_lambda=float(ppo["gae_lambda"]),
                clip_range=float(ppo["clip_range"]),
                ent_coef=float(ppo["ent_coef"]),
                policy_kwargs={
                    "net_arch": list(ppo["policy_layers"]),
                    "log_std_init": float(ppo["log_std_init"]),
                },
                seed=int(config["master_seed"]),
                verbose=1,
                device="cpu",
            )
        model.learn(total_timesteps=timesteps, callback=checkpoint, progress_bar=False)
        model.save(run_dir / "final_model")
        finish_run(state_path, state, "complete")
    except BaseException:
        finish_run(state_path, state, "failed", error=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
    print(f"saved vector PPO run to {run_dir}")


if __name__ == "__main__":
    main()
