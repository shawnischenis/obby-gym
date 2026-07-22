#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from obby_rl.config import ROOT, load_json
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxBatchedVecEnv, RobloxObbyBatch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from train_vector_ppo import parse_curriculum_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recurrent PPO over Roblox lanes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "m3_fixed_ppo.json")
    parser.add_argument("--run-name")
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 24), default=23)
    parser.add_argument("--curriculum-replay")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1024)
    parser.add_argument("--jump-threshold", type=float, default=0.65)
    parser.add_argument("--jump-cooldown-steps", type=int, default=8)
    parser.add_argument("--jump-timing-weight", type=float, default=0.15)
    parser.add_argument("--jump-mistiming-weight", type=float, default=0.02)
    parser.add_argument("--checkpoint-credit-weight", type=float, default=0.2)
    parser.add_argument("--mask-jump-to-takeoff-window", action="store_true")
    parser.add_argument("--reset-all-on-any-done", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    replay = parse_curriculum_replay(args.curriculum_replay)
    run_name = args.run_name or f"m4-recurrent-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=args.curriculum_stage,
        action_repeat_ticks=3,
    )
    batch = RobloxObbyBatch(
        transport,
        args.num_envs,
        jump_threshold=args.jump_threshold,
        jump_cooldown_steps=args.jump_cooldown_steps,
        yaw_scale=0,
        terminate_on_hazard=True,
        jump_timing_weight=args.jump_timing_weight,
        jump_mistiming_weight=args.jump_mistiming_weight,
        checkpoint_credit_weight=args.checkpoint_credit_weight,
        mask_jump_to_takeoff_window=args.mask_jump_to_takeoff_window,
    )
    vector_env = RobloxBatchedVecEnv(
        batch,
        course_seed=int(config["course_seed"]),
        max_episode_steps=int(config["episode_max_steps"]),
        vary_course_seeds=True,
        reset_all_on_any_done=args.reset_all_on_any_done,
        curriculum_replay=replay,
        curriculum_sampler_seed=int(config["master_seed"]),
    )
    env = VecMonitor(vector_env, filename=str(run_dir / "monitor.csv"))
    rollout_size = args.n_steps * args.num_envs
    batch_size = min(64, rollout_size)
    if args.init_model:
        model = RecurrentPPO.load(
            args.init_model,
            env=env,
            device="cpu",
            custom_objects={
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": batch_size,
                "n_epochs": args.n_epochs,
                "target_kl": 0.01,
            },
        )
        model.verbose = 1
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=0.01,
            policy_kwargs={
                "lstm_hidden_size": args.lstm_hidden_size,
                "n_lstm_layers": 1,
                "net_arch": {"pi": [128], "vf": [128]},
                "log_std_init": -0.5,
            },
            verbose=1,
            seed=int(config["master_seed"]),
            device="cpu",
        )
    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every_steps // args.num_envs),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="recurrent_ppo",
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint, progress_bar=False)
        model.save(run_dir / "final_model")
    finally:
        env.close()
    print(f"saved recurrent PPO run to {run_dir}")


if __name__ == "__main__":
    main()
