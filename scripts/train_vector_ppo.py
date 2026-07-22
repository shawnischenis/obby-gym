#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from obby_rl.config import ROOT, load_json
from obby_rl.feasibility import JumpFeasibilityModel
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
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 24))
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--terminate-on-hazard", action="store_true")
    parser.add_argument("--jump-threshold", type=float)
    parser.add_argument("--jump-timing-weight", type=float, default=0.0)
    parser.add_argument("--jump-timing-min-distance", type=float, default=12.0)
    parser.add_argument("--jump-timing-max-distance", type=float, default=18.0)
    parser.add_argument("--jump-cooldown-steps", type=int)
    parser.add_argument("--mask-jump-to-takeoff-window", action="store_true")
    parser.add_argument("--scripted-forward", action="store_true")
    parser.add_argument(
        "--scripted-forward-replay-probability",
        type=float,
        default=0.0,
        help="probability that a synchronous cohort uses scripted forward movement",
    )
    parser.add_argument("--jump-mistiming-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-credit-weight", type=float, default=0.0)
    parser.add_argument(
        "--freeze-actor",
        action="store_true",
        help="update only the critic while preserving deterministic actor behavior",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--checkpoint-every-steps", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--vary-course-seeds", action="store_true")
    parser.add_argument("--reset-all-on-any-done", action="store_true")
    parser.add_argument("--wait-for-all-done", action="store_true")
    parser.add_argument(
        "--post-landing-reset-probability",
        type=float,
        default=0.0,
        help="fraction of Stage 20-23 lanes reset from captured intermediate states",
    )
    parser.add_argument(
        "--privileged-observations",
        action="store_true",
        help="train a fresh teacher on the separate exact geometry/physics payload",
    )
    parser.add_argument("--jump-feasibility-model", type=Path)
    parser.add_argument("--jump-feasibility-threshold", type=float, default=0.5)
    parser.add_argument(
        "--curriculum-replay",
        help="Comma-separated stage:weight pairs sampled once per synchronous cohort",
    )
    return parser.parse_args()


def parse_curriculum_replay(specification: str | None) -> list[tuple[int, float]]:
    if not specification:
        return []
    replay: list[tuple[int, float]] = []
    for item in specification.split(","):
        stage_text, separator, weight_text = item.partition(":")
        if not separator:
            raise ValueError("curriculum replay entries must use stage:weight")
        stage = int(stage_text)
        weight = float(weight_text)
        if stage not in range(1, 24) or weight <= 0:
            raise ValueError("curriculum replay stage or weight is invalid")
        replay.append((stage, weight))
    return replay


def main() -> None:
    args = parse_args()
    curriculum_replay = parse_curriculum_replay(args.curriculum_replay)
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
        "action_repeat_ticks": int(args.action_repeat_ticks),
        "initial_model": str(args.init_model) if args.init_model else None,
        "smoothness_weight": float(args.smoothness_weight),
        "terminate_on_hazard": bool(args.terminate_on_hazard),
        "jump_timing_weight": float(args.jump_timing_weight),
        "jump_timing_distance": [
            float(args.jump_timing_min_distance),
            float(args.jump_timing_max_distance),
        ],
        "mask_jump_to_takeoff_window": bool(args.mask_jump_to_takeoff_window),
        "scripted_forward": bool(args.scripted_forward),
        "scripted_forward_replay_probability": float(args.scripted_forward_replay_probability),
        "jump_mistiming_weight": float(args.jump_mistiming_weight),
        "checkpoint_credit_weight": float(args.checkpoint_credit_weight),
        "freeze_actor": bool(args.freeze_actor),
        "total_timesteps": timesteps,
        "vary_course_seeds": bool(args.vary_course_seeds),
        "reset_all_on_any_done": bool(args.reset_all_on_any_done),
        "wait_for_all_done": bool(args.wait_for_all_done),
        "post_landing_reset_probability": float(args.post_landing_reset_probability),
        "curriculum_replay": curriculum_replay,
        "privileged_observations": bool(args.privileged_observations),
        "jump_feasibility_model": (
            str(args.jump_feasibility_model) if args.jump_feasibility_model else None
        ),
        "jump_feasibility_threshold": float(args.jump_feasibility_threshold),
        "ppo": {**config["ppo"], "n_steps": n_steps, "batch_size": batch_size},
    }
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None else config["ppo"]["learning_rate"]
    )
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    resolved["ppo"] = {**resolved["ppo"], "learning_rate": learning_rate}
    n_epochs = int(args.n_epochs if args.n_epochs is not None else config["ppo"]["n_epochs"])
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    target_kl = float(args.target_kl) if args.target_kl is not None else None
    if target_kl is not None and target_kl <= 0:
        raise ValueError("target_kl must be positive")
    resolved["ppo"] = {
        **resolved["ppo"],
        "n_epochs": n_epochs,
        "target_kl": target_kl,
    }
    jump_threshold = float(
        args.jump_threshold
        if args.jump_threshold is not None
        else config["action_mapping"]["jump_threshold"]
    )
    resolved["action_mapping"] = {
        **config["action_mapping"],
        "jump_threshold": jump_threshold,
        "jump_cooldown_steps": int(
            args.jump_cooldown_steps
            if args.jump_cooldown_steps is not None
            else config["action_mapping"]["jump_cooldown_steps"]
        ),
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "provenance.json", collect_m3_provenance(resolved))
    state_path = run_dir / "run_state.json"
    state = start_run(state_path, target_timesteps=timesteps)
    env: VecMonitor | None = None
    try:
        transport = StudioHTTPTransport(
            timeout=120,
            curriculum_stage=int(resolved["curriculum_stage"]),
            action_repeat_ticks=int(args.action_repeat_ticks),
        )
        mapping = config["action_mapping"]
        feasibility_model: JumpFeasibilityModel | None = None
        if args.jump_feasibility_model:
            feasibility_model = JumpFeasibilityModel()
            feasibility_model.load_state_dict(
                torch.load(args.jump_feasibility_model, map_location="cpu", weights_only=True)
            )
            feasibility_model.eval()
        batch = RobloxObbyBatch(
            transport,
            num_envs,
            jump_threshold=jump_threshold,
            jump_cooldown_steps=int(resolved["action_mapping"]["jump_cooldown_steps"]),
            yaw_scale=float(mapping["yaw_scale"]),
            smoothness_weight=float(args.smoothness_weight),
            terminate_on_hazard=bool(args.terminate_on_hazard),
            jump_timing_weight=float(args.jump_timing_weight),
            jump_timing_distance=(
                float(args.jump_timing_min_distance),
                float(args.jump_timing_max_distance),
            ),
            mask_jump_to_takeoff_window=bool(args.mask_jump_to_takeoff_window),
            scripted_forward=bool(args.scripted_forward),
            jump_mistiming_weight=float(args.jump_mistiming_weight),
            checkpoint_credit_weight=float(args.checkpoint_credit_weight),
            privileged_observations=bool(args.privileged_observations),
            jump_feasibility_model=feasibility_model,
            jump_feasibility_threshold=float(args.jump_feasibility_threshold),
        )
        vector_env = RobloxBatchedVecEnv(
            batch,
            course_seed=int(config["course_seed"]),
            max_episode_steps=int(config["episode_max_steps"]),
            vary_course_seeds=bool(args.vary_course_seeds),
            reset_all_on_any_done=bool(args.reset_all_on_any_done),
            wait_for_all_done=bool(args.wait_for_all_done),
            curriculum_replay=curriculum_replay,
            curriculum_sampler_seed=int(config["master_seed"]),
            scripted_forward_replay_probability=float(args.scripted_forward_replay_probability),
            post_landing_reset_probability=float(args.post_landing_reset_probability),
        )
        env = VecMonitor(vector_env, filename=str(run_dir / "monitor.csv"))
        checkpoint = CheckpointCallback(
            save_freq=max(
                1,
                int(
                    args.checkpoint_every_steps
                    if args.checkpoint_every_steps is not None
                    else config["checkpoint_every_steps"]
                )
                // num_envs,
            ),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_vector",
        )
        ppo = config["ppo"]
        if args.init_model:
            # PPO.load validates the checkpoint against env.observation_space. That
            # allows teacher curriculum transfer but rejects 22/48-input mixing.
            # Continuation rollout settings must be injected during reconstruction;
            # assigning n_steps afterward leaves the old rollout buffer allocated.
            model = PPO.load(
                args.init_model,
                env=env,
                device="cpu",
                custom_objects={
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "target_kl": target_kl,
                },
            )
            model.verbose = 1
        else:
            model = PPO(
                str(ppo["policy"]),
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=float(ppo["gamma"]),
                gae_lambda=float(ppo["gae_lambda"]),
                clip_range=float(ppo["clip_range"]),
                ent_coef=float(ppo["ent_coef"]),
                policy_kwargs={
                    "net_arch": list(ppo["policy_layers"]),
                    "log_std_init": float(ppo["log_std_init"]),
                },
                seed=int(config["master_seed"]),
                target_kl=target_kl,
                verbose=1,
                device="cpu",
            )
        if args.freeze_actor:
            for parameter in model.policy.mlp_extractor.policy_net.parameters():
                parameter.requires_grad_(False)
            for parameter in model.policy.action_net.parameters():
                parameter.requires_grad_(False)
            model.policy.log_std.requires_grad_(False)
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
