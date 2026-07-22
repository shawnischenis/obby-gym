#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from obby_rl.config import ROOT
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from stable_baselines3 import PPO

DEFAULT_MODEL = ROOT / "runs" / "m3-stage1-2-student-ppo-v1" / "final_model.zip"


def load_action_mapping(model_path: Path) -> dict[str, float | int]:
    for directory in (model_path.parent, model_path.parent.parent):
        config_path = directory / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            mapping = config.get("action_mapping", {})
            if isinstance(mapping, dict):
                return mapping
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show eight simultaneous policy rollouts for website recording"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 24), default=2)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="idle time after lane creation so Studio can render before actions begin",
    )
    parser.add_argument(
        "--reset-delay-seconds",
        type=float,
        default=0.0,
        help="hold terminal poses before generating the next synchronized cohort",
    )
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    parser.add_argument(
        "--jump-threshold",
        type=float,
        help="override the checkpoint run's recorded jump threshold",
    )
    parser.add_argument(
        "--jump-cooldown-steps",
        type=int,
        help="override the checkpoint run's recorded jump cooldown",
    )
    parser.add_argument(
        "--camera-view",
        choices=("auto", "parallel", "side", "behind", "completion"),
        default="auto",
        help="recording camera and compatible lane arrangement",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="disable PPO sampling; stochastic actions better resemble rollout collection",
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.warmup_seconds < 0:
        raise ValueError("--warmup-seconds must be non-negative")
    if args.reset_delay_seconds < 0:
        raise ValueError("--reset-delay-seconds must be non-negative")
    if not args.model.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {args.model}")

    action_mapping = load_action_mapping(args.model)
    jump_threshold = float(
        args.jump_threshold
        if args.jump_threshold is not None
        else action_mapping.get("jump_threshold", 0.0)
    )
    jump_cooldown_steps = int(
        args.jump_cooldown_steps
        if args.jump_cooldown_steps is not None
        else action_mapping.get("jump_cooldown_steps", 8)
    )
    if jump_cooldown_steps < 0:
        raise ValueError("jump cooldown must be non-negative")

    model = PPO.load(args.model, device="cpu")
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=args.curriculum_stage,
        action_repeat_ticks=args.action_repeat_ticks,
        recording_view=True,
        recording_camera=args.camera_view,
    )
    batch = RobloxObbyBatch(
        transport,
        args.num_envs,
        jump_threshold=jump_threshold,
        jump_cooldown_steps=jump_cooldown_steps,
        yaw_scale=0,
        terminate_on_hazard=True,
    )
    expected_shape = (batch.observation_size,)
    if model.observation_space.shape != expected_shape:
        batch.close()
        raise ValueError(
            f"model expects observations {model.observation_space.shape}, "
            f"but this rollout provides {expected_shape}"
        )

    next_seed = args.seed_start
    transitions = 0
    cohorts = 0
    print("Waiting for the ObbyRL Studio plugin at http://127.0.0.1:8765 ...", flush=True)
    print(
        f"action mapping jump_threshold={jump_threshold:g} "
        f"jump_cooldown_steps={jump_cooldown_steps}",
        flush=True,
    )
    try:
        seeds = list(range(next_seed, next_seed + args.num_envs))
        next_seed += args.num_envs
        observations, _ = batch.reset(seeds)
        cohorts = 1
        if args.warmup_seconds:
            print(
                f"lanes ready; waiting {args.warmup_seconds:g}s for rendering before rollout ...",
                flush=True,
            )
            time.sleep(args.warmup_seconds)
        started = time.perf_counter()
        while time.perf_counter() - started < args.duration_seconds:
            actions, _ = model.predict(observations, deterministic=args.deterministic)
            observations, _, terminated, truncated, _ = batch.step(
                np.asarray(actions, dtype=np.float32)
            )
            transitions += args.num_envs
            if np.any(terminated | truncated):
                if args.reset_delay_seconds:
                    time.sleep(args.reset_delay_seconds)
                seeds = list(range(next_seed, next_seed + args.num_envs))
                next_seed += args.num_envs
                observations, _ = batch.reset(seeds)
                cohorts += 1
        elapsed = time.perf_counter() - started
        print(
            f"parallel rollout finished envs={args.num_envs} cohorts={cohorts} "
            f"transitions={transitions} seconds={elapsed:.1f} "
            f"aggregate_fps={transitions / elapsed:.1f}",
            flush=True,
        )
    except KeyboardInterrupt:
        print("Stopped recording rollout.", flush=True)
    finally:
        batch.close()


if __name__ == "__main__":
    main()
