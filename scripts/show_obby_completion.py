#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from obby_rl.config import ROOT, load_json
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from show_parallel_rollout import load_action_mapping
from stable_baselines3 import PPO

DEFAULT_DEPLOYMENT = (
    ROOT / "runs" / "m4-stage23-required-jumps-replay-v1" / "deployment.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record deployed two-jump policy attempts at eight-lane cadence"
    )
    parser.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 25))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--seeds",
        help="comma-separated visible-lane seeds; overrides --seed and --episodes",
    )
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--between-course-seconds", type=float, default=1.0)
    parser.add_argument("--terminal-hold-seconds", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--lives",
        type=int,
        default=1,
        help="falls allowed per course before ending the trial",
    )
    parser.add_argument(
        "--retry-timing-shift",
        type=float,
        default=1.0,
        help="studs to shift retry jump windows: earlier on retry 1, later on retry 2",
    )
    parser.add_argument("--trace-actions", action="store_true")
    parser.add_argument(
        "--visible-lane",
        type=int,
        default=1,
        help="one-based rendered lane; use 0 to disable recording-only rig effects",
    )
    parser.add_argument(
        "--camera-view",
        choices=("completion", "completion-side", "completion-follow"),
        default="completion",
    )
    args = parser.parse_args()
    if not args.deployment.is_file():
        raise FileNotFoundError(f"deployment manifest not found: {args.deployment}")
    deployment = load_json(args.deployment)
    configured_model = Path(deployment["model"])
    if not configured_model.is_absolute():
        configured_model = ROOT / configured_model
    model_path = args.model or configured_model
    if not model_path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {model_path}")
    visible_seeds = (
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [args.seed + episode * 8 for episode in range(args.episodes)]
    )
    if not visible_seeds:
        raise ValueError("--seeds must contain at least one integer")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.lives < 1:
        raise ValueError("--lives must be positive")
    if args.retry_timing_shift < 0:
        raise ValueError("--retry-timing-shift must be non-negative")
    if not 0 <= args.visible_lane <= 8:
        raise ValueError("--visible-lane must be between 0 and 8")
    if min(args.warmup_seconds, args.between_course_seconds, args.terminal_hold_seconds) < 0:
        raise ValueError("recording delays must be non-negative")

    mapping = {
        **load_action_mapping(model_path),
        **deployment.get("action_mapping", {}),
    }
    jump_threshold = float(mapping.get("jump_threshold", 0.75))
    jump_cooldown_steps = int(mapping.get("jump_cooldown_steps", 8))
    mask_jump_to_takeoff_window = bool(mapping.get("mask_jump_to_takeoff_window", False))
    jump_timing_distance = tuple(mapping.get("jump_timing_distance", [12.0, 18.0]))
    if len(jump_timing_distance) != 2:
        raise ValueError("deployment jump_timing_distance must contain two values")
    curriculum_stage = int(
        args.curriculum_stage or deployment.get("curriculum_stage", 23)
    )
    model = PPO.load(model_path, device="cpu")
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=curriculum_stage,
        action_repeat_ticks=3,
        recording_view=True,
        recording_camera=args.camera_view,
        recording_visible_lane=args.visible_lane,
    )
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=jump_threshold,
        jump_cooldown_steps=jump_cooldown_steps,
        jump_timing_distance=(float(jump_timing_distance[0]), float(jump_timing_distance[1])),
        mask_jump_to_takeoff_window=mask_jump_to_takeoff_window,
        yaw_scale=0,
        # Roblox recovers a fall at the latest checkpoint. The recording loop
        # decides when the life budget is exhausted instead of treating every
        # hazard as an immediate terminal state.
        terminate_on_hazard=False,
    )
    try:
        print(
            f"deployment model={model_path} stage={curriculum_stage} "
            f"jump_threshold={jump_threshold:g} "
            f"takeoff_gate={mask_jump_to_takeoff_window}",
            flush=True,
        )
        completed_count = 0
        base_jump_timing_distance = batch.jump_timing_distance
        for episode, cohort_seed in enumerate(visible_seeds):
            batch.jump_timing_distance = base_jump_timing_distance
            seeds = list(range(cohort_seed, cohort_seed + 8))
            observations, _ = batch.reset(seeds)
            print(
                f"episode={episode + 1}/{len(visible_seeds)} visible_seed={cohort_seed}; "
                "preserving eight-lane cadence with lanes 2-8 hidden",
                flush=True,
            )
            time.sleep(args.warmup_seconds if episode == 0 else args.between_course_seconds)
            terminal = False
            lives_remaining = args.lives
            retry_timing_offset = 0.0
            previous_checkpoint = 0
            for step in range(1, args.max_steps + 1):
                actions, _ = model.predict(observations, deterministic=True)
                actions = np.asarray(actions, dtype=np.float32)
                if retry_timing_offset != 0.0:
                    target_delta = observations[0, 6:9] * np.asarray(
                        [64.0, 32.0, 64.0], dtype=np.float32
                    )
                    target_distance = float(np.linalg.norm(target_delta))
                    retry_window = (
                        base_jump_timing_distance[0] + retry_timing_offset,
                        base_jump_timing_distance[1] + retry_timing_offset,
                    )
                    grounded = bool(observations[0, 4] > 0.5)
                    jump_ready = bool(observations[0, 5] > 0.5)
                    if grounded and jump_ready and retry_window[0] <= target_distance <= retry_window[1]:
                        # This is a controlled retry intervention, not PPO
                        # exploration: retain movement and only request takeoff.
                        actions[0, 3] = 1.0
                observations, _, terminated, truncated, infos = batch.step(
                    actions
                )
                checkpoint = int(infos[0].get("checkpoint_index", 0))
                if args.trace_actions:
                    velocity = infos[0].get("pre_step_local_velocity", [0.0, 0.0, 0.0])
                    geometry = infos[0].get("pre_step_route_features", [0.0, 0.0, 0.0])
                    print(
                        f"episode={episode + 1} step={step} checkpoint={checkpoint} "
                        f"distance={infos[0].get('jump_takeoff_distance'):.2f} "
                        f"grounded={infos[0].get('pre_step_grounded')} "
                        f"velocity=({velocity[0] * 32:.2f},{velocity[1] * 32:.2f},"
                        f"{velocity[2] * 32:.2f}) "
                        f"geometry=({geometry[0] * 10:.2f},{geometry[1] * 3:.2f},"
                        f"{geometry[2] * 18:.2f}) "
                        f"progress={infos[0].get('pre_step_progress'):.3f} "
                        f"intent_value={infos[0].get('jump_intent_value'):.3f} "
                        f"intent={infos[0].get('jump_intent_active')} "
                        f"allowed={infos[0].get('jump_takeoff_allowed')} "
                        f"applied={infos[0].get('jump_command_applied')} "
                        f"cooldown={infos[0].get('jump_cooldown_remaining')}",
                        flush=True,
                    )
                if checkpoint != previous_checkpoint:
                    print(
                        f"episode={episode + 1} checkpoint={checkpoint} step={step} "
                        f"jump_applied={infos[0].get('jump_command_applied')} "
                        f"cooldown={infos[0].get('jump_cooldown_remaining')}",
                        flush=True,
                    )
                    previous_checkpoint = checkpoint
                if bool(infos[0].get("hazard_recovered")):
                    lives_remaining -= 1
                    print(
                        f"episode={episode + 1} fall step={step} "
                        f"checkpoint={checkpoint} lives_remaining={lives_remaining}",
                        flush=True,
                    )
                    if lives_remaining == 0:
                        print(
                            f"episode={episode + 1} terminal step={step} "
                            f"completed=False checkpoint={checkpoint} "
                            "hazard=True lives_exhausted=True",
                            flush=True,
                        )
                        terminal = True
                        time.sleep(args.terminal_hold_seconds)
                        break
                    attempt_number = args.lives - lives_remaining + 1
                    # Distance decreases during the approach, so a positive
                    # offset opens the gate earlier and a negative one later.
                    direction = 1.0 if attempt_number % 2 == 0 else -1.0
                    offset = direction * args.retry_timing_shift
                    retry_timing_offset = offset
                    batch.jump_timing_distance = (
                        max(0.0, base_jump_timing_distance[0] + offset),
                        max(0.01, base_jump_timing_distance[1] + offset),
                    )
                    print(
                        f"episode={episode + 1} retry_attempt={attempt_number} "
                        f"jump_window_offset={offset:+.2f}_studs "
                        f"jump_window={batch.jump_timing_distance}",
                        flush=True,
                    )
                if bool(terminated[0] or truncated[0]):
                    info = infos[0]
                    completed = bool(terminated[0]) and not bool(info.get("hazard_recovered"))
                    completed_count += int(completed)
                    print(
                        f"episode={episode + 1} terminal step={step} completed={completed} "
                        f"checkpoint={info.get('checkpoint_index')} "
                        f"hazard={bool(info.get('hazard_recovered'))}",
                        flush=True,
                    )
                    terminal = True
                    time.sleep(args.terminal_hold_seconds)
                    break
            if not terminal:
                print(
                    f"episode={episode + 1} reached max_steps={args.max_steps} "
                    "without terminating",
                    flush=True,
                )
        print(
            f"recording sequence complete clean={completed_count}/{len(visible_seeds)}",
            flush=True,
        )
    finally:
        batch.close()


if __name__ == "__main__":
    main()
