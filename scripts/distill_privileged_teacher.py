#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from obby_rl.config import ROOT
from obby_rl.dagger import fit_action_head
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from stable_baselines3 import PPO


def collect_teacher_trajectories(
    teacher: PPO,
    batch: RobloxObbyBatch,
    *,
    curriculum_stage: int,
    episodes: int,
    seed_start: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    batch.transport.curriculum_stage = curriculum_stage
    next_seed = seed_start
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    finished = 0
    clean = 0
    hazards = 0
    while finished < episodes:
        cohort_size = min(batch.num_envs, episodes - finished)
        teacher_observations, _ = batch.reset(list(range(next_seed, next_seed + batch.num_envs)))
        next_seed += batch.num_envs
        active = np.zeros(batch.num_envs, dtype=np.bool_)
        active[:cohort_size] = True
        while np.any(active):
            teacher_actions, _ = teacher.predict(teacher_observations, deterministic=True)
            checked_actions = np.asarray(teacher_actions, dtype=np.float32)
            checked_actions[~active] = 0
            observations.append(batch.student_observations[active])
            actions.append(checked_actions[active].copy())
            teacher_observations, _, terminated, truncated, infos = batch.step(checked_actions)
            dones = active & (terminated | truncated)
            for index in np.flatnonzero(dones):
                hazard = bool(infos[index].get("hazard_recovered"))
                hazards += int(hazard)
                clean += int(bool(terminated[index]) and not hazard)
                finished += 1
                active[index] = False
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        {
            "curriculum_stage": curriculum_stage,
            "episodes": episodes,
            "clean": clean,
            "hazards": hazards,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a privileged PPO teacher into the 22-input student"
    )
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage1-episodes", type=int, default=64)
    parser.add_argument("--stage2-episodes", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=2_000_000)
    parser.add_argument("--clone-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--jump-loss-weight", type=float, default=4.0)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    teacher = PPO.load(args.teacher, device="cpu")
    student = PPO.load(args.student, device="cpu")
    if teacher.observation_space.shape != (48,) or student.observation_space.shape != (22,):
        raise ValueError("expected a 48-input teacher and a 22-input student")
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=1)
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=0.0,
        jump_cooldown_steps=8,
        yaw_scale=0,
        terminate_on_hazard=True,
        privileged_observations=True,
    )
    datasets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    try:
        for stage, episodes, offset in (
            (1, args.stage1_episodes, 0),
            (2, args.stage2_episodes, args.stage1_episodes),
        ):
            stage_observations, stage_labels, stage_metrics = collect_teacher_trajectories(
                teacher,
                batch,
                curriculum_stage=stage,
                episodes=episodes,
                seed_start=args.seed_start + offset,
            )
            datasets.append(stage_observations)
            labels.append(stage_labels)
            metrics.append(stage_metrics)
            print(json.dumps(stage_metrics))
    finally:
        batch.close()
    aggregate_observations = np.concatenate(datasets)
    aggregate_labels = np.concatenate(labels)
    losses = fit_action_head(
        student,
        aggregate_observations,
        aggregate_labels,
        epochs=args.clone_epochs,
        learning_rate=args.learning_rate,
        jump_loss_weight=args.jump_loss_weight,
        seed=20260720,
    )
    np.savez_compressed(
        run_dir / "teacher_dataset.npz",
        observations=aggregate_observations,
        action_labels=aggregate_labels,
    )
    student.save(run_dir / "final_model")
    summary = {
        "teacher": str(args.teacher),
        "initial_student": str(args.student),
        "samples": len(aggregate_observations),
        "metrics": metrics,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "clone_epochs": args.clone_epochs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
