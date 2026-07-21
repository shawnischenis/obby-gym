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


def collect_iteration(
    teacher: PPO,
    student: PPO,
    batch: RobloxObbyBatch,
    *,
    episodes: int,
    beta: float,
    seed_start: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    next_seed = seed_start
    collected_observations: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
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
            student_observations = batch.student_observations
            teacher_actions, _ = teacher.predict(teacher_observations, deterministic=True)
            student_actions, _ = student.predict(student_observations, deterministic=True)
            teacher_actions = np.asarray(teacher_actions, dtype=np.float32)
            student_actions = np.asarray(student_actions, dtype=np.float32)
            execute_teacher = rng.random(batch.num_envs) < beta
            actions = np.where(execute_teacher[:, None], teacher_actions, student_actions)
            actions[~active] = 0
            collected_observations.append(student_observations[active].copy())
            collected_labels.append(teacher_actions[active].copy())
            teacher_observations, _, terminated, truncated, infos = batch.step(actions)
            dones = active & (terminated | truncated)
            for index in np.flatnonzero(dones):
                hazard = bool(infos[index].get("hazard_recovered"))
                hazards += int(hazard)
                clean += int(bool(terminated[index]) and not hazard)
                finished += 1
                active[index] = False
    return (
        np.concatenate(collected_observations),
        np.concatenate(collected_labels),
        {"episodes": episodes, "clean": clean, "hazards": hazards, "beta": beta},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAgger a limited-sensing student with a privileged PPO teacher"
    )
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--base-dataset", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=4_000_000)
    parser.add_argument("--clone-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--jump-loss-weight", type=float, default=4.0)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    teacher = PPO.load(args.teacher, device="cpu")
    student = PPO.load(args.student, device="cpu")
    all_observations: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    if args.base_dataset:
        base = np.load(args.base_dataset)
        all_observations.append(np.asarray(base["observations"], dtype=np.float32))
        all_labels.append(np.asarray(base["action_labels"], dtype=np.float32))
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=2)
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=0.0,
        jump_cooldown_steps=8,
        yaw_scale=0,
        terminate_on_hazard=True,
        privileged_observations=True,
    )
    history: list[dict[str, Any]] = []
    betas = np.linspace(0.75, 0.0, args.iterations)
    try:
        for iteration, beta in enumerate(betas):
            observations, labels, metrics = collect_iteration(
                teacher,
                student,
                batch,
                episodes=args.episodes,
                beta=float(beta),
                seed_start=args.seed_start + iteration * args.episodes,
                rng=np.random.default_rng(20260720 + iteration),
            )
            all_observations.append(observations)
            all_labels.append(labels)
            aggregate_observations = np.concatenate(all_observations)
            aggregate_labels = np.concatenate(all_labels)
            losses = fit_action_head(
                student,
                aggregate_observations,
                aggregate_labels,
                epochs=args.clone_epochs,
                learning_rate=args.learning_rate,
                jump_loss_weight=args.jump_loss_weight,
                seed=20260720 + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **metrics,
                "new_samples": len(observations),
                "aggregate_samples": len(aggregate_observations),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
            }
            history.append(record)
            student.save(run_dir / f"dagger_{iteration + 1}")
            np.savez_compressed(
                run_dir / f"dataset_{iteration + 1}.npz",
                observations=aggregate_observations,
                action_labels=aggregate_labels,
            )
            print(json.dumps(record))
    finally:
        batch.close()
    student.save(run_dir / "final_model")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
