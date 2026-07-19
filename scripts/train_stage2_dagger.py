#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from obby_rl.config import ROOT
from obby_rl.dagger import TakeoffOracle, fit_action_head, fit_jump_head
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from stable_baselines3 import PPO


def collect_iteration(
    model: PPO,
    batch: RobloxObbyBatch,
    oracle: TakeoffOracle,
    *,
    episodes: int,
    beta: float,
    seed: int,
    learn_movement: bool = False,
    course_seed_start: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    next_course_seed = course_seed_start + batch.num_envs
    observations, _ = batch.reset(
        list(range(course_seed_start, course_seed_start + batch.num_envs))
    )
    collected_observations: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    finished = 0
    clean = 0
    hazards = 0
    while finished < episodes:
        labels = oracle.labels(observations)
        policy_actions, _ = model.predict(observations, deterministic=True)
        oracle_actions = np.zeros((batch.num_envs, 4), dtype=np.float32)
        oracle_actions[:, 1] = 1.0
        oracle_actions[:, 3] = labels
        execute_oracle = rng.random(batch.num_envs) < beta
        actions = np.asarray(policy_actions, dtype=np.float32)
        if learn_movement:
            actions = np.where(execute_oracle[:, None], oracle_actions, actions)
        else:
            actions[:, 3] = np.where(execute_oracle, labels, actions[:, 3])
        collected_observations.append(observations.copy())
        collected_labels.append(oracle_actions.copy() if learn_movement else labels.copy())
        next_observations, _, terminated, truncated, infos = batch.step(actions)
        dones = terminated | truncated
        for index in np.flatnonzero(dones):
            if finished >= episodes:
                break
            hazard = bool(infos[index].get("hazard_recovered"))
            hazards += int(hazard)
            clean += int(not hazard and bool(terminated[index]))
            finished += 1
        if np.any(dones) and finished < episodes:
            reset_seeds = [0] * batch.num_envs
            for index in np.flatnonzero(dones):
                reset_seeds[index] = next_course_seed
                next_course_seed += 1
            reset_observations, _ = batch.reset_lanes(reset_seeds, dones)
            next_observations[dones] = reset_observations[dones]
        observations = next_observations
    return (
        np.concatenate(collected_observations),
        np.concatenate(collected_labels),
        {"episodes": finished, "clean": clean, "hazards": hazards, "beta": beta},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight DAgger for Stage 2 jump timing")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "runs/m3-vector-stage1-continuous-hold-lease250-8192/final_model.zip",
    )
    parser.add_argument("--run-name", default="m3-stage2-dagger")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--curriculum-stage", type=int, choices=range(2, 15), default=2)
    parser.add_argument("--course-seed-start", type=int, default=0)
    parser.add_argument("--oracle-min-distance", type=float, default=13.5)
    parser.add_argument("--oracle-max-distance", type=float, default=17.5)
    parser.add_argument(
        "--learn-movement",
        action="store_true",
        help="DAgger all four action outputs and execute policy movement in Roblox",
    )
    parser.add_argument("--clone-epochs", type=int, default=40)
    parser.add_argument("--jump-loss-weight", type=float, default=4.0)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    model = PPO.load(args.model, device="cpu")
    oracle = TakeoffOracle(
        minimum_distance=args.oracle_min_distance,
        maximum_distance=args.oracle_max_distance,
    )
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=0.0,
        jump_cooldown_steps=8,
        yaw_scale=0,
        terminate_on_hazard=True,
        scripted_forward=not args.learn_movement,
    )
    all_observations: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    history: list[dict[str, Any]] = []
    betas = np.linspace(1.0, 0.0, args.iterations)
    try:
        for iteration, beta in enumerate(betas):
            observations, labels, metrics = collect_iteration(
                model,
                batch,
                oracle,
                episodes=args.episodes,
                beta=float(beta),
                seed=20260717 + iteration,
                learn_movement=args.learn_movement,
                course_seed_start=args.course_seed_start + iteration * args.episodes,
            )
            all_observations.append(observations)
            all_labels.append(labels)
            aggregate_observations = np.concatenate(all_observations)
            aggregate_labels = np.concatenate(all_labels)
            if args.learn_movement:
                losses = fit_action_head(
                    model,
                    aggregate_observations,
                    aggregate_labels,
                    epochs=args.clone_epochs,
                    seed=20260717 + iteration,
                    jump_loss_weight=args.jump_loss_weight,
                )
            else:
                losses = fit_jump_head(
                    model,
                    aggregate_observations,
                    aggregate_labels,
                    epochs=args.clone_epochs,
                    seed=20260717 + iteration,
                )
            model.save(run_dir / f"dagger_{iteration + 1}")
            np.savez_compressed(
                run_dir / f"dataset_{iteration + 1}.npz",
                observations=aggregate_observations,
                action_labels=aggregate_labels,
            )
            record = {
                "iteration": iteration + 1,
                **metrics,
                "oracle_min_distance": oracle.minimum_distance,
                "oracle_max_distance": oracle.maximum_distance,
                "samples": len(aggregate_labels),
                "positive_fraction": float(
                    np.mean(
                        (
                            aggregate_labels[:, 3]
                            if args.learn_movement
                            else aggregate_labels
                        )
                        > 0
                    )
                ),
                "learn_movement": args.learn_movement,
                "curriculum_stage": args.curriculum_stage,
                "jump_loss_weight": args.jump_loss_weight,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
            }
            history.append(record)
            print(json.dumps(record))
    finally:
        batch.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    model.save(run_dir / "final_model")
    print(f"saved Stage 2 DAgger run to {run_dir}")


if __name__ == "__main__":
    main()
