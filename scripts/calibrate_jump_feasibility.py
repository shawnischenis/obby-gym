#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from obby_rl.config import ROOT
from obby_rl.feasibility import fit_feasibility_model, predict_probabilities
from obby_rl.transport import StudioHTTPTransport
from validate_vector_jump_timing import run_group


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    high_confidence = probabilities >= 0.7
    return {
        "samples": len(labels),
        "mean_target_probability": float(np.mean(labels)),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "mean_absolute_error": float(np.mean(np.abs(probabilities - labels))),
        "high_confidence_samples": int(np.sum(high_confidence)),
        "high_confidence_target_probability": float(
            np.mean(labels[high_confidence]) if np.any(high_confidence) else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Stage 17 jump feasibility")
    parser.add_argument("--run-name", default="m3-stage17-feasibility-v1")
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--development-seeds", type=int, default=24)
    parser.add_argument("--validation-seeds", type=int, default=8)
    parser.add_argument("--max-jump-step", type=int, default=9)
    parser.add_argument("--min-jump-step", type=int, default=0)
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--lane-balanced", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if not 0 <= args.min_jump_step <= args.max_jump_step:
        raise ValueError("jump-step range must be non-negative and increasing")
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path: Path = run_dir / "progress.json"
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=17,
        action_repeat_ticks=args.action_repeat_ticks,
    )
    records: list[dict[str, object]] = (
        json.loads(progress_path.read_text()) if progress_path.exists() else []
    )
    completed_seeds = {int(record["seed"]) for record in records}
    seed_count = args.development_seeds + args.validation_seeds
    try:
        for offset in range(seed_count):
            seed = args.seed_start + offset
            if seed in completed_seeds:
                print(json.dumps({"seed": seed, "status": "resumed"}), flush=True)
                continue
            groups = (
                [
                    [jump_step] * 8
                    for jump_step in range(args.min_jump_step, args.max_jump_step + 1)
                ]
                if args.lane_balanced
                else [
                    list(range(start, min(start + 8, args.max_jump_step + 1)))
                    for start in range(args.min_jump_step, args.max_jump_step + 1, 8)
                ]
            )
            repetitions = 1 if args.lane_balanced else args.repetitions
            for repetition in range(repetitions):
                for jump_steps in groups:
                    group = run_group(transport, jump_steps, seed)
                    for result in group:
                        if result["observation"] is not None:
                            records.append(
                                {
                                    "seed": seed,
                                    "repetition": repetition,
                                    "jump_step": int(result["jump_step"]),
                                    "distance": float(result["distance"]),
                                    "gap": float(result["gap"]),
                                    "outcome": str(result["outcome"]),
                                    "observation": result["observation"],
                                }
                            )
            progress_path.write_text(json.dumps(records) + "\n")
            successes = sum(
                record["outcome"] == "complete" for record in records if record["seed"] == seed
            )
            print(json.dumps({"seed": seed, "successful_timings": successes}), flush=True)
    finally:
        transport.close()
    observations = np.asarray([record["observation"] for record in records], dtype=np.float32)
    group_success: dict[tuple[int, int], list[bool]] = {}
    for record in records:
        key = (int(record["seed"]), int(record["jump_step"]))
        group_success.setdefault(key, []).append(record["outcome"] == "complete")
    labels = np.asarray(
        [
            np.mean(group_success[(int(record["seed"]), int(record["jump_step"]))])
            for record in records
        ],
        dtype=np.float32,
    )
    seeds = np.asarray([record["seed"] for record in records], dtype=np.int64)
    development = seeds < args.seed_start + args.development_seeds
    model, losses = fit_feasibility_model(
        observations[development], labels[development], epochs=args.epochs, seed=20260720
    )
    development_probabilities = predict_probabilities(model, observations[development])
    validation_probabilities = predict_probabilities(model, observations[~development])
    summary = {
        "action_repeat_ticks": args.action_repeat_ticks,
        "repetitions": args.repetitions,
        "lane_balanced": bool(args.lane_balanced),
        "development": metrics(labels[development], development_probabilities),
        "validation": metrics(labels[~development], validation_probabilities),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    np.savez_compressed(
        run_dir / "dataset.npz",
        observations=observations,
        labels=labels,
        seeds=seeds,
        gaps=np.asarray([record["gap"] for record in records], dtype=np.float32),
        distances=np.asarray([record["distance"] for record in records], dtype=np.float32),
        jump_steps=np.asarray([record["jump_step"] for record in records], dtype=np.int32),
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
