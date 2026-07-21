#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from obby_rl.config import ROOT
from obby_rl.dagger import fit_jump_head
from obby_rl.vector_env import (
    PRIVILEGED_DECISION_PHASE_INDEX,
    PRIVILEGED_DECISION_PHASE_SCALE,
)
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Stage 17 macro timing into a teacher")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--jump-step", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-loss-weight", type=float, default=8.0)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    model = PPO.load(args.teacher, device="cpu")
    dataset = np.load(args.dataset)
    observations = np.asarray(dataset["observations"], dtype=np.float32).copy()
    jump_steps = np.asarray(dataset["jump_steps"], dtype=np.int32)
    observations[:, PRIVILEGED_DECISION_PHASE_INDEX] = np.clip(
        jump_steps / PRIVILEGED_DECISION_PHASE_SCALE, 0.0, 1.0
    )
    labels = np.where(jump_steps == args.jump_step, 1.0, -1.0).astype(np.float32)
    losses = fit_jump_head(
        model,
        observations,
        labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=20260720,
        train_actor_representation=True,
        positive_loss_weight=args.positive_loss_weight,
    )
    model.save(run_dir / "final_model")
    summary = {
        "samples": len(observations),
        "positive_samples": int(np.sum(labels > 0)),
        "jump_step": args.jump_step,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
