#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO


SEGMENT_PROGRESS_INDEX = 12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the legacy absolute-progress dependency from a PPO checkpoint"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    changed: list[str] = []
    with torch.no_grad():
        for name, network in (
            ("actor", model.policy.mlp_extractor.policy_net),
            ("critic", model.policy.mlp_extractor.value_net),
        ):
            first_linear = next(layer for layer in network if isinstance(layer, torch.nn.Linear))
            first_linear.weight[:, SEGMENT_PROGRESS_INDEX] = 0
            changed.append(f"{name}.first_linear.weight[:,{SEGMENT_PROGRESS_INDEX}]")

    args.output.mkdir(parents=True, exist_ok=False)
    model.save(args.output / "model")
    (args.output / "migration.json").write_text(
        json.dumps(
            {
                "source_model": str(args.model),
                "output_model": str(args.output / "model.zip"),
                "observation_index": SEGMENT_PROGRESS_INDEX,
                "old_semantics": "absolute_course_progress",
                "new_semantics": "segment_relative_progress",
                "operation": "zero input column",
                "changed_parameters": changed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved migrated checkpoint to {args.output / 'model.zip'}")


if __name__ == "__main__":
    main()
