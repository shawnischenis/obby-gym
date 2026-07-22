#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from obby_rl.config import ROOT
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from stable_baselines3 import PPO


MODEL = (
    ROOT
    / "runs"
    / "m4-stage23-required-jumps-replay-v1"
    / "checkpoints"
    / "ppo_vector_4096_steps.zip"
)


def main() -> None:
    model = PPO.load(MODEL, device="cpu")
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=23)
    # Continuous jump intent plus the geometry gate supplies reliable takeoff
    # timing while learned movement produces natural landing states.
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=-2.0,
        mask_jump_to_takeoff_window=True,
        terminate_on_hazard=False,
    )
    seeds = list(range(24000, 24008))
    try:
        observations, _ = batch.reset(seeds)
        captured = np.zeros(8, dtype=np.bool_)
        for _ in range(160):
            actions, _ = model.predict(observations, deterministic=True)
            observations, _, _, _, infos = batch.step(actions)
            captured |= np.asarray(
                [int(info.get("checkpoint_index", 0)) >= 1 for info in infos],
                dtype=np.bool_,
            )
            if np.all(captured):
                break
        if not np.all(captured):
            raise RuntimeError(f"failed to capture natural landings for lanes {np.flatnonzero(~captured) + 1}")

        observations, infos = batch.reset(seeds, np.ones(8, dtype=np.bool_))
        for index, (observation, info) in enumerate(zip(observations, infos, strict=True)):
            velocity = observation[:3] * np.asarray([32.0, 32.0, 32.0])
            previous_action = observation[18:22]
            print(
                f"lane={index + 1} replayed={bool(info.get('post_landing_reset'))} "
                f"seed={int(info.get('course_seed', -1))} "
                f"checkpoint={int(info.get('checkpoint_index', -1))} "
                f"grounded={bool(observation[4] > 0.5)} "
                f"cooldown={int(info.get('restored_jump_cooldown', -1))} "
                f"velocity=({velocity[0]:.2f},{velocity[1]:.2f},{velocity[2]:.2f}) "
                f"previous_action={previous_action.tolist()} "
                f"segment_progress={float(observation[12]):.3f}"
            )
        if any(int(info.get("checkpoint_index", -1)) != 1 for info in infos):
            raise RuntimeError("landing-state replay did not restore checkpoint one")
        if not all(bool(info.get("post_landing_reset")) for info in infos):
            raise RuntimeError("one or more lanes fell back instead of replaying a snapshot")
    finally:
        batch.close()


if __name__ == "__main__":
    main()
