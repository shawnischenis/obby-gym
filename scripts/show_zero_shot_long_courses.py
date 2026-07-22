#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from obby_rl.config import ROOT


DEFAULT_MODEL = ROOT / "runs" / "m4-stage20-segment-progress-migration-v1" / "model.zip"


def run_stage(
    *,
    model: Path,
    stage: int,
    seeds: str,
    warmup_seconds: float,
    between_course_seconds: float,
    terminal_hold_seconds: float,
    camera_view: str,
    lives: int,
    retry_timing_shift: float,
) -> None:
    print(f"\n=== zero-shot Stage {stage}: seeds {seeds} ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "show_obby_completion.py"),
            "--model",
            str(model),
            "--curriculum-stage",
            str(stage),
            "--seeds",
            seeds,
            "--warmup-seconds",
            str(warmup_seconds),
            "--between-course-seconds",
            str(between_course_seconds),
            "--terminal-hold-seconds",
            str(terminal_hold_seconds),
            "--visible-lane",
            "1",
            "--camera-view",
            camera_view,
            "--lives",
            str(lives),
            "--retry-timing-shift",
            str(retry_timing_shift),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record zero-shot four- and eight-obstacle mixed-course attempts"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stage21-seeds", default="21010,21012")
    parser.add_argument("--stage22-seeds", default="22005")
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--between-course-seconds", type=float, default=1.5)
    parser.add_argument("--terminal-hold-seconds", type=float, default=3.0)
    parser.add_argument(
        "--lives",
        type=int,
        default=3,
        help="falls allowed per zero-shot course before moving to the next seed",
    )
    parser.add_argument(
        "--retry-timing-shift",
        type=float,
        default=1.0,
        help="stud offset used for controlled earlier/later retry timing",
    )
    parser.add_argument(
        "--camera-view",
        choices=("completion", "completion-side", "completion-follow"),
        default="completion",
    )
    args = parser.parse_args()

    for stage, seeds in ((21, args.stage21_seeds), (22, args.stage22_seeds)):
        run_stage(
            model=args.model,
            stage=stage,
            seeds=seeds,
            warmup_seconds=args.warmup_seconds,
            between_course_seconds=args.between_course_seconds,
            terminal_hold_seconds=args.terminal_hold_seconds,
            camera_view=args.camera_view,
            lives=args.lives,
            retry_timing_shift=args.retry_timing_shift,
        )


if __name__ == "__main__":
    main()
