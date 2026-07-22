# Roblox place archive

This folder keeps the reusable Studio places separate from source code and run
artifacts.

## `training/`

- `ObbyRL-M4-Landing-Replay-v8.rbxlx` is the current training place with
  segment-relative progress and real landing-state replay.
- `ObbyRL-M4-Required-Jumps-v8.rbxlx` is a v8-compatible rebuild for the
  required-jump training workflow.
- `ObbyRL-Mixed-Landing-Replay-v9.rbxlx` is the current mixed-course training
  place. Stages 20, 21, and 22 contain two, four, and eight randomized segments
  respectively, sampling gaps, angled jumps, beams, and stairs. Intermediate
  replay can restore any captured nonterminal checkpoint.

## `recording/`

- `ObbyRL-Obby-Completion-Recording-v8.rbxlx` is the current completion capture
  place.
- `ObbyRL-Parallel-Recording-v8.rbxlx` and
  `ObbyRL-Varied-Jumps-Recording-v8.rbxlx` are v8-compatible rebuilds for the
  parallel and varied-jump capture workflows.

## `historical/`

These nine binaries were recovered byte-for-byte from Git after the cleanup.
They retain the source and observation semantics embedded at their original
milestones. They are useful for reproducing historical runs, but should not be
combined with the v8 segment-progress policy unless explicitly migrated.

The later untracked v3-v7 recording binaries could not be recovered exactly
after deletion. Their current-source v8 replacements are stored under
`recording/` instead.

## Mixed-obstacle demo capture

Open `training/ObbyRL-Mixed-Landing-Replay-v9.rbxlx`, press Play, and run
`scripts/show_obby_completion.py` with the migrated Stage 20 model. The vetted
visible-lane seeds `20002,20005,20006,20012,20013` completed cleanly in the live
recording cadence on July 22, 2026. Explicit `--seeds` support keeps the demo
sequence independent of hidden-lane cohort numbering.

For a single recording sequence containing two zero-shot four-obstacle attempts
followed by two zero-shot eight-obstacle attempts, run
`scripts/show_zero_shot_long_courses.py`. The default seeds completed in batch
evaluation, but longer-course footage should still be labeled as attempts because
recording-mode reruns are not deterministic.
