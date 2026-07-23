# Obby Gym media

This directory stages authentic Roblox captures before they are optimized and
copied into `docs/media/` for the project site. Do not use generated gameplay or
reconstructed scenes as experimental evidence.

## Capture order

Capture these four clips first. The filenames are intentional; keeping them
stable lets the site be redesigned before every asset is final.

| Priority | Capture | Candidate filename | What it should prove |
| --- | --- | --- | --- |
| 1 | Eight agents training | `candidates/video/parallel-training.mov` | Eight independent lanes advance from one batched Python action step. |
| 2 | Varied jumps | `candidates/video/varied-jumps.mov` | The policy handles visibly different gap, height, and angle combinations. |
| 3 | Full completion | `candidates/video/obby-completion.mov` | One deterministic, uninterrupted evaluation completes a multi-obstacle course. |
| 4 | Seeded generation | `candidates/video/course-generation.mov` | Multiple seeds produce different valid layouts without manual construction. |

Also save one still frame from each clip under `candidates/images/`. These become
video posters and work as fallbacks for reduced-motion visitors.

## Visual direction

- Record the game viewport only; hide Studio panels, selection outlines, and the
  mouse cursor whenever possible.
- Use the same camera height, field of view, lighting, and avatar appearance
  across clips.
- Prefer a neutral three-quarter camera that makes gap length and landing height
  readable. Avoid rapid camera movement.
- Keep diagnostic overlays minimal. The parallel clip may label lanes 1–8; the
  varied-jump clip may show seed, gap, height, and angle in one quiet corner.
- Record deterministic evaluation for completion footage. Do not select only a
  successful fragment from a failed episode.
- Capture at 1920×1080 or higher, 60 FPS if available. The final website export
  will normally be 1280×720 or 1600×900 at 30 FPS.
- Record without music or microphone audio. The website videos will be muted.

## Shot specifications

### Parallel training

- Duration: 8–12 seconds.
- Frame all eight lanes simultaneously from a fixed elevated camera.
- Begin with all agents grounded and include at least one reset or visibly
  different action per lane.
- This should be the clearest systems clip, not the hero clip.

For a clean capture, show stochastic rollout collection from the promoted
student policy. This is the environment-facing portion of PPO training, without
the visually distracting optimization pauses between rollout batches:

```bash
.venv/bin/python scripts/show_parallel_rollout.py \
  --curriculum-stage 2 \
  --num-envs 4 \
  --duration-seconds 20 \
  --warmup-seconds 5 \
  --seed-start 3000
```

The command loads `runs/m3-stage1-2-student-ppo-v1/final_model.zip` by default.
Pass `--model PATH` to record another compatible 22-input PPO checkpoint. Add
`--deterministic` for matched evaluation behavior; leave it off when depicting
stochastic training rollouts.

### Varied jumps

- Duration: 10–16 seconds.
- Show at least four geometries: shorter/longer gap, high-to-low, low-to-high,
  and an angled landing.
- A four-panel synchronized capture is ideal. A restrained sequence of four
  cuts is the fallback.
- Keep the takeoff and landing visible in every panel.

The two-agent recording view uses Stage 11, where gap, landing height, and
approach angle vary together. It loads the promoted steering checkpoint and
places the camera behind both agents:

```bash
.venv/bin/python scripts/show_parallel_rollout.py \
  --model runs/m3-stage9-small-angle-dagger-steering-v1/final_model.zip \
  --curriculum-stage 11 \
  --num-envs 2 \
  --duration-seconds 35 \
  --warmup-seconds 5 \
  --reset-delay-seconds 0.5 \
  --camera-view side \
  --seed-start 5000 \
  --deterministic
```

Each synchronized reset advances to two new seeds, so a longer take naturally
contains multiple gap, height, and angle combinations.

### Obby completion

- Duration: 15–30 seconds, or the full episode if shorter.
- Use a held-out procedural seed and deterministic policy actions.
- Start before the first action and end after the finish state is unambiguous.
- This is the preferred hero media because it communicates the result without
  requiring technical context.

The strongest promoted multi-obstacle policy currently completes Stage 20's two
chained mixed segments at 122/128 across its promotion partitions. Record that
validated scope rather than implying the unpromoted eight-segment course is
solved:

```bash
.venv/bin/python scripts/show_obby_completion.py \
  --seed 0 \
  --episodes 5 \
  --warmup-seconds 5 \
  --between-course-seconds 1 \
  --terminal-hold-seconds 2
```

The script preserves the original eight-lane simulation cadence while rendering
only lane 1. The two-second terminal hold makes the finish state readable. If seed 0 is one
of the uncommon failures, increment `--seed-start`; keep the complete attempt in
the final clip rather than cutting around a failure.

### Course generation

- Duration: 6–10 seconds.
- Use one fixed overview camera and cycle through at least four named seeds.
- Hold each generated layout long enough to see the route change.
- If live regeneration looks visually noisy, capture four stills instead and
  present them as a simple 2×2 grid.

With the latest place open and playing in Studio, cycle 12 full procedural
courses at a two-second cadence:

```bash
.venv/bin/python scripts/show_course_generation.py \
  --curriculum-stage 4 \
  --count 12 \
  --seed-start 1000 \
  --hold-seconds 2
```

Start screen recording before running the command. Use a fixed overview camera
and trim the connection/setup time from the final clip. Stage 4 is the full
eight-segment procedural course; the other curriculum stages are mostly focused
skill lessons and will not show the same layout variety.

## Selection and export

Keep original recordings in `candidates/`. Once selected, export:

```text
docs/media/course-generation.mp4      + course-generation.jpg
docs/media/parallel-training.mp4      + parallel-training.jpg
docs/media/varied-jumps.mp4           + varied-jumps.jpg      (side view)
docs/media/back-view-jumps.mp4        + back-view-jumps.jpg   (back view)
docs/media/varied-jumps-dual.mp4      + varied-jumps-dual.jpg (side + back stitched vertically; used on the site)
docs/media/two-segment-completion.mp4 + two-segment-completion.jpg
docs/media/two-segment-front.mp4      + two-segment-front.jpg
docs/media/zero-shot.mp4              + zero-shot.jpg
```

Current exports were produced from the original captures with
`ffmpeg -crf 20 -preset slow -pix_fmt yuv420p -movflags +faststart`, native
resolution, 30 FPS, audio stripped. Posters are single frames at up to 1200 px
wide, JPEG quality 4.

The site's "Episode trace" figure in the Results section reads
`docs/media/episode-trace.json` and stays hidden until that file exists.
Generate it from a real deterministic episode (Studio playing, bridge plugin
installed) with:

```bash
.venv/bin/python scripts/export_episode_trace.py \
  --model runs/m4-stage20-two-segment-replay-v1/checkpoints/ppo_vector_4096_steps.zip \
  --curriculum-stage 20 --seed 0
```

Do not hand-write this file; the figure is presented as a recorded episode.

Recommended delivery settings:

- MP4, H.264, `yuv420p`, no audio, 30 FPS.
- Aim for 2–6 MB per clip and less than 300 KB per poster.
- Avoid GIF for gameplay; it is larger and has worse color and motion quality.
- Preserve the original recording until the page is finished.

The final page should use native `<video autoplay muted loop playsinline>` with
the matching WebP poster and a still-image fallback for reduced motion.
