# General Roblox Obby Agent — Project Plan

## Goal

Train an agent that can complete procedurally generated Roblox obstacle courses, then run the trained policy entirely in Roblox through a Luau inference module.

The core comparison is:

1. **Fixed-course policy** — trained on one course/seed.
2. **Domain-randomized policy** — trained on randomized, procedurally valid courses.
3. **Recurrent randomized policy** — trained on randomized courses with memory (for partial observability).

The main success criterion is not training return. It is completion rate on a held-out suite of course seeds and obstacle combinations that were never used for training.

## Recommended technical direction

Use **PPO as the primary algorithm** for all three comparisons. Its action space can naturally represent movement direction, camera/heading change, and jumping. Use the same feed-forward PPO architecture for experiments 1 and 2, then swap only the network for recurrent PPO (LSTM) in experiment 3.

DQN can be added as a secondary baseline, but it requires enumerating a small discrete action set such as `forward`, `left`, `right`, and combinations with `jump`. That makes the comparison partly about action-space design rather than generalization. If retained, use it as an ablation rather than the project's main line.

Suggested Python stack:

- [Gymnasium](https://gymnasium.farama.org/) for the environment contract.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for feed-forward experiments.
- [sb3-contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) for the memory experiment.
- PyTorch during training; export plain tensors and implement only the required forward pass in Luau.
- TensorBoard plus CSV/JSON episode logs for reproducible comparisons.

## Proposed architecture

```text
CourseConfig + seed
        |
        v
Python experiment runner -----> Roblox course generator (Luau)
        ^                                |
        |                                v
Gymnasium adapter <------ observations, rewards, episode events
        |
        v
PPO / RecurrentPPO
        |
        v
versioned policy artifact (weights + normalization + schema)
        |
        v
Luau MLP/LSTM inference module -----> Roblox agent controller
```

The repository's earlier MCP skeleton compiled JSON specs and reusable primitives into deterministic Luau build plans; it is preserved under `legacy/mcp_build_skeleton/`. Reuse that data-driven idea where it helps, but keep the active obby runtime in the Rojo project and make a seeded `CourseConfig`—not the generated Roblox instance tree—the source of truth.

### Roblox-side modules

- `CourseGenerator`: accepts a seed and difficulty/config, samples obstacles, validates them, and builds tagged geometry.
- `CourseManifest`: records the exact sampled parameters, ordered checkpoints, safe spawn transforms, bounds, and optimal/nominal route metadata.
- `AgentHarness`: owns reset, action application, fixed simulation cadence, observation collection, termination, and reward events.
- `Bridge`: batches rollout data and receives actions/configuration from Python.
- `PolicyRuntime`: loads exported weights, applies the saved observation normalization, and runs MLP or LSTM inference in Luau.
- `Evaluator`: runs fixed seed lists without training-time exploration and emits machine-readable results.

### Python-side modules

- `RobloxObbyEnv(gymnasium.Env)`: validates observation/action schemas and exposes `reset()`/`step()`.
- `transport`: handles request IDs, timeouts, reconnects, batching, and protocol-version checks.
- `train.py`: creates the selected policy and records all experiment configuration.
- `evaluate.py`: evaluates every checkpoint on the same held-out seeds.
- `export_policy.py`: converts PyTorch parameters into a deterministic, versioned Luau artifact.
- `parity_test.py`: feeds identical observations through PyTorch and Luau and compares logits, values, and recurrent state.

## Observation and action contract (v1)

Start with structured, agent-relative observations rather than pixels. Keep the contract small enough for fast Luau inference and explicit enough to diagnose failures.

Suggested observation vector:

- Root linear velocity and angular/heading velocity in agent-local coordinates.
- Grounded flag, humanoid state, jump cooldown, and time since last grounded.
- Relative vector to the current checkpoint and next checkpoint.
- Current checkpoint index and normalized progress.
- A compact geometry probe: ray distances in a fixed fan (forward/down/left/right and diagonals), surface normals, and hazard flags.
- Previous action.
- Optional obstacle metadata visible to the agent, but never the generator seed or privileged future layout.

Every scalar needs a documented unit, range, clipping rule, and normalization rule. Store the observation schema hash beside every checkpoint and exported policy.

Suggested action:

- Two bounded values for local movement (`strafe`, `forward`).
- One bounded heading/yaw command, if the controller needs explicit facing.
- One Bernoulli/discrete jump command.

If library integration makes the mixed action space cumbersome, discretize movement into 8 directions plus idle and keep jump as a binary branch. Avoid a large Cartesian enumeration.

## Reward and episode design

Use rewards that encourage course progress without teaching seed-specific shortcuts:

- Positive delta in route/checkpoint progress.
- One-time checkpoint bonus.
- Large completion bonus.
- Small time penalty.
- Fall/death penalty.
- Optional small control-change penalty only if the learned policy visibly chatters.

Do not reward raw world-axis movement; randomized courses make that exploitable. Cap or make checkpoint rewards one-shot. Terminate on completion, death/fall, stuck detection, or a fixed time limit. Log every reward component separately.

## Procedural course generation

Build the generator as a grammar of obstacle segments with explicit entry/exit transforms:

- Flat platform and gap jump.
- Offset/lateral jump.
- Narrow beam.
- Stairs or height change.
- Moving platform (later curriculum phase).
- Rotating/sweeping hazard (later curriculum phase).
- Choice/fork or temporarily occluded route for the memory experiment.

Each segment should declare parameter ranges, required clearance, difficulty features, and a conservative reachability condition. Assemble segments by matching the previous exit transform to the next entry transform, then perform overlap and bounds checks. Initially validate reachability analytically with conservative avatar jump limits; later add scripted oracle/bot rollouts as a stronger validation layer.

Use a counter-based or otherwise deterministic seed strategy so `generator_version + config + seed` reconstructs the exact course. Split seeds before training:

- Training seeds/configurations.
- Validation seeds for model selection and curriculum decisions.
- Test seeds, locked until final evaluation.
- Stress seeds that deliberately push parameter boundaries.

Existing “AI obby maker” products can help with visual ideation, but most produce a finished level rather than a deterministic research generator. The smoother path here is a small in-repo parametric generator. The current primitive/manifest system is a good foundation for reusable segment builders. Rojo is also worth adopting if the Roblox project grows, because it keeps Luau and project structure in normal source control ([Rojo documentation](https://rojo.space/docs/)).

## Studio ↔ Python bridge

Prototype with a local Python HTTP service and Roblox `HttpService`. Roblox supports outbound requests and JSON encoding, but HTTP must be enabled and requests have platform limits; Roblox currently documents an overall external-request limit of 500 requests/minute per game server ([HttpService reference](https://create.roblox.com/docs/reference/engine/classes/HttpService)). Therefore:

- Do not make one HTTP request per physics frame.
- Prefer action repeat (for example, hold an action for several simulation ticks).
- Batch transitions/rollout fragments where possible.
- Include `protocol_version`, `episode_id`, `step_id`, and `course_seed` in messages.
- Add timeouts, idempotent reset semantics, and clear failure termination.

Measure throughput in milestone 1 before committing to a long training campaign. Roblox Studio is unlikely to match purpose-built simulators, so environment throughput is a first-class project risk. The official Studio CLI can run Luau scripts against a local or published place and write output logs, which is useful for generator tests, smoke tests, and repeatable evaluation orchestration ([Studio command-line interface](https://create.roblox.com/docs/studio/command-line-interface)). The official Studio MCP supports exploration and playtest scenarios and is useful for development/debugging, but the training data plane should remain a simple versioned protocol ([Studio MCP documentation](https://create.roblox.com/docs/studio/mcp)).

## Milestones

### M0 — Reproducible project skeleton

Deliverables:

- Decide and document avatar rig, physics settings, simulation/action cadence, and maximum episode length.
- Add Python packaging, pinned dependencies, configuration files, and experiment output conventions.
- Add a Roblox project layout (preferably Rojo) without breaking the current build-plan workflow.
- Define v1 observation, action, reset, and rollout message schemas.
- Define global seeds and deterministic naming/versioning rules.

Exit criteria:

- A clean checkout can build/open the test place and run a scripted smoke test.
- An experiment configuration fully identifies code version, generator version, seed sets, and hyperparameters.

### M1 — End-to-end single-obstacle environment

Deliverables:

- One simple gap-jump course generated from a seed.
- Agent reset and action control.
- Structured observation collection.
- Python `Gymnasium` environment with random-agent and scripted-agent tests.
- Batched transport benchmark and failure recovery.

Exit criteria:

- At least 1,000 unattended resets complete without leaked characters/instances or protocol desynchronization.
- Identical seeds reproduce identical manifests.
- Report measured environment steps/second and reset latency.

### M2 — Procedural obby generator v1

Deliverables:

- Four static segment families: gap, offset jump, beam, and stairs/height change.
- Constraint-based assembly, overlap checking, kill plane, checkpoints, and finish trigger.
- Serializable course manifest and replay-by-seed command.
- Generator property tests over thousands of seeds.
- A small hand-authored/scripted oracle to reject obviously impossible courses.

Exit criteria:

- No overlaps/out-of-bounds geometry in the validation sample.
- Every sampled course has a continuous checkpoint chain satisfying conservative reachability constraints.
- Failed generation resamples deterministically and reports a reason.

### M3 — Fixed-course PPO baseline

Deliverables:

- Feed-forward PPO training on one fixed seed.
- Reward-component dashboards and periodic deterministic checkpoints.
- Evaluation on the training course plus the untouched randomized validation suite.

Exit criteria:

- High completion rate on the fixed course (target: at least 90% over 100 deterministic episodes).
- Generalization result recorded even if it is poor; this is the control condition.

### M4 — Domain-randomized PPO

Deliverables:

- Curriculum from easy parameter ranges/short courses to the full training distribution.
- Use bridge stages for narrow beams, stairs, then 2-, 4-, and 8-segment mixed
  courses; 9-10 stud jumps are excluded from the core distribution.
- Per-episode seeded course randomization.
- Same observation, action, reward, model capacity, and training-step budget as the fixed baseline wherever possible.
- Optional parallel Studio workers only after one worker is reliable.

Exit criteria:

- Material improvement over fixed-course PPO on held-out seeds.
- Training and validation performance broken down by obstacle family and difficulty bin.

### M5 — Recurrent randomized PPO

Deliverables:

- LSTM policy using RecurrentPPO.
- Correct hidden-state reset on episode boundaries.
- Partially observable course cases where memory is plausibly useful: occluded landing information, delayed cues, forks, or moving hazards.
- Sequence-length and memory ablations.

Exit criteria:

- Demonstrate benefit on a predeclared memory-sensitive subset, not merely a larger-network benefit.
- Compare against a parameter-matched feed-forward model and, ideally, a frame/action history baseline.

### M6 — Luau policy export and parity

Deliverables:

- Versioned policy artifact containing architecture, tensors, observation normalization, action mapping, schema hash, and numeric precision.
- Luau dense-layer/activation implementation; LSTM implementation for the recurrent policy.
- Golden test vectors produced by Python.
- Runtime performance profiling in Studio and on a representative Roblox client.

Exit criteria:

- PyTorch and Luau outputs agree within a declared tolerance across at least 1,000 test observations/sequences.
- Exported policy completes the same evaluation suite with no statistically meaningful regression from the Python-driven policy.
- Inference stays within the chosen frame-time budget with no per-step allocations large enough to cause visible spikes.

### M7 — Final controlled comparison

Freeze all three selected checkpoints before opening the test set. Evaluate with the same avatar, action cadence, observation schema, episode time limit, and test seeds.

Primary metrics:

- Course completion rate with bootstrap confidence intervals.
- Median progress before termination.
- Median completion time among successes.
- Death/fall rate and stuck/time-out rate.
- Completion by obstacle family, difficulty, course length, and seed.

Secondary metrics:

- Sample efficiency (environment steps to threshold performance).
- Wall-clock training time and environment throughput.
- Policy size and Luau inference time.
- Robustness under modest physics/observation perturbations.

Recommended evaluation matrix:

| Policy | Fixed train seed | Held-out in-distribution seeds | Harder geometry | Memory-sensitive courses |
|---|---:|---:|---:|---:|
| Fixed feed-forward PPO | yes | yes | yes | yes |
| Randomized feed-forward PPO | yes | yes | yes | yes |
| Randomized recurrent PPO | yes | yes | yes | yes |

Run multiple independent training seeds per condition (preferably 3–5). A single run per policy is too noisy to support a useful conclusion.

## Tooling recommendations

- **Rojo**: source-controlled Roblox project syncing/building.
- **Wally**: optional Luau package management if dependencies are introduced.
- **StyLua + Selene**: formatting and static analysis for Luau.
- **Ruff + pytest + mypy/pyright**: Python quality and tests.
- **Gymnasium + Stable-Baselines3 + sb3-contrib**: environment and algorithms.
- **TensorBoard**: training curves; keep raw CSV/JSON alongside it.
- **Roblox Studio CLI**: unattended generator/smoke scripts and log capture.
- **Roblox Studio MCP**: interactive inspection and playtest debugging, not high-frequency rollout transport.
- **Open Cloud Place Publishing**: optional later CI deployment; Roblox documents it as a way to automate updating an existing place ([publishing guide](https://create.roblox.com/docs/cloud/guides/usage-place-publishing)). It is not needed for the local training prototype.

## Major risks and mitigations

| Risk | Consequence | Early mitigation |
|---|---|---|
| Studio simulation throughput is low | Training takes days or is impractical | Benchmark in M1; use action repeat, compact observations, rollout batching, and multiple workers only after reliability |
| Generator emits impossible courses | Agent is punished for unsolvable episodes | Conservative reachability constraints, deterministic rejection/resampling, oracle checks, manifest replay |
| Reward hacking | High return without completing courses | One-shot checkpoint rewards, route-relative progress, completion-centric evaluation, replay suspicious episodes |
| Fixed and randomized experiments are not comparable | Invalid conclusions | Freeze shared schemas, architecture/capacity, step budgets, reward, and evaluation seeds |
| Recurrent policy wins only because it is larger | Misattributed memory benefit | Parameter-matched feed-forward control and explicit partially observable test subset |
| Python/Luau numeric mismatch | Exported policy performs worse | Golden-vector parity tests for preprocessing, logits/actions, and LSTM state |
| Policy uses privileged generator data | Unrealistic generalization result | Separate manifest/oracle data from the agent-visible observation builder |
| Engine updates change physics | Results drift | Record Studio/client version when possible and maintain a short physics regression suite |

## Suggested first implementation slice

Keep the first slice deliberately narrow:

1. Add seeded `GapSegment` and `CourseGenerator` modules to the active Rojo project, borrowing the legacy manifest system's data-driven approach where useful.
2. Generate a 3-segment course from `seed`, with a manifest containing entry/exit transforms and checkpoint positions.
3. Add a Roblox harness that accepts an action, advances for a fixed action-repeat window, and returns the v1 structured observation and reward components.
4. Wrap it in a Gymnasium environment and run random-policy soak tests.
5. Train a small PPO on one fixed 3-segment course.
6. Export that small MLP to Luau and prove Python/Luau parity before scaling course diversity or adding recurrence.

This slice attacks the two largest unknowns—training throughput and export parity—before substantial work is invested in a sophisticated generator.

## Decisions to lock before M1

- R6 versus R15 avatar and exact humanoid/controller implementation.
- Whether actions drive `Humanoid:Move()` or a lower-level custom character controller.
- Physics step/action-repeat convention and whether training relies on real-time Studio execution.
- Observation ray layout and maximum observation dimension.
- Local-only training bridge security assumptions.
- Minimum acceptable steps/second and maximum acceptable Luau inference time.
