# M3 status — fixed-course PPO baseline

M3 is in progress.

## Implemented

- Fixed-course Gymnasium wrapper that forces course seed `0` on every episode reset.
- Python-side 400-step episode time limit, corresponding to 20 seconds at 20 decisions/second.
- Stable-Baselines3 PPO entry point with configurable rollout length, batch size, network architecture, checkpoint interval, and master seed.
- Per-run resolved configuration, Monitor CSV output, periodic checkpoints, and final model serialization.
- Unit tests for fixed-seed resets and episode truncation.
- Edge-triggered jump mapping: activation above `0.75`, release required before retriggering, and an eight-decision cooldown.
- Reduced initial PPO exploration standard deviation (`log_std_init = -0.5`) to avoid untrained action thrashing.
- Deterministic evaluator with separate fixed-course and held-out-seed metrics for completion, return, episode length, hazards, and checkpoint progress.

## Default baseline

- Policy: `MlpPolicy`, two hidden layers of 128 units.
- Budget: 100,000 environment steps.
- PPO rollout: 256 steps; batch size 64; 10 epochs.
- Training course: procedural generator seed `0` for every episode.
- Evaluation: 20 deterministic episodes on seed `0`, then the untouched validation partition beginning at seed `1,000,000`.

## Live smoke gate

The second 128-step smoke invocation completed successfully against course seed `0`:

- Two 64-step rollout iterations at 7 environment steps/second.
- Ten PPO optimization epochs after the first rollout.
- Approximate KL `0.0084`, clip fraction `0.0375`, and entropy loss `-5.67`.
- Final artifact saved as `runs/m3-fixed-smoke-2/final_model.zip` (504,162 bytes).
- Reloaded model produced a finite four-component deterministic action for a test observation.

The smoke budget is intentionally too small to assess policy quality. The negative explained variance (`-8.28`) is not treated as a learning result.

A second 128-step smoke with the corrected action mapping also completed at 7 steps/second. The measured policy standard deviation was `0.606` and entropy loss was `-3.68`, down from approximately `0.999` and `-5.67` in the original twitchy smoke run. The model was saved under `runs/m3-controlled-actions-smoke/`.

The evaluation smoke ran one fixed-seed episode and two held-out seeds for 64 steps each. As expected for an untrained 128-step model, completion and checkpoint progress were zero and mean return was `-0.064`; the purpose of this run was to validate deterministic model loading and metric collection. Results are saved in `runs/m3-controlled-actions-smoke/evaluation-smoke.json`.

## Fixed-course pilot finding

A 2,048-step pilot completed eight PPO rollouts and five full 400-step episodes. Every episode returned exactly `-0.4`, the accumulated time penalty, with no checkpoint progress. This is a valid negative result: the original global-progress reward was too sparse for the calmer exploration policy, so the 100,000-step run was not started.

The bridge now uses potential-based checkpoint-distance shaping and reports each component in `info.reward_components`: checkpoint distance, global progress, checkpoint bonus, finish reward, hazard penalty, and time penalty. A second pilot is required before committing the full budget.

The shaped 2,048-step pilot completed with five episode returns from `-0.3771` to `-0.3509`; deterministic seed-0 evaluation returned `-0.2057` but still reached checkpoint index 0. This confirms meaningful approach progress without successful obstacle traversal.

Freezing yaw produced straight course-relative movement, but a 512-step comparison returned `-5.13`: the avatar reliably reached the first obstacle and incurred repeated hazard penalties instead of wandering near spawn. The remaining learning problem is jump timing, not basic forward locomotion. Episode Monitor logs now include accumulated reward components, hazard count, and maximum checkpoint index so subsequent pilots can measure that directly.

## Curriculum revision

M3 now supports four explicit course stages over the same observation, reward, checkpoint, and transport contracts:

1. One wide, continuous flat traversal for forward/strafe control.
2. One fixed 3.5-stud gap for approach momentum and jump timing.
3. One isolated randomized jump with varying gap, lateral offset, and landing height from -3 to +3 studs.
4. The full eight-stage procedural course.

The trainer and evaluator accept `--curriculum-stage 1..4`. Each resolved run config and provenance record the selected stage. Promotion thresholds will be based on deterministic completion and hazard rates rather than a fixed number of training steps.

The first stage-1 pilot exposed two environment issues rather than a PPO result. A preceding scripted validator left a persistent `Humanoid:Move` command, causing four one-step completions after position reset. Subsequent episodes fell three times each because the original 12-stud-wide runway punished normal lateral exploration. Reset/recovery now explicitly clears Humanoid movement, and stage 1 uses a 40-stud-wide runway. The corrected course passes the Studio curriculum smoke test and requires a clean rerun.

A subsequent clean attempt showed that 40 studs still permitted correlated strafe exploration to leave the runway. Before the play session paused, its second full episode improved to return `-0.148` with zero hazards, but the run correctly ended with `failed` status after a transport timeout. Stage 1 is now 200 studs wide so lateral boundaries do not interfere with the locomotion lesson.

The 200-stud pilot completed normally. Its first episode had one hazard; the second had zero hazards, positive checkpoint-distance reward `+0.216`, and return `-0.181`. The remaining fall was consistent with backward exploration off the spawn platform, so stage 1 now includes a 100-stud safety apron behind spawn without changing the forward target distance. The updated curriculum passes the Studio smoke suite.

The final stage-1 pilot learned successfully in 2,048 steps. After two exploratory failures, its last seven training episodes all finished. Deterministic evaluation completed 3/3 fixed episodes and 2/2 held-out-seed episodes, with zero hazards, checkpoint index 1, and mean lengths of 44.7 and 47.5 steps respectively. Stage 1 is promoted. The trainer supports `--init-model` so stage 2 can inherit this locomotion policy.

## Dynamics synchronization audit

Curriculum progression is paused for an environment audit after visual movement remained jerky. Two concrete risks were found and corrected:

- Roblox's default client `PlayerModule` controls were still active and could issue `Humanoid:Move` calls that competed with the PPO server controller. A `StarterPlayerScripts` client script now disables them and reports its state through bridge telemetry.
- The action was observed after a fixed three-heartbeat hold, but remained active during the slower HTTP exchange. The bridge now stops Humanoid movement immediately after each observation, so no commanded motion occurs between the recorded next state and the next action.
- `Humanoid:Move` was initially issued only once at the start of that hold. Continuous forward/strafe input is now refreshed on every physics heartbeat, while yaw and jump remain one-shot controls. Reset and hazard recovery also wait for a grounded, nearly stationary character before returning an observation.

Each transition now reports the requested action, actual hold time, before/after checkpoint distance, before/after position, post-action velocity, reward components, AutoRotate state, and default-control state. `scripts/audit_live_dynamics.py` verifies reset velocity, previous-action alignment, fixed hold duration, controller ownership, and reward directionality before training resumes.

The post-fix 36-transition audit passed. Compared with the single-issue movement command, cumulative forward distance over the scripted block improved from `1.136` to `1.961` studs (about 73%). Forward reward was `+0.3059`, backward reward was `-0.3196`, idle reward was exactly the accumulated time penalty (`-0.012`), and action holds ranged from 50 to 72 ms.

Stage 1 was then retrained from scratch for 2,048 steps under the audited dynamics. After two 400-step exploratory episodes, every subsequent training episode completed, with the final rollout reporting mean episode length 215 and mean return `+1.71`. Deterministic evaluation completed 3/3 fixed-seed and 2/2 held-out-seed episodes with zero hazards; mean episode lengths were 51.3 and 51.5 steps. The promoted checkpoint is `runs/m3-stage1-synced-2048-rerun/final_model.zip`, with full evaluation in the adjacent `evaluation.json`.

## Remaining M3 work

An eight-lane synchronous vector environment is now implemented and live-validated; see `docs/M3_VECTOR_ENV.md`. Its scripted physics smoke sustained 56.3 aggregate transitions/second, and its end-to-end PPO smoke sustained 58 aggregate FPS. Independent lane auto-reset was verified by finishing only lane 1 while lanes 2 through 8 remained intact.

The promoted Stage 1 policy was subsequently fine-tuned for 8,192 aggregate transitions across eight lanes. Mean stochastic episode length improved from 94.8 to 80.2 steps and policy standard deviation decreased from about `0.594` to `0.572`. Deterministic evaluation remained perfect (3/3 fixed and 2/2 held-out, zero hazards), but fixed-course traversal stayed essentially unchanged at 51.7 steps. This indicates additional training on the current objective improves exploration behavior but does not materially smooth the deterministic controller. Evaluation now reports action variation and direction reversals so future smoothing changes have an explicit metric.

An action-change penalty experiment then fine-tuned for another 8,192 transitions with weight `0.005`. Stochastic mean length improved to 56.4 and policy standard deviation fell to `0.534`. Deterministic evaluation remained 5/5 with zero hazards and zero direction reversals, but mean action variation increased rather than decreased: fixed `0.0347 -> 0.0377`, held-out `0.0367 -> 0.0426`. The coefficient is therefore not promoted as a deterministic smoothing improvement. It is useful if reduced stochastic training noise is desired, but a stronger coefficient or a different action parameterization must pass the explicit action-variation gate before replacing the prior checkpoint.

The remaining visible tapping was traced to the actuator rather than PPO: the bridge issued `Humanoid:Move(Vector3.zero)` after every observation while waiting for the next Python action. The bridge now implements a zero-order hold. A heartbeat controller continuously refreshes the latest forward/strafe command through the Python/HTTP interval and clears it only on reset, termination, or recovery. The timing audit measured a mean action replacement interval of 132 ms while retaining a minimum 50 ms observation window. Over the same 12 scripted decisions, forward displacement increased from `1.961` to `10.482` studs and backward displacement remained correctly signed at `-9.178`.

The pre-change deterministic policy still completed both visible Stage 1 evaluations under the new actuator in 18 decisions, down from roughly 48-53, with zero hazards, zero direction reversals, and mean action variation around `0.021`. This validates continuous held movement, but the environment dynamics have materially changed; future promoted checkpoints must be retrained and evaluated under zero-order hold rather than relying on policies optimized for the old stop/start actuator.

The first from-scratch zero-order-hold training attempt exposed an actuator lease requirement. PPO optimization pauses Python action delivery for seconds; an unlimited hold allowed the last exploratory command to continue through that pause, causing invalid 1-6 step episodes and returns as low as `-24`. That run was stopped and marked failed. Held actions now expire after 250 ms, comfortably above the measured 132 ms normal replacement interval, so ordinary inference remains continuous while PPO updates automatically release movement.

Stage 1 was then retrained from scratch for 8,192 aggregate transitions across eight lanes with the 250 ms lease and no smoothing penalty. Final stochastic mean episode length was 41 with return `+2.22`. Deterministic evaluation completed 3/3 fixed and 2/2 held-out episodes with zero hazards and zero direction reversals; mean lengths were 18.7 and 19.0 decisions, and mean action variation was about `0.038`. The promoted continuous-hold Stage 1 checkpoint is `runs/m3-vector-stage1-continuous-hold-lease250-8192/final_model.zip`.

1. Save complete provenance and explicit run state (`running`, `complete`, or `failed`) with each experiment.
2. Add periodic evaluation during long training, using a separately scheduled Studio pass so it does not perturb rollouts.
3. Run the declared 100,000-step fixed-course budget and evaluate the frozen checkpoint.
