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
2. One fixed 7-stud gap that requires a real jump rather than momentum-assisted walking.
3. One isolated randomized jump with varying gap, lateral offset, and landing height from -3 to +3 studs.
4. The full eight-stage procedural course.

The trainer and evaluator accept curriculum stages `1..16`. Stage 4 remains the legacy full course for saved-run compatibility; stages 3 and 5-16 are progressive single-jump geometry tiers. Each resolved run config and provenance record the selected stage. Promotion thresholds are based on deterministic completion and hazard rates rather than a fixed number of training steps.

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

The initial Stage 2 experiment inherited that checkpoint and trained for 8,192 aggregate transitions across eight fixed 3.5-stud-gap lanes. It completed deterministic evaluation without hazards, but visual inspection showed the avatar could cross through momentum-assisted ground contact without jumping. That checkpoint is retained as a traversal result but is not a valid jump-timing promotion. Stage 2 now uses a fixed 7-stud gap, inside the validated jump envelope but large enough to require an actual jump.

The 7-stud revision passed two scripted controllability gates. Holding forward without jump for 60 decisions produced five hazards and zero completions. Sweeping one jump pulse across approach decisions 0 through 13 showed that decisions 3, 4, and 5 complete in 13 steps while every earlier or later pulse falls. This proves the course both requires jumping and has a reachable timing window.

Several rejected pilots exposed jump-curriculum issues: the original `0.75` edge threshold made jump exploration too rare; held-high signals could not retry after release; repeated recoveries buried the success signal; spawn jumps consumed the cooldown before takeoff; and stochastic movement disturbed the narrow approach. The Stage 2 curriculum now exposes jump readiness in observation slot 5, uses cooldown-gated level triggering, terminates a lane on its first hazard, masks jump execution to the validated 12-18 stud takeoff window during training, adds a `+0.2` grounded takeoff reward, and temporarily scripts forward locomotion while PPO learns the jump dimension.

The resulting isolated-jump run trained for 8,192 transitions and initially passed a five-episode evaluation. Repeated visible trials and a later 20-attempt evaluation showed that this was a small-sample false promotion: the policy frequently jumped too early or late and fell. `runs/m3-vector-stage2-isolated-jump-8192/final_model.zip` is therefore retained only as an experimental artifact, not a promoted checkpoint.

## Stage 2 DAgger and conservative PPO

A lightweight DAgger pipeline now trains only the jump row of the continuous policy head, preserving the promoted Stage 1 movement outputs. It aggregates labels on both oracle-visited and learner-visited observations, progressively reduces oracle execution probability from 1 to 0, and saves each aggregate dataset and iteration checkpoint. The first DAgger attempt failed because its 14-20 stud oracle window had been inferred from single-agent cadence and did not transfer exactly to the eight-lane vector cadence.

The vector timing sweep measured successful takeoffs at 13.60, 15.42, and 17.26 studs. A pulse at 19.14 studs was too early and one at 11.36 studs was too late, so the calibrated oracle window is 13.5-17.5 studs. With that correction, the learner-only fourth DAgger rollout completed 24/32 episodes cleanly. Independent deterministic evaluation of the final cloned model completed 51/64 cleanly (79.7%).

The DAgger model was then fine-tuned with PPO at learning rate `2e-5`, scripted forward movement, termination on the first hazard, and timing rewards using the same calibrated window. Checkpoint evaluation was essential: performance peaked before the end of training and subsequently regressed. The 3,072-step checkpoint completed 24/24 in screening and 62/64 in the confirmation run (96.9%, two hazards). The final 4,096-step model completed only 16/24 in screening and must not be used.

The promoted Stage 2 checkpoint is `runs/m3-stage2-dagger-v2-ppo-conservative/checkpoints/ppo_vector_3072_steps.zip`. Evaluation/deployment uses deterministic actions, jump threshold `0.0`, cooldown `8`, yaw disabled, and scripted forward movement for this isolated jump curriculum. The next promotion gate is to remove scripted forward movement while preserving robust jump completion.

The first attempt to remove scripted forward movement exposed a distribution mismatch. The promoted scripted-movement checkpoint completed only 7/64 with policy movement enabled, and the pre-PPO DAgger model completed 6/64. An 8,192-step low-learning-rate joint PPO run did not improve this: screened checkpoints remained between 8.3% and 12.5% clean. That run is rejected.

Movement-aware DAgger now labels the full oracle action `[strafe=0, forward=1, yaw=0, jump]` on learner-visited observations and fits all four output rows while leaving the shared Stage 1 representation frozen. Its learner-only fourth rollout completed 31/32 episodes cleanly with policy movement enabled. Independent deterministic evaluation without scripted movement completed 56/64 cleanly (87.5%), with eight hazards and mean episode length 11.03 decisions. The promoted joint-control checkpoint is `runs/m3-stage2-dagger-v3-joint-actions/final_model.zip`. The scripted-forward checkpoint remains the higher-success reference at 96.9%, so further joint-policy improvement should target the remaining 9.4 percentage-point gap before Stage 3.

## Progressive geometry randomization

Generator `0.5.0` adds explicit single-jump variation tiers while preserving stage 4 as the legacy full obby for saved-run compatibility:

| Order | Stage | Geometry |
| --- | ---: | --- |
| 1 | 3 | Gap 6.5-7.5 studs, level landing |
| 2 | 5 | Gap 6-8.5 studs, level landing |
| 3 | 13 | Gap 6-9 studs, level landing |
| 4 | 14 | Gap 5-8.5 studs, level landing |
| 5 | 15 | Overlapping replay of 5-8.5 and 6-9 stud ranges |
| 6 | 12 | Gap 5-9 studs, level landing |
| 7 | 6 | Gap 5-9 studs, level landing |
| 8 | 7 | High-to-low landing, -0.5 to -3 studs |
| 9 | 8 | Low-to-high landing, +0.5 to +3 studs |
| 10 | 9 | Approach angle up to 8 degrees |
| 11 | 10 | Approach angle up to 18 degrees |
| 12 | 11 | Gap 5-9, height -3 to +3, and angle up to 18 degrees together |
| Final | 4 | Full eight-segment procedural obby |

Stages 16 and 17 retain the 9-10 stud boundary only as optional stress
benchmarks. They are not promotion gates and are not sampled by the core
curriculum.

Gap, height, and approach angle are sampled deterministically from the course seed. Angles are converted to lateral landing offsets using the actual platform-center distance rather than treated as arbitrary strafe offsets. Evaluation and DAgger assign distinct deterministic course seeds to lanes and reset episodes, preventing a randomized tier from silently training or evaluating only seed 0.

A tier advances only after deterministic evaluation over 64 development seeds and 64 untouched validation seeds reaches at least 90% combined clean completion, neither partition falls below 85%, and visual inspection confirms the policy is jumping rather than exploiting contact or recovery behavior. Failed tiers retain their checkpoint and dataset but do not expand the geometry range.

Stage 3 initially exposed seed overfitting: the Stage 2 joint policy completed 63/64 development courses but only 50/64 held-out courses. Movement-aware DAgger over distinct Stage 3 seeds produced a learner-only 64/64 rollout. Its frozen checkpoint then completed 64/64 development and 56/64 held-out courses, or 93.8% combined with both partitions above the floor. `runs/m3-stage3-narrow-gap-dagger-v1/final_model.zip` is promoted for Stage 3. The same checkpoint generalized directly to Stage 5 at 63/64 development and 64/64 held-out, so no redundant Stage 5 training was performed.

Stage 6's 5-10 stud range exposed an observation/teacher limitation. The Stage 3 checkpoint completed only 46/64 development courses, and a first DAgger attempt reached only 42/64 learner-only because its fixed checkpoint-distance oracle itself completed just 39/64. The takeoff distance must shift with gap length, but the original feed-forward observation duplicated checkpoint-relative and finish-relative vectors on a one-segment course and did not identify the sampled gap.

The structured observation retains its 22-value shape for checkpoint compatibility but now uses route-feature slots 9-11 for normalized gap length, landing height, and approach angle on jump segments. Non-jump segments retain finish-relative values there. The DAgger oracle shifts the calibrated takeoff window with the observed gap (`gap + 6.5` through `gap + 10.5` studs). Stage 6 training must restart under this geometry-aware observation; the failed `m3-stage6-wide-gap-dagger-v1` artifact is not promotable.

Geometry-aware DAgger and weighted cloning did not promote either Stage 6 or Stage 12 because the scripted oracle remained weaker than the inherited policy. Varied-seed PPO on Stage 12 improved the best 64-course development score from 75.0% to 82.8% at 1,024 steps, but missed the 85% partition floor and regressed with further training. Stage 13 (6-9 studs) was inserted so the upper boundary can be learned before lowering the minimum gap; no Stage 12 or Stage 6 artifact is promoted.

Stage 13 varied-seed PPO peaked at 2,048 steps. The frozen checkpoint completed 59/64 development and 62/64 held-out courses, or 94.5% combined, so `runs/m3-stage13-gap6-9-ppo-v1/checkpoints/ppo_vector_2048_steps.zip` is promoted. PPO training now has an explicit `vary_course_seeds` mode that advances a deterministic seed counter independently whenever a vector lane resets; an earlier seed-0-only Stage 12 attempt was stopped and marked failed.

The promoted Stage 13 checkpoint still completed only 48/64 when the lower Stage 12 boundary moved from 6 to 5 studs. A 4,096-step varied-seed fine-tune produced misleading 23/24 checkpoint screens: full 64-course confirmation fell to 37/64 at 2,048 steps and 48/64 at 4,096 steps. This run is rejected and demonstrates that 24-course screens are useful only for ranking candidates, never promotion. The next curriculum change should isolate the lower boundary at 5-8.5 studs before recombining it with the learned 9-stud upper boundary.

Stage 14 isolated that lower boundary successfully. Its 2,048-step checkpoint completed 62/64 development and 64/64 held-out courses (98.4% combined) and is promoted as `runs/m3-stage14-gap5-8p5-ppo-v1/checkpoints/ppo_vector_2048_steps.zip`. Direct evaluation on uniform Stage 12 remained 48/64, indicating interference between the separately learned upper and lower boundary behaviors. Stage 15 therefore samples from the overlapping 5-8.5 and 6-9 ranges to rehearse both mastered tasks before returning to uniform 5-9 sampling.

Independent lane rebuilding was subsequently identified as a source of invalid transition timing and large evaluation variance. Promotion evaluation now runs synchronous eight-lane cohorts and never rebuilds a course while another lane is active. The PPO vector adapter also supports `reset_all_on_any_done`, which truncates and resets the entire cohort together for transition-correct training. Under the corrected gate, the Stage 15 4,096-step checkpoint completed 55/64 development and 64/64 held-out Stage 12 courses (93.0% combined), then generalized to Stage 6 at 61/64 development and 62/64 held-out (96.1% combined). The same checkpoint completed both high-to-low and low-to-high height stages at 128/128 each.

Small-angle failures were asymmetric: the inherited policy achieved only 52.9% for -8 to -4 degrees. Synchronous DAgger with an oracle strafe proportional to local checkpoint offset corrected this; `runs/m3-stage9-small-angle-dagger-steering-v1/final_model.zip` completed 63/64 development and 64/64 held-out Stage 9 courses (99.2% combined). It generalized directly to Stage 10's ±18-degree range at 60/64 development and 62/64 held-out (95.3% combined).

On Stage 11's fully combined geometry, the steering checkpoint completed 50/64. Steering-aware DAgger improved the frozen model to 53/64, but synchronous PPO confirmation remained 51/64. Performance below 8 studs was 90-100%; the 9-10 stud bin was only 33.3%. Stage 16 isolates 9-10 stud jumps with height and angle randomized together before returning to the full combined distribution. No Stage 11 checkpoint is promoted yet.

The synchronous PPO adapter now supports deterministic weighted curriculum replay at the cohort level. A Stage 16 run sampled 55% current-task cohorts and 45% rehearsal across full-gap, height, and angle stages. Rollout returns remained positive across the 8,192-step run, demonstrating that replay and synchronized resets work operationally, but deterministic Stage 16 checkpoints remained between 37.5% and 45.8%. The run is not promoted.

A direct scripted-expert audit then completed only 19/64 Stage 16 courses. The current DAgger teacher is therefore not an authoritative expert for simultaneous 9-10 stud distance, ±3-stud height, and ±18-degree angle. Stage 16 is paused pending a privileged geometry/physics teacher or empirical feasibility filtering. More PPO or behavior cloning against the current teacher is not justified. Weighted replay remains required for future training, and promotion must include regression evaluation over the earlier gap, height, and angle skill families.

1. Save complete provenance and explicit run state (`running`, `complete`, or `failed`) with each experiment.
2. Add periodic evaluation during long training, using a separately scheduled Studio pass so it does not perturb rollouts.
3. Run the declared 100,000-step fixed-course budget and evaluate the frozen checkpoint.
# Privileged teacher track

The Stage 16 scripted expert is not authoritative enough for further DAgger
(19/64 in the latest audit). Work has therefore moved to a privileged PPO
teacher followed by limited-sensing student distillation. The separate 48-value
teacher contract and Python trainer mode are implemented; see
`docs/M3_PRIVILEGED_TEACHER.md`. Live transport validation and teacher training
are the next gates.

## Long-gap scope decision

The 9-10 stud boundary is removed from the core curriculum. It consumed
substantial training and calibration effort while remaining sensitive to lane,
reset, and takeoff-frame variation. Core Stages 6 and 11 now cap gaps at 9 studs;
Stages 16 and 17 remain reproducible optional stress tests only.

Height and angle prerequisites are already complete: the promoted Stage 15
checkpoint achieved 128/128 on both high-to-low and low-to-high stages, and the
Stage 9 steering checkpoint achieved 127/128 on small angles and 122/128 on the
full ±18-degree stage. Work therefore resumes at capped combined geometry
(Stage 11), followed by the multi-segment procedural Stage 4 course and M4
domain-randomized training.

The existing Stage 9 steering checkpoint required no additional capped-combined
training. On two disjoint 64-course Stage 11 partitions it completed 60/64 and
62/64 cleanly: 122/128 (95.3%) combined, with both partitions above 90%. It is
therefore promoted for capped combined geometry. Remaining failures concentrate
in the 8-9 stud and steep negative-angle slices, which remain evaluation bins
rather than blockers. The next gate is the full eight-segment Stage 4 course.

The first direct Stage 4 baseline completed only 1/32 courses, usually failing
near the first checkpoint transition. Stage 4 also introduces realistic narrow
beams and stairs that were not covered by the prior jump curriculum. Bridge
Stages 18-22 now isolate narrow beams, isolate stairs, and then chain 2, 4, and 8
mixed segments. Stage 22 matches the final eight-segment structure while Stage 4
remains the stable final benchmark identifier.

Terminal checkpoint instrumentation on a second Stage 20 screen found nine of
12 hazards before checkpoint 1 and three after it. The cause was a student
observation semantic mismatch: route features 9-11 represented gap geometry on
jumps but fell back to the final-course vector on beams and stairs. Thus an
isolated beam and the same first beam in a multi-segment course looked different.
The procedural harness now emits the existing `(-1, 0, 0)` non-jump sentinel
consistently, independent of remaining course length. This is an observation
contract correction and requires a new place build and regression evaluation.

Under the corrected contract, the inherited steering policy retained 31/32
narrow-beam and 32/32 stair completions, but achieved only 31/64 on two-segment
Stage 20. A conservative 4,096-transition PPO continuation used synchronized
cohort resets and replay weights `20:0.60,18:0.15,19:0.15,11:0.10`. Checkpoint
screens on the same 24 seeds scored 10/24 at 1,024 steps, 23/24 at 2,048, and
24/24 at both 3,072 and 4,096. The final checkpoint is
`runs/m4-stage20-two-segment-replay-v1/checkpoints/ppo_vector_4096_steps.zip`.
It is a promotion candidate, not yet promoted: the required 64-course
development and 64-course untouched validation gates plus Stage 11/18/19
regressions remain to be run.

The final Stage 20 checkpoint subsequently passed its full promotion gate:
60/64 development and 62/64 untouched validation courses, or 122/128 (95.3%)
combined. Regression remained strong on capped combined jumps (62/64), narrow
beams (32/32), and stairs (31/32). The promoted two-segment checkpoint is
`runs/m4-stage20-two-segment-replay-v1/checkpoints/ppo_vector_4096_steps.zip`.
Work advances to Stage 21's four-segment mixed courses.

The promoted Stage 20 policy began Stage 21 at 27/32 (84.4%), with every course
reaching at least checkpoint 2. An eight-lane 4,096-transition continuation was
rejected: checkpoint screens fell from 20/24 at 1,024 steps to 12/24 at 2,048
and 7/24 at 4,096. With `reset_all_on_any_done`, one failure truncates seven
other four-segment trajectories, leaving rollout mean lengths near 10-12 despite
successful deterministic episodes taking about 44 decisions. The next diagnostic
uses four synchronized lanes to reduce cross-lane interruption without returning
to invalid independent course rebuilding.

### Required-jump correction

Visual review found that Stage 20 overstates jump ability: it inherits the base
3.5-7 stud gap range, and half of its uniformly sampled segment kinds are beams
or stairs. Short gap samples can also be crossed using running momentum. The
mixed completion gate therefore does not measure success conditional on a jump
actually being required.

Stage 23 is a two-transition required-jump correction. Every segment samples a
6-8 stud gap, -3 to +3 studs of landing height, and -18 to +18 degrees of
approach angle. It uses the `offset` generator for both transitions so flat and
near-zero-angle cases remain possible while beam/stair samples cannot dilute the
new-task distribution. Training continues from the promoted Stage 20 checkpoint
with replay weights `23:0.65,11:0.10,7:0.04,8:0.04,3:0.05,18:0.05,19:0.05,20:0.02`.
Promotion must report Stage 23 separately from Stage 18/19 regressions; mixed
course completion alone is not evidence of required-jump competence.

The first 8,192-transition continuation exposed and corrected a separate
evaluation error: `evaluate_stage2_vector.py` had hard-coded a zero jump
threshold even though training and deployment use `0.75`. Under the corrected
deployment threshold, the inherited Stage 20 policy completed 8/24 Stage 23
development courses. The best new checkpoint was the 4,096-transition snapshot
at 19/24; later snapshots regressed, with the 8,192 snapshot reaching only
10/24. On a separate 32-course partition the 4,096 snapshot completed 20/32.
It retained beams at 32/32 and stairs at 31/32, but the older Stage 11 combined
jump regression was only 24/32. A second conservative continuation with a
`5e-6` learning rate and 20% Stage 11 replay did not exceed the first run's
4,096 snapshot. The current candidate is therefore
`runs/m4-stage23-required-jumps-replay-v1/checkpoints/ppo_vector_4096_steps.zip`,
but it is not promoted: required-jump generalization and Stage 11 retention are
both below gate.

A checkpoint-compatible controller correction then interprets the fourth policy
output as jump intent, lowers its threshold from `0.75` to `0.65`, and executes
it only while grounded inside the existing geometry-aware takeoff window. This
uses the unchanged 4,096-transition network. It completed 20/24 development and
29/32 held-out Stage 23 courses, compared with 19/24 and 20/32 without the gate.
The held-out gap bins were 17/19 for 6-7 studs and 12/13 for 7-8 studs. Stage 11
also improved from 24/32 to 27/32 under this mapping. A PPO continuation trained
with the same gate and timing rewards did not exceed the unchanged network, so
its weights are rejected. The selected model and controller settings are pinned
in `runs/m4-stage23-required-jumps-replay-v1/deployment.json`.

Visual completion review then exposed a checkpoint-validity bug: entering the
checkpoint's ±6-stud volume while airborne counted as advancement and could
terminate a course without landing. Checkpoint advancement now requires ground
contact and emits the newly advanced target observation immediately. Metrics
from before this correction are not grounded-landing success rates.

Under the corrected contract, the selected feed-forward model completed 28/32
courses across all eight lanes, but the five seeds used as recording lane 1
completed 0/5. Tracing seed 123008 showed that cooldown reached zero before the
second takeoff window, while deterministic jump intent remained negative for
every second-approach state. Eight-lane PPO, transition-valid single-lane PPO,
frozen-representation DAgger, positive-weighted DAgger, and movement-anchored
representation DAgger did not produce a promotable correction; the best weighted
DAgger screen completed only 1/5 recording seeds, and representation learning
collapsed learner-only movement. These artifacts are rejected. Further ordinary
PPO/DAgger against the same feed-forward representation is paused. The next
model should expose checkpoint phase explicitly or use the already-installed
recurrent PPO stack so post-landing state can be distinguished without a
scripted second-jump override.

### Recurrent policy attempt

An `sb3-contrib` `RecurrentPPO` path now provides separate training and
state-correct deterministic evaluation scripts using `MlpLstmPolicy`. A direct
8,192-transition Stage 23 run from scratch failed at 0/32 because memory does not
remove the need for locomotion curriculum. Restarting from Stage 1 produced a
32/32 recurrent locomotion policy after 4,096 transitions. The policy then
received 4,096 Stage 2 transitions followed by an 8,192-transition lower-rate
continuation with Stage 1 replay. Deterministic single-jump performance remained
4/32; lowering the geometry-gated jump threshold to zero scored 3/32. The
recurrent run is operational but not promotable and must not advance to the
two-jump stage. This isolates the current bottleneck to jump-action learning,
not merely missing temporal memory. The next architecture experiment should use
a hybrid action distribution with categorical jump intent and continuous
movement instead of thresholding a Gaussian action dimension.

The subsequent checkpoint-transition audit found that Studio was still running
an older installed `ObbyRLBridge.rbxm` even after the v7 place was rebuilt. The
place carried the grounded-checkpoint module, but the stale plugin returned the
just-landed target observation for an extra decision. The plugin was rebuilt and
reinstalled. After restarting Studio, seed 123008 switched immediately from the
first checkpoint distance to the second (`7.72` then `23.22`). At the valid
takeoff windows both approaches were grounded with near-zero vertical velocity
and about 16 studs/s forward speed. The main differences were lateral velocity
and geometry: the successful first jump was high-to-low at roughly +17.5
degrees, while the failed second jump was a nearly straight low-to-high landing.
Corrected isolated evaluation confirmed a skill imbalance: Stage 7 high-to-low
scored 28/32, while Stage 8 low-to-high scored only 21/32. The immediate blocker
is therefore low-to-high skill regression plus geometry-conditioned state shift,
not stale targets, cooldown, or temporal memory alone. Earlier Stage 7/8 results
that predate grounded completion must not be treated as valid landing metrics.

The corrected Stage 8 prerequisite now samples 4-5 stud gaps at +0.5 to +3
studs of landing height. Its previous 6.5-7.5 stud range reused the flat jump
envelope despite asking for positive vertical displacement. The longer uphill
range is deferred until the shorter grounded-landing prerequisite is promoted.
Corrected evaluation on this range was 64/64 with the existing feed-forward
policy, so no Stage 8 weight update was needed. Stage 23 now samples its positive-
height segments from the same 4-5 stud range while retaining 6-8 studs for level
and downhill segments.

A recording/evaluation mismatch then exposed vector-lane contamination. The
standalone course created by `Main.server` remained collidable at the origin in
normal vector mode, where it overlapped lane 1's generated course. Recording
mode disabled that geometry and therefore produced worse but more truthful lane
1 behavior. The bridge now disables standalone base-course geometry for every
vector reset, not only recording resets. The apparent 54/64 Stage 23 result and
other vector results measured before this correction require revalidation.

After removing base-course overlap, corrected results were 57/64 for the 4-5
stud low-to-high prerequisite and 42/64 for height-conditioned two-jump Stage
23. A conservative 8,192-transition PPO continuation regressed at every screened
checkpoint (28/64 at 2,048, 33/64 at 4,096, and 32/64 at 8,192) and is rejected.
To expose episode phase without changing checkpoint tensor dimensions, student
observation slot 20 (zero-based)—previous yaw, whose action is disabled—is now
normalized checkpoint phase (`checkpointIndex / checkpointCount`). Course
progress already provides a monotonic position signal; elapsed time remains
excluded to avoid timing shortcuts.

That checkpoint-compatible semantic substitution was rejected immediately: the
unchanged policy fell from 42/64 to 29/64. Even though yaw is not applied to the
avatar, previous yaw was functioning as learned action history. The source was
reverted. Checkpoint phase must be appended under a new observation schema with
explicit model migration or retraining; it cannot safely replace an existing
feature.

Neither reducing the synchronized cohort to four lanes nor adding an all-lanes
completion barrier produced a promotable model. Four-lane screening peaked at
19/24. The barrier correctly restored rollout lengths from roughly 10-14 to
51-58 decisions, but waiting lanes necessarily stored policy actions that the
environment overrode with zero, making those transitions off-policy; its best
24-course screen was 21/24 but full confirmation was only 53/64. A transition-
correct single-lane run also regressed (16/24 at its early screen) and exposed a
trainer bug: loaded PPO continuations retained the checkpoint's old `n_steps`
despite the CLI override. Continuation loading now injects learning rate,
`n_steps`, and batch size during model reconstruction so rollout-buffer settings
match the resolved run configuration. No Stage 21 training artifact is promoted;
the promoted Stage 20 model remains the best Stage 21 baseline at 27/32.

A Python-side checkpoint-credit experiment added configurable reward for actual
checkpoint-index advancement. With `+0.4` extra credit per checkpoint, a
transition-correct single-lane rollout produced stable shaped returns from
`+1.84` to `+1.96`, demonstrating that intermediate success is now visible.
Nevertheless, the first PPO-updated checkpoint fell to 14/24 deterministic
completion. The run is rejected. Low aggregate KL is not sufficient protection
because small mean-action changes can cross jump/steering thresholds and alter
the deterministic controller sharply.

Stage 21's inherited 27/32 completion corresponds to approximately 95% success
per attempted obstacle; four compounded attempts predict roughly 81-84% whole-
course completion. A 90% four-obstacle gate requires about 97.4% reliability per
obstacle. Future Stage 21 optimization should preserve the promoted actor with
an explicit behavior/KL anchor or warm up the new long-horizon critic before
allowing actor updates. Ordinary PPO continuation is paused. The checkpoint-
credit mechanism remains implemented and tested but is not part of evaluation
reward and has not produced a promoted model.

Critic-only warm-up preserved all actor tensors exactly and increased shaped
rollout return, but subsequent actor updates remained extremely threshold-
sensitive: even measured KL around `1e-8` reduced a paired 32-course screen from
30/32 for the unchanged actor to 21/32. All actor-updated artifacts are rejected.
The unchanged Stage 20 policy then scored 51/64 on a full Stage 21 partition,
below promotion. Instrumentation found one reported clean completion at
checkpoint 3/4 because termination used projected route progress rather than the
checkpoint chain. Studio termination now requires advancing through every
checkpoint. Multi-segment evaluations made before this correction may overcount
rare route-projection shortcuts and must not be used alone for promotion.

After reopening the corrected place, the unchanged Stage 20 policy completed
25/32 Stage 21 courses. Critic-only PPO warm-up increased shaped return while
preserving all seven actor tensors bit-exactly. One-epoch actor continuations at
`1e-6` and target KL `1e-4` avoided immediate collapse on one screen (27/32),
but extended continuation fell to 21/32; a paired unchanged-policy evaluation on
the identical seeds scored 30/32. Even actor KL around `1e-8` is therefore too
large a reliability guarantee for this deterministic threshold controller.

Mixed-course DAgger was corrected so its jump oracle never labels the non-jump
`(-1, 0, 0)` sentinel. One expert partition achieved 31/32, but a second achieved
only 25/32, so the scripted expert is not consistently authoritative. Direct
low-rate cloning scored 24/32. Adding a 10x behavior-anchor loss against the
promoted policy still scored only 21/32. Both DAgger artifacts are rejected. The
promoted Stage 20 checkpoint remains the active agent; further Stage 21 training
requires either a demonstrably stronger expert or a continuous/margin-aware
action formulation that is less sensitive to tiny output changes.

The restored, uncontaminated Stage 8 low-to-high prerequisite was revalidated
on 64 fixed seeds at 57/64 (89.1%). A conservative 4,096-transition continuation
used 70% short-uphill samples plus Stage 7, Stage 3, and Stage 11 replay. All four
checkpoints tied at 29/32 on the initial screen, but full evaluation exposed
regression: the 1,024-step checkpoint scored 56/64 and the 4,096-step checkpoint
scored 52/64. The entire continuation is rejected; the original promoted model
remains selected. Further work on the seven Stage 8 failures should use targeted
failure-state aggregation with a validated teacher rather than another ordinary
PPO continuation.

The Stage 8 takeoff-window teacher was then audited by forcing jump intent while
retaining learned movement and geometry gating. It completed 64/64 fixed-seed
courses with zero hazards, so its timing labels are authoritative on this
partition. A learner-state DAgger pass collected 564 observations and updated
only the jump output row at `1e-5`; movement outputs and the shared actor stayed
frozen. The candidate scored 58/64 on the development partition, 57/64 held out,
and tied the baseline on Stage 7 and Stage 3 replay. It improved one Stage 11
mixed-angle screen from 25/32 to 29/32. However, the required paired Stage 23
gate regressed from 40/64 for the baseline to 35/64 for the candidate, with more
failures after checkpoint one. The candidate is rejected. This confirms that
isolated-jump timing labels alone do not cover the post-landing state
distribution of the second approach.

A second DAgger experiment filtered collection to checkpoint index >= 1 and
trained on 401 post-landing observations only. Despite that data filter, the
global jump output row also moved on first-approach states; Stage 23 collapsed to
24/64 with 14 failures before checkpoint one, so the artifact is rejected.
Threshold calibration on the unchanged promoted weights was more effective.
With takeoff-window masking still mandatory, lowering jump intent threshold from
0.65 to 0.0 scored 43/64 on Stage 23, 58/64 on Stage 8, 28/32 on Stages 7 and 3,
and 28/32 on Stage 11. The deployment manifest now selects threshold 0.0 with
the original model weights. This is controller calibration, not a new policy
promotion; removing the geometry gate would invalidate these results.

Stage 23 training now supports mixed initial-state sampling with
`--post-landing-reset-probability`. At `0.5`, each eight-lane Stage 23 cohort has
exactly four normal course starts and four checkpoint-one starts. Post-landing
resets retarget checkpoint two immediately, settle the rig on the intermediate
platform, then apply deterministic seed-conditioned residual motion spanning
12-16 studs/s forward and -1.5 to +1.5 studs/s lateral. This directly trains the
second-approach distribution while retaining first-jump replay in the same PPO
rollout. The new protocol is built into
`ObbyRL-M4-Post-Landing-Mix-v5.rbxlx`; live validation and training remain pending
a Studio restart with the rebuilt plugin.

Live validation confirmed the 4+4 reset contract: normal lanes reported
checkpoint 0 with zero velocity, while post-landing lanes reported checkpoint 1,
were grounded, targeted checkpoint 2, and retained seed-conditioned horizontal
velocity. A 4,096-transition PPO continuation at `1e-6` used this 50/50 initial-
state distribution. Its best checkpoint was 3,072 steps. Post-landing-only
evaluation improved from 49/64 for the baseline to 53/64, demonstrating that the
targeted second-approach skill did improve. However, normal-start two-jump
completion was only 40/64 versus the deployed 43/64, so the checkpoint is not
promoted. The next experiment should reduce the post-landing fraction or add an
explicit behavior anchor on normal-start states to retain first-jump reliability.

The synthetic post-landing reset is superseded by real landing-state replay in
`ObbyRL-M4-Landing-Replay-v6.rbxlx`. On grounded checkpoint-one advancement, the
Studio bridge now stores the course seed, exact root transform, linear and
angular velocity, previous action, and Python-reported jump cooldown. A later
post-landing reset rebuilds that same seeded course, settles the rig, and restores
the captured state. If no natural snapshot exists for that lane, the request
falls back to a normal start instead of fabricating a landing. Student observation
slot 12 now contains segment-relative rather than absolute course progress, so it
no longer directly identifies first versus second approach. This is an
observation-semantics change: old checkpoints remain tensor-compatible but must
not be considered behavior-compatible without migration or retraining. Live
snapshot capture/replay validation is pending a Studio restart with v6.

The first live v6 replay validated seed, checkpoint, cooldown, previous-action,
and segment-progress restoration for all eight lanes. It also exposed impact-
frame captures with vertical velocity as large as -32 and +13 studs/s despite a
grounded flag. Snapshot eligibility now additionally requires absolute vertical
speed below 3 studs/s, preserving horizontal landing momentum while excluding
unstable impact transients. This refinement is built as
`ObbyRL-M4-Landing-Replay-v7.rbxlx` and requires one more Studio restart before
training.

The v7 stability filter was itself rejected after live validation: one lane
advanced into the next jump without ever producing a grounded checkpoint-one
state below the 3 studs/s vertical threshold, so its replay request fell back to
checkpoint zero. Impact-frame velocity is part of the actual observation handed
to the policy immediately after checkpoint advancement; filtering it would again
create a synthetic distribution and omit hard transitions. V8 therefore captures
the exact first decision-boundary observation after checkpoint one, including
its authentic vertical velocity. `ObbyRL-M4-Landing-Replay-v8.rbxlx` supersedes
v7.

V8 live validation restored all eight exact decision-boundary snapshots with
their original seed, checkpoint, velocity, previous action, and cooldown. To
migrate the existing policy to segment-relative progress without relearning all
other features, actor and critic first-layer input column 12 were zeroed. This
no-update migrated policy scored 45/64 on normal-start Stage 23, improving the
previous deployment's 43/64, while retaining Stage 8 at 58/64, Stages 7 and 3 at
28/32, and Stage 11 at 28/32. It is the new v8-compatible deployment baseline.

A 4,096-transition PPO continuation then used 50% real landing-state replay at
`1e-6`. The 1,024-step checkpoint led the initial screen at 26/32 but confirmed
at only 44/64; later checkpoints were worse. All PPO-updated artifacts are
rejected. The result supports the user's diagnosis: removing the phase-identifying
absolute-progress feature helped more than training separate post-landing
behavior. Further optimization should anchor the migrated actor or change the
non-smooth thresholded action formulation rather than continue ordinary PPO.

Landing-state replay is generalized in v9 from Stage 23's checkpoint-one jump
state to every nonterminal checkpoint in Stages 20-23. Stage 20, 21, and 22
provide two-, four-, and eight-segment procedural courses sampling gaps, offset
jumps, beams, and stairs. Each lane maintains a checkpoint-indexed snapshot
reservoir; reset selection chooses an available state deterministically from the
requested seed, rebuilds the snapshot's original course seed, and restores its
physics/action/cooldown state. This prevents geometry/state mismatches while
allowing replay after any obstacle family. The current build is
`roblox_places/training/ObbyRL-Mixed-Landing-Replay-v9.rbxlx`.

Without additional training, the migrated Stage 20 two-segment policy was tested
zero-shot on longer mixed courses under v9. It completed 10/16 four-obstacle
Stage 21 courses (62.5%); all failures still reached checkpoint 2 or 3. It
completed 9/16 eight-obstacle Stage 22 courses (56.2%); failed runs terminated at
checkpoints 2, 3, 5, or 6. This is meaningful length generalization, but not
one-take recording reliability. Two Stage 21 seeds that completed in batch
evaluation failed when rerun under recording mode, illustrating remaining Roblox
physics variance. The website completion clip should therefore use the vetted
two-obstacle sequence; longer-course footage should be labeled zero-shot attempts
or selected from recorded successful trials.
