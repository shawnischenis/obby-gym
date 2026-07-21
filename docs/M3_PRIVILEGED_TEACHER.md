# M3 Privileged Teacher and Student Distillation

## Decision

Train a fresh PPO teacher with simulator-only geometry and physics, qualify it on
held-out seeds, distill its actions into the existing limited-sensing student,
then fine-tune the student with conservative PPO updates. The teacher is a
training instrument, not the final deployable policy.

The existing `obby-structured-v1` student observation remains exactly 22 values.
Roblox now emits a separate `obby-privileged-v1` payload with 48 values. Python
selects it only when `RobloxObbyBatch(privileged_observations=True)` or the
trainer flag `--privileged-observations` is used.

## Privileged observation contract

Values 0–21 are the unchanged student observation. Values 22–47 append:

- root position relative to course start;
- exact checkpoint vector in world coordinates;
- exact current-segment entry vector;
- exact landing-platform dimensions;
- exact gap, height, approach angle, and lateral offset;
- distance from the current segment entry;
- Humanoid walk speed, jump power, jump height, and hip height;
- workspace gravity;
- checkpoint index and segment count;
- entry- and landing-platform presence flags.

All values are normalized and clipped to `[-1, 1]` by Python. The raw payload is
kept separate so privileged data cannot accidentally leak into a student model.

## Experiment sequence

1. Smoke-test the 48-value Studio transport on all eight lanes.
2. Train a fresh privileged PPO teacher, starting with the single-jump stages and
   using synchronous cohort resets plus weighted replay of earlier stages.
3. Require high deterministic success on held-out seeds and each prerequisite
   stage before accepting the teacher.
4. Roll out the teacher while recording `(student observation, teacher action)`.
5. Behavior-clone the 22-input student, then run DAgger on student-visited states
   with the privileged teacher supplying labels.
6. Fine-tune only the student with PPO at a low learning rate and conservative
   clipping. Compare teacher, distilled student, and PPO-fine-tuned student on the
   same held-out seed suite.

## Trainer entry point

Use the normal vector trainer with `--privileged-observations`. A prior 48-input
teacher may be passed through `--init-model` for curriculum transfer. Do not pass
a student checkpoint: Stable-Baselines3 rejects its incompatible 22-input space.

## First live results (2026-07-20)

- Privileged transport smoke: eight lanes, `(8, 48)` batch, approximately 62
  transitions/second.
- Stage 1 teacher (`m3-privileged-teacher-stage1-v1`): 16,384 steps; deterministic
  held-out evaluation 16/16, zero hazards, mean length 20.81.
- Initial Stage 2 transfer without hazard termination was stopped at 6,144 steps
  because repeated recovery produced progressively negative returns.
- Corrected Stage 2 transfer (`m3-privileged-teacher-stage2-v2`) used hazard
  termination, 25% Stage 1 replay, a `1e-4` learning rate, and jump-timing
  shaping. It retained Stage 1 at 16/16 but achieved only 3/64 (4.7%) on held-out
  Stage 2 jumps, so it is rejected as a distillation teacher.

The next experiment should isolate the jump decision with scripted forward
movement. Once deterministic jumping is reliable, restore policy-controlled
movement and fine-tune jointly. This avoids asking PPO to discover takeoff timing
while simultaneously perturbing a newly learned locomotion controller.

That experiment was completed:

- Jump-isolated teacher (`m3-privileged-teacher-stage2-jump-isolated-v1`):
  deterministic Stage 2 success 64/64 with scripted forward movement.
- Joint fine-tune (`m3-privileged-teacher-stage2-joint-v1`): deterministic Stage
  2 success 49/64 (76.6%) with policy-controlled movement; Stage 1 retention
  remained 16/16.

The decomposition works, but the joint teacher remains below the proposed 90%
promotion threshold. Do not begin student distillation from this checkpoint yet.

### Mixed movement replay result

`m3-privileged-teacher-stage2-mixed-movement-v1` continued from the joint
checkpoint with 25% scripted-forward cohorts, 75% policy-movement cohorts, 20%
Stage 1 course replay, and a `3e-5` learning rate. With scripted assistance fully
disabled during evaluation it achieved:

- Stage 2: 64/64 (100%), zero hazards, mean length 13.33;
- Stage 1 retention: 16/16 (100%), zero hazards, mean length 23.19.

This checkpoint clears the 90% promotion threshold and is qualified as the first
teacher for Stage 1–2 student distillation.

## Stage 1–2 student distillation result

The initial behavior-cloning dataset contains 3,070 paired transitions from 64
Stage 1 and 128 Stage 2 teacher episodes. Only the 22-value student observation
was stored with each four-action teacher label. Cloning reduced action-head loss
from 0.297 to 0.0029; the initial student achieved 54/64 (84.4%) on a held-out
Stage 2 partition and retained Stage 1 at 16/16.

Privileged DAgger then aggregated student-visited states over four iterations.
The third checkpoint is promoted rather than the final checkpoint because the
fourth cloning update regressed. `dagger_3.zip` achieved 64/64 on each of two
independent Stage 2 partitions (128/128 combined), zero hazards, and retained
Stage 1 at 16/16. The final checkpoint achieved only 57/64 on the first partition.

Promoted limited-sensing student:
`runs/m3-stage1-2-privileged-dagger-v1/dagger_3.zip`.

### Same-seed teacher comparison and student PPO

On the exact two Stage 2 partitions used to promote the DAgger student, the
privileged teacher achieved 113/128 (88.3%) while the 22-input DAgger student
achieved 128/128. Thus the limited-sensing student exceeded its teacher rather
than merely matching labels on the teacher's state distribution.

An 8,192-transition conservative PPO run then continued from `dagger_3.zip` at a
`2e-5` learning rate with 20% Stage 1 replay. The PPO student retained 128/128
Stage 2 and 16/16 Stage 1 success, while reducing Stage 2 mean completion length
from approximately 14 decisions to 12.25 and 12.48 on the two partitions. Stage
1 mean completion length was 17 decisions.

Promoted Stage 1–2 limited-sensing policy:
`runs/m3-stage1-2-student-ppo-v1/final_model.zip`.

## Gap expansion status

The promoted student generalized without additional training to Stage 3
(6.5–7.5 studs) at 128/128 and Stage 5 (6–8.5 studs) at 126/128. On Stage 6
(5–10 studs), it achieved 55/64; the 9–10 stud bin was only 6/11. The original
privileged teacher achieved 57/64 on the same seeds and was therefore not yet an
authoritative Stage 6 teacher.

An isolated scripted-forward Stage 6 teacher improved to 60/64 overall, but the
9–10 stud bin remained 8/11. Stage 17 was added as a flat, straight 9–10 stud
boundary curriculum so long-gap timing can be mastered without height or angle
confounds before returning to uniform Stage 6. The updated place is
`ObbyRL-M3-Privileged-Stage17.rbxlx`.

Stage 17 baseline success was 40/64. Focused privileged PPO improved the best
checkpoint to 55/64 (85.9%), below promotion. A lower-rate continuation regressed
to 48/64, and empirically rewarding an early takeoff mode regressed to 36/64.
Live sweeps showed seed-sensitive, multi-modal successful takeoff distances near
25, 16, and 10 studs, so the existing single checkpoint-distance reward window
is not a faithful model of long-gap physics.

An edge-trigger-only jump experiment was also rejected. It reduced the Stage 17
teacher to 47/64 and regressed the promoted Stage 2 student from 64/64 to 58/64.
Cooldown-based held-jump semantics were restored to preserve the established
action contract. Stage 17 remains unpromoted; the next solution should improve
action cadence/takeoff representation or use an empirically learned feasibility
model rather than add more PPO steps against the current single-window shaping.

### Variable-cadence and macro-action audit

The bridge now accepts `action_repeat_ticks` per transport reset while retaining
three ticks (20 Hz) as the compatibility default. Two ticks produced about 74
transitions/second and exposed additional successful timing samples around
15.5–17.4 studs, compared with the coarser 20 Hz sweep. Cooldown was scaled from
8 to 12 decisions to preserve its 0.4-second real-time duration.

However, the trained 30 Hz teacher achieved only 46/64 at the end and 54/64 at
its best saved checkpoint, below the 20 Hz best of 55/64. A scripted
jump-on-first-decision macro completed only 8/64 Stage 17 seeds. Therefore neither
higher cadence nor a universal immediate jump solves the boundary. Optimal
takeoff timing is seed/state dependent, and future shaping should come from an
empirical feasibility dataset that records successful timing intervals for many
gaps rather than another fixed or linearly shifted distance window.

### First empirical feasibility model

A resumable calibration sweep collected 230 valid takeoff-state samples across
16 development and 8 untouched validation seeds. Every seed exposed three or
four successful timing choices. A 48-input MLP reported 96.1% development and
97.4% validation accuracy, with 100% recall on both partitions.

Using its probability for reward shaping did not improve the unshielded PPO
teacher (45/64). A suppress-only action shield peaked at 60/64 on its calibration
partition with threshold 0.3 but fell to 47/64 on an independent partition. An
authoritative shield that forced a jump at classifier-approved states achieved
only 8/64, matching the universal immediate-jump macro.

The high offline classification score is therefore not a valid feasibility
estimate: one Roblox physics rollout per `(seed, timing)` was treated as ground
truth, while fresh rollouts reveal outcome noise. The next calibration must run
each timing repeatedly and learn empirical success probability. No feasibility
model or Stage 17 teacher from this experiment is promoted.

### Lane-balanced timing and phase-aware teacher

The first timing dataset was also confounded because each candidate timing was
tied to a different simulation lane. A corrected lane-balanced sweep applies
every candidate timing to all eight lanes. Across 12 seeds, decision 5 was the
best single macro (94.7% in-sample); decisions 4 and 6 achieved 63.8% and 12.1%.
An initial fresh partition happened to score 64/64, but a later untouched
partition scored only 50/64. The macro is useful but not universal.

Distilling decision 5 into the snapshot-only teacher reached 42/64. The
privileged observation now replaces its otherwise-unused normalized checkpoint
index (feature 44) with normalized episode decision phase. This does not change
or leak into the student's 22-value observation. A phase-aware teacher then
achieved 57/64 on two fresh partitions while emitting a clean single jump on
decision 5. Because the exact scripted macro scored only 50/64 on the matching
partition, the residual is seed/physics variability, not failure to represent
the macro. Neither checkpoint is promoted. Calibration must use more seeds and
lane-balanced outcomes before state-dependent timing or distillation resumes.

A larger focused calibration then collected 554 valid samples over 24 new
seeds, testing decisions 4–6 in every lane. Its held-out Brier score was 0.061,
and states assigned at least 0.7 feasibility had 93.6% empirical target
probability. Despite those offline metrics, an authoritative gate achieved only
48/64 (75%) on a completely new live partition. This model is also rejected.
The repeated offline-to-live gap, plus seed totals ranging from 3 to 17
successful lane/timing trials out of 24, makes reset settling, lane-specific
physics, and unobserved initial state the next audit target. More PPO or teacher
distillation is paused until that simulator variance is measured and controlled.

This privileged-teacher branch is now closed as a core-curriculum blocker. The
9-10 stud range is retained only as an optional stress benchmark; no checkpoint
from this branch is promoted. Mainline work continues with capped combined
geometry and multi-segment randomized courses.
