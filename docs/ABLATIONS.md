# ObbyRL ablations

This document distinguishes completed comparisons from proposed experiments. A
row is a result only when both policies were evaluated on the recorded protocol;
otherwise it is marked **not run**.

## Completed comparisons

| Question | Variant | Evaluation | Result | Artifact |
| --- | --- | --- | ---: | --- |
| Does privileged sensing produce the strongest deployed policy? | 48-input privileged PPO teacher | Same two held-out Stage 2 partitions | 113/128 (88.3%) | `m3-privileged-teacher-stage2-mixed-movement-v1` |
|  | 22-input limited-sensing DAgger student | Same two held-out Stage 2 partitions | 128/128 (100%) | `m3-stage1-2-privileged-dagger-v1/dagger_3.zip` |
| How should privileged demonstrations be transferred? | One-pass behavior clone | One held-out Stage 2 partition | 54/64 (84.4%) | `m3-stage1-2-privileged-distillation-v1` |
|  | Behavior clone + DAgger | Two held-out Stage 2 partitions | 128/128 (100%) | `m3-stage1-2-privileged-dagger-v1/dagger_3.zip` |
|  | DAgger + conservative PPO | Same two Stage 2 partitions | 128/128 (100%); 12.25–12.48 mean decisions | `m3-stage1-2-student-ppo-v1/final_model.zip` |

The privileged comparison is not a claim that privileged information is
harmful. The teacher made the student dataset possible. It shows that privileged
PPO performance is not an upper bound: DAgger can train a limited student on the
student-induced state distribution and surpass the labeling policy.

The imitation + RL comparison shows no completion-rate gain after DAgger reached
100%. Its measured benefit was efficiency: mean Stage 2 completion fell from
approximately 14 decisions to 12.25–12.48 without Stage 1 regression.

## Pre-registered memory ablation — not run

| Item | Feedforward | Recurrent |
| --- | --- | --- |
| Input | Current 22-value observation | Same current observation |
| Capacity | Match recurrent parameter count | One LSTM layer + action/value heads |
| Algorithm | PPO | Recurrent PPO |
| Seeds | Identical training, development, and untouched validation seeds | Identical |
| Budget | Equal simulator transitions and evaluation episodes | Equal |
| State reset | N/A | Reset hidden state on every termination/truncation |
| Primary metric | Clean course completion | Clean course completion |
| Memory subset | Delayed/occluded cue courses | Same courses |

The current observation exposes local checkpoint geometry, so standard obbies may
not require memory. A meaningful memory ablation must include a predeclared
partially observable subset—such as an occluded landing, delayed direction cue,
or moving hazard—and compare against both a parameter-matched feedforward model
and a short action/observation-history baseline.

## Evaluation policy

- Deterministic action selection.
- Distinct development and untouched validation course seeds.
- Clean completion excludes hazard recovery.
- Promotion normally requires at least 90% combined completion, with neither
  partition below 85%.
- Results are frozen checkpoint evaluations, not training-rollout return.
