# M2 complete — procedural obby generator v1

M2 is complete. The generator, deterministic feasibility oracle/resampling path, checkpoint runtime, action-level route planner, and live bridge integration are implemented and tested.

## Implemented

- Versioned procedural-course JSON configuration and JSON Schema.
- Deterministic replay from an integer seed with a serializable manifest signature.
- Eight-stage assembly from four static segment families: gap, offset jump, beam, and stairs.
- Conservative jump-gap and lateral-offset limits.
- Cross-stage axis-aligned overlap validation.
- Ordered checkpoint markers, finish marker, and a course-wide kill plane.
- Ordered checkpoint state: observations target the next stage, and recovery returns the avatar to the latest reached checkpoint.
- Procedural-course support in the Studio plugin, including manifest signatures and checkpoint progress in result metadata.
- Agent-owned yaw control (`Humanoid.AutoRotate` disabled), preventing sustained strafe actions from curling into circles.
- Conservative feasibility oracle for checkpoint continuity, beam width, stair controller limits, and total course rise.
- Deterministic candidate resampling with generation seed, attempt number, and human-readable rejection audit entries in every manifest.
- Action-level scripted plans for jump landings, beam traversal, stair steps, and stair landings.
- An ordered checkpoint-to-finish state-machine test that reaches all eight checkpoints and finish progress of at least 0.98.
- Studio property test over seeds `0..1999`, including deterministic replay and obstacle-family coverage.

The first sample produced 16,000 segments with this distribution:

| Family | Count |
| --- | ---: |
| Gap | 3,992 |
| Offset jump | 4,108 |
| Beam | 3,963 |
| Stairs | 3,937 |

All 2,000 accepted manifests passed the current reachability, overlap, and feasibility checks. The oracle rejected 241 candidates; 206 requested seeds deterministically resampled at least once. The accepted obstacle distribution was:

| Family | Count |
| --- | ---: |
| Gap | 4,118 |
| Offset jump | 4,295 |
| Beam | 4,132 |
| Stairs | 3,455 |

## Live runtime validation

The Studio/Python bridge passed the focused M2 live gate:

- Two reset passes over 25 seeds reproduced identical manifest signatures and instance counts.
- All 25 seeds produced distinct manifest signatures.
- Procedural instance counts varied as expected from `22` to `43` because stair counts vary.
- Sustained strafe moved in a straight line and forced a real fall.
- The fall emitted `hazard_recovered`, applied the hazard penalty, returned the avatar to its checkpoint, and did not terminate the episode.
- A 100-step random-action run measured 7.43 environment steps/second.
- The final extended gate completed 200/200 resets: 100 seeds across two replay passes, with 100 identical replays and 100 unique signatures.
- Instance counts across that accepted sample ranged from 20 to 38, reflecting obstacle and stair-count variation without leaked course instances.
- A forced hazard recovery passed immediately after the extended reset sequence, confirming the bridge remained synchronized.

## Exit decision

M2 exit criteria are satisfied: the validation sample has no accepted overlap/out-of-bounds failures, every accepted course has an oracle-validated checkpoint chain, and rejected candidates resample deterministically with auditable reasons. Work can proceed to the fixed-course PPO baseline in M3.

## Reproduce the current gate

```bash
rojo build -o /tmp/ObbyRL-m2.rbxlx
/Applications/RobloxStudio.app/Contents/MacOS/RobloxStudio \
  --task RunScript \
  --localPlaceFile /tmp/ObbyRL-m2.rbxlx \
  --runScriptFile "$PWD/tests/luau/m2_generator_properties.luau" \
  --outputFile /tmp/obbyrl-m2-properties.log \
  --quitAfterExecution
```
