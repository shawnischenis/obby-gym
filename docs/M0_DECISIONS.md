# M0 decisions

These defaults are locked for the first end-to-end environment and may only change through a versioned configuration update.

| Decision | M0 value | Reason |
|---|---|---|
| Avatar | R15 | Current standard Roblox rig and representative humanoid physics |
| Controller | `Humanoid:Move()` | Smallest path to a working agent harness; custom physics can be evaluated later |
| Decision rate | 20 Hz | Responsive enough for jumping while keeping bridge/inference load bounded |
| Action repeat | 3 physics heartbeats (~50 ms at 60 Hz) | Targets a 20 Hz environment decision rate while holding each action across physics updates |
| Episode limit | 20 seconds | Appropriate for the initial short course and bounds stalled episodes |
| Transport | Studio plugin over loopback HTTP | Roblox officially supports localhost communication from plugins; the playtest experience itself does not own the external connection |
| Observation cap | 128 floats | Initial engineering budget; the exact v1 layout is an M1 deliverable |
| Training security | Loopback only | Python binds to `127.0.0.1`; the plugin permission is limited to the local broker |
| Throughput gate | 20 environment steps/s per Studio worker | Below this, M1 must optimize or reconsider the data plane before full training |
| Luau inference gate | 2 ms median, 5 ms p99 per action on target client | Keeps policy evaluation safely below a frame budget |

The Roblox and Python constants intentionally duplicate a small set of values in M0. `scripts/validate_m0.py` validates the canonical JSON configuration and checks the shared schema/version constants against Luau. M1 should generate the entire Luau configuration from canonical JSON before expanding the contract.
