# M0 decisions

These defaults are locked for the first end-to-end environment and may only change through a versioned configuration update.

| Decision | M0 value | Reason |
|---|---|---|
| Avatar | R15 | Current standard Roblox rig and representative humanoid physics |
| Controller | `Humanoid:Move()` | Smallest path to a working agent harness; custom physics can be evaluated later |
| Decision rate | 20 Hz | Responsive enough for jumping while keeping bridge/inference load bounded |
| Action repeat | 3 decision ticks | Reduces transport pressure; M1 will measure whether this harms control |
| Episode limit | 20 seconds | Appropriate for the initial short course and bounds stalled episodes |
| Transport | Local HTTP | Simple to inspect and recover; rollout messages will be batched |
| Observation cap | 128 floats | Initial engineering budget; the exact v1 layout is an M1 deliverable |
| Training security | Loopback only | The prototype bridge must not bind to a public interface |
| Throughput gate | 20 environment steps/s per Studio worker | Below this, M1 must optimize or reconsider the data plane before full training |
| Luau inference gate | 2 ms median, 5 ms p99 per action on target client | Keeps policy evaluation safely below a frame budget |

The Roblox and Python constants intentionally duplicate a small set of values in M0. `scripts/validate_m0.py` validates the canonical JSON configuration; M1 should generate the Luau configuration from that JSON or add a cross-language parity check before expanding the contract.
