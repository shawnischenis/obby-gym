# M1 status

## Implemented and verified

- [x] Deterministic single-gap manifest and Roblox geometry generation.
- [x] Replay by seed; seed `12345` generated a `5.6463` stud gap in Studio.
- [x] R15 reset and `Humanoid:Move()` action harness with strafe, forward, yaw, and jump.
- [x] Hazard-touch and below-kill-plane recovery reset the same character in place, with debounce and reset diagnostics.
- [x] Fixed 22-float `obby-structured-v1` observation layout with local velocity, targets, progress, rays, and previous action.
- [x] Gymnasium environment adapter with checked observation/action spaces.
- [x] Deterministic fake transport for contract tests without Studio.
- [x] Python 3.12 workspace environment with the pinned project dependencies.
- [x] Eight Python tests, Ruff, mypy, StyLua, Selene, and Rojo build pass.
- [x] Studio CLI generator smoke test passes.
- [x] Studio CLI reset smoke test passes.

## Live-loop results

- [x] Python transport connects to a running playtest through the installed Studio plugin.
- [x] Initial benchmark: 100 random-policy steps in 23.193 seconds, or **4.31 environment steps/second**.
- [x] Physics-timing fix benchmark: 100 steps in 13.349 seconds, or **7.49 environment steps/second**.
- [x] 1,000 live resets completed without protocol desynchronization.
- [x] Reset soak: 205.762 seconds total, 4.86 resets/s, 85.8 ms median, 101.9 ms p95.
- [x] Generated-course descendant count remained exactly `4` across all 1,000 resets; no geometry leak detected.

The optimized single-agent throughput remains below the M0 gate of 20 environment steps/second per worker. A trial with HTTP/1.1 keep-alive regressed to 5.52 steps/s and was reverted. The next throughput strategy should vectorize multiple agents per plugin exchange or run parallel Studio workers; further single-request tuning is unlikely to close the gap.

The bridge runs in `plugin/ObbyRLBridge.plugin.lua`, because Roblox supports local-machine HTTP for Studio plugins. The playtest experience owns course physics and reusable harness modules but does not make external requests.
