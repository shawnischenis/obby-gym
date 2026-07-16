# M1 performance decision

## Measurements

| Test | Result |
|---|---:|
| Original 150 ms action window | 4.31 steps/s |
| Corrected 3-heartbeat (~50 ms) action window | 7.49 steps/s |
| HTTP/1.1 keep-alive experiment | 5.52 steps/s (reverted) |
| 1,000-reset soak | 4.86 resets/s |
| Reset latency | 85.8 ms median / 101.9 ms p95 |

The synchronous plugin HTTP exchange contributes roughly 80–100 ms per command. Reducing the physics action window further cannot make a single worker reach 20 environment steps/s.

## Decision

Keep the simple HTTP protocol for correctness and M2 generator work. Before large PPO runs, add vectorized collection: several agents/environments should share one plugin exchange so aggregate throughput scales without increasing request rate. Parallel Studio workers remain a secondary option after one vectorized worker is reliable.

Do not adopt the HTTP/1.1 experiment; it reduced measured throughput. Do not increase request frequency beyond the documented Roblox limits.
