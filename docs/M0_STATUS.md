# M0 status

M0 is complete as of 2026-07-16.

- [x] R15 avatar, `Humanoid:Move()` controller, cadence, action repeat, and episode limit documented.
- [x] Python package and exact dependency versions defined.
- [x] Rojo project builds successfully with Rojo 7.5.1.
- [x] Configuration, protocol, action, reset, and step-result schemas defined.
- [x] Train, validation, test, and stress seeds are disjoint and versioned.
- [x] Configuration SHA-256 and run provenance collection implemented.
- [x] Python tests pass.
- [x] StyLua formatting check passes.
- [x] Selene reports zero errors/warnings.
- [x] Roblox Studio CLI smoke test passes against the built place.

Verified Studio version: `0.730.0.7300790`. The next work item is M1: one seeded gap obstacle, reset/action control, structured observations, and a measured Studio↔Python transport loop.
