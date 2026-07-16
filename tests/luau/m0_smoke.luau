-- Run with Studio CLI --task RunScript after building/opening the Rojo place.
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local config = require(ReplicatedStorage.ObbyRL.Config)

assert(config.CONFIG_VERSION == "0.1.0", "unexpected config version")
assert(config.PROTOCOL_VERSION == "0.1.0", "unexpected protocol version")
assert(config.GENERATOR_VERSION == "0.1.0", "unexpected generator version")
assert(config.AVATAR_RIG == Enum.HumanoidRigType.R15, "M0 requires R15")
assert(config.BRIDGE_LOCAL_ONLY == true, "M0 bridge must remain local-only")

print("[ObbyRL] M0 Studio smoke test passed")
