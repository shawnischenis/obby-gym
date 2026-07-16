-- Run with Studio CLI --task RunScript after building/opening the Rojo place.
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local config = require(ReplicatedStorage.ObbyRL.Config)

assert(config.CONFIG_VERSION == "0.1.0")
assert(config.PROTOCOL_VERSION == "0.1.0")
assert(config.GENERATOR_VERSION == "0.1.0")
assert(config.AVATAR_RIG == Enum.HumanoidRigType.R15)
assert(config.BRIDGE_LOCAL_ONLY == true)

print("[ObbyRL] M0 Studio smoke test passed")
