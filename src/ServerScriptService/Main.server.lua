--!strict

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Config = require(ReplicatedStorage.ObbyRL.Config)

workspace.Gravity = Config.WORKSPACE_GRAVITY
print(
	string.format(
		"[ObbyRL] M0 ready (config=%s protocol=%s generator=%s)",
		Config.CONFIG_VERSION,
		Config.PROTOCOL_VERSION,
		Config.GENERATOR_VERSION
	)
)
