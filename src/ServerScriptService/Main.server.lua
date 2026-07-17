--!strict

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")

local AgentHarness = require(ReplicatedStorage.ObbyRL.AgentHarness)
local Config = require(ReplicatedStorage.ObbyRL.Config)
local CourseGenerator = require(ReplicatedStorage.ObbyRL.ProceduralCourseGenerator)
local CourseConfig = require(ReplicatedStorage.ObbyRL.ProceduralCourseConfig)
local ResetController = require(ReplicatedStorage.ObbyRL.ResetController)

local controlStatus = ReplicatedStorage.ObbyRL:FindFirstChild("ControlStatus")
if not controlStatus then
	controlStatus = Instance.new("RemoteEvent")
	controlStatus.Name = "ControlStatus"
	controlStatus.Parent = ReplicatedStorage.ObbyRL
end
local controlStatusEvent = controlStatus :: RemoteEvent
controlStatusEvent.OnServerEvent:Connect(function(player: Player, disabled: boolean)
	player:SetAttribute("ObbyRLDefaultControlsDisabled", disabled == true)
end)

workspace.Gravity = Config.WORKSPACE_GRAVITY
local manifest = CourseGenerator.build(0, CourseConfig, workspace)
local course = workspace:WaitForChild("GeneratedCourseV2")
local controllersByCharacter: { [Model]: ResetController.Controller } = {}

local function connectKillPlane(model: Model)
	local currentKillPlane = model:WaitForChild("KillPlane") :: BasePart
	currentKillPlane.Touched:Connect(function(hit: BasePart)
		local character = hit:FindFirstAncestorOfClass("Model")
		if character then
			local controller = controllersByCharacter[character]
			if controller then
				AgentHarness.recover(controller.state)
				character:SetAttribute("LastResetReason", "hazard")
			end
		end
	end)
end

connectKillPlane(course)

for index = 1, CourseConfig.stageCount do
	local checkpoint = course:WaitForChild(string.format("Checkpoint_%02d", index)) :: BasePart
	checkpoint.Touched:Connect(function(hit: BasePart)
		local character = hit:FindFirstAncestorOfClass("Model")
		local controller = if character then controllersByCharacter[character] else nil
		if controller then
			AgentHarness.advanceCheckpoint(controller.state)
		end
	end)
end

local function onCharacter(character: Model)
	local state = AgentHarness.new(character, manifest)
	AgentHarness.reset(state)
	-- The plugin observes the configured kill-plane crossing to assign the RL penalty.
	-- Keep this heartbeat reset only as a deep-fall safety net so it cannot win that race.
	controllersByCharacter[character] = ResetController.new(state, CourseConfig.killPlaneY - 25)
	character.Destroying:Once(function()
		controllersByCharacter[character] = nil
	end)
	character:SetAttribute("ObbyRLReady", true)
	character:SetAttribute("ObservationSize", #AgentHarness.observe(state))
end

RunService.Heartbeat:Connect(function()
	for character, controller in controllersByCharacter do
		if character.Parent then
			ResetController.step(controller)
		end
	end
end)

local function onPlayer(player: Player)
	player.CharacterAdded:Connect(onCharacter)
	if player.Character then
		task.spawn(onCharacter, player.Character)
	end
end

Players.PlayerAdded:Connect(onPlayer)
for _, player in Players:GetPlayers() do
	onPlayer(player)
end

print(
	string.format(
		"[ObbyRL] M0 ready (config=%s protocol=%s generator=%s)",
		Config.CONFIG_VERSION,
		Config.PROTOCOL_VERSION,
		Config.GENERATOR_VERSION
	)
)
