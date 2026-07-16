--!strict

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")

local AgentHarness = require(ReplicatedStorage.ObbyRL.AgentHarness)
local Config = require(ReplicatedStorage.ObbyRL.Config)
local CourseGenerator = require(ReplicatedStorage.ObbyRL.CourseGenerator)
local GapCourseConfig = require(ReplicatedStorage.ObbyRL.GapCourseConfig)
local ResetController = require(ReplicatedStorage.ObbyRL.ResetController)

workspace.Gravity = Config.WORKSPACE_GRAVITY
local manifest = CourseGenerator.build(0, GapCourseConfig, workspace)
local course = workspace:WaitForChild("GeneratedCourse")
local controllersByCharacter: { [Model]: ResetController.Controller } = {}

local function connectKillPlane(model: Model)
	local currentKillPlane = model:WaitForChild("KillPlane") :: BasePart
	currentKillPlane.Touched:Connect(function(hit: BasePart)
		local character = hit:FindFirstAncestorOfClass("Model")
		if character then
			local controller = controllersByCharacter[character]
			if controller then
				ResetController.reset(controller, "hazard")
			end
		end
	end)
end

connectKillPlane(course)

local function onCharacter(character: Model)
	local state = AgentHarness.new(character, manifest)
	AgentHarness.reset(state)
	controllersByCharacter[character] = ResetController.new(state, Config.FALL_Y)
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
