local ReplicatedStorage = game:GetService("ReplicatedStorage")
local AgentHarness = require(ReplicatedStorage.ObbyRL.AgentHarness)
local CourseGenerator = require(ReplicatedStorage.ObbyRL.CourseGenerator)
local GapCourseConfig = require(ReplicatedStorage.ObbyRL.GapCourseConfig)
local ResetController = require(ReplicatedStorage.ObbyRL.ResetController)

local character = Instance.new("Model")
character.Name = "ResetSmokeCharacter"
local humanoid = Instance.new("Humanoid")
humanoid.Parent = character
local root = Instance.new("Part")
root.Name = "HumanoidRootPart"
root.Anchored = true
root.Parent = character
character.PrimaryPart = root
character.Parent = workspace

local manifest = CourseGenerator.manifest(7, GapCourseConfig)
local state = AgentHarness.new(character, manifest)
local controller = ResetController.new(state, -25)
root.Position = Vector3.new(0, -30, 0)
assert(ResetController.step(controller), "fall should trigger a reset")
assert(root.Position.Y > -25, "reset should return the root above the kill plane")
assert(character:GetAttribute("LastResetReason") == "fall", "fall reason was not recorded")
assert(character:GetAttribute("ResetCount") == 1, "reset count was not recorded")
character:Destroy()

print("[ObbyRL] M1 reset smoke passed")
