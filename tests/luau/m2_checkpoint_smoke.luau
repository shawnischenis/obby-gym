local ReplicatedStorage = game:GetService("ReplicatedStorage")
local AgentHarness = require(ReplicatedStorage.ObbyRL.AgentHarness)
local Config = require(ReplicatedStorage.ObbyRL.ProceduralCourseConfig)
local Generator = require(ReplicatedStorage.ObbyRL.ProceduralCourseGenerator)

local character = Instance.new("Model")
character.Name = "M2CheckpointCharacter"
local humanoid = Instance.new("Humanoid")
humanoid.Parent = character
local root = Instance.new("Part")
root.Name = "HumanoidRootPart"
root.Anchored = true
root.Parent = character
character.PrimaryPart = root
character.Parent = workspace

local manifest = Generator.manifest(17, Config)
local state = AgentHarness.new(character, manifest)
assert(not humanoid.AutoRotate, "RL harness must own yaw instead of Humanoid.AutoRotate")
AgentHarness.reset(state)
assert(state.checkpointIndex == 0, "episode reset should clear checkpoint progress")
assert(state.checkpoint == manifest.segments[1].checkpointPosition, "first target mismatch")

root.Position = manifest.segments[1].checkpointPosition
assert(AgentHarness.advanceCheckpoint(state), "first checkpoint was not reached")
assert(state.checkpointIndex == 1, "checkpoint index did not advance")
assert(state.checkpoint == manifest.segments[2].checkpointPosition, "next target mismatch")

root.Position = Vector3.new(100, -100, 100)
AgentHarness.recover(state)
local recoveryDelta = root.Position - manifest.segments[1].checkpointPosition
assert(math.abs(recoveryDelta.X) < 0.01, "recovery X mismatch")
assert(math.abs(recoveryDelta.Z - 3) < 0.01, "recovery Z mismatch")
assert(
	root.Position.Y > manifest.segments[1].checkpointPosition.Y,
	"recovery should be above platform"
)

root.Position = manifest.segments[3].checkpointPosition
assert(not AgentHarness.advanceCheckpoint(state), "checkpoints must be reached in order")
assert(state.checkpointIndex == 1, "out-of-order touch advanced checkpoint")

for index = 2, Config.stageCount do
	root.Position = manifest.segments[index].checkpointPosition
	assert(AgentHarness.advanceCheckpoint(state), `checkpoint {index} was not reached`)
end
assert(state.checkpointIndex == Config.stageCount, "scripted route did not reach final checkpoint")
assert(state.checkpoint == manifest.finishPosition, "finish should be the final observation target")
root.Position = manifest.finishPosition
local observation = AgentHarness.observe(state)
assert(observation[13] >= 0.98, "scripted route did not satisfy finish progress")

character:Destroy()
print("[ObbyRL] M2 checkpoint recovery smoke passed")
