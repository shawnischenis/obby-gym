--!strict

local ObservationBuilder = require(script.Parent.ObservationBuilder)

local AgentHarness = {}

export type Action = { strafe: number, forward: number, yaw: number, jump: boolean }
export type State = {
	character: Model,
	humanoid: Humanoid,
	root: BasePart,
	spawnCFrame: CFrame,
	checkpoint: Vector3,
	finish: Vector3,
	courseStart: Vector3,
	courseEnd: Vector3,
	previousAction: Action,
}

local ZERO_ACTION: Action = { strafe = 0, forward = 0, yaw = 0, jump = false }

function AgentHarness.new(character: Model, manifest: any): State
	local humanoid = character:FindFirstChildOfClass("Humanoid")
	local root = character:FindFirstChild("HumanoidRootPart")
	assert(humanoid ~= nil, "character requires a Humanoid")
	assert(root ~= nil and root:IsA("BasePart"), "character requires HumanoidRootPart")
	return {
		character = character,
		humanoid = humanoid,
		root = root,
		spawnCFrame = manifest.spawnCFrame,
		checkpoint = manifest.checkpointPosition,
		finish = manifest.finishPosition,
		courseStart = manifest.courseStart,
		courseEnd = manifest.courseEnd,
		previousAction = table.clone(ZERO_ACTION),
	}
end

function AgentHarness.reset(state: State)
	state.character:PivotTo(state.spawnCFrame + Vector3.new(0, 3, 0))
	state.root.AssemblyLinearVelocity = Vector3.zero
	state.root.AssemblyAngularVelocity = Vector3.zero
	state.humanoid.Health = state.humanoid.MaxHealth
	state.previousAction = table.clone(ZERO_ACTION)
end

function AgentHarness.applyAction(state: State, action: Action)
	local strafe = math.clamp(action.strafe, -1, 1)
	local forward = math.clamp(action.forward, -1, 1)
	local localMove = Vector3.new(strafe, 0, -forward)
	if localMove.Magnitude > 1 then
		localMove = localMove.Unit
	end
	local worldMove = state.root.CFrame:VectorToWorldSpace(localMove)
	state.humanoid:Move(worldMove, false)
	local yaw = math.clamp(action.yaw, -1, 1)
	if math.abs(yaw) > 0.001 then
		state.root.CFrame = state.root.CFrame * CFrame.Angles(0, math.rad(yaw * 8), 0)
	end
	if action.jump then
		state.humanoid.Jump = true
	end
	state.previousAction = { strafe = strafe, forward = forward, yaw = yaw, jump = action.jump }
end

function AgentHarness.observe(state: State): { number }
	local route = state.courseEnd - state.courseStart
	local progress = (state.root.Position - state.courseStart):Dot(route.Unit) / route.Magnitude
	return ObservationBuilder.build(
		state.root,
		state.humanoid,
		state.checkpoint,
		state.finish,
		progress,
		state.previousAction,
		{ state.character }
	)
end

return AgentHarness
