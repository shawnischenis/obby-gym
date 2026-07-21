--!strict

local ObservationBuilder = {}
ObservationBuilder.SIZE = 22

local function clip(value: number, low: number, high: number): number
	return math.clamp(value, low, high)
end

local function scaledVector(root: BasePart, target: Vector3): Vector3
	return root.CFrame:VectorToObjectSpace(target - root.Position)
end

local function rayFraction(
	origin: Vector3,
	direction: Vector3,
	length: number,
	params: RaycastParams
): number
	local result = workspace:Raycast(origin, direction.Unit * length, params)
	if result == nil then
		return 1
	end
	return clip(result.Distance / length, 0, 1)
end

function ObservationBuilder.build(
	root: BasePart,
	humanoid: Humanoid,
	checkpoint: Vector3,
	finish: Vector3,
	progress: number,
	previousAction: { strafe: number, forward: number, yaw: number, jump: boolean },
	ignore: { Instance },
	jumpGeometry: Vector3?
): { number }
	local localVelocity = root.CFrame:VectorToObjectSpace(root.AssemblyLinearVelocity)
	local checkpointLocal = scaledVector(root, checkpoint)
	local finishLocal = scaledVector(root, finish)
	local params = RaycastParams.new()
	params.FilterType = Enum.RaycastFilterType.Exclude
	params.FilterDescendantsInstances = ignore

	local origin = root.Position
	local forward = root.CFrame.LookVector
	local right = root.CFrame.RightVector
	local down = Vector3.new(0, -1, 0)
	local grounded = humanoid.FloorMaterial ~= Enum.Material.Air
	-- AgentHarness supplies a stable per-segment geometry vector. The fallback is
	-- retained only for legacy callers outside the procedural harness.
	local routeFeatures = jumpGeometry
		or Vector3.new(
			clip(finishLocal.X / 64, -1, 1),
			clip(finishLocal.Y / 32, -1, 1),
			clip(finishLocal.Z / 64, -1, 1)
		)

	local values = {
		clip(localVelocity.X / 32, -1, 1),
		clip(localVelocity.Y / 32, -1, 1),
		clip(localVelocity.Z / 32, -1, 1),
		clip(root.AssemblyAngularVelocity.Y / 8, -1, 1),
		if grounded then 1 else 0,
		if grounded then 1 else 0,
		clip(checkpointLocal.X / 64, -1, 1),
		clip(checkpointLocal.Y / 32, -1, 1),
		clip(checkpointLocal.Z / 64, -1, 1),
		routeFeatures.X,
		routeFeatures.Y,
		routeFeatures.Z,
		clip(progress, 0, 1),
		rayFraction(origin, forward, 24, params),
		rayFraction(origin, forward + down, 24, params),
		rayFraction(origin, down, 12, params),
		rayFraction(origin, -right + down, 16, params),
		rayFraction(origin, right + down, 16, params),
		clip(previousAction.strafe, -1, 1),
		clip(previousAction.forward, -1, 1),
		clip(previousAction.yaw, -1, 1),
		if previousAction.jump then 1 else 0,
	}
	assert(#values == ObservationBuilder.SIZE, "observation layout size mismatch")
	return values
end

return ObservationBuilder
