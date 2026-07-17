--!strict

local ScriptedCourseOracle = require(script.Parent.ScriptedCourseOracle)

export type PartSpec = {
	name: string,
	kind: string,
	position: Vector3,
	size: Vector3,
	stage: number,
}

export type Segment = {
	index: number,
	kind: string,
	entryPosition: Vector3,
	exitPosition: Vector3,
	checkpointPosition: Vector3,
	parameters: { [string]: number },
}

export type Manifest = {
	seed: number,
	generationSeed: number,
	generationAttempt: number,
	rejections: { string },
	generatorVersion: string,
	parts: { PartSpec },
	segments: { Segment },
	spawnCFrame: CFrame,
	finishPosition: Vector3,
	courseStart: Vector3,
	courseEnd: Vector3,
	signature: string,
}

local Generator = {}
local VERSION = "0.4.0"
local KINDS = { "gap", "offset", "beam", "stairs" }

local function platformSpec(index: number, position: Vector3, config: any): PartSpec
	return {
		name = string.format("Platform_%02d", index),
		kind = "platform",
		position = position,
		size = Vector3.new(config.platformWidth, config.platformHeight, config.platformLength),
		stage = index,
	}
end

local function appendNumber(parts: { string }, value: number)
	table.insert(parts, string.format("%.5f", value))
end

local function signature(
	seed: number,
	generationSeed: number,
	segments: { Segment },
	parts: { PartSpec }
): string
	local values = { tostring(seed), tostring(generationSeed), VERSION }
	for _, segment in segments do
		table.insert(values, segment.kind)
		appendNumber(values, segment.exitPosition.X)
		appendNumber(values, segment.exitPosition.Y)
		appendNumber(values, segment.exitPosition.Z)
	end
	for _, part in parts do
		table.insert(values, part.kind)
		appendNumber(values, part.position.X)
		appendNumber(values, part.position.Y)
		appendNumber(values, part.position.Z)
		appendNumber(values, part.size.X)
		appendNumber(values, part.size.Y)
		appendNumber(values, part.size.Z)
	end
	return table.concat(values, "|")
end

local function addJump(
	random: Random,
	index: number,
	kind: string,
	current: Vector3,
	config: any,
	parts: { PartSpec }
): (Vector3, Segment)
	local gap = random:NextNumber(config.gapMin, config.gapMax)
	local height = random:NextNumber(config.jumpHeightMin or 0, config.jumpHeightMax or 0)
	local offset = if kind == "offset"
		then random:NextNumber(config.offsetMin, config.offsetMax)
		else 0
	local exit = current + Vector3.new(offset, height, -(config.platformLength + gap))
	table.insert(parts, platformSpec(index, exit, config))
	return exit,
		{
			index = index,
			kind = kind,
			entryPosition = current,
			exitPosition = exit,
			checkpointPosition = exit,
			parameters = { gap = gap, offset = offset, height = height },
		}
end

local function addBeam(
	random: Random,
	index: number,
	current: Vector3,
	config: any,
	parts: { PartSpec }
): (Vector3, Segment)
	local length = random:NextNumber(config.beamLengthMin, config.beamLengthMax)
	local width = random:NextNumber(config.beamWidthMin, config.beamWidthMax)
	local entryEdgeZ = current.Z - config.platformLength / 2
	local beamCenter = Vector3.new(current.X, current.Y, entryEdgeZ - length / 2)
	table.insert(parts, {
		name = string.format("Beam_%02d", index),
		kind = "beam",
		position = beamCenter,
		size = Vector3.new(width, config.platformHeight, length),
		stage = index,
	})
	local exit = Vector3.new(current.X, current.Y, entryEdgeZ - length - config.platformLength / 2)
	table.insert(parts, platformSpec(index, exit, config))
	return exit,
		{
			index = index,
			kind = "beam",
			entryPosition = current,
			exitPosition = exit,
			checkpointPosition = exit,
			parameters = { length = length, width = width },
		}
end

local function addStairs(
	random: Random,
	index: number,
	current: Vector3,
	config: any,
	parts: { PartSpec }
): (Vector3, Segment)
	local count = random:NextInteger(config.stairCountMin, config.stairCountMax)
	local entryEdgeZ = current.Z - config.platformLength / 2
	for stair = 1, count do
		table.insert(parts, {
			name = string.format("Stair_%02d_%02d", index, stair),
			kind = "stair",
			position = Vector3.new(
				current.X,
				current.Y + stair * config.stairRise,
				entryEdgeZ - (stair - 0.5) * config.stairRun
			),
			size = Vector3.new(config.platformWidth, config.platformHeight, config.stairRun),
			stage = index,
		})
	end
	local exit = Vector3.new(
		current.X,
		current.Y + count * config.stairRise,
		entryEdgeZ - count * config.stairRun - config.platformLength / 2
	)
	table.insert(parts, platformSpec(index, exit, config))
	return exit,
		{
			index = index,
			kind = "stairs",
			entryPosition = current,
			exitPosition = exit,
			checkpointPosition = exit,
			parameters = { count = count, run = config.stairRun, rise = config.stairRise },
		}
end

local function candidate(requestedSeed: number, generationSeed: number, config: any): Manifest
	local random = Random.new(generationSeed)
	local start = Vector3.new(0, 4, 0)
	local current = start
	local parts: { PartSpec } = { platformSpec(0, start, config) }
	if config.startSafetyDepth and config.startSafetyDepth > 0 then
		table.insert(parts, {
			name = "StartSafetyApron",
			kind = "platform",
			position = start
				+ Vector3.new(0, 0, config.platformLength / 2 + config.startSafetyDepth / 2),
			size = Vector3.new(
				config.platformWidth,
				config.platformHeight,
				config.startSafetyDepth
			),
			stage = 0,
		})
	end
	local segments: { Segment } = {}
	local kinds = config.segmentKinds or KINDS
	for index = 1, config.stageCount do
		local kind = kinds[random:NextInteger(1, #kinds)]
		local segment: Segment
		if kind == "gap" or kind == "offset" then
			current, segment = addJump(random, index, kind, current, config, parts)
		elseif kind == "beam" then
			current, segment = addBeam(random, index, current, config, parts)
		else
			current, segment = addStairs(random, index, current, config, parts)
		end
		table.insert(segments, segment)
	end
	return {
		seed = requestedSeed,
		generationSeed = generationSeed,
		generationAttempt = 0,
		rejections = {},
		generatorVersion = VERSION,
		parts = parts,
		segments = segments,
		spawnCFrame = CFrame.lookAt(start + Vector3.new(0, 3, 3), start + Vector3.new(0, 3, -1)),
		finishPosition = current,
		courseStart = start,
		courseEnd = current,
		signature = signature(requestedSeed, generationSeed, segments, parts),
	}
end

local function overlaps(a: PartSpec, b: PartSpec): boolean
	local delta = a.position - b.position
	local half = (a.size + b.size) / 2
	local epsilon = 0.01
	return math.abs(delta.X) < half.X - epsilon
		and math.abs(delta.Y) < half.Y - epsilon
		and math.abs(delta.Z) < half.Z - epsilon
end

function Generator.validate(manifest: Manifest, config: any): (boolean, string?)
	if #manifest.segments ~= config.stageCount then
		return false, "stage count mismatch"
	end
	for _, segment in manifest.segments do
		if segment.kind == "gap" or segment.kind == "offset" then
			if segment.parameters.gap > config.maxJumpGap then
				return false, "jump gap exceeds limit"
			end
			if math.abs(segment.parameters.offset) > config.maxLateralOffset then
				return false, "lateral offset exceeds limit"
			end
		end
	end
	for first = 1, #manifest.parts do
		for second = first + 1, #manifest.parts do
			local a = manifest.parts[first]
			local b = manifest.parts[second]
			if a.stage ~= b.stage and overlaps(a, b) then
				return false, string.format("parts overlap: %s and %s", a.name, b.name)
			end
		end
	end
	return true, nil
end

function Generator.oracle(manifest: Manifest, config: any): (boolean, string?)
	local plan, reason = ScriptedCourseOracle.plan(manifest, config)
	return plan ~= nil, reason
end

function Generator.manifest(seed: number, config: any): Manifest
	local rejections: { string } = {}
	for attempt = 0, config.maxGenerationAttempts - 1 do
		local generationSeed = (seed + attempt * 104729) % 2147483647
		local manifest = candidate(seed, generationSeed, config)
		manifest.generationAttempt = attempt
		manifest.rejections = table.clone(rejections)
		local valid, reason = Generator.validate(manifest, config)
		if valid then
			valid, reason = Generator.oracle(manifest, config)
		end
		if valid then
			return manifest
		end
		table.insert(
			rejections,
			string.format(
				"attempt=%d generation_seed=%d reason=%s",
				attempt,
				generationSeed,
				reason or "unknown"
			)
		)
	end
	error(
		string.format(
			"seed %d exhausted %d attempts: %s",
			seed,
			config.maxGenerationAttempts,
			table.concat(rejections, "; ")
		)
	)
end

local COLORS = {
	platform = Color3.fromRGB(72, 145, 210),
	beam = Color3.fromRGB(235, 170, 65),
	stair = Color3.fromRGB(150, 105, 200),
}

local function markerPart(name: string, size: Vector3, position: Vector3, color: Color3): Part
	local part = Instance.new("Part")
	part.Name = name
	part.Anchored = true
	part.CanCollide = false
	part.Size = size
	part.Position = position
	part.Color = color
	part.TopSurface = Enum.SurfaceType.Smooth
	part.BottomSurface = Enum.SurfaceType.Smooth
	return part
end

local function translateManifest(manifest: Manifest, offset: Vector3): Manifest
	if offset.Magnitude == 0 then
		return manifest
	end
	for _, spec in manifest.parts do
		spec.position += offset
	end
	for _, segment in manifest.segments do
		segment.entryPosition += offset
		segment.exitPosition += offset
		segment.checkpointPosition += offset
	end
	manifest.spawnCFrame += offset
	manifest.finishPosition += offset
	manifest.courseStart += offset
	manifest.courseEnd += offset
	return manifest
end

function Generator.build(seed: number, config: any, parent: Instance, origin: Vector3?): Manifest
	local manifest = Generator.manifest(seed, config)
	local valid, reason = Generator.validate(manifest, config)
	assert(valid, reason)
	translateManifest(manifest, origin or Vector3.zero)
	local old = parent:FindFirstChild("GeneratedCourseV2")
	if old then
		old:Destroy()
	end
	local model = Instance.new("Model")
	model.Name = "GeneratedCourseV2"
	model:SetAttribute("CourseSeed", seed)
	model:SetAttribute("GenerationSeed", manifest.generationSeed)
	model:SetAttribute("GenerationAttempt", manifest.generationAttempt)
	model:SetAttribute("RejectedCandidates", #manifest.rejections)
	model:SetAttribute("GeneratorVersion", VERSION)
	model:SetAttribute("Signature", manifest.signature)
	for _, spec in manifest.parts do
		local part = Instance.new("Part")
		part.Name = spec.name
		part.Anchored = true
		part.Size = spec.size
		part.Position = spec.position - Vector3.new(0, spec.size.Y / 2, 0)
		part.Color = COLORS[spec.kind] or COLORS.platform
		part.TopSurface = Enum.SurfaceType.Smooth
		part.BottomSurface = Enum.SurfaceType.Smooth
		part:SetAttribute("SegmentKind", spec.kind)
		part:SetAttribute("Stage", spec.stage)
		part.Parent = model
	end
	for _, segment in manifest.segments do
		local checkpoint = markerPart(
			string.format("Checkpoint_%02d", segment.index),
			Vector3.new(config.platformWidth - 2, 0.2, 2),
			segment.checkpointPosition + Vector3.new(0, 0.1, 0),
			Color3.fromRGB(80, 220, 125)
		)
		checkpoint.Transparency = 0.35
		checkpoint:SetAttribute("CheckpointIndex", segment.index)
		checkpoint.Parent = model
	end
	local finish = markerPart(
		"Finish",
		Vector3.new(config.platformWidth, 0.25, 3),
		manifest.finishPosition + Vector3.new(0, 0.125, 0),
		Color3.fromRGB(245, 210, 60)
	)
	finish:SetAttribute("Finish", true)
	finish.Parent = model
	local killPlane = markerPart(
		"KillPlane",
		Vector3.new(300, 1, 500),
		Vector3.new(
			manifest.courseStart.X,
			config.killPlaneY,
			(manifest.courseStart.Z + manifest.courseEnd.Z) / 2
		),
		Color3.fromRGB(190, 45, 45)
	)
	killPlane.Transparency = 0.35
	killPlane:SetAttribute("Hazard", true)
	killPlane.Parent = model
	model.Parent = parent
	return manifest
end

return Generator
