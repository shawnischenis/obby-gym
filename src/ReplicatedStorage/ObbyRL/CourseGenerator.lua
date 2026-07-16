--!strict

export type GapConfig = {
	startPosition: Vector3,
	headingDegrees: number,
	platformWidth: number,
	platformLength: number,
	platformHeight: number,
	gapMin: number,
	gapMax: number,
	finishDepth: number,
	killPlaneY: number,
}

export type Manifest = {
	seed: number,
	generatorVersion: string,
	gap: number,
	spawnCFrame: CFrame,
	checkpointPosition: Vector3,
	finishPosition: Vector3,
	courseStart: Vector3,
	courseEnd: Vector3,
}

local CourseGenerator = {}
local GENERATOR_VERSION = "0.1.0"

local function makePart(name: string, size: Vector3, cframe: CFrame, color: Color3): Part
	local part = Instance.new("Part")
	part.Name = name
	part.Size = size
	part.CFrame = cframe
	part.Color = color
	part.Anchored = true
	part.TopSurface = Enum.SurfaceType.Smooth
	part.BottomSurface = Enum.SurfaceType.Smooth
	return part
end

function CourseGenerator.manifest(seed: number, config: GapConfig): Manifest
	assert(config.gapMax >= config.gapMin, "gapMax must be >= gapMin")
	local random = Random.new(seed)
	local gap = random:NextNumber(config.gapMin, config.gapMax)
	local heading = CFrame.Angles(0, math.rad(config.headingDegrees), 0)
	local forward = heading.LookVector
	local start = config.startPosition
	local secondCenter = start + forward * (config.platformLength + gap + config.platformLength / 2)
	local checkpoint = secondCenter - forward * (config.platformLength / 2 - 2)
	local finish = secondCenter + forward * (config.platformLength / 2 - config.finishDepth / 2)

	return {
		seed = seed,
		generatorVersion = GENERATOR_VERSION,
		gap = gap,
		spawnCFrame = CFrame.lookAt(start - forward * (config.platformLength / 2 - 3), start),
		checkpointPosition = checkpoint,
		finishPosition = finish,
		courseStart = start - forward * (config.platformLength / 2),
		courseEnd = secondCenter + forward * (config.platformLength / 2),
	}
end

function CourseGenerator.build(seed: number, config: GapConfig, parent: Instance): Manifest
	local manifest = CourseGenerator.manifest(seed, config)
	local old = parent:FindFirstChild("GeneratedCourse")
	if old then
		old:Destroy()
	end

	local model = Instance.new("Model")
	model.Name = "GeneratedCourse"
	model:SetAttribute("CourseSeed", seed)
	model:SetAttribute("GeneratorVersion", manifest.generatorVersion)
	model:SetAttribute("GapStuds", manifest.gap)

	local heading = CFrame.Angles(0, math.rad(config.headingDegrees), 0)
	local forward = heading.LookVector
	local firstCenter = config.startPosition
	local secondCenter = firstCenter + forward * (config.platformLength + manifest.gap)
	local platformSize =
		Vector3.new(config.platformWidth, config.platformHeight, config.platformLength)
	local platformOffset = Vector3.new(0, -config.platformHeight / 2, 0)
	local first = makePart(
		"StartPlatform",
		platformSize,
		CFrame.new(firstCenter + platformOffset) * heading,
		Color3.fromRGB(75, 145, 210)
	)
	first.Parent = model
	local second = makePart(
		"LandingPlatform",
		platformSize,
		CFrame.new(secondCenter + platformOffset) * heading,
		Color3.fromRGB(80, 190, 110)
	)
	second.Parent = model

	local finish = makePart(
		"Finish",
		Vector3.new(config.platformWidth, 0.2, config.finishDepth),
		CFrame.new(manifest.finishPosition + Vector3.new(0, 0.1, 0)) * heading,
		Color3.fromRGB(245, 210, 60)
	)
	finish.CanCollide = false
	finish:SetAttribute("Finish", true)
	finish.Parent = model

	local killPlane = makePart(
		"KillPlane",
		Vector3.new(200, 1, 200),
		CFrame.new(config.startPosition.X, config.killPlaneY, config.startPosition.Z),
		Color3.fromRGB(190, 45, 45)
	)
	killPlane.Transparency = 0.35
	killPlane:SetAttribute("Hazard", true)
	killPlane.Parent = model
	model.Parent = parent
	return manifest
end

return CourseGenerator
