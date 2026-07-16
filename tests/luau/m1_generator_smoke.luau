local ReplicatedStorage = game:GetService("ReplicatedStorage")
local CourseGenerator = require(ReplicatedStorage.ObbyRL.CourseGenerator)

local config = {
	startPosition = Vector3.new(0, 4, 0),
	headingDegrees = 0,
	platformWidth = 12,
	platformLength = 14,
	platformHeight = 1,
	gapMin = 4,
	gapMax = 8,
	finishDepth = 4,
	killPlaneY = -25,
}

local first = CourseGenerator.manifest(12345, config)
local second = CourseGenerator.manifest(12345, config)
local other = CourseGenerator.manifest(12346, config)
assert(first.gap == second.gap, "same seed must reproduce the same gap")
assert(first.spawnCFrame == second.spawnCFrame, "same seed must reproduce the same spawn")
assert(first.gap ~= other.gap, "different smoke-test seeds should produce different gaps")

local built = CourseGenerator.build(12345, config, workspace)
assert(workspace:FindFirstChild("GeneratedCourse") ~= nil, "course was not built")
assert(built.generatorVersion == "0.1.0", "unexpected generator version")
print(string.format("[ObbyRL] M1 generator smoke passed gap=%.4f", built.gap))
