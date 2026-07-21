local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Curriculum = require(ReplicatedStorage.ObbyRL.CurriculumCourseConfig)
local Generator = require(ReplicatedStorage.ObbyRL.ProceduralCourseGenerator)

local teachingStages = { 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }
for _, stage in teachingStages do
	local config = Curriculum.forStage(stage)
	local first = Generator.manifest(123, config)
	local replay = Generator.manifest(123, config)
	assert(first.signature == replay.signature, `stage {stage} was not deterministic`)
	local valid, reason = Generator.validate(first, config)
	assert(valid, `stage {stage} was invalid: {tostring(reason)}`)
	local feasible, oracleReason = Generator.oracle(first, config)
	assert(feasible, `stage {stage} failed oracle: {tostring(oracleReason)}`)
	assert(#first.segments == 1, `stage {stage} should contain one teaching segment`)
end
local full = Generator.manifest(123, Curriculum.forStage(4))
assert(#full.segments == 8, "stage 4 should remain the full legacy course")

local flat = Generator.manifest(0, Curriculum.forStage(1))
assert(flat.segments[1].kind == "beam", "stage 1 should be flat traversal")
assert(flat.segments[1].parameters.width == 200, "stage 1 runway should be boundary-free")
assert(flat.parts[2].name == "StartSafetyApron", "stage 1 should protect backward exploration")
local fixedJump = Generator.manifest(0, Curriculum.forStage(2))
assert(fixedJump.segments[1].kind == "gap", "stage 2 should be a fixed gap")
assert(fixedJump.segments[1].parameters.gap == 7, "stage 2 must require a real jump")

local narrowGap = Generator.manifest(42, Curriculum.forStage(3)).segments[1]
assert(
	narrowGap.parameters.gap >= 6.5 and narrowGap.parameters.gap <= 7.5,
	"stage 3 gap must remain in its narrow range"
)
local highToLow = Generator.manifest(42, Curriculum.forStage(7)).segments[1]
assert(highToLow.parameters.height < 0, "stage 7 must land below takeoff")
local lowToHigh = Generator.manifest(42, Curriculum.forStage(8)).segments[1]
assert(lowToHigh.parameters.height > 0, "stage 8 must land above takeoff")
local angled = Generator.manifest(42, Curriculum.forStage(9)).segments[1]
assert(math.abs(angled.parameters.angle) <= 8, "stage 9 angle must be small")
local combined = Generator.manifest(42, Curriculum.forStage(11)).segments[1]
assert(
	combined.parameters.gap >= 5 and combined.parameters.gap <= 9,
	"stage 11 gap must remain in its combined range"
)
assert(math.abs(combined.parameters.angle) <= 18, "stage 11 angle must remain bounded")
local intermediate = Generator.manifest(42, Curriculum.forStage(12)).segments[1]
assert(
	intermediate.parameters.gap >= 5 and intermediate.parameters.gap <= 9,
	"stage 12 gap must remain in its intermediate range"
)
local upperExpansion = Generator.manifest(42, Curriculum.forStage(13)).segments[1]
assert(
	upperExpansion.parameters.gap >= 6 and upperExpansion.parameters.gap <= 9,
	"stage 13 gap must remain in its upper-expansion range"
)
local lowerExpansion = Generator.manifest(42, Curriculum.forStage(14)).segments[1]
assert(
	lowerExpansion.parameters.gap >= 5 and lowerExpansion.parameters.gap <= 8.5,
	"stage 14 gap must remain in its lower-expansion range"
)
local boundaryReplay = Generator.manifest(42, Curriculum.forStage(15)).segments[1]
assert(
	boundaryReplay.parameters.gap >= 5 and boundaryReplay.parameters.gap <= 9,
	"stage 15 gap must remain inside the boundary-replay union"
)
local combinedLongJump = Generator.manifest(42, Curriculum.forStage(16)).segments[1]
assert(
	combinedLongJump.parameters.gap >= 9 and combinedLongJump.parameters.gap <= 10,
	"stage 16 gap must isolate the combined long-jump range"
)
assert(math.abs(combinedLongJump.parameters.height) <= 3, "stage 16 height must be bounded")
assert(math.abs(combinedLongJump.parameters.angle) <= 18, "stage 16 angle must be bounded")
local flatLongJump = Generator.manifest(42, Curriculum.forStage(17)).segments[1]
assert(
	flatLongJump.parameters.gap >= 9 and flatLongJump.parameters.gap <= 10,
	"stage 17 gap must isolate the flat long-jump range"
)
assert(flatLongJump.parameters.height == 0, "stage 17 must keep the landing level")
assert(flatLongJump.parameters.angle == 0, "stage 17 must keep the approach straight")

local narrowBeam = Generator.manifest(42, Curriculum.forStage(18)).segments[1]
assert(narrowBeam.kind == "beam", "stage 18 must isolate beams")
assert(narrowBeam.parameters.width >= 3, "stage 18 beam must use a learnable width")
local stairs = Generator.manifest(42, Curriculum.forStage(19)).segments[1]
assert(stairs.kind == "stairs", "stage 19 must isolate stairs")
for stage, expectedCount in { [20] = 2, [21] = 4, [22] = 8 } do
	local mixed = Generator.manifest(42, Curriculum.forStage(stage))
	assert(#mixed.segments == expectedCount, `stage {stage} segment count mismatch`)
	local valid, reason = Generator.validate(mixed, Curriculum.forStage(stage))
	assert(valid, `stage {stage} was invalid: {tostring(reason)}`)
end

print("[ObbyRL] M3 curriculum smoke passed variation stages")
