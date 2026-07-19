local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Curriculum = require(ReplicatedStorage.ObbyRL.CurriculumCourseConfig)
local Generator = require(ReplicatedStorage.ObbyRL.ProceduralCourseGenerator)

local teachingStages = { 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }
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
	combined.parameters.gap >= 5 and combined.parameters.gap <= 10,
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

print("[ObbyRL] M3 curriculum smoke passed variation stages")
