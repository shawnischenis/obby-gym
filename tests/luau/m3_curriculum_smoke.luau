local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Curriculum = require(ReplicatedStorage.ObbyRL.CurriculumCourseConfig)
local Generator = require(ReplicatedStorage.ObbyRL.ProceduralCourseGenerator)

for stage = 1, 4 do
	local config = Curriculum.forStage(stage)
	local first = Generator.manifest(123, config)
	local replay = Generator.manifest(123, config)
	assert(first.signature == replay.signature, `stage {stage} was not deterministic`)
	local valid, reason = Generator.validate(first, config)
	assert(valid, `stage {stage} was invalid: {tostring(reason)}`)
	local feasible, oracleReason = Generator.oracle(first, config)
	assert(feasible, `stage {stage} failed oracle: {tostring(oracleReason)}`)
	if stage < 4 then
		assert(#first.segments == 1, `stage {stage} should contain one teaching segment`)
	else
		assert(#first.segments == 8, "stage 4 should contain the full course")
	end
end

local flat = Generator.manifest(0, Curriculum.forStage(1))
assert(flat.segments[1].kind == "beam", "stage 1 should be flat traversal")
assert(flat.segments[1].parameters.width == 200, "stage 1 runway should be boundary-free")
assert(flat.parts[2].name == "StartSafetyApron", "stage 1 should protect backward exploration")
local fixedJump = Generator.manifest(0, Curriculum.forStage(2))
assert(fixedJump.segments[1].kind == "gap", "stage 2 should be a fixed gap")
assert(fixedJump.segments[1].parameters.gap == 3.5, "stage 2 gap changed")

print("[ObbyRL] M3 curriculum smoke passed stages=4")
