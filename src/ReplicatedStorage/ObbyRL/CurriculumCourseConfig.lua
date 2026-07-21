--!strict

local Base = require(script.Parent.ProceduralCourseConfig)

local Curriculum = {}

local function mutableBase(): any
	return table.clone(Base)
end

function Curriculum.forStage(stage: number): any
	assert(stage >= 1 and stage <= 22, "curriculum stage must be 1..22")
	local config = mutableBase()
	if stage == 1 then
		config.stageCount = 1
		config.segmentKinds = { "beam" }
		config.platformWidth = 200
		config.beamWidthMin = 200
		config.beamWidthMax = 200
		config.beamLengthMin = 18
		config.beamLengthMax = 18
		config.startSafetyDepth = 100
	elseif stage == 2 then
		config.stageCount = 1
		config.segmentKinds = { "gap" }
		-- Seven studs is inside the validated jump envelope but cannot be crossed
		-- through ordinary ground contact at running speed like the old 3.5-stud gap.
		config.gapMin = 7
		config.gapMax = 7
		config.jumpHeightMin = 0
		config.jumpHeightMax = 0
	elseif stage == 3 then
		config.stageCount = 1
		config.segmentKinds = { "gap" }
		config.gapMin = 6.5
		config.gapMax = 7.5
		config.jumpHeightMin = 0
		config.jumpHeightMax = 0
	elseif stage == 4 then
		-- Stage 4 remains the legacy full course for saved-run compatibility.
		config.jumpHeightMin = 0
		config.jumpHeightMax = 0
	else
		config.stageCount = 1
		config.segmentKinds = { "gap" }
		config.jumpHeightMin = 0
		config.jumpHeightMax = 0
		if stage == 5 then
			config.gapMin = 6
			config.gapMax = 8.5
			config.maxJumpGap = 8.5
		elseif stage == 6 then
			config.gapMin = 5
			-- Nine studs is the core upper bound. The unreliable 9-10 boundary is
			-- retained only in optional stress-test stages 16 and 17.
			config.gapMax = 9
			config.maxJumpGap = 9
			config.maxOracleJumpDistance = 9
		elseif stage == 7 then
			config.gapMin = 6.5
			config.gapMax = 7.5
			config.jumpHeightMin = -3
			config.jumpHeightMax = -0.5
		elseif stage == 8 then
			config.gapMin = 6.5
			config.gapMax = 7.5
			config.jumpHeightMin = 0.5
			config.jumpHeightMax = 3
		elseif stage == 9 then
			config.segmentKinds = { "offset" }
			config.gapMin = 6.5
			config.gapMax = 7.5
			config.approachAngleMin = -8
			config.approachAngleMax = 8
		elseif stage == 10 then
			config.segmentKinds = { "offset" }
			config.gapMin = 6
			config.gapMax = 8.5
			config.maxJumpGap = 8.5
			config.approachAngleMin = -18
			config.approachAngleMax = 18
			config.maxLateralOffset = 7
			config.maxOracleJumpDistance = 11
		elseif stage == 11 then
			config.segmentKinds = { "offset" }
			config.gapMin = 5
			config.gapMax = 9
			config.jumpHeightMin = -3
			config.jumpHeightMax = 3
			config.approachAngleMin = -18
			config.approachAngleMax = 18
			config.maxJumpGap = 9
			config.maxLateralOffset = 8
			config.maxOracleJumpDistance = 13
		elseif stage == 12 then
			-- Intermediate expansion before the full 5-10 stud Stage 6 range.
			config.gapMin = 5
			config.gapMax = 9
			config.maxJumpGap = 9
		elseif stage == 13 then
			-- Expand the difficult upper boundary before lowering the minimum gap.
			config.gapMin = 6
			config.gapMax = 9
			config.maxJumpGap = 9
		elseif stage == 14 then
			-- Teach the lower boundary while holding the mastered 8.5 upper limit.
			config.gapMin = 5
			config.gapMax = 8.5
			config.maxJumpGap = 8.5
		elseif stage == 15 then
			-- Rehearse both mastered boundary tasks before uniform 5-9 sampling.
			config.gapRanges = { { 5, 8.5 }, { 6, 9 } }
			config.gapMin = 5
			config.gapMax = 9
			config.maxJumpGap = 9
		elseif stage == 16 then
			-- Optional stress benchmark, excluded from the core curriculum.
			config.segmentKinds = { "offset" }
			config.gapMin = 9
			config.gapMax = 10
			config.jumpHeightMin = -3
			config.jumpHeightMax = 3
			config.approachAngleMin = -18
			config.approachAngleMax = 18
			config.maxJumpGap = 10
			config.maxLateralOffset = 8
			config.maxOracleJumpDistance = 13
		elseif stage == 17 then
			-- Optional flat stress benchmark, excluded from the core curriculum.
			config.gapMin = 9
			config.gapMax = 10
			config.maxJumpGap = 10
			config.maxOracleJumpDistance = 10
		elseif stage == 18 then
			-- Teach realistic narrow-beam traversal before mixed courses.
			config.segmentKinds = { "beam" }
			config.beamWidthMin = 3
			config.beamWidthMax = 6
			config.beamLengthMin = 8
			config.beamLengthMax = 16
		elseif stage == 19 then
			-- Teach stair ascent/descent as an isolated obstacle family.
			config.segmentKinds = { "stairs" }
		elseif stage == 20 then
			-- First checkpoint-transition curriculum.
			config.stageCount = 2
			config.segmentKinds = { "gap", "offset", "beam", "stairs" }
		elseif stage == 21 then
			config.stageCount = 4
			config.segmentKinds = { "gap", "offset", "beam", "stairs" }
		else
			-- Eight-segment training alias for the final Stage 4 benchmark.
			config.stageCount = 8
			config.segmentKinds = { "gap", "offset", "beam", "stairs" }
		end
	end
	config.curriculumStage = stage
	return table.freeze(config)
end

return Curriculum
