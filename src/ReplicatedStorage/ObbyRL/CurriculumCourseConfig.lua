--!strict

local Base = require(script.Parent.ProceduralCourseConfig)

local Curriculum = {}

local function mutableBase(): any
	return table.clone(Base)
end

function Curriculum.forStage(stage: number): any
	assert(stage >= 1 and stage <= 14, "curriculum stage must be 1..14")
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
			config.gapMax = 10
			config.maxJumpGap = 10
			config.maxOracleJumpDistance = 10
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
			config.gapMax = 10
			config.jumpHeightMin = -3
			config.jumpHeightMax = 3
			config.approachAngleMin = -18
			config.approachAngleMax = 18
			config.maxJumpGap = 10
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
		else
			-- Teach the lower boundary while holding the mastered 8.5 upper limit.
			config.gapMin = 5
			config.gapMax = 8.5
			config.maxJumpGap = 8.5
		end
	end
	config.curriculumStage = stage
	return table.freeze(config)
end

return Curriculum
