--!strict

local Base = require(script.Parent.ProceduralCourseConfig)

local Curriculum = {}

local function mutableBase(): any
	return table.clone(Base)
end

function Curriculum.forStage(stage: number): any
	assert(stage >= 1 and stage <= 4, "curriculum stage must be 1..4")
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
		config.gapMin = 3.5
		config.gapMax = 3.5
		config.jumpHeightMin = 0
		config.jumpHeightMax = 0
	elseif stage == 3 then
		config.stageCount = 1
		config.segmentKinds = { "gap", "offset" }
		config.jumpHeightMin = -3
		config.jumpHeightMax = 3
	else
		config.jumpHeightMin = 0
		config.jumpHeightMax = 0
	end
	config.curriculumStage = stage
	return table.freeze(config)
end

return Curriculum
