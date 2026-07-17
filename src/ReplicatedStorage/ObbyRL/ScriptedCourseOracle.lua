--!strict

export type Primitive = { stage: number, kind: string, target: Vector3, jump: boolean }

local Oracle = {}

function Oracle.plan(manifest: any, config: any): ({ Primitive }?, string?)
	local primitives: { Primitive } = {}
	local expectedEntry = manifest.courseStart
	for _, segment in manifest.segments do
		if (segment.entryPosition - expectedEntry).Magnitude > 0.01 then
			return nil, string.format("checkpoint chain breaks before stage %d", segment.index)
		end
		if (segment.checkpointPosition - segment.exitPosition).Magnitude > 0.01 then
			return nil, string.format("checkpoint is not on stage %d exit", segment.index)
		end
		if segment.kind == "gap" or segment.kind == "offset" then
			if math.abs(segment.parameters.height) > 3 then
				return nil, string.format("jump at stage %d exceeds oracle height", segment.index)
			end
			local airDistance =
				Vector2.new(segment.parameters.offset, segment.parameters.gap).Magnitude
			if airDistance > config.maxOracleJumpDistance then
				return nil, string.format("jump at stage %d exceeds oracle distance", segment.index)
			end
			table.insert(primitives, {
				stage = segment.index,
				kind = "jump_land",
				target = segment.exitPosition,
				jump = true,
			})
		elseif segment.kind == "beam" then
			if segment.parameters.width < config.minOracleBeamWidth then
				return nil, string.format("beam at stage %d is below oracle width", segment.index)
			end
			table.insert(primitives, {
				stage = segment.index,
				kind = "beam_traverse",
				target = segment.exitPosition,
				jump = false,
			})
		elseif segment.kind == "stairs" then
			if segment.parameters.rise > 2 or segment.parameters.run < 2 then
				return nil,
					string.format("stairs at stage %d exceed controller limits", segment.index)
			end
			for stair = 1, segment.parameters.count do
				table.insert(primitives, {
					stage = segment.index,
					kind = "stair_step",
					target = segment.entryPosition + Vector3.new(
						0,
						stair * segment.parameters.rise,
						-stair * segment.parameters.run
					),
					jump = true,
				})
			end
			table.insert(primitives, {
				stage = segment.index,
				kind = "stair_land",
				target = segment.exitPosition,
				jump = false,
			})
		else
			return nil, string.format("unknown segment kind at stage %d", segment.index)
		end
		expectedEntry = segment.exitPosition
	end
	if (manifest.courseEnd - expectedEntry).Magnitude > 0.01 then
		return nil, "course end does not match checkpoint chain"
	end
	if manifest.courseEnd.Y - manifest.courseStart.Y > config.maxCourseRise then
		return nil, "total course rise exceeds oracle limit"
	end
	return primitives, nil
end

return Oracle
