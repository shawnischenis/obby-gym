local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Config = require(ReplicatedStorage.ObbyRL.ProceduralCourseConfig)
local Generator = require(ReplicatedStorage.ObbyRL.ProceduralCourseGenerator)
local Oracle = require(ReplicatedStorage.ObbyRL.ScriptedCourseOracle)

local kinds = { gap = 0, offset = 0, beam = 0, stairs = 0 }
local rejectedCandidates = 0
local resampledSeeds = 0
local primitiveKinds: { [string]: number } = {}
for seed = 0, 1999 do
	local first = Generator.manifest(seed, Config)
	local replay = Generator.manifest(seed, Config)
	assert(first.signature == replay.signature, `seed {seed} did not replay deterministically`)
	assert(
		first.generationSeed == replay.generationSeed,
		`seed {seed} replay chose another candidate`
	)
	assert(#first.rejections == #replay.rejections, `seed {seed} replay changed rejection count`)
	local valid, reason = Generator.validate(first, Config)
	assert(valid, `seed {seed} invalid: {tostring(reason)}`)
	local feasible, oracleReason = Generator.oracle(first, Config)
	assert(feasible, `seed {seed} failed oracle: {tostring(oracleReason)}`)
	local plan, planReason = Oracle.plan(first, Config)
	assert(plan ~= nil, `seed {seed} did not produce a plan: {tostring(planReason)}`)
	for _, primitive in plan do
		primitiveKinds[primitive.kind] = (primitiveKinds[primitive.kind] or 0) + 1
	end
	rejectedCandidates += #first.rejections
	if first.generationAttempt > 0 then
		resampledSeeds += 1
		assert(#first.rejections == first.generationAttempt, "missing rejection audit entries")
		assert(string.find(first.rejections[1], "reason=") ~= nil, "rejection reason missing")
	end
	for _, segment in first.segments do
		kinds[segment.kind] += 1
	end
end

assert(resampledSeeds > 0, "validation sample did not exercise deterministic resampling")
assert(rejectedCandidates >= resampledSeeds, "rejected candidate count is inconsistent")
for _, kind in { "jump_land", "beam_traverse", "stair_step", "stair_land" } do
	assert((primitiveKinds[kind] or 0) > 0, `scripted primitive {kind} was never planned`)
end

for kind, count in kinds do
	assert(count > 0, `segment kind {kind} was never generated`)
end

local built = Generator.build(20260716, Config, workspace)
assert(#built.segments == Config.stageCount, "built course stage count mismatch")
local model = workspace:FindFirstChild("GeneratedCourseV2")
assert(model ~= nil, "v2 course model missing")
assert(model:FindFirstChild("KillPlane") ~= nil, "kill plane missing")
assert(model:FindFirstChild("Finish") ~= nil, "finish trigger missing")
assert(
	model:GetAttribute("GenerationSeed") == built.generationSeed,
	"generation seed attribute missing"
)
assert(
	model:GetAttribute("RejectedCandidates") == #built.rejections,
	"rejection count attribute missing"
)
for index = 1, Config.stageCount do
	assert(
		model:FindFirstChild(string.format("Checkpoint_%02d", index)) ~= nil,
		"checkpoint missing"
	)
end
print(
	string.format(
		"[ObbyRL] M2 properties passed seeds=2000 gap=%d offset=%d beam=%d stairs=%d resampled=%d rejected=%d",
		kinds.gap,
		kinds.offset,
		kinds.beam,
		kinds.stairs,
		resampledSeeds,
		rejectedCandidates
	)
)
