--!strict

local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")

local toolbar = plugin:CreateToolbar("ObbyRL")
local toggle = toolbar:CreateButton("Bridge", "Toggle the local Python RL bridge", "")
toggle.ClickableWhenViewportHidden = true

local enabled = true
local generation = 0
local DEFAULT_ACTION_REPEAT_TICKS = 3
local RESET_SETTLE_TIMEOUT = 1
local RESET_STABLE_TICKS = 3
local VECTOR_LANE_SPACING = 400
local MAX_ACTION_HOLD_SECONDS = 0.25

local function vectorLaneOrigin(
	index: number,
	laneCount: number,
	recordingView: boolean,
	recordingCamera: string
): Vector3
	if recordingView then
		if laneCount == 2 and recordingCamera == "side" then
			-- Separate the courses along their travel axis so an orthogonal side
			-- camera sees two distinct jump profiles instead of overlapping lanes.
			return Vector3.new(0, 0, (index - 1) * 55)
		end
		local columns = if laneCount <= 4 then 2 else 4
		local xSpacing = if laneCount == 2 then 44 elseif laneCount <= 4 then 60 else 100
		local zSpacing = if laneCount <= 4 then 80 else 140
		local column = (index - 1) % columns
		local row = math.floor((index - 1) / columns)
		return Vector3.new(column * xSpacing, 0, row * zSpacing)
	end
	return Vector3.new((index - 1) * VECTOR_LANE_SPACING, 0, 0)
end

local function hideRecordingGeometry(root: Instance)
	for _, descendant in root:GetDescendants() do
		if descendant:IsA("BasePart") then
			descendant.Transparency = 1
			descendant.CanCollide = false
		end
	end
end

local function hideRecordingVisuals(root: Instance)
	for _, descendant in root:GetDescendants() do
		if descendant:IsA("BasePart") then
			descendant.Transparency = 1
		end
	end
end

local function settleCharacter(AgentHarness: any, state: any): number
	local started = os.clock()
	local stableTicks = 0
	repeat
		AgentHarness.stop(state)
		RunService.Heartbeat:Wait()
		local grounded = state.humanoid.FloorMaterial ~= Enum.Material.Air
		local nearlyStill = state.root.AssemblyLinearVelocity.Magnitude < 2
		stableTicks = if grounded and nearlyStill then stableTicks + 1 else 0
	until stableTicks >= RESET_STABLE_TICKS or os.clock() - started >= RESET_SETTLE_TIMEOUT
	AgentHarness.stop(state)
	state.root.AssemblyLinearVelocity = Vector3.zero
	state.root.AssemblyAngularVelocity = Vector3.zero
	AgentHarness.neutralizePose(state)
	return os.clock() - started
end

local function settleCharacters(AgentHarness: any, states: { any }): number
	local started = os.clock()
	local stableTicks = table.create(#states, 0)
	local allStable = false
	repeat
		allStable = true
		for index, state in states do
			AgentHarness.stop(state)
			local grounded = state.humanoid.FloorMaterial ~= Enum.Material.Air
			local nearlyStill = state.root.AssemblyLinearVelocity.Magnitude < 2
			stableTicks[index] = if grounded and nearlyStill then stableTicks[index] + 1 else 0
			if stableTicks[index] < RESET_STABLE_TICKS then
				allStable = false
			end
		end
		if not allStable then
			RunService.Heartbeat:Wait()
		end
	until allStable or os.clock() - started >= RESET_SETTLE_TIMEOUT
	for _, state in states do
		AgentHarness.stop(state)
		state.root.AssemblyLinearVelocity = Vector3.zero
		state.root.AssemblyAngularVelocity = Vector3.zero
		AgentHarness.neutralizePose(state)
	end
	return os.clock() - started
end

local function exchange(payload: any): (boolean, any)
	local success, result = pcall(function()
		return HttpService:RequestAsync({
			Url = "http://127.0.0.1:8765/exchange",
			Method = "POST",
			Headers = { ["Content-Type"] = "application/json" },
			Body = HttpService:JSONEncode(payload),
			Timeout = 5,
		})
	end)
	if not success then
		return false, result
	end
	if not result.Success then
		return false, string.format("HTTP %s: %s", result.StatusCode, result.StatusMessage)
	end
	return true, HttpService:JSONDecode(result.Body)
end

local function selectLandingSnapshot(snapshots: any, selectorSeed: number): any?
	if snapshots == nil then
		return nil
	end
	local available = {}
	for checkpointIndex, snapshot in snapshots do
		if snapshot ~= nil then
			table.insert(available, { checkpointIndex = checkpointIndex, snapshot = snapshot })
		end
	end
	table.sort(available, function(a, b)
		return a.checkpointIndex < b.checkpointIndex
	end)
	if #available == 0 then
		return nil
	end
	local selected = (math.abs(selectorSeed) % #available) + 1
	return available[selected].snapshot
end

local function landingSnapshotCount(snapshots: any): number
	local count = 0
	for _, snapshot in snapshots or {} do
		if snapshot ~= nil then
			count += 1
		end
	end
	return count
end

local function waitForRuntime(): (any, any, any, any, Model)
	if not RunService:IsRunning() then
		return nil
	end
	local folder = ReplicatedStorage:FindFirstChild("ObbyRL")
	local player = Players:GetPlayers()[1]
	local character = if player then player.Character else nil
	if not folder or not character then
		return nil
	end
	local AgentHarness = require(folder:WaitForChild("AgentHarness"))
	local CourseGenerator = require(folder:WaitForChild("ProceduralCourseGenerator"))
	local CurriculumConfig = require(folder:WaitForChild("CurriculumCourseConfig"))
	local CourseConfig = CurriculumConfig.forStage(4)
	local manifest = CourseGenerator.manifest(0, CourseConfig)
	local state = AgentHarness.new(character, manifest)
	return AgentHarness, CourseGenerator, CurriculumConfig, state, character
end

local function resultPayload(
	requestId: string,
	episodeId: string,
	stepId: number,
	AgentHarness: any,
	state: any,
	reward: number,
	terminated: boolean,
	seed: number,
	commandStarted: number,
	hazardRecovered: boolean?,
	rewardComponents: any?,
	transition: any?
): any
	local course = workspace:FindFirstChild("GeneratedCourseV2")
	return {
		protocol_version = "0.1.0",
		message_type = "step_result",
		ack_request_id = requestId,
		episode_id = episodeId,
		step_id = stepId,
		observation = { schema = "obby-structured-v1", values = AgentHarness.observe(state) },
		privileged_observation = {
			schema = "obby-privileged-v1",
			values = AgentHarness.observePrivileged(state),
		},
		reward = reward,
		terminated = terminated,
		truncated = false,
		info = {
			default_controls_disabled = if Players:GetPlayers()[1]
				then Players:GetPlayers()[1]:GetAttribute("ObbyRLDefaultControlsDisabled") == true
				else false,
			humanoid_auto_rotate = state.humanoid.AutoRotate,
			course_seed = seed,
			curriculum_stage = if course then course:GetAttribute("CurriculumStage") else 4,
			course_signature = if course then course:GetAttribute("Signature") else nil,
			course_instances = if course then #course:GetDescendants() else 0,
			checkpoint_index = state.checkpointIndex,
			checkpoint_count = #state.checkpoints,
			hazard_recovered = hazardRecovered == true,
			reward_components = rewardComponents or {},
			transition = transition or {},
			studio_command_seconds = os.clock() - commandStarted,
		},
	}
end

local function cloneAgent(character: Model, parent: Instance, index: number): Model
	character.Archivable = true
	local clone = character:Clone()
	clone.Name = string.format("Agent_%02d", index)
	for _, descendant in clone:GetDescendants() do
		if descendant:IsA("Animation") and string.find(string.lower(descendant.Name), "jump") then
			clone:SetAttribute("RecordingJumpAnimationId", descendant.AnimationId)
			break
		end
	end
	for _, descendant in clone:GetDescendants() do
		if descendant:IsA("Script") or descendant:IsA("LocalScript") then
			descendant:Destroy()
		elseif descendant:IsA("Motor6D") then
			-- The player is a rig template, not the simulated agent. Cloning while
			-- its Animate script is mid-pose otherwise bakes raised arms or a jump
			-- pose into later recording episodes after the scripts are removed.
			descendant.Transform = CFrame.identity
		end
	end
	clone.Parent = parent
	local root = clone:FindFirstChild("HumanoidRootPart")
	if root and root:IsA("BasePart") then
		root.Anchored = false
	end
	local humanoid = clone:FindFirstChildOfClass("Humanoid")
	if humanoid then
		humanoid.PlatformStand = false
		humanoid.Sit = false
		humanoid.Jump = false
		local animator = humanoid:FindFirstChildOfClass("Animator")
		if animator then
			for _, track in animator:GetPlayingAnimationTracks() do
				track:Stop(0)
			end
		end
	end
	return clone
end

local function addRecordingEffects(character: Model)
	local root = character:FindFirstChild("HumanoidRootPart")
	if not root or not root:IsA("BasePart") then
		return
	end
	local upper = Instance.new("Attachment")
	upper.Position = Vector3.new(0, 1.5, 0)
	upper.Parent = root
	local lower = Instance.new("Attachment")
	lower.Position = Vector3.new(0, -1.5, 0)
	lower.Parent = root
	local trail = Instance.new("Trail")
	trail.Name = "RecordingMotionTrail"
	trail.Attachment0 = upper
	trail.Attachment1 = lower
	trail.Color = ColorSequence.new(Color3.fromRGB(255, 210, 60))
	trail.Transparency = NumberSequence.new(0.15, 1)
	trail.Lifetime = 0.45
	trail.MinLength = 0.1
	trail.FaceCamera = true
	trail.Parent = root

	local humanoid = character:FindFirstChildOfClass("Humanoid")
	local animationId = character:GetAttribute("RecordingJumpAnimationId")
	if not humanoid or not animationId or animationId == "" then
		return
	end
	local animator = humanoid:FindFirstChildOfClass("Animator")
	if not animator then
		animator = Instance.new("Animator")
		animator.Parent = humanoid
	end
	local animation = Instance.new("Animation")
	animation.Name = "RecordingJumpAnimation"
	animation.AnimationId = animationId
	animation.Parent = character
	local success, jumpTrack = pcall(function()
		return animator:LoadAnimation(animation)
	end)
	if not success or not jumpTrack then
		warn("[ObbyRL] unable to load recording jump animation")
		return
	end
	jumpTrack.Priority = Enum.AnimationPriority.Action
	jumpTrack.Looped = false
	humanoid.StateChanged:Connect(function(_, newState: Enum.HumanoidStateType)
		if newState == Enum.HumanoidStateType.Jumping then
			jumpTrack:Play(0.05, 1, 1)
		elseif newState == Enum.HumanoidStateType.Landed
			or newState == Enum.HumanoidStateType.Running
		then
			jumpTrack:Stop(0.1)
		end
	end)
end

local function vectorLaneResult(
	AgentHarness: any,
	lane: any,
	action: any,
	elapsed: number,
	heldActions: { [any]: any }
): any
	local observation = AgentHarness.observe(lane.state)
	local progress = observation[13]
	local checkpointDistance = (lane.state.root.Position - lane.state.checkpoint).Magnitude
	local checkpointReward = (lane.lastCheckpointDistance - checkpointDistance) / 20
	local progressReward = (progress - lane.lastProgress) * 0.1
	local reachedCheckpoint = AgentHarness.advanceCheckpoint(lane.state)
	if reachedCheckpoint then
		-- advanceCheckpoint changes both the target and its geometry. Return that
		-- new contract immediately instead of leaking the previous checkpoint for
		-- one policy decision.
		observation = AgentHarness.observe(lane.state)
		progress = observation[13]
		if lane.state.checkpointIndex < #lane.state.checkpoints then
			lane.landingSnapshots = lane.landingSnapshots or {}
			local snapshot = AgentHarness.captureLandingSnapshot(
				lane.state,
				action.jump_cooldown_remaining or 0
			)
			snapshot.seed = lane.seed
			lane.landingSnapshots[lane.state.checkpointIndex] = snapshot
		end
	end
	local fell = lane.state.root.Position.Y < -17
	local terminated = reachedCheckpoint
		and lane.state.checkpointIndex >= #lane.state.checkpoints
	local checkpointBonus = if reachedCheckpoint then 0.1 else 0
	local finishReward = if terminated then 1 else 0
	local hazardPenalty = if fell then -1 else 0
	local reward = checkpointReward
		+ progressReward
		+ checkpointBonus
		+ finishReward
		+ hazardPenalty
		- 0.001
	if reachedCheckpoint then
		checkpointDistance = (lane.state.root.Position - lane.state.checkpoint).Magnitude
	end
	if fell and not terminated then
		if lane.recordingVisible then
			task.wait(0.5)
		end
		heldActions[lane.state] = nil
		lane.actionStartedAt = nil
		AgentHarness.stop(lane.state)
		AgentHarness.recover(lane.state)
		observation = AgentHarness.observe(lane.state)
		progress = observation[13]
		checkpointDistance = (lane.state.root.Position - lane.state.checkpoint).Magnitude
	end
	lane.lastProgress = progress
	lane.lastCheckpointDistance = checkpointDistance
	if terminated then
		heldActions[lane.state] = nil
		lane.actionStartedAt = nil
		AgentHarness.stop(lane.state)
	end
	return {
		observation = { schema = "obby-structured-v1", values = observation },
		privileged_observation = {
			schema = "obby-privileged-v1",
			values = AgentHarness.observePrivileged(lane.state),
		},
		reward = reward,
		terminated = terminated,
		truncated = false,
		info = {
			lane_index = lane.index,
			course_seed = lane.seed,
			checkpoint_index = lane.state.checkpointIndex,
			hazard_recovered = fell,
			hold_seconds = elapsed,
			action_lease_seconds = MAX_ACTION_HOLD_SECONDS,
			previous_action_total_seconds = lane.previousActionTotalSeconds or 0,
			reward_components = {
				checkpoint = checkpointReward,
				progress = progressReward,
				checkpoint_bonus = checkpointBonus,
				finish = finishReward,
				hazard = hazardPenalty,
				time = -0.001,
			},
			action = action,
		},
	}
end

local function runWorker(myGeneration: number)
	print("[ObbyRL Plugin] waiting for playtest and Python broker")
	while enabled and generation == myGeneration do
		local runtime = table.pack(waitForRuntime())
		if runtime.n == 0 or runtime[1] == nil then
			task.wait(0.25)
			continue
		end
		local AgentHarness, CourseGenerator, CurriculumConfig, state, character =
			runtime[1], runtime[2], runtime[3], runtime[4], runtime[5]
		local outgoing: any = { protocol_version = "0.1.0", message_type = "worker_ready" }
		local lastProgress = 0
		local lastCheckpointDistance = 0
		local seed = 0
		local curriculumStage = 4
		local actionRepeatTicks = DEFAULT_ACTION_REPEAT_TICKS
		local vectorLanes: { any } = {}
		local heldActions: { [any]: any } = {}
		local lastActionStartedAt: number? = nil
		local holdConnection = RunService.Heartbeat:Connect(function()
			for heldState, held in heldActions do
				if heldState.character.Parent then
					if os.clock() < held.expiresAt then
						AgentHarness.refreshMovement(heldState, held.action)
					else
						AgentHarness.stop(heldState)
						heldActions[heldState] = nil
					end
				else
					heldActions[heldState] = nil
				end
			end
		end)
		local function clearHeldActions()
			for heldState in heldActions do
				AgentHarness.stop(heldState)
			end
			table.clear(heldActions)
			lastActionStartedAt = nil
		end
		print("[ObbyRL Plugin] playtest found; connecting to Python")

		while
			enabled
			and generation == myGeneration
			and RunService:IsRunning()
			and character.Parent
		do
			local ok, command = exchange(outgoing)
			if not ok then
				task.wait(0.5)
				continue
			end
			if command.message_type == "vector_reset_command" then
				local commandStarted = os.clock()
				clearHeldActions()
				local previousSnapshots = {}
				for index, lane in vectorLanes do
					previousSnapshots[index] = lane.landingSnapshots
				end
				local vectorRoot = workspace:FindFirstChild("ObbyRLVectorLanes")
				if vectorRoot then
					vectorRoot:Destroy()
				end
				vectorRoot = Instance.new("Folder")
				vectorRoot.Name = "ObbyRLVectorLanes"
				vectorRoot.Parent = workspace
				vectorLanes = {}
				curriculumStage = command.curriculum_stage or 4
				local recordingView = command.recording_view == true
				local recordingCamera = command.recording_camera or "auto"
				local recordingVisibleLane = command.recording_visible_lane or 0
				local postLandingMask = command.post_landing_mask or {}
				if recordingCamera == "auto" then
					recordingCamera = if #command.course_seeds == 2 then "side" else "parallel"
				end
				workspace:SetAttribute("ObbyRLRecordingView", recordingView)
				workspace:SetAttribute("ObbyRLRecordingLaneCount", #command.course_seeds)
				workspace:SetAttribute("ObbyRLRecordingCamera", recordingCamera)
				-- Lane 1 is generated at the origin, where Main.server also creates the
				-- standalone base course. Leaving it collidable superimposes unrelated
				-- platforms under lane 1 and contaminates both training and evaluation.
				local baseCourse = workspace:FindFirstChild("GeneratedCourseV2")
				if baseCourse then
					hideRecordingGeometry(baseCourse)
				end
				actionRepeatTicks = math.clamp(command.action_repeat_ticks or 3, 1, 6)
				local CourseConfig = CurriculumConfig.forStage(curriculumStage)
				for index, requestedSeed in command.course_seeds do
					local requestedPostLanding = postLandingMask[index] == true
					local landingSnapshot = if requestedPostLanding
						then selectLandingSnapshot(previousSnapshots[index], requestedSeed)
						else nil
					local laneSeed = if landingSnapshot then landingSnapshot.seed else requestedSeed
					local laneFolder = Instance.new("Folder")
					laneFolder.Name = string.format("Lane_%02d", index)
					laneFolder.Parent = vectorRoot
					local origin = vectorLaneOrigin(
						index,
						#command.course_seeds,
						recordingView,
						recordingCamera
					)
					local manifest =
						CourseGenerator.build(laneSeed, CourseConfig, laneFolder, origin)
					if recordingView then
						local laneCourse = laneFolder:FindFirstChild("GeneratedCourseV2")
						local killPlane = if laneCourse then laneCourse:FindFirstChild("KillPlane") else nil
						if killPlane and killPlane:IsA("BasePart") then
							killPlane.Transparency = 1
						end
					end
					local agent = cloneAgent(character, laneFolder, index)
					local laneState = AgentHarness.new(agent, manifest)
					AgentHarness.reset(laneState)
					local postLanding = landingSnapshot ~= nil
					local recordingVisible = recordingVisibleLane == index
					if recordingVisible then
						addRecordingEffects(agent)
					end
					if recordingVisibleLane > 0 and index ~= recordingVisibleLane then
						hideRecordingVisuals(laneFolder)
					end
					table.insert(vectorLanes, {
						index = index,
						seed = laneSeed,
						state = laneState,
						lastProgress = 0,
						lastCheckpointDistance = (laneState.root.Position - laneState.checkpoint).Magnitude,
						actionStartedAt = nil,
						previousActionTotalSeconds = 0,
						recordingVisible = recordingVisible,
						postLanding = postLanding,
						pendingSnapshot = landingSnapshot,
						landingSnapshots = previousSnapshots[index] or {},
					})
				end
				-- The real player is only a rig template in vector mode. Park it outside
				-- the simulation so it cannot collide with lane 1.
				local playerRoot = character:FindFirstChild("HumanoidRootPart")
				if playerRoot and playerRoot:IsA("BasePart") then
					playerRoot.Anchored = true
					character:PivotTo(CFrame.new(0, 1000, 0))
				end
				local states = {}
				for _, lane in vectorLanes do
					table.insert(states, lane.state)
				end
				settleCharacters(AgentHarness, states)
				for _, lane in vectorLanes do
					if lane.pendingSnapshot then
						AgentHarness.restoreLandingSnapshot(lane.state, lane.pendingSnapshot)
					end
				end
				for _, lane in vectorLanes do
					lane.lastCheckpointDistance = (lane.state.root.Position - lane.state.checkpoint).Magnitude
				end
				local results = {}
				for _, lane in vectorLanes do
					table.insert(results, {
						observation = {
							schema = "obby-structured-v1",
							values = AgentHarness.observe(lane.state),
						},
						privileged_observation = {
							schema = "obby-privileged-v1",
							values = AgentHarness.observePrivileged(lane.state),
						},
						reward = 0,
						terminated = false,
						truncated = false,
						info = {
							lane_index = lane.index,
							course_seed = lane.seed,
							checkpoint_index = lane.state.checkpointIndex,
							post_landing_reset = lane.postLanding == true,
							restored_jump_cooldown = if lane.pendingSnapshot
								then lane.pendingSnapshot.jumpCooldown
								else 0,
							replayed_checkpoint_index = if lane.pendingSnapshot
								then lane.pendingSnapshot.checkpointIndex
								else 0,
							available_snapshot_count = landingSnapshotCount(lane.landingSnapshots),
						},
					})
				end
				outgoing = {
					protocol_version = "0.1.0",
					message_type = "vector_step_result",
					ack_request_id = command.request_id,
					episode_id = command.episode_id,
					step_id = 0,
					results = results,
					info = {
						num_envs = #vectorLanes,
						studio_command_seconds = os.clock() - commandStarted,
					},
				}
			elseif command.message_type == "vector_reset_lanes_command" then
				clearHeldActions()
				for _, lane in vectorLanes do
					lane.actionStartedAt = nil
				end
				assert(
					#command.course_seeds == #vectorLanes,
					"vector seed count does not match lanes"
				)
				assert(
					#command.reset_mask == #vectorLanes,
					"vector reset mask does not match lanes"
				)
				local postLandingMask = command.post_landing_mask or {}
				local CourseConfig = CurriculumConfig.forStage(curriculumStage)
				local resetStates = {}
				for index, shouldReset in command.reset_mask do
					if shouldReset then
						local oldLane = vectorLanes[index]
						local parent = oldLane.state.character.Parent
						assert(parent ~= nil, "vector lane folder is missing")
						for _, child in parent:GetChildren() do
							child:Destroy()
						end
						local requestedPostLanding = postLandingMask[index] == true
						local landingSnapshot = if requestedPostLanding
							then selectLandingSnapshot(oldLane.landingSnapshots, command.course_seeds[index])
							else nil
						local laneSeed = if landingSnapshot
							then landingSnapshot.seed
							else command.course_seeds[index]
						local origin = Vector3.new((index - 1) * VECTOR_LANE_SPACING, 0, 0)
						local manifest =
							CourseGenerator.build(laneSeed, CourseConfig, parent, origin)
						local agent = cloneAgent(character, parent, index)
						local laneState = AgentHarness.new(agent, manifest)
						AgentHarness.reset(laneState)
						local postLanding = landingSnapshot ~= nil
						vectorLanes[index] = {
							index = index,
							seed = laneSeed,
							state = laneState,
							lastProgress = 0,
							lastCheckpointDistance = 0,
							actionStartedAt = nil,
							previousActionTotalSeconds = 0,
							postLanding = postLanding,
							pendingSnapshot = landingSnapshot,
							landingSnapshots = oldLane.landingSnapshots or {},
						}
						table.insert(resetStates, laneState)
					end
				end
				settleCharacters(AgentHarness, resetStates)
				for index, lane in vectorLanes do
					if command.reset_mask[index] and lane.pendingSnapshot then
						AgentHarness.restoreLandingSnapshot(lane.state, lane.pendingSnapshot)
					end
				end
				local results = {}
				for _, lane in vectorLanes do
					lane.lastCheckpointDistance = (lane.state.root.Position - lane.state.checkpoint).Magnitude
					table.insert(results, {
						observation = {
							schema = "obby-structured-v1",
							values = AgentHarness.observe(lane.state),
						},
						privileged_observation = {
							schema = "obby-privileged-v1",
							values = AgentHarness.observePrivileged(lane.state),
						},
						reward = 0,
						terminated = false,
						truncated = false,
						info = {
							lane_index = lane.index,
							course_seed = lane.seed,
							checkpoint_index = lane.state.checkpointIndex,
							post_landing_reset = lane.postLanding == true,
							restored_jump_cooldown = if lane.pendingSnapshot
								then lane.pendingSnapshot.jumpCooldown
								else 0,
							replayed_checkpoint_index = if lane.pendingSnapshot
								then lane.pendingSnapshot.checkpointIndex
								else 0,
							available_snapshot_count = landingSnapshotCount(lane.landingSnapshots),
						},
					})
				end
				outgoing = {
					protocol_version = "0.1.0",
					message_type = "vector_step_result",
					ack_request_id = command.request_id,
					episode_id = command.episode_id,
					step_id = command.step_id,
					results = results,
					info = { num_envs = #vectorLanes },
				}
			elseif command.message_type == "vector_action_command" then
				local commandStarted = os.clock()
				assert(#command.actions == #vectorLanes, "vector action count does not match lanes")
				for index, lane in vectorLanes do
					lane.previousActionTotalSeconds = if lane.actionStartedAt
						then commandStarted - lane.actionStartedAt
						else 0
					lane.actionStartedAt = commandStarted
					AgentHarness.applyAction(lane.state, command.actions[index])
					heldActions[lane.state] = {
						action = command.actions[index],
						expiresAt = commandStarted + MAX_ACTION_HOLD_SECONDS,
					}
				end
				local elapsed = 0
				repeat
					elapsed += RunService.Heartbeat:Wait()
					if elapsed < actionRepeatTicks / 60 then
						for index, lane in vectorLanes do
							AgentHarness.refreshMovement(lane.state, command.actions[index])
							AgentHarness.refreshJump(lane.state, command.actions[index])
						end
					end
				until elapsed >= actionRepeatTicks / 60
				local results = {}
				local recoveredStates = {}
				for index, lane in vectorLanes do
					local result = vectorLaneResult(
						AgentHarness,
						lane,
						command.actions[index],
						elapsed,
						heldActions
					)
					table.insert(results, result)
					if result.info.hazard_recovered then
						if lane.recordingVisible then
							table.insert(recoveredStates, lane.state)
						else
							-- Invisible cadence lanes should never freeze the visible
							-- recording while waiting for their rigs to settle.
							AgentHarness.neutralizePose(lane.state)
						end
					end
				end
				if #recoveredStates > 0 then
					settleCharacters(AgentHarness, recoveredStates)
					for index, lane in vectorLanes do
						if results[index].info.hazard_recovered then
							local settledObservation = AgentHarness.observe(lane.state)
							results[index].observation.values = settledObservation
							results[index].privileged_observation.values =
								AgentHarness.observePrivileged(lane.state)
							lane.lastProgress = settledObservation[13]
							lane.lastCheckpointDistance = (
								lane.state.root.Position - lane.state.checkpoint
							).Magnitude
						end
					end
				end
				outgoing = {
					protocol_version = "0.1.0",
					message_type = "vector_step_result",
					ack_request_id = command.request_id,
					episode_id = command.episode_id,
					step_id = command.step_id,
					results = results,
					info = {
						num_envs = #vectorLanes,
						hold_seconds = elapsed,
						action_repeat_ticks = actionRepeatTicks,
						studio_command_seconds = os.clock() - commandStarted,
					},
				}
			elseif command.message_type == "reset_command" then
				local commandStarted = os.clock()
				clearHeldActions()
				local playerRoot = character:FindFirstChild("HumanoidRootPart")
				if playerRoot and playerRoot:IsA("BasePart") then
					playerRoot.Anchored = false
				end
				seed = command.course_seed
				curriculumStage = command.curriculum_stage or 4
				actionRepeatTicks = math.clamp(command.action_repeat_ticks or 3, 1, 6)
				local CourseConfig = CurriculumConfig.forStage(curriculumStage)
				local manifest = CourseGenerator.build(seed, CourseConfig, workspace)
				local course = workspace:FindFirstChild("GeneratedCourseV2")
				if course then
					course:SetAttribute("CurriculumStage", curriculumStage)
				end
				state = AgentHarness.new(character, manifest)
				AgentHarness.reset(state)
				local settleSeconds = settleCharacter(AgentHarness, state)
				lastProgress = 0
				lastCheckpointDistance = (state.root.Position - state.checkpoint).Magnitude
				outgoing = resultPayload(
					command.request_id,
					command.episode_id,
					0,
					AgentHarness,
					state,
					0,
					false,
					seed,
					commandStarted,
					false,
					{
						checkpoint = 0,
						progress = 0,
						checkpoint_bonus = 0,
						finish = 0,
						hazard = 0,
						time = 0,
					},
					{ reset_settle_seconds = settleSeconds }
				)
			elseif command.message_type == "action_command" then
				local commandStarted = os.clock()
				local previousActionTotalSeconds = if lastActionStartedAt
					then commandStarted - lastActionStartedAt
					else 0
				local positionBefore = state.root.Position
				local checkpointBefore = state.checkpoint
				local distanceBefore = (positionBefore - checkpointBefore).Magnitude
				AgentHarness.applyAction(state, command.action)
				heldActions[state] = {
					action = command.action,
					expiresAt = commandStarted + MAX_ACTION_HOLD_SECONDS,
				}
				lastActionStartedAt = commandStarted
				local elapsed = 0
				repeat
					elapsed += RunService.Heartbeat:Wait()
					if elapsed < actionRepeatTicks / 60 then
						-- Humanoid:Move is an input command, so refresh continuous axes each
						-- physics tick. Jump and yaw remain one-shot in applyAction.
						AgentHarness.refreshMovement(state, command.action)
						AgentHarness.refreshJump(state, command.action)
					end
				until elapsed >= actionRepeatTicks / 60
				local observation = AgentHarness.observe(state)
				local positionAfter = state.root.Position
				local velocityAfter = state.root.AssemblyLinearVelocity
				local progress = observation[13]
				local checkpointDistance = (state.root.Position - state.checkpoint).Magnitude
				local checkpointReward = (lastCheckpointDistance - checkpointDistance) / 20
				local progressReward = (progress - lastProgress) * 0.1
				local reachedCheckpoint = AgentHarness.advanceCheckpoint(state)
				-- Detect before the character body touches the physical plane. Its root stays
				-- several studs above the contact point, so checking the plane center is too late.
				local fell = state.root.Position.Y < -17
				local terminated = reachedCheckpoint
					and state.checkpointIndex >= #state.checkpoints
				local checkpointBonus = if reachedCheckpoint then 0.1 else 0
				local finishReward = if terminated then 1 else 0
				local hazardPenalty = if fell then -1 else 0
				local timePenalty = -0.001
				local reward = checkpointReward
					+ progressReward
					+ checkpointBonus
					+ finishReward
					+ hazardPenalty
					+ timePenalty
				if reachedCheckpoint then
					checkpointDistance = (state.root.Position - state.checkpoint).Magnitude
				end
				if fell and not terminated then
					heldActions[state] = nil
					AgentHarness.stop(state)
					AgentHarness.recover(state)
					settleCharacter(AgentHarness, state)
					observation = AgentHarness.observe(state)
					progress = observation[13]
					checkpointDistance = (state.root.Position - state.checkpoint).Magnitude
				end
				lastProgress = progress
				lastCheckpointDistance = checkpointDistance
				if terminated then
					heldActions[state] = nil
					AgentHarness.stop(state)
				end
				outgoing = resultPayload(
					command.request_id,
					command.episode_id,
					command.step_id,
					AgentHarness,
					state,
					reward,
					terminated,
					seed,
					commandStarted,
					fell,
					{
						checkpoint = checkpointReward,
						progress = progressReward,
						checkpoint_bonus = checkpointBonus,
						finish = finishReward,
						hazard = hazardPenalty,
						time = timePenalty,
					},
					{
						action = command.action,
						hold_seconds = elapsed,
						action_lease_seconds = MAX_ACTION_HOLD_SECONDS,
						previous_action_total_seconds = previousActionTotalSeconds,
						distance_before = distanceBefore,
						distance_after = (positionAfter - checkpointBefore).Magnitude,
						position_before = { positionBefore.X, positionBefore.Y, positionBefore.Z },
						position_after = { positionAfter.X, positionAfter.Y, positionAfter.Z },
						velocity_after = { velocityAfter.X, velocityAfter.Y, velocityAfter.Z },
					}
				)
			else
				outgoing = { protocol_version = "0.1.0", message_type = "worker_ready" }
				task.wait(0.05)
			end
		end
		clearHeldActions()
		holdConnection:Disconnect()
	end
end

local function restart()
	generation += 1
	if enabled then
		task.spawn(runWorker, generation)
	end
end

toggle.Click:Connect(function()
	enabled = not enabled
	toggle:SetActive(enabled)
	restart()
end)

toggle:SetActive(true)
restart()
