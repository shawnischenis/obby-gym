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
local ACTION_REPEAT_TICKS = 3

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
	local CourseConfig = require(folder:WaitForChild("ProceduralCourseConfig"))
	local manifest = CourseGenerator.manifest(0, CourseConfig)
	local state = AgentHarness.new(character, manifest)
	return AgentHarness, CourseGenerator, CourseConfig, state, character
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
	hazardRecovered: boolean?
): any
	local course = workspace:FindFirstChild("GeneratedCourseV2")
	return {
		protocol_version = "0.1.0",
		message_type = "step_result",
		ack_request_id = requestId,
		episode_id = episodeId,
		step_id = stepId,
		observation = { schema = "obby-structured-v1", values = AgentHarness.observe(state) },
		reward = reward,
		terminated = terminated,
		truncated = false,
		info = {
			course_seed = seed,
			course_signature = if course then course:GetAttribute("Signature") else nil,
			course_instances = if course then #course:GetDescendants() else 0,
			checkpoint_index = state.checkpointIndex,
			checkpoint_count = #state.checkpoints,
			hazard_recovered = hazardRecovered == true,
			studio_command_seconds = os.clock() - commandStarted,
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
		local AgentHarness, CourseGenerator, CourseConfig, state, character =
			runtime[1], runtime[2], runtime[3], runtime[4], runtime[5]
		local outgoing: any = { protocol_version = "0.1.0", message_type = "worker_ready" }
		local lastProgress = 0
		local seed = 0
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
			if command.message_type == "reset_command" then
				local commandStarted = os.clock()
				seed = command.course_seed
				local manifest = CourseGenerator.build(seed, CourseConfig, workspace)
				state = AgentHarness.new(character, manifest)
				AgentHarness.reset(state)
				lastProgress = 0
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
					false
				)
			elseif command.message_type == "action_command" then
				local commandStarted = os.clock()
				AgentHarness.applyAction(state, command.action)
				local elapsed = 0
				repeat
					elapsed += RunService.Heartbeat:Wait()
				until elapsed >= ACTION_REPEAT_TICKS / 60
				local observation = AgentHarness.observe(state)
				local progress = observation[13]
				local reachedCheckpoint = AgentHarness.advanceCheckpoint(state)
				-- Detect before the character body touches the physical plane. Its root stays
				-- several studs above the contact point, so checking the plane center is too late.
				local fell = state.root.Position.Y < CourseConfig.killPlaneY + 8
				local terminated = progress >= 0.98
				local reward = (progress - lastProgress) - 0.001
				if reachedCheckpoint then
					reward += 0.1
				end
				if terminated then
					reward += 1
				elseif fell then
					reward -= 1
					AgentHarness.recover(state)
					observation = AgentHarness.observe(state)
					progress = observation[13]
				end
				lastProgress = progress
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
					fell
				)
			else
				outgoing = { protocol_version = "0.1.0", message_type = "worker_ready" }
				task.wait(0.05)
			end
		end
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
