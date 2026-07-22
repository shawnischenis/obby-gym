--!strict

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Workspace = game:GetService("Workspace")

local player = Players.LocalPlayer
local playerScripts = player:WaitForChild("PlayerScripts")
local playerModule = require(playerScripts:WaitForChild("PlayerModule"))
local controls = playerModule:GetControls()
local controlStatus =
	ReplicatedStorage:WaitForChild("ObbyRL"):WaitForChild("ControlStatus") :: RemoteEvent

local function disableDefaultControls()
	controls:Disable()
	controlStatus:FireServer(true)
end

disableDefaultControls()
player.CharacterAdded:Connect(function()
	task.defer(disableDefaultControls)
end)

local function currentCamera(): Camera
	local camera = Workspace.CurrentCamera
	while not camera do
		Workspace:GetPropertyChangedSignal("CurrentCamera"):Wait()
		camera = Workspace.CurrentCamera
	end
	return camera
end

task.spawn(function()
	while true do
		local lanes = Workspace:FindFirstChild("ObbyRLVectorLanes")
		if lanes then
			local camera = currentCamera()
			camera.CameraType = Enum.CameraType.Scriptable
			camera.FieldOfView = 55
			local recordingCamera = Workspace:GetAttribute("ObbyRLRecordingCamera")
			if recordingCamera == "completion-follow" then
				local lane = lanes:FindFirstChild("Lane_01")
				local agent = if lane then lane:FindFirstChild("Agent_01") else nil
				local root = if agent then agent:FindFirstChild("HumanoidRootPart") else nil
				if root and root:IsA("BasePart") then
					camera.FieldOfView = 48
					local target = root.Position + Vector3.new(0, 2, -7)
					local desired = CFrame.lookAt(
						root.Position + Vector3.new(0, 16, 25),
						target
					)
					camera.CFrame = camera.CFrame:Lerp(desired, 0.12)
				end
			elseif recordingCamera == "completion" then
				-- Single-agent overview for the promoted two-segment mixed course.
				camera.FieldOfView = 45
				camera.CFrame = CFrame.lookAt(Vector3.new(0, 45, 50), Vector3.new(0, 5, -25))
			elseif Workspace:GetAttribute("ObbyRLRecordingCamera") == "completion-side" then
				-- Orthogonal side replay of the same deterministic completion seed.
				camera.FieldOfView = 45
				camera.CFrame = CFrame.lookAt(Vector3.new(-55, 30, -25), Vector3.new(0, 5, -25))
			elseif Workspace:GetAttribute("ObbyRLRecordingLaneCount") == 2 then
				if Workspace:GetAttribute("ObbyRLRecordingCamera") == "behind" then
					-- Centered elevated chase view: directly behind in plan, tilted down
					-- to show takeoff timing, gap geometry, and landing outcomes.
					camera.FieldOfView = 40
					camera.CFrame = CFrame.lookAt(Vector3.new(22, 75, 95), Vector3.new(22, 4, -8))
				else
					-- Orthogonal side profile: plan-view direction is exactly along X.
					camera.CFrame = CFrame.lookAt(Vector3.new(-70, 14, 15), Vector3.new(0, 5, 15))
				end
			elseif Workspace:GetAttribute("ObbyRLRecordingLaneCount") == 4 then
				camera.CFrame = CFrame.lookAt(Vector3.new(30, 90, 125), Vector3.new(30, 4, 25))
			else
				camera.CFrame = CFrame.lookAt(Vector3.new(150, 300, 400), Vector3.new(150, 4, 50))
			end
			player:SetAttribute("ObbyRLRecordingCameraActive", true)
		else
			player:SetAttribute("ObbyRLRecordingCameraActive", false)
		end
		if Workspace:GetAttribute("ObbyRLRecordingCamera") == "completion-follow" then
			RunService.RenderStepped:Wait()
		else
			task.wait(0.25)
		end
	end
end)
