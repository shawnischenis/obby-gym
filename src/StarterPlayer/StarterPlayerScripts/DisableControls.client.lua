--!strict

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

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
