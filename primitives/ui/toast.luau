--!strict
-- TEMPLATE: ui.toast
-- LocalScript-side ModuleScript. Singleton transient text overlay on the bottom of the screen.
-- Designed to be required from any client script (e.g. tool LocalScripts).
-- Usage:
--   local Toast = require(ReplicatedStorage.UI.Toast)
--   Toast.show("BITE! Click to REEL!", Color3.fromRGB(255,190,60), 5)

local Players = game:GetService("Players")
local player = Players.LocalPlayer

local M = {}

local gui: ScreenGui? = nil
local label: TextLabel? = nil

local function ensureGui()
    if gui and gui.Parent then return end
    local pg = player:WaitForChild("PlayerGui")
    local existing = pg:FindFirstChild("ToastHUD")
    if existing then
        gui = existing :: ScreenGui
        label = (existing :: any):FindFirstChild("ToastLabel")
        return
    end

    local g = Instance.new("ScreenGui")
    g.Name = "ToastHUD"
    g.ResetOnSpawn = false
    g.IgnoreGuiInset = true
    g.Parent = pg

    local lbl = Instance.new("TextLabel")
    lbl.Name = "ToastLabel"
    lbl.Size = UDim2.new(0.45, 0, 0, 60)
    lbl.Position = UDim2.new(0.275, 0, 0.72, 0)
    lbl.BackgroundTransparency = 0.35
    lbl.BackgroundColor3 = Color3.new(0, 0, 0)
    lbl.BorderSizePixel = 0
    lbl.Font = Enum.Font.GothamBold
    lbl.TextScaled = true
    lbl.TextColor3 = Color3.new(1, 1, 1)
    lbl.Visible = false
    lbl.Parent = g

    gui = g
    label = lbl
end

function M.show(text: string, color: Color3?, durationSeconds: number?)
    ensureGui()
    if not label then return end
    label.Text = text
    label.TextColor3 = color or Color3.new(1, 1, 1)
    label.Visible = true
    if durationSeconds then
        local snapshot = text
        task.delay(durationSeconds, function()
            if label and label.Text == snapshot then label.Visible = false end
        end)
    end
end

function M.hide()
    if label then label.Visible = false end
end

return M
