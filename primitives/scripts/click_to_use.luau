--!strict
-- TEMPLATE: script.click_to_use
-- LocalScript that lives on a Tool. Forwards Tool.Activated to a server RemoteEvent.
-- Optionally validates the click target client-side before firing.
--
-- Substitution slots:
--   {{REMOTE_FOLDER_PATH}}  e.g. "ReplicatedStorage.Fishing.Remotes"
--   {{REMOTE_NAME}}         e.g. "CastRequest"
--   {{REQUIRES_TARGET}}     boolean literal — `true` if a mouse.Hit position is required
--   {{TARGET_FILTER_LUAU}}  optional Luau expression evaluating to `true` if the target is valid.
--                            Available locals: `mouse`, `targetPos`, `targetInstance`.
--                            Use `true` to skip filtering.
--                            Example: `targetInstance == workspace.Terrain`

local Tool = script.Parent
local Players = game:GetService("Players")
local player = Players.LocalPlayer

local function resolveRemote()
    local parts = string.split("{{REMOTE_FOLDER_PATH}}", ".")
    local cur: any = game
    for _, name in ipairs(parts) do
        cur = cur:WaitForChild(name)
    end
    return cur:WaitForChild("{{REMOTE_NAME}}")
end

local Remote = resolveRemote()

local mouse: Mouse?
Tool.Equipped:Connect(function(m) mouse = m end)

Tool.Activated:Connect(function()
    if {{REQUIRES_TARGET}} then
        if not mouse then return end
        local targetPos = mouse.Hit.Position
        local targetInstance = mouse.Target
        if not ({{TARGET_FILTER_LUAU}}) then
            return
        end
        Remote:FireServer(targetPos)
    else
        Remote:FireServer()
    end
end)
