--!strict
-- TEMPLATE: script.inventory_award
-- ModuleScript helper. Server-side. Creates a Tool from a recipe and parents it to a player's Backpack.
-- Designed to be required from a server Script. No substitution slots; pure module.
--
-- Usage:
--   local Award = require(ReplicatedStorage.Common.InventoryAward)
--   Award.give(player, {
--       name = "Anchovy (1.3kg)",
--       handle = { shape = Enum.PartType.Ball, color = Color3.fromRGB(180,180,170), size = Vector3.new(1,1,1.4), material = Enum.Material.SmoothPlastic },
--       halo = { color = Color3.fromRGB(110, 220, 120), range = 6, brightness = 1.5 },
--       can_be_dropped = true,
--   })

local M = {}

type HandleRecipe = {
    shape: Enum.PartType?,
    color: Color3?,
    size: Vector3?,
    material: Enum.Material?,
}

type HaloRecipe = {
    color: Color3?,
    range: number?,
    brightness: number?,
}

export type ToolRecipe = {
    name: string,
    handle: HandleRecipe?,
    halo: HaloRecipe?,
    can_be_dropped: boolean?,
    grip_offset: CFrame?,
}

function M.give(player: Player, recipe: ToolRecipe): Tool?
    if not player or not player.Parent then return nil end
    local backpack = player:FindFirstChildOfClass("Backpack")
    if not backpack then return nil end

    local tool = Instance.new("Tool")
    tool.Name = recipe.name
    tool.RequiresHandle = true
    tool.CanBeDropped = recipe.can_be_dropped ~= false
    if recipe.grip_offset then tool.Grip = recipe.grip_offset end

    local h = recipe.handle or {}
    local handle = Instance.new("Part")
    handle.Name = "Handle"
    handle.Shape = h.shape or Enum.PartType.Ball
    handle.Material = h.material or Enum.Material.SmoothPlastic
    handle.Color = h.color or Color3.new(1, 1, 1)
    handle.Size = h.size or Vector3.new(1, 1, 1)
    handle.CanCollide = false
    handle.TopSurface = Enum.SurfaceType.Smooth
    handle.BottomSurface = Enum.SurfaceType.Smooth
    handle.Parent = tool

    if recipe.halo then
        local pl = Instance.new("PointLight")
        pl.Color = recipe.halo.color or Color3.new(1, 1, 1)
        pl.Range = recipe.halo.range or 6
        pl.Brightness = recipe.halo.brightness or 1.5
        pl.Parent = handle
    end

    tool.Parent = backpack
    return tool
end

return M
