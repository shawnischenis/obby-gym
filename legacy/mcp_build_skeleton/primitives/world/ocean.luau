--!strict
-- PRIMITIVE: world.ocean
-- Lays terrain Water in a rectangular slab. Run before any island/landmass so subsequent
-- terrain fills overwrite the water where land emerges.
-- Expects global PARAMS:
--   center: Vector3?   default (0, -15, 0)
--   size_x: number?    default 600
--   size_z: number?    default 600
--   depth: number?     default 30  (slab thickness along Y)
-- Returns: { y_top: number — water surface y (top of the slab) }

local Terrain = game:GetService("Workspace").Terrain
local center = PARAMS.center or Vector3.new(0, -15, 0)
local sx = PARAMS.size_x or 600
local sz = PARAMS.size_z or 600
local d  = PARAMS.depth or 30

Terrain:FillBlock(CFrame.new(center.X, center.Y, center.Z), Vector3.new(sx, d, sz), Enum.Material.Water)

return {
    y_top = center.Y + d / 2,
}
