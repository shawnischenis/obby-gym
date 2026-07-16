--!strict
-- PRIMITIVE: world.island
-- Builds a sand-dome island via Terrain:FillBall. Water plane is a separate primitive (world.ocean).
-- Expects global PARAMS:
--   center: Vector3                  (world center of the ball)
--   radius: number                   (sphere radius in studs)
--   peak_height: number              (target peak above y=0, used to validate; informational)
--   material: Enum.Material?         (default Sand)
--   submerge_depth: number?          (how far below y=0 the ball center sits; default 15)
-- Returns (via `return` table consumed by execute_luau):
--   waterline_radius: number, peak_y: number, model_path: string

local Workspace = game:GetService("Workspace")
local Terrain = Workspace.Terrain

local center: Vector3 = PARAMS.center
local radius: number = PARAMS.radius
local material: Enum.Material = PARAMS.material or Enum.Material.Sand
local submergeDepth: number = PARAMS.submerge_depth or 15

-- Place ball so that radius-submergeDepth equals peak height above water
local ballCenter = Vector3.new(center.X, center.Y - submergeDepth, center.Z)
Terrain:FillBall(ballCenter, radius, material)

local peakY = ballCenter.Y + radius
local waterlineRadius = (radius * radius - submergeDepth * submergeDepth)
waterlineRadius = waterlineRadius > 0 and math.sqrt(waterlineRadius) or 0

-- Drop an invisible marker model so other primitives can anchor to this island by path
local marker = Instance.new("Model")
marker.Name = PARAMS.name or "Island"
local center_part = Instance.new("Part")
center_part.Name = "Anchor"
center_part.Anchored = true
center_part.CanCollide = false
center_part.Transparency = 1
center_part.Size = Vector3.new(1, 1, 1)
center_part.CFrame = CFrame.new(center.X, peakY, center.Z)
center_part.Parent = marker
marker.PrimaryPart = center_part
marker.Parent = Workspace

return {
    waterline_radius = waterlineRadius,
    peak_y = peakY,
    model_path = marker:GetFullName(),
}
