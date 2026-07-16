--!strict
-- PRIMITIVE: world.spawn_location
-- Places a single SpawnLocation as a player respawn anchor.
-- Expects global PARAMS:
--   position: Vector3      world position of the spawn center
--   name: string?          default "SpawnLocation"
--   size: Vector3?         default Vector3.new(6, 1, 6)
--   material: Material?    default Enum.Material.Pebble
--   color: Color3?         default Color3.fromRGB(130, 160, 130)
--   neutral: boolean?      default true
--   team_color: BrickColor? optional
-- Returns: { model_path: string }

local Workspace = game:GetService("Workspace")

local spawn = Instance.new("SpawnLocation")
spawn.Name = PARAMS.name or "SpawnLocation"
spawn.Anchored = true
spawn.Size = PARAMS.size or Vector3.new(6, 1, 6)
spawn.CFrame = CFrame.new(PARAMS.position)
spawn.TopSurface = Enum.SurfaceType.Smooth
spawn.BottomSurface = Enum.SurfaceType.Smooth
spawn.Material = PARAMS.material or Enum.Material.Pebble
spawn.Color = PARAMS.color or Color3.fromRGB(130, 160, 130)
spawn.Neutral = PARAMS.neutral ~= false
spawn.Parent = Workspace

return {
    model_path = spawn:GetFullName(),
}
