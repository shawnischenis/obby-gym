--!strict
-- PRIMITIVE: world.dock
-- Builds a wooden dock as a Model: deck plank + evenly spaced posts that extend underwater.
-- Direction-agnostic — works for docks pointing any horizontal direction.
-- Expects global PARAMS:
--   anchor_pos: Vector3        shore-end position (where dock meets land)
--   direction: Vector3         unit vector pointing into water (Y component ignored)
--   length: number?            default 20
--   width: number?             default 4
--   deck_top_y: number?        default 2 (above water y=0)
--   deck_thick: number?        default 1.5
--   post_spacing: number?      default 4
--   post_submerge_depth: number? default 6 (post bottom y = deck_top_y - this)
--   color_deck: Color3?
--   color_post: Color3?
--   name: string?              default "Dock"
-- Returns: { model_path, tip_position }

local Workspace = game:GetService("Workspace")

local anchorPos: Vector3 = PARAMS.anchor_pos
local rawDir: Vector3 = PARAMS.direction
local dir = Vector3.new(rawDir.X, 0, rawDir.Z)
if dir.Magnitude < 1e-4 then
    error("world.dock: direction must have non-zero horizontal component")
end
dir = dir.Unit

local length: number = PARAMS.length or 20
local width: number = PARAMS.width or 4
local deckTopY: number = PARAMS.deck_top_y or 2
local deckThick: number = PARAMS.deck_thick or 1.5
local postSpacing: number = PARAMS.post_spacing or 4
local postSubmerge: number = PARAMS.post_submerge_depth or 6
local colorDeck: Color3 = PARAMS.color_deck or Color3.fromRGB(150, 110, 70)
local colorPost: Color3 = PARAMS.color_post or Color3.fromRGB(95, 65, 40)

local model = Instance.new("Model")
model.Name = PARAMS.name or "Dock"

local centerPos = anchorPos + dir * (length / 2)
local deckCenterY = deckTopY - deckThick / 2
local deckCenter = Vector3.new(centerPos.X, deckCenterY, centerPos.Z)

-- Orient deck so its long axis (local +Z) aligns with dir
local deck = Instance.new("Part")
deck.Name = "Deck"
deck.Anchored = true
deck.Material = Enum.Material.WoodPlanks
deck.Color = colorDeck
deck.TopSurface = Enum.SurfaceType.Smooth
deck.BottomSurface = Enum.SurfaceType.Smooth
deck.Size = Vector3.new(width, deckThick, length)
deck.CFrame = CFrame.lookAt(deckCenter, deckCenter + dir)
deck.Parent = model

local postHeight = postSubmerge + deckThick
local postCenterY = deckTopY - postHeight / 2
local postSize = 0.6

local rightVec = Vector3.new(dir.Z, 0, -dir.X)
local numPosts = math.floor(length / postSpacing) + 1
for i = 0, numPosts - 1 do
    local along = anchorPos + dir * (i * postSpacing)
    for sign = -1, 1, 2 do
        local center = along + rightVec * (sign * width / 2)
        local post = Instance.new("Part")
        post.Name = string.format("Post_%d_%s", i, sign < 0 and "L" or "R")
        post.Anchored = true
        post.Material = Enum.Material.Wood
        post.Color = colorPost
        post.TopSurface = Enum.SurfaceType.Smooth
        post.BottomSurface = Enum.SurfaceType.Smooth
        post.Size = Vector3.new(postSize, postHeight, postSize)
        post.CFrame = CFrame.new(center.X, postCenterY, center.Z)
        post.Parent = model
    end
end

model.PrimaryPart = deck
model.Parent = Workspace

local tipPos = anchorPos + dir * length
tipPos = Vector3.new(tipPos.X, deckTopY, tipPos.Z)

return {
    model_path = model:GetFullName(),
    tip_position = tipPos,
}
