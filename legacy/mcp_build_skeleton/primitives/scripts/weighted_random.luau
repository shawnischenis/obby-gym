--!strict
-- TEMPLATE: script.weighted_random
-- Module returning a picker over a static weighted table. Use for fish tables, loot drops, rarity rolls, etc.
-- Substitution slots:
--   TABLE_LITERAL  -- a Luau array-of-tables literal; each entry MUST have a positive numeric .weight,
--                     plus any other fields the caller wants returned (name, color, etc.).

local Entries = {{TABLE_LITERAL}}

local total = 0
for _, e in ipairs(Entries) do
    assert(type(e.weight) == "number" and e.weight > 0, "weighted_random: every entry needs a positive .weight")
    total = total + e.weight
end

local M = {}

function M.entries()
    return Entries
end

function M.pick(rngOverride: Random?): any
    local roll
    if rngOverride then
        roll = rngOverride:NextNumber() * total
    else
        roll = math.random() * total
    end
    local acc = 0
    for _, e in ipairs(Entries) do
        acc = acc + e.weight
        if roll <= acc then
            return e
        end
    end
    return Entries[#Entries]
end

return M
