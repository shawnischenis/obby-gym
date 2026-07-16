--!strict

local AgentHarness = require(script.Parent.AgentHarness)

local ResetController = {}

export type Controller = {
	state: AgentHarness.State,
	fallY: number,
	resetCount: number,
	lastResetAt: number,
}

function ResetController.new(state: AgentHarness.State, fallY: number): Controller
	return { state = state, fallY = fallY, resetCount = 0, lastResetAt = -math.huge }
end

function ResetController.reset(controller: Controller, reason: string): boolean
	local now = os.clock()
	if now - controller.lastResetAt < 0.25 then
		return false
	end
	controller.lastResetAt = now
	controller.resetCount += 1
	AgentHarness.reset(controller.state)
	controller.state.character:SetAttribute("LastResetReason", reason)
	controller.state.character:SetAttribute("ResetCount", controller.resetCount)
	return true
end

function ResetController.step(controller: Controller): boolean
	if controller.state.root.Position.Y < controller.fallY then
		return ResetController.reset(controller, "fall")
	end
	return false
end

return ResetController
