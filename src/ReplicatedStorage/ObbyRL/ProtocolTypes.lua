--!strict

export type Action = {
	strafe: number,
	forward: number,
	yaw: number,
	jump: boolean,
}

export type Observation = {
	schema: "obby-structured-v1",
	values: { number },
}

export type ResetRequest = {
	protocol_version: "0.1.0",
	message_type: "reset_request",
	request_id: string,
	episode_id: string,
	course_seed: number,
	generator_version: "0.1.0",
}

export type ActionRequest = {
	protocol_version: "0.1.0",
	message_type: "action_request",
	request_id: string,
	episode_id: string,
	step_id: number,
	action: Action,
}

export type StepResult = {
	protocol_version: "0.1.0",
	message_type: "step_result",
	request_id: string,
	episode_id: string,
	step_id: number,
	course_seed: number,
	observation: Observation,
	reward: number,
	reward_components: { [string]: number },
	terminated: boolean,
	truncated: boolean,
}

return table.freeze({})
