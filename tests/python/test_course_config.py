from __future__ import annotations

from jsonschema import Draft202012Validator
from obby_rl.config import ROOT, load_json


def test_procedural_course_config_schema() -> None:
    config = load_json(ROOT / "configs" / "procedural_course.v1.json")
    schema = load_json(ROOT / "schemas" / "procedural-course.schema.json")
    Draft202012Validator(schema).validate(config)
    assert config["gap_range"][1] <= config["max_jump_gap"]
    assert max(abs(value) for value in config["offset_range"]) <= config["max_lateral_offset"]
    assert config["beam_width_range"][0] >= config["min_oracle_beam_width"]
    assert config["max_oracle_jump_distance"] >= config["max_jump_gap"]
    assert config["max_generation_attempts"] >= 1
