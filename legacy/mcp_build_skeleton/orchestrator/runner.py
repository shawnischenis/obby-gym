#!/usr/bin/env python3
"""
runner.py — compile a GameSpec JSON into a deterministic build plan.

Inputs:  specs/<game>.json  +  primitives/manifest.json + primitives/**/*.luau
Output:  builds/<game>.plan.json — an ordered, executable build plan.

The plan is consumed by an executor that has a Roblox Studio MCP connection.
Refs that depend on prior steps' return values are left as `{"$ref": "id.field"}` markers
in the plan; the executor resolves them against the runtime context.

Usage:
    python3 orchestrator/runner.py specs/fishing_game.v1.json
    python3 orchestrator/runner.py specs/fishing_game.v1.json --out builds/fishing.plan.json --print
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PRIMITIVES_DIR = ROOT / "primitives"
MANIFEST_PATH = PRIMITIVES_DIR / "manifest.json"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_manifest() -> dict:
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def load_primitive_body(prim_name: str, manifest: dict) -> str:
    entry = manifest["primitives"].get(prim_name)
    if not entry:
        raise KeyError(f"primitive not in manifest: {prim_name}")
    return (PRIMITIVES_DIR / entry["file"]).read_text()


def load_spec(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Luau emission (for static, non-ref params)
# ----------------------------------------------------------------------------

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REF_PLACEHOLDER = "__ROBLOXCURSOR_REF_{}__"


def emit_luau(value: Any, type_hint: str | None = None, refs: list | None = None) -> str:
    """Convert a Python value into a Luau literal expression.

    Recognized type_hints from the primitive's params_schema:
      Vector3   -> Vector3.new(x, y, z)
      Color3    -> Color3.fromRGB(r, g, b)
      Material  -> Enum.Material.<Name>
      CFrame    -> CFrame.new(x, y, z)  (position only for now)

    Refs are deferred — they become placeholder tokens collected in `refs`,
    which the executor will substitute against the runtime context.
    """
    refs = refs if refs is not None else []

    # Ref marker: {"$ref": "step.field", optional "scale"/"offset"}
    if isinstance(value, dict) and "$ref" in value:
        idx = len(refs)
        refs.append({"index": idx, "ref": value["$ref"],
                     **{k: v for k, v in value.items() if k != "$ref"}})
        return _REF_PLACEHOLDER.format(idx)

    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)

    if isinstance(value, str):
        if type_hint == "Material":
            return f"Enum.Material.{value}"
        # Escape backslash and quote
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    if isinstance(value, list):
        if type_hint == "Vector3" and len(value) == 3:
            return f"Vector3.new({value[0]}, {value[1]}, {value[2]})"
        if type_hint == "Color3" and len(value) == 3:
            return f"Color3.fromRGB({value[0]}, {value[1]}, {value[2]})"
        return "{" + ", ".join(emit_luau(v, None, refs) for v in value) + "}"

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            key = k if _IDENT_RE.match(k) else f'["{k}"]'
            parts.append(f"{key} = {emit_luau(v, None, refs)}")
        return "{" + ", ".join(parts) + "}"

    raise ValueError(f"don't know how to emit {value!r}")


def emit_params_block(params: dict, schema: dict) -> tuple[str, list]:
    """Emit `local PARAMS = { ... }` honoring per-field type hints from the schema."""
    refs: list = []
    parts = []
    for k, v in params.items():
        hint = (schema.get(k) or {}).get("type") if isinstance(schema, dict) else None
        parts.append(f"    {k} = {emit_luau(v, hint, refs)}")
    body = "local PARAMS = {\n" + ",\n".join(parts) + "\n}\n"
    return body, refs


# ----------------------------------------------------------------------------
# Spec walking & step extraction
# ----------------------------------------------------------------------------

def collect_steps(spec: dict) -> list[dict]:
    """Walk the spec and return an ordered list of build steps.

    Recognized top-level keys: `world`, `spawn`, `entities`, `mechanics`, `ui`.
    Each entry must have `id` (auto-assigned for `spawn`) and `primitive`.
    """
    steps: list[dict] = []

    for entry in spec.get("world", []) or []:
        steps.append({
            "id": entry["id"],
            "primitive": entry["primitive"],
            "params": entry.get("params", {}),
            "exports": entry.get("exports", []),
            "depends_on": entry.get("depends_on", []),
        })

    if "spawn" in spec:
        s = spec["spawn"]
        steps.append({
            "id": s.get("id", "spawn"),
            "primitive": s["primitive"],
            "params": s.get("params", {}),
            "exports": [],
            "depends_on": s.get("depends_on", []),
        })

    for entry in spec.get("entities", []) or []:
        steps.append({
            "id": entry["id"],
            "primitive": entry["primitive"],
            "params": entry.get("params", {}),
            "exports": [],
            "depends_on": entry.get("depends_on", []),
            "attach_scripts": entry.get("attach_scripts", []),
        })

    for entry in spec.get("mechanics", []) or []:
        steps.append({
            "id": entry["id"],
            "primitive": entry["primitive"],
            "params": entry.get("params", {}),
            "exports": [],
            "depends_on": entry.get("depends_on", []),
        })

    for i, entry in enumerate(spec.get("ui", []) or []):
        steps.append({
            "id": entry.get("id", f"ui_{i}"),
            "primitive": entry["primitive"],
            "params": entry.get("params", {}),
            "exports": [],
            "depends_on": entry.get("depends_on", []),
        })

    return steps


# ----------------------------------------------------------------------------
# Ref discovery & topo sort
# ----------------------------------------------------------------------------

def discover_refs(value: Any) -> list[str]:
    """Find all `$ref: "id.field"` references nested anywhere inside a value."""
    found: list[str] = []
    if isinstance(value, dict):
        if "$ref" in value and isinstance(value["$ref"], str):
            found.append(value["$ref"])
        for v in value.values():
            found.extend(discover_refs(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(discover_refs(v))
    return found


def topo_sort(steps: list[dict]) -> list[dict]:
    """Sort steps so that referenced ids come before their referents."""
    by_id = {s["id"]: s for s in steps}
    for s in steps:
        ref_deps = {ref.split(".", 1)[0] for ref in discover_refs(s["params"])}
        s["_deps"] = list(set(s["depends_on"]) | (ref_deps & set(by_id.keys())))

    ordered: list[dict] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(sid: str):
        if sid in visited:
            return
        if sid in visiting:
            raise ValueError(f"dependency cycle through step '{sid}'")
        if sid not in by_id:
            return
        visiting.add(sid)
        for dep in by_id[sid]["_deps"]:
            visit(dep)
        visiting.discard(sid)
        visited.add(sid)
        ordered.append(by_id[sid])

    for s in steps:
        visit(s["id"])
    for s in ordered:
        s.pop("_deps", None)
    return ordered


# ----------------------------------------------------------------------------
# Plan compilation
# ----------------------------------------------------------------------------

def render_world_step(step: dict, manifest: dict) -> dict:
    """For a world primitive: render its Luau by concatenating a PARAMS block + the primitive body."""
    entry = manifest["primitives"][step["primitive"]]
    schema = entry.get("params_schema", {})
    body = load_primitive_body(step["primitive"], manifest)

    params_block, refs = emit_params_block(step["params"], schema)
    luau = (
        f"-- AUTO-GENERATED by runner.py for step '{step['id']}' "
        f"(primitive {step['primitive']})\n"
        + params_block
        + "\n"
        + body
    )
    return {
        "id": step["id"],
        "primitive": step["primitive"],
        "kind": "world",
        "luau_template": luau,
        "deferred_refs": refs,  # placeholders the executor must fill before sending to MCP
        "exports": entry.get("returns", {}),
        "depends_on": step["depends_on"],
        "raw_params": step["params"],
    }


def render_script_step(step: dict, manifest: dict) -> dict:
    """For a script/ui template: substitute slots in the source."""
    entry = manifest["primitives"][step["primitive"]]
    src = load_primitive_body(step["primitive"], manifest)
    # Slot substitution: any "{{NAME}}" tokens replaced with str(params[NAME])
    substituted = src
    missing = []
    for slot in entry.get("slots", []):
        token = slot  # e.g. "{{TABLE_LITERAL}}"
        m = re.fullmatch(r"\{\{(.+)\}\}", slot)
        if not m:
            continue
        name = m.group(1)
        if name not in step["params"]:
            missing.append(name)
            continue
        substituted = substituted.replace(token, _render_slot_value(step["params"][name]))
    return {
        "id": step["id"],
        "primitive": step["primitive"],
        "kind": entry["kind"],
        "module_type": entry.get("module_type", "Script"),
        "parent_kind": entry.get("parent_kind"),
        "source": substituted,
        "missing_slots": missing,
        "depends_on": step["depends_on"],
        "raw_params": step["params"],
    }


def _render_slot_value(value: Any) -> str:
    """Render a Python value as the literal text to splice into a Luau source template."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value  # raw splice: caller is responsible for whether to quote
    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
        # Table-of-tables (e.g., fish weighted table)
        lines = []
        for entry in value:
            inner = ", ".join(
                f"{k} = {_render_slot_value(v) if isinstance(v, (int, float, bool)) else _luau_literal(v)}"
                for k, v in entry.items()
            )
            lines.append("    { " + inner + " }")
        return "{\n" + ",\n".join(lines) + "\n}"
    return _luau_literal(value)


def _luau_literal(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
        return f"Color3.fromRGB({value[0]}, {value[1]}, {value[2]})"
    return emit_luau(value)


def compile_plan(spec: dict, manifest: dict) -> dict:
    steps = collect_steps(spec)

    # Validate primitive refs against manifest
    known = set(manifest["primitives"].keys())
    unknown_steps = [s for s in steps if s["primitive"] not in known]
    known_steps = [s for s in steps if s["primitive"] in known]

    ordered = topo_sort(known_steps)

    rendered = []
    for s in ordered:
        entry = manifest["primitives"][s["primitive"]]
        if entry["kind"] == "world":
            rendered.append(render_world_step(s, manifest))
        elif entry["kind"] in ("script", "ui"):
            rendered.append(render_script_step(s, manifest))
        else:
            rendered.append({
                "id": s["id"], "primitive": s["primitive"], "kind": entry["kind"],
                "raw_params": s["params"], "depends_on": s["depends_on"],
                "note": "executor must handle this kind",
            })

    return {
        "spec_id": spec.get("meta", {}).get("title", "untitled"),
        "ordered_ids": [s["id"] for s in rendered],
        "steps": rendered,
        "unresolved_primitives": [
            {"id": s["id"], "primitive": s["primitive"]} for s in unknown_steps
        ],
        "verification": spec.get("verification", {}),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("spec", help="path to spec JSON")
    p.add_argument("--out", default=None, help="path to write plan JSON")
    p.add_argument("--print", action="store_true", help="pretty-print the plan to stdout")
    args = p.parse_args(argv)

    spec_path = Path(args.spec)
    if not spec_path.exists():
        sys.exit(f"spec not found: {spec_path}")
    spec = load_spec(spec_path)
    manifest = load_manifest()

    plan = compile_plan(spec, manifest)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2))
        print(f"wrote {out}")
    else:
        default_out = ROOT / "builds" / (spec_path.stem + ".plan.json")
        default_out.parent.mkdir(parents=True, exist_ok=True)
        default_out.write_text(json.dumps(plan, indent=2))
        print(f"wrote {default_out}")

    print(f"\nspec: {plan['spec_id']}")
    print(f"resolved steps ({len(plan['steps'])}): " + ", ".join(plan["ordered_ids"]))
    if plan["unresolved_primitives"]:
        print(f"unresolved primitives ({len(plan['unresolved_primitives'])}):")
        for u in plan["unresolved_primitives"]:
            print(f"  - {u['id']}: {u['primitive']}")

    if args.print:
        print("\n--- full plan ---")
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
