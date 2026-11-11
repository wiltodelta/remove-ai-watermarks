"""Shared operation catalog and isolated execution for user-facing integrations.

The CLI tree is the source of truth for parameters. Catalog construction imports
Click lazily; execution and the MCP transport can use this module with stdlib
imports only. Optional model runtimes remain in the child that performs the work.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

log = logging.getLogger(__name__)


def operation_catalog() -> list[dict[str, Any]]:
    """Describe every non-server CLI leaf and the public API-only backends."""
    import click

    from remove_ai_watermarks.cli import main

    operations: list[dict[str, Any]] = []

    def walk(command: click.Command, path: list[str]) -> None:
        if isinstance(command, click.Group):
            for name, child in command.commands.items():
                walk(child, [*path, name])
            return
        if path == ["mcp"]:
            return
        parameters: list[dict[str, Any]] = []
        for parameter in command.params:
            kind = parameter.type
            schema: dict[str, Any] = {"type": "string"}
            if isinstance(kind, click.Choice):
                schema["enum"] = list(kind.choices)
            elif isinstance(kind, click.types.BoolParamType):
                schema["type"] = "boolean"
            elif isinstance(kind, click.types.IntParamType):
                schema["type"] = "integer"
            elif isinstance(kind, click.types.FloatParamType):
                schema["type"] = "number"
            for attribute, key in (("min", "minimum"), ("max", "maximum")):
                bound = getattr(kind, attribute, None)
                if bound is not None:
                    schema[key] = bound
            if parameter.multiple:
                schema = {"type": "array", "items": schema}
            default: Any = parameter.default
            if isinstance(default, (str, bool, int, float, list, tuple, dict)):
                schema["default"] = list(cast("tuple[Any, ...]", default)) if isinstance(default, tuple) else default
            schema["description"] = getattr(parameter, "help", None) or parameter.name.replace("_", " ")
            parameters.append(
                {
                    "name": parameter.name,
                    "schema": schema,
                    "required": parameter.required,
                    "positional": isinstance(parameter, click.Argument),
                    "flags": parameter.opts if isinstance(parameter, click.Option) else [],
                    "negative_flags": parameter.secondary_opts,
                    "is_flag": getattr(parameter, "is_flag", False),
                    "secret": parameter.name == "hf_token",
                }
            )
        operations.append(
            {
                "id": "cli_" + "_".join(path).replace("-", "_"),
                "title": " ".join(path),
                "description": command.help or " ".join(path),
                "command": path,
                "parameters": parameters,
            }
        )

    walk(main, [])
    source = {"name": "source", "schema": {"type": "string", "description": "Source image path"}, "required": True}
    operations.extend(
        [
            {
                "id": "api_classify_onnx",
                "extra": "classify-onnx",
                "modules": ["numpy", "cv2", "torch", "transformers", "onnxruntime"],
                "title": "classify with ONNX",
                "description": "Classify a photograph with the CPU ONNX backend. This is a guess, not provenance.",
                "parameters": [source],
            },
            {
                "id": "api_draft_text",
                "extra": "text-draft",
                "modules": ["numpy", "cv2", "paddle", "paddleocr"],
                "title": "draft text for verification",
                "description": "Propose OCR manifest lines. Every accepted proposal still requires human verification.",
                "parameters": [
                    source,
                    {"name": "min_score", "schema": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.85}},
                    {"name": "stable", "schema": {"type": "boolean", "default": True}},
                ],
            },
        ]
    )
    return operations


def input_schema(operation: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON Schema shared by the protocol and visual controls."""
    return {
        "type": "object",
        "properties": {p["name"]: p["schema"] for p in operation["parameters"]},
        "required": [p["name"] for p in operation["parameters"] if p.get("required")],
        "additionalProperties": False,
    }


def widget_arguments(operation: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Translate optional visual controls into the same typed operation payload."""
    values: dict[str, Any] = {}
    for parameter in operation["parameters"]:
        name, spec = parameter["name"], parameter["schema"]
        value = raw.get(name)
        if value is None or value == "":
            continue
        if spec["type"] == "array" and isinstance(value, str):
            value = [line.strip() for line in value.splitlines() if line.strip()]
        elif spec["type"] == "boolean" and isinstance(value, str):
            if value == "default":
                continue
            if value not in {"enabled", "disabled"}:
                raise ValueError(f"Invalid boolean control: {name}")
            value = value == "enabled"
        elif spec["type"] in {"integer", "number"} and isinstance(value, str):
            value = int(value) if spec["type"] == "integer" else float(value)
        if not parameter.get("required") and "default" in spec and value == spec["default"]:
            continue
        values[name] = value
    return validate_arguments(operation, values)


def validate_arguments(operation: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Validate values before execution, retaining omitted CLI defaults."""
    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be an object")
    arguments = cast("dict[str, Any]", arguments)
    schema = input_schema(operation)
    unknown = set(arguments) - schema["properties"].keys()
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
    missing = set(schema["required"]) - arguments.keys()
    if missing:
        raise ValueError(f"Missing parameters: {', '.join(sorted(missing))}")

    def check(name: str, value: Any, spec: dict[str, Any]) -> None:
        kind = spec["type"]
        valid = {
            "string": lambda: isinstance(value, str) and bool(value.strip()) and "\0" not in value,
            "boolean": lambda: isinstance(value, bool),
            "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
            "array": lambda: isinstance(value, list),
        }[kind]()
        if not valid:
            raise ValueError(f"{name} must be a {kind}")
        if kind == "number" and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{name} must be one of {spec['enum']}")
        if "minimum" in spec and value < spec["minimum"]:
            raise ValueError(f"{name} must be at least {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ValueError(f"{name} must be at most {spec['maximum']}")
        if kind == "array":
            for item in value:
                check(name, item, spec["items"])

    for name, value in arguments.items():
        check(name, value, schema["properties"][name])
    return dict(arguments)


def plan_operation(operation: dict[str, Any], arguments: dict[str, Any]) -> tuple[list[str], str | None]:
    """Build a CLI invocation without a shell, always preserving the source."""
    values = validate_arguments(operation, arguments)
    if "command" not in operation:
        return [], None
    parameters = operation["parameters"]
    output_parameter = next((p for p in parameters if p["name"] in {"output", "output_dir"}), None)
    if operation["command"][-1] == "metadata" and not values.get("remove"):
        output_parameter = None
    if operation["command"] == ["visible"] and values.get("detect"):
        output_parameter = None
    output = None
    if output_parameter is not None:
        key = output_parameter["name"]
        source = Path(values.get("source", values.get("directory", ""))).resolve()
        default = (
            source.with_name(source.name + "_clean")
            if key == "output_dir"
            else source.with_stem(source.stem + "_clean")
        )
        output = values.get(key) or str(default)
        if Path(output).resolve() == source:
            raise ValueError("Output must differ from the source; integrations never overwrite an input")
        values[key] = output
    flags: list[str] = []
    positional: list[str] = []
    for parameter in parameters:
        name = parameter["name"]
        if name not in values:
            continue
        value = values[name]
        if parameter["positional"]:
            positional.append(str(value))
        elif parameter["is_flag"]:
            if value:
                flags.append(parameter["flags"][0])
            elif parameter["negative_flags"]:
                flags.append(parameter["negative_flags"][0])
            elif parameter["schema"].get("default") is True:
                raise ValueError(f"{name} has no false option in the CLI")
        else:
            for item in value if parameter["schema"]["type"] == "array" else [value]:
                flags.extend([parameter["flags"][0], str(item)])
    return [*operation["command"], *flags, "--", *positional], output


def _output_state(output: str | None) -> dict[str, tuple[int, int, int]]:
    if output is None:
        return {}
    path = Path(output)
    files = path.rglob("*") if path.is_dir() else [path]
    state: dict[str, tuple[int, int, int]] = {}
    for file in files:
        try:
            if file.is_file():
                stat = file.stat()
                state[str(file)] = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            continue
    return state


def run_operation(operation: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one catalog operation in an isolated child of this interpreter."""
    values = validate_arguments(operation, arguments)
    _command, output = plan_operation(operation, values)
    before = _output_state(output)
    public_values = {key: "[redacted]" if key == "hf_token" else value for key, value in values.items()}
    log.info("Integration request: operation=%s arguments=%s", operation["id"], public_values)
    completed = subprocess.run(  # noqa: S603 -- fixed module, no shell, payload on stdin
        [sys.executable, "-m", "remove_ai_watermarks.integration", "execute", operation["id"]],
        input=json.dumps(values),
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1", "COLUMNS": "120", "TERM": "dumb"},
        check=False,
    )
    stdout, stderr = completed.stdout, completed.stderr
    token = values.get("hf_token")
    if token:
        stdout, stderr = stdout.replace(token, "[redacted]"), stderr.replace(token, "[redacted]")
    if _output_state(output) == before:
        output = None
    result = {"exit_code": completed.returncode, "stdout": stdout, "stderr": stderr, "output": output}
    log.info("Integration response: operation=%s result=%s", operation["id"], result)
    return result


def _execute(operation: dict[str, Any], arguments: dict[str, Any]) -> None:
    if "command" in operation:
        from remove_ai_watermarks.cli import main

        command, _output = plan_operation(operation, arguments)
        main.main(command, prog_name="remove-ai-watermarks")
        return
    values = validate_arguments(operation, arguments)
    from remove_ai_watermarks.optional_deps import module_available

    if not module_available(*operation["modules"]):
        extra = operation["extra"]
        raise RuntimeError(f"Missing backend. Install: pip install 'remove-ai-watermarks[{extra}]'")
    if operation["id"] == "api_classify_onnx":
        from remove_ai_watermarks.classify import classify_pixels

        result = classify_pixels(Path(values["source"]), backend="onnx")
    else:
        from remove_ai_watermarks.text_draft import draft_text_lines

        result = draft_text_lines(
            Path(values["source"]), min_score=values.get("min_score", 0.85), stable=values.get("stable", True)
        )
    sys.stdout.write(json.dumps(dataclasses.asdict(result), default=str))


def _main() -> None:
    catalog = operation_catalog()
    if sys.argv[1:] == ["catalog"]:
        sys.stdout.write(json.dumps(catalog))
        return
    if len(sys.argv) != 3 or sys.argv[1] != "execute":
        raise SystemExit("Expected catalog, or execute OPERATION with JSON arguments on stdin")
    operation = next((item for item in catalog if item["id"] == sys.argv[2]), None)
    if operation is None:
        raise SystemExit("Unknown operation")
    try:
        _execute(operation, json.load(sys.stdin))
    except (ValueError, OSError, RuntimeError) as error:
        log.error("Operation failed: %s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    _main()
