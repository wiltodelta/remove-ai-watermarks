"""Cross-integration coverage, argument fidelity, and source preservation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import click
import pytest
from PIL import Image

from remove_ai_watermarks import integration, mcp_server
from remove_ai_watermarks.cli import main


def operation(name: str) -> dict:
    return next(item for item in integration.operation_catalog() if item["id"] == name)


def test_catalog_covers_every_leaf_and_every_parameter() -> None:
    expected = {}

    def walk(command, path):
        if isinstance(command, click.Group):
            for name, child in command.commands.items():
                walk(child, [*path, name])
        elif path != ["mcp"]:
            expected["cli_" + "_".join(path).replace("-", "_")] = {p.name for p in command.params}

    walk(main, [])
    catalog = integration.operation_catalog()
    actual = {item["id"]: {p["name"] for p in item["parameters"]} for item in catalog if "command" in item}
    assert actual == expected
    assert {item["id"] for item in catalog if "command" not in item} == {"api_classify_onnx", "api_draft_text"}
    json.dumps(catalog)
    advertised = {tool["name"]: tool["inputSchema"] for tool in mcp_server.tools()}
    for item in catalog:
        assert advertised[item["id"]] == integration.input_schema(item)


def test_plan_preserves_repeated_regions_false_flags_and_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    args, output = integration.plan_operation(
        operation("cli_erase"),
        {
            "source": str(source),
            "regions": ["1,2,3,4", "5,6,7,8"],
            "strip_metadata": False,
        },
    )
    assert args.count("--region") == 2
    assert "--keep-metadata" in args
    assert args[-2:] == ["--", str(source)]
    assert output == str(source.with_stem("source_clean"))
    with pytest.raises(ValueError, match="differ"):
        integration.plan_operation(
            operation("cli_erase"), {"source": str(source), "output": str(source), "regions": ["1,2,3,4"]}
        )
    alias = tmp_path / "alias.png"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="differ"):
        integration.plan_operation(
            operation("cli_erase"), {"source": str(source), "output": str(alias), "regions": ["1,2,3,4"]}
        )


def test_metadata_inspection_does_not_advertise_an_output(tmp_path: Path) -> None:
    _, output = integration.plan_operation(
        operation("cli_metadata"), {"source": str(tmp_path / "source.png"), "check": True}
    )
    assert output is None


@pytest.mark.parametrize(
    "values",
    [
        {"source": "x", "unknown": True},
        {"source": "x", "seed": True},
        {"source": "x", "seed": "7"},
        {"source": "x", "profile": "invented"},
        {"source": "x", "fps": float("nan")},
        {"source": "x", "fps": float("inf")},
        {"source": "x\0y"},
        {},
    ],
)
def test_invalid_values_are_rejected_before_execution(values) -> None:
    with pytest.raises(ValueError, match=r"parameters|must be"):
        integration.plan_operation(operation("cli_video_invisible"), values)


def test_real_text_and_image_operations_preserve_inputs(tmp_path: Path) -> None:
    text = tmp_path / "article.txt"
    text.write_text("My own article.\u200b\u200b\u200b\u200b\u200b")
    before = hashlib.sha256(text.read_bytes()).hexdigest()
    result = integration.run_operation(operation("cli_text_metadata"), {"source": str(text), "remove": True})
    assert result["exit_code"] == 0, result
    assert Path(result["output"]).is_file()
    assert hashlib.sha256(text.read_bytes()).hexdigest() == before
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(image)
    before = image.read_bytes()
    result = integration.run_operation(operation("cli_erase"), {"source": str(image), "regions": ["2,2,8,8"]})
    assert result["exit_code"] == 0, result
    assert Path(result["output"]).is_file()
    assert image.read_bytes() == before


def test_secret_never_reaches_logs_or_responses(monkeypatch, caplog) -> None:
    import subprocess

    token = "private-test-credential"

    def fake_run(*args, **kwargs):
        assert token not in str(args)
        assert token in kwargs["input"]
        return subprocess.CompletedProcess([], 1, token, f"request failed: {token}")

    monkeypatch.setattr(integration.subprocess, "run", fake_run)
    with caplog.at_level("INFO"):
        result = integration.run_operation(operation("cli_invisible"), {"source": "x.png", "hf_token": token})
    assert token not in caplog.text
    assert token not in json.dumps(result)
    assert "[redacted]" in result["stderr"]


def test_api_onnx_passes_the_selected_backend(monkeypatch, capsys) -> None:
    from remove_ai_watermarks import classify, optional_deps

    @dataclasses.dataclass
    class Result:
        label: str = "unknown"

    calls = []

    def fake(path, **kwargs):
        calls.append((path, kwargs))
        return Result()

    monkeypatch.setattr(classify, "classify_pixels", fake)
    monkeypatch.setattr(optional_deps, "module_available", lambda *modules: True)
    integration._execute(operation("api_classify_onnx"), {"source": "photo.png"})
    assert calls == [(Path("photo.png"), {"backend": "onnx"})]
    assert json.loads(capsys.readouterr().out) == {"label": "unknown"}


def test_widget_defaults_do_not_override_the_selected_video_recipe():
    selected = operation("cli_video_invisible")
    raw = {p["name"]: p["schema"].get("default") for p in selected["parameters"]}
    raw.update(source="source.mp4", profile="seedance-experimental")
    values = integration.widget_arguments(selected, raw)
    assert values == {"source": "source.mp4", "profile": "seedance-experimental"}
    args, _ = integration.plan_operation(selected, values)
    assert "--noise-std" not in args
    assert "--long-side" not in args
    assert "--fps" not in args


def test_missing_api_backend_names_the_extra(monkeypatch):
    from remove_ai_watermarks import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *names: False)
    with pytest.raises(RuntimeError, match="classify-onnx"):
        integration._execute(operation("api_classify_onnx"), {"source": "photo.png"})


def test_visible_detection_does_not_advertise_an_output():
    _, output = integration.plan_operation(operation("cli_visible"), {"source": "image.png", "detect": True})
    assert output is None


def test_failed_operation_never_returns_a_stale_existing_output(tmp_path, monkeypatch):
    import subprocess

    output = tmp_path / "old.png"
    output.write_bytes(b"previous result")
    monkeypatch.setattr(
        integration.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", "failed")
    )
    result = integration.run_operation(
        operation("cli_erase"),
        {
            "source": str(tmp_path / "source.png"),
            "regions": ["1,2,3,4"],
            "output": str(output),
        },
    )
    assert result["output"] is None
    assert output.read_bytes() == b"previous result"
