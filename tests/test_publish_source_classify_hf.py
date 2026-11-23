"""Local staging checks for the source-classifier Hub publisher."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import publish_source_classify_hf as publish  # noqa: E402


def test_hub_ids_match_the_runtime_loader() -> None:
    # The workflow runs this script with --no-project, so it cannot import the package constants.
    from remove_ai_watermarks.source_classify import (
        CONFIG_FILE,
        GITHUB_WEIGHTS_REPO,
        GITHUB_WEIGHTS_TAG,
        MODEL_FILE,
        WEIGHTS_REPO,
    )

    assert publish.HUB_REPO == WEIGHTS_REPO
    assert publish.GITHUB_REPO == GITHUB_WEIGHTS_REPO
    assert publish.GITHUB_DEFAULT_RELEASE_TAG == GITHUB_WEIGHTS_TAG
    assert publish.CONFIG_FILE == CONFIG_FILE
    assert publish.MODEL_FILE == MODEL_FILE
    assert "multi-provider-source-v6.npz" not in publish.RETIRED_FILES
    assert "multi-provider-source-v7.npz" not in publish.RETIRED_FILES
    assert "multi-provider-source-v8.npz" not in publish.RETIRED_FILES
    assert "wiltodelta/raiw-photo-classify" in publish.RETIRED_HUB_REPOS


def test_stage_release_copies_public_artifacts_and_synthesizes_config(tmp_path: Path) -> None:
    from remove_ai_watermarks.source_classify import (
        ALL_PROVIDER_CLASSES,
        FEATURE_DIM_V4,
        MODEL_FILE,
        MODEL_SCHEMA,
        MODEL_SHA256,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / publish.MODEL_FILE).write_bytes(b"weights")
    destination = tmp_path / "stage"
    publish.stage_release(destination, source)
    assert sorted(path.name for path in destination.iterdir()) == sorted(publish.PUBLIC_FILES)

    config = json.loads((destination / publish.CONFIG_FILE).read_text(encoding="utf-8"))
    assert config["model_type"] == "raiw-source-classify"
    assert config["schema"] == MODEL_SCHEMA
    assert config["artifact_file"] == MODEL_FILE
    assert config["artifact_sha256"] == MODEL_SHA256
    assert config["feature_dimension"] == FEATURE_DIM_V4
    assert config["providers"] == list(ALL_PROVIDER_CLASSES)
    assert config["abstain_label"] == "unknown"


def test_stage_release_requires_the_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=re.escape(publish.MODEL_FILE)):
        publish.stage_release(tmp_path / "stage", tmp_path)


def test_card_matches_the_metrics_manifest() -> None:
    from remove_ai_watermarks.source_classify import GITHUB_WEIGHTS_TAG, MODEL_FILE, MODEL_SCHEMA, MODEL_SHA256

    metrics = json.loads((publish.CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    card = (publish.CARD_DIR / "README.md").read_text(encoding="utf-8")
    assert metrics["schema"] == MODEL_SCHEMA
    assert metrics["artifact_file"] == MODEL_FILE
    assert metrics["artifact_sha256"] == MODEL_SHA256
    assert metrics["github_release_tag"] == GITHUB_WEIGHTS_TAG
    assert MODEL_FILE in card
    assert metrics["artifact_sha256"] in card


def test_publish_uploads_allowlist_and_deletes_retired_hub_repos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    import huggingface_hub

    created: list[tuple[str, str, bool, bool]] = []
    uploads: list[dict[str, object]] = []
    deleted: list[tuple[str, str, bool]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "tok"

        def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool, private: bool) -> None:
            created.append((repo_id, repo_type, exist_ok, private))

        def upload_folder(self, **kwargs: object) -> object:
            uploads.append(kwargs)
            return types.SimpleNamespace(oid="abc123")

        def delete_repo(self, repo_id: str, *, repo_type: str, missing_ok: bool) -> None:
            deleted.append((repo_id, repo_type, missing_ok))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    rev = publish.publish(tmp_path, token="tok", message="msg")
    assert rev == "abc123"
    assert created == [(publish.HUB_REPO, "model", True, False)]
    assert len(uploads) == 1
    assert uploads[0]["allow_patterns"] == list(publish.PUBLIC_FILES)
    assert ("wiltodelta/raiw-photo-classify", "model", True) in deleted


def test_stage_release_includes_standalone_scripts_and_colab_notebook() -> None:
    assert {"website-content.json", "infer.py", "benchmark.py", "demo.ipynb"} <= set(publish.TRACKED_CARD_FILES)
    assert all((publish.CARD_DIR / name).is_file() for name in publish.TRACKED_CARD_FILES)


def test_published_scripts_and_notebook_require_v8_compatible_package_floor() -> None:
    floor_re = re.compile(r"remove-ai-watermarks\[source-classify\]>=(\d+)\.(\d+)\.(\d+)")
    for rel_name in ("benchmark.py", "infer.py", "demo.ipynb"):
        content = (publish.CARD_DIR / rel_name).read_text(encoding="utf-8")
        match = floor_re.search(content)
        assert match is not None, f"Missing package floor in {rel_name}"
        ver = tuple(int(part) for part in match.groups())
        assert ver >= (0, 64, 0), f"{rel_name} floor {ver} predates v8 runtime (0.64.0)"


def test_publish_dry_run_performs_zero_hub_mutations_and_revision_out_records_oid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    class ForbiddenApi:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("HfApi must not be instantiated during --dry-run")

    monkeypatch.setattr(huggingface_hub, "HfApi", ForbiddenApi)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    src_dir = tmp_path / "weights"
    src_dir.mkdir()
    (src_dir / publish.MODEL_FILE).write_bytes(b"weights")
    rev_file = tmp_path / "rev.txt"

    rc = publish.main(["--src", str(src_dir), "--revision-out", str(rev_file), "--dry-run"])
    assert rc == 0
    assert not rev_file.exists()


def test_stage_github_repo_strips_frontmatter_and_excludes_binary_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    import yaml

    gh_stage = tmp_path / "gh_stage"
    rc_stage_only = publish.main(["--stage-github-only", str(gh_stage), "--github-release-tag", "v8"])
    assert rc_stage_only == 0
    staged_rels = sorted(p.relative_to(gh_stage).as_posix() for p in gh_stage.rglob("*") if p.is_file())
    assert staged_rels == sorted(publish.GITHUB_TRACKED_FILES)
    assert publish.MODEL_FILE not in staged_rels

    readme = (gh_stage / "README.md").read_text(encoding="utf-8")
    assert not readme.startswith("---")
    assert readme.startswith("# RAIW multi-provider source-pipeline classifier")
    for heading in publish.WEBSITE_SECTIONS:
        assert heading in readme

    config = json.loads((gh_stage / publish.CONFIG_FILE).read_text(encoding="utf-8"))
    assert config["github_repo"] == publish.GITHUB_REPO
    assert config["github_release_tag"] == "v8"
    assert "Apache License" in (gh_stage / "LICENSE").read_text(encoding="utf-8")
    assert "*.npz" in (gh_stage / ".gitignore").read_text(encoding="utf-8")

    sync_wf = yaml.safe_load((gh_stage / publish.GITHUB_WORKFLOW_REL).read_text(encoding="utf-8"))
    assert sync_wf["permissions"]["contents"] == "write"
    assert {"push", "schedule", "workflow_dispatch"} <= set(sync_wf[True])

    parent_wf = yaml.safe_load(
        (publish.REPO_ROOT / ".github/workflows/publish-source-classify-hf.yml").read_text(encoding="utf-8")
    )
    assert {"push", "workflow_dispatch"} <= set(parent_wf[True])
    assert "docs/source-classify-hf/**" in parent_wf[True]["push"]["paths"]

    # Dry-run with --target both must not spawn any external git/gh commands.
    def forbid_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(subprocess, "run", forbid_run)
    src_dir = tmp_path / "weights"
    src_dir.mkdir()
    weights_path = src_dir / publish.MODEL_FILE
    weights_path.write_bytes(b"unverified-weights")
    gh_rev_file = tmp_path / "gh_rev.txt"

    rc_both = publish.main(
        [
            "--src",
            str(src_dir),
            "--target",
            "both",
            "--github-revision-out",
            str(gh_rev_file),
            "--dry-run",
        ]
    )
    assert rc_both == 0
    assert not gh_rev_file.exists()

    # Non-dry-run publish_github must reject SHA-256 mismatch before calling any external CLI.
    with pytest.raises(SystemExit, match="SHA-256 hash mismatch"):
        publish.publish_github(gh_stage, weights_path, release_tag="v8", message="msg", dry_run=False)

    # SSH deploy-key mode must push over git@github.com without requiring local weights or calling `gh`.
    called_cmds: list[list[str]] = []

    def fake_ssh_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        called_cmds.append(list(cmd))
        assert cmd[0] == "git", f"SSH deploy-key mode must not invoke {cmd[0]}"
        env_arg = kwargs.get("env")
        assert isinstance(env_arg, dict)
        assert "GIT_SSH_COMMAND" in env_arg
        assert "synthetic-ed25519-key" not in " ".join(cmd)
        if cmd[:2] == ["git", "clone"]:
            assert cmd[-2] == f"git@github.com:{publish.GITHUB_REPO}.git"
            target_repo = Path(cmd[-1])
            (target_repo / ".git").mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, "M  README.md\n", "")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, "deadbeef1234567\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_ssh_run)
    returned_sha = publish.publish_github(
        gh_stage,
        None,
        release_tag="v8",
        message="sync",
        deploy_key="synthetic-ed25519-key",
        dry_run=False,
    )
    assert returned_sha == "deadbeef1234567"
    assert any(c[:2] == ["git", "push"] for c in called_cmds)

    # `--sync-github-checkout` must preserve `.git` while removing stale files/dirs.
    existing_checkout = tmp_path / "checkout_sync"
    (existing_checkout / ".git").mkdir(parents=True)
    (existing_checkout / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (existing_checkout / "stale_dir").mkdir()
    (existing_checkout / "stale_dir" / "old.txt").write_text("stale", encoding="utf-8")
    assert publish.main(["--sync-github-checkout", str(existing_checkout)]) == 0
    assert (existing_checkout / ".git" / "HEAD").is_file()
    assert not (existing_checkout / "stale_dir").exists()
    assert (existing_checkout / publish.GITHUB_WORKFLOW_REL).is_file()


def test_website_content_matches_personal_website_visitor_copy_contract(tmp_path: Path) -> None:
    publish.validate_website_contract(publish.CARD_DIR)
    bad_dir = tmp_path / "bad_card"
    bad_dir.mkdir()
    (bad_dir / "README.md").write_text((publish.CARD_DIR / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    copy = json.loads((publish.CARD_DIR / "website-content.json").read_text(encoding="utf-8"))
    (bad_dir / "website-content.json").write_text(
        json.dumps({**copy, "intro": "Run `pip install remove-ai-watermarks`"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match=r"website-content\.json"):
        publish.validate_website_contract(bad_dir)


def _load_module(name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_standalone_infer_script_inspects_weights_and_classifies_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from PIL import Image

    sc_mod = _load_module("raiw_test_source_classify", Path(__file__).with_name("test_source_classify.py"))
    sc_mod.install_stub_ort_if_missing(monkeypatch)
    infer_mod = _load_module("raiw_hf_infer", publish.CARD_DIR / "infer.py")

    weights = sc_mod._write_synthetic_v7_npz(
        tmp_path / "synthetic_v7.npz",
        det_winner="openai",
        attr_winner="openai",
        det_score=-2.0,
        det_threshold=2.0,
        syn_threshold=0.5,
    )
    summary = infer_mod.inspect_weights(weights)
    assert summary["schema"] == "multi-provider-source-v7"
    assert summary["feature_dim"] == 512
    assert len(summary["providers"]) == 10
    assert summary["abstain_label"] == "unknown"
    assert summary["watermark_truth"] == "unknown"
    assert summary["synthetic_threshold"] == 0.5
    assert summary["vit_onnx_size_bytes"] > 64
    assert "openai" in summary["version_subfamilies"]

    sample_img = tmp_path / "gradient.png"
    y, x = np.ogrid[:320, :320]
    arr = np.stack(
        (
            (x * 3 + y * 5) % 256,
            np.broadcast_to((x * 7) % 256, (320, 320)),
            np.broadcast_to((y * 11) % 256, (320, 320)),
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(sample_img)

    exit_inspect = infer_mod.main(["--inspect-weights", "--weights", str(weights), "--json"])
    assert exit_inspect == 0

    rows = infer_mod.classify_targets([sample_img], weights_path=weights)
    assert len(rows) == 1
    assert rows[0]["label"] == "unknown"
    assert rows[0]["reason"] == "abstained"
    assert rows[0]["synthetic_detected"] is True
    assert rows[0]["watermark_truth"] == "unknown"


def test_demo_notebook_is_valid_nbformat_and_compiles_cleanly() -> None:
    nb_path = publish.CARD_DIR / "demo.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) >= 4

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 3
    for idx, cell in enumerate(code_cells):
        assert cell.get("outputs") == []
        source_lines = [line for line in cell["source"] if not line.lstrip().startswith(("%", "!"))]
        compile("".join(source_lines), f"demo.ipynb#cell-{idx}", "exec")


def test_standalone_benchmark_script_summarizes_and_formats_metrics() -> None:
    bench_mod = _load_module("raiw_hf_benchmark", publish.CARD_DIR / "benchmark.py")
    samples = [
        bench_mod.SampleItem(
            sha256="a" * 64,
            category="ui_screenshot",
            expected_provider=None,
            container_format="PNG",
            width=1024,
            height=768,
            raw_bytes=b"",
            regime="non_ai",
        ),
        bench_mod.SampleItem(
            sha256="b" * 64,
            category="native_flux",
            expected_provider="flux",
            container_format="PNG",
            width=1024,
            height=1024,
            raw_bytes=b"",
            regime="native",
        ),
        bench_mod.SampleItem(
            sha256="c" * 64,
            category="ai_openai",
            expected_provider="openai",
            container_format="JPEG",
            width=1024,
            height=1024,
            raw_bytes=b"",
            regime="arena",
        ),
    ]
    preds = [
        {"pred_ai": False, "pred_provider": None},
        {"pred_ai": True, "pred_provider": "flux"},
        {"pred_ai": False, "pred_provider": None},
    ]
    summary = bench_mod._summarize_predictions(
        model_id="wiltodelta/raiw-source-classify",
        revision="deadbeef",
        runtime="NumPy + OpenCV (CPU)",
        weights_mb=15.52,
        license_id="apache-2.0",
        median_cpu_ms=500.0,
        samples=samples,
        preds=preds,
    )
    assert summary["non_ai_fpr_pct"] == 0.0
    assert summary["native_ai"]["recall_pct"] == 100.0
    assert summary["native_ai"]["provider_attribution_pct"] == 100.0
    assert summary["arena_ai"]["recall_pct"] == 0.0
    assert summary["ai_recall_pct"] == 50.0
    assert summary["precision_pct"] == 100.0
    assert summary["provider_attribution_pct"] == 50.0
    md = bench_mod.format_markdown_table([summary])
    assert "`wiltodelta/raiw-source-classify`" in md
    assert "0.0% (0/1)" in md
    assert "Native AI Recall (`1`)" in md
    assert "Arena AI Recall (`1`)" in md
