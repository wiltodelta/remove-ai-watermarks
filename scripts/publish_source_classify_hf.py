#!/usr/bin/env python3
"""Publish the source-pipeline model to Hugging Face and GitHub.

The derivative weight file is downloaded by the GitHub Action from a model
freeze or release tag. The private corpus, feature caches, and training catalog
never enter this repository, the Hugging Face Hub, or the standalone GitHub
model repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = REPO_ROOT / "src"
if (_REPO_SRC / "remove_ai_watermarks").is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from remove_ai_watermarks import source_classify as _sc  # noqa: E402

CONFIG_FILE = _sc.CONFIG_FILE
MODEL_FILE = _sc.MODEL_FILE
HUB_REPO = _sc.WEIGHTS_REPO
GITHUB_REPO = _sc.GITHUB_WEIGHTS_REPO
GITHUB_DEFAULT_RELEASE_TAG = _sc.GITHUB_WEIGHTS_TAG
_verify_sha256 = _sc._verify_sha256

log = logging.getLogger(__name__)

GITHUB_DESCRIPTION = (
    "Open-set 10-provider AI image source-pipeline and synthetic-image classifier "
    "(CPU ONNX + 512-d pixel forensics, pickle-free weights). Not a SynthID detector. "
    "By Victor Kuznetsov."
)
GITHUB_HOMEPAGE = f"https://huggingface.co/{HUB_REPO}"
GITHUB_TOPICS = (
    "ai-image-detection",
    "image-forensics",
    "source-attribution",
    "synthetic-image-detection",
    "open-set-classification",
    "onnx",
    "onnxruntime",
    "computer-vision",
)
GITHUB_GITIGNORE = "__pycache__/\n*.py[cod]\n.venv/\n*.npz\n.DS_Store\nbenchmark_cache/\n"
RETIRED_HUB_REPOS = (
    "wiltodelta/openai-google-image-source-classifier",
    "wiltodelta/raiw-photo-classify",
)
RETIRED_FILES = (
    "source-pipeline-mlp.npz",
    "openai-google-source-v1.npz",
    "multi-provider-source-v3.npz",
    "multi-provider-source-v4.npz",
    "multi-provider-source-v5.npz",
)
CARD_DIR = REPO_ROOT / "docs" / "source-classify-hf"
TRACKED_CARD_FILES = (
    "README.md",
    "metrics.json",
    "website-content.json",
    "infer.py",
    "benchmark.py",
    "demo.ipynb",
)
PUBLIC_FILES = (*TRACKED_CARD_FILES, CONFIG_FILE, MODEL_FILE)
GITHUB_TRACKED_FILES = (*TRACKED_CARD_FILES, CONFIG_FILE, "LICENSE", ".gitignore")
WEBSITE_SECTIONS = ("## Model architecture", "## Evaluation")
FORBIDDEN_VISITOR_COPY = (
    "<",
    ">",
    "`",
    "{",
    "}",
)
FORBIDDEN_VISITOR_PATTERN = (
    r"\b(?:sudo|pip install|brew install|npm install|MCP|CLI|CUDA|API)\b"
    r"|receipt|invoice|raiw-photo-classify"
)


def _strip_yaml_frontmatter(markdown: str) -> str:
    """Remove leading YAML frontmatter so GitHub renders the README heading directly."""
    if not markdown.startswith("---\n"):
        return markdown
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return markdown
    return markdown[end + 5 :].lstrip("\n")


def validate_website_contract(card_dir: Path) -> None:
    """Enforce wiltodelta-app's import contract before staging Hub or GitHub files."""
    import re

    card = (card_dir / "README.md").read_text(encoding="utf-8")
    for heading in WEBSITE_SECTIONS:
        if heading not in card:
            raise SystemExit(f"README.md missing personal-website section: {heading}")
    value = json.loads((card_dir / "website-content.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or sorted(value) != ["intro", "requirements", "sections"]:
        raise SystemExit("website-content.json: invalid visitor copy shape")
    forbidden_re = re.compile(FORBIDDEN_VISITOR_PATTERN)

    def check_plain(text: Any, limit: int) -> None:
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > limit
            or any(ch in text for ch in FORBIDDEN_VISITOR_COPY)
            or forbidden_re.search(text)
        ):
            raise SystemExit("website-content.json: visitor copy violates plain-text or scope rules")

    check_plain(value["intro"], 400)
    check_plain(value["requirements"], 600)
    sections = value["sections"]
    if not isinstance(sections, list) or not (1 <= len(sections) <= 6):
        raise SystemExit("website-content.json: invalid visitor copy sections")
    for section in sections:
        if not isinstance(section, list) or len(section) != 2:
            raise SystemExit("website-content.json: invalid visitor section")
        check_plain(section[0], 80)
        check_plain(section[1], 650)
    if re.search(r"receipt|invoice|raiw-photo-classify", json.dumps(value), re.IGNORECASE):
        raise SystemExit("website-content.json: prohibited term in visitor copy")


def _build_hub_config(
    metrics: dict[str, Any],
    *,
    github_release_tag: str | None = None,
) -> dict[str, Any]:
    """Synthesize model ``config.json`` from the tracked metrics manifest."""
    classes = [str(item) for item in metrics["classes"]]
    resolved_tag = github_release_tag or str(metrics.get("github_release_tag") or GITHUB_DEFAULT_RELEASE_TAG)
    return {
        "model_type": "raiw-source-classify",
        "schema": metrics["schema"],
        "artifact_file": metrics["artifact_file"],
        "artifact_sha256": metrics["artifact_sha256"],
        "feature_dimension": metrics["feature_dimension"],
        "stacked_stage2_dimension": metrics["stacked_stage2_dimension"],
        "providers": [cls for cls in classes if cls != "unknown"],
        "abstain_label": "unknown",
        "library_name": "onnx",
        "package": "remove-ai-watermarks[source-classify]",
        "github_repo": GITHUB_REPO,
        "github_release_tag": resolved_tag,
        "license": "apache-2.0",
    }


def _stage_card_bundle(
    dest: Path,
    extra_sources: Mapping[str, Path],
    *,
    github_release_tag: str | None = None,
    strip_frontmatter: bool = False,
) -> None:
    sources = {name: CARD_DIR / name for name in TRACKED_CARD_FILES}
    sources.update(extra_sources)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing publication files: {', '.join(missing)}")
    validate_website_contract(CARD_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        if name == "README.md" and strip_frontmatter:
            cleaned = _strip_yaml_frontmatter(path.read_text(encoding="utf-8"))
            (dest / name).write_text(cleaned, encoding="utf-8")
        else:
            shutil.copy2(path, dest / name)
    metrics = json.loads((CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    (dest / CONFIG_FILE).write_text(
        json.dumps(_build_hub_config(metrics, github_release_tag=github_release_tag), indent=2) + "\n",
        encoding="utf-8",
    )


def stage_release(
    dest: Path,
    src: Path,
    *,
    github_release_tag: str | None = None,
) -> None:
    """Stage tracked metadata, synthesized Hub config, and one derivative model artifact."""
    _stage_card_bundle(
        dest,
        {MODEL_FILE: src.expanduser().resolve() / MODEL_FILE},
        github_release_tag=github_release_tag,
        strip_frontmatter=False,
    )


def stage_github_repo(
    dest: Path,
    *,
    github_release_tag: str | None = None,
) -> None:
    """Stage the lightweight public GitHub repository tree (weights stay in GitHub Releases)."""
    _stage_card_bundle(
        dest,
        {"LICENSE": REPO_ROOT / "LICENSE"},
        github_release_tag=github_release_tag,
        strip_frontmatter=True,
    )
    (dest / ".gitignore").write_text(GITHUB_GITIGNORE, encoding="utf-8")


def publish(stage: Path, *, token: str, message: str, dry_run: bool = False) -> str:
    """Upload the exact staged allowlist, delete retired weights/repos, and return the Hub revision."""
    if dry_run:
        staged_files = sorted(path.name for path in stage.iterdir() if path.is_file())
        log.info(
            "dry-run: staged %s for %s (allow=%s, delete=%s)",
            staged_files,
            HUB_REPO,
            list(PUBLIC_FILES),
            list(RETIRED_FILES),
        )
        return "(dry-run)"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    log.info("HF API request: create_repo repo_id=%s repo_type=model private=False", HUB_REPO)
    api.create_repo(HUB_REPO, repo_type="model", exist_ok=True, private=False)
    log.info(
        "HF API request: upload_folder repo_id=%s allow_patterns=%s delete_patterns=%s message=%r",
        HUB_REPO,
        list(PUBLIC_FILES),
        list(RETIRED_FILES),
        message,
    )
    commit = api.upload_folder(
        folder_path=str(stage),
        repo_id=HUB_REPO,
        repo_type="model",
        commit_message=message,
        allow_patterns=list(PUBLIC_FILES),
        delete_patterns=list(RETIRED_FILES),
    )
    oid = getattr(commit, "oid", "") or ""
    log.info("HF API response: upload_folder repo_id=%s oid=%s", HUB_REPO, oid or "(unchanged)")
    for retired_repo in RETIRED_HUB_REPOS:
        log.info("HF API request: delete_repo repo_id=%s repo_type=model missing_ok=True", retired_repo)
        api.delete_repo(retired_repo, repo_type="model", missing_ok=True)
    return oid


def _run_checked(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    log.info("External command request: %s", " ".join(cmd))
    proc = subprocess.run(  # noqa: S603 - fixed git/gh subcommands, no shell
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_summary = (proc.stdout or "").strip()
    stderr_summary = (proc.stderr or "").strip()
    if proc.returncode != 0 and not allow_failure:
        log.error(
            "External command failed (rc=%s): %s | stderr=%s",
            proc.returncode,
            " ".join(cmd),
            stderr_summary or "(empty)",
        )
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    log.info(
        "External command response: rc=%s cmd=%s stdout=%s",
        proc.returncode,
        cmd[0],
        (stdout_summary[:240] + "...") if len(stdout_summary) > 240 else (stdout_summary or "(empty)"),
    )
    return proc


def _write_git_askpass(helper_dir: Path) -> Path:
    askpass = helper_dir / "git-askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  *Username*) printf "%s\\n" "x-access-token" ;;\n'
        '  *) printf "%s\\n" "${GH_TOKEN:-}" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    return askpass


def publish_github(
    stage: Path,
    weights_file: Path,
    *,
    release_tag: str | None = None,
    message: str,
    token: str | None = None,
    dry_run: bool = False,
) -> str:
    """Sync the standalone GitHub repository and attach the verified `.npz` release asset."""
    if not weights_file.is_file():
        raise SystemExit(f"missing model artifact for GitHub release: {weights_file}")
    metrics = json.loads((CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    expected_sha = str(metrics["artifact_sha256"]).lower()
    schema = str(metrics["schema"])
    resolved_tag = release_tag or str(metrics.get("github_release_tag") or GITHUB_DEFAULT_RELEASE_TAG)

    if dry_run:
        staged_files = sorted(path.name for path in stage.iterdir() if path.is_file())
        log.info(
            "dry-run: staged %s for GitHub %s release=%s weights=%s",
            staged_files,
            GITHUB_REPO,
            resolved_tag,
            weights_file.name,
        )
        return "(dry-run)"

    try:
        _verify_sha256(weights_file, expected_sha)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    env = dict(os.environ)
    clean_token = (token or "").strip()
    if clean_token:
        env["GH_TOKEN"] = clean_token

    repo_check = _run_checked(["gh", "api", f"repos/{GITHUB_REPO}"], env=env, allow_failure=True)
    existing_topics: set[str] = set()
    need_repo_meta = repo_check.returncode != 0
    if repo_check.returncode == 0:
        repo_meta = json.loads(repo_check.stdout or "{}")
        existing_topics = {str(t) for t in (repo_meta.get("topics") or [])}
        need_repo_meta = (
            repo_meta.get("description") != GITHUB_DESCRIPTION or repo_meta.get("homepage") != GITHUB_HOMEPAGE
        )
    if need_repo_meta:
        repo_subcmd = ["create", GITHUB_REPO, "--public"] if repo_check.returncode != 0 else ["edit", GITHUB_REPO]
        _run_checked(
            [
                "gh",
                "repo",
                *repo_subcmd,
                "--description",
                GITHUB_DESCRIPTION,
                "--homepage",
                GITHUB_HOMEPAGE,
            ],
            env=env,
        )

    if existing_topics != set(GITHUB_TOPICS):
        topics_payload = json.dumps({"names": list(GITHUB_TOPICS)})
        _run_checked(
            ["gh", "api", "-X", "PUT", f"repos/{GITHUB_REPO}/topics", "--input", "-"],
            env=env,
            input_text=topics_payload,
        )

    with tempfile.TemporaryDirectory(prefix="raiw-source-gh-git-") as tmp:
        tmp_root = Path(tmp)
        git_env = dict(env)
        git_env["GIT_TERMINAL_PROMPT"] = "0"
        if clean_token:
            askpass = _write_git_askpass(tmp_root)
            git_env["GIT_ASKPASS"] = str(askpass)

        checkout = tmp_root / "repo"
        remote_url = f"https://github.com/{GITHUB_REPO}.git"
        clone_res = _run_checked(
            ["git", "clone", "--depth", "1", remote_url, str(checkout)],
            env=git_env,
            allow_failure=True,
        )
        if clone_res.returncode != 0 or not (checkout / ".git").is_dir():
            shutil.rmtree(checkout, ignore_errors=True)
            checkout.mkdir(parents=True, exist_ok=True)
            _run_checked(["git", "init", "-b", "main"], cwd=checkout, env=git_env)
            _run_checked(["git", "remote", "add", "origin", remote_url], cwd=checkout, env=git_env)
        else:
            _run_checked(["git", "checkout", "-B", "main"], cwd=checkout, env=git_env)

        for existing in list(checkout.iterdir()):
            if existing.name == ".git":
                continue
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()

        for name in GITHUB_TRACKED_FILES:
            shutil.copy2(stage / name, checkout / name)
            if name in ("infer.py", "benchmark.py"):
                (checkout / name).chmod(0o755)

        _run_checked(["git", "add", "-A"], cwd=checkout, env=git_env)
        status = _run_checked(["git", "status", "--porcelain"], cwd=checkout, env=git_env)
        if status.stdout.strip():
            _run_checked(
                [
                    "git",
                    "-c",
                    "user.name=github-actions[bot]",
                    "-c",
                    "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                    "commit",
                    "-m",
                    message,
                ],
                cwd=checkout,
                env=git_env,
            )
            _run_checked(["git", "push", "-u", "origin", "HEAD:refs/heads/main"], cwd=checkout, env=git_env)

        head_sha = _run_checked(["git", "rev-parse", "HEAD"], cwd=checkout, env=git_env).stdout.strip()

    release_title = f"RAIW Source Classify {resolved_tag} ({schema})"
    size_mb = metrics.get("artifact_size_mb", "")
    size_note = f"{size_mb} MB, " if size_mb else ""
    release_notes = (
        f"Pickle-free CPU-only multimodal weights `{weights_file.name}` "
        f"({size_note}`allow_pickle=False`, `sha256={expected_sha}`).\n\n"
        f"- Hugging Face model card: https://huggingface.co/{HUB_REPO}\n"
        f'- Python package: `pip install "remove-ai-watermarks[source-classify]"`\n'
        f"- Standalone runner: `uv run https://raw.githubusercontent.com/{GITHUB_REPO}/main/infer.py <IMAGE> --json`"
    )

    rel_view = _run_checked(
        [
            "gh",
            "release",
            "view",
            resolved_tag,
            "--repo",
            GITHUB_REPO,
            "--json",
            "assets,body,isPrerelease,name",
        ],
        env=env,
        allow_failure=True,
    )
    if rel_view.returncode != 0:
        _run_checked(
            [
                "gh",
                "release",
                "create",
                resolved_tag,
                str(weights_file),
                "--repo",
                GITHUB_REPO,
                "--target",
                head_sha,
                "--title",
                release_title,
                "--notes",
                release_notes,
                "--latest",
            ],
            env=env,
        )
    else:
        rel_data = json.loads(rel_view.stdout or "{}")
        if (
            rel_data.get("name") != release_title
            or rel_data.get("body") != release_notes
            or bool(rel_data.get("isPrerelease"))
        ):
            _run_checked(
                [
                    "gh",
                    "release",
                    "edit",
                    resolved_tag,
                    "--repo",
                    GITHUB_REPO,
                    "--title",
                    release_title,
                    "--notes",
                    release_notes,
                    "--prerelease=false",
                    "--latest",
                ],
                env=env,
            )
        matching_digest = f"sha256:{expected_sha}"
        has_verified_asset = any(
            asset.get("name") == weights_file.name and str(asset.get("digest", "")).lower() == matching_digest
            for asset in rel_data.get("assets", [])
        )
        if not has_verified_asset:
            _run_checked(
                [
                    "gh",
                    "release",
                    "upload",
                    resolved_tag,
                    str(weights_file),
                    "--repo",
                    GITHUB_REPO,
                    "--clobber",
                ],
                env=env,
            )

    return head_sha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--message", default="Publish source-pipeline classifier")
    parser.add_argument(
        "--target",
        choices=("hf", "github", "both"),
        default="hf",
        help="Publication surface to update (default: hf).",
    )
    parser.add_argument(
        "--github-release-tag",
        default=None,
        help="Release tag in wiltodelta/raiw-source-classify (defaults to metrics.json github_release_tag).",
    )
    parser.add_argument("--revision-out", type=Path, default=None, help="Optional file to record the published OID.")
    parser.add_argument(
        "--github-revision-out",
        type=Path,
        default=None,
        help="Optional file to record the published GitHub commit SHA.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage and validate in a temp dir without remote writes.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    do_hf = args.target in ("hf", "both")
    do_github = args.target in ("github", "both")

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if do_hf and not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN is not set (write role required)")

    gh_token = (
        os.environ.get("RAIW_SOURCE_CLASSIFY_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip() or None
    )
    weights_file = args.src.expanduser().resolve() / MODEL_FILE

    hf_revision = ""
    if do_hf:
        with tempfile.TemporaryDirectory(prefix="raiw-source-hf-") as tmp:
            stage = Path(tmp)
            stage_release(stage, args.src, github_release_tag=args.github_release_tag)
            hf_revision = publish(stage, token=hf_token, message=args.message, dry_run=args.dry_run)
        if args.revision_out is not None and not args.dry_run:
            args.revision_out.parent.mkdir(parents=True, exist_ok=True)
            args.revision_out.write_text(hf_revision + "\n", encoding="utf-8")
        log.info("published %s revision=%s", HUB_REPO, hf_revision or "(unknown)")

    if do_github:
        with tempfile.TemporaryDirectory(prefix="raiw-source-gh-") as tmp:
            gh_stage = Path(tmp)
            stage_github_repo(gh_stage, github_release_tag=args.github_release_tag)
            gh_revision = publish_github(
                gh_stage,
                weights_file,
                release_tag=args.github_release_tag,
                message=args.message,
                token=gh_token,
                dry_run=args.dry_run,
            )
        if args.github_revision_out is not None and not args.dry_run:
            args.github_revision_out.parent.mkdir(parents=True, exist_ok=True)
            args.github_revision_out.write_text(gh_revision + "\n", encoding="utf-8")
        log.info(
            "published GitHub %s@%s commit=%s",
            GITHUB_REPO,
            args.github_release_tag or GITHUB_DEFAULT_RELEASE_TAG,
            gh_revision or "(unknown)",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
