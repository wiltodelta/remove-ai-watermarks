"""Contract tests for the unified development-only provider-oracle tooling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import provider_oracles as oracles
import pytest
from click.testing import CliRunner
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def test_catalog_covers_every_project_vendor_oracle_surface() -> None:
    assert set(oracles.SURFACES) == {
        "gemini-web",
        "meta-web",
        "microsoft-api",
        "microsoft-web",
        "openai-api",
        "openai-web",
    }
    assert {"google", "meta", "microsoft", "openai"} == oracles.PROVIDER_KEYS
    assert oracles.SURFACES["openai-api"].automation == "api"
    assert oracles.SURFACES["microsoft-api"].automation == "api"
    assert oracles.SURFACES["gemini-web"].automation == "real_browser"
    assert oracles.SURFACES["meta-web"].automation == "playwright"
    assert oracles.SURFACES["gemini-web"].media_types == ("image", "video", "audio")


def test_default_slot_config_is_repository_local() -> None:
    assert oracles.DEFAULT_SLOTS_PATH == oracles.REPOSITORY_ROOT / ".oracle-slots.json"


def test_list_json_exposes_scope_and_surface() -> None:
    result = CliRunner().invoke(oracles.cli, ["list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["key"] for row in payload] == [
        "microsoft-api",
        "openai-api",
        "gemini-web",
        "meta-web",
        "microsoft-web",
        "openai-web",
    ]
    assert next(row for row in payload if row["key"] == "meta-web")["signal_family"] == "content_seal"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("google", ["gemini-web"]),
        ("meta", ["meta-web"]),
        ("microsoft", ["microsoft-api", "microsoft-web"]),
        ("openai", ["openai-api", "openai-web"]),
    ],
)
def test_provider_plan_prefers_api_before_web(provider: str, expected: list[str]) -> None:
    assert [surface.key for surface in oracles.provider_plan(provider)] == expected


def test_plan_json_marks_web_as_an_explicit_fallback_when_api_exists() -> None:
    result = CliRunner().invoke(oracles.cli, ["plan", "openai", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [(row["key"], row["preference"]) for row in payload] == [
        ("openai-api", "primary"),
        ("openai-web", "fallback"),
    ]


def test_slots_support_multiple_accounts_and_networks(tmp_path: Path) -> None:
    config = tmp_path / "slots.json"
    config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "slots": [
                    {
                        "name": "gemini-personal",
                        "surface": "gemini-web",
                        "account_label": "personal-a",
                        "browser_profile": "Profile 2",
                        "google_account_index": 0,
                        "network_label": "home",
                    },
                    {
                        "name": "gemini-research",
                        "surface": "gemini-web",
                        "account_label": "research-b",
                        "browser_profile": "Profile 4",
                        "google_account_index": 2,
                        "network_label": "office",
                    },
                    {
                        "name": "openai-west",
                        "surface": "openai-web",
                        "account_label": None,
                        "browser_profile": None,
                        "network_label": "west-egress",
                        "proxy_url_env": "OPENAI_WEST_PROXY",
                    },
                    {
                        "name": "openai-api-1",
                        "surface": "openai-api",
                        "account_label": "project-1",
                        "browser_profile": None,
                        "network_label": "home",
                        "api_key_env": "OPENAI_API_KEY_1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    slots = oracles.load_slots(config)

    assert set(slots) == {
        "gemini-personal",
        "gemini-research",
        "openai-api-1",
        "openai-west",
    }
    assert slots["gemini-research"]["account_label"] == "research-b"
    assert slots["gemini-research"]["google_account_index"] == 2
    assert slots["openai-west"]["network_label"] == "west-egress"
    assert slots["openai-west"]["proxy_url_env"] == "OPENAI_WEST_PROXY"
    assert slots["openai-api-1"]["api_key_env"] == "OPENAI_API_KEY_1"


def test_named_slot_uses_repository_local_config_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "slots.json"
    config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "slots": [
                    {
                        "name": "openai-api-1",
                        "surface": "openai-api",
                        "account_label": "project-1",
                        "browser_profile": None,
                        "network_label": "home",
                        "api_key_env": "OPENAI_API_KEY_1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(oracles, "DEFAULT_SLOTS_PATH", config)

    slot = oracles.resolve_slot("openai-api", slots_path=None, slot_name="openai-api-1")

    assert slot is not None
    assert slot["api_key_env"] == "OPENAI_API_KEY_1"


def test_slots_override_requires_a_named_slot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--slots requires --slot"):
        oracles.resolve_slot("openai-api", slots_path=tmp_path / "slots.json", slot_name=None)


@pytest.mark.parametrize(
    ("slot", "message"),
    [
        (
            {
                "name": "gemini-unidentified",
                "surface": "gemini-web",
                "account_label": "personal-a",
                "browser_profile": "Profile 2",
                "google_account_index": 0,
                "network_label": "home",
                "proxy_url_env": "GOOGLE_PROXY",
            },
            "does not use proxy_url_env",
        ),
        (
            {
                "name": "gemini-no-profile",
                "surface": "gemini-web",
                "account_label": "personal-a",
                "browser_profile": None,
                "google_account_index": 0,
                "network_label": "home",
            },
            "requires a browser_profile",
        ),
        (
            {
                "name": "openai-api-proxied",
                "surface": "openai-api",
                "account_label": "project-1",
                "browser_profile": None,
                "network_label": "proxy",
                "proxy_url_env": "THORDATA_ROUTE_US",
                "api_key_env": "OPENAI_API_KEY_1",
            },
            "does not use proxy_url_env",
        ),
        (
            {
                "name": "gemini-no-account-index",
                "surface": "gemini-web",
                "account_label": "personal-a",
                "browser_profile": "Profile 2",
                "network_label": "home",
            },
            "requires a google_account_index",
        ),
        (
            {
                "name": "openai-api-browser",
                "surface": "openai-api",
                "account_label": "project-a",
                "browser_profile": "Profile 2",
                "network_label": "home",
            },
            "does not use a browser_profile",
        ),
        (
            {
                "name": "openai-api-no-key",
                "surface": "openai-api",
                "account_label": "project-a",
                "browser_profile": None,
                "network_label": "home",
            },
            "requires an api_key_env",
        ),
        (
            {
                "name": "meta-with-api-key",
                "surface": "meta-web",
                "account_label": None,
                "browser_profile": None,
                "network_label": "home",
                "api_key_env": "OPENAI_API_KEY_1",
            },
            "only openai-api uses api_key_env",
        ),
    ],
)
def test_slots_require_the_identity_needed_by_the_surface(
    tmp_path: Path,
    slot: dict[str, str | None],
    message: str,
) -> None:
    config = tmp_path / "slots.json"
    config.write_text(json.dumps({"format_version": 1, "slots": [slot]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        oracles.load_slots(config)


def test_prepare_snapshots_the_selected_execution_slot(tmp_path: Path, tmp_clean_png: Path) -> None:
    slot: oracles.ExecutionSlot = {
        "name": "gemini-research",
        "surface": "gemini-web",
        "account_label": "research-b",
        "browser_profile": "Profile 4",
        "google_account_index": 2,
        "network_label": "office",
    }

    manifest_path = oracles.prepare_batch(
        "gemini-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
        slot=slot,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["slot"] == slot


def test_prepare_strips_image_metadata_without_changing_pixels(
    tmp_path: Path,
    tmp_png_with_ai_metadata: Path,
) -> None:
    output_dir = tmp_path / "oracle-batch"

    manifest_path = oracles.prepare_batch(
        "gemini-web",
        [tmp_png_with_ai_metadata],
        output_dir=output_dir,
        repository_root=oracles.REPOSITORY_ROOT,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rows"][0]
    upload = output_dir / row["upload_path"]
    assert row["media_type"] == "image"
    assert row["metadata_stripped"] is True
    assert row["pixels_preserved"] is True
    assert row["source_sha256"] != row["upload_sha256"]
    with Image.open(upload) as image:
        assert "parameters" not in image.info
        assert "prompt" not in image.info
    assert oracles.verify_batch(manifest_path)["complete"] is False


def test_prepare_keeps_non_image_bytes_exact(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic video fixture")
    output_dir = tmp_path / "oracle-batch"

    manifest_path = oracles.prepare_batch(
        "gemini-web",
        [source],
        output_dir=output_dir,
        repository_root=oracles.REPOSITORY_ROOT,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rows"][0]
    assert row["media_type"] == "video"
    assert row["metadata_stripped"] is False
    assert row["bytes_preserved"] is True
    assert (output_dir / row["upload_path"]).read_bytes() == source.read_bytes()


def test_prepare_rejects_unsupported_provider_media(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic video fixture")

    with pytest.raises(ValueError, match="does not support video"):
        oracles.prepare_batch(
            "openai-web",
            [source],
            output_dir=tmp_path / "oracle-batch",
            repository_root=oracles.REPOSITORY_ROOT,
        )


def test_record_and_verify_complete_batch(tmp_path: Path, tmp_clean_png: Path) -> None:
    output_dir = tmp_path / "oracle-batch"
    manifest_path = oracles.prepare_batch(
        "meta-web",
        [tmp_clean_png],
        output_dir=output_dir,
        repository_root=oracles.REPOSITORY_ROOT,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    oracles.record_result(
        manifest_path,
        artifact_id=manifest["rows"][0]["artifact_id"],
        watermark_result="detected",
        provenance_result="absent",
        raw_response="AI signature from Meta was found",
        checked_at="2026-09-07T18:30:00Z",
    )

    report = oracles.verify_batch(manifest_path, require_complete=True)
    assert report == {
        "provider": "meta",
        "surface": "meta-web",
        "total": 1,
        "recorded": 1,
        "complete": True,
        "watermark_results": {"detected": 1},
    }


def test_record_refuses_to_overwrite_result(tmp_path: Path, tmp_clean_png: Path) -> None:
    manifest_path = oracles.prepare_batch(
        "microsoft-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
    )
    artifact_id = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"][0]["artifact_id"]
    kwargs = {
        "artifact_id": artifact_id,
        "watermark_result": "not_detected",
        "provenance_result": "absent",
        "raw_response": "Inconclusive",
        "checked_at": "2026-09-07T18:30:00Z",
    }
    oracles.record_result(manifest_path, **kwargs)

    with pytest.raises(ValueError, match="already has a recorded result"):
        oracles.record_result(manifest_path, **kwargs)


def test_verify_rejects_tampered_upload(tmp_path: Path, tmp_clean_png: Path) -> None:
    output_dir = tmp_path / "oracle-batch"
    manifest_path = oracles.prepare_batch(
        "gemini-web",
        [tmp_clean_png],
        output_dir=output_dir,
        repository_root=oracles.REPOSITORY_ROOT,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (output_dir / manifest["rows"][0]["upload_path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="upload hash mismatch"):
        oracles.verify_batch(manifest_path)


def test_openai_check_requires_acknowledgement(tmp_clean_png: Path) -> None:
    result = CliRunner().invoke(oracles.cli, ["check-openai", str(tmp_clean_png)])

    assert result.exit_code != 0
    assert "--acknowledge-upload" in result.output


def test_openai_check_uses_the_hardened_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    calls: list[tuple[Path, bool, str | None]] = []

    def verify(path: Path, *, acknowledge_upload: bool, api_key: str | None) -> SimpleNamespace:
        calls.append((path, acknowledge_upload, api_key))
        return SimpleNamespace(to_dict=lambda: {"status": "detected", "provider_scope": "openai"})

    monkeypatch.setattr(oracles, "verify_openai", verify)
    result = CliRunner().invoke(
        oracles.cli,
        ["check-openai", str(tmp_clean_png), "--acknowledge-upload"],
        env={"OPENAI_API_KEY": "test-default-key"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [(tmp_clean_png, True, "test-default-key")]
    assert json.loads(result.output)["status"] == "detected"


def test_openai_check_selects_a_named_key_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "slots": [
                    {
                        "name": "openai-api-2",
                        "surface": "openai-api",
                        "account_label": "project-2",
                        "browser_profile": None,
                        "network_label": "home",
                        "api_key_env": "OPENAI_API_KEY_2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY_1=test-first-key\nOPENAI_API_KEY_2=test-second-key\n",
        encoding="utf-8",
    )
    calls: list[str | None] = []

    def verify(_path: Path, *, acknowledge_upload: bool, api_key: str | None) -> SimpleNamespace:
        assert acknowledge_upload is True
        calls.append(api_key)
        return SimpleNamespace(to_dict=lambda: {"status": "not_detected"})

    monkeypatch.setattr(oracles, "verify_openai", verify)
    result = CliRunner().invoke(
        oracles.cli,
        [
            "check-openai",
            str(tmp_clean_png),
            "--acknowledge-upload",
            "--slots",
            str(slots_path),
            "--slot",
            "openai-api-2",
            "--env-file",
            str(env_path),
        ],
        env={"OPENAI_API_KEY_1": "", "OPENAI_API_KEY_2": ""},
    )

    assert result.exit_code == 0, result.output
    assert calls == ["test-second-key"]
    assert "test-second-key" not in result.output


def test_secret_environment_interpolates_proxy_routes_without_mutating_process(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "THORDATA_PASSWORD=test-thordata-password\n"
        "THORDATA_ROUTE_US=http://td-customer-user-country-US:${THORDATA_PASSWORD}"
        "@proxy.thordata.test:9999\n",
        encoding="utf-8",
    )

    values = oracles.secret_environment(env_path, environ={})

    assert values["THORDATA_PASSWORD"] == "test-thordata-password"
    assert values["THORDATA_ROUTE_US"] == (
        "http://td-customer-user-country-US:test-thordata-password@proxy.thordata.test:9999"
    )


def test_nonempty_process_secret_overrides_dotenv(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY_1=file-key\n", encoding="utf-8")

    values = oracles.secret_environment(env_path, environ={"OPENAI_API_KEY_1": "process-key"})

    assert values["OPENAI_API_KEY_1"] == "process-key"


def test_microsoft_check_requires_acknowledgement_before_configuration(tmp_clean_png: Path) -> None:
    result = CliRunner().invoke(
        oracles.cli,
        [
            "check-microsoft",
            str(tmp_clean_png),
            "--content-uri",
            "https://media.example.test/sample.png",
            "--env-file",
            "/does/not/exist",
        ],
    )

    assert result.exit_code == 2
    assert "--acknowledge-upload" in result.output
    assert "AZURE_CONTENT_SAFETY_ENDPOINT" not in result.output


def test_microsoft_check_uses_named_slot_and_dotenv_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "slots": [
                    {
                        "name": "microsoft-api-1",
                        "surface": "microsoft-api",
                        "account_label": "content-safety-project",
                        "browser_profile": None,
                        "network_label": "direct",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AZURE_CONTENT_SAFETY_ENDPOINT=https://content-safety.example.test\n"
        "AZURE_CONTENT_SAFETY_SUBSCRIPTION_ID=test-subscription\n"
        "AZURE_CONTENT_SAFETY_RESOURCE_GROUP=test-resource-group\n"
        "AZURE_CONTENT_SAFETY_ACCOUNT_NAME=test-content-safety\n"
        "AZURE_ORACLE_STORAGE_RESOURCE_GROUP=test-storage-resource-group\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def verify(path: Path, **kwargs: object) -> SimpleNamespace:
        calls.append({"path": path, **kwargs})
        return SimpleNamespace(to_dict=lambda: {"status": "not_detected", "provenance_status": "absent"})

    monkeypatch.setattr(oracles, "verify_microsoft", verify)
    result = CliRunner().invoke(
        oracles.cli,
        [
            "check-microsoft",
            str(tmp_clean_png),
            "--content-uri",
            "https://media.example.test/sample.png",
            "--acknowledge-upload",
            "--slots",
            str(slots_path),
            "--slot",
            "microsoft-api-1",
            "--env-file",
            str(env_path),
        ],
        env={"AZURE_CONTENT_SAFETY_ENDPOINT": "", "AZURE_CONTENT_SAFETY_SUBSCRIPTION_ID": ""},
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "path": tmp_clean_png,
            "content_uri": "https://media.example.test/sample.png",
            "endpoint": "https://content-safety.example.test",
            "subscription_id": "test-subscription",
            "resource_group": "test-resource-group",
            "account_name": "test-content-safety",
            "storage_resource_group": "test-storage-resource-group",
            "acknowledge_upload": True,
        }
    ]
    payload = json.loads(result.output)
    assert payload["status"] == "not_detected"
    assert payload["execution_slot"]["name"] == "microsoft-api-1"


def test_check_at_is_timezone_aware(tmp_path: Path, tmp_clean_png: Path) -> None:
    manifest_path = oracles.prepare_batch(
        "gemini-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
    )
    artifact_id = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"][0]["artifact_id"]

    with pytest.raises(ValueError, match="timezone-aware"):
        oracles.record_result(
            manifest_path,
            artifact_id=artifact_id,
            watermark_result="detected",
            provenance_result="absent",
            raw_response="SynthID detected",
            checked_at="2026-09-07T18:30:00",
        )
