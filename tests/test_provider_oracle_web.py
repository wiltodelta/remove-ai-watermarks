"""Contract tests for isolated Playwright provider-oracle automation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import provider_oracle_web as web
import provider_oracles as oracles
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("surface", "page_text", "watermark", "provenance"),
    [
        (
            "openai-web",
            "Generated with OpenAI tools\nSynthID detected\nContent Credentials not detected",
            "detected",
            "absent",
        ),
        (
            "openai-web",
            "No OpenAI signals detected",
            "not_detected",
            "unavailable",
        ),
        (
            "microsoft-web",
            "Microsoft AI detected\nWatermark detected\nC2PA detected",
            "detected",
            "present",
        ),
        (
            "microsoft-web",
            "Inconclusive",
            "not_detected",
            "unavailable",
        ),
        (
            "microsoft-web",
            "No watermark detected\nC2PA detected",
            "not_detected",
            "present",
        ),
        (
            "meta-web",
            "AI signature from Meta was found\nUpload another file",
            "detected",
            "unavailable",
        ),
        (
            "meta-web",
            "AI signatures were found\nUpload another file",
            "detected",
            "unavailable",
        ),
        (
            "meta-web",
            "No AI signatures from Meta were found\nUpload another file",
            "not_detected",
            "unavailable",
        ),
    ],
)
def test_parse_provider_result(
    surface: str,
    page_text: str,
    watermark: str,
    provenance: str,
) -> None:
    verdict = web.parse_provider_result(surface, page_text)

    assert verdict.watermark_result == watermark
    assert verdict.provenance_result == provenance
    assert verdict.raw_response == page_text


def test_parse_provider_result_keeps_throttle_separate_from_clean() -> None:
    verdict = web.parse_provider_result(
        "meta-web",
        "You\u2019ve reached the daily limit for identifications. Try again tomorrow",
    )

    assert verdict.watermark_result == "refused"
    assert verdict.provenance_result == "unavailable"


def test_openai_static_explanation_is_not_a_settled_result() -> None:
    page_text = (
        "Verify OpenAI-generated content\n"
        "Upload an image or audio file to check for signals that it was generated with OpenAI tools.\n"
        "Upload a file\n"
        "This tool checks whether an uploaded file contains provenance signals associated with OpenAI tools."
    )

    assert not web._settled("openai-web", page_text)
    assert web.parse_provider_result("openai-web", page_text).watermark_result == "indeterminate"


def test_openai_provider_error_is_a_settled_indeterminate_result() -> None:
    page_text = "Something went wrong. Please try again. If the issue persists, contact support."

    assert web._settled("openai-web", page_text)
    assert web.parse_provider_result("openai-web", page_text).watermark_result == "indeterminate"


@pytest.mark.parametrize(
    ("surface", "ready_state"),
    [("meta-web", "networkidle"), ("microsoft-web", "networkidle"), ("openai-web", "load")],
)
def test_react_uploads_wait_for_surface_readiness_before_assigning_the_file(
    tmp_path: Path,
    surface: str,
    ready_state: str,
) -> None:
    events: list[str] = []

    class FakeInput:
        def wait_for(self, *, state: str, timeout: int) -> None:
            assert state == "attached"
            assert timeout == 5_000
            events.append("input-attached")

        def set_input_files(self, path: str, *, timeout: int) -> None:
            assert path == str(tmp_path / "upload.webp")
            assert timeout == 5_000
            events.append("file-assigned")

    class FakeLocator:
        @property
        def first(self) -> FakeInput:
            return FakeInput()

    class FakePage:
        def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            assert state == ready_state
            assert timeout == 5_000
            events.append(f"ready:{state}")

        def locator(self, selector: str) -> FakeLocator:
            assert selector == 'input[type="file"]'
            events.append("input-located")
            return FakeLocator()

    web._set_upload_file(FakePage(), surface, tmp_path / "upload.webp", 5.0)  # type: ignore[arg-type]

    assert events == [f"ready:{ready_state}", "input-located", "input-attached", "file-assigned"]


def test_proxy_settings_are_loaded_from_an_environment_reference() -> None:
    slot: oracles.ExecutionSlot = {
        "name": "openai-west",
        "surface": "openai-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "west-egress",
        "proxy_url_env": "OPENAI_WEST_PROXY",
    }

    proxy = web.proxy_settings(slot, {"OPENAI_WEST_PROXY": "http://alice:secret@proxy.example:8080"})

    assert proxy == {
        "server": "http://proxy.example:8080",
        "username": "alice",
        "password": "secret",
    }


def test_proxy_settings_do_not_treat_a_missing_secret_as_direct_access() -> None:
    slot: oracles.ExecutionSlot = {
        "name": "openai-west",
        "surface": "openai-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "west-egress",
        "proxy_url_env": "OPENAI_WEST_PROXY",
    }

    with pytest.raises(ValueError, match="OPENAI_WEST_PROXY is not set"):
        web.proxy_settings(slot, {})


def test_proxy_context_accepts_proxy_certificate() -> None:
    calls: list[dict[str, object]] = []

    class FakeBrowser:
        def new_context(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    proxy = {"server": "http://proxy.example:9999", "username": "route", "password": "secret"}

    web._new_context(FakeBrowser(), proxy)  # type: ignore[arg-type]

    assert calls == [{"proxy": proxy, "ignore_https_errors": True}]


def test_proxy_browser_accepts_proxy_certificate() -> None:
    calls: list[dict[str, object]] = []

    class FakeChromium:
        def launch(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    proxy = {"server": "http://proxy.example:9999", "username": "route", "password": "secret"}

    web._launch_browser(FakeChromium(), proxy, headed=False)  # type: ignore[arg-type]

    assert calls == [{"headless": True, "args": ["--ignore-certificate-errors"]}]


def test_direct_browser_keeps_normal_certificate_checks() -> None:
    calls: list[dict[str, object]] = []

    class FakeChromium:
        def launch(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    web._launch_browser(FakeChromium(), None, headed=True)  # type: ignore[arg-type]

    assert calls == [{"headless": False}]


def test_run_web_batch_records_playwright_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    slot: oracles.ExecutionSlot = {
        "name": "meta-direct",
        "surface": "meta-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "home",
        "proxy_url_env": None,
    }
    manifest_path = oracles.prepare_batch(
        "meta-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
        slot=slot,
    )
    calls: list[tuple[str, list[Path], dict[str, str] | None, bool, float]] = []

    def submit(
        surface: str,
        uploads: list[Path],
        proxy: dict[str, str] | None,
        *,
        headed: bool,
        timeout_seconds: float,
    ) -> list[web.WebVerdict]:
        calls.append((surface, uploads, proxy, headed, timeout_seconds))
        return [
            web.WebVerdict(
                watermark_result="detected",
                provenance_result="unavailable",
                raw_response="AI signature from Meta was found\nUpload another file",
            )
        ]

    monkeypatch.setattr(web, "_submit_batch", submit)
    report = web.run_web_batch(
        manifest_path,
        acknowledge_uploads=True,
        headed=False,
        timeout_seconds=30,
        environ={},
    )

    assert report["complete"] is True
    assert calls[0][0] == "meta-web"
    assert calls[0][2] is None
    results = json.loads((manifest_path.parent / "results.json").read_text(encoding="utf-8"))
    assert results["rows"][0]["watermark_result"] == "detected"


def test_run_web_batch_loads_the_selected_proxy_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    slot: oracles.ExecutionSlot = {
        "name": "openai-thordata-us",
        "surface": "openai-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "thordata-us-residential",
        "proxy_url_env": "THORDATA_ROUTE_US",
    }
    manifest_path = oracles.prepare_batch(
        "openai-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
        slot=slot,
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "THORDATA_PASSWORD=test-thordata-password\n"
        "THORDATA_ROUTE_US=http://td-customer-user-country-US:${THORDATA_PASSWORD}"
        "@proxy.thordata.test:9999\n",
        encoding="utf-8",
    )
    proxies: list[dict[str, str] | None] = []

    def submit(
        _surface: str,
        _uploads: list[Path],
        proxy: dict[str, str] | None,
        *,
        headed: bool,
        timeout_seconds: float,
    ) -> list[web.WebVerdict]:
        assert headed is False
        assert timeout_seconds == 30
        proxies.append(proxy)
        return [web.WebVerdict("not_detected", "unavailable", "No OpenAI signals detected")]

    monkeypatch.setattr(web, "_submit_batch", submit)
    report = web.run_web_batch(
        manifest_path,
        acknowledge_uploads=True,
        headed=False,
        timeout_seconds=30,
        env_file=env_path,
        environ={},
    )

    assert report["watermark_results"] == {"not_detected": 1}
    assert proxies == [
        {
            "server": "http://proxy.thordata.test:9999",
            "username": "td-customer-user-country-US",
            "password": "test-thordata-password",
        }
    ]


def test_run_web_batch_requires_explicit_upload_acknowledgement(
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    manifest_path = oracles.prepare_batch(
        "meta-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
    )

    with pytest.raises(ValueError, match="acknowledge_uploads"):
        web.run_web_batch(manifest_path, acknowledge_uploads=False)


def test_run_web_batch_refuses_google_real_browser_workflow(
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    manifest_path = oracles.prepare_batch(
        "gemini-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
    )

    with pytest.raises(ValueError, match="real Chrome"):
        web.run_web_batch(manifest_path, acknowledge_uploads=True)
