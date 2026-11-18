"""Contract tests for isolated Playwright provider-oracle automation."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import provider_oracle_web as web
import provider_oracles as oracles
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _thordata_slot() -> oracles.ExecutionSlot:
    return {
        "name": "openai-thordata-us",
        "surface": "openai-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "thordata-us-residential",
        "proxy_url_env": "THORDATA_ROUTE_US",
    }


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
        (
            "synthid-web",
            "Watermark detected",
            "detected",
            "unavailable",
        ),
        (
            "synthid-web",
            "Watermark not detected",
            "not_detected",
            "unavailable",
        ),
        (
            "synthid-web",
            "SynthID was not detected\nIt is unlikely that this media was made with AI from a SynthID partner",
            "not_detected",
            "unavailable",
        ),
        (
            "synthid-web",
            "Please sign in before detection.",
            "indeterminate",
            "unavailable",
        ),
        (
            "synthid-web",
            "Watermark detected\nWatermark not detected",
            "indeterminate",
            "unavailable",
        ),
        (
            "kling-web",
            (
                "AI Content Detection Results\n"
                "Conclusion: Kling AI generation signals were detected.\n"
                "Invisible watermark\tDetected\tKling AI\n"
                "AIGC metadata\tDetected\tKling AI\n"
                "C2PA manifest\tNot detected\t-"
            ),
            "detected",
            "present",
        ),
        (
            "kling-web",
            (
                "AI Content Detection Results\n"
                "Conclusion: Kling AI generation signals were detected.\n"
                "Invisible watermark\tNot detected\t-\n"
                "AIGC metadata\tDetected\tKling AI\n"
                "C2PA manifest\tNot detected\t-"
            ),
            "not_detected",
            "present",
        ),
        (
            "kling-web",
            (
                "AI Content Detection Results\n"
                "Conclusion: No supported signal was detected. This does not prove the content was not AI-generated.\n"
                "Invisible watermark\tNot detected\t-\n"
                "AIGC metadata\tNot detected\t-\n"
                "C2PA manifest\tNot detected\t-"
            ),
            "not_detected",
            "absent",
        ),
        (
            "kling-web",
            (
                "AI Content Detection Results\n"
                "Conclusion: No supported signal was detected. This does not prove the content was not AI-generated.\n"
                "C2PA manifest\tNot detected\t—\n"
                "Invisible watermark\tFailed\t—\n"
                "AIGC metadata\tFailed\t—"
            ),
            "not_detected",
            "absent",
        ),
        (
            "kling-web",
            (
                "AI Content Detection Results\n"
                "Conclusion: Kling AI generation signals were detected.\n"
                "C2PA manifest\tNot detected\t—\n"
                "Invisible watermark\tFailed\t—\n"
                "AIGC metadata\tDetected\tKling AI"
            ),
            "not_detected",
            "present",
        ),
        (
            "kling-web",
            "Welcome to Kling AI\nContinue with Google\nContinue with Apple\nContinue with email",
            "indeterminate",
            "unavailable",
        ),
        (
            "kling-web",
            "Your detection quota has been used up.",
            "refused",
            "unavailable",
        ),
        (
            "cartesia-web",
            "Watermark detected\nThis audio contains a Cartesia watermark.",
            "detected",
            "unavailable",
        ),
        (
            "cartesia-web",
            "No watermark detected\nThis audio does not contain a Cartesia watermark.",
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


def test_openai_link_wrapped_provider_error_is_settled() -> None:
    page_text = (
        "Something went wrong. Please try again. If the issue persists, contact support\n(opens in a new window)\n."
    )

    assert web._settled("openai-web", page_text)
    assert web.parse_provider_result("openai-web", page_text).watermark_result == "indeterminate"


@pytest.mark.parametrize(
    ("surface", "ready_state", "expected_events"),
    [
        ("meta-web", "load", ["ready:load", "dropzone-visible", "input-located", "input-attached", "file-assigned"]),
        ("microsoft-web", "networkidle", ["ready:networkidle", "input-located", "input-attached", "file-assigned"]),
        ("openai-web", "load", ["ready:load", "input-located", "input-attached", "file-assigned"]),
        ("synthid-web", "networkidle", ["ready:networkidle", "input-located", "input-attached", "file-assigned"]),
    ],
)
def test_react_uploads_wait_for_surface_readiness_before_assigning_the_file(
    tmp_path: Path,
    surface: str,
    ready_state: str,
    expected_events: list[str],
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

    class FakeDropzone:
        def wait_for(self, *, state: str, timeout: int) -> None:
            assert state == "visible"
            assert timeout == 5_000
            events.append("dropzone-visible")

    class FakeDropzoneLocator:
        @property
        def first(self) -> FakeDropzone:
            return FakeDropzone()

    class FakeLocator:
        @property
        def first(self) -> FakeInput:
            return FakeInput()

    class FakePage:
        def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            assert state == ready_state
            assert timeout == 5_000
            events.append(f"ready:{state}")

        def get_by_text(self, text: str) -> FakeDropzoneLocator:
            assert text == "Upload or drag an image, video, or audio file"
            return FakeDropzoneLocator()

        def locator(self, selector: str) -> FakeLocator:
            assert selector == 'input[type="file"]'
            events.append("input-located")
            return FakeLocator()

    web._set_upload_file(FakePage(), surface, tmp_path / "upload.webp", 5.0)  # type: ignore[arg-type]

    assert events == expected_events


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


def test_thordata_proxy_settings_take_a_fresh_europe_or_us_session() -> None:
    proxy = web.proxy_settings(
        _thordata_slot(),
        {
            "THORDATA_ROUTE_US": "http://td-customer-user-country-JP-sessid-prepared-sesstime-15:secret@gateway.pr.thordata.net:9999"
        },
        country="DE",
        session_id="freshsession01",
    )

    assert proxy == {
        "server": "http://gateway.pr.thordata.net:9999",
        "username": "td-customer-user-country-DE-sessid-freshsession01-sesstime-90",
        "password": "secret",
    }


def test_thordata_proxy_settings_reject_a_country_outside_europe_or_the_us() -> None:
    assert "US" in web.EUROPE_OR_US_COUNTRIES
    assert "DE" in web.EUROPE_OR_US_COUNTRIES
    assert "JP" not in web.EUROPE_OR_US_COUNTRIES

    with pytest.raises(ValueError, match="Europe or the US"):
        web.proxy_settings(
            _thordata_slot(),
            {"THORDATA_ROUTE_US": "http://route:secret@gateway.pr.thordata.net:9999"},
            country="JP",
            session_id="freshsession01",
        )


def test_new_proxy_session_changes_between_calls() -> None:
    first = web.new_proxy_session()
    second = web.new_proxy_session()

    assert first != second
    assert re.fullmatch(r"[A-Za-z0-9]{1,32}", first)
    assert re.fullmatch(r"[A-Za-z0-9]{1,32}", second)


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
    manifest_path = oracles.prepare_batch(
        "openai-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
        slot=_thordata_slot(),
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "THORDATA_PASSWORD=test-thordata-password\n"
        "THORDATA_ROUTE_US=http://td-customer-user-country-US:${THORDATA_PASSWORD}"
        "@proxy.pr.thordata.net:9999\n",
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

    monkeypatch.setattr(web, "choose_proxy_country", lambda rng=None: "NL")
    monkeypatch.setattr(web, "new_proxy_session", lambda: "freshsession01")
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
            "server": "http://proxy.pr.thordata.net:9999",
            "username": "td-customer-user-country-NL-sessid-freshsession01-sesstime-90",
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


def test_run_web_batch_refuses_gemini_real_browser_workflow(
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


def test_run_web_batch_requires_a_slot_for_synthid(
    tmp_path: Path,
    tmp_clean_png: Path,
) -> None:
    manifest_path = oracles.prepare_batch(
        "synthid-web",
        [tmp_clean_png],
        output_dir=tmp_path / "oracle-batch",
        repository_root=oracles.REPOSITORY_ROOT,
    )

    with pytest.raises(ValueError, match="explicit execution slot"):
        web.run_web_batch(manifest_path, acknowledge_uploads=True)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("SynthID was detected\nThis media was made or edited with Google AI", "detected"),
        ("SynthID was detected\nSynthID was not detected", "indeterminate"),
    ],
)
def test_current_synthid_positive_label_settles_without_losing_conflicts(label: str, expected: str) -> None:
    assert web._settled("synthid-web", label)
    assert web.parse_provider_result("synthid-web", label).watermark_result == expected


@pytest.mark.parametrize("outcome", ["detected", "not_detected", "indeterminate", "refused", "unreachable"])
def test_batch_continues_only_after_conclusive_verdicts_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_clean_png: Path,
    outcome: str,
) -> None:
    slot: oracles.ExecutionSlot = {
        "name": "synthid-direct",
        "surface": "synthid-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "direct",
        "proxy_url_env": None,
    }
    from PIL import Image

    other = tmp_path / "other.png"
    Image.new("RGB", (3, 2), "navy").save(other)
    manifest = oracles.prepare_batch(
        "synthid-web", [tmp_clean_png, other], output_dir=tmp_path / "bounded-batch", slot=slot
    )
    submitted: list[int] = []
    closed: list[bool] = []
    held: list[object] = []

    def submit(*_args: object, **_kwargs: object):
        def stream():
            try:
                for index in range(2):
                    submitted.append(index)
                    yield web.WebVerdict(outcome if index == 0 else "not_detected", "unavailable", "Observed reply")
            finally:
                closed.append(True)

        iterator = stream()
        held.append(iterator)  # Explicit cleanup must work even while another reference keeps the generator alive.
        return iterator

    monkeypatch.setattr(web, "_submit_batch", submit)
    report = web.run_web_batch(manifest, acknowledge_uploads=True, headed=False, environ={})
    conclusive = outcome in {"detected", "not_detected"}
    assert submitted == ([0, 1] if conclusive else [0])
    assert closed == [True]
    assert report["complete"] is conclusive
    assert report["recorded"] == len(submitted)
    results = json.loads((manifest.parent / "results.json").read_text())
    assert results["rows"][0]["watermark_result"] == outcome
    assert results["rows"][1]["watermark_result"] == ("not_detected" if conclusive else None)


def test_cartesia_web_runs_directly_under_acknowledge_uploads_without_extra_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "clip.wav"
    import wave

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 6)

    slot: oracles.ExecutionSlot = {
        "name": "cartesia-direct",
        "surface": "cartesia-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "direct",
        "proxy_url_env": None,
        "api_key_env": None,
    }
    manifest = oracles.prepare_batch("cartesia-web", [wav_path], output_dir=tmp_path / "cartesia-batch", slot=slot)
    monkeypatch.setattr(
        web,
        "_submit_batch",
        lambda *_a, **_kw: iter([web.WebVerdict("not_detected", "unavailable", "No watermark detected")]),
    )
    report = web.run_web_batch(manifest, acknowledge_uploads=True, headed=False, environ={})
    assert report["complete"] is True
    assert report["watermark_results"] == {"not_detected": 1}
