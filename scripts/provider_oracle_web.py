#!/usr/bin/env python3
"""Drive public provider-oracle pages in one isolated Playwright context."""

from __future__ import annotations

import logging
import os
import random
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlsplit

import provider_oracles as oracles

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from playwright.sync_api import Browser, BrowserContext, BrowserType, Page, ProxySettings

log = logging.getLogger(__name__)

PLAYWRIGHT_SURFACES = frozenset({"openai-web", "microsoft-web", "meta-web", "synthid-web"})
# A proxy exit is chosen when the run starts. These are the countries that
# choice may use: the United States and European ISO codes.
EUROPE_OR_US_COUNTRIES = (
    "AT",
    "BE",
    "CH",
    "CZ",
    "DE",
    "DK",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HU",
    "IE",
    "IT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "US",
)
_THORDATA_COUNTRY = re.compile(r"-country-[A-Za-z]{2}", re.IGNORECASE)
_THORDATA_SESSION = re.compile(r"-sessid-[A-Za-z0-9]+", re.IGNORECASE)
_THORDATA_SESSION_TIME = re.compile(r"-sesstime-\d+", re.IGNORECASE)


@dataclass(frozen=True)
class WebVerdict:
    """One settled Web result or a distinct could-not-ask outcome."""

    watermark_result: Literal["detected", "not_detected", "indeterminate", "refused", "unreachable"]
    provenance_result: Literal["present", "absent", "indeterminate", "unavailable"]
    raw_response: str


_RESULT_MARKERS = {
    "openai-web": (
        "Generated with OpenAI tools",
        "No OpenAI signals detected",
    ),
    "microsoft-web": (
        "Microsoft AI detected",
        "Inconclusive",
        "No provenance detected",
    ),
    "meta-web": (
        "AI signature from Meta was found",
        "AI signatures from Meta were found",
        "AI signatures were found",
        "No AI signatures from Meta were found",
        "No AI signatures were found",
    ),
    "synthid-web": (
        "Watermark detected",
        "SynthID was detected",
        "Watermark not detected",
        "SynthID was not detected",
    ),
}

_REFUSAL_MARKERS = (
    "you've reached the daily limit",
    "you\u2019ve reached the daily limit",
    "rate_limited",
    "too many requests",
    "verify you are human",
    "just a moment...",
    "access denied",
)

_INDETERMINATE_MARKERS = (
    "Something went wrong. Please try again. If the issue persists, contact support",
    "Please sign in before detection.",
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(marker.casefold() in lowered for marker in markers)


def _provenance_result(text: str) -> Literal["present", "absent", "unavailable"]:
    lowered = text.casefold()
    absent = (
        "content credentials not detected",
        "no content credentials detected",
        "c2pa not detected",
        "no c2pa detected",
    )
    present = ("content credentials detected", "c2pa detected")
    if any(marker in lowered for marker in absent):
        return "absent"
    if any(marker in lowered for marker in present):
        return "present"
    return "unavailable"


def parse_provider_result(surface: str, page_text: str) -> WebVerdict:
    """Map settled provider wording without collapsing refusal or uncertainty."""
    if surface not in PLAYWRIGHT_SURFACES:
        raise ValueError(f"unsupported Playwright oracle surface: {surface}")
    response = page_text.strip()
    lowered = response.casefold()
    provenance = _provenance_result(response)
    if _contains_any(response, _REFUSAL_MARKERS):
        return WebVerdict("refused", provenance, response or "Provider refused the browser request")
    if surface == "openai-web":
        # The landing-page FAQ repeats the lowercase phrase "generated with
        # OpenAI tools" before any upload. The result card uses these exact,
        # title-cased labels, so keep this seam case-sensitive.
        if "No OpenAI signals detected" in response:
            return WebVerdict("not_detected", provenance, response)
        if "Generated with OpenAI tools" in response:
            return WebVerdict("detected", provenance, response)
    elif surface == "microsoft-web":
        if "no watermark detected" in lowered or "watermark not detected" in lowered:
            return WebVerdict("not_detected", provenance, response)
        if "microsoft ai detected" in lowered or "watermark detected" in lowered:
            return WebVerdict("detected", provenance, response)
        if "inconclusive" in lowered or "no provenance detected" in lowered:
            return WebVerdict("not_detected", provenance, response)
    elif surface == "synthid-web":
        # The exact labels are the verdict. A page that shows both, or that
        # asks for a sign-in, has no single watermark result. The sign-in
        # sentence is kept; the skill then continues in the user's existing Chrome.
        detected = "Watermark detected" in response or "SynthID was detected" in response
        not_detected = "Watermark not detected" in response or "SynthID was not detected" in response
        if not_detected and not detected:
            return WebVerdict("not_detected", "unavailable", response)
        if detected and not not_detected:
            return WebVerdict("detected", "unavailable", response)
    else:
        if "no ai signatures from meta were found" in lowered or "no ai signatures were found" in lowered:
            return WebVerdict("not_detected", "unavailable", response)
        if (
            "ai signature from meta was found" in lowered
            or "ai signatures from meta were found" in lowered
            or "ai signatures were found" in lowered
        ):
            return WebVerdict("detected", "unavailable", response)
    return WebVerdict("indeterminate", provenance, response or "Provider page returned no readable result")


def choose_proxy_country(rng: random.Random | None = None) -> str:
    """Pick one exit country from Europe or the United States."""
    picker = random.SystemRandom() if rng is None else rng
    return picker.choice(EUROPE_OR_US_COUNTRIES)


def new_proxy_session() -> str:
    """Return a new ThorData session id. A later call does not reuse it."""
    return "s" + secrets.token_hex(8)


def fresh_thordata_username(username: str, *, country: str, session_id: str) -> str:
    """Drop a prepared country and session, then attach the ones for this run."""
    if country not in EUROPE_OR_US_COUNTRIES:
        raise ValueError("proxy country must be in Europe or the US")
    if re.fullmatch(r"[A-Za-z0-9]{1,32}", session_id) is None:
        raise ValueError("proxy session id must be 1-32 ASCII letters or digits")
    base = _THORDATA_SESSION_TIME.sub("", username)
    base = _THORDATA_SESSION.sub("", base)
    base = _THORDATA_COUNTRY.sub("", base).strip("-")
    return f"{base}-country-{country}-sessid-{session_id}-sesstime-90"


def proxy_settings(
    slot: oracles.ExecutionSlot,
    environ: Mapping[str, str] = os.environ,
    *,
    country: str | None = None,
    session_id: str | None = None,
) -> ProxySettings | None:
    """Resolve a secret proxy URL. A ThorData exit is chosen for this call."""
    variable = slot.get("proxy_url_env")
    if variable is None:
        return None
    value = environ.get(variable)
    if not value:
        raise ValueError(f"proxy environment variable {variable} is not set")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https", "socks5"} or parsed.hostname is None:
        raise ValueError(f"proxy environment variable {variable} is not a valid HTTP, HTTPS, or SOCKS5 URL")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    server = f"{parsed.scheme}://{host}"
    if parsed.port is not None:
        server = f"{server}:{parsed.port}"
    settings: ProxySettings = {"server": server}
    if parsed.username is not None:
        username = unquote(parsed.username)
        is_thordata = parsed.hostname.endswith((".thordata.net", ".thordata.online"))
        if is_thordata:
            chosen_country = choose_proxy_country() if country is None else country
            chosen_session = new_proxy_session() if session_id is None else session_id
            username = fresh_thordata_username(username, country=chosen_country, session_id=chosen_session)
            log.info("ThorData proxy exit country=%s session=%s", chosen_country, chosen_session)
        settings["username"] = username
    if parsed.password is not None:
        settings["password"] = unquote(parsed.password)
    return settings


def _new_context(browser: Browser, proxy: ProxySettings | None) -> BrowserContext:
    """Create one isolated route, accepting the proxy service's TLS certificate."""
    if proxy is None:
        return browser.new_context()
    return browser.new_context(proxy=proxy, ignore_https_errors=True)


def _launch_browser(chromium: BrowserType, proxy: ProxySettings | None, *, headed: bool) -> Browser:
    """Launch Chromium with proxy certificate handling scoped to routed runs."""
    if proxy is None:
        return chromium.launch(headless=not headed)
    return chromium.launch(headless=not headed, args=["--ignore-certificate-errors"])


def _page_evidence(page: Page) -> str:
    """Capture enough provider state to distinguish a response from transport failure."""
    body = page.locator("body").inner_text(timeout=5_000).strip()
    if body:
        return body
    return f"URL: {page.url}\nTitle: {page.title()}".strip()


def _settled(surface: str, text: str) -> bool:
    if _contains_any(text, _REFUSAL_MARKERS):
        return True
    if _contains_any(text, _INDETERMINATE_MARKERS):
        return True
    if surface == "openai-web":
        return any(marker in text for marker in _RESULT_MARKERS[surface])
    if not _contains_any(text, _RESULT_MARKERS[surface]):
        return False
    return surface != "meta-web" or "upload another file" in text.casefold()


def _set_upload_file(page: Page, surface: str, upload: Path, timeout_seconds: float) -> None:
    """Wait for a provider's upload handler before assigning one prepared file."""
    timeout_ms = int(timeout_seconds * 1_000)
    if surface == "openai-web":
        # OpenAI keeps background requests alive, so network-idle is
        # unreachable. Its visible input becomes handler-ready at page load.
        page.wait_for_load_state("load", timeout=timeout_ms)
    elif surface == "meta-web":
        # Meta keeps background /monitoring telemetry open, preventing
        # network-idle; wait for page load and the hydrated upload dropzone.
        page.wait_for_load_state("load", timeout=timeout_ms)
        page.get_by_text("Upload or drag an image, video, or audio file").first.wait_for(
            state="visible",
            timeout=timeout_ms,
        )
    elif surface in {"microsoft-web", "synthid-web"}:
        # These pages mount hidden inputs before their selection handlers are
        # active. Network-idle is the observable handler-readiness seam.
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    file_input = page.locator('input[type="file"]').first
    file_input.wait_for(state="attached", timeout=timeout_ms)
    file_input.set_input_files(str(upload), timeout=timeout_ms)


def _check_page(page: Page, surface: str, upload: Path, timeout_seconds: float) -> WebVerdict:
    """Upload one file and wait for one settled, fresh-navigation result."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    oracle = oracles.SURFACES[surface]
    try:
        page.goto(oracle.url, wait_until="domcontentloaded", timeout=int(timeout_seconds * 1_000))
    except PlaywrightError as error:
        log.warning("Browser navigation to %s failed: %s", oracle.url, error)
        return WebVerdict("unreachable", "unavailable", f"Browser navigation failed: {error}")

    try:
        _set_upload_file(page, surface, upload, timeout_seconds)
    except PlaywrightTimeoutError:
        evidence = _page_evidence(page)
        log.warning("Provider upload control did not become ready at %s: %s", oracle.url, evidence)
        return parse_provider_result(surface, evidence)
    except PlaywrightError as error:
        log.warning("Browser upload of %s to %s failed: %s", upload.name, oracle.url, error)
        return WebVerdict("unreachable", "unavailable", f"Browser file upload failed: {error}")

    deadline = monotonic() + timeout_seconds
    evidence = ""
    while monotonic() < deadline:
        try:
            evidence = _page_evidence(page)
        except PlaywrightError:
            evidence = ""
        if _settled(surface, evidence):
            return parse_provider_result(surface, evidence)
        page.wait_for_timeout(500)
    log.warning("Provider result at %s did not settle within %s seconds: %s", oracle.url, timeout_seconds, evidence)
    return parse_provider_result(surface, evidence)


def _submit_batch(
    surface: str,
    uploads: list[Path],
    proxy: ProxySettings | None,
    *,
    headed: bool,
    timeout_seconds: float,
) -> Iterator[WebVerdict]:
    """Submit every row in one isolated session without retries or route changes."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright.chromium, proxy, headed=headed)
        context = _new_context(browser, proxy)
        try:
            for upload in uploads:
                page = context.new_page()
                try:
                    log.info("Submitting %s to %s", upload.name, oracles.SURFACES[surface].url)
                    verdict = _check_page(page, surface, upload, timeout_seconds)
                    log.info("Provider response for %s: %s", upload.name, verdict.raw_response)
                    yield verdict
                finally:
                    page.close()
        finally:
            context.close()
            browser.close()


def run_web_batch(
    manifest_path: Path,
    *,
    acknowledge_uploads: bool,
    headed: bool = True,
    timeout_seconds: float = 120.0,
    env_file: Path | None = None,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, object]:
    """Run unrecorded rows until a refusal, incomplete result, or transport failure."""
    if not acknowledge_uploads:
        raise ValueError("acknowledge_uploads=True is required before transmitting batch files")
    manifest_path = manifest_path.resolve()
    oracles.verify_batch(manifest_path)
    manifest, results = oracles.load_batch(manifest_path)
    surface = manifest["surface"]
    if oracles.SURFACES[surface].automation == "real_browser":
        raise ValueError(f"{surface} requires the real Chrome workflow, not isolated Playwright")
    if surface not in PLAYWRIGHT_SURFACES:
        raise ValueError(f"surface {surface} is not a Playwright Web oracle")
    slot = manifest["slot"]
    if slot is None:
        raise ValueError("Playwright batches require an explicit execution slot")
    secret_values = dict(environ) if env_file is None else oracles.secret_environment(env_file, environ=environ)
    proxy = proxy_settings(slot, secret_values)
    pending = [
        (manifest_row, result_row)
        for manifest_row, result_row in zip(manifest["rows"], results["rows"], strict=True)
        if result_row["watermark_result"] is None
    ]
    if not pending:
        return oracles.verify_batch(manifest_path, require_complete=True)
    uploads = [manifest_path.parent / row["upload_path"] for row, _result in pending]
    verdicts = _submit_batch(
        surface,
        uploads,
        proxy,
        headed=headed,
        timeout_seconds=timeout_seconds,
    )
    try:
        for (manifest_row, _result), verdict in zip(pending, verdicts, strict=True):
            checked_at = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
            oracles.record_result(
                manifest_path,
                artifact_id=manifest_row["artifact_id"],
                watermark_result=verdict.watermark_result,
                provenance_result=verdict.provenance_result,
                raw_response=verdict.raw_response,
                checked_at=checked_at,
            )
            if verdict.watermark_result not in {"detected", "not_detected"}:
                break
    finally:
        close = getattr(verdicts, "close", None)
        if callable(close):
            close()
    return oracles.verify_batch(manifest_path)
