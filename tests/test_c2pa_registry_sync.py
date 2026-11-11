"""Guards for the generated official C2PA soft-binding registry snapshot."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

import scripts.sync_c2pa_soft_bindings as sync
from remove_ai_watermarks._internal._generated_c2pa_soft_bindings import (
    C2PA_SOFT_BINDING_ROWS,
    C2PA_SOFT_BINDING_SOURCE_LICENSE,
    C2PA_SOFT_BINDING_SOURCE_REVISION,
    C2PA_SOFT_BINDING_SOURCE_URL,
)
from remove_ai_watermarks._internal.constants import C2PA_SOFT_BINDING_REGISTRY, C2PA_SOFT_BINDINGS
from scripts.sync_c2pa_soft_bindings import parse_registry, render_module


def _entry(*, identifier: int = 1, algorithm: str = "com.example.mark.1") -> dict[str, object]:
    return {
        "identifier": identifier,
        "alg": algorithm,
        "type": "watermark",
        "encodedMediaTypes": ["text/plain"],
        "entryMetadata": {
            "description": "Example watermark. Longer implementation detail.",
            "dateEntered": "2026-01-01T00:00:00Z",
            "contact": "registry@example.com",
            "informationalUrl": "https://example.com/mark",
        },
    }


def test_packaged_registry_covers_the_current_official_baseline():
    algorithms = [entry.algorithm for entry in C2PA_SOFT_BINDING_REGISTRY]
    identifiers = [entry.identifier for entry in C2PA_SOFT_BINDING_REGISTRY]

    assert len(C2PA_SOFT_BINDING_REGISTRY) >= 53
    assert identifiers == sorted(identifiers)
    assert len(algorithms) == len(set(algorithms))
    assert set(C2PA_SOFT_BINDINGS) == {algorithm.encode() for algorithm in algorithms}
    assert sum(entry.date_entered.startswith("2026-") for entry in C2PA_SOFT_BINDING_REGISTRY) >= 28
    assert {entry.kind for entry in C2PA_SOFT_BINDING_REGISTRY} == {"watermark", "fingerprint"}
    assert "com.adobe.hiermark.A" in algorithms
    assert "com.adobe.flowmark.A" in algorithms

    assert C2PA_SOFT_BINDING_ROWS
    assert C2PA_SOFT_BINDING_SOURCE_URL.endswith("/softbinding-algorithm-list.json")
    assert re.fullmatch(r"[0-9a-f]{40}", C2PA_SOFT_BINDING_SOURCE_REVISION)
    assert C2PA_SOFT_BINDING_SOURCE_LICENSE == "CC BY 4.0"


def test_sync_accepts_encoded_media_and_renders_attribution():
    rows = parse_registry([_entry()])

    assert rows[0].encoded_media_types == ("text/plain",)
    assert rows[0].decoded_media_types == ()
    assert rows[0].display_label == "Example watermark"
    rendered = render_module(rows, source="fixture.json", revision="fixture-revision")
    assert "Source license: CC BY 4.0" in rendered
    assert "Changes: validated, ordered" in rendered


def test_sync_rejects_duplicate_algorithms():
    with pytest.raises(ValueError, match="algorithms must be unique"):
        parse_registry([_entry(identifier=1), _entry(identifier=2)])


def test_official_source_is_fetched_through_its_resolved_revision(monkeypatch: pytest.MonkeyPatch):
    revision = "a" * 40
    requested: list[str] = []
    monkeypatch.setattr(sync, "_latest_revision", lambda: revision)
    monkeypatch.setattr(sync, "_read_bytes", lambda source: requested.append(source) or b"[]")

    payload, resolved = sync._source_snapshot(sync.DEFAULT_SOURCE_URL, None)

    assert payload == b"[]"
    assert resolved == revision
    assert requested == [sync.PINNED_SOURCE_URL.format(revision=revision)]


def test_packaged_snapshot_preserves_published_resolution_apis():
    """Guard the resolution-API subset the runtime deliberately never calls.

    Nine algorithms publish ``softBindingResolutionApis`` today. The runtime
    resolves nothing over the network by design (see
    docs/c2pa-resolution-research.md for the contracts and the product
    decision); this guard keeps the packaged snapshot honest about what the
    ecosystem publishes, so an upstream change surfaces here consciously
    instead of passing silently through a registry regeneration.
    """
    documented = {
        "ai.trufo.pawprint.watermark": ("https://c2pa.trufo.ai/v1",),
        "ai.trufo.pawprint.fingerprint": ("https://c2pa.trufo.ai/v1",),
        "com.aiwatermark.videoseal.1": ("https://aiwatermark.com/api/v1",),
        "com.aiwatermark.pixelseal.1": ("https://aiwatermark.com/api/v1",),
        "com.aiwatermark.audioseal.1": ("https://aiwatermark.com/api/v1",),
        "me.deepmark.audio.vigil.128": ("https://resolution-api.deepmark.me",),
        "io.blockfact.audio.watermark.32": ("https://api.blockfact.io/api/soft-binding/v1",),
        "com.joinmonolith.sha256": ("https://api.joinmonolith.com/api/c2pa",),
        "studio.aidr.silentcipher.1": ("https://aidr.studio/api/c2pa/v1",),
        "com.etchv.watermark.1": ("https://api.etchv.com/c2pa/v1",),
    }
    packaged = {entry.algorithm: entry.resolution_apis for entry in C2PA_SOFT_BINDING_REGISTRY}

    for algorithm, apis in documented.items():
        assert packaged.get(algorithm) == apis, (
            f"{algorithm} resolution APIs changed: {packaged.get(algorithm)} != {apis}; "
            "update docs/c2pa-resolution-research.md with the new contract"
        )
    with_apis = {algorithm for algorithm, apis in packaged.items() if apis}
    assert with_apis == set(documented), (
        f"upstream softbinding-algorithm-list resolution-API set changed: {sorted(with_apis)}; "
        "revisit docs/c2pa-resolution-research.md before regenerating"
    )


class TestReadBytesAuth:
    """api.github.com rate-limits unauthenticated requests to 60/hour PER SOURCE IP,
    which CI runners exhaust fast since many workflows share GitHub's runner IP pool
    (observed: --check failing with 403 on every platform in one push). A GITHUB_TOKEN
    raises that to 5000/hour; these pin that the header is sent to api.github.com when
    a token is present, withheld when it is not, and never sent to the unrelated raw
    CDN host even with a token set."""

    def _urlopen_headers(self, source: str) -> dict[str, str]:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with patch("scripts.sync_c2pa_soft_bindings.urlopen", return_value=response) as mock_urlopen:
            sync._read_bytes(source)
        request = mock_urlopen.call_args[0][0]
        return dict(request.headers)

    def test_sends_bearer_token_to_github_api(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        headers = self._urlopen_headers("https://api.github.com/repos/example/example/commits")
        assert headers.get("Authorization") == "Bearer test-token"

    def test_omits_authorization_without_a_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        headers = self._urlopen_headers("https://api.github.com/repos/example/example/commits")
        assert "Authorization" not in headers

    def test_never_sends_the_token_to_the_raw_cdn_host(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        headers = self._urlopen_headers("https://raw.githubusercontent.com/example/example/main/x.json")
        assert "Authorization" not in headers


@pytest.mark.parametrize("entry", C2PA_SOFT_BINDING_REGISTRY, ids=lambda entry: entry.algorithm)
def test_every_soft_binding_algorithm_keeps_exact_token_boundaries(entry):
    from remove_ai_watermarks._internal.c2pa import soft_binding_registry_entries_in

    token = entry.algorithm.encode()
    assert entry in soft_binding_registry_entries_in(b'"' + token + b'"')
    for prefix, suffix in ((b"prefix", b""), (b"", b".suffix"), (b"-", b""), (b"", b"1")):
        assert entry not in soft_binding_registry_entries_in(prefix + token + suffix)
