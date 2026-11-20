"""API rewriting is opt-in and preserves the shared local acceptance boundary."""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import text as lib
from remove_ai_watermarks import text_api as api


@dataclass
class Endpoint:
    url: str = ""
    requests: list[dict] = field(default_factory=list)
    status: int = 200
    body: object = None
    headers: dict[str, str] = field(default_factory=dict)


def completion(content: str, finish: str = "stop", **message: object) -> dict:
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content, **message}}]}


@pytest.fixture
def endpoint() -> Iterator[Endpoint]:
    state = Endpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
            body = state.body
            if callable(body):
                body = body(payload)
            if body is None:
                body = completion(payload["messages"][1]["content"].replace("sent", "shipped"))
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(state.status)
            for name, value in state.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_endpoint_is_opt_in_and_protected_facts_roundtrip(endpoint: Endpoint, caplog: pytest.LogCaptureFixture) -> None:
    assert raiw.build_text_api_rewriter is lib.build_text_api_rewriter
    assert api.build_text_api_rewriter is lib.build_text_api_rewriter
    caplog.set_level(logging.INFO, logger=api.__name__)
    rewrite = api.build_text_api_rewriter(
        base_url=endpoint.url, model="test-model", api_key="synthetic-secret", max_completion_tokens=75, timeout=5
    )
    assert endpoint.requests == []
    result = lib.remove_text_invisible("The curator Maya sent 12 crates.", rewriter=rewrite)
    assert result.cleaned_text == "The curator Maya shipped 12 crates."
    assert len(endpoint.requests) == 1
    request = endpoint.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer synthetic-secret"
    assert request["payload"]["model"] == "test-model"
    assert request["payload"]["max_completion_tokens"] == 75
    assert request["payload"]["stream"] is False
    messages = request["payload"]["messages"]
    assert messages[0]["content"] == lib._TEXT_REWRITE_SYSTEM_PROMPT
    assert "12" not in messages[1]["content"]
    assert "Maya" not in messages[1]["content"]
    assert "[[RAIW_SPAN_" in messages[1]["content"]
    assert "request endpoint=" in caplog.text
    assert "response status=200" in caplog.text
    assert "synthetic-secret" not in caplog.text
    assert "test-model" in caplog.text


@pytest.mark.parametrize("entry", ["api", "invisible", "all"])
@pytest.mark.parametrize("refuse", [False, True])
def test_strict_residual_resends_only_restored_sentences(
    endpoint: Endpoint, tmp_path, entry: str, refuse: bool
) -> None:
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    original = "The crew counted 12 crates.\n\nStaff will record the seal number in the log."
    destination = tmp_path / "result.txt"
    source = tmp_path / "source.txt"
    source.write_text(original)
    destination.write_bytes(b"existing")
    first = ["The crew counted 13 crates.", "Staff will enter the seal number in the log."]
    result = "The crew tallied 12 crates.\n\nStaff will enter the seal number in the log."
    endpoint.body = lambda payload: completion(
        json.dumps(
            first
            if payload["seed"] == 7291
            else ["The crew tallied 13 crates." if refuse else "The crew tallied 12 crates."]
        ),
        role="assistant",
    )
    if entry == "api":
        callbacks = api.build_text_api_strict_residual_rewriter(base_url=endpoint.url, model="local", seed=7291)
        if refuse:
            with pytest.raises(RuntimeError, match="Strict residual rewrite"):
                lib.remove_text_invisible(source, destination, **callbacks)
        else:
            assert lib.remove_text_invisible(source, destination, **callbacks).cleaned_text == result
    else:
        arguments = [
            "text",
            entry,
            str(source),
            "-o",
            str(destination),
            "--rewriter-backend",
            "ollama",
            "--rewriter-model",
            "local",
            "--ollama-url",
            endpoint.url,
            "--seed",
            "7291",
            "--strict-residual",
        ]
        if entry == "all":
            arguments.append("--invisible")
        cli = CliRunner().invoke(main, arguments)
        assert cli.exit_code == int(refuse), cli.output
    assert destination.read_bytes() == (b"existing" if refuse else result.encode())
    requests = [item["payload"] for item in endpoint.requests]
    assert [item["seed"] for item in requests] == [7291, 84237]
    assert json.loads(requests[1]["messages"][1]["content"])["original"] == ["The crew counted 12 crates."]


def test_api_preserves_locked_technical_text(endpoint: Endpoint) -> None:
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    result = lib.remove_text_invisible("Maya sent 12 crates, tagged `x=1`.", rewriter=rewrite)
    assert result.cleaned_text == "Maya shipped 12 crates, tagged `x=1`."
    assert "Authorization" not in endpoint.requests[0]["headers"]


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff",
        [],
        {},
        {"choices": []},
        {"choices": [{}, {}]},
        {"choices": completion("Tools require repairs.")["choices"] * 2},
        completion("Tools require repairs.", finish="length"),
        completion("Tools require repairs.", finish="content_filter"),
        completion("Tools require repairs.", finish="tool_calls"),
        completion("Tools require repairs.", refusal="Cannot comply"),
        completion("Tools require repairs.", tool_calls=[{"id": "call"}]),
        completion("Tools require repairs.", function_call={"name": "unexpected"}),
        completion("Tools require repairs.", role="tool"),
        completion(""),
        completion(None),
        completion("Tools need repairs."),
        completion("Инструменты нуждаются в ремонте."),
    ],
)
def test_incomplete_malformed_or_unacceptable_output_is_not_returned(endpoint: Endpoint, body: object) -> None:
    endpoint.body = body
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError):
        rewrite("Tools need repairs.")
    assert len(endpoint.requests) == 1


def test_missing_or_invented_fact_is_rejected(endpoint: Endpoint) -> None:
    endpoint.body = completion("The curator Maya shipped 99 crates.")
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError, match="acceptance"):
        rewrite("The curator Maya sent 12 crates.")


@pytest.mark.parametrize("status", [429, 500])
def test_http_failures_do_not_retry_or_expose_credentials(
    endpoint: Endpoint, status: int, caplog: pytest.LogCaptureFixture
) -> None:
    key = 'synthetic-"secret'
    endpoint.status = status
    endpoint.body = {"error": key}
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model", api_key=key)
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        rewrite("Tools need repairs.")
    assert len(endpoint.requests) == 1
    assert key not in str(error.value)
    assert key not in caplog.text
    assert json.dumps(key)[1:-1] not in caplog.text
    assert "[redacted]" in caplog.text
    assert "test-model" in caplog.text


def test_redirect_does_not_forward_the_request_or_key(endpoint: Endpoint) -> None:
    endpoint.status = 307
    endpoint.headers["Location"] = endpoint.url + "/another"
    endpoint.body = {"redirect": True}
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model", api_key="synthetic-secret")
    with pytest.raises(RuntimeError, match="HTTP 307"):
        rewrite("Tools need repairs.")
    assert len(endpoint.requests) == 1


def test_reflected_credential_is_never_returned(endpoint: Endpoint, caplog: pytest.LogCaptureFixture) -> None:
    endpoint.body = completion("Tools require repairs. synthetic-secret")
    caplog.set_level(logging.INFO, logger=api.__name__)
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model", api_key="synthetic-secret")
    with pytest.raises(RuntimeError, match="configured credential") as error:
        rewrite("Tools need repairs.")
    assert "synthetic-secret" not in caplog.text + str(error.value)


def test_json_escaped_credential_is_never_returned(endpoint: Endpoint) -> None:
    endpoint.body = (
        json.dumps(completion("Tools require repairs. synthetic-secret"))
        .replace("synthetic-secret", r"synthetic-\u0073ecret")
        .encode()
    )
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model", api_key="synthetic-secret")
    with pytest.raises(RuntimeError, match="configured credential"):
        rewrite("Tools need repairs.")


def test_json_escaped_error_credential_is_redacted(endpoint: Endpoint, caplog: pytest.LogCaptureFixture) -> None:
    endpoint.status = 500
    endpoint.body = rb'{"error":"synthetic-\u0073ecret"}'
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model", api_key="synthetic-secret")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        rewrite("Tools need repairs.")
    assert "[redacted]" in caplog.text


@pytest.mark.parametrize(
    "settings",
    [
        {"base_url": "file:///tmp/model"},
        {"base_url": "https://user:secret@example.test/v1"},
        {"base_url": "https://example.test/v1?token=secret"},
        {"base_url": "https://example.test/v1#secret"},
        {"model": ""},
        {"api_key": "secret\r\nInjected: true"},
        {"api_key": "secret\x00"},
        {"max_completion_tokens": True},
        {"max_completion_tokens": 0},
        {"timeout": True},
        {"timeout": 0},
        {"timeout": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected_before_requests(endpoint: Endpoint, settings: dict) -> None:
    values = {"base_url": endpoint.url, "model": "test-model", **settings}
    with pytest.raises(ValueError, match=r"base_url|model|api_key|max_completion_tokens|timeout"):
        api.build_text_api_rewriter(**values)
    assert endpoint.requests == []


def test_missing_language_dependency_is_rejected_before_requests(
    endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="text-api"):
        api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    assert endpoint.requests == []


def test_default_local_path_does_not_contact_endpoint(endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    def local_factory(model: str, **options: object):
        assert model == lib.DEFAULT_TEXT_REWRITER_MODEL
        assert options["backend"] == "transformers"
        return lambda text: text.replace("sent", "shipped")

    monkeypatch.setattr(lib, "_build_local_lm_rewriter", local_factory)
    result = lib.remove_text_invisible("Maya sent crates.")
    assert result.cleaned_text == "Maya shipped crates."
    assert endpoint.requests == []


def test_explicit_outer_validator_retains_existing_retry_policy(endpoint: Endpoint) -> None:
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError, match="validator rejected"):
        lib.remove_text_invisible("Tools sent crates.", rewriter=rewrite, rewrite_validator=lambda *_args: False)
    assert len(endpoint.requests) == 3


def test_transport_failure_cannot_expose_key(
    endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic-secret")

    monkeypatch.setattr(api.HTTPConnection, "request", fail)
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model", api_key="synthetic-secret")
    with pytest.raises(RuntimeError, match="request failed") as error:
        rewrite("Tools need repairs.")
    assert "synthetic-secret" not in caplog.text + str(error.value)
    assert endpoint.requests == []


def test_api_uses_packaged_language_model_for_same_language_output(endpoint: Endpoint) -> None:
    source = "The volunteers sent fresh fruit to the kitchen, where the staff prepared lunch for the visitors."
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    assert rewrite(source) == source.replace("sent", "shipped")


def test_api_rejects_confident_language_change_within_latin_script(endpoint: Endpoint) -> None:
    source = "The volunteers sent fresh fruit to the kitchen, where the staff prepared lunch for the visitors."
    endpoint.body = completion(
        "Los voluntarios enviaron fruta fresca a la cocina, donde el personal preparo el almuerzo para los visitantes."
    )
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError, match="acceptance"):
        rewrite(source)
    assert len(endpoint.requests) == 1


def test_api_uses_membership_guard_after_restoring_fact_tokens(endpoint: Endpoint) -> None:
    endpoint.body = lambda payload: completion(payload["messages"][1]["content"].replace("part of", "critical of"))
    rewrite = api.build_text_api_rewriter(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError, match="acceptance"):
        rewrite("This blog is part of Transforming Manufacturing.")


def test_document_api_draft_correction_and_review_use_only_two_posts(endpoint, tmp_path):
    draft = ("Staff used the room.", "Guests borrowed books.")
    corrected = ("Guests used the room.", "Readers checked out books.")
    seen = []

    def respond(payload):
        data = json.loads(payload["messages"][1]["content"])
        assert data["original"] == ["Visitors used the room.", "Readers borrowed books."]
        if "rejected" in data:
            assert data["rejected"] == list(draft)
            return completion(json.dumps(corrected))
        return completion(json.dumps(draft))

    def validate(_, candidate):
        seen.append(candidate)
        return candidate.startswith(corrected[0])

    endpoint.body = respond
    rewrite, repair = api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model")
    assert not endpoint.requests
    output = tmp_path / "output.txt"
    result = lib.remove_text_invisible(
        "Visitors used the room.\r\n\r\n`value = 12`\r\n\r\nReaders borrowed books.",
        output,
        document_rewriter=rewrite,
        document_validator=validate,
        document_repairer=repair,
    )
    assert len(endpoint.requests) == 2
    for request in endpoint.requests:
        prompt = request["payload"]["messages"][0]["content"]
        assert "The input contains exactly 2 prose slots. Output exactly 2 JSON strings." in prompt
    assert seen == ["\r\n\r\n`value = 12`\r\n\r\n".join(parts) for parts in (draft, corrected)]
    assert result.cleaned_text == seen[1]
    assert output.read_bytes() == seen[1].encode()


@pytest.mark.parametrize(
    "response",
    ['["Only one."]', '{"0":"Staff used the room.","1":"Guests borrowed books."}', '[1,"Text"]'],
)
def test_document_api_declines_malformed_slots(endpoint, response):
    endpoint.body = completion(response)
    rewrite, _ = api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError, match="one nonempty string per prose span"):
        rewrite(("Visitors used the room.", "Readers borrowed books."))
    assert len(endpoint.requests) == 1


def test_document_api_can_repair_a_fact_vetoed_by_the_reviewer(endpoint, tmp_path):
    def respond(payload):
        data = json.loads(payload["messages"][1]["content"])
        count = 12 if "rejected" in data else 19
        return completion(json.dumps([f"The team shipped {count} files."]))

    endpoint.body = respond
    rewrite, repair = api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model")
    output = tmp_path / "output.txt"
    lib.remove_text_invisible(
        "The team sent 12 files.",
        output,
        document_rewriter=rewrite,
        document_validator=lib.verify_text_rewrite_facts,
        document_repairer=repair,
    )
    assert len(endpoint.requests) == 2
    assert output.read_text() == "The team shipped 12 files."


def test_document_api_accepts_retained_slots_without_repair(endpoint):
    endpoint.body = completion('["Guests used the room.","Readers borrowed books."]')
    rewrite, repair = api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model")
    result = lib.remove_text_invisible(
        "Visitors used the room.\n\nReaders borrowed books.",
        document_rewriter=rewrite,
        document_validator=lambda *_: True,
        document_repairer=repair,
    )
    assert result.rewritten_spans == result.retained_prose_spans == 1
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize("document", [False, True])
def test_api_can_send_max_tokens_without_another_limit_field(endpoint, document):
    settings = {
        "base_url": endpoint.url,
        "model": "test-model",
        "max_completion_tokens": 42,
        "completion_token_field": "max_tokens",
    }
    if document:
        endpoint.body = completion('["Guests used the room."]')
        rewrite_document, _ = api.build_text_api_document_rewriters(**settings)
        rewrite_document(("Visitors used the room.",))
    else:
        endpoint.body = completion("Guests used the room.")
        api.build_text_api_rewriter(**settings)("Visitors used the room.")
    payload = endpoint.requests[0]["payload"]
    assert payload["max_tokens"] == 42
    assert "max_completion_tokens" not in payload
    assert len(endpoint.requests) == 1


def test_api_rejects_unknown_completion_token_field(endpoint):
    with pytest.raises(ValueError, match="completion_token_field"):
        api.build_chat_completion_client(base_url=endpoint.url, model="test-model", completion_token_field="other")
    assert not endpoint.requests


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_document_api_accepts_one_fenced_json_array(endpoint, newline):
    endpoint.body = completion(newline.join((" ```json", '["Guests used the room."]', "```  ")))
    rewrite, _ = api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model")
    result = lib.remove_text_invisible("Visitors used the room.", document_rewriter=rewrite)
    assert result.cleaned_text == "Guests used the room."


@pytest.mark.parametrize(
    "response",
    [
        'Here is the result:\n```json\n["Text."]\n```',
        '```json\n["Text."]\n```\nMore text',
        '```json\n["Text."]\n```\n```json\n["Other."]\n```',
    ],
)
def test_document_api_rejects_content_outside_single_json_array(endpoint, response):
    endpoint.body = completion(response)
    rewrite, _ = api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        rewrite(("Visitors used the room.",))


def test_document_api_sends_explicit_recast_and_sampling_settings(endpoint):
    endpoint.body = completion('["Guests used the room."]')
    rewrite, _ = api.build_text_api_document_rewriters(
        base_url=endpoint.url, model="test-model", rewrite_style="recast", temperature=0.0, seed=7291
    )
    assert not endpoint.requests
    assert rewrite(("Visitors used the room.",)) == ("Guests used the room.",)
    request = endpoint.requests[0]["payload"]
    prompt = request["messages"][0]["content"]
    assert "Rewrite every prose slot in fresh wording" in prompt
    assert prompt.index("Rewrite every prose slot") < prompt.index("The input contains exactly 1 prose slots")
    assert request["temperature"] == 0.0
    assert request["seed"] == 7291


def test_document_api_rejects_unknown_rewrite_style(endpoint):
    with pytest.raises(ValueError, match="rewrite_style"):
        api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model", rewrite_style="other")
    assert not endpoint.requests


def test_sentence_api_preserves_separators_and_sends_fixed_shape_for_repair(endpoint):
    original = ("Dr. Reed uses a ruler.\tThe block is 1.5 meters wide.", "They note the height.")
    drafts = ("Dr. Reed measures with a ruler.", "The block measures 1.5 meters across.", "They record the height.")
    rejected = (drafts[0] + "\t" + drafts[1], drafts[2])

    def respond(payload):
        data = json.loads(payload["messages"][1]["content"])
        assert data["original"] == ["Dr. Reed uses a ruler.", "The block is 1.5 meters wide.", "They note the height."]
        if "rejected" in data:
            assert data["rejected"] == list(drafts)
        options = payload["response_format"]
        assert options["type"] == "json_schema"
        assert options["json_schema"]["strict"] is True
        assert options["json_schema"]["schema"] == {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 3,
            "maxItems": 3,
        }
        assert payload["reasoning_effort"] == "none"
        return completion(json.dumps(drafts))

    endpoint.body = respond
    rewrite, repair = api.build_text_api_document_rewriters(
        base_url=endpoint.url,
        model="test-model",
        rewrite_style="sentence",
        structured_output=True,
        disable_reasoning=True,
    )
    assert rewrite(original) == rejected
    assert repair(original, rejected) == rejected
    with pytest.raises(ValueError, match="same sentence slots"):
        repair(original, ("Merged sentence.", "They record the height."))
    assert len(endpoint.requests) == 2


@pytest.mark.parametrize("extra", ["Two sentences. Here.", "Extra\nline", "Extra\rline", "The room empties"])
def test_sentence_api_rejects_changed_boundaries(endpoint, extra):
    endpoint.body = completion(json.dumps([extra, "The door swings shut."]))
    rewrite, _ = api.build_text_api_document_rewriters(
        base_url=endpoint.url, model="test-model", rewrite_style="sentence"
    )
    with pytest.raises(RuntimeError, match="sentence boundary"):
        rewrite(("The room emptied. The door closed.",))
    assert len(endpoint.requests) == 1
    assert "response_format" not in endpoint.requests[0]["payload"]


def test_schema_rejection_does_not_retry_without_schema(endpoint):
    endpoint.status = 400
    rewrite, _ = api.build_text_api_document_rewriters(
        base_url=endpoint.url, model="test-model", structured_output=True
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        rewrite(("Visitors used the room.",))
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize("stage", [["invisible"], ["all", "--invisible"]])
def test_cli_local_ollama_uses_document_sentence_api(endpoint, tmp_path, stage):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "article.txt"
    output = tmp_path / "result.txt"
    source.write_bytes(b"Readers visit the library.\tIt closes at noon.\r\n")
    endpoint.body = completion('["People visit the library.", "It shuts at noon."]')
    result = CliRunner().invoke(
        main,
        [
            "text",
            *stage,
            str(source),
            "-o",
            str(output),
            "--rewriter-backend",
            "ollama",
            "--rewriter-model",
            "local-test",
            "--ollama-url",
            endpoint.url,
            "--seed",
            "7291",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b"People visit the library.\tIt shuts at noon.\r\n"
    assert len(endpoint.requests) == 1
    assert "Authorization" not in endpoint.requests[0]["headers"]
    assert endpoint.requests[0]["payload"]["seed"] == 7291
    assert endpoint.requests[0]["payload"]["response_format"]["type"] == "json_schema"


def test_cli_local_ollama_refuses_nonlocal_endpoint_and_fact_changes(endpoint, tmp_path):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "article.txt"
    output = tmp_path / "result.txt"
    source.write_text("Readers borrowed 12 books.")
    output.write_bytes(b"existing")
    command = [
        "text",
        "invisible",
        str(source),
        "-o",
        str(output),
        "--rewriter-backend",
        "ollama",
        "--rewriter-model",
        "local-test",
        "--ollama-url",
    ]
    runner = CliRunner()
    nonlocal_result = runner.invoke(main, [*command, "http://example.com/v1"])
    assert nonlocal_result.exit_code == 2
    assert "loopback" in nonlocal_result.output
    assert not endpoint.requests
    endpoint.body = completion('["Readers checked out 19 books."]')
    result = runner.invoke(main, [*command, endpoint.url])
    assert result.exit_code == 1
    assert "document_validator" in result.output
    assert len(endpoint.requests) == 1
    assert output.read_bytes() == b"existing"


def test_cli_local_ollama_retains_unchanged_spans_only_on_request(endpoint, tmp_path):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "article.txt"
    output = tmp_path / "result.txt"
    source.write_text("Readers visit the library.\n\nIt closes at noon.")
    endpoint.body = completion('["People visit the library.", "It closes at noon."]')
    command = [
        "text",
        "invisible",
        str(source),
        "-o",
        str(output),
        "--rewriter-backend",
        "ollama",
        "--rewriter-model",
        "local-test",
        "--ollama-url",
        endpoint.url,
    ]
    runner = CliRunner()
    strict = runner.invoke(main, command)
    assert strict.exit_code == 1
    assert not output.exists()
    partial = runner.invoke(main, [*command, "--allow-partial"])
    assert partial.exit_code == 0, partial.output
    assert "PARTIAL" in partial.output
    assert output.read_text() == "People visit the library.\n\nIt closes at noon."


@pytest.mark.parametrize(
    ("setting", "value"),
    [("temperature", True), ("temperature", float("nan")), ("temperature", -0.1), ("seed", True), ("seed", 1.5)],
)
def test_document_api_rejects_invalid_sampling_settings(endpoint, setting, value):
    with pytest.raises(ValueError, match=setting):
        api.build_text_api_document_rewriters(base_url=endpoint.url, model="test-model", **{setting: value})
    assert not endpoint.requests
