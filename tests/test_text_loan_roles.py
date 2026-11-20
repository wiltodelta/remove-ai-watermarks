"""Replay frozen dependency trees without importing an optional language model."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from remove_ai_watermarks.text_roles import verify_text_rewrite_loan_roles

FIXTURE = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic/text-loan-role-parses.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


class Token:
    def __init__(self, doc, index, annotation):
        self.doc = doc
        self.i = index
        self.lemma_, self.pos_, self.dep_, head = annotation.split("/")
        self.head_index = int(head)

    @property
    def head(self):
        return self.doc.tokens[self.head_index]

    @property
    def children(self):
        return [t for t in self.doc if t.head_index == self.i and t.i != self.i]

    @property
    def lefts(self):
        return [t for t in self.children if t.i < self.i]

    @property
    def conjuncts(self):
        root = self
        while root.dep_ == "conj":
            root = root.head
        pending = [root]
        result = []
        seen = set()
        while pending:
            token = pending.pop()
            if token.i in seen:
                continue
            seen.add(token.i)
            if token.i != self.i:
                result.append(token)
            pending.extend(t for t in token.children if t.dep_ == "conj")
        return result


class Doc:
    def __init__(self, annotation, missing=None, partial=False, text=""):
        self.text = text
        self.tokens = [Token(self, i, row) for i, row in enumerate(annotation.split())]
        self.missing = missing
        self.partial = partial

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def has_annotation(self, name, *, require_complete=False):
        return name != self.missing and not (require_complete and self.partial)


class Parser:
    lang = "en"

    def __init__(self, row, missing=None):
        self.mapping = {row["source"]: row["source_parse"], row["candidate"]: row["candidate_parse"]}
        self.missing = missing

    def pipe(self, texts):
        return (Doc(self.mapping[text], self.missing, text=text) for text in texts)


@pytest.mark.parametrize("row", CASES, ids=lambda row: row["id"])
@pytest.mark.parametrize("reverse", [False, True])
def test_selected_reversals_and_equivalent_voices(row, reverse):
    source, candidate = row["source"], row["candidate"]
    if reverse:
        source, candidate = candidate, source
    assert verify_text_rewrite_loan_roles(source, candidate, parser=Parser(row)) is row["no_selected_reversal"]


@pytest.mark.parametrize("missing", ["DEP", "POS", "LEMMA"])
def test_incomplete_parse_is_an_error(missing):
    row = CASES[0]
    with pytest.raises(ValueError, match="complete dependency"):
        verify_text_rewrite_loan_roles(row["source"], row["candidate"], parser=Parser(row, missing))


def test_non_english_parser_is_an_error():
    parser = Parser(CASES[0])
    parser.lang = "ru"
    with pytest.raises(ValueError, match="English parser"):
        verify_text_rewrite_loan_roles(CASES[0]["source"], CASES[0]["candidate"], parser=parser)


@pytest.mark.parametrize(("source", "candidate"), [("", "Text."), ("Text.", "  ")])
def test_empty_prose_is_an_error(source, candidate):
    with pytest.raises(ValueError, match="nonempty"):
        verify_text_rewrite_loan_roles(source, candidate, parser=Parser(CASES[0]))


@pytest.mark.parametrize("count", [1, 3])
def test_parser_document_count_is_checked(count):
    parser = Parser(CASES[0])
    doc = Doc(CASES[0]["source_parse"])
    parser.pipe = lambda _texts: [doc] * count
    with pytest.raises(ValueError, match="one complete document per input"):
        verify_text_rewrite_loan_roles(CASES[0]["source"], CASES[0]["candidate"], parser=parser)


def test_parser_failure_propagates():
    parser = Parser(CASES[0])

    def broken(_texts):
        raise RuntimeError("Parser unavailable")

    parser.pipe = broken
    with pytest.raises(RuntimeError, match="Parser unavailable"):
        verify_text_rewrite_loan_roles(CASES[0]["source"], CASES[0]["candidate"], parser=parser)


def test_partially_annotated_tokens_are_an_error():
    parser = Parser(CASES[0])
    parser.pipe = lambda texts: [Doc(parser.mapping[text], partial=True, text=text) for text in texts]
    with pytest.raises(ValueError, match="complete dependency"):
        verify_text_rewrite_loan_roles(CASES[0]["source"], CASES[0]["candidate"], parser=parser)


def test_empty_parsed_documents_are_an_error():
    parser = Parser(CASES[0])
    parser.pipe = lambda texts: [Doc("", text=text) for text in texts]
    with pytest.raises(ValueError, match="complete dependency"):
        verify_text_rewrite_loan_roles(CASES[0]["source"], CASES[0]["candidate"], parser=parser)


def test_stale_or_truncated_parse_is_an_error():
    parser = Parser(CASES[0])
    parser.pipe = lambda _texts: [Doc(CASES[0]["source_parse"], text="Truncated.")] * 2
    with pytest.raises(ValueError, match="exact input text"):
        verify_text_rewrite_loan_roles(CASES[0]["source"], CASES[0]["candidate"], parser=parser)


def test_import_without_optional_dependencies():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SOURCE_ROOT)!r})\n"
        "from remove_ai_watermarks.text_roles import verify_text_rewrite_loan_roles\n"
        "assert callable(verify_text_rewrite_loan_roles)\n"
    )
    result = subprocess.run(  # noqa: S603 - Fixed local interpreter and synthetic test code, no shell.
        [sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
