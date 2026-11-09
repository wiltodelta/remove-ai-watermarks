"""The user-facing docs must name every knob the public surface actually has.

The code-to-code seams are already guarded: `test_api.py` compares
`InvisibleOptions` against the engine signature field by field, and
`test_cli.py` drives the commands. Nothing guarded the code-to-DOCS seam, and it
drifted exactly the way an unguarded seam does -- silently, one addition at a
time. When this was first measured, `docs/cli.md` named none of
`--adaptive-polish`, `--controlnet-scale`, `--detect/--no-detect`, `--humanize`,
`--strength`, `--tile-size`, `--tile-overlap` or `--unsharp`, and
`docs/python-api.md` named 9 of the 16 `InvisibleOptions` fields.

These tests read the real Click tree and the real dataclass rather than a
hand-kept list, so a knob added tomorrow fails here until someone documents it.
Both accept EITHER spelling of a boolean pair: `--keep-metadata` documents
`--strip-metadata/--keep-metadata` perfectly well, and requiring the on-form
would force noise into the guides.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from urllib.parse import unquote

import click

from remove_ai_watermarks.api import InvisibleOptions
from remove_ai_watermarks.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]

# The pages a user is sent to for command and API reference. A knob mentioned
# only in a research-archive page is not documented for the purpose of this gate.
CLI_PAGES = ("docs/cli.md", "README.md", "docs/installation.md")
API_PAGES = ("docs/python-api.md", "README.md")


def _text(pages: tuple[str, ...]) -> str:
    return "".join((REPO_ROOT / page).read_text(encoding="utf-8") for page in pages)


def _cli_options() -> list[tuple[str, list[str]]]:
    """Every ``(command, spellings)`` pair in the tree.

    Group by the Click parameter itself, never by string surgery on the names: a
    boolean pair can be spelled anything (``--strip-metadata/--keep-metadata``,
    ``--keep-standard/--remove-all``), so deriving the partner from a ``--no-``
    prefix silently splits those pairs into two groups and demands both halves.
    """
    found: list[tuple[str, list[str]]] = []

    def walk(command: click.Command, prefix: str = "") -> None:
        if isinstance(command, click.Group):
            for name, sub in sorted(command.commands.items()):
                walk(sub, f"{prefix}{name} ")
            return
        for param in command.params:
            if not isinstance(param, click.Option):
                continue
            names = [o for o in param.opts + param.secondary_opts if o.startswith("--")]
            if names and names != ["--help"]:
                found.append((prefix.strip() or "(root)", names))

    walk(main)
    return found


def test_every_cli_option_is_named_in_the_command_docs() -> None:
    docs = _text(CLI_PAGES)
    undocumented = [
        f"{command}: {' / '.join(names)}"
        for command, names in _cli_options()
        if not any(name in docs for name in names)
    ]
    assert not undocumented, (
        "CLI options absent from " + ", ".join(CLI_PAGES) + ":\n  " + "\n  ".join(sorted(set(undocumented)))
    )


def test_every_invisible_option_field_is_named_in_the_api_docs() -> None:
    docs = _text(API_PAGES)
    fields = [f.name for f in dataclasses.fields(InvisibleOptions)]
    assert fields, "InvisibleOptions has no fields; the guard would pass vacuously"
    undocumented = [name for name in fields if name not in docs]
    assert not undocumented, (
        "InvisibleOptions fields absent from "
        + ", ".join(API_PAGES)
        + f": {undocumented}. Document the field or explain in python-api.md why it is internal."
    )


_RAW_PREFIX = "https://raw.githubusercontent.com/wiltodelta/remove-ai-watermarks/main/"
_BLOB_PREFIX = "https://github.com/wiltodelta/remove-ai-watermarks/blob/main/"
_MARKDOWN_LINK_RE = re.compile(r"(!?\[[^\[\]]*\]|\])\(([^)]+)\)")


def test_readme_links_and_images_work_on_pypi() -> None:
    """PyPI renders README.md at pypi.org/project/remove-ai-watermarks/ without rewriting relative URLs."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    matches = list(_MARKDOWN_LINK_RE.finditer(readme))
    assert matches, "README.md has no Markdown links or images; the guard would pass vacuously"

    problems: list[str] = []
    for match in matches:
        label, target = match.group(1), match.group(2).strip()
        is_image = label.startswith("!")
        if not target.startswith(("https://", "#")):
            problems.append(f"{label}({target}): relative URL breaks on PyPI")
            continue
        if is_image and target.startswith(_BLOB_PREFIX):
            problems.append(
                f"{label}({target}): image uses github.com/blob HTML URL instead of raw.githubusercontent.com"
            )
            continue
        for prefix in (_RAW_PREFIX, _BLOB_PREFIX):
            if target.startswith(prefix):
                rel_path = unquote(target.removeprefix(prefix).split("#", maxsplit=1)[0])
                if not (REPO_ROOT / rel_path).is_file():
                    problems.append(f"{label}({target}): target file {rel_path!r} does not exist in repo")

    assert not problems, "Broken or PyPI-incompatible links in README.md:\n  " + "\n  ".join(problems)
