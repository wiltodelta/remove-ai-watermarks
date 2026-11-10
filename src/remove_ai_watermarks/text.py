"""Text provenance inspection, character-steganography scrubbing, and token-watermark removal.

Covers two distinct technical layers:

1. **Deterministic text provenance and character steganography** (standard library +
   ``_internal/c2pa.py``, no ML or pixel extras required):
   - C2PA 2.3/2.4 Appendix A.8 plain-text ``C2PATextManifestWrapper`` (``C2PATXT\\x00``
     in trailing Variation Selectors ``U+FE00..U+FE0F`` and ``U+E0100..U+E01EF``).
   - C2PA 2.3/2.4 Appendix A.9 structured-text ASCII-armored manifests
     (``-----BEGIN C2PA MANIFEST----- ... -----END C2PA MANIFEST-----``).
   - C2PA 2.3/2.4 Appendix A.7 HTML ``<script type="application/c2pa">`` and
     ``<link rel="c2pa-manifest">`` elements inside ``<head>``.
   - ``VSRMark`` (``com.cognitive-proof.vsrmark.1``, C2PA soft-binding registry ID 39)
     interleaved Variation Selector payloads (``VSR1`` header + ``zlib.crc32``).
   - Unicode Tags (``U+E0000..U+E007F``) and zero-width / Variation Selector
     steganography, preserving valid Emoji presentation selectors (``U+FE0E``/``U+FE0F``)
     and Emoji ``U+200D`` ZWJ sequences.

2. **Token-level statistical LLM watermarks** (Google ``SynthID-Text``, Anthropic
   Claude Fable 5.1/5.2 & Mythos 5.1/5.2, OpenAI GPT-6 Astra ``textGrain``, KGW
   Red-Green List, and Unigram):
   - Span-protected prose regeneration (:func:`remove_text_invisible`) that locks
     low-entropy technical and structural spans verbatim (fenced code blocks, inline
     code, LaTeX math, URLs, file paths, HTML tags, Markdown headings) and rewrites
     only unlocked prose spans with an unwatermarked local causal LM or a
     caller-supplied rewriter.
   - Keyed statistical scorers (:func:`score_synthid_text`, :func:`score_gumbel_text`,
     :func:`score_kgw_text`, :func:`score_unigram_text`) for local calibration and
     verification when the deployment's watermark key is known.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import logging
import math
import re
import statistics
import struct
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from remove_ai_watermarks._internal.c2pa import c2pa_vendor_platform, claim_generator_platform, parse_c2pa_chunk
from remove_ai_watermarks._internal.utils import staged_sibling

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".html",
        ".htm",
        ".xml",
        ".svg",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".tex",
    }
)

C2PATXT_MAGIC = b"C2PATXT\x00"
C2PATXT_VERSION = 1
_C2PATXT_HEADER = struct.Struct("!8sBI")

VSRMARK_MAGIC = b"VSR1"
VSRMARK_VERSION = 1
VSRMARK_ALGORITHM = "com.cognitive-proof.vsrmark.1"
_VSRMARK_HEADER = struct.Struct(">4sBH")
_VSRMARK_CRC = struct.Struct(">I")

C2PA_ASCII_ARMOR_BEGIN = "-----BEGIN C2PA MANIFEST-----"
C2PA_ASCII_ARMOR_END = "-----END C2PA MANIFEST-----"

DEFAULT_TEXT_REWRITER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ALPHA_001_Z = 2.3263478740408408

_ZERO_WIDTH_CODEPOINTS: frozenset[int] = frozenset(
    {
        0x00AD,  # SOFT HYPHEN
        0x034F,  # COMBINING GRAPHEME JOINER
        0x061C,  # ARABIC LETTER MARK
        0x115F,  # HANGUL CHOSEONG FILLER
        0x1160,  # HANGUL JUNGSEONG FILLER
        0x17B4,  # KHMER VOWEL INHERENT AQ
        0x17B5,  # KHMER VOWEL INHERENT AA
        0x180B,  # MONGOLIAN FREE VARIATION SELECTOR ONE
        0x180C,  # MONGOLIAN FREE VARIATION SELECTOR TWO
        0x180D,  # MONGOLIAN FREE VARIATION SELECTOR THREE
        0x180E,  # MONGOLIAN VOWEL SEPARATOR
        0x180F,  # MONGOLIAN FREE VARIATION SELECTOR FOUR
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER (preserved inside joining scripts)
        0x200D,  # ZERO WIDTH JOINER (preserved inside valid Emoji ZWJ and joining scripts)
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        0x202A,  # LEFT-TO-RIGHT EMBEDDING
        0x202B,  # RIGHT-TO-LEFT EMBEDDING
        0x202C,  # POP DIRECTIONAL FORMATTING
        0x202D,  # LEFT-TO-RIGHT OVERRIDE
        0x202E,  # RIGHT-TO-LEFT OVERRIDE
        0x2060,  # WORD JOINER
        0x2061,  # FUNCTION APPLICATION
        0x2062,  # INVISIBLE TIMES
        0x2063,  # INVISIBLE SEPARATOR
        0x2064,  # INVISIBLE PLUS
        0x2065,  # UNASSIGNED DEFAULT IGNORABLE
        0x2066,  # LEFT-TO-RIGHT ISOLATE
        0x2067,  # RIGHT-TO-LEFT ISOLATE
        0x2068,  # FIRST STRONG ISOLATE
        0x2069,  # POP DIRECTIONAL ISOLATE
        0x206A,  # INHIBIT SYMMETRIC SWAPPING
        0x206B,  # ACTIVATE SYMMETRIC SWAPPING
        0x206C,  # INHIBIT ARABIC FORM SHAPING
        0x206D,  # ACTIVATE ARABIC FORM SHAPING
        0x206E,  # NATIONAL DIGIT SHAPES
        0x206F,  # NOMINAL DIGIT SHAPES
        0x3164,  # HANGUL FILLER
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
        0xFFA0,  # HALFWIDTH HANGUL FILLER
        0xFFF9,  # INTERLINEAR ANNOTATION ANCHOR
        0xFFFA,  # INTERLINEAR ANNOTATION SEPARATOR
        0xFFFB,  # INTERLINEAR ANNOTATION TERMINATOR
        0x1BCA0,  # SHORTHAND FORMAT LETTER OVERLAP
        0x1BCA1,  # SHORTHAND FORMAT CONTINUING OVERLAP
        0x1BCA2,  # SHORTHAND FORMAT DOWN STEP
        0x1BCA3,  # SHORTHAND FORMAT UP STEP
        0x1D159,  # MUSICAL SYMBOL NULL NOTEHEAD
        0x1D173,  # MUSICAL SYMBOL BEGIN BEAM
        0x1D174,  # MUSICAL SYMBOL END BEAM
        0x1D175,  # MUSICAL SYMBOL BEGIN TIE
        0x1D176,  # MUSICAL SYMBOL END TIE
        0x1D177,  # MUSICAL SYMBOL BEGIN SLUR
        0x1D178,  # MUSICAL SYMBOL END SLUR
        0x1D179,  # MUSICAL SYMBOL BEGIN PHRASE
        0x1D17A,  # MUSICAL SYMBOL END PHRASE
        0xE0000,  # UNASSIGNED TAG PREFIX
    }
)

_NON_JOINING_ARABIC_CPS: frozenset[int] = frozenset(
    {
        0x0622,  # ALEF WITH MADDA ABOVE
        0x0623,  # ALEF WITH HAMZA ABOVE
        0x0624,  # WAW WITH HAMZA ABOVE
        0x0625,  # ALEF WITH HAMZA BELOW
        0x0627,  # ALEF
        0x062F,  # DAL
        0x0630,  # THAL
        0x0631,  # REH
        0x0632,  # ZAIN
        0x0648,  # WAW
        0x0698,  # JEH
    }
)

_INNAMARK_SEPARATOR = "\u2004"
_INNAMARK_ALPHABET: tuple[str, str, str, str] = ("\u2008", "\u2009", "\u202f", "\u205f")
_INNAMARK_DIGIT_MAP: dict[str, int] = {ch: idx for idx, ch in enumerate(_INNAMARK_ALPHABET)}
_INNAMARK_CHARS: frozenset[str] = frozenset((*_INNAMARK_ALPHABET, _INNAMARK_SEPARATOR))

_STEGCLOAK_ZWC: tuple[str, ...] = ("\u200c", "\u200d", "\u2061", "\u2062", "\u2063", "\u2064")
_STEGCLOAK_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u200c", "\u200d"),
    ("\u200c", "\u2061"),
    ("\u200c", "\u2062"),
    ("\u200d", "\u2061"),
    ("\u200d", "\u2062"),
    ("\u2061", "\u2062"),
)
_STEGCLOAK_DIBITS: dict[str, str] = {
    "\u200c": "00",
    "\u200d": "01",
    "\u2061": "10",
    "\u2062": "11",
}

_EXOTIC_SPACE_CPS: frozenset[int] = frozenset(
    {
        0x00A0,  # NO-BREAK SPACE
        0x1680,  # OGHAM SPACE MARK
        0x2000,  # EN QUAD
        0x2001,  # EM QUAD
        0x2002,  # EN SPACE
        0x2003,  # EM SPACE
        0x2004,  # THREE-PER-EM SPACE
        0x2005,  # FOUR-PER-EM SPACE
        0x2006,  # SIX-PER-EM SPACE
        0x2007,  # FIGURE SPACE
        0x2008,  # PUNCTUATION SPACE
        0x2009,  # THIN SPACE
        0x200A,  # HAIR SPACE
        0x202F,  # NARROW NO-BREAK SPACE
        0x205F,  # MEDIUM MATHEMATICAL SPACE
        0x3000,  # IDEOGRAPHIC SPACE
    }
)

_INVISIBLE_NAMED_ENTITIES: dict[str, str] = {
    "zerowidthspace": "\u200b",
    "zwnj": "\u200c",
    "zwj": "\u200d",
    "lrm": "\u200e",
    "rlm": "\u200f",
    "shy": "\u00ad",
    "nobreak": "\u2060",
    "wordjoiner": "\u2060",
    "applyfunction": "\u2061",
    "af": "\u2061",
    "invisibletimes": "\u2062",
    "it": "\u2062",
    "invisiblecomma": "\u2063",
    "ic": "\u2063",
}
_HTML_ENTITY_RE = re.compile(r"&(?:#(?P<dec>\d{1,7})|#x(?P<hex>[0-9A-Fa-f]{1,6})|(?P<name>[A-Za-z]{2,24}));")
_ASSISTANT_CITATION_RE = re.compile(
    r"\ue200[^\ue200\ue201]{0,160}\ue201"
    r"|\bcite(?:turn\d+(?:search|news|view|image|file|fetch|forecast|sports|finance|academia|product|map)\d+)+\b"
    r"|\[cite(?:_start|_end|:\s*\d+(?:\s*,\s*\d+)*)\]"
    r"|\[span_\d+\](?:\((?:start|end)_span\))?"
    r"|\bgrok_render_citation_card_json\b(?:=\{[^}\n]*\})?"
)

_CONFUSABLE_TO_LATIN: dict[str, str] = {
    # Cyrillic uppercase & lowercase lookalikes
    "\u0410": "A",
    "\u0412": "B",
    "\u0421": "C",
    "\u0415": "E",
    "\u041d": "H",
    "\u0406": "I",
    "\u0408": "J",
    "\u041a": "K",
    "\u041c": "M",
    "\u041e": "O",
    "\u0420": "P",
    "\u0405": "S",
    "\u0422": "T",
    "\u0425": "X",
    "\u04ae": "Y",
    "\u0430": "a",
    "\u0441": "c",
    "\u0501": "d",
    "\u0435": "e",
    "\u04bb": "h",
    "\u0456": "i",
    "\u0458": "j",
    "\u043e": "o",
    "\u0440": "p",
    "\u0455": "s",
    "\u0443": "y",
    "\u0445": "x",
    # Greek uppercase & lowercase lookalikes
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0396": "Z",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a5": "Y",
    "\u03a7": "X",
    "\u03b1": "a",
    "\u03b9": "i",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03bd": "v",
    # Armenian, Cherokee, and Latin IPA confusables
    "\u0578": "n",
    "\u0585": "o",
    "\u13a0": "D",
    "\u13a1": "R",
    "\u13a2": "T",
    "\u13aa": "A",
    "\u13ab": "J",
    "\u13ac": "E",
    "\u13b3": "W",
    "\u13b7": "M",
    "\u13bb": "H",
    "\u13c0": "G",
    "\u13c2": "h",
    "\u13c3": "Z",
    "\u13cf": "b",
    "\u13d9": "V",
    "\u13de": "L",
    "\u13df": "C",
    "\u13e2": "P",
    "\u13e6": "K",
    "\u13f4": "B",
    "\u0251": "a",
    "\u0261": "g",
    "\u1d0f": "o",
    "\u212a": "K",
    "\u0423": "Y",
    "\u043a": "k",
    # Fullwidth Latin uppercase & lowercase lookalikes (U+FF21..U+FF3A, U+FF41..U+FF5A)
    **{chr(0xFF21 + i): chr(ord("A") + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(ord("a") + i) for i in range(26)},
}

_CONFUSABLE_TO_CYRILLIC: dict[str, str] = {
    # ASCII Latin uppercase & lowercase lookalikes injected into Cyrillic words
    "A": "\u0410",
    "B": "\u0412",
    "C": "\u0421",
    "E": "\u0415",
    "H": "\u041d",
    "I": "\u0406",
    "J": "\u0408",
    "K": "\u041a",
    "M": "\u041c",
    "O": "\u041e",
    "P": "\u0420",
    "S": "\u0405",
    "T": "\u0422",
    "X": "\u0425",
    "Y": "\u0423",
    "a": "\u0430",
    "c": "\u0441",
    "e": "\u0435",
    "i": "\u0456",
    "j": "\u0458",
    "k": "\u043a",
    "o": "\u043e",
    "p": "\u0440",
    "s": "\u0455",
    "x": "\u0445",
    "y": "\u0443",
    # Greek uppercase & lowercase lookalikes injected into Cyrillic words
    "\u0391": "\u0410",
    "\u0392": "\u0412",
    "\u0395": "\u0415",
    "\u0397": "\u041d",
    "\u0399": "\u0406",
    "\u039a": "\u041a",
    "\u039c": "\u041c",
    "\u039f": "\u041e",
    "\u03a1": "\u0420",
    "\u03a4": "\u0422",
    "\u03a5": "\u0423",
    "\u03a7": "\u0425",
    "\u03b1": "\u0430",
    "\u03b9": "\u0456",
    "\u03bf": "\u043e",
    "\u03c1": "\u0440",
}

_TEXTGRAIN_SUB_PART = 0x1F3D5B79
_TEXTGRAIN_SUB_COST = 0x2C4E6A8F
_TEXTGRAIN_SUB_COL = 0x3B5D7F91
_TEXTSEAL_DUAL_KEY_OFFSET = 12345
_STD_NORMAL = statistics.NormalDist()

_C2PA_ASCII_ARMOR_COMMENT_RE = re.compile(
    r"(?:<!--[ \t]*\r?\n?)?"
    r"(?:^[ \t]*(?:#|//|\*)?[ \t]*)?-----BEGIN C2PA MANIFEST-----[ \t]*\r?\n"
    r"(?P<body>.*?)"
    r"(?:^[ \t]*(?:#|//|\*)?[ \t]*)?-----END C2PA MANIFEST-----[ \t]*(?:\r?\n?[ \t]*-->)?[ \t]*(?:\r?\n)?",
    re.DOTALL | re.MULTILINE,
)
_HTML_HEAD_RE = re.compile(r"<head\b[^>]*>(?P<head>.*?)</head\s*>", re.IGNORECASE | re.DOTALL)
_HTML_C2PA_SCRIPT_RE = re.compile(
    r"[ \t]*<script\b[^>]*\btype\s*=\s*['\"]application/c2pa['\"][^>]*>(?P<payload>.*?)</script\s*>[ \t]*\r?\n?",
    re.IGNORECASE | re.DOTALL,
)
_HTML_C2PA_LINK_RE = re.compile(
    r"[ \t]*<link\b[^>]*\brel\s*=\s*['\"]c2pa-manifest['\"][^>]*>[ \t]*\r?\n?",
    re.IGNORECASE | re.DOTALL,
)
_HTML_HREF_RE = re.compile(r"\bhref\s*=\s*['\"](?P<href>[^'\"]+)['\"]", re.IGNORECASE)

_STATISTICAL_SAMPLING_CAVEAT = (
    "Token-level statistical LLM watermarks (Google SynthID-Text, Anthropic Claude Fable 5.1/5.2 "
    "and Mythos 5.1/5.2, OpenAI GPT-6 Astra textGrain, KGW, Unigram) are keyed in the token-sampling "
    "distribution and cannot be verified locally without the provider's secret key."
)


def is_text_invisible_available() -> bool:
    """Return whether the optional local LM runtime (``torch`` + ``transformers``) is installed."""
    from remove_ai_watermarks.optional_deps import module_available

    return module_available("torch", "transformers")


def _require_text_invisible_runtime() -> None:
    """Raise with a shell-quoted install hint when the local LM runtime is absent."""
    if not is_text_invisible_available():
        raise RuntimeError(
            "Token-level text watermark removal requires 'torch' and 'transformers' "
            "(install 'remove-ai-watermarks[diffusion]' or pass a custom rewriter=... callable)."
        )


def _read_text_input(text_or_path: str | Path) -> tuple[Path | None, str]:
    """Return ``(source_path, text_content)`` for a path or in-memory string."""
    if isinstance(text_or_path, Path):
        return text_or_path, text_or_path.read_text(encoding="utf-8")
    return None, text_or_path


def _resolve_output_path(source: Path | None, output_path: str | Path | None) -> Path | None:
    """Resolve and validate the output path before running any transformation."""
    resolved: Path | None = None
    if output_path is not None:
        resolved = Path(output_path)
    elif source is not None:
        resolved = source.with_stem(f"{source.stem}_clean")
    if resolved is not None and source is not None and resolved.resolve() == source.resolve():
        raise ValueError("Output path must differ from source text file")
    return resolved


def _prepare_text_io(
    text_or_path: str | Path,
    output_path: str | Path | None,
) -> tuple[Path | None, Path | None, str]:
    """Validate output destination before reading and return ``(source, resolved_output, text)``."""
    source: Path | None = text_or_path if isinstance(text_or_path, Path) else None
    resolved_output = _resolve_output_path(source, output_path)
    _, text = _read_text_input(text_or_path)
    return source, resolved_output, text


def _write_output_text(output_path: Path | None, text: str) -> None:
    """Atomically write ``text`` to ``output_path`` when an output destination is active."""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with staged_sibling(output_path) as staged:
            staged.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class TextProvenanceReport:
    """Locally verifiable provenance and character-steganography signals in text."""

    source: Path | None
    is_ai_generated: Literal[True] | None
    confidence: Literal["high", "medium", "unknown"]
    platform: str | None
    has_ai_metadata: bool
    signals: tuple[str, ...]
    metadata_markers: dict[str, str]
    c2pa_info: dict[str, Any]
    vsrmark_payload: str | None
    unicode_tags_payload: str | None
    stego_codepoint_count: int
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class TextMetadataResult:
    """Result of stripping deterministic C2PA wrappers and character steganography from text."""

    source: Path | None
    output: Path | None
    cleaned_text: str
    removed_signals: tuple[str, ...]
    detected_markers: dict[str, str]
    remaining_markers: dict[str, str]
    removed_codepoints: int


@dataclass(frozen=True)
class TextSpan:
    """A contiguous segment of a text document classified as locked or rewritable prose."""

    kind: Literal[
        "prose",
        "fenced_code",
        "inline_code",
        "math_block",
        "math_inline",
        "html_block",
        "html_tag",
        "markdown_structure",
        "url",
        "path",
        "whitespace",
    ]
    text: str
    locked: bool


@dataclass(frozen=True)
class TextInvisibleResult:
    """Result of span-protected prose regeneration for token-level LLM watermarks."""

    source: Path | None
    output: Path | None
    cleaned_text: str
    rewritten_spans: int
    locked_spans: int
    locked_characters: int
    rewritten_characters: int
    model_id: str

    @property
    def token_watermark_action(self) -> Literal["regenerated", "no_prose_spans"]:
        """Describe whether high-entropy prose spans were regenerated."""
        return "regenerated" if self.rewritten_spans > 0 else "no_prose_spans"


@dataclass(frozen=True)
class TextAllResult:
    """Combined result of deterministic text metadata scrubbing and prose regeneration."""

    source: Path | None
    output: Path | None
    cleaned_text: str
    removed_signals: tuple[str, ...]
    detected_markers: dict[str, str]
    remaining_markers: dict[str, str]
    removed_codepoints: int
    rewritten_spans: int
    locked_spans: int
    model_id: str | None

    @property
    def token_watermark_action(self) -> Literal["not_run", "regenerated", "no_prose_spans"]:
        """Describe whether token-watermark prose regeneration ran."""
        if self.model_id is None:
            return "not_run"
        return "regenerated" if self.rewritten_spans > 0 else "no_prose_spans"


@dataclass(frozen=True)
class TextWatermarkScore:
    """Keyed statistical watermark detector result for a token sequence."""

    scheme: Literal["synthid", "gumbel", "textgrain", "textseal2", "kgw", "unigram"]
    z_score: float
    p_value: float
    detected: bool
    num_tokens: int
    threshold_z: float


# ── Variation Selector byte bijection (C2PATXT & VSRMark) ──


def byte_to_variation_selector(byte_val: int) -> str:
    """Map one byte ``0..255`` to its Unicode Variation Selector codepoint."""
    if not 0 <= byte_val <= 255:
        raise ValueError(f"byte value must be in 0..255, got {byte_val}")
    if byte_val < 16:
        return chr(0xFE00 + byte_val)
    return chr(0xE0100 + (byte_val - 16))


def variation_selector_to_byte(char: str) -> int | None:
    """Map one Unicode Variation Selector character back to ``0..255``, or ``None``."""
    if len(char) != 1:
        return None
    cp = ord(char)
    if 0xFE00 <= cp <= 0xFE0F:
        return cp - 0xFE00
    if 0xE0100 <= cp <= 0xE01EF:
        return cp - 0xE0100 + 16
    return None


def encode_variation_selectors(payload: bytes) -> str:
    """Encode arbitrary bytes as a contiguous sequence of Unicode Variation Selectors."""
    return "".join(byte_to_variation_selector(b) for b in payload)


def decode_variation_selectors(chars: str) -> bytes:
    """Decode every Variation Selector in ``chars`` into a byte string."""
    out = bytearray()
    for ch in chars:
        byte_val = variation_selector_to_byte(ch)
        if byte_val is not None:
            out.append(byte_val)
    return bytes(out)


# ── Encoders for C2PATXT, C2PA ASCII Armor, VSRMark, and Unicode Tags ──


def encode_c2patxt(text: str, manifest_bytes: bytes, *, padding_bytes: int = 0) -> str:
    """Embed ``manifest_bytes`` at the end of ``text`` per C2PA 2.3/2.4 Appendix A.8."""
    if padding_bytes < 0:
        raise ValueError("padding_bytes must be non-negative")
    normalized = unicodedata.normalize("NFC", _strip_c2patxt_suffix(text)[0])
    header = _C2PATXT_HEADER.pack(C2PATXT_MAGIC, C2PATXT_VERSION, len(manifest_bytes))
    wrapper_bytes = header + manifest_bytes + (b"\x00" * padding_bytes)
    return normalized + "\ufeff" + encode_variation_selectors(wrapper_bytes)


def encode_c2pa_ascii_armor(
    manifest_bytes_or_uri: bytes | str,
    *,
    comment_style: Literal["none", "html", "hash"] = "none",
) -> str:
    """Format a C2PA Appendix A.9 ASCII-armored manifest block."""
    if isinstance(manifest_bytes_or_uri, bytes):
        b64 = base64.b64encode(manifest_bytes_or_uri).decode("ascii")
        body = f"data:application/c2pa;base64,{b64}"
    else:
        body = manifest_bytes_or_uri.strip()
    block = f"{C2PA_ASCII_ARMOR_BEGIN}\n{body}\n{C2PA_ASCII_ARMOR_END}"
    if comment_style == "html":
        return f"<!--\n{block}\n-->\n"
    if comment_style == "hash":
        return "\n".join(f"# {line}" for line in block.splitlines()) + "\n"
    return block + "\n"


def encode_vsrmark(text: str, payload: str | bytes) -> str:
    """Interleave a ``VSRMark`` (``VSR1`` + CRC32) payload into ``text``."""
    payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if len(payload_bytes) > 65535:
        raise ValueError("VSRMark payload cannot exceed 65535 bytes")
    header = _VSRMARK_HEADER.pack(VSRMARK_MAGIC, VSRMARK_VERSION, len(payload_bytes))
    body = header + payload_bytes
    crc = zlib.crc32(body) & 0xFFFFFFFF
    frame = body + _VSRMARK_CRC.pack(crc)

    carrier_indices = [
        idx
        for idx, ch in enumerate(text)
        if variation_selector_to_byte(ch) is None and not ch.isspace() and ord(ch) not in _ZERO_WIDTH_CODEPOINTS
    ]
    if len(carrier_indices) < len(frame):
        raise ValueError(
            f"VSRMark frame requires at least {len(frame)} visible carrier characters, got {len(carrier_indices)}"
        )

    out: list[str] = []
    frame_pos = 0
    carrier_set = set(carrier_indices[: len(frame)])
    for idx, ch in enumerate(text):
        out.append(ch)
        if idx in carrier_set:
            out.append(byte_to_variation_selector(frame[frame_pos]))
            frame_pos += 1
    return "".join(out)


def encode_unicode_tags(text: str, hidden_ascii: str) -> str:
    """Append a hidden ASCII string encoded in the Unicode Tags block (``U+E0000..U+E007F``)."""
    encoded: list[str] = ["\U000e0001"]
    for ch in hidden_ascii:
        cp = ord(ch)
        if not 0x20 <= cp <= 0x7E:
            raise ValueError(f"Unicode Tags payload must be printable ASCII (0x20..0x7E), got {ch!r}")
        encoded.append(chr(0xE0000 + cp))
    encoded.append("\U000e007f")
    return text + "".join(encoded)


def encode_innamark(
    text: str,
    payload: str | bytes,
    *,
    compressed: bool = False,
    crc32: bool = True,
) -> str:
    """Embed an ``Innamark`` (``arXiv:2502.12710``) whitespace watermark into ``text``.

    Replaces ASCII spaces (``U+0020``) with the base-4 LSB-first transcoding alphabet
    (``U+2008``, ``U+2009``, ``U+202F``, ``U+205F``) separated across repetitions by
    ``U+2004`` (``THREE-PER-EM SPACE``), wrapped in a standard ``InnamarkTag`` header.
    """
    raw_body = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if compressed:
        compressor = zlib.compressobj(level=6, wbits=-zlib.MAX_WBITS)
        body = compressor.compress(raw_body) + compressor.flush()
    else:
        body = raw_body

    type_tag = 0x20 | (0x40 if compressed else 0x00) | (0x10 if crc32 else 0x00)
    total_len = 1 + 4 + (4 if crc32 else 0) + len(body)
    if crc32:
        placeholder = bytes([type_tag]) + struct.pack("<II", total_len, 0) + body
        checksum = zlib.crc32(placeholder) & 0xFFFFFFFF
        tag_bytes = bytes([type_tag]) + struct.pack("<II", total_len, checksum) + body
    else:
        tag_bytes = bytes([type_tag]) + struct.pack("<I", total_len) + body

    transcoded_chars: list[str] = [_INNAMARK_SEPARATOR]
    for byte_val in tag_bytes:
        val = byte_val
        for _ in range(4):
            transcoded_chars.append(_INNAMARK_ALPHABET[val & 3])
            val >>= 2

    space_positions = [idx for idx, ch in enumerate(text) if ch == " "]
    if len(space_positions) < len(transcoded_chars):
        raise ValueError(
            f"Innamark frame requires at least {len(transcoded_chars)} ASCII spaces, got {len(space_positions)}"
        )

    out = list(text)
    cycle_len = len(transcoded_chars)
    for pos_idx, text_idx in enumerate(space_positions):
        out[text_idx] = transcoded_chars[pos_idx % cycle_len]
    return "".join(out)


def _decode_printable_utf8(raw: bytes, *, min_printable_ratio: float = 0.8) -> str | None:
    """Decode ``raw`` as UTF-8 when at least ``min_printable_ratio`` characters are printable."""
    if not raw:
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not decoded:
        return None
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in ("\n", "\r", "\t"))
    if printable / len(decoded) < min_printable_ratio:
        return None
    return decoded


def _parse_innamark_tag(raw_bytes: bytes) -> str | None:
    """Parse and verify a standard ``InnamarkTag`` frame from decoded base-4 whitespace bytes."""
    if len(raw_bytes) < 2:
        return None
    type_tag = raw_bytes[0]
    if (type_tag & 0x80) != 0 or (type_tag & 0x07) != 0:
        return None
    is_compressed = (type_tag & 0x40) != 0
    is_sized = (type_tag & 0x20) != 0
    hash_bits = type_tag & 0x18
    if hash_bits == 0x18:
        return None

    offset = 1
    tag_end = len(raw_bytes)
    if is_sized:
        if len(raw_bytes) < offset + 4:
            return None
        (sized_len,) = struct.unpack_from("<I", raw_bytes, offset)
        if sized_len < offset + 4 or sized_len > len(raw_bytes):
            return None
        tag_end = sized_len
        offset += 4

    tag_slice = raw_bytes[:tag_end]
    if hash_bits == 0x10:
        if len(tag_slice) < offset + 4:
            return None
        (expected_crc,) = struct.unpack_from("<I", tag_slice, offset)
        zeroed = tag_slice[:offset] + b"\x00\x00\x00\x00" + tag_slice[offset + 4 :]
        if (zlib.crc32(zeroed) & 0xFFFFFFFF) != expected_crc:
            return None
        offset += 4
    elif hash_bits == 0x08:
        if len(tag_slice) < offset + 32:
            return None
        expected_digest = tag_slice[offset : offset + 32]
        masked = tag_slice[:offset] + (b"\xff" * 32) + tag_slice[offset + 32 :]
        if hashlib.sha3_256(masked).digest() != expected_digest:
            return None
        offset += 32

    body = tag_slice[offset:]
    if is_compressed:
        try:
            body = zlib.decompress(body, -zlib.MAX_WBITS)
        except zlib.error:
            return None
    return _decode_printable_utf8(body, min_printable_ratio=0.75)


def _detect_innamark_payload(text: str) -> str | None:
    """Detect and decode an ``Innamark`` (``arXiv:2502.12710``) whitespace watermark."""
    space_stream = "".join(ch for ch in text if ch in _INNAMARK_CHARS)
    if len(space_stream) < 8:
        return None

    segments = [seg for seg in space_stream.split(_INNAMARK_SEPARATOR) if len(seg) >= 4]
    if not segments:
        return None

    decoded_candidates: list[str] = []
    for seg in segments:
        usable_len = (len(seg) // 4) * 4
        raw = bytes(
            _INNAMARK_DIGIT_MAP[seg[i]]
            | (_INNAMARK_DIGIT_MAP[seg[i + 1]] << 2)
            | (_INNAMARK_DIGIT_MAP[seg[i + 2]] << 4)
            | (_INNAMARK_DIGIT_MAP[seg[i + 3]] << 6)
            for i in range(0, usable_len, 4)
        )
        parsed_tag = _parse_innamark_tag(raw)
        if parsed_tag is not None:
            decoded_candidates.append(parsed_tag)
        elif len(segments) >= 2:
            plain = _decode_printable_utf8(raw)
            if plain is not None:
                decoded_candidates.append(plain)

    if not decoded_candidates:
        return None
    counts: dict[str, int] = {}
    for item in decoded_candidates:
        counts[item] = counts.get(item, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _detect_stegcloak_stream(text: str) -> str | None:
    """Detect a ``KuroLabs/stegcloak`` zero-width stream and decode its framing metadata or payload."""
    for match in re.finditer(r"[\u200c\u200d\u2061\u2062\u2063\u2064]{6,}", text):
        run = match.group(0)
        if not any(ch in run for ch in ("\u2061", "\u2062", "\u2063", "\u2064")):
            continue
        flag_idx = _STEGCLOAK_ZWC.index(run[0])
        zwc_a, zwc_b = _STEGCLOAK_PAIRS[flag_idx]
        expanded = run[1:].replace("\u2064", zwc_b + zwc_b).replace("\u2063", zwc_a + zwc_a)
        mode_ch = expanded[0]
        if mode_ch not in ("\u200c", "\u200d", "\u2061"):
            continue
        body_zwc = expanded[1:]
        if len(body_zwc) % 4 != 0:
            continue
        bits = "".join(_STEGCLOAK_DIBITS[ch] for ch in body_zwc)
        raw_bytes = bytes((int(bits[i : i + 8], 2) ^ 0xFF) for i in range(0, len(bits), 8))
        if mode_ch == "\u200c" and len(raw_bytes) > 40:
            return f"aes-256-ctr+hmac-sha256 ({len(raw_bytes) - 40} bytes ciphertext)"
        if mode_ch == "\u200d" and len(raw_bytes) > 8:
            return f"aes-256-ctr ({len(raw_bytes) - 8} bytes ciphertext)"
        if mode_ch == "\u2061":
            plain = _decode_printable_utf8(raw_bytes, min_printable_ratio=0.7)
            if plain is not None:
                return f"unencrypted:{plain}"
            return f"unencrypted ({len(raw_bytes)} bytes lzutf8)"
    return None


def _detect_zero_width_binary_payload(text: str) -> dict[str, str]:
    """Decode covert payloads from ``GroundTruth-MCP``, ``zengkan0703``, ``BitJacker``, and ``phantomoji``."""
    found: dict[str, str] = {}

    gt_match = re.search(r"[\u2061\u2062]{64}", text)
    if gt_match is not None:
        bits = "".join("0" if ch == "\u2061" else "1" for ch in gt_match.group(0))
        install_id = int(bits[:32], 2)
        nonce = int(bits[32:], 2)
        found["groundtruth_mcp"] = f"install_id=0x{install_id:08x}, nonce=0x{nonce:08x}"

    for match in re.finditer(r"[\u200b\u200c\u200d\u2060\ufeff]{8,}", text):
        run = match.group(0)
        distinct = sorted(set(run))
        if "\u200d" in distinct and set(distinct).issubset({"\u200b", "\u200c", "\u200d"}):
            parts = [p for p in run.split("\u200d") if p]
            if parts and all(len(p) <= 16 and set(p).issubset({"\u200b", "\u200c"}) for p in parts):
                for zero_ch in ("\u200b", "\u200c"):
                    try:
                        chars = [chr(int("".join("0" if c == zero_ch else "1" for c in p), 2)) for p in parts]
                    except ValueError:
                        continue
                    candidate = "".join(chars)
                    if candidate and sum(1 for c in candidate if c.isprintable()) / len(candidate) >= 0.8:
                        found.setdefault("zero_width_binary", candidate)
                        break

        if len(distinct) == 2 and len(run) >= 8:
            byte_len = (len(run) // 8) * 8
            for zero_ch in distinct:
                raw = bytes(
                    int("".join("0" if c == zero_ch else "1" for c in run[i : i + 8]), 2) for i in range(0, byte_len, 8)
                )
                decoded = _decode_printable_utf8(raw)
                if decoded is not None:
                    found.setdefault("zero_width_binary", decoded)
                    break

        if len(run) >= 8 and len(run) % 4 == 0 and set(distinct).issubset({"\u200b", "\u200c", "\u200d", "\u2060"}):
            dibit_map = {"\u200b": 0, "\u200c": 1, "\u200d": 2, "\u2060": 3}
            raw = bytes(
                (dibit_map[run[i]] << 6)
                | (dibit_map[run[i + 1]] << 4)
                | (dibit_map[run[i + 2]] << 2)
                | dibit_map[run[i + 3]]
                for i in range(0, len(run), 4)
            )
            decoded = _decode_printable_utf8(raw)
            if decoded is not None:
                found.setdefault("zero_width_binary", decoded)

    return found


def _detect_variation_selector_stego_payload(
    text: str,
    *,
    excluded_indices: frozenset[int] = frozenset(),
) -> dict[str, str]:
    """Decode ``paulgb/emoji-encoder``, ``prospectorlabs/subtext-spec``, and ``chinmay29hub/stegmoji`` VS runs."""
    found: dict[str, str] = {}
    runs: list[list[int]] = []
    current: list[int] = []

    for idx, ch in enumerate(text):
        if idx in excluded_indices:
            if current:
                runs.append(current)
                current = []
            continue
        byte_val = variation_selector_to_byte(ch)
        if byte_val is not None:
            current.append(byte_val)
        elif ch == "\u200a" and current:
            # prospectorlabs/subtext-spec inserts U+200A every 30 VS codepoints for LINE compatibility
            continue
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    for run in runs:
        if len(run) < 3:
            continue
        raw = bytes(run)
        if len(raw) >= 8 and raw[:2] == b"\xab\xcd" and raw[6] == 0x01:
            body_len = int.from_bytes(raw[2:6], "big")
            flags = raw[7]
            body = raw[8 : 8 + body_len]
            if flags == 0x00 and len(body) >= 2:
                decoded = _decode_printable_utf8(body[1:])
                if decoded is not None:
                    found["subtext"] = decoded
                    continue
            found["subtext"] = f"Subtext v1 frame (flags=0x{flags:02x}, len={body_len})"
            continue

        if len(run) >= 16 and set(run).issubset({14, 15}):
            # chinmay29hub/stegmoji binary channel: VS15 (14) -> 0, VS16 (15) -> 1
            byte_len = (len(run) // 8) * 8
            bit_bytes = bytes(
                int("".join("0" if b == 14 else "1" for b in run[i : i + 8]), 2) for i in range(0, byte_len, 8)
            )
            if len(bit_bytes) >= 2 and bit_bytes[0] in (0x00, 0x01):
                payload_bytes = bit_bytes[1:]
                if bit_bytes[0] == 0x01:
                    try:
                        payload_bytes = zlib.decompress(payload_bytes)
                    except zlib.error:
                        payload_bytes = b""
                decoded = _decode_printable_utf8(payload_bytes)
                if decoded is not None:
                    found["stegmoji"] = decoded
                    continue

        decoded = _decode_printable_utf8(raw)
        if decoded is not None:
            found.setdefault("variation_selector", decoded)

    return found


def decode_stego_payloads(text: str) -> dict[str, str]:
    """Decode covert steganography payloads across Innamark, StegCloak, VS encoders, and ZW binary."""
    payloads: dict[str, str] = {}
    after_c2patxt, c2patxt_match = _strip_c2patxt_suffix(text)
    if c2patxt_match is not None:
        payloads["c2patxt"] = f"{len(c2patxt_match.manifest_bytes)} bytes JUMBF"
    vsrmark = _detect_vsrmark(after_c2patxt)
    excluded: frozenset[int] = vsrmark.indices if vsrmark is not None else frozenset()
    if vsrmark is not None:
        payloads["vsrmark"] = vsrmark.payload

    innamark_payload = _detect_innamark_payload(after_c2patxt)
    if innamark_payload is not None:
        payloads["innamark"] = innamark_payload

    stegcloak_info = _detect_stegcloak_stream(after_c2patxt)
    if stegcloak_info is not None:
        payloads["stegcloak"] = stegcloak_info

    payloads.update(_detect_zero_width_binary_payload(after_c2patxt))
    payloads.update(_detect_variation_selector_stego_payload(after_c2patxt, excluded_indices=excluded))
    return payloads


# ── Internal inspection and stripping helpers ──


def _inspect_jumbf_bytes(payload: bytes) -> dict[str, Any]:
    """Extract normalized C2PA metadata fields from raw JUMBF manifest bytes."""
    if not payload:
        return {}
    info: dict[str, Any] = {}
    parse_c2pa_chunk(payload, info)
    return info


@dataclass(frozen=True)
class _C2paTxtMatch:
    manifest_bytes: bytes
    padding_bytes: int
    had_bom_prefix: bool
    removed_codepoints: int
    c2pa_info: dict[str, Any]


def _strip_c2patxt_suffix(text: str) -> tuple[str, _C2paTxtMatch | None]:
    """Detect and strip a trailing C2PA Appendix A.8 ``C2PATXT\\x00`` wrapper."""
    end = len(text)
    while end > 0 and text[end - 1] in ("\r", "\n"):
        end -= 1
    trailing_newlines = text[end:]
    cursor = end
    while cursor > 0 and variation_selector_to_byte(text[cursor - 1]) is not None:
        cursor -= 1

    vs_run = text[cursor:end]
    if len(vs_run) < _C2PATXT_HEADER.size:
        return text, None

    raw = decode_variation_selectors(vs_run)
    magic, version, manifest_len = _C2PATXT_HEADER.unpack_from(raw, 0)
    if magic != C2PATXT_MAGIC or version != C2PATXT_VERSION:
        return text, None
    if _C2PATXT_HEADER.size + manifest_len > len(raw):
        return text, None

    manifest_bytes = raw[_C2PATXT_HEADER.size : _C2PATXT_HEADER.size + manifest_len]
    padding_len = len(raw) - (_C2PATXT_HEADER.size + manifest_len)
    had_bom = cursor > 0 and text[cursor - 1] == "\ufeff"
    prefix_end = cursor - 1 if had_bom else cursor
    removed_cp = end - prefix_end
    cleaned = text[:prefix_end] + trailing_newlines
    c2pa_info = _inspect_jumbf_bytes(manifest_bytes)
    return (
        cleaned,
        _C2paTxtMatch(
            manifest_bytes=manifest_bytes,
            padding_bytes=padding_len,
            had_bom_prefix=had_bom,
            removed_codepoints=removed_cp,
            c2pa_info=c2pa_info,
        ),
    )


@dataclass(frozen=True)
class _AsciiArmorMatch:
    manifest_bytes: bytes | None
    manifest_uri: str | None
    c2pa_info: dict[str, Any]
    count: int


def _decode_inline_c2pa_payload(raw_body: str) -> tuple[bytes | None, str | None]:
    """Parse an inline ``data:application/c2pa;base64,...`` or external URI body."""
    cleaned_lines: list[str] = []
    for line in raw_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        elif stripped.startswith("//"):
            stripped = stripped[2:].strip()
        elif stripped.startswith("*"):
            stripped = stripped.lstrip("*").strip()
        if stripped:
            cleaned_lines.append(stripped)
    joined = "".join(cleaned_lines)
    if not joined:
        return None, None
    if joined.startswith(("http://", "https://", "urn:")):
        return None, joined
    prefix = "data:application/c2pa;base64,"
    b64_text = joined[len(prefix) :] if joined.lower().startswith(prefix) else joined
    try:
        decoded = base64.b64decode(b64_text, validate=True)
    except (binascii.Error, ValueError):
        return None, joined
    return (decoded if decoded else None), None


def _strip_c2pa_ascii_armor(text: str) -> tuple[str, _AsciiArmorMatch | None]:
    """Detect and strip C2PA Appendix A.9 ``-----BEGIN C2PA MANIFEST-----`` blocks."""
    if C2PA_ASCII_ARMOR_BEGIN not in text:
        return text, None

    manifest_bytes: bytes | None = None
    manifest_uri: str | None = None
    c2pa_info: dict[str, Any] = {}
    count = 0

    def _record_body(body: str) -> str:
        nonlocal manifest_bytes, manifest_uri, c2pa_info, count
        count += 1
        decoded, uri = _decode_inline_c2pa_payload(body)
        if decoded is not None and manifest_bytes is None:
            manifest_bytes = decoded
            c2pa_info = _inspect_jumbf_bytes(decoded)
        if uri is not None and manifest_uri is None:
            manifest_uri = uri
        return ""

    cleaned = _C2PA_ASCII_ARMOR_COMMENT_RE.sub(lambda m: _record_body(m.group("body")), text)
    if count == 0:
        return text, None
    return (
        cleaned,
        _AsciiArmorMatch(
            manifest_bytes=manifest_bytes,
            manifest_uri=manifest_uri,
            c2pa_info=c2pa_info,
            count=count,
        ),
    )


@dataclass(frozen=True)
class _HtmlC2paMatch:
    manifest_bytes: bytes | None
    manifest_uri: str | None
    c2pa_info: dict[str, Any]
    element_count: int
    outside_head: bool


def _strip_html_c2pa(text: str) -> tuple[str, _HtmlC2paMatch | None]:
    """Detect and strip C2PA Appendix A.7 ``<script type="application/c2pa">`` and ``<link rel="c2pa-manifest">``."""
    lowered = text.lower()
    if "application/c2pa" not in lowered and "c2pa-manifest" not in lowered:
        return text, None

    manifest_bytes: bytes | None = None
    manifest_uri: str | None = None
    c2pa_info: dict[str, Any] = {}
    element_count = 0

    head_match = _HTML_HEAD_RE.search(text)
    head_span = head_match.span() if head_match is not None else None
    outside_head = False

    def _on_script(match: re.Match[str]) -> str:
        nonlocal manifest_bytes, manifest_uri, c2pa_info, element_count, outside_head
        element_count += 1
        if head_span is None or not (head_span[0] <= match.start() and match.end() <= head_span[1]):
            outside_head = True
        decoded, uri = _decode_inline_c2pa_payload(match.group("payload"))
        if decoded is not None and manifest_bytes is None:
            manifest_bytes = decoded
            c2pa_info = _inspect_jumbf_bytes(decoded)
        if uri is not None and manifest_uri is None:
            manifest_uri = uri
        return ""

    def _on_link(match: re.Match[str]) -> str:
        nonlocal manifest_uri, element_count, outside_head
        element_count += 1
        if head_span is None or not (head_span[0] <= match.start() and match.end() <= head_span[1]):
            outside_head = True
        href_match = _HTML_HREF_RE.search(match.group(0))
        if href_match is not None and manifest_uri is None:
            manifest_uri = href_match.group("href")
        return ""

    cleaned = _HTML_C2PA_SCRIPT_RE.sub(_on_script, text)
    cleaned = _HTML_C2PA_LINK_RE.sub(_on_link, cleaned)
    if element_count == 0:
        return text, None
    return (
        cleaned,
        _HtmlC2paMatch(
            manifest_bytes=manifest_bytes,
            manifest_uri=manifest_uri,
            c2pa_info=c2pa_info,
            element_count=element_count,
            outside_head=outside_head,
        ),
    )


@dataclass(frozen=True)
class _VsrMarkMatch:
    payload: str
    raw_payload: bytes
    crc32: int
    indices: frozenset[int]


def _detect_vsrmark(text: str) -> _VsrMarkMatch | None:
    """Scan interleaved Variation Selectors in ``text`` for a valid ``VSR1`` + CRC32 frame."""
    vs_positions: list[int] = []
    vs_bytes = bytearray()
    for idx, ch in enumerate(text):
        byte_val = variation_selector_to_byte(ch)
        if byte_val is not None:
            vs_positions.append(idx)
            vs_bytes.append(byte_val)

    min_frame_len = _VSRMARK_HEADER.size + _VSRMARK_CRC.size
    if len(vs_bytes) < min_frame_len:
        return None

    stream = bytes(vs_bytes)
    search_pos = 0
    while True:
        offset = stream.find(VSRMARK_MAGIC, search_pos)
        if offset < 0 or offset + min_frame_len > len(stream):
            return None
        magic, version, length = _VSRMARK_HEADER.unpack_from(stream, offset)
        frame_end = offset + _VSRMARK_HEADER.size + length + _VSRMARK_CRC.size
        if magic == VSRMARK_MAGIC and version == VSRMARK_VERSION and frame_end <= len(stream):
            body = stream[offset : offset + _VSRMARK_HEADER.size + length]
            (expected_crc,) = _VSRMARK_CRC.unpack_from(stream, offset + _VSRMARK_HEADER.size + length)
            actual_crc = zlib.crc32(body) & 0xFFFFFFFF
            if actual_crc == expected_crc:
                raw_payload = stream[offset + _VSRMARK_HEADER.size : offset + _VSRMARK_HEADER.size + length]
                decoded_payload = raw_payload.decode("utf-8", errors="replace")
                indices = frozenset(vs_positions[offset:frame_end])
                return _VsrMarkMatch(
                    payload=decoded_payload,
                    raw_payload=raw_payload,
                    crc32=expected_crc,
                    indices=indices,
                )
        search_pos = offset + 1


def _is_emoji_base(ch: str) -> bool:
    """Return whether ``ch`` is a valid base character for an Emoji VS15/VS16 or ZWJ sequence."""
    cp = ord(ch)
    if (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2300 <= cp <= 0x23FF
        or 0x2B00 <= cp <= 0x2BFF
        or cp
        in {
            0x00A9,
            0x00AE,
            0x203C,
            0x2049,
            0x2122,
            0x2139,
            0x2194,
            0x2195,
            0x2196,
            0x2197,
            0x2198,
            0x2199,
            0x21A9,
            0x21AA,
            0x25AA,
            0x25AB,
            0x25B6,
            0x25C0,
            0x25FB,
            0x25FC,
            0x25FD,
            0x25FE,
            0x2934,
            0x2935,
            0x3030,
            0x303D,
            0x3297,
            0x3299,
        }
    ):
        return True
    return unicodedata.category(ch) == "So"


def _is_emoji_modifier(ch: str) -> bool:
    """Return whether ``ch`` is an Emoji presentation selector, keycap, or skin-tone modifier."""
    cp = ord(ch)
    return cp in (0xFE0E, 0xFE0F, 0x20E3) or 0x1F3FB <= cp <= 0x1F3FF


def _neighbor_skipping_modifiers(text: str, start: int, step: int) -> str:
    """Return the nearest neighbor character skipping Emoji modifiers in direction ``step``."""
    idx = start + step
    n = len(text)
    while 0 <= idx < n and _is_emoji_modifier(text[idx]):
        idx += step
    return text[idx] if 0 <= idx < n else ""


def _is_joining_script_char(ch: str) -> bool:
    """Return whether ``ch`` belongs to an orthographic script that requires ``U+200C``/``U+200D``."""
    cp = ord(ch)
    in_joining_block = (
        0x0600 <= cp <= 0x08FF
        or 0x0900 <= cp <= 0x0DFF
        or 0x0F00 <= cp <= 0x109F
        or 0x1780 <= cp <= 0x18AF
        or 0xFB50 <= cp <= 0xFDFF
        or 0xFE70 <= cp <= 0xFEFC
    )
    return in_joining_block and unicodedata.category(ch).startswith(("L", "M"))


def _is_pua_codepoint(cp: int) -> bool:
    """Return whether ``cp`` is in a Unicode Private Use Area (e.g. ``U+E200..U+E203`` markers)."""
    return 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD


def _is_cjk_ideograph(ch: str) -> bool:
    """Return whether ``ch`` is a CJK Unified or Compatibility Ideograph valid for a single IVS."""
    if not ch:
        return False
    cp = ord(ch)
    return 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x323AF


def _is_reserved_ignorable_or_noncharacter(cp: int) -> bool:
    """Return whether ``cp`` is an unassigned Default_Ignorable_Code_Point or Unicode noncharacter."""
    return (
        0xFFF0 <= cp <= 0xFFF8
        or 0xE0080 <= cp <= 0xE00FF
        or 0xE01F0 <= cp <= 0xE0FFF
        or 0xFDD0 <= cp <= 0xFDEF
        or ((cp & 0xFFFE) == 0xFFFE and (cp >> 16) <= 0x10)
    )


def _is_rtl_char(ch: str) -> bool:
    """Return whether ``ch`` has right-to-left Unicode bidirectional class."""
    return unicodedata.bidirectional(ch) in ("R", "AL", "AN")


_BIDI_CONTROL_CHARS = "\u202a\u202b\u202c\u2066\u2067\u2068\u2069\u200e\u200f\u061c"


def _valid_rtl_bidi_indices(text: str) -> frozenset[int]:
    """Return indices of balanced Bidi controls on lines containing RTL script characters.

    Always excludes ``U+202D`` (``LRO``) and ``U+202E`` (``RLO``) Trojan Source overrides.
    """
    if not any(ch in text for ch in _BIDI_CONTROL_CHARS):
        return frozenset()
    valid: set[int] = set()
    offset = 0
    for line in text.splitlines(keepends=True):
        if any(ch in line for ch in _BIDI_CONTROL_CHARS) and any(_is_rtl_char(ch) for ch in line):
            emb_stack: list[int] = []
            iso_stack: list[int] = []
            for i, ch in enumerate(line):
                cp = ord(ch)
                if cp in (0x202A, 0x202B):
                    emb_stack.append(offset + i)
                elif cp == 0x202C and emb_stack:
                    valid.add(emb_stack.pop())
                    valid.add(offset + i)
                elif cp in (0x2066, 0x2067, 0x2068):
                    iso_stack.append(offset + i)
                elif cp == 0x2069 and iso_stack:
                    valid.add(iso_stack.pop())
                    valid.add(offset + i)
                elif cp in (0x200E, 0x200F, 0x061C):
                    left = line[i - 1] if i > 0 else ""
                    right = line[i + 1] if i + 1 < len(line) else ""
                    if (left and _is_rtl_char(left)) or (right and _is_rtl_char(right)):
                        valid.add(offset + i)
        offset += len(line)
    return frozenset(valid)


def _subdivision_flag_tag_indices(text: str) -> frozenset[int]:
    """Return indices of Plane-14 Tag characters belonging to valid subdivision flag emojis.

    Requires ``U+1F3F4`` followed by 2..6 alphanumeric tag characters (``0-9a-z`` in
    ``U+E0030..U+E0039`` or ``U+E0061..U+E007A``) terminated by ``U+E007F``.
    """
    if "\U0001f3f4" not in text:
        return frozenset()
    preserved: set[int] = set()
    n = len(text)
    idx = 0
    while idx < n:
        if ord(text[idx]) == 0x1F3F4:
            cursor = idx + 1
            while cursor < n and (cursor - (idx + 1)) < 6:
                cp = ord(text[cursor])
                if 0xE0030 <= cp <= 0xE0039 or 0xE0061 <= cp <= 0xE007A:
                    cursor += 1
                else:
                    break
            tag_len = cursor - (idx + 1)
            if 2 <= tag_len <= 6 and cursor < n and ord(text[cursor]) == 0xE007F:
                preserved.update(range(idx + 1, cursor + 1))
                idx = cursor + 1
                continue
        idx += 1
    return frozenset(preserved)


def _is_invisible_or_stego_cp(cp: int) -> bool:
    """Return whether codepoint ``cp`` is in an invisible, variation-selector, tag, or PUA range."""
    return (
        cp in _ZERO_WIDTH_CODEPOINTS
        or 0xFE00 <= cp <= 0xFE0F
        or 0xE0000 <= cp <= 0xE007F
        or 0xE0100 <= cp <= 0xE01EF
        or _is_pua_codepoint(cp)
        or _is_reserved_ignorable_or_noncharacter(cp)
    )


def _decode_invisible_html_entities_in_prose(text: str) -> tuple[str, int]:
    """Decode HTML entities that resolve to zero-width, format, or variation-selector codepoints."""
    if "&" not in text:
        return text, 0
    decoded_count = 0

    def _replace_entity(match: re.Match[str]) -> str:
        nonlocal decoded_count
        dec_str = match.group("dec")
        hex_str = match.group("hex")
        name_str = match.group("name")
        if name_str is not None:
            mapped = _INVISIBLE_NAMED_ENTITIES.get(name_str.lower())
            if mapped is not None:
                decoded_count += 1
                return mapped
            return match.group(0)
        cp = int(dec_str, 10) if dec_str is not None else int(hex_str, 16)
        if 0 <= cp <= 0x10FFFF and _is_invisible_or_stego_cp(cp):
            decoded_count += 1
            return chr(cp)
        return match.group(0)

    return _HTML_ENTITY_RE.sub(_replace_entity, text), decoded_count


def _strip_assistant_citations_in_prose(text: str) -> tuple[str, int]:
    """Strip ChatGPT PUA citation blocks, Gemini span/cite markers, and Grok citation cards."""
    count = 0

    def _on_match(_m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return ""

    cleaned = _ASSISTANT_CITATION_RE.sub(_on_match, text)
    return cleaned, count


def _is_legitimate_typographic_nbsp(text: str, idx: int) -> bool:
    """Return whether ``text[idx]`` is a legitimate typographic non-breaking space."""
    cp = ord(text[idx])
    if cp not in (0x00A0, 0x202F, 0x2007, 0x2009):
        return False
    left = text[idx - 1] if idx > 0 else ""
    right = text[idx + 1] if idx + 1 < len(text) else ""
    if (left and left.isdigit()) or (right and right.isdigit()):
        return True
    return left in ("«", "\u2039") or right in ("»", "\u203a", ":", ";", "!", "?")


def _normalize_exotic_spaces_in_prose(
    text: str,
    *,
    force_innamark_clean: bool = False,
) -> tuple[str, int]:
    """Normalize exotic Unicode ``Zs`` space-substitution watermarks to ASCII ``' '``."""
    count = 0
    out: list[str] = []
    for idx, ch in enumerate(text):
        cp = ord(ch)
        if cp in _EXOTIC_SPACE_CPS:
            if (
                cp == 0x200A
                and idx > 0
                and idx + 1 < len(text)
                and variation_selector_to_byte(text[idx - 1]) is not None
                and variation_selector_to_byte(text[idx + 1]) is not None
            ):
                count += 1
                continue
            if not force_innamark_clean and _is_legitimate_typographic_nbsp(text, idx):
                out.append(ch)
            else:
                out.append(" ")
                count += 1
        else:
            out.append(ch)
    return "".join(out), count


def _is_transparent_in_word(ch: str) -> bool:
    """Return whether ``ch`` is an invisible/format codepoint that should not split a word token."""
    return _is_invisible_or_stego_cp(ord(ch))


def _is_native_latin_letter(ch: str) -> bool:
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return True
    if ch in _CONFUSABLE_TO_LATIN:
        return False
    return "LATIN" in unicodedata.name(ch, "")


def _is_native_cyrillic_letter(ch: str) -> bool:
    cp = ord(ch)
    if 0x0400 <= cp <= 0x052F:
        return True
    if ch in _CONFUSABLE_TO_CYRILLIC:
        return False
    return "CYRILLIC" in unicodedata.name(ch, "")


def _normalize_mixed_script_homoglyphs_in_prose(text: str) -> tuple[str, int]:
    """Normalize confusable homoglyphs inside predominantly Latin or Cyrillic words."""
    replaced_count = 0
    out = list(text)
    n = len(text)
    idx = 0
    span_counts: tuple[int, int] | None = None

    while idx < n:
        ch = text[idx]
        if ch.isalpha() or _is_transparent_in_word(ch):
            letter_indices: list[int] = []
            while idx < n and (text[idx].isalpha() or _is_transparent_in_word(text[idx])):
                if text[idx].isalpha():
                    letter_indices.append(idx)
                idx += 1
            if len(letter_indices) >= 2:
                latin_conf = [i for i in letter_indices if text[i] in _CONFUSABLE_TO_LATIN]
                cyrillic_conf = [i for i in letter_indices if text[i] in _CONFUSABLE_TO_CYRILLIC]
                if not latin_conf and not cyrillic_conf:
                    continue

                latin_native = [i for i in letter_indices if _is_native_latin_letter(text[i])] if latin_conf else []
                can_latin = bool(
                    latin_native and latin_conf and len(latin_native) + len(latin_conf) == len(letter_indices)
                )

                cyrillic_native = (
                    [i for i in letter_indices if _is_native_cyrillic_letter(text[i])] if cyrillic_conf else []
                )
                can_cyrillic = bool(
                    cyrillic_native
                    and cyrillic_conf
                    and len(cyrillic_native) + len(cyrillic_conf) == len(letter_indices)
                )

                if can_latin and can_cyrillic and len(latin_native) == len(cyrillic_native):
                    if span_counts is None:
                        span_counts = (
                            sum(1 for c in text if c.isalpha() and _is_native_latin_letter(c)),
                            sum(1 for c in text if c.isalpha() and _is_native_cyrillic_letter(c)),
                        )
                    prefer_latin = span_counts[0] >= span_counts[1]
                else:
                    prefer_latin = not can_cyrillic or len(latin_native) > len(cyrillic_native)

                if can_latin and prefer_latin:
                    for ci in latin_conf:
                        out[ci] = _CONFUSABLE_TO_LATIN[text[ci]]
                        replaced_count += 1
                elif can_cyrillic:
                    for ci in cyrillic_conf:
                        out[ci] = _CONFUSABLE_TO_CYRILLIC[text[ci]]
                        replaced_count += 1
        else:
            idx += 1

    return "".join(out), replaced_count


def _extract_and_strip_stego_codepoints(
    text: str,
    *,
    vsrmark_indices: frozenset[int] = frozenset(),
) -> tuple[str, str | None, int, int, int]:
    """Strip Unicode Tags, VSRMark selectors, PUA markers, and steganographic zero-width/VS codepoints.

    Preserves valid Emoji presentation selectors (``U+FE0E``/``U+FE0F``), single CJK
    Ideographic Variation Sequences (``U+FE00..U+FE0D`` / ``U+E0100..U+E01EF``), valid
    Mongolian/Khmer/Hangul script formatters, balanced RTL Bidi controls, valid regional
    subdivision flag tag sequences (``U+1F3F4``..``U+E007F``), and valid Emoji or
    joining-script ``U+200C``/``U+200D`` sequences.
    """
    flag_tag_indices = _subdivision_flag_tag_indices(text)
    valid_bidi_indices = _valid_rtl_bidi_indices(text)
    tag_chars: list[str] = []
    tag_codepoint_count = 0
    anomalous_zw_count = 0
    removed_count = 0
    out: list[str] = []
    n = len(text)

    for idx, ch in enumerate(text):
        cp = ord(ch)

        if idx in vsrmark_indices:
            removed_count += 1
            continue

        if 0xE0000 <= cp <= 0xE007F:
            if idx in flag_tag_indices:
                out.append(ch)
                continue
            tag_codepoint_count += 1
            removed_count += 1
            ascii_cp = cp - 0xE0000
            if 0x20 <= ascii_cp <= 0x7E:
                tag_chars.append(chr(ascii_cp))
            continue

        if _is_pua_codepoint(cp) or _is_reserved_ignorable_or_noncharacter(cp):
            anomalous_zw_count += 1
            removed_count += 1
            continue

        if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
            prev_ch = text[idx - 1] if idx > 0 else ""
            next_ch = text[idx + 1] if idx + 1 < n else ""
            next_is_vs = variation_selector_to_byte(next_ch) is not None
            valid_base = (
                bool(prev_ch)
                and not next_is_vs
                and (_is_emoji_base(prev_ch) if cp in (0xFE0E, 0xFE0F) else _is_cjk_ideograph(prev_ch))
            )
            if valid_base:
                out.append(ch)
            else:
                anomalous_zw_count += 1
                removed_count += 1
            continue

        if cp in _ZERO_WIDTH_CODEPOINTS:
            if cp in (0x202A, 0x202B, 0x202C, 0x2066, 0x2067, 0x2068, 0x2069, 0x200E, 0x200F, 0x061C):
                if idx in valid_bidi_indices:
                    out.append(ch)
                    continue
            elif 0x180B <= cp <= 0x180F:
                prev_cp = ord(text[idx - 1]) if idx > 0 else 0
                next_cp = ord(text[idx + 1]) if idx + 1 < n else 0
                if 0x1800 <= prev_cp <= 0x18AF and not (0x180B <= next_cp <= 0x180F):
                    out.append(ch)
                    continue
            elif cp in (0x17B4, 0x17B5):
                prev_cp = ord(text[idx - 1]) if idx > 0 else 0
                if 0x1780 <= prev_cp <= 0x17FF:
                    out.append(ch)
                    continue
            elif cp in (0x115F, 0x1160):
                prev_cp = ord(text[idx - 1]) if idx > 0 else 0
                if 0x1100 <= prev_cp <= 0x11FF or 0xA960 <= prev_cp <= 0xA97C or 0xD7B0 <= prev_cp <= 0xD7C6:
                    out.append(ch)
                    continue
            elif cp == 0x200D:
                left = _neighbor_skipping_modifiers(text, idx, -1)
                right = _neighbor_skipping_modifiers(text, idx, 1)
                if (
                    left
                    and right
                    and (
                        (_is_emoji_base(left) and _is_emoji_base(right))
                        or (_is_joining_script_char(left) and _is_joining_script_char(right))
                    )
                ):
                    out.append(ch)
                    continue
            elif cp == 0x200C:
                left = text[idx - 1] if idx > 0 else ""
                right = text[idx + 1] if idx + 1 < n else ""
                if (
                    left
                    and right
                    and _is_joining_script_char(left)
                    and _is_joining_script_char(right)
                    and ord(left) not in _NON_JOINING_ARABIC_CPS
                ):
                    out.append(ch)
                    continue
            anomalous_zw_count += 1
            removed_count += 1
            continue

        out.append(ch)

    tags_payload = "".join(tag_chars) if tag_chars else None
    return "".join(out), tags_payload, removed_count, tag_codepoint_count, anomalous_zw_count


# ── Public provenance identification and deterministic metadata stripping ──


def _platform_from_c2pa_info(c2pa_info: dict[str, Any]) -> str | None:
    generator = c2pa_info.get("claim_generator")
    if isinstance(generator, str):
        resolved = claim_generator_platform(generator) or c2pa_vendor_platform(generator)
        if resolved is not None:
            return resolved
    for key in ("ai_tool", "issuer", "claim_generator", "soft_binding"):
        val = c2pa_info.get(key)
        if isinstance(val, str) and val.strip():
            return c2pa_vendor_platform(val) or val.strip()
    return None


def _preprocess_unlocked_prose_segments(
    text: str,
    *,
    force_innamark_clean: bool = False,
) -> tuple[str, int, int, int, int]:
    """Decode invisible entities, strip citations, and normalize homoglyphs/spaces outside code spans."""
    if not text:
        return text, 0, 0, 0, 0

    code_locked_kinds = {
        "fenced_code",
        "inline_code",
        "math_block",
        "math_inline",
        "html_block",
        "html_tag",
        "url",
        "path",
    }
    spans = split_protected_text_spans(text)
    out_parts: list[str] = []
    entity_count = 0
    citation_count = 0
    homoglyph_count = 0
    space_count = 0

    for span in spans:
        if span.kind in code_locked_kinds:
            out_parts.append(span.text)
            continue
        seg, e_cnt = _decode_invisible_html_entities_in_prose(span.text)
        seg, c_cnt = _strip_assistant_citations_in_prose(seg)
        seg, h_cnt = _normalize_mixed_script_homoglyphs_in_prose(seg)
        seg, s_cnt = _normalize_exotic_spaces_in_prose(seg, force_innamark_clean=force_innamark_clean)
        entity_count += e_cnt
        citation_count += c_cnt
        homoglyph_count += h_cnt
        space_count += s_cnt
        out_parts.append(seg)

    return "".join(out_parts), entity_count, citation_count, homoglyph_count, space_count


def _inspect_and_strip_text(
    text: str,
    source: Path | None = None,
) -> tuple[TextProvenanceReport, str]:
    """Run all deterministic text provenance, steganography, and homoglyph inspection/stripping passes."""
    signals: list[str] = []
    markers: dict[str, str] = {}
    c2pa_info: dict[str, Any] = {}
    caveats: list[str] = []

    after_c2patxt, c2patxt_match = _strip_c2patxt_suffix(text)
    if c2patxt_match is not None:
        signals.append("c2pa_text_manifest_wrapper")
        markers["c2pa_text_wrapper"] = (
            f"C2PATXT v{C2PATXT_VERSION} ({len(c2patxt_match.manifest_bytes)} bytes JUMBF"
            + (f", {c2patxt_match.padding_bytes} bytes padding" if c2patxt_match.padding_bytes else "")
            + ")"
        )
        c2pa_info.update(c2patxt_match.c2pa_info)
        if not c2patxt_match.had_bom_prefix:
            caveats.append("C2PATXT wrapper lacked the leading U+FEFF sentinel required by C2PA Appendix A.8.")

    after_armor, armor_match = _strip_c2pa_ascii_armor(after_c2patxt)
    if armor_match is not None:
        signals.append("c2pa_ascii_armor")
        if armor_match.manifest_bytes is not None:
            markers["c2pa_ascii_armor"] = f"Inline JUMBF ({len(armor_match.manifest_bytes)} bytes)"
        elif armor_match.manifest_uri is not None:
            markers["c2pa_ascii_armor"] = f"External manifest URI ({armor_match.manifest_uri})"
        else:
            markers["c2pa_ascii_armor"] = f"{armor_match.count} block(s)"
        for k, v in armor_match.c2pa_info.items():
            c2pa_info.setdefault(k, v)

    after_html, html_match = _strip_html_c2pa(after_armor)
    if html_match is not None:
        signals.append("c2pa_html_head")
        if html_match.manifest_bytes is not None:
            markers["c2pa_html_manifest"] = f"Inline HTML C2PA JUMBF ({len(html_match.manifest_bytes)} bytes)"
        elif html_match.manifest_uri is not None:
            markers["c2pa_html_manifest"] = f"External HTML C2PA link ({html_match.manifest_uri})"
        else:
            markers["c2pa_html_manifest"] = f"{html_match.element_count} HTML C2PA element(s)"
        for k, v in html_match.c2pa_info.items():
            c2pa_info.setdefault(k, v)
        if html_match.element_count > 1:
            caveats.append("C2PA Appendix A.7 violation: HTML <head> contains multiple C2PA manifest elements.")
        if html_match.outside_head:
            caveats.append("C2PA Appendix A.7 warning: C2PA HTML element found outside <head>.")

    vsrmark_match = _detect_vsrmark(after_html)
    vsrmark_payload: str | None = None
    vsrmark_indices: frozenset[int] = frozenset()
    if vsrmark_match is not None:
        vsrmark_payload = vsrmark_match.payload
        vsrmark_indices = vsrmark_match.indices
        signals.append("vsrmark")
        markers["vsrmark"] = (
            f"{VSRMARK_ALGORITHM} ({len(vsrmark_match.raw_payload)} bytes, crc32=0x{vsrmark_match.crc32:08x})"
        )

    innamark_payload = _detect_innamark_payload(after_html)
    if innamark_payload is not None:
        signals.append("innamark_whitespace")
        markers["innamark_payload"] = innamark_payload

    stegcloak_info = _detect_stegcloak_stream(after_html)
    if stegcloak_info is not None:
        signals.append("stegcloak")
        markers["stegcloak"] = stegcloak_info

    zw_payloads = _detect_zero_width_binary_payload(after_html)
    if "groundtruth_mcp" in zw_payloads:
        signals.append("groundtruth_mcp")
        markers["groundtruth_mcp_watermark"] = zw_payloads["groundtruth_mcp"]
    if "zero_width_binary" in zw_payloads:
        markers["zero_width_payload"] = zw_payloads["zero_width_binary"]

    vs_payloads = _detect_variation_selector_stego_payload(after_html, excluded_indices=vsrmark_indices)
    for vs_key, vs_val in vs_payloads.items():
        markers[f"{vs_key}_payload"] = vs_val

    preprocessed, entity_cnt, citation_cnt, homoglyph_cnt, space_cnt = _preprocess_unlocked_prose_segments(
        after_html,
        force_innamark_clean=innamark_payload is not None,
    )
    if entity_cnt > 0:
        markers["invisible_html_entities"] = f"{entity_cnt} invisible HTML entity(ies)"
    if citation_cnt > 0:
        signals.append("assistant_citation_artifact")
        markers["assistant_citation_artifact"] = f"{citation_cnt} assistant citation artifact(s)"
    if homoglyph_cnt > 0:
        signals.append("homoglyph_steganography")
        markers["homoglyph_steganography"] = f"{homoglyph_cnt} mixed-script confusable homoglyph(s)"
    if space_cnt > 0 and "innamark_whitespace" not in signals:
        signals.append("whitespace_steganography")
        markers["whitespace_steganography"] = f"{space_cnt} exotic space codepoint(s)"

    pre_vsrmark = (
        vsrmark_match
        if preprocessed == after_html
        else (_detect_vsrmark(preprocessed) if vsrmark_match is not None else None)
    )
    pre_vsrmark_indices: frozenset[int] = pre_vsrmark.indices if pre_vsrmark is not None else frozenset()
    cleaned_text, tags_payload, stego_removed, tag_cp_count, anomalous_zw_count = _extract_and_strip_stego_codepoints(
        preprocessed,
        vsrmark_indices=pre_vsrmark_indices,
    )
    if tag_cp_count > 0:
        signals.append("unicode_tags")
        markers["unicode_tags"] = f"Unicode Tags hidden ASCII ({len(tags_payload or '')} chars)"
    if anomalous_zw_count > 0:
        signals.append("zero_width_steganography")
        markers["zero_width_steganography"] = f"{anomalous_zw_count} steganographic codepoint(s)"

    total_stego_cp = (
        (c2patxt_match.removed_codepoints if c2patxt_match is not None else 0)
        + stego_removed
        + citation_cnt
        + homoglyph_cnt
        + space_cnt
    )

    for key in ("issuer", "ai_tool", "source_type", "claim_generator", "soft_binding", "synthid_watermark"):
        if key in c2pa_info:
            markers[f"c2pa_{key}"] = str(c2pa_info[key])

    has_c2pa_or_vsrmark = any(
        sig in signals
        for sig in (
            "c2pa_text_manifest_wrapper",
            "c2pa_ascii_armor",
            "c2pa_html_head",
            "vsrmark",
            "innamark_whitespace",
        )
    )
    has_any_signal = bool(signals)
    platform = _platform_from_c2pa_info(c2pa_info)
    if platform is None and vsrmark_match is not None:
        platform = "VSRMark (com.cognitive-proof.vsrmark.1)"
    elif platform is None and innamark_payload is not None:
        platform = "Innamark (arXiv:2502.12710)"

    if has_c2pa_or_vsrmark:
        is_ai: Literal[True] | None = True
        confidence: Literal["high", "medium", "unknown"] = "high"
    elif has_any_signal:
        is_ai = True
        confidence = "medium"
    else:
        is_ai = None
        confidence = "unknown"

    caveats.append(_STATISTICAL_SAMPLING_CAVEAT)

    report = TextProvenanceReport(
        source=source,
        is_ai_generated=is_ai,
        confidence=confidence,
        platform=platform,
        has_ai_metadata=has_any_signal,
        signals=tuple(signals),
        metadata_markers=markers,
        c2pa_info=c2pa_info,
        vsrmark_payload=vsrmark_payload,
        unicode_tags_payload=tags_payload,
        stego_codepoint_count=total_stego_cp,
        caveats=tuple(caveats),
    )
    return report, cleaned_text


def identify_text(text_or_path: str | Path) -> TextProvenanceReport:
    """Inspect a text string or text file for C2PA wrappers and character steganography."""
    source, text = _read_text_input(text_or_path)
    report, _ = _inspect_and_strip_text(text, source=source)
    return report


def remove_text_metadata(
    text_or_path: str | Path,
    output_path: str | Path | None = None,
) -> TextMetadataResult:
    """Strip C2PA text/HTML wrappers, VSRMark soft bindings, Unicode Tags, and zero-width steganography."""
    source, resolved_output, text = _prepare_text_io(text_or_path, output_path)

    before_report, cleaned_text = _inspect_and_strip_text(text, source=source)
    if cleaned_text == text and not before_report.has_ai_metadata:
        after_report = before_report
    else:
        after_report, _ = _inspect_and_strip_text(cleaned_text)
    _write_output_text(resolved_output, cleaned_text)

    return TextMetadataResult(
        source=source,
        output=resolved_output,
        cleaned_text=cleaned_text,
        removed_signals=before_report.signals,
        detected_markers=before_report.metadata_markers,
        remaining_markers=after_report.metadata_markers,
        removed_codepoints=before_report.stego_codepoint_count,
    )


# ── Span-protected segmentation and local prose regeneration ──

_BLOCK_SPAN_RE = re.compile(
    r"(?P<fenced_code>^[ \t]*(?:```|~~~)[^\n]*\n[\s\S]*?\n[ \t]*(?:```|~~~)[ \t]*(?=\n|$))"
    r"|(?P<math_block>\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])"
    r"|(?P<html_block><(?:script|style|pre|code|head|svg|math)\b[^>]*>[\s\S]*?"
    r"</(?:script|style|pre|code|head|svg|math)\s*>)"
    r"|(?P<markdown_structure>^(?:[ \t]*#{1,6}[ \t]+[^\n]+|[ \t]*\|[^\n]+\|[ \t]*(?=\n|$)"
    r"|[ \t]*\[[^\]]+\]:[ \t]+\S+[^\n]*))",
    re.IGNORECASE | re.MULTILINE,
)

_INLINE_SPAN_RE = re.compile(
    r"(?P<inline_code>`[^`\n]+`)"
    r"|(?P<math_inline>(?<!\$)\$(?!\$)[^\$\n]+?\$(?!\$)|\\\([\s\S]+?\\\))"
    r"|(?P<url>https?://[^\s<>)\]]+)"
    r"|(?P<html_tag></?[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*?)?>)"
    r"|(?P<path>(?<![A-Za-z0-9_])(?:\.{0,2}/)[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8}\b)",
)


def _make_prose_or_ws_span(piece: str) -> TextSpan:
    is_ws = piece.isspace()
    return TextSpan(kind="whitespace" if is_ws else "prose", text=piece, locked=is_ws)


def split_protected_text_spans(text: str) -> tuple[TextSpan, ...]:
    """Segment ``text`` into locked technical/structural spans and unlocked prose spans."""
    if not text:
        return ()

    spans: list[TextSpan] = []

    def _emit_prose_and_inline(segment: str) -> None:
        if not segment:
            return
        for part in re.split(r"(\n\s*\n)", segment):
            if not part:
                continue
            if part.isspace():
                spans.append(TextSpan(kind="whitespace", text=part, locked=True))
                continue
            cursor = 0
            for match in _INLINE_SPAN_RE.finditer(part):
                if match.start() > cursor:
                    spans.append(_make_prose_or_ws_span(part[cursor : match.start()]))
                group_name = match.lastgroup or "inline_code"
                spans.append(
                    TextSpan(
                        kind=group_name,  # type: ignore[arg-type]
                        text=match.group(0),
                        locked=True,
                    )
                )
                cursor = match.end()
            if cursor < len(part):
                spans.append(_make_prose_or_ws_span(part[cursor:]))

    cursor = 0
    for match in _BLOCK_SPAN_RE.finditer(text):
        start, end = match.span()
        if start > cursor:
            _emit_prose_and_inline(text[cursor:start])
        group_name = match.lastgroup or "fenced_code"
        spans.append(TextSpan(kind=group_name, text=match.group(0), locked=True))  # type: ignore[arg-type]
        cursor = end

    if cursor < len(text):
        _emit_prose_and_inline(text[cursor:])

    return tuple(spans)


def _select_local_lm_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=2)
def _load_local_rewriter_cached(model_id: str, resolved_device: str) -> tuple[Any, Any, str]:
    """Load and cache an unwatermarked local causal LM keyed on resolved device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if resolved_device in ("cuda", "mps") else torch.float32
    log.info("Loading local text rewriter model %s on %s", model_id, resolved_device)
    tokenizer: Any = cast("Any", AutoTokenizer).from_pretrained(model_id)
    model: Any = cast("Any", AutoModelForCausalLM).from_pretrained(model_id, torch_dtype=dtype)
    model.to(resolved_device)
    model.train(False)
    return tokenizer, model, resolved_device


def _load_local_rewriter_runtime(model_id: str, device: str) -> tuple[Any, Any, str]:
    _require_text_invisible_runtime()
    resolved_device = _select_local_lm_device(device)
    return _load_local_rewriter_cached(model_id, resolved_device)


_PLACEHOLDER_TOKEN_RE = re.compile(r"\[\[RAIW_SPAN_\d+\]\]")
_NUMERAL_TOKEN_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_WORD_NUMBERS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}
_PSEUDO_NEGATION_RE = re.compile(
    r"\b(?:not\s+only|not\s+just|not\s+merely|not\s+simply|not\s+solely|not\s+necessarily|without\s+difficulty)\b",
    re.IGNORECASE,
)
_NEGATION_TOKEN_RE = re.compile(
    r"\b(?:not|no|never|none|neither|nor|without|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't)\b",
    re.IGNORECASE,
)
_ANTONYM_POLES: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset(
            {"increase", "increases", "increased", "increasing", "rise", "rises", "rose", "grow", "grows", "grew"}
        ),
        frozenset(
            {
                "decrease",
                "decreases",
                "decreased",
                "decreasing",
                "fall",
                "falls",
                "fell",
                "shrink",
                "shrinks",
                "shrank",
                "drop",
                "drops",
                "dropped",
            }
        ),
    ),
    (
        frozenset({"succeed", "succeeds", "succeeded", "pass", "passes", "passed"}),
        frozenset({"fail", "fails", "failed", "failing"}),
    ),
    (frozenset({"always"}), frozenset({"never"})),
    (
        frozenset({"before", "prior", "precede", "precedes", "preceded"}),
        frozenset({"after", "subsequent", "follow", "follows", "followed"}),
    ),
    (
        frozenset({"above", "over", "exceed", "exceeds", "exceeded"}),
        frozenset({"below", "under"}),
    ),
    (
        frozenset({"true", "correct", "accurate", "valid"}),
        frozenset({"false", "incorrect", "inaccurate", "invalid"}),
    ),
    (
        frozenset({"allowed", "permitted", "legal", "authorized"}),
        frozenset({"forbidden", "prohibited", "illegal", "banned", "unauthorized"}),
    ),
    (frozenset({"open", "opened"}), frozenset({"closed", "shut"})),
)
_CLAIM_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
    }
)


def _normalize_numeral_token(token: str) -> str:
    cleaned = token.replace(",", "")
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def _extract_normalized_numerals(text: str) -> set[str]:
    without_placeholders = _PLACEHOLDER_TOKEN_RE.sub(" ", text)
    nums = {_normalize_numeral_token(m.group(0)) for m in _NUMERAL_TOKEN_RE.finditer(without_placeholders)}
    for word in re.findall(r"[^\W\d_]+", without_placeholders.lower()):
        if word in _WORD_NUMBERS:
            nums.add(_WORD_NUMBERS[word])
    return nums


def _extract_mid_sentence_proper_nouns(text: str) -> set[str]:
    """Extract capitalized mid-sentence proper nouns (skipping sentence-initial tokens)."""
    without_placeholders = _PLACEHOLDER_TOKEN_RE.sub(" ", text)
    entities: set[str] = set()
    for sentence in re.split(r"[.!?]+", without_placeholders):
        words = re.findall(r"[\"'(\[]?[^\W\d_][\w-]*", sentence.strip())
        for idx, raw_word in enumerate(words):
            if idx == 0 or raw_word[0] in ('"', "'", "(", "["):
                continue
            if raw_word[0].isupper() and any(ch.islower() for ch in raw_word[1:]):
                entities.add(raw_word.lower())
    return entities


def _has_active_negation(text: str) -> bool:
    scrubbed = _PSEUDO_NEGATION_RE.sub(" ", _PLACEHOLDER_TOKEN_RE.sub(" ", text))
    return _NEGATION_TOKEN_RE.search(scrubbed) is not None


def _sentence_content_words(sentence: str) -> set[str]:
    without_placeholders = _PLACEHOLDER_TOKEN_RE.sub(" ", sentence)
    return {
        w for w in re.findall(r"[^\W_]+", without_placeholders.lower()) if len(w) >= 2 and w not in _CLAIM_STOPWORDS
    }


def _has_antonym_pole_inversion(src_words: set[str], cand_words: set[str]) -> bool:
    for pole_a, pole_b in _ANTONYM_POLES:
        src_a = bool(src_words & pole_a)
        src_b = bool(src_words & pole_b)
        cand_a = bool(cand_words & pole_a)
        cand_b = bool(cand_words & pole_b)
        if (src_a and not src_b and cand_b and not cand_a) or (src_b and not src_a and cand_a and not cand_b):
            return True
    return False


def verify_text_rewrite_facts(source_text: str, candidate_text: str) -> bool:
    """Verify that ``candidate_text`` preserves numerals, proper nouns, NegEx polarity, and antonym poles."""
    source_nums = _extract_normalized_numerals(source_text)
    if source_nums:
        candidate_nums = _extract_normalized_numerals(candidate_text)
        if not source_nums.issubset(candidate_nums):
            return False

    source_entities = _extract_mid_sentence_proper_nouns(source_text)
    if source_entities:
        candidate_words = {
            w.lower() for w in re.findall(r"[^\W\d_][\w-]*", _PLACEHOLDER_TOKEN_RE.sub(" ", candidate_text))
        }
        if not source_entities.issubset(candidate_words):
            return False

    source_sentences = [s.strip() for s in re.split(r"[.!?]+", source_text) if s.strip()]
    candidate_sentences = [s.strip() for s in re.split(r"[.!?]+", candidate_text) if s.strip()]
    if len(source_sentences) == 1 and len(candidate_sentences) == 1:
        src_s, cand_s = source_sentences[0], candidate_sentences[0]
        if _has_active_negation(src_s) != _has_active_negation(cand_s):
            return False
        return not _has_antonym_pole_inversion(_sentence_content_words(src_s), _sentence_content_words(cand_s))

    cand_word_sets = [_sentence_content_words(c) for c in candidate_sentences]
    used_cand: set[int] = set()
    for src_s in source_sentences:
        src_words = _sentence_content_words(src_s)
        if not src_words:
            continue
        best_idx = -1
        best_sim = 0.0
        for idx, cand_words in enumerate(cand_word_sets):
            if idx in used_cand or not cand_words:
                continue
            sim = len(src_words & cand_words) / math.sqrt(len(src_words) * len(cand_words))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= 0.45:
            used_cand.add(best_idx)
            matched_cand = candidate_sentences[best_idx]
            if _has_active_negation(src_s) != _has_active_negation(matched_cand):
                return False
            if _has_antonym_pole_inversion(src_words, cand_word_sets[best_idx]):
                return False
    return True


def _build_local_lm_rewriter(
    model_id: str = DEFAULT_TEXT_REWRITER_MODEL,
    *,
    device: str = "auto",
    seed: int = 0,
) -> Callable[[str], str]:
    """Create a prose-rewriting callable backed by an unwatermarked local causal LM."""
    tokenizer, model, resolved_device = _load_local_rewriter_runtime(model_id, device)
    import torch

    system_prompt = (
        "Rewrite the user's passage in fresh, natural wording while preserving every fact, "
        "number, and meaning. Keep any placeholder tokens matching [[RAIW_SPAN_N]] exactly "
        "unchanged in their appropriate positions. Output only the rewritten passage."
    )

    def _rewrite(passage: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system_prompt}\n\nPassage:\n{passage}\n\nRewritten passage:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(resolved_device)
        input_len = int(inputs["input_ids"].shape[1])
        max_new = max(64, min(1024, input_len * 2))
        best_decoded = ""
        for attempt in range(3):
            cast("Any", torch).manual_seed(seed + attempt)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][input_len:]
            decoded = str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()
            if not decoded:
                continue
            if not best_decoded:
                best_decoded = decoded
            if verify_text_rewrite_facts(passage, decoded):
                return decoded
        return best_decoded if best_decoded else passage

    return _rewrite


def _rewrite_preserving_ws(text: str, rewriter: Callable[[str], str]) -> str:
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    core = text.strip()
    return lead + rewriter(core).strip() + trail if core else text


def _rewrite_paragraph_cluster(
    cluster: Sequence[TextSpan],
    rewriter: Callable[[str], str],
) -> tuple[str, int, int]:
    """Rewrite a contiguous paragraph cluster of prose and inline locked spans.

    Inline locked spans are substituted with ``[[RAIW_SPAN_k]]`` placeholders during
    paragraph-level rewriting so sentence grammar stays coherent. If the rewriter
    drops or duplicates any placeholder, falls back to rewriting each prose span
    individually around the locked inline spans so locked spans are 100% preserved.
    """
    prose_indices = [i for i, span in enumerate(cluster) if not span.locked]
    if not prose_indices:
        return "".join(span.text for span in cluster), 0, 0

    placeholders: list[tuple[str, str]] = []
    templated_parts: list[str] = []
    rewritten_chars = 0
    for span in cluster:
        if span.locked:
            token = f"[[RAIW_SPAN_{len(placeholders)}]]"
            placeholders.append((token, span.text))
            templated_parts.append(token)
        else:
            templated_parts.append(span.text)
            rewritten_chars += len(span.text)

    templated = "".join(templated_parts)
    leading_ws = templated[: len(templated) - len(templated.lstrip())]
    trailing_ws = templated[len(templated.rstrip()) :]
    rewritten_core = rewriter(templated.strip()).strip()

    if all(rewritten_core.count(token) == 1 for token, _ in placeholders):
        restored = rewritten_core
        for token, original_text in placeholders:
            restored = restored.replace(token, original_text)
        return leading_ws + restored + trailing_ws, len(prose_indices), rewritten_chars

    out_parts = [span.text if span.locked else _rewrite_preserving_ws(span.text, rewriter) for span in cluster]
    return "".join(out_parts), len(prose_indices), rewritten_chars


def remove_text_invisible(
    text_or_path: str | Path,
    output_path: str | Path | None = None,
    *,
    rewriter: Callable[[str], str] | None = None,
    rewriter_model: str = DEFAULT_TEXT_REWRITER_MODEL,
    device: str = "auto",
    seed: int = 0,
) -> TextInvisibleResult:
    """Remove statistical token watermarks by regenerating unlocked prose spans.

    Low-entropy technical and structural spans (fenced code blocks, inline code,
    LaTeX math, URLs, file paths, HTML blocks/tags, Markdown headings) are locked
    and verified to remain byte-for-byte identical in the output.
    """
    source, resolved_output, text = _prepare_text_io(text_or_path, output_path)

    spans = split_protected_text_spans(text)
    has_prose = any(not span.locked for span in spans)

    active_rewriter: Callable[[str], str] | None = rewriter
    model_label = "custom_rewriter" if rewriter is not None else rewriter_model
    if has_prose and active_rewriter is None:
        active_rewriter = _build_local_lm_rewriter(rewriter_model, device=device, seed=seed)

    out_chunks: list[str] = []
    rewritten_spans = 0
    locked_spans = 0
    locked_characters = 0
    rewritten_characters = 0

    block_kinds = {"fenced_code", "math_block", "html_block", "markdown_structure", "whitespace"}
    current_cluster: list[TextSpan] = []

    def _flush_cluster() -> None:
        nonlocal rewritten_spans, locked_spans, locked_characters, rewritten_characters
        if not current_cluster:
            return
        for sp in current_cluster:
            if sp.locked:
                locked_spans += 1
                locked_characters += len(sp.text)
        if active_rewriter is not None and any(not sp.locked for sp in current_cluster):
            rebuilt, r_spans, r_chars = _rewrite_paragraph_cluster(current_cluster, active_rewriter)
            out_chunks.append(rebuilt)
            rewritten_spans += r_spans
            rewritten_characters += r_chars
        else:
            out_chunks.append("".join(sp.text for sp in current_cluster))
        current_cluster.clear()

    for span in spans:
        if span.kind in block_kinds:
            _flush_cluster()
            out_chunks.append(span.text)
            if span.kind != "whitespace":
                locked_spans += 1
                locked_characters += len(span.text)
        else:
            current_cluster.append(span)
    _flush_cluster()

    cleaned_text = "".join(out_chunks)

    search_cursor = 0
    for span in spans:
        if span.locked and span.kind != "whitespace":
            found_at = cleaned_text.find(span.text, search_cursor)
            if found_at < 0:
                raise RuntimeError(f"Locked {span.kind} span failed byte-identical verification: {span.text!r}")
            search_cursor = found_at + len(span.text)

    _write_output_text(resolved_output, cleaned_text)

    return TextInvisibleResult(
        source=source,
        output=resolved_output,
        cleaned_text=cleaned_text,
        rewritten_spans=rewritten_spans,
        locked_spans=locked_spans,
        locked_characters=locked_characters,
        rewritten_characters=rewritten_characters,
        model_id=model_label,
    )


def remove_text_all(
    text_or_path: str | Path,
    output_path: str | Path | None = None,
    *,
    include_invisible: bool = False,
    rewriter: Callable[[str], str] | None = None,
    rewriter_model: str = DEFAULT_TEXT_REWRITER_MODEL,
    device: str = "auto",
    seed: int = 0,
) -> TextAllResult:
    """Strip deterministic text provenance and optionally regenerate prose spans."""
    source, resolved_output, raw_text = _prepare_text_io(text_or_path, output_path)
    meta_result = remove_text_metadata(raw_text, output_path=None)

    rewritten_spans = 0
    locked_spans = 0
    model_id: str | None = None
    final_text = meta_result.cleaned_text
    remaining_markers = meta_result.remaining_markers

    if include_invisible:
        inv_result = remove_text_invisible(
            meta_result.cleaned_text,
            output_path=None,
            rewriter=rewriter,
            rewriter_model=rewriter_model,
            device=device,
            seed=seed,
        )
        final_text = inv_result.cleaned_text
        rewritten_spans = inv_result.rewritten_spans
        locked_spans = inv_result.locked_spans
        model_id = inv_result.model_id
        remaining_markers = identify_text(final_text).metadata_markers

    _write_output_text(resolved_output, final_text)

    return TextAllResult(
        source=source,
        output=resolved_output,
        cleaned_text=final_text,
        removed_signals=meta_result.removed_signals,
        detected_markers=meta_result.detected_markers,
        remaining_markers=remaining_markers,
        removed_codepoints=meta_result.removed_codepoints,
        rewritten_spans=rewritten_spans,
        locked_spans=locked_spans,
        model_id=model_id,
    )


# ── Keyed local statistical watermark scorers (SynthID-Text, Gumbel/textGrain, KGW, Unigram) ──


def _normal_one_sided_p_value(z_score: float) -> float:
    """Return the one-sided standard normal p-value ``P(Z >= z_score)``."""
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _make_watermark_score(
    scheme: Literal["synthid", "gumbel", "textgrain", "textseal2", "kgw", "unigram"],
    z_score: float,
    num_tokens: int,
    threshold_z: float,
) -> TextWatermarkScore:
    if num_tokens == 0:
        return TextWatermarkScore(scheme, 0.0, 0.5, False, 0, threshold_z)
    p_val = _normal_one_sided_p_value(z_score)
    return TextWatermarkScore(
        scheme=scheme,
        z_score=z_score,
        p_value=p_val,
        detected=z_score >= threshold_z,
        num_tokens=num_tokens,
        threshold_z=threshold_z,
    )


def _prf_bit_or_uniform(context: Sequence[int], token_id: int, key: int) -> int:
    """Deterministic 64-bit keyed pseudorandom digest for ``(context, token_id, key)``."""
    items = (*context, token_id, key)
    packed = b"".join(int(item).to_bytes(8, "little", signed=True) for item in items)
    return int.from_bytes(hashlib.blake2s(packed, digest_size=8).digest(), "little")


def _prf_uniform_01(context: Sequence[int], token_id: int, key: int) -> float:
    """Return a uniform float in ``(0, 1)`` from the 64-bit keyed digest."""
    raw = _prf_bit_or_uniform(context, token_id, key)
    return min(max((raw + 0.5) / float(1 << 64), 1e-15), 1.0 - 1e-15)


def _iter_unique_ngrams(
    token_ids: Sequence[int],
    window_len: int,
) -> list[tuple[tuple[int, ...], int]]:
    """Return deduplicated ``(context, token_id)`` pairs of length ``window_len``."""
    seen: set[tuple[int, ...]] = set()
    pairs: list[tuple[tuple[int, ...], int]] = []
    for idx in range(window_len - 1, len(token_ids)):
        ngram = tuple(int(t) for t in token_ids[idx - window_len + 1 : idx + 1])
        if ngram not in seen:
            seen.add(ngram)
            pairs.append((ngram[:-1], ngram[-1]))
    return pairs


def _exp_race_score(
    scheme: Literal["gumbel", "textgrain", "textseal2"],
    exp_sum: float,
    scored_tokens: int,
    threshold_z: float,
) -> TextWatermarkScore:
    if scored_tokens == 0:
        return _make_watermark_score(scheme, 0.0, 0, threshold_z)
    z_score = (exp_sum - scored_tokens) / math.sqrt(scored_tokens)
    return _make_watermark_score(scheme, z_score, scored_tokens, threshold_z)


def score_synthid_text(
    token_ids: Sequence[int],
    keys: Sequence[int],
    *,
    ngram_len: int = 5,
    threshold_z: float = DEFAULT_ALPHA_001_Z,
) -> TextWatermarkScore:
    """Compute the keyed ``SynthID-Text`` / tournament-sampling z-score over ``token_ids``.

    Deduplicates repeated ``ngram_len`` windows (matching the repeated-context masking
    used by Google ``SynthID-Text`` and Anthropic Claude Fable 5.2 / Mythos 5.2).
    """
    if ngram_len < 2:
        raise ValueError("ngram_len must be at least 2")
    if not keys:
        raise ValueError("keys must be non-empty")
    pairs = _iter_unique_ngrams(token_ids, ngram_len)
    if not pairs:
        return _make_watermark_score("synthid", 0.0, 0, threshold_z)

    g_sum = sum(
        _prf_bit_or_uniform(context, token_id, layer_key) & 1 for context, token_id in pairs for layer_key in keys
    )
    scored_tokens = len(pairs)
    m_layers = len(keys)
    mean_g = g_sum / (scored_tokens * m_layers)
    z_score = (mean_g - 0.5) / math.sqrt(0.25 / (scored_tokens * m_layers))
    return _make_watermark_score("synthid", z_score, scored_tokens, threshold_z)


def score_gumbel_text(
    token_ids: Sequence[int],
    secret_key: int,
    *,
    context_len: int = 3,
    threshold_z: float = DEFAULT_ALPHA_001_Z,
) -> TextWatermarkScore:
    """Compute the keyed distortion-free Gumbel-max / ``textGrain`` z-score over ``token_ids``."""
    if context_len < 1:
        raise ValueError("context_len must be at least 1")
    pairs = _iter_unique_ngrams(token_ids, context_len + 1)
    exp_sum = sum(-math.log(1.0 - _prf_uniform_01(context, token_id, secret_key)) for context, token_id in pairs)
    return _exp_race_score("gumbel", exp_sum, len(pairs), threshold_z)


def score_textgrain_text(
    token_ids: Sequence[int],
    secret_key: int,
    *,
    context_len: int = 3,
    num_blocks: int = 256,
    num_columns: int = 256,
    threshold_z: float = DEFAULT_ALPHA_001_Z,
) -> TextWatermarkScore:
    """Compute OpenAI's 2026 ``textGrain`` block-column Sinkhorn coupling z-score.

    Implements the three-subkey detection law from Li et al. (2026) / ``textseal``:
    block assignment ``b = floor(PRF(w, x, k + SUB_PART) * B)``, active column
    ``J = floor(PRF(w, 0, k + SUB_COL) * m)``, and exponential cell score
    ``Y_t = -ln(1 - PRF(w, b * m + J, k + SUB_COST))`` deduplicated per unique
    ``(context, token)`` window.
    """
    if context_len < 1:
        raise ValueError("context_len must be at least 1")
    if num_blocks < 2 or num_columns < 2:
        raise ValueError("num_blocks and num_columns must be at least 2")
    pairs = _iter_unique_ngrams(token_ids, context_len + 1)
    exp_sum = 0.0
    for context, token_id in pairs:
        u_part = _prf_uniform_01(context, token_id, secret_key + _TEXTGRAIN_SUB_PART)
        block = min(int(u_part * num_blocks), num_blocks - 1)
        col = min(int(_prf_uniform_01(context, 0, secret_key + _TEXTGRAIN_SUB_COL) * num_columns), num_columns - 1)
        cell = block * num_columns + col
        u_cost = _prf_uniform_01(context, cell, secret_key + _TEXTGRAIN_SUB_COST)
        exp_sum += -math.log(1.0 - u_cost)
    return _exp_race_score("textgrain", exp_sum, len(pairs), threshold_z)


def score_textseal2_text(
    token_ids: Sequence[int],
    key_a: int,
    key_b: int | None = None,
    *,
    context_len: int = 3,
    threshold_z: float = DEFAULT_ALPHA_001_Z,
) -> TextWatermarkScore:
    """Compute Meta's ``TextSeal2`` two-key Gaussian copula virtual-key z-score.

    Evaluates ``G_A = Phi^{-1}(u_A)`` and ``G_B = Phi^{-1}(u_B)``, combines them into
    the virtual key ``G* = (G_A + G_B) / sqrt(2)``, and scores each unique ``(w, x)``
    pair via ``s_t = -ln Phi(-G*_t) ~ Exp(1)`` under ``H_0``.
    """
    if context_len < 1:
        raise ValueError("context_len must be at least 1")
    pairs = _iter_unique_ngrams(token_ids, context_len + 1)
    resolved_key_b = key_a + _TEXTSEAL_DUAL_KEY_OFFSET if key_b is None else key_b
    exp_sum = 0.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for context, token_id in pairs:
        u_a = _prf_uniform_01(context, token_id, key_a)
        u_b = _prf_uniform_01(context, token_id, resolved_key_b)
        g_star = (_STD_NORMAL.inv_cdf(u_a) + _STD_NORMAL.inv_cdf(u_b)) * inv_sqrt2
        phi_neg = min(max(_STD_NORMAL.cdf(-g_star), 1e-15), 1.0 - 1e-15)
        exp_sum += -math.log(phi_neg)
    return _exp_race_score("textseal2", exp_sum, len(pairs), threshold_z)


def _score_greenlist(
    scheme: Literal["kgw", "unigram"],
    token_ids: Sequence[int],
    secret_key: int,
    *,
    context_width: int,
    greenlist_ratio: float,
    threshold_z: float,
) -> TextWatermarkScore:
    if not 0.0 < greenlist_ratio < 1.0:
        raise ValueError("greenlist_ratio must be in (0, 1)")
    if len(token_ids) <= context_width:
        return _make_watermark_score(scheme, 0.0, 0, threshold_z)

    green_hits = 0
    scored_tokens = 0
    cutoff = int(greenlist_ratio * (1 << 64))

    for idx in range(context_width, len(token_ids)):
        context = tuple(int(t) for t in token_ids[idx - context_width : idx]) if context_width > 0 else ()
        if _prf_bit_or_uniform(context, int(token_ids[idx]), secret_key) < cutoff:
            green_hits += 1
        scored_tokens += 1

    z_score = (green_hits - greenlist_ratio * scored_tokens) / math.sqrt(
        scored_tokens * greenlist_ratio * (1.0 - greenlist_ratio)
    )
    return _make_watermark_score(scheme, z_score, scored_tokens, threshold_z)


def score_kgw_text(
    token_ids: Sequence[int],
    secret_key: int,
    *,
    context_width: int = 1,
    greenlist_ratio: float = 0.25,
    threshold_z: float = DEFAULT_ALPHA_001_Z,
) -> TextWatermarkScore:
    """Compute the keyed KGW Red-Green List z-score over ``token_ids``."""
    if context_width < 1:
        raise ValueError("context_width must be at least 1")
    return _score_greenlist(
        "kgw",
        token_ids,
        secret_key,
        context_width=context_width,
        greenlist_ratio=greenlist_ratio,
        threshold_z=threshold_z,
    )


def score_unigram_text(
    token_ids: Sequence[int],
    secret_key: int,
    *,
    greenlist_ratio: float = 0.25,
    threshold_z: float = DEFAULT_ALPHA_001_Z,
) -> TextWatermarkScore:
    """Compute the keyed context-free Unigram watermark z-score over ``token_ids``."""
    return _score_greenlist(
        "unigram",
        token_ids,
        secret_key,
        context_width=0,
        greenlist_ratio=greenlist_ratio,
        threshold_z=threshold_z,
    )
