"""Tests for text provenance inspection, character steganography removal, and token watermark regeneration."""

from __future__ import annotations

import json
import struct
import sys
import zlib
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

import remove_ai_watermarks as raiw
from remove_ai_watermarks.cli import main
from remove_ai_watermarks.text import (
    C2PATXT_MAGIC,
    C2PATXT_VERSION,
    VSRMARK_MAGIC,
    VSRMARK_VERSION,
    byte_to_variation_selector,
    decode_stego_payloads,
    decode_variation_selectors,
    encode_c2pa_ascii_armor,
    encode_c2patxt,
    encode_innamark,
    encode_unicode_tags,
    encode_variation_selectors,
    encode_vsrmark,
    identify_text,
    remove_text_all,
    remove_text_invisible,
    remove_text_metadata,
    score_gumbel_text,
    score_kgw_text,
    score_synthid_text,
    score_textgrain_text,
    score_textseal2_text,
    score_unigram_text,
    split_protected_text_spans,
    variation_selector_to_byte,
    verify_text_rewrite_facts,
)

if TYPE_CHECKING:
    from pathlib import Path

SYNTHETIC_JUMBF_MANIFEST = (
    b"jumbc2pa\x00\x00\x00\x01"
    b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    b"OpenAI ChatGPT GPT-6 Astra com.cognitive-proof.vsrmark.1"
    b"20261109120000Z"
)


def test_variation_selector_byte_bijection_covers_all_256_bytes() -> None:
    assert C2PATXT_MAGIC == b"C2PATXT\x00"
    assert C2PATXT_VERSION == 1
    assert VSRMARK_MAGIC == b"VSR1"
    assert VSRMARK_VERSION == 1
    raw = bytes(range(256))
    encoded = encode_variation_selectors(raw)
    assert len(encoded) == 256
    assert ord(encoded[0]) == 0xFE00
    assert ord(encoded[15]) == 0xFE0F
    assert ord(encoded[16]) == 0xE0100
    assert ord(encoded[255]) == 0xE01EF
    assert decode_variation_selectors(encoded) == raw
    assert variation_selector_to_byte("A") is None
    assert variation_selector_to_byte("ab") is None
    with pytest.raises(ValueError, match=r"0\.\.255"):
        byte_to_variation_selector(256)


def test_c2patxt_appendix_a8_roundtrip_inspect_and_strip(tmp_path: Path) -> None:
    carrier = "Generated analysis paragraph with ❤️ and 👩‍💻 emoji.\n"
    watermarked = encode_c2patxt(carrier, SYNTHETIC_JUMBF_MANIFEST, padding_bytes=8)

    report = identify_text(watermarked)
    assert report.is_ai_generated is True
    assert report.confidence == "high"
    assert "c2pa_text_manifest_wrapper" in report.signals
    assert report.c2pa_info["ai_source_kind"] == "generated"
    assert "ChatGPT" in (report.platform or "")
    assert "8 bytes padding" in report.metadata_markers["c2pa_text_wrapper"]
    assert report.stego_codepoint_count == 1 + 13 + len(SYNTHETIC_JUMBF_MANIFEST) + 8

    src = tmp_path / "doc.txt"
    src.write_text(watermarked, encoding="utf-8")
    res = remove_text_metadata(src)
    assert res.output == tmp_path / "doc_clean.txt"
    assert res.cleaned_text == carrier
    assert res.remaining_markers == {}
    assert res.output is not None
    assert res.output.read_text(encoding="utf-8") == carrier


def test_c2patxt_without_bom_sentinel_reports_caveat() -> None:
    carrier = "Plain text body."
    header = struct.pack("!8sBI", b"C2PATXT\x00", 1, len(SYNTHETIC_JUMBF_MANIFEST))
    without_bom = carrier + encode_variation_selectors(header + SYNTHETIC_JUMBF_MANIFEST)

    report = identify_text(without_bom)
    assert report.is_ai_generated is True
    assert "c2pa_text_manifest_wrapper" in report.signals
    assert any("U+FEFF" in caveat for caveat in report.caveats)
    assert remove_text_metadata(without_bom).cleaned_text == carrier


def test_c2pa_ascii_armor_appendix_a9_html_and_hash_comments() -> None:
    body = "# Heading\n\nBody paragraph.\n"
    html_armored = encode_c2pa_ascii_armor(SYNTHETIC_JUMBF_MANIFEST, comment_style="html") + body
    report = identify_text(html_armored)
    assert report.is_ai_generated is True
    assert report.confidence == "high"
    assert "c2pa_ascii_armor" in report.signals
    assert "Inline JUMBF" in report.metadata_markers["c2pa_ascii_armor"]
    assert remove_text_metadata(html_armored).cleaned_text == body

    hash_armored = encode_c2pa_ascii_armor("https://example.com/manifests/123.c2pa", comment_style="hash") + body
    uri_report = identify_text(hash_armored)
    assert uri_report.is_ai_generated is True
    assert "https://example.com/manifests/123.c2pa" in uri_report.metadata_markers["c2pa_ascii_armor"]
    assert remove_text_metadata(hash_armored).cleaned_text == body


def test_c2pa_html_appendix_a7_single_and_multiple_elements() -> None:
    import base64

    b64 = base64.b64encode(SYNTHETIC_JUMBF_MANIFEST).decode("ascii")
    html_doc = (
        "<!doctype html>\n<html>\n<head>\n"
        "  <title>Test</title>\n"
        f'  <script type="application/c2pa">data:application/c2pa;base64,{b64}</script>\n'
        '  <link rel="c2pa-manifest" href="https://example.com/manifest.c2pa">\n'
        "</head>\n<body><p>Hello world.</p></body>\n</html>\n"
    )
    report = identify_text(html_doc)
    assert report.is_ai_generated is True
    assert "c2pa_html_head" in report.signals
    assert any("multiple C2PA manifest elements" in caveat for caveat in report.caveats)

    cleaned = remove_text_metadata(html_doc).cleaned_text
    assert "application/c2pa" not in cleaned
    assert "c2pa-manifest" not in cleaned
    assert "<title>Test</title>" in cleaned
    assert "<p>Hello world.</p>" in cleaned


def test_vsrmark_soft_binding_roundtrip_and_crc_validation() -> None:
    carrier = "This carrier sentence has enough visible characters to hold a VSRMark claim."
    payload = '{"claim":"claude-fable-5-2","id":"9f8a"}'
    marked = encode_vsrmark(carrier, payload)

    report = identify_text(marked)
    assert report.is_ai_generated is True
    assert report.confidence == "high"
    assert "vsrmark" in report.signals
    assert report.vsrmark_payload == payload
    assert "com.cognitive-proof.vsrmark.1" in report.metadata_markers["vsrmark"]

    cleaned = remove_text_metadata(marked)
    assert cleaned.cleaned_text == carrier
    assert cleaned.remaining_markers == {}

    # Corrupting the CRC32 must disqualify VSRMark attribution while still flagging steganography
    bad_body = struct.pack(">4sBH", VSRMARK_MAGIC, VSRMARK_VERSION, 4) + b"test"
    bad_crc = (zlib.crc32(bad_body) ^ 0xFFFFFFFF) & 0xFFFFFFFF
    bad_frame = bad_body + struct.pack(">I", bad_crc)
    corrupted = carrier[:15] + encode_variation_selectors(bad_frame) + carrier[15:]
    corrupted_report = identify_text(corrupted)
    assert "vsrmark" not in corrupted_report.signals
    assert "zero_width_steganography" in corrupted_report.signals
    assert remove_text_metadata(corrupted).cleaned_text == carrier


def test_unicode_tags_and_zero_width_stego_preserve_emojis() -> None:
    clean_with_emojis = " Shipping ❤️ and 👩‍💻 with zero steganography."
    clean_report = identify_text(clean_with_emojis)
    assert clean_report.is_ai_generated is None
    assert clean_report.has_ai_metadata is False
    assert remove_text_metadata(clean_with_emojis).cleaned_text == clean_with_emojis

    tagged = encode_unicode_tags("Hello\u200b\u200c world ❤️ 👩‍💻!", "ASTRA-TEXTGRAIN")
    report = identify_text(tagged)
    assert report.is_ai_generated is True
    assert report.confidence == "medium"
    assert "unicode_tags" in report.signals
    assert "zero_width_steganography" in report.signals
    assert report.unicode_tags_payload == "ASTRA-TEXTGRAIN"

    stripped = remove_text_metadata(tagged)
    assert stripped.cleaned_text == "Hello world ❤️ 👩‍💻!"
    assert stripped.remaining_markers == {}


def test_span_protected_regeneration_locks_technical_spans_verbatim(tmp_path: Path) -> None:
    document = (
        "# Architecture Overview\n\n"
        "The model computes `y = f(x)` using threshold $\\alpha = 0.01$ and publishes "
        "results to https://raiw.cc/docs/spec from ./src/runner.py.\n\n"
        "```python\n"
        "def compute(x: int) -> int:\n"
        "    return x * 2\n"
        "```\n\n"
        "$$\\sum_{i=1}^{n} x_i = 1$$\n\n"
        "A second prose paragraph explains the deployment parameters in detail.\n"
    )
    spans = split_protected_text_spans(document)
    kinds = {span.kind for span in spans}
    assert {
        "markdown_structure",
        "prose",
        "inline_code",
        "math_inline",
        "url",
        "path",
        "fenced_code",
        "math_block",
    } <= kinds

    # 1) Rewriter that preserves [[RAIW_SPAN_k]] placeholders
    def preserving_rewriter(text: str) -> str:
        return text.replace("The model computes", "Our system evaluates").replace(
            "A second prose paragraph explains", "The concluding section describes"
        )

    out_path = tmp_path / "doc_clean.md"
    res1 = remove_text_invisible(document, out_path, rewriter=preserving_rewriter)
    assert res1.token_watermark_action == "regenerated"
    assert res1.rewritten_spans >= 2
    assert "Our system evaluates `y = f(x)` using threshold $\\alpha = 0.01$" in res1.cleaned_text
    assert "https://raiw.cc/docs/spec" in res1.cleaned_text
    assert "`./src/runner.py`" in res1.cleaned_text or "./src/runner.py" in res1.cleaned_text
    assert "def compute(x: int) -> int:\n    return x * 2" in res1.cleaned_text
    assert "$$\\sum_{i=1}^{n} x_i = 1$$" in res1.cleaned_text

    # 2) Rewriter that drops placeholders (verifies the per-span fallback still preserves locked spans)
    def dropping_rewriter(_text: str) -> str:
        return "Rewritten prose segment"

    res2 = remove_text_invisible(document, rewriter=dropping_rewriter)
    for locked_literal in (
        "# Architecture Overview",
        "`y = f(x)`",
        "$\\alpha = 0.01$",
        "https://raiw.cc/docs/spec",
        "./src/runner.py",
        "def compute(x: int) -> int:\n    return x * 2",
        "$$\\sum_{i=1}^{n} x_i = 1$$",
    ):
        assert locked_literal in res2.cleaned_text


def test_keyed_statistical_watermark_scorers_detect_and_clear_after_regeneration() -> None:
    unwatermarked = [((i * 1103515245 + 12345) >> 16) % 4990 + 10 for i in range(240)]
    synthid_keys = tuple(range(101, 131))
    secret_key = 99991

    assert not score_synthid_text(unwatermarked, synthid_keys).detected
    assert not score_gumbel_text(unwatermarked, secret_key).detected
    assert not score_kgw_text(unwatermarked, secret_key).detected
    assert not score_unigram_text(unwatermarked, secret_key).detected

    # Construct a sequence watermarked by tournament selection (SynthID-Text)
    watermarked_synthid = list(unwatermarked[:4])
    for step in range(4, 240):
        candidates = [step * 31 + c for c in range(8)]
        context = tuple(watermarked_synthid[-4:])
        best_tok = max(
            candidates,
            key=lambda tok: score_synthid_text([*context, tok], synthid_keys, ngram_len=5).z_score,
        )
        watermarked_synthid.append(best_tok)

    detected_score = score_synthid_text(watermarked_synthid, synthid_keys)
    assert detected_score.detected is True
    assert detected_score.z_score > 5.0

    # Keeping 20% locked low-entropy tokens and regenerating 80% prose tokens clears the watermark
    locked_code = [100 + i for i in range(48)]
    fresh_prose = [((i * 214013 + 2531011) >> 16) % 9000 + 6000 for i in range(192)]
    regenerated = locked_code + fresh_prose
    cleared_score = score_synthid_text(regenerated, synthid_keys)
    assert cleared_score.detected is False
    assert cleared_score.z_score < 2.326


def test_cli_text_commands_and_missing_runtime_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    src = tmp_path / "sample.md"
    watermarked = encode_c2patxt("Sample prose with ❤️.\n", SYNTHETIC_JUMBF_MANIFEST)
    src.write_text(watermarked, encoding="utf-8")

    id_res = runner.invoke(main, ["text", "identify", str(src), "--json"], catch_exceptions=False)
    assert id_res.exit_code == 0
    payload = json.loads(id_res.output)
    assert payload["is_ai_generated"] is True
    assert "c2pa_text_manifest_wrapper" in payload["signals"]

    meta_out = tmp_path / "sample_meta.md"
    meta_res = runner.invoke(
        main,
        ["text", "metadata", str(src), "--remove", "-o", str(meta_out)],
        catch_exceptions=False,
    )
    assert meta_res.exit_code == 0
    assert meta_out.read_text(encoding="utf-8") == "Sample prose with ❤️.\n"

    all_out = tmp_path / "sample_all.md"
    all_res = runner.invoke(
        main,
        ["text", "all", str(src), "-o", str(all_out)],
        catch_exceptions=False,
    )
    assert all_res.exit_code == 0
    assert all_out.read_text(encoding="utf-8") == "Sample prose with ❤️.\n"

    # When torch/transformers are absent, `text invisible` prints an install hint instead of a traceback
    monkeypatch.setitem(sys.modules, "transformers", None)
    inv_res = runner.invoke(main, ["text", "invisible", str(meta_out), "-o", str(tmp_path / "inv.md")])
    assert inv_res.exit_code != 0
    assert "remove-ai-watermarks[diffusion]" in inv_res.output

    # Top-level raiw exports resolve lazily
    assert raiw.identify_text is identify_text
    assert raiw.remove_text_all is remove_text_all
    assert raiw.score_textgrain_text is score_textgrain_text
    assert raiw.score_textseal2_text is score_textseal2_text
    assert raiw.verify_text_rewrite_facts is verify_text_rewrite_facts


def test_subdivision_flag_emoji_and_joining_scripts_preserved_while_pua_and_zw_delimiters_stripped() -> None:
    england_flag = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    skin_tone_couple = "\U0001f468\U0001f3fb\u200d\u2764\ufe0f\u200d\U0001f468\U0001f3fb"
    persian_zwnj = "می\u200cشود"
    clean_base = f"Flag {england_flag}, emoji {skin_tone_couple}, and Persian {persian_zwnj}."

    # Clean base with subdivision flag, skin-tone/VS16 ZWJ emoji, and Persian ZWNJ reports no watermark
    clean_report = identify_text(clean_base)
    assert clean_report.is_ai_generated is None
    assert clean_report.stego_codepoint_count == 0
    assert remove_text_metadata(clean_base).cleaned_text == clean_base

    # Inject standalone Unicode Tags, zengkan0703-style U+200D between Latin chars, ChatGPT PUA U+E200..U+E202,
    # and jaceddd-style deprecated format controls U+206A..U+206F + U+00AD
    dirty = (
        encode_unicode_tags(f"Flag {england_flag} ", "claude-fable-5-2")
        + " H\u200de\u200cl\u00adl\u206co\ue200cite\ue202turn0search0\ue201 "
        + f"emoji {skin_tone_couple}, and Persian {persian_zwnj}."
    )
    dirty_report = identify_text(dirty)
    assert dirty_report.is_ai_generated is True
    assert dirty_report.unicode_tags_payload == "claude-fable-5-2"
    assert "unicode_tags" in dirty_report.signals
    assert "zero_width_steganography" in dirty_report.signals

    stripped = remove_text_metadata(dirty)
    assert stripped.cleaned_text == (
        f"Flag {england_flag}  Hello emoji {skin_tone_couple}, and Persian {persian_zwnj}."
    )
    assert identify_text(stripped.cleaned_text).is_ai_generated is None


def test_innamark_stegcloak_subtext_stegmoji_groundtruth_and_zw_binary_forensics() -> None:
    cover = " ".join(f"word{i}" for i in range(120))
    innamark_text = encode_innamark(cover, "author:astra-6", compressed=True, crc32=True)
    report = identify_text(innamark_text)
    assert report.is_ai_generated is True
    assert report.confidence == "high"
    assert "innamark_whitespace" in report.signals
    assert report.metadata_markers["innamark_payload"] == "author:astra-6"
    assert decode_stego_payloads(innamark_text)["innamark"] == "author:astra-6"

    cleaned_innamark = remove_text_metadata(innamark_text)
    assert cleaned_innamark.cleaned_text == cover
    assert identify_text(cleaned_innamark.cleaned_text).is_ai_generated is None

    # StegCloak unencrypted stream: flag U+200C (pair [\u200c, \u200d]) + mode U+2061 (unencrypted)
    # + dibits for complement of b"ok" (0x6F ^ 0xFF = 0x90 = 10_01_00_00, 0x6B ^ 0xFF = 0x94 = 10_01_01_00)
    # where \u200c\u200c is compressed to \u2063
    stegcloak_stream = "\u200c\u2061\u2061\u200d\u2063\u2061\u200d\u200d\u200c"
    stegcloak_doc = f"Normal {stegcloak_stream} sentence."
    sc_report = identify_text(stegcloak_doc)
    assert "stegcloak" in sc_report.signals
    assert sc_report.metadata_markers["stegcloak"] == "unencrypted:ok"
    assert remove_text_metadata(stegcloak_doc).cleaned_text == "Normal  sentence."

    # GroundTruth-MCP 64-bit U+2061 (0) / U+2062 (1) watermark
    gt_bits = f"{0xCAFEBABE:032b}{0x12345678:032b}"
    gt_stream = "".join("\u2061" if b == "0" else "\u2062" for b in gt_bits)
    gt_doc = f"First line\n{gt_stream}Second line."
    gt_report = identify_text(gt_doc)
    assert "groundtruth_mcp" in gt_report.signals
    assert gt_report.metadata_markers["groundtruth_mcp_watermark"] == "install_id=0xcafebabe, nonce=0x12345678"
    assert remove_text_metadata(gt_doc).cleaned_text == "First line\nSecond line."

    # Subtext v1 frame (0xAB 0xCD + uint32 len + version 1 + flags 0 + type 1 + UTF-8 body) with U+200A splitter
    subtext_body = b"\x01hi!"
    subtext_frame = b"\xab\xcd" + len(subtext_body).to_bytes(4, "big") + b"\x01\x00" + subtext_body
    subtext_vs = (
        encode_variation_selectors(subtext_frame[:6]) + "\u200a" + encode_variation_selectors(subtext_frame[6:])
    )
    subtext_doc = f"Carrier{subtext_vs} text."
    assert decode_stego_payloads(subtext_doc)["subtext"] == "hi!"
    assert remove_text_metadata(subtext_doc).cleaned_text == "Carrier text."

    # StegMoji VS15 (0) / VS16 (1) binary channel with flags=0x00 + b"AI"
    stegmoji_bytes = b"\x00AI"
    stegmoji_bits = "".join(f"{b:08b}" for b in stegmoji_bytes)
    stegmoji_vs = "".join("\ufe0e" if bit == "0" else "\ufe0f" for bit in stegmoji_bits)
    stegmoji_doc = f"Emoji{stegmoji_vs} carrier."
    assert decode_stego_payloads(stegmoji_doc)["stegmoji"] == "AI"
    assert remove_text_metadata(stegmoji_doc).cleaned_text == "Emoji carrier."


def test_cjk_ivs_rtl_bidi_non_joining_arabic_and_homoglyph_entity_scrubbing() -> None:
    # Single CJK IVS selector (U+E0100 after CJK ideograph 葛 U+845B) is preserved,
    # and balanced RTL Bidi isolate (U+2067..U+2069) on an Arabic line is preserved,
    # and legitimate typographic NBSP next to digits (100\u00a0km) is preserved.
    legit = "漢字葛\U000e0100と100\u00a0km\nمرحب\u0627 \u2067ع\u0627لم\u2069 جميل"
    legit_report = identify_text(legit)
    assert legit_report.is_ai_generated is None
    assert remove_text_metadata(legit).cleaned_text == legit

    # Right-non-joining Arabic Alef (U+0627) followed by U+200C (ZWNJ) is detected and stripped as a watermark,
    # Trojan Source U+202E override is stripped, smuggled tag after a valid flag's U+E007F is stripped,
    # invisible HTML entity &#x200B; and &ZeroWidthSpace; are decoded and stripped outside inline code,
    # and mixed-script homoglyph with transparent ZWSP (passw\u200b\u043erd) is normalized to password!
    england_flag = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    smuggled = (
        f"{england_flag}\U000e0078 "
        "\u0627\u200cب "
        "\u202ealert "
        "passw&#x200B;\u043erd &ZeroWidthSpace; "
        "`code &#x200B; passw\u043erd` "
        "пароль"
    )
    smuggled_report = identify_text(smuggled)
    assert smuggled_report.is_ai_generated is True
    assert "homoglyph_steganography" in smuggled_report.signals
    assert "unicode_tags" in smuggled_report.signals
    assert "zero_width_steganography" in smuggled_report.signals

    cleaned = remove_text_metadata(smuggled).cleaned_text
    assert cleaned == f"{england_flag} \u0627ب alert password  `code &#x200B; passw\u043erd` пароль"

    # Bidirectional homoglyph normalization: Latin 'a'/'o'/'c' injected inside Russian Cyrillic words
    # (U+043F + Latin "a" + U+0440 + Latin "o" + U+043B U+044C -> "пароль", and Latin "c" + U+043E U+0443 U+0441)
    # and Fullwidth Latin (U+FF41) injected inside ASCII Latin ("p\uff41ssword" -> "password")
    cyr_spoofed = "\u043f" + "a" + "\u0440" + "o" + "\u043b\u044c " + "c" + "\u043e\u0443\u0441 p\uff41ssword"
    cyr_report = identify_text(cyr_spoofed)
    assert cyr_report.is_ai_generated is True
    assert "homoglyph_steganography" in cyr_report.signals
    assert remove_text_metadata(cyr_spoofed).cleaned_text == "пароль \u0441\u043e\u0443\u0441 password"


def test_verify_text_rewrite_facts_spelled_numbers_proper_nouns_and_antonym_poles() -> None:
    # Spelled-out number words match digit equivalents ("forty" <-> "40")
    assert verify_text_rewrite_facts(
        "The cluster processed forty requests per second.",
        "Throughput reached 40 requests every second.",
    )
    # Missing mid-sentence proper noun ("Postgres" and Cyrillic "Казахстан") is rejected
    assert not verify_text_rewrite_facts(
        "The service stores ledger rows in Postgres for durability.",
        "The service stores ledger rows in a database for durability.",
    )
    assert verify_text_rewrite_facts(
        "Серверы расположены в Казахстан для низкой задержки.",
        "Узлы кластера развернуты в Казахстан для низкой задержки.",
    )
    assert not verify_text_rewrite_facts(
        "Серверы расположены в Казахстан для низкой задержки.",
        "Серверы расположены в регионе для низкой задержки.",
    )
    # Antonym pole inversion ("increased" -> "decreased", "allowed" -> "prohibited") is rejected
    assert not verify_text_rewrite_facts(
        "Latency increased during the migration window.",
        "Latency decreased during the migration window.",
    )
    assert not verify_text_rewrite_facts(
        "External access is allowed for verified operators.",
        "External access is prohibited for verified operators.",
    )


def test_verify_text_rewrite_facts_and_textgrain_textseal2_scorers() -> None:
    # Numeral preservation (including comma normalization)
    assert verify_text_rewrite_facts("Revenue grew by 1,250 units in 2026.", "In 2026, output rose by 1250 units.")
    assert not verify_text_rewrite_facts("Revenue grew by 1,250 units in 2026.", "In 2026, output rose significantly.")

    # NegEx single-sentence polarity check vs pseudo-negations ("not only")
    assert not verify_text_rewrite_facts(
        "PlayStation consoles are computers.",
        "PlayStation consoles aren't computers.",
    )
    assert verify_text_rewrite_facts(
        "The system is not only fast but accurate.",
        "The architecture is both rapid and precise.",
    )

    # Keyed textGrain and TextSeal2 scorers
    unwatermarked = [((i * 1103515245 + 12345) >> 16) % 4990 + 10 for i in range(240)]
    secret_key = 424242
    tg_null = score_textgrain_text(unwatermarked, secret_key)
    ts2_null = score_textseal2_text(unwatermarked, secret_key)
    assert tg_null.scheme == "textgrain"
    assert ts2_null.scheme == "textseal2"
    assert not tg_null.detected
    assert not ts2_null.detected

    # Construct a sequence watermarked under textGrain and verify detection
    watermarked_tg = list(unwatermarked[:3])
    for step in range(3, 180):
        candidates = [step * 37 + c for c in range(8)]
        ctx = tuple(watermarked_tg[-3:])
        best_tok = max(candidates, key=lambda tok: score_textgrain_text([*ctx, tok], secret_key).z_score)
        watermarked_tg.append(best_tok)
    tg_pos = score_textgrain_text(watermarked_tg, secret_key)
    assert tg_pos.detected is True
    assert tg_pos.z_score > 5.0

    # Construct a sequence watermarked under TextSeal2 and verify detection
    watermarked_ts2 = list(unwatermarked[:3])
    for step in range(3, 180):
        candidates = [step * 41 + c for c in range(8)]
        ctx = tuple(watermarked_ts2[-3:])
        best_tok = max(candidates, key=lambda tok: score_textseal2_text([*ctx, tok], secret_key).z_score)
        watermarked_ts2.append(best_tok)
    ts2_pos = score_textseal2_text(watermarked_ts2, secret_key)
    assert ts2_pos.detected is True
    assert ts2_pos.z_score > 5.0
