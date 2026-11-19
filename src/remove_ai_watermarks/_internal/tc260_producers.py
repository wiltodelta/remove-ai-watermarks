"""Registered TC260 producer identities, mark routing, and pinned signing keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class Tc260Producer:
    """One producer's measured codes and independently established identity."""

    key: str
    codes: tuple[str, ...]
    vendor: str
    image_mark: str | None = None
    video_mark: str | None = None
    video_platform: str | None = None
    signer: str | None = None
    signing_key_x: int | None = None
    image_platform: str | None = None


TC260_PRODUCERS = (
    # YOYO JPEG exports; USCC independently verified in Honor's privacy statement.
    Tc260Producer("honor", ("91440300MA5G49LC9K",), "Honor", image_platform="Honor"),
    Tc260Producer("doubao", ("91110102MACQD9K640", "doubao"), "ByteDance", "doubao", "doubao", "ByteDance Doubao"),
    Tc260Producer(
        "jimeng", ("9144030008867405X2",), "ByteDance", "jimeng", "dreamina", video_platform="ByteDance Dreamina"
    ),
    Tc260Producer(
        "volcengine",
        ("91110108MA01R70K8D",),
        "ByteDance",
        video_platform="ByteDance Volcano Engine",
        image_platform="ByteDance Volcano Engine",
    ),
    Tc260Producer("qwen", ("91440101MA9Y9T4H7A",), "Alibaba", "qwen", video_platform="Alibaba Cloud Qwen"),
    # Tongyi Yunqi (Hangzhou), owned by Alibaba Cloud and Tongyi Lab, signs
    # both Wan and HappyHorse exports, not just one model.
    Tc260Producer(
        "wan", ("91330106MA2CFLDG4R",), "Alibaba", "wan", "wan", video_platform="Alibaba Tongyi (Wan, HappyHorse)"
    ),
    # Kling 3.0 Turbo served by Higgsfield (2026-09-24) writes the bare "kling".
    Tc260Producer("kling", ("91110108335469089C", "kling"), "Kuaishou", "kling", "kling", "Kuaishou Kling AI"),
    Tc260Producer(
        "yuanbao",
        ("91440300708461136T", "9144030071526726XG", "91440101327598294H"),
        "Tencent",
        "yuanbao",
        video_platform="Tencent Yuanbao",
        image_platform="Tencent Yuanbao",
    ),
    Tc260Producer(
        "workbuddy",
        ("911101085636549482",),
        "Tencent",
        "workbuddy",
        image_platform="Tencent WorkBuddy",
    ),
    Tc260Producer("runninghub", ("91340100MAEB4N8H76", "RunningHub"), "RunningHub", "runninghub"),
    Tc260Producer("baidu", ("91110000802100433B",), "Baidu", "baidu", video_platform="Baidu"),
    Tc260Producer("liblib", ("91110105MACJ6K1C8A",), "LiblibAI", "liblib"),
    Tc260Producer("meitu", ("913502007378955153",), "Meitu", image_platform="Meitu"),
    Tc260Producer("vivo", ("91441900557262083U",), "vivo", image_platform="vivo"),
    Tc260Producer("xiaomi", ("91110108596084056A",), "Xiaomi", image_platform="Xiaomi"),
    Tc260Producer("agnes", ("91320191MAKHJ19D9J",), "Agnes AI", video_platform="Agnes AI", image_platform="Agnes AI"),
    # ShengShu's USCC, verified against the Beijing municipal list of 2025-03-03.
    Tc260Producer("vidu", ("91110108MACC4D63XF",), "ShengShu", video_mark="vidu", video_platform="ShengShu Vidu"),
    # Even-y SM2 point for the x coordinate carried by measured MiniMax exports
    # (Hailuo 2.3, H3 and H3 Max, 2026-09-29). Source: docs/research-sweep-2026-09.md.
    Tc260Producer(
        "minimax",
        ("MiniMax",),
        "MiniMax",
        video_mark="hailuo",
        video_platform="MiniMax Hailuo AI",
        signer="MiniMax",
        signing_key_x=0xA0B3B0B6A0C9B0C89CAB328342AF4E8221EC5B40799CBE835AB4251F7B47E4FD,
    ),
)


def uscc_of(code: str) -> str:
    """Extract the USCC from ``001`` + type digit + USCC + product suffix."""
    return code[4:22] if len(code) >= 22 and code[:3] == "001" else code


_PRODUCERS_BY_CODE = {code.casefold(): row for row in TC260_PRODUCERS for code in row.codes}


def tc260_producer_in(producer: str, codes: Iterable[str]) -> bool:
    """Match a whole producer identity, USCC-normalized and casefolded."""
    code = uscc_of(producer.strip()).casefold()
    return bool(code) and any(code == candidate.casefold() for candidate in codes)


def producer_for_code(code: str) -> Tc260Producer | None:
    """Resolve a producer field; partial names never establish an identity."""
    return _PRODUCERS_BY_CODE.get(uscc_of(code.strip()).casefold())


def producer_for_signer(signer: str) -> Tc260Producer | None:
    """Resolve the identity returned by successful pinned-key verification."""
    return next((row for row in TC260_PRODUCERS if row.signer == signer), None)


def producer_codes_for_mark(mark: str) -> tuple[str, ...]:
    """Measured producer identities for an image or video mark, if registered."""
    row = next((row for row in TC260_PRODUCERS if mark in (row.image_mark, row.video_mark)), None)
    return row.codes if row else ()
