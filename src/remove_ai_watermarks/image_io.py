"""Unicode-safe cv2 image IO (issue #17).

``cv2.imread`` / ``cv2.imwrite`` pass the path to the platform C runtime, which
on Windows uses the narrow (ANSI) code-page API and therefore fails on paths
containing non-ASCII characters (Chinese, Cyrillic, ...). The symptom is a
``can't open/read file`` warning and a ``None`` decode even though the file
exists.

These wrappers route through numpy buffers instead: ``np.fromfile`` /
``ndarray.tofile`` open the path in Python (full Unicode), and
``cv2.imdecode`` / ``cv2.imencode`` do the codec work. The decoded/encoded
bytes are byte-for-byte identical to ``imread`` / ``imwrite``. On macOS/Linux
cv2 already accepts UTF-8 paths, so the wrappers are behavior-neutral there.

cv2/numpy are imported lazily inside the functions so importing this module
stays cheap in a bare environment (matching the rest of the package).
"""

# cv2 ships no type stubs; mirror the pragma used by the other cv2-using modules.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks._internal.constants import PNG_SIGNATURE

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


def imread(path: str | Path, flags: int | None = None) -> NDArray[Any] | None:
    """Unicode-safe ``cv2.imread`` with a Pillow fallback for HEIC/AVIF.

    ``flags`` defaults to ``cv2.IMREAD_COLOR`` (same as ``cv2.imread``). Returns
    ``None`` when the file is missing or cannot be decoded, matching ``cv2.imread``
    semantics so existing ``if img is None`` checks keep working.

    OpenCV cannot decode HEIC/AVIF (and some other containers), so when its decode
    returns None we fall back to Pillow (:func:`_pil_read`): AVIF is native in modern
    Pillow, HEIC works when the optional ``pillow-heif`` plugin is installed. This lets
    the pixel path (visible removal) read the same formats the metadata path already
    scans; normal PNG/JPEG/WebP never reach the fallback, so they are unaffected.
    """
    import cv2
    import numpy as np

    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    # cv2.imdecode returns None on an undecodable container (HEIC/AVIF); the type stub
    # omits that, hence the ignore.
    if img is not None:  # pyright: ignore[reportUnnecessaryComparison]
        return img
    return _pil_read(path, flags)


_heif_registered = False


def _pil_read(path: str | Path, flags: int) -> NDArray[Any] | None:
    """Decode via Pillow (HEIC/AVIF and any other Pillow-readable container) into the
    cv2 layout ``flags`` implies: grayscale, 3-channel BGR, or BGRA when the source has
    alpha and ``IMREAD_UNCHANGED`` was requested. Returns None if Pillow (with the
    optional HEIF plugin) still cannot open it. No EXIF auto-rotation, matching cv2."""
    import cv2
    import numpy as np

    try:
        from PIL import Image
    except Exception:
        return None
    _register_heif()
    try:
        with Image.open(path) as im:
            im.load()
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(im.convert("L"))
            has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            if flags == cv2.IMREAD_UNCHANGED and has_alpha:
                return cv2.cvtColor(np.asarray(im.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
            return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read ``path`` as a BGR ndarray, raising instead of returning ``None``.

    :func:`imread` keeps cv2's ``None``-on-failure contract because the removal paths
    branch on it. Scripts and tests want the opposite -- fail loudly at the read -- so
    they call this. Each vendor engine used to carry its own verbatim copy.
    """
    image = imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def to_bgr(image: NDArray[Any]) -> NDArray[Any]:
    """Return a 3-channel BGR view of ``image``, promoting grayscale and BGRA.

    The cv2-based engines (sparkle + the text-mark detectors/localizers) assume a
    3-channel BGR array for their channel reductions (``mean(axis=2)``, the top-hat
    glyph extraction). A 2D grayscale or 4-channel BGRA input -- a real Gemini-app
    export is opaque RGBA -- would otherwise crash or mis-broadcast.
    Centralizes the shape coercion that was inlined across the engines. A 3-channel
    input is returned unchanged (no copy).
    """
    import cv2

    if image.ndim == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _register_heif() -> None:
    """Register the HEIF+AVIF Pillow opener/saver via libheif (idempotent, best-effort)."""
    global _heif_registered
    if _heif_registered:
        return
    _heif_registered = True
    import contextlib

    with contextlib.suppress(Exception):
        import pillow_heif  # pyright: ignore[reportMissingImports]

        # Documented public API; pillow-heif does not mark its typed re-export.
        pillow_heif.register_heif_opener()  # pyright: ignore[reportPrivateImportUsage]


# ── Display-tag carry across re-encodes (issue #98) ─────────────────────────
#
# cv2's encoders write no container metadata, so every pixel-path output used to
# come back without the source's ICC colour profile and EXIF orientation: a
# Display P3 portrait lost a chunk of its saturation and displayed sideways even
# though the user never asked the tool to touch metadata. Those two tags are
# display fidelity, not provenance, so they travel from the decode source to the
# re-encoded output. Only those two: wholesale EXIF copying would resurrect the
# AI-provenance tags the metadata strip exists to drop.

_ORIENT_EXIF_TAG = 0x0112  # EXIF ImageIFD.Orientation
# Orientations that transpose width and height (the 90-degree rotations). Only these
# can be detected by comparing the raster's shape with the source's stored size;
# the mirror/180 values (2, 3, 4) keep today's tagless behaviour because no cheap
# signal distinguishes a rotated decode from a stored one for them.
_TRANSPOSING_ORIENTATIONS = frozenset({5, 6, 7, 8})


def _orientation_exif(orient: int | None, *, prefixed: bool) -> bytes | None:
    """EXIF bytes holding ONLY the orientation tag, in the form a container wants.

    ``piexif.dump`` returns the JPEG APP1 payload (``Exif\\x00\\x00`` + TIFF); the
    PNG ``eXIf`` chunk and the WebP ``EXIF`` chunk carry the bare TIFF structure, so
    the prefix is stripped there. Orientation 1 (upright) is the absence of a tag.
    """
    if orient is None or orient == 1:
        return None
    import piexif

    dumped = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orient}})
    if prefixed:
        return dumped
    return dumped[6:] if dumped.startswith(b"Exif\x00\x00") else dumped


def _read_display_tags(
    source: Path, raster_shape: tuple[int, int], *, orientation_applied: bool | None = None
) -> tuple[bytes | None, int | None]:
    """Extract ``(icc_profile, orientation)`` from ``source`` for a raster of ``raster_shape``.

    The contract is that ``source`` is the file whose decode produced the pixels
    being written, and ``raster_shape`` is that raster's ``(rows, cols)``. Whether the
    decode turned the raster upright cannot be assumed from the container: cv2
    applies EXIF orientation for JPEG and (build-dependent) PNG -- but only under
    ``IMREAD_COLOR``, never ``IMREAD_UNCHANGED``, which is what the pixel paths read
    with -- and the reporter of issue #98 ran a build that left PNG flat too.
    ``orientation_applied`` carries that decode state explicitly when known.
    Otherwise the legacy inference uses raster geometry: for the
    transposing orientations (5-8) the tag is carried exactly when the raster still
    has the source's STORED dimensions, and dropped when it has the upright ones;
    anything else (a resized pipeline output, the rare mirror/180 values) drops the
    tag and keeps today's behaviour rather than risk a double rotation. ICC travels
    in every case: it describes colour, not geometry. Failures read as "no tags":
    the restore is an enhancement, never a reason to fail a write.
    """
    _register_heif()
    try:
        from PIL import Image
    except Exception:
        return None, None
    try:
        with Image.open(source) as im:
            icc = im.info.get("icc_profile")
            orient = im.getexif().get(_ORIENT_EXIF_TAG)
            stored_w, stored_h = im.size
    except Exception:
        return None, None
    if not (isinstance(icc, bytes) and icc):
        icc = None
    if orientation_applied is True:
        orient = None
    elif orientation_applied is False:
        orient = orient if isinstance(orient, int) and 1 <= orient <= 8 else None
    elif orient in _TRANSPOSING_ORIENTATIONS:
        rows, cols = raster_shape
        # Upright dimensions are the stored ones swapped; only a raster still in
        # stored orientation may carry the tag.
        if (cols, rows) != (stored_w, stored_h):
            orient = None
    else:
        orient = None
    return icc, orient


def _png_splice_display_tags(data: bytes, icc: bytes | None, orient: int | None) -> bytes:
    """Insert ``iCCP`` / ``eXIf`` chunks after IHDR, leaving IDAT verbatim.

    Replaces same-kind chunks rather than duplicating them, so rewriting an
    already-tagged PNG cannot stack profiles. Returns the input unchanged when the
    chunk list does not walk cleanly (the file still decodes exactly as written).
    """
    import struct
    import zlib

    def _chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    iccp = _chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc, 9)) if icc else None
    exif_payload = _orientation_exif(orient, prefixed=False)
    exif_chunk = _chunk(b"eXIf", exif_payload) if exif_payload else None
    if iccp is None and exif_chunk is None:
        return data

    out = bytearray(data[:8])
    pos, n = 8, len(data)
    inserted = False
    while pos + 8 <= n:
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > n:
            return data  # malformed length: leave the encoded bytes alone
        replaced = (kind == b"iCCP" and iccp is not None) or (kind == b"eXIf" and exif_chunk is not None)
        if not replaced:
            out += data[pos:end]
        if kind == b"IHDR" and not inserted:
            out += (iccp or b"") + (exif_chunk or b"")
            inserted = True
        if kind == b"IEND":
            break
        pos = end
    if not inserted:
        return data
    return bytes(out)


def _jpeg_splice_display_tags(data: bytes, icc: bytes | None, orient: int | None) -> bytes:
    """Insert APP1 (EXIF) and APP2 (ICC) markers without touching the DCT scan.

    The new segments go at the end of the existing APP block (after JFIF APP0, ahead
    of DQT/SOF), and same-purpose segments are replaced instead of duplicated. A large
    ICC profile is chunked across APP2 markers per the ICC-in-JPEG spec: one-byte
    sequence and total counters, the layout Pillow's writer and the spec share.
    Returns the input unchanged when the marker walk does not reach the scan cleanly.
    """
    import struct

    exif = _orientation_exif(orient, prefixed=True)
    segments: list[bytes] = []
    if exif:
        segments.append(b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif)
    if icc:
        # 65535 per segment minus the 2-byte length and the 14-byte ICC header
        # ("ICC_PROFILE\x00" + 1-byte sequence + 1-byte total).
        part_size = 65535 - 2 - 14
        parts = [icc[i : i + part_size] for i in range(0, len(icc), part_size)]
        for seq, part in enumerate(parts, start=1):
            payload = b"ICC_PROFILE\x00" + bytes((seq, len(parts))) + part
            segments.append(b"\xff\xe2" + struct.pack(">H", len(payload) + 2) + payload)
    if not segments:
        return data

    out = bytearray(data[:2])  # SOI
    pos, n = 2, len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            return data  # lost the marker boundary: leave the encoded bytes alone
        marker = data[pos + 1]
        if marker not in range(0xE0, 0xF0):
            break  # first segment outside the APP block: insertion point found
        (seg_len,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        end = pos + 2 + seg_len
        if seg_len < 2 or end > n:
            return data
        payload = data[pos + 4 : end]
        replaced = (marker == 0xE1 and exif and payload.startswith(b"Exif\x00\x00")) or (
            marker == 0xE2 and icc and payload.startswith(b"ICC_PROFILE\x00")
        )
        if not replaced:
            out += data[pos:end]
        pos = end
    out += b"".join(segments)
    out += data[pos:]
    return bytes(out)


def _webp_restore_display_tags(path: Path, icc: bytes | None, orient: int | None) -> None:
    """Attach the display tags to a written WebP by re-saving it losslessly.

    A hand-spliced ``ICCP``/``EXIF`` RIFF chunk is invisible to readers: libwebp
    surfaces metadata only when a ``VP8X`` header announces it, and stitching that
    header by hand duplicates what Pillow's writer already does correctly (measured:
    a spliced file fails to open in both Pillow and cv2). Pillow's WebP save in
    lossless mode is pixel-identical, so the container bytes change and the picture
    does not.
    """
    try:
        from PIL import Image
    except Exception:
        return
    exif = _orientation_exif(orient, prefixed=True)
    save_kwargs: dict[str, Any] = {"format": "WEBP", "lossless": True, "exact": True}
    if icc:
        save_kwargs["icc_profile"] = icc
    if exif:
        save_kwargs["exif"] = exif
    try:
        with Image.open(path) as im:
            im.load()
            im.save(path, **save_kwargs)
    except Exception:
        log.debug("webp display-tag restore failed on %s", path, exc_info=True)


def _restore_display_tags(path: Path, icc: bytes | None, orient: int | None) -> None:
    """Best-effort splice of the display tags into an already-written output.

    Never raises and never re-encodes pixels: a failure leaves the un-tagged (but
    valid) file the codec wrote, which is exactly the pre-fix behaviour.
    """
    if icc is None and (orient is None or orient == 1):
        return
    try:
        data = path.read_bytes()
    except OSError:
        log.debug("display-tag restore could not read back %s", path, exc_info=True)
        return
    try:
        if data.startswith(PNG_SIGNATURE):
            path.write_bytes(_png_splice_display_tags(data, icc, orient))
        elif data[:2] == b"\xff\xd8":
            path.write_bytes(_jpeg_splice_display_tags(data, icc, orient))
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            _webp_restore_display_tags(path, icc, orient)
        # else: a container without a restore path (BMP/TIFF): leave as written
    except Exception:
        log.debug("display-tag restore failed on %s", path, exc_info=True)


# Containers cv2 cannot encode -> written via Pillow (pillow-heif).
_HEIF_WRITE_EXTS = {".heic", ".heif", ".avif"}


def _encode_params(ext: str) -> list[int]:
    """cv2 encode params that PRESERVE quality. The removal only touches the mark's
    footprint, so the container re-encode must not degrade the untouched pixels:
    JPEG at quality 100 with 4:4:4 chroma (no subsampling), WebP at max. Lossless
    containers (PNG/BMP/TIFF) need no params. getattr-guarded so an older OpenCV
    build without the chroma/subsampling flags still gets quality 100."""
    import cv2

    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        cq = getattr(cv2, "IMWRITE_JPEG_CHROMA_QUALITY", None)
        if cq is not None:
            params += [cq, 100]
        sf = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        sf444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
        if sf is not None and sf444 is not None:
            params += [sf, sf444]
        return params
    if ext == ".webp":
        # cv2 WebP: quality 1-100 is LOSSY; a value > 100 selects LOSSLESS mode.
        # "work with originals" requires lossless so a mark-removal re-encode does not
        # degrade the untouched pixels the fill composites over (regression: q100
        # round-tripped a random image at maxdiff ~230, q101 at 0).
        return [cv2.IMWRITE_WEBP_QUALITY, 101]
    return []


def _pil_write(
    path: str | Path,
    img: NDArray[Any],
    *,
    icc_profile: bytes | None = None,
    orientation: int | None = None,
) -> bool:
    """Encode HEIC/AVIF via Pillow (+pillow-heif) at high quality -- cv2 has no encoder
    for them. BGR / BGRA in; returns False if Pillow (with the plugin) cannot save.
    The display tags, when given, ride along in the save call: Pillow's HEIF encoder
    writes both natively, so this branch needs no post-write splice."""
    import cv2
    import numpy as np
    from PIL import Image

    _register_heif()
    if img.ndim == 3 and img.shape[2] == 4:
        arr, mode = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), "RGBA"
    else:
        arr, mode = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB), "RGB"
    save_kwargs: dict[str, Any] = {"quality": 100}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    exif = _orientation_exif(orientation, prefixed=True)
    if exif:
        save_kwargs["exif"] = exif
    try:
        Image.fromarray(np.ascontiguousarray(arr), mode).save(str(path), **save_kwargs)
        return True
    except Exception:
        return False


def imwrite(
    path: str | Path,
    img: NDArray[Any],
    *,
    display_tags_from: str | Path | None = None,
    orientation_applied: bool | None = None,
) -> bool:
    """Unicode-safe image write that PRESERVES the input format at max quality.

    Format is taken from the path extension. HEIC/AVIF (which cv2 cannot encode) go
    through Pillow; everything else through cv2 with quality-preserving params (see
    :func:`_encode_params`) so a lossy re-encode of the untouched pixels stays near-
    lossless. Returns ``True`` on success, ``False`` if the codec rejects the image or
    the path cannot be written (matching ``cv2.imwrite``, never raising).

    When ``display_tags_from`` names the file whose decode produced ``img``'s pixels,
    the source's ICC profile and EXIF orientation are carried into the re-encoded
    output (issue #98): cv2's encoders write no container metadata, and colour and
    geometry are display fidelity, not provenance. The tags are read BEFORE the write
    so an in-place rewrite (source == output) still finds them.
    Set ``orientation_applied=False`` for an unchanged/raw decode, or ``True``
    when EXIF has already been applied, including mirrors and square images.
    Without that explicit state, orientation is re-tagged only when the raster
    still has the source's stored dimensions, using the legacy heuristic in
    :func:`_read_display_tags`. Pass the known decode state to avoid ambiguous
    geometry, especially square images and mirrors."""
    import cv2

    icc, orient = (
        _read_display_tags(
            Path(display_tags_from), (img.shape[0], img.shape[1]), orientation_applied=orientation_applied
        )
        if display_tags_from is not None
        else (None, None)
    )
    ext = (Path(path).suffix or ".png").lower()
    if ext in _HEIF_WRITE_EXTS:
        return _pil_write(path, img, icc_profile=icc, orientation=orient)
    try:
        ok, buf = cv2.imencode(ext, img, _encode_params(ext))
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    _restore_display_tags(Path(path), icc, orient)
    return True


# Container extensions that carry an alpha channel (for read/write-with-alpha).
ALPHA_FORMATS = {".png", ".webp", ".heic", ".heif", ".avif"}


def read_bgr_and_alpha(path: str | Path) -> tuple[NDArray[Any] | None, NDArray[Any] | None]:
    """Read an image preserving its alpha channel separately.

    Returns ``(bgr, alpha)`` where ``alpha`` is a single-channel ndarray when the
    source has transparency, else ``None``. Grayscale inputs are promoted to BGR.
    Returns ``(None, None)`` if the image cannot be decoded.
    """
    import cv2

    image = imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()
    return image, None


def write_bgr_with_alpha(
    path: str | Path,
    bgr: NDArray[Any],
    alpha: NDArray[Any] | None,
    *,
    display_tags_from: str | Path | None = None,
    orientation_applied: bool | None = None,
) -> bool:
    """Write BGR (with optional alpha) to ``path``. Returns ``imwrite``'s success flag.

    When ``alpha`` is provided and the output extension supports it, the original
    alpha plane is rejoined unchanged. The watermark region is NOT made transparent:
    the fill reconstructs real pixels there, so zeroing alpha would punch a
    transparent hole that renders as a white box on any non-transparent viewer
    (issue #30). Preserving the input alpha keeps genuinely transparent backgrounds
    intact without inventing new holes.

    Returning the flag is load-bearing: :func:`imwrite` is contractually non-raising, so
    this is the ONLY signal a caller gets that the file was not created. Discarding it let
    a failed write (read-only directory, full disk) run on to ``output.stat()`` and die
    with a bare ``FileNotFoundError`` traceback instead of a readable error.
    Regression: ``tests/test_cli_robustness.py::TestFailedWriteIsReported``.
    """
    import numpy as np

    if alpha is None or Path(path).suffix.lower() not in ALPHA_FORMATS:
        return imwrite(path, bgr, display_tags_from=display_tags_from, orientation_applied=orientation_applied)
    return imwrite(
        path, np.dstack([bgr, alpha]), display_tags_from=display_tags_from, orientation_applied=orientation_applied
    )
