"""C2PA (Coalition for Content Provenance and Authenticity) metadata handling.

C2PA metadata is embedded in PNG files as a JUMBF container chunk
(``caBX``).  This module can detect, extract, and re-inject those
chunks.  Supported issuers:

- Google Imagen
- Adobe Firefly
- Microsoft Designer
- OpenAI (ChatGPT, GPT-4o, Sora, DALL-E)
- Truepic (signing authority)

The parser uses byte-level scanning — it does not validate JUMBF/CBOR
structure but reliably identifies known signatures, issuers, tools,
and actions.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

from remove_ai_watermarks.noai.constants import (
    C2PA_ACTIONS,
    C2PA_AI_TOOLS,
    C2PA_CHUNK_TYPE,
    C2PA_ISSUERS,
    C2PA_SIGNATURES,
    PNG_SIGNATURE,
)


def has_c2pa_metadata(image_path: Path) -> bool:
    """
    Check if an image contains C2PA metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        True if C2PA metadata is detected, False otherwise.
    """
    image_path = Path(image_path)

    if image_path.suffix.lower() != ".png":
        return False

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return False

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(length)
                    # Check for any C2PA signature
                    for sig in C2PA_SIGNATURES:
                        if sig in chunk_data:
                            return True
                    # Also check if chunk_data itself contains C2PA-like patterns
                    if b"jumb" in chunk_data.lower() or b"c2pa" in chunk_data.lower():
                        return True
                    f.read(4)
                else:
                    f.read(length + 4)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return False


def extract_c2pa_info(image_path: Path) -> dict[str, Any]:
    """
    Extract basic C2PA metadata information from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary containing C2PA metadata info.
    """
    c2pa_info: dict[str, Any] = {}

    if not has_c2pa_metadata(image_path):
        return c2pa_info

    c2pa_info["has_c2pa"] = True
    c2pa_info["type"] = "C2PA (Coalition for Content Provenance and Authenticity)"

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return c2pa_info

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(length)
                    _parse_c2pa_chunk(chunk_data, c2pa_info)
                    f.read(4)
                else:
                    f.read(length + 4)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return c2pa_info


def _parse_c2pa_chunk(chunk_data: bytes, c2pa_info: dict[str, Any]) -> None:
    """Parse C2PA chunk data and populate info dictionary."""
    # Find issuers
    issuers = []
    for sig, name in C2PA_ISSUERS.items():
        if sig in chunk_data:
            issuers.append(name)
    if issuers:
        c2pa_info["issuer"] = ", ".join(set(issuers))

    # Find AI tools
    ai_tools = []
    for sig, name in C2PA_AI_TOOLS.items():
        if sig in chunk_data:
            ai_tools.append(name)
    if ai_tools:
        c2pa_info["ai_tool"] = ", ".join(set(ai_tools))

    # Extract software agent (multiple patterns)
    patterns = [
        rb"softwareAgent.*?dname([^\x00]+?)(?:q|l|m|n)",
        rb"software_agent[^\x00]*?([A-Za-z0-9_\-\.]+)",
        rb"Software[^\x00]*?([A-Za-z0-9_\-\. ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, chunk_data, re.DOTALL | re.IGNORECASE)
        if match:
            agent = match.group(1).decode("utf-8", errors="ignore").strip()
            if agent and len(agent) < 100:
                c2pa_info["software_agent"] = agent
                break

    # Extract claim generator (multiple patterns)
    claim_patterns = [
        rb"claim_generator[^\x00]*?([A-Za-z0-9_\-\.\/\:]+)",
        rb"claimGenerator[^\x00]*?([A-Za-z0-9_\-\.\/\:]+)",
        rb"dname([^\x00]{3,50})(?:q|l|m|n|i)",
    ]
    for pattern in claim_patterns:
        match = re.search(pattern, chunk_data, re.DOTALL | re.IGNORECASE)
        if match:
            gen_name = match.group(1).decode("utf-8", errors="ignore").strip()
            # Filter out common false positives
            if gen_name and len(gen_name) < 100 and not gen_name.startswith(("\\x", "\\\\x")):
                c2pa_info["claim_generator"] = gen_name
                break

    # Find actions
    actions = []
    for sig, name in C2PA_ACTIONS.items():
        if sig in chunk_data:
            actions.append(name)
    if actions:
        c2pa_info["actions"] = ", ".join(actions)

    # Find timestamps
    timestamp_matches = re.findall(rb"(\d{14}Z)", chunk_data)
    if timestamp_matches:
        c2pa_info["timestamp"] = timestamp_matches[0].decode("utf-8")
        if len(timestamp_matches) > 1:
            c2pa_info["timestamps"] = [t.decode("utf-8") for t in timestamp_matches[:3]]

    # Find digital source type
    if b"trainedAlgorithmicMedia" in chunk_data:
        c2pa_info["source_type"] = "trainedAlgorithmicMedia (AI-generated)"
    elif b"algorithmicMedia" in chunk_data:
        c2pa_info["source_type"] = "algorithmicMedia"
    elif b"compositeWithTrainedAlgorithmicMedia" in chunk_data:
        c2pa_info["source_type"] = "compositeWithTrainedAlgorithmicMedia (AI-enhanced)"


def extract_c2pa_chunk(image_path: Path) -> bytes | None:
    """
    Extract the raw C2PA JUMBF chunk from a PNG file.

    Args:
        image_path: Path to the source PNG file.

    Returns:
        Raw bytes of the C2PA chunk or None.
    """
    if image_path.suffix.lower() != ".png":
        return None

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return None

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(length)
                    crc = f.read(4)

                    # Check for any C2PA signature
                    for sig in C2PA_SIGNATURES:
                        if sig in chunk_data:
                            return chunk_header + chunk_data + crc

                    # Also check lowercase variants
                    if b"jumb" in chunk_data.lower() or b"c2pa" in chunk_data.lower():
                        return chunk_header + chunk_data + crc
                else:
                    f.read(length + 4)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return None


def inject_c2pa_chunk(target_path: Path, output_path: Path, c2pa_chunk: bytes) -> None:
    """
    Inject a C2PA JUMBF chunk into a PNG file.

    Args:
        target_path: Path to the target PNG file.
        output_path: Path where the output file will be saved.
        c2pa_chunk: Raw bytes of the C2PA chunk to inject.

    Raises:
        ValueError: If not PNG files.
    """
    if target_path.suffix.lower() != ".png" or output_path.suffix.lower() != ".png":
        raise ValueError("C2PA chunk injection is only supported for PNG files")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            f_out.write(f_in.read(8))

            c2pa_injected = False
            while True:
                chunk_header = f_in.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]
                chunk_data = f_in.read(length)
                crc = f_in.read(4)

                if chunk_type == b"IDAT" and not c2pa_injected:
                    f_out.write(c2pa_chunk)
                    c2pa_injected = True

                if chunk_type == C2PA_CHUNK_TYPE:
                    continue

                f_out.write(chunk_header)
                f_out.write(chunk_data)
                f_out.write(crc)

                if chunk_type == b"IEND":
                    break
