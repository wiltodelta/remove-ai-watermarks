"""Research-only periodic lattice detector, not a package export."""

from synthid_runtime.synthid_detector import SynthIDDetection, detect_synthid, is_available

__all__ = ["SynthIDDetection", "detect_synthid", "is_available"]
