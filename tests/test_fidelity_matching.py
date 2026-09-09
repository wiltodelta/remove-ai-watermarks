"""Regression test for the one-to-one face matcher in ``scripts/fidelity_metrics.py``.

The shipped per-face nearest matcher collided on multi-face images (two original faces
both picking the same variant face when regeneration dropped a face), which inflated/
corrupted the identity metric. ``assign_faces_one_to_one`` is the collision-free
replacement. The function is pure (centers + diagonals in, index map out), so it is
tested here without insightface / the heavy PEP723 env. Caught on the gemini_3 Qwen
ControlNet experiment, where the original had 18 faces but the regenerated variants had
17, producing two collisions under the old matcher.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_module():
    # fidelity_metrics is a standalone PEP723 script, not an installed module; load it by
    # path with scripts/ on sys.path so its `_plain_console` shim import resolves.
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("fidelity_metrics", _SCRIPTS / "fidelity_metrics.py")
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # @dataclass introspection needs the module registered
        spec.loader.exec_module(mod)
    except ImportError as exc:  # cv2/click absent in a bare env -> skip, not fail
        pytest.skip(f"fidelity_metrics import deps missing: {exc}")
    finally:
        sys.path.remove(str(_SCRIPTS))
    return mod


def _load_assign():
    return _load_module().assign_faces_one_to_one


def test_text_ned_remains_case_sensitive() -> None:
    assert _load_module()._text_ned("A", "a") == 1.0


def test_distinct_faces_match_nearest() -> None:
    assign = _load_assign()
    ref = [(0.0, 0.0), (100.0, 100.0)]
    var = [(2.0, 1.0), (98.0, 102.0)]
    diags = [50.0, 50.0]
    assert assign(ref, var, diags) == {0: 0, 1: 1}


def test_no_collision_when_variant_drops_a_face() -> None:
    # Two original faces near the SAME single variant face: the old nearest matcher mapped
    # BOTH to index 0; one-to-one must give the nearer ref the match and drop the other.
    assign = _load_assign()
    ref = [(10.0, 10.0), (14.0, 10.0)]  # both close to the lone variant
    var = [(12.0, 10.0)]
    diags = [50.0, 50.0]
    matched = assign(ref, var, diags)
    assert sorted(matched.values()) == [0]  # variant 0 used at most once
    assert len(matched) == 1


def test_gate_drops_implausibly_far_match() -> None:
    assign = _load_assign()
    ref = [(0.0, 0.0)]
    var = [(1000.0, 1000.0)]  # far beyond 0.6 * diag
    diags = [50.0]
    assert assign(ref, var, diags) == {}


def test_assignment_is_one_to_one_over_many_faces() -> None:
    assign = _load_assign()
    ref = [(float(i * 100), 0.0) for i in range(18)]
    var = [(float(i * 100) + 3.0, 0.0) for i in range(17)]  # one fewer, as in the experiment
    diags = [50.0] * 18
    matched = assign(ref, var, diags)
    assert len(matched) == 17
    assert len(set(matched.values())) == 17  # every variant used at most once
