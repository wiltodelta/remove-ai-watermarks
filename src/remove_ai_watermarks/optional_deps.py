"""Shared availability guard for optional dependencies.

A bare ``importlib.util.find_spec(name) is not None`` check lies when only a
leftover data directory exists in site-packages: e.g. ``trustmark`` downloads
model weights into its own package dir, so after the package is uninstalled
(``uv sync`` pruning an extra) a ``trustmark/models/`` remnant survives and
``find_spec`` resolves it to a namespace-package spec (``loader is None``)
while the actual import fails. Every ``is_available()`` guard routes through
``module_available`` so a pure namespace package counts as absent.
"""

from __future__ import annotations

import importlib.util


def module_available(*names: str) -> bool:
    """True when every named module resolves to a real, importable package.

    A spec with ``loader is None`` is a pure namespace package -- for our
    optional deps that means a stale directory remnant, not an installed
    package -- so it is treated as not available.
    """
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return False
    return True


def pixels_available() -> bool:
    """True when this build can decode and inpaint pixels.

    The default package ships without the ``visible`` extra -- the Homebrew formula
    installs exactly that build -- so every visible-mark command and the visible arm
    of ``identify`` has to ask this. It is one predicate rather than three, because
    the three places that needed it each grew their own ``exc.name in (...)`` test
    and all three then agreed only by coincidence: an ``ImportError`` raised INSIDE
    ``cv2/__init__.py`` carries ``name == "cv2.cv2"`` and a numpy ABI mismatch
    carries ``name is None``, so a name match answered "that is some other import
    error" and let the traceback through.
    """
    from remove_ai_watermarks._internal.watermark_profiles import PIXELS_MODULES

    return module_available(*PIXELS_MODULES)
