#!/usr/bin/env python3
"""Development facade for the shared hash-pinned public Perth v1 model."""

# ruff: noqa: E402 - Bootstrap the worktree source before importing the runtime.
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks._internal.perth import (
    CHECKPOINT_RELATIVE_PATH as CHECKPOINT_RELATIVE_PATH,
)
from remove_ai_watermarks._internal.perth import (
    CHECKPOINT_SHA256 as CHECKPOINT_SHA256,
)
from remove_ai_watermarks._internal.perth import (
    PACKAGE_NAME as PACKAGE_NAME,
)
from remove_ai_watermarks._internal.perth import (
    PACKAGE_VERSION as PACKAGE_VERSION,
)
from remove_ai_watermarks._internal.perth import (
    SAMPLE_RATE as SAMPLE_RATE,
)
from remove_ai_watermarks._internal.perth import (
    SOURCE_TREE_SHA256 as SOURCE_TREE_SHA256,
)
from remove_ai_watermarks._internal.perth import (
    PerthReading as PerthReading,
)
from remove_ai_watermarks._internal.perth import (
    available as available,
)
from remove_ai_watermarks._internal.perth import (
    checkpoint_path as checkpoint_path,
)
from remove_ai_watermarks._internal.perth import (
    embed as embed,
)
from remove_ai_watermarks._internal.perth import (
    import_failure as import_failure,
)
from remove_ai_watermarks._internal.perth import (
    load_model as load_model,
)
from remove_ai_watermarks._internal.perth import (
    read as read,
)
