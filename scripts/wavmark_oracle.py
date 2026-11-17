#!/usr/bin/env python3
"""Development import interface for the shared pinned public WavMark model."""

# ruff: noqa: E402 - Bootstrap the source-layout package before explicit re-exports.

from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from remove_ai_watermarks._internal.wavmark import (
    MESSAGE as MESSAGE,
)
from remove_ai_watermarks._internal.wavmark import (
    MIN_EMBED_SAMPLES as MIN_EMBED_SAMPLES,
)
from remove_ai_watermarks._internal.wavmark import (
    MIN_READ_SAMPLES as MIN_READ_SAMPLES,
)
from remove_ai_watermarks._internal.wavmark import (
    MODEL_FILENAME as MODEL_FILENAME,
)
from remove_ai_watermarks._internal.wavmark import (
    MODEL_REPOSITORY as MODEL_REPOSITORY,
)
from remove_ai_watermarks._internal.wavmark import (
    MODEL_REVISION as MODEL_REVISION,
)
from remove_ai_watermarks._internal.wavmark import (
    MODEL_SHA256 as MODEL_SHA256,
)
from remove_ai_watermarks._internal.wavmark import (
    SAMPLE_RATE as SAMPLE_RATE,
)
from remove_ai_watermarks._internal.wavmark import (
    SOURCE_COMMIT as SOURCE_COMMIT,
)
from remove_ai_watermarks._internal.wavmark import (
    SOURCE_TREE_SHA256 as SOURCE_TREE_SHA256,
)
from remove_ai_watermarks._internal.wavmark import (
    WavMarkReading as WavMarkReading,
)
from remove_ai_watermarks._internal.wavmark import (
    available as available,
)
from remove_ai_watermarks._internal.wavmark import (
    checkpoint_path as checkpoint_path,
)
from remove_ai_watermarks._internal.wavmark import (
    embed as embed,
)
from remove_ai_watermarks._internal.wavmark import (
    import_failure as import_failure,
)
from remove_ai_watermarks._internal.wavmark import (
    load_model as load_model,
)
from remove_ai_watermarks._internal.wavmark import (
    read as read,
)
