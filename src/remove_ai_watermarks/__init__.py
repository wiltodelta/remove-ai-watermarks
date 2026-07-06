"""Remove-AI-Watermarks: Unified tool for removing visible and invisible AI watermarks."""

import os as _os
import warnings as _warnings

# transformers prints a noisy deprecation for the Siglip2ImageProcessorFast
# alias when it is imported (by the optional GPU/ML path). Silence it before
# any submodule pulls transformers in, so the CLI startup stays quiet. Uses
# setdefault so a user-set TRANSFORMERS_VERBOSITY still wins.
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_warnings.filterwarnings("ignore", message=r".*ImageProcessorFast.*")


__version__ = "0.13.0"
