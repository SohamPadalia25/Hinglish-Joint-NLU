"""Filesystem paths and runtime settings for the backend.

Every path can be overridden with an environment variable, which keeps the
service deployable without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/config.py -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


# --- Trained artefacts -------------------------------------------------------
# best_model.pt and hinglish_joint_model1.pt were verified bit-identical
# (all 397 tensors compared with torch.equal). best_model.pt is the canonical one.
CHECKPOINT_PATH = _path_from_env("HINGLISH_CHECKPOINT", PROJECT_ROOT / "best_model.pt")
METADATA_PATH = _path_from_env("HINGLISH_METADATA", PROJECT_ROOT / "model_metadata2.json")

# --- Encoder config + tokenizer ----------------------------------------------
# Local copies of xlm-roberta-large's config.json and tokenizer.json ONLY.
# The pretrained encoder *weights* are deliberately not stored here: every
# encoder tensor is overwritten by the fine-tuned checkpoint on load.
ASSETS_DIR = _path_from_env("HINGLISH_ASSETS", PROJECT_ROOT / "assets")
TOKENIZER_DIR = _path_from_env("HINGLISH_TOKENIZER_DIR", ASSETS_DIR / "xlm-roberta-large")

# Fallback used only when TOKENIZER_DIR has not been populated yet.
HF_MODEL_ID = "xlm-roberta-large"

# --- Runtime -----------------------------------------------------------------
# None => pick cuda when available, else cpu.
DEVICE = os.environ.get("HINGLISH_DEVICE") or None

# Pad every request to max_len, exactly as the training notebook did.
# Attention masking makes dynamic padding numerically equivalent, but the
# faithful setting is the default so inference cannot drift from training.
PAD_TO_MAX_LENGTH = os.environ.get("HINGLISH_PAD_TO_MAX_LENGTH", "1") not in ("0", "false", "False")

# --- Conversation memory -----------------------------------------------------
MAX_CONTEXT_TURNS = int(os.environ.get("HINGLISH_MAX_CONTEXT_TURNS", "5"))

# --- CORS --------------------------------------------------------------------
# The browser blocks the Vite dev server from calling this API unless the origin
# is allowed. Both hostnames are listed because Vite prints "localhost" while
# some setups resolve to 127.0.0.1, and the two are distinct origins.
# Override with a comma-separated list for other deployments.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "HINGLISH_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]
