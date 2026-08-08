"""Text preprocessing - lifted verbatim from the training notebook.

`HINGLISH_NORMALIZE` and `normalize_hinglish` below are character-for-character
identical to Section 3 of `Another_copy_of_Maybe_final.ipynb`. The model was
trained on the output of exactly this function, so any change here silently
degrades accuracy at inference time. Do not "improve" it.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# --- VERBATIM FROM THE TRAINING NOTEBOOK - DO NOT EDIT -----------------------
HINGLISH_NORMALIZE = {
    r"\bnhi\b":    "nahi",
    r"\bnai\b":    "nahi",
    r"\bkaroon\b": "karu",
    r"\bkarna\b":  "karo",
    r"\bho gya\b": "ho gaya",
    r"\bkr\b":     "kar",
    r"\bpls\b":    "please",
    r"\bplz\b":    "please",
    r"\bthx\b":    "thanks",
    r"\bty\b":     "thanks",
    r"\brupee\b":  "rupaye",
    r"\brs\b":     "rupaye",
}


def normalize_hinglish(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    for pattern, replacement in HINGLISH_NORMALIZE.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text)
    return text
# --- END VERBATIM ------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Whitespace tokenization, matching the notebook's `text_clean.split()`."""
    return normalize_hinglish(text).split()


def align_to_original(raw_text: str, normalized_tokens: Sequence[str]) -> Tuple[List[str], bool]:
    """Best-effort recovery of the raw surface form for each normalized token.

    `normalize_hinglish` lowercases, so a slot value extracted from the
    normalized text reads ``#abc123`` even when the user typed ``#ABC123``.
    Every substitution in `HINGLISH_NORMALIZE` is word-count preserving
    (``ho gya`` -> ``ho gaya`` is 2 -> 2, the rest are 1 -> 1), and neither
    ``.lower()`` nor the whitespace squash changes the token count. So when the
    raw text splits into the same number of tokens, index i in the normalized
    sequence corresponds to index i in the raw sequence.

    Returns ``(tokens, aligned)``. When the counts disagree - which should not
    happen for the substitutions above, but is not worth crashing over - the
    normalized tokens are returned with ``aligned=False`` and callers fall back
    to lowercased slot values.
    """
    original_tokens = raw_text.split()
    if len(original_tokens) == len(normalized_tokens):
        return original_tokens, True
    return list(normalized_tokens), False
