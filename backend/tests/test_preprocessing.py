"""Preprocessing must stay identical to training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.model.preprocessing import (
    HINGLISH_NORMALIZE,
    align_to_original,
    normalize_hinglish,
    tokenize,
)

NOTEBOOK = Path(__file__).resolve().parents[2] / "Another_copy_of_Maybe_final.ipynb"


def test_normalization_table_matches_the_notebook():
    """Guard against anyone 'improving' the regex table.

    Reads the notebook source directly, so this fails if the two ever diverge.
    """
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = "\n".join("".join(cell["source"]) for cell in cells)

    for pattern, replacement in HINGLISH_NORMALIZE.items():
        assert pattern in source, f"pattern {pattern!r} is not in the notebook"
        assert f'"{replacement}"' in source, f"replacement {replacement!r} is not in the notebook"

    assert len(HINGLISH_NORMALIZE) == 12


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Mera order NHI aaya", "mera order nahi aaya"),
        ("PLZ refund KR do", "please refund kar do"),
        ("200 RS zyada", "200 rupaye zyada"),
        ("  extra   spaces  ", "extra spaces"),
        ("Payment ho gya", "payment ho gaya"),
        ("THX for help", "thanks for help"),
    ],
)
def test_normalize_hinglish(raw, expected):
    assert normalize_hinglish(raw) == expected


@pytest.mark.parametrize("bad", [None, 42, [], {}])
def test_normalize_hinglish_type_guard(bad):
    assert normalize_hinglish(bad) == ""


@pytest.mark.parametrize(
    "raw",
    [
        "Mera order #ABC123 5 din se deliver nahi hua",
        "PLZ refund KR do 200 RS",
        "Payment ho gya but order NHI aaya",
        "Order #45231 abhi tak nahi aaya",
        "Bahut hi WORST experience raha",
    ],
)
def test_word_count_is_preserved_by_normalization(raw):
    """Every substitution is word-count preserving, which is what makes the
    index-wise mapping back to the original surface form valid."""
    assert len(raw.split()) == len(tokenize(raw))


def test_align_to_original_recovers_casing():
    raw = "Mera order #ABC123 nahi aaya"
    tokens = tokenize(raw)
    surface, aligned = align_to_original(raw, tokens)

    assert aligned is True
    assert surface[2] == "#ABC123"      # original casing preserved
    assert tokens[2] == "#abc123"       # normalized form is lowercased


def test_align_to_original_falls_back_when_counts_disagree():
    tokens = ["a", "b"]
    surface, aligned = align_to_original("a b c", tokens)

    assert aligned is False
    assert surface == tokens
