"""Shared fixtures.

The checkpoint is ~2.2 GB and takes ~11s to load, so it is built once per test
session and shared. `get_pipeline()` returns a process-wide singleton, which
means the API tests reuse the same loaded model rather than loading a second copy.
"""

from __future__ import annotations

import pytest

from backend.pipeline import get_pipeline


@pytest.fixture(scope="session")
def pipeline():
    """The shared NLU pipeline with the real checkpoint loaded."""
    return get_pipeline()


@pytest.fixture(scope="session")
def nlu(pipeline):
    return pipeline.nlu


@pytest.fixture()
def session_id(pipeline):
    """A fresh session, cleaned up afterwards."""
    sid = pipeline.create_session()
    yield sid
    pipeline.clear_session(sid)
