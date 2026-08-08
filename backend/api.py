"""FastAPI surface over the NLU pipeline.

Run with:
    python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000

Endpoints
    GET    /health
    POST   /api/predict              stateless single-message NLU
    POST   /api/session              create a session
    GET    /api/session/{id}         session history
    DELETE /api/session/{id}         forget a session
    POST   /api/chat                 context-aware turn
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import __version__
from backend import config as cfg
from backend.pipeline import NLUPipeline, get_pipeline
from backend.response import ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# --- schemas -----------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Hinglish sentence to analyse")
    verbose: bool = Field(False, description="Include top-k intents and the sentiment distribution")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(None, description="Omit to start a new session")


class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Supply to use a specific id")


class SlotSpan(BaseModel):
    type: str
    value: str
    confidence: float
    value_normalized: Optional[str] = None


class NLUBlock(BaseModel):
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_confidence: float
    slots: List[SlotSpan]


# --- app ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the 2.2 GB checkpoint once at startup so the first request is fast
    # and a broken checkpoint fails the boot rather than a user's request.
    logger.info("Loading NLU pipeline...")
    app.state.pipeline = get_pipeline()
    logger.info("Pipeline ready: %s", app.state.pipeline.info()["model"])
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Hinglish Joint NLU API",
    version=__version__,
    description=(
        "Intent, sentiment and slot extraction from a fine-tuned XLM-RoBERTa Large "
        "joint model, with a conversation-context layer and a rule-based priority "
        "routing layer."
    ),
    lifespan=lifespan,
)


# Browser clients (the Vite dev server) are a different origin from the API, so
# the allowed origins have to be declared explicitly. No ML logic is involved.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# Stateless and cheap - no model, no network. One shared instance is enough.
_responder = ResponseGenerator()


def _pipeline() -> NLUPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:  # e.g. when routes are exercised without lifespan
        pipeline = get_pipeline()
        app.state.pipeline = pipeline
    return pipeline


@app.get("/health", tags=["ops"])
def health() -> Dict[str, Any]:
    """Liveness plus what is actually loaded."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return {"status": "loading", "model_loaded": False}
    return {"status": "ok", "model_loaded": True, "version": __version__, **pipeline.info()}


@app.post("/api/predict", tags=["nlu"])
def predict(request: PredictRequest) -> Dict[str, Any]:
    """Single-query NLU with no conversation context."""
    try:
        return _pipeline().predict(request.text, verbose=request.verbose)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/chat", tags=["chat"])
def chat(request: ChatRequest) -> Dict[str, Any]:
    """Context-aware turn. Creates the session when `session_id` is omitted.

    Returns the pipeline result unchanged, plus the generated reply
    (`response`, `response_type`, `requires_human`).
    """
    pipeline = _pipeline()
    try:
        result = pipeline.process_message(request.session_id, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    reply = _responder.generate(result)
    # Record the reply against the turn the pipeline just stored, so session
    # history shows both sides of the conversation.
    pipeline.context.set_bot_response(result["session_id"], reply["response"])

    return {**result, **reply}


@app.post("/api/session", status_code=201, tags=["session"])
def create_session(request: Optional[SessionRequest] = None) -> Dict[str, Any]:
    """Open a conversation session."""
    session_id = _pipeline().create_session(request.session_id if request else None)
    return {"session_id": session_id, "created": True}


@app.get("/api/session/{session_id}", tags=["session"])
def get_session(session_id: str) -> Dict[str, Any]:
    """Retained history for a session."""
    info = _pipeline().get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return info


@app.delete("/api/session/{session_id}", tags=["session"])
def delete_session(session_id: str) -> Dict[str, Any]:
    """Forget a session and its history."""
    if not _pipeline().clear_session(session_id):
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return {"session_id": session_id, "cleared": True}
