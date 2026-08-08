"""Master pipeline: message in, structured decision out.

    message
       -> normalize / preprocess          (backend.model.preprocessing)
       -> XLM-R Large joint model         (backend.model.inference)
       -> intent + sentiment + slots
       -> conversation memory             (backend.context.context_manager)
       -> context interpretation          (backend.context.context_interpreter)
       -> priority routing layer          (backend.priority.priority_engine)
       -> structured result

Ordering detail that matters: history is read *before* the current turn is
appended, so a message is never treated as its own context.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from backend import config as cfg
from backend.context.context_interpreter import ContextInterpreter
from backend.context.context_manager import ConversationContext
from backend.model.inference import HinglishNLU
from backend.model.preprocessing import normalize_hinglish
from backend.priority.priority_engine import PriorityEngine

logger = logging.getLogger(__name__)


class NLUPipeline:
    """Wires the NLU model, conversation memory and priority layer together."""

    def __init__(
        self,
        nlu: Optional[HinglishNLU] = None,
        context: Optional[ConversationContext] = None,
        interpreter: Optional[ContextInterpreter] = None,
        priority_engine: Optional[PriorityEngine] = None,
    ):
        self.nlu = nlu or HinglishNLU()
        self.context = context or ConversationContext(max_turns=cfg.MAX_CONTEXT_TURNS)
        self.interpreter = interpreter or ContextInterpreter()
        self.priority_engine = priority_engine or PriorityEngine()

    # ------------------------------------------------------------- main entry

    def process_message(self, session_id: Optional[str], message: str) -> Dict[str, Any]:
        """Full context-aware pass over one message in a conversation."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        session_id = self.context.create_session(session_id)

        # 1. NLU on the current message only - the model never sees history.
        nlu_result = self.nlu.predict(message)

        # 2. History as it stood *before* this message.
        relevant_context = self.context.get_relevant_context(session_id)

        # 3. Is this a follow-up, and what carries over?
        interpretation = self.interpreter.interpret(message, nlu_result, relevant_context)

        # 4. Priority and routing.
        effective_slots: List[Dict[str, Any]] = list(nlu_result["slots"])
        if interpretation["context_used"]:
            effective_slots += interpretation["inherited_slots"]

        priority = self.priority_engine.predict_priority(
            intent=nlu_result["intent"],
            sentiment=nlu_result["sentiment"],
            slots=effective_slots,
            context=relevant_context,
        )

        # 5. Remember this turn for the next one.
        self.context.add_turn(
            session_id=session_id,
            user_message=message,
            nlu=nlu_result,
            priority=priority["priority"],
            normalized_message=normalize_hinglish(message),
        )

        return {
            "session_id": session_id,
            "message": message,
            "nlu": {
                "intent": nlu_result["intent"],
                "intent_confidence": nlu_result["intent_confidence"],
                "sentiment": nlu_result["sentiment"],
                "sentiment_confidence": nlu_result["sentiment_confidence"],
                "slots": nlu_result["slots"],
            },
            "context": {
                "used": interpretation["context_used"],
                "reason": interpretation["context_reason"],
                "previous_intent": interpretation["previous_intent"],
                "previous_sentiment": interpretation["previous_sentiment"],
                "inherited_slots": interpretation["inherited_slots"],
                "signals": interpretation["signals"],
                "relevant_history": interpretation["relevant_history"],
            },
            "priority": {
                "label": priority["priority"],
                "score": priority["priority_score"],
                "routing": priority["routing"],
                "reason": priority["reason"],
                "rules_fired": priority["rules_fired"],
            },
        }

    # ------------------------------------------------------------- stateless

    def predict(self, message: str, verbose: bool = False) -> Dict[str, Any]:
        """Single-shot NLU with no session and no conversation memory."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        nlu_result = (
            self.nlu.predict_verbose(message) if verbose else self.nlu.predict(message)
        )
        priority = self.priority_engine.predict_priority(
            intent=nlu_result["intent"],
            sentiment=nlu_result["sentiment"],
            slots=nlu_result["slots"],
            context=None,
        )
        return {
            "message": message,
            "nlu": nlu_result,
            "priority": {
                "label": priority["priority"],
                "score": priority["priority_score"],
                "routing": priority["routing"],
                "reason": priority["reason"],
            },
        }

    # ------------------------------------------------------- session passthru

    def create_session(self, session_id: Optional[str] = None) -> str:
        return self.context.create_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.context.session_info(session_id)

    def clear_session(self, session_id: str) -> bool:
        return self.context.clear_session(session_id)

    def info(self) -> Dict[str, Any]:
        return {
            "model": self.nlu.info(),
            "max_context_turns": self.context.max_turns,
            "active_sessions": len(self.context.list_sessions()),
        }


# --- process-wide singleton --------------------------------------------------
# The checkpoint is ~2.2 GB; one instance per process, built on first use.
_pipeline: Optional[NLUPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> NLUPipeline:
    """Return the shared pipeline, constructing it on first call."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                logger.info("Building NLU pipeline (loading checkpoint)...")
                _pipeline = NLUPipeline()
    return _pipeline


def process_message(session_id: Optional[str], message: str) -> Dict[str, Any]:
    """Module-level shortcut over the shared pipeline."""
    return get_pipeline().process_message(session_id, message)
