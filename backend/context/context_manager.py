"""Session-scoped conversation memory.

Holds a bounded window of recent turns per session. In-process and in-memory:
sessions vanish on restart, which is fine for a single-node demo but is the
first thing to replace (Redis, a table) before running more than one worker.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from backend import config as cfg


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Turn:
    """One user message plus whatever the pipeline concluded about it."""

    turn_index: int
    timestamp: str
    user_message: str
    normalized_message: str = ""
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    sentiment: Optional[str] = None
    sentiment_confidence: float = 0.0
    slots: List[Dict[str, Any]] = field(default_factory=list)
    priority: Optional[str] = None
    bot_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationContext:
    """Bounded per-session turn history.

    Args:
        max_turns: how many recent turns to retain per session (3-5 is the
            useful range; beyond that older turns stop being relevant and start
            dragging irrelevant intents into follow-up interpretation).
    """

    def __init__(self, max_turns: int = cfg.MAX_CONTEXT_TURNS):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self._sessions: Dict[str, Deque[Turn]] = {}
        self._created_at: Dict[str, str] = {}
        self._turn_counter: Dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- lifecycle

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a session (or return an existing id unchanged)."""
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            if sid not in self._sessions:
                self._sessions[sid] = deque(maxlen=self.max_turns)
                self._created_at[sid] = _now()
                self._turn_counter[sid] = 0
            return sid

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def clear_session(self, session_id: str) -> bool:
        """Forget a session. Returns False if it did not exist."""
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            self._created_at.pop(session_id, None)
            self._turn_counter.pop(session_id, None)
            return existed

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions)

    # ------------------------------------------------------------------ write

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        nlu: Optional[Dict[str, Any]] = None,
        bot_response: Optional[str] = None,
        priority: Optional[str] = None,
        normalized_message: str = "",
    ) -> Turn:
        """Append a turn, creating the session if needed.

        Oldest turns fall off automatically once `max_turns` is exceeded.
        """
        nlu = nlu or {}
        with self._lock:
            if session_id not in self._sessions:
                self.create_session(session_id)

            self._turn_counter[session_id] += 1
            turn = Turn(
                turn_index=self._turn_counter[session_id],
                timestamp=_now(),
                user_message=user_message,
                normalized_message=normalized_message,
                intent=nlu.get("intent"),
                intent_confidence=float(nlu.get("intent_confidence", 0.0)),
                sentiment=nlu.get("sentiment"),
                sentiment_confidence=float(nlu.get("sentiment_confidence", 0.0)),
                slots=list(nlu.get("slots", [])),
                priority=priority,
                bot_response=bot_response,
            )
            self._sessions[session_id].append(turn)
            return turn

    def set_bot_response(self, session_id: str, response: str) -> bool:
        """Attach a reply to the most recent turn."""
        with self._lock:
            history = self._sessions.get(session_id)
            if not history:
                return False
            history[-1].bot_response = response
            return True

    # ------------------------------------------------------------------- read

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retained turns, oldest first. Unknown session -> empty list."""
        with self._lock:
            return [turn.to_dict() for turn in self._sessions.get(session_id, ())]

    def session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if session_id not in self._sessions:
                return None
            history = self._sessions[session_id]
            return {
                "session_id": session_id,
                "created_at": self._created_at[session_id],
                "total_turns": self._turn_counter[session_id],
                "retained_turns": len(history),
                "max_turns": self.max_turns,
                "history": [turn.to_dict() for turn in history],
            }

    def get_relevant_context(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarise what earlier turns established, for the interpreter.

        `known_slots` keeps the most recent value per slot type - when a user
        restates an order id, the newer one wins.
        """
        with self._lock:
            history = list(self._sessions.get(session_id, ()))

        if max_turns is not None:
            history = history[-max_turns:]

        if not history:
            return {
                "session_id": session_id,
                "turn_count": 0,
                "previous_intent": None,
                "previous_intent_confidence": 0.0,
                "previous_sentiment": None,
                "recent_intents": [],
                "recent_sentiments": [],
                "known_slots": [],
                "recent_turns": [],
            }

        known: Dict[str, Dict[str, Any]] = {}
        for turn in history:
            for slot in turn.slots:
                slot_type = slot.get("type")
                if slot_type:
                    known[slot_type] = {
                        "type": slot_type,
                        "value": slot.get("value"),
                        "from_turn": turn.turn_index,
                    }

        last = history[-1]
        return {
            "session_id": session_id,
            "turn_count": len(history),
            "previous_intent": last.intent,
            "previous_intent_confidence": last.intent_confidence,
            "previous_sentiment": last.sentiment,
            "recent_intents": [t.intent for t in history if t.intent],
            "recent_sentiments": [t.sentiment for t in history if t.sentiment],
            "known_slots": list(known.values()),
            "recent_turns": [t.to_dict() for t in history],
        }
