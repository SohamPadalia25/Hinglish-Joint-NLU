"""Decide whether a message should be read against earlier turns.

The trained NLU model is single-utterance: it never sees conversation history.
Rather than concatenating the transcript and pushing it through XLM-R - which
would put the encoder far outside its training distribution and let an older
turn override the current one - this layer keeps the model's prediction for the
*current* message and reports which earlier facts are relevant to it.

The rule is deliberately conservative:

    self-contained query  -> ignore history, trust the current prediction
    short / ambiguous     -> surface the previous intent, sentiment and slots

Nothing here rewrites the model's intent or sentiment. It reports and it
inherits slots the user did not repeat; it never fabricates a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from backend.model.preprocessing import normalize_hinglish

# Words that typically point back at something already said, rather than
# introducing it. Hinglish + English, matched on normalized (lowercased) tokens.
ANAPHORA_MARKERS: FrozenSet[str] = frozenset({
    "iska", "iski", "isko", "isme", "is",
    "uska", "uski", "usko", "usme", "us",
    "ye", "yeh", "wo", "woh", "vo",
    "same", "it", "its", "that", "this", "these", "those",
})

# Words that signal a follow-up on an established topic ("when will it come?").
FOLLOW_UP_MARKERS: FrozenSet[str] = frozenset({
    "kab", "kitna", "kitne", "kitni", "abhi", "ab",
    "phir", "dobara", "again", "still", "yet", "update", "status",
    "aur", "bhi", "then", "next",
})

# Slot types worth carrying forward. Identifiers and amounts stay true across a
# conversation; a sentiment-ish or wording-specific slot does not.
INHERITABLE_SLOT_TYPES: FrozenSet[str] = frozenset({
    "ORDER_ID", "AMOUNT", "ACTUAL_AMOUNT", "PRODUCT_NAME", "PRODUCT_TYPE",
    "ITEM", "PAYMENT_METHOD", "PAYMENT_TYPE", "COURIER", "DATE",
    "PROMISED_DATE", "ACTUAL_DATE", "TAX_ID", "PROMO_CODE", "CONTACT",
    "ACCOUNT_TYPE", "SUBSCRIPTION", "PLAN",
})


@dataclass
class InterpreterConfig:
    """Thresholds for the ambiguity rules. All tunable, none learned."""

    short_query_tokens: int = 5
    low_intent_confidence: float = 0.60
    anaphora_markers: FrozenSet[str] = field(default_factory=lambda: ANAPHORA_MARKERS)
    follow_up_markers: FrozenSet[str] = field(default_factory=lambda: FOLLOW_UP_MARKERS)
    inheritable_slot_types: FrozenSet[str] = field(default_factory=lambda: INHERITABLE_SLOT_TYPES)
    max_relevant_turns: int = 3


class ContextInterpreter:
    """Applies the self-contained / ambiguous rules to one message."""

    def __init__(self, config: Optional[InterpreterConfig] = None):
        self.config = config or InterpreterConfig()

    def interpret(
        self,
        message: str,
        nlu: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return whether history was used, why, and what was carried forward.

        Args:
            message: the raw user message.
            nlu: the current message's prediction from `HinglishNLU.predict`.
            context: `ConversationContext.get_relevant_context(session_id)`.
        """
        cfg = self.config
        tokens = normalize_hinglish(message).split()
        token_set = set(tokens)

        if context.get("turn_count", 0) == 0:
            return self._result(
                used=False,
                reason="No prior turns in this session; nothing to interpret against.",
                signals=[],
                context=context,
                inherited=[],
            )

        signals: List[Dict[str, str]] = []

        if len(tokens) <= cfg.short_query_tokens:
            signals.append({
                "signal": "short_query",
                "detail": f"{len(tokens)} tokens (<= {cfg.short_query_tokens})",
            })

        anaphora = sorted(token_set & cfg.anaphora_markers)
        if anaphora:
            signals.append({
                "signal": "anaphora_marker",
                "detail": f"refers back via {anaphora}",
            })

        follow_ups = sorted(token_set & cfg.follow_up_markers)
        if follow_ups:
            signals.append({
                "signal": "follow_up_marker",
                "detail": f"follow-up phrasing {follow_ups}",
            })

        confidence = float(nlu.get("intent_confidence", 0.0))
        if confidence < cfg.low_intent_confidence:
            signals.append({
                "signal": "low_intent_confidence",
                "detail": f"intent confidence {confidence:.2f} < {cfg.low_intent_confidence}",
            })

        current_types = {s.get("type") for s in nlu.get("slots", [])}
        known_slots = context.get("known_slots", [])
        if not current_types and known_slots:
            signals.append({
                "signal": "no_slots_in_query",
                "detail": "current message carries no entities, earlier turns do",
            })

        if not signals:
            return self._result(
                used=False,
                reason=(
                    "Current query is self-contained "
                    f"({len(tokens)} tokens, intent confidence {confidence:.2f}); "
                    "answered on its own."
                ),
                signals=[],
                context=context,
                inherited=[],
            )

        inherited = [
            {**slot, "source": "context"}
            for slot in known_slots
            if slot.get("type") in cfg.inheritable_slot_types
            and slot.get("type") not in current_types
        ]

        previous_intent = context.get("previous_intent")
        reason = (
            f"Query looks like a follow-up ({', '.join(s['signal'] for s in signals)}). "
            f"Carried forward previous_intent={previous_intent}"
        )
        if inherited:
            reason += f" and slots {[s['type'] for s in inherited]}"
        reason += "."

        return self._result(
            used=True,
            reason=reason,
            signals=signals,
            context=context,
            inherited=inherited,
        )

    def _result(
        self,
        used: bool,
        reason: str,
        signals: List[Dict[str, str]],
        context: Dict[str, Any],
        inherited: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        relevant_history = context.get("recent_turns", [])[-self.config.max_relevant_turns:] if used else []
        return {
            "context_used": used,
            "context_reason": reason,
            "relevant_history": relevant_history,
            "signals": signals,
            "previous_intent": context.get("previous_intent") if used else None,
            "previous_sentiment": context.get("previous_sentiment") if used else None,
            "inherited_slots": inherited,
        }
