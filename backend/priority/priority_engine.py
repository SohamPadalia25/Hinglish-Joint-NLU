"""Priority-based routing layer.

This is a transparent rule engine, NOT a trained model. No transformer was
trained on the `priority` column. Every number below is a hand-set weight, and
every decision comes back with the list of rules that fired.

The intent base scores are not invented: they are the *observed* priority mix
per intent in `hinglish_support_dataset_expanded_final.tsv`, collapsed to a
single number with

    base = (count(HIGH) * 1.0 + count(MEDIUM) * 0.5 + count(LOW) * 0.0) / count(all)

so REQUEST_ESCALATE (171/171 HIGH) scores 1.00 and FEEDBACK_POSITIVE
(239/239 LOW) scores 0.00. Using the dataset's own distribution keeps the rules
defensible instead of arbitrary; `test_priority_engine.py` asserts the engine
reproduces the dataset's majority priority label for all 15 intents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Priority-weighted share of HIGH/MEDIUM per intent, measured over all 2,989 rows.
DEFAULT_INTENT_BASE_SCORES: Mapping[str, float] = {
    "REQUEST_ESCALATE":    1.000,   # 171 HIGH /   0 MED /   0 LOW
    "COMPLAINT_DELIVERY":  0.968,   # 205 / 14 /  0
    "PAYMENT_ISSUE":       0.944,   # 165 / 21 /  0
    "COMPLAINT_PRODUCT":   0.906,   # 193 / 36 /  4
    "COMPLAINT_SERVICE":   0.862,   # 175 / 55 /  5
    "COMPLAINT_BILLING":   0.858,   # 148 / 49 /  4
    "RETURN_REQUEST":      0.778,   #  99 / 62 /  6
    "FEEDBACK_NEGATIVE":   0.759,   #  99 / 54 / 13
    "REQUEST_REFUND":      0.713,   #  74 / 100 / 0
    "QUERY_REFUND_STATUS": 0.544,   #  40 / 119 / 24
    "REQUEST_CANCEL":      0.515,   #  21 / 134 / 16
    "REQUEST_CALLBACK":    0.461,   #  17 / 107 / 29
    "QUERY_ORDER_STATUS":  0.410,   #   8 / 147 / 44
    "QUERY_PRODUCT_INFO":  0.238,   #   0 / 139 / 153
    "FEEDBACK_POSITIVE":   0.000,   #   0 /   0 / 239
}

DEFAULT_SENTIMENT_MODIFIERS: Mapping[str, float] = {
    "NEGATIVE": +0.15,
    "NEUTRAL": 0.00,
    "POSITIVE": -0.30,
}

# Slot types that indicate time pressure, contractual exposure or a threat.
URGENCY_SLOT_TYPES: frozenset = frozenset({
    "URGENCY", "SLA", "THREAT", "ESCALATION_LEVEL", "ESCALATION_TYPE",
    "WAIT_TIME", "COMPLAINT_COUNT", "IMPACT", "PRIORITY", "DEMAND",
    "COMPLIANCE", "RESTRICTION",
})

# Intents that always reach a person regardless of the numeric score.
FORCE_HUMAN_INTENTS: frozenset = frozenset({"REQUEST_ESCALATE", "REQUEST_CALLBACK"})


@dataclass
class PriorityRules:
    """Every knob the engine uses. Swap one in to retune without code edits."""

    intent_base_scores: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INTENT_BASE_SCORES)
    )
    sentiment_modifiers: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SENTIMENT_MODIFIERS)
    )
    urgency_slot_types: frozenset = URGENCY_SLOT_TYPES
    force_human_intents: frozenset = FORCE_HUMAN_INTENTS

    unknown_intent_base: float = 0.50
    urgency_slot_bonus: float = 0.10
    max_urgency_bonus: float = 0.20
    repeated_intent_bonus: float = 0.10
    sustained_negative_bonus: float = 0.10
    sustained_negative_min_turns: int = 2

    high_threshold: float = 0.75
    medium_threshold: float = 0.35

    routing_by_priority: Mapping[str, str] = field(
        default_factory=lambda: {
            "HIGH": "HUMAN_AGENT",
            "MEDIUM": "ASSISTED_BOT",
            "LOW": "AUTO_BOT",
        }
    )


class PriorityEngine:
    """Scores a message HIGH / MEDIUM / LOW and picks a routing target."""

    def __init__(self, rules: Optional[PriorityRules] = None):
        self.rules = rules or PriorityRules()

    def predict_priority(
        self,
        intent: Optional[str],
        sentiment: Optional[str],
        slots: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score one message.

        Args:
            intent: predicted intent label.
            sentiment: predicted sentiment label.
            slots: slot spans from `HinglishNLU.predict`.
            context: `ConversationContext.get_relevant_context` output, or None.

        Returns:
            ``{"priority", "priority_score", "routing", "reason"}`` plus the
            individual rule contributions under ``"rules_fired"``.
        """
        rules = self.rules
        slots = list(slots or [])
        context = dict(context or {})

        fired: List[Dict[str, Any]] = []

        known_intent = intent in rules.intent_base_scores
        score = rules.intent_base_scores.get(intent, rules.unknown_intent_base)
        fired.append({
            "rule": "intent_base",
            "delta": round(score, 3),
            "detail": (
                f"intent={intent} base={score:.3f}" if known_intent
                else f"unrecognised intent={intent!r}, using default {score:.2f}"
            ),
        })

        modifier = rules.sentiment_modifiers.get(sentiment, 0.0)
        if modifier:
            score += modifier
            fired.append({
                "rule": "sentiment_modifier",
                "delta": round(modifier, 3),
                "detail": f"sentiment={sentiment}",
            })

        urgency_types = sorted({
            s.get("type") for s in slots if s.get("type") in rules.urgency_slot_types
        })
        if urgency_types:
            bonus = min(len(urgency_types) * rules.urgency_slot_bonus, rules.max_urgency_bonus)
            score += bonus
            fired.append({
                "rule": "urgency_slots",
                "delta": round(bonus, 3),
                "detail": f"urgency-bearing slots {urgency_types}",
            })

        recent_intents = list(context.get("recent_intents", []))
        if intent and recent_intents.count(intent) >= 1:
            score += rules.repeated_intent_bonus
            fired.append({
                "rule": "repeated_intent",
                "delta": round(rules.repeated_intent_bonus, 3),
                "detail": f"user already raised {intent} in this session",
            })

        negatives = [s for s in context.get("recent_sentiments", []) if s == "NEGATIVE"]
        if len(negatives) >= rules.sustained_negative_min_turns:
            score += rules.sustained_negative_bonus
            fired.append({
                "rule": "sustained_negative",
                "delta": round(rules.sustained_negative_bonus, 3),
                "detail": f"{len(negatives)} negative turns in recent history",
            })

        score = max(0.0, min(1.0, score))

        if score >= rules.high_threshold:
            label = "HIGH"
        elif score >= rules.medium_threshold:
            label = "MEDIUM"
        else:
            label = "LOW"

        routing = rules.routing_by_priority.get(label, "ASSISTED_BOT")
        if intent in rules.force_human_intents:
            routing = "HUMAN_AGENT"
            fired.append({
                "rule": "force_human_routing",
                "delta": 0.0,
                "detail": f"{intent} always routes to a human",
            })

        return {
            "priority": label,
            "priority_score": round(score, 3),
            "routing": routing,
            "reason": self._explain(label, score, routing, fired),
            "rules_fired": fired,
        }

    def _explain(
        self,
        label: str,
        score: float,
        routing: str,
        fired: Sequence[Mapping[str, Any]],
    ) -> str:
        parts = [
            f"{rule['detail']} ({rule['delta']:+.2f})" if rule["delta"] else rule["detail"]
            for rule in fired
        ]
        return (
            f"{label} (score {score:.2f}, thresholds HIGH>={self.rules.high_threshold} "
            f"MEDIUM>={self.rules.medium_threshold}) -> {routing}. "
            + "; ".join(parts)
        )


_DEFAULT_ENGINE = PriorityEngine()


def predict_priority(
    intent: Optional[str],
    sentiment: Optional[str],
    slots: Optional[Sequence[Mapping[str, Any]]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Module-level shortcut using the default rule set."""
    return _DEFAULT_ENGINE.predict_priority(intent, sentiment, slots, context)
