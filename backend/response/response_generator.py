"""Deterministic, template-based reply generation.

Consumes the structured result from `NLUPipeline.process_message` and turns it
into an agent-facing reply. No model is called here, no external LLM is required,
and the same input always produces the same output.

The hard rule this module is built around: **the generator may only restate what
the pipeline actually extracted.** It never states an order status, a refund or
delivery date, an amount, a transaction outcome, an account detail or a policy.
Where a fact is needed and missing, it asks the user for it instead.

Two mechanisms enforce that:

* Every template string is fact-free and contains no digits. Anything concrete in
  a reply came from a slot value, verbatim.
* `_is_referenceable` drops partial slots before they are ever quoted. The model
  emits `AMOUNT = "rupaye"` (dropping the "799") often enough that quoting slot
  values blindly would produce nonsense; identifier- and quantity-like slots must
  contain a digit to be quoted at all. A dropped slot is simply not mentioned -
  it is never repaired from the raw message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Disposition of a generated reply. Precedence is resolved in `_response_type`.
RESPONSE_TYPES = (
    "ESCALATION",           # going to a human now
    "CALLBACK",             # a call-back was requested
    "CLARIFICATION",        # we need a detail before we can act
    "ACTION_CONFIRMATION",  # a cancel / refund / return request was taken down
    "ACKNOWLEDGEMENT",      # a complaint was recorded
    "INFORMATION",          # an informational query
    "APPRECIATION",         # positive feedback
    "APOLOGY",              # negative feedback
    "FALLBACK",             # intent not recognised
)

# Intents that always involve a person, independent of the priority score.
HUMAN_INTENTS = frozenset({"REQUEST_ESCALATE", "REQUEST_CALLBACK"})

# Slot types whose value is only meaningful when it carries a number. The model
# frequently emits a partial span (AMOUNT = "rupaye"), and quoting that reads as
# nonsense - so these are dropped unless a digit survived in the span.
DIGIT_REQUIRED_SLOTS = frozenset({
    "ORDER_ID", "AMOUNT", "ACTUAL_AMOUNT", "QUANTITY", "ORDER_COUNT",
    "COMPLAINT_COUNT", "TAX_ID", "CONTACT", "RATING", "THRESHOLD",
    "BUDGET", "DISCOUNT", "CHARGE", "WAIT_TIME",
})

SLOT_DISPLAY_NAMES: Mapping[str, str] = {
    "ORDER_ID": "order ID",
    "AMOUNT": "amount",
    "ACTUAL_AMOUNT": "amount charged",
    "PAYMENT_METHOD": "payment method",
    "PAYMENT_TYPE": "payment type",
    "PRODUCT_NAME": "product",
    "PRODUCT_TYPE": "product type",
    "ITEM": "item",
    "COURIER": "courier",
    "DATE": "timeframe",
    "PROMO_CODE": "promo code",
    "CONTACT": "contact number",
    "REASON": "reason",
    "ISSUE": "issue",
}

SENTIMENT_OPENERS: Mapping[str, str] = {
    "NEGATIVE": "I'm sorry for the trouble this has caused.",
    "NEUTRAL": "Thanks for getting in touch.",
    "POSITIVE": "Thank you for the kind words!",
}

URGENT_OPENERS: Mapping[str, str] = {
    "NEGATIVE": "I'm really sorry about this, and I understand it needs urgent attention.",
    "NEUTRAL": "Thanks for flagging this - I can see it needs prompt attention.",
    "POSITIVE": "Thank you for the kind words!",
}

DEFAULT_SENTIMENT = "NEUTRAL"


@dataclass(frozen=True)
class IntentStrategy:
    """How one intent is answered.

    Attributes:
        summary: what we understood, phrased as a restatement of the user's own
            message. Never a claim about system state.
        action: what support can do next. Phrased as capability ("we can check"),
            never as a completed outcome ("your refund is processed").
        response_type: the disposition when no override applies.
        helpful_slots: slot types that would let support act. When none of these
            are available - freshly extracted or carried in from context - the
            reply asks for them.
        ask: the question to ask when `helpful_slots` are missing.
    """

    summary: str
    action: str
    response_type: str
    helpful_slots: Tuple[str, ...] = ()
    ask: str = ""


# One strategy per intent label in model_metadata2.json. No template contains a
# digit, a date, a status or a policy statement - see test_no_invented_numbers.
INTENT_STRATEGIES: Mapping[str, IntentStrategy] = {
    "COMPLAINT_BILLING": IntentStrategy(
        summary="there's a problem with the way you've been billed",
        action="We can review the charges on your account and get any billing error corrected.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "AMOUNT", "ACTUAL_AMOUNT"),
        ask="Could you share the order ID or invoice number, and the amount you were charged?",
    ),
    "COMPLAINT_DELIVERY": IntentStrategy(
        summary="your order hasn't reached you as expected",
        action="We can check the shipment with the courier and chase up the delivery.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID",),
        ask="Could you share your order ID so we can pull up the shipment?",
    ),
    "COMPLAINT_PRODUCT": IntentStrategy(
        summary="the product you received isn't up to the mark",
        action="We can review the item and arrange a replacement or a return.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "PRODUCT_NAME", "ITEM"),
        ask="Could you share the order ID and a photo of the item you received?",
    ),
    "COMPLAINT_SERVICE": IntentStrategy(
        summary="the support you've received so far hasn't been good enough",
        action="We'll review how this was handled and make sure it's picked up properly.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID",),
        ask="If you have a previous complaint or ticket reference, sharing it will help us trace it.",
    ),
    "FEEDBACK_NEGATIVE": IntentStrategy(
        summary="your experience with us hasn't been a good one",
        action="Your feedback is noted and passed on to the team responsible.",
        response_type="APOLOGY",
        helpful_slots=("ORDER_ID",),
        ask="If there's a particular order or interaction behind this, tell me and I'll have it looked into.",
    ),
    "FEEDBACK_POSITIVE": IntentStrategy(
        summary="you're happy with how things went",
        action="I'll pass this on to the team - it genuinely means a lot to them.",
        response_type="APPRECIATION",
    ),
    "PAYMENT_ISSUE": IntentStrategy(
        summary="your payment didn't go through the way it should have",
        action="We can help you check the payment and the order status, and get a deducted or failed payment traced.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "AMOUNT", "PAYMENT_METHOD"),
        ask="Could you share the order ID and the payment or UPI reference from your bank?",
    ),
    "QUERY_ORDER_STATUS": IntentStrategy(
        summary="you'd like an update on your order",
        action="I can get the current status looked up for you.",
        response_type="INFORMATION",
        helpful_slots=("ORDER_ID",),
        ask="Could you share your order ID?",
    ),
    "QUERY_PRODUCT_INFO": IntentStrategy(
        summary="you'd like more details about a product",
        action="I can share the specifications, availability and pricing we have on record.",
        response_type="INFORMATION",
        helpful_slots=("PRODUCT_NAME", "PRODUCT_TYPE", "ITEM"),
        ask="Which product would you like to know about?",
    ),
    "QUERY_REFUND_STATUS": IntentStrategy(
        summary="you'd like to know where your refund has reached",
        action="I can get the refund checked against your order and tell you where it stands.",
        response_type="INFORMATION",
        helpful_slots=("ORDER_ID",),
        ask="Could you share the order ID the refund relates to?",
    ),
    "REQUEST_CALLBACK": IntentStrategy(
        summary="you'd like someone from our team to call you",
        action="I've noted your callback request and passed it to the support team.",
        response_type="CALLBACK",
        helpful_slots=("CONTACT",),
        ask="Could you confirm the number to call and a time window that suits you?",
    ),
    "REQUEST_CANCEL": IntentStrategy(
        summary="you'd like to cancel your order",
        action="We can put a cancellation request through once the order is confirmed.",
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID",),
        ask="Could you share the order ID you'd like cancelled?",
    ),
    "REQUEST_ESCALATE": IntentStrategy(
        summary="you'd like this taken further",
        action="I'm escalating this to a senior support specialist.",
        response_type="ESCALATION",
        helpful_slots=("ORDER_ID",),
        ask="If you have an order ID or an earlier complaint reference, sharing it will speed this up.",
    ),
    "REQUEST_REFUND": IntentStrategy(
        summary="you'd like a refund",
        action="We can raise a refund request and have it reviewed against your order.",
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID", "AMOUNT"),
        ask="Could you share the order ID and what went wrong, so we can raise the request?",
    ),
    "RETURN_REQUEST": IntentStrategy(
        summary="you'd like to return the item",
        action="We can start a return and arrange a pickup once the order is confirmed.",
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID", "PRODUCT_NAME", "ITEM"),
        ask="Could you share the order ID and what's wrong with the item?",
    ),
}

FALLBACK_STRATEGY = IntentStrategy(
    summary="you need some help",
    action="I want to make sure this reaches the right place.",
    response_type="FALLBACK",
    ask="Could you tell me a little more about what you need help with?",
)


class ResponseGenerator:
    """Turns a pipeline result into a reply.

    Deterministic and offline: no network call, no LLM, no randomness.

    Example
    -------
    >>> ResponseGenerator().generate(pipeline_result)["response_type"]
    'ESCALATION'
    """

    def __init__(self, strategies: Optional[Mapping[str, IntentStrategy]] = None):
        self.strategies = dict(strategies or INTENT_STRATEGIES)

    # ------------------------------------------------------------------ public

    def generate(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        """Build the reply for one pipeline result.

        Args:
            result: output of `NLUPipeline.process_message` (or `predict`, whose
                `context` block is simply absent).

        Returns:
            ``{"response": str, "response_type": str, "requires_human": bool}``
        """
        nlu = dict(result.get("nlu") or {})
        context = dict(result.get("context") or {})
        priority = dict(result.get("priority") or {})

        intent = nlu.get("intent") or "UNKNOWN"
        sentiment = nlu.get("sentiment") or DEFAULT_SENTIMENT
        priority_label = (priority.get("label") or "MEDIUM").upper()
        routing = priority.get("routing") or ""

        strategy = self.strategies.get(intent, FALLBACK_STRATEGY)

        own_slots = self._referenceable(nlu.get("slots") or [])
        inherited = self._referenceable(context.get("inherited_slots") or [])
        context_used = bool(context.get("used"))

        needs_detail = bool(strategy.ask) and not self._has_helpful_slot(
            strategy, own_slots, inherited if context_used else []
        )
        requires_human = self._requires_human(intent, priority_label, routing)

        return {
            "response": self._compose(
                strategy=strategy,
                sentiment=sentiment,
                priority_label=priority_label,
                own_slots=own_slots,
                inherited=inherited if context_used else [],
                needs_detail=needs_detail,
                requires_human=requires_human,
            ),
            "response_type": self._response_type(
                intent, strategy, requires_human, needs_detail
            ),
            "requires_human": requires_human,
        }

    # ------------------------------------------------------------------ slots

    @staticmethod
    def _is_referenceable(slot: Mapping[str, Any]) -> bool:
        """Whether a slot value is complete enough to quote back at the user.

        The model routinely returns partial spans. `AMOUNT = "rupaye"` carries no
        information, and repairing it from the raw message would mean inventing
        the number - so it is dropped instead.
        """
        value = str(slot.get("value") or "").strip()
        if not value:
            return False
        if slot.get("type") in DIGIT_REQUIRED_SLOTS and not any(c.isdigit() for c in value):
            return False
        return True

    @classmethod
    def _referenceable(cls, slots: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(s) for s in slots if cls._is_referenceable(s)]

    @staticmethod
    def _has_helpful_slot(
        strategy: IntentStrategy,
        own_slots: Sequence[Mapping[str, Any]],
        inherited: Sequence[Mapping[str, Any]],
    ) -> bool:
        available = {s.get("type") for s in own_slots} | {s.get("type") for s in inherited}
        return bool(set(strategy.helpful_slots) & available)

    @staticmethod
    def _describe(slot: Mapping[str, Any]) -> str:
        slot_type = str(slot.get("type", ""))
        name = SLOT_DISPLAY_NAMES.get(slot_type, slot_type.replace("_", " ").lower())
        return f"{name} {slot['value']}"

    @classmethod
    def _describe_all(cls, slots: Sequence[Mapping[str, Any]]) -> str:
        """Comma-separated, with 'and' before the last - reads as a sentence."""
        described = [cls._describe(s) for s in slots]
        if len(described) == 1:
            return described[0]
        return ", ".join(described[:-1]) + f" and the {described[-1]}"

    # -------------------------------------------------------------- assembly

    def _compose(
        self,
        strategy: IntentStrategy,
        sentiment: str,
        priority_label: str,
        own_slots: Sequence[Mapping[str, Any]],
        inherited: Sequence[Mapping[str, Any]],
        needs_detail: bool,
        requires_human: bool,
    ) -> str:
        openers = URGENT_OPENERS if priority_label == "HIGH" else SENTIMENT_OPENERS
        parts = [
            openers.get(sentiment, openers[DEFAULT_SENTIMENT]),
            f"I understand that {strategy.summary}.",
        ]

        if own_slots:
            parts.append(f"I've noted the {self._describe_all(own_slots)}.")

        if inherited:
            parts.append(
                f"I'm carrying forward the {self._describe_all(inherited)} "
                "from your earlier message."
            )

        parts.append(strategy.action)

        if needs_detail and strategy.ask:
            parts.append(strategy.ask)

        parts.append(self._closer(priority_label, requires_human))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _closer(priority_label: str, requires_human: bool) -> str:
        if priority_label == "HIGH":
            return "I've marked this as high priority and a support specialist will pick it up."
        if requires_human:
            return "I've passed this to our support team so someone can get back to you."
        if priority_label == "LOW":
            return "Happy to help with anything else right here."
        return "Our support team will follow this up with you."

    # ------------------------------------------------------------ disposition

    @staticmethod
    def _requires_human(intent: str, priority_label: str, routing: str) -> bool:
        """A person is involved for escalations, callbacks and anything HIGH."""
        return (
            intent in HUMAN_INTENTS
            or priority_label == "HIGH"
            or routing == "HUMAN_AGENT"
        )

    @staticmethod
    def _response_type(
        intent: str,
        strategy: IntentStrategy,
        requires_human: bool,
        needs_detail: bool,
    ) -> str:
        """Precedence: callback > escalation > clarification > intent default."""
        if intent == "REQUEST_CALLBACK":
            return "CALLBACK"
        if requires_human:
            return "ESCALATION"
        if needs_detail:
            return "CLARIFICATION"
        return strategy.response_type
