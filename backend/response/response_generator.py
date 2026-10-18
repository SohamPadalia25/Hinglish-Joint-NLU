"""Deterministic, template-based reply generation - in natural Hinglish.

Consumes the structured result from `NLUPipeline.process_message` and turns it
into an agent-facing reply. No model is called here, no external LLM is required,
and the same input always produces the same output.

Replies are written in Roman-script Hinglish, the way an Indian support agent
actually speaks: conversational Hindi carrying the sentence, with the ordinary
English support vocabulary (order, payment, refund, status, priority, callback)
left in English because that is what people really say. The templates are not
mechanical translations of the earlier English ones - each was rewritten so the
sentence sounds spoken rather than converted. No Devanagari, no formal register
("mahoday", "bhugtan"), no word-for-word Hindi.

The hard rule this module is built around is unchanged: **the generator may only
restate what the pipeline actually extracted.** It never states an order status,
a refund or delivery date, an amount, a transaction outcome, an account detail or
a policy. Where a fact is needed and missing, it asks the user for it instead.

Two mechanisms enforce that, and both survive the language change:

* Every template string is fact-free and contains no digits. Anything concrete in
  a reply came from a slot value, verbatim.
* `_is_referenceable` drops partial slots before they are ever quoted. The model
  emits `AMOUNT = "rupaye"` (dropping the "799") often enough that quoting slot
  values blindly would produce nonsense; identifier- and quantity-like slots must
  contain a digit to be quoted at all. A dropped slot is simply not mentioned -
  it is never repaired from the raw message.

Routing language is deliberately careful. The system routes; it does not raise
tickets. So a reply says a case has been marked high priority and routed to a
human agent - both of which the priority engine really did - and never claims a
ticket was created or an agent has been assigned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Disposition of a generated reply. Precedence is resolved in `_response_type`.
# Unchanged from the English implementation - only the wording moved to Hinglish.
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

# Slot labels stay in English - "order ID" and "amount" are what a Hinglish
# speaker says out loud, and translating them would read as stilted.
SLOT_DISPLAY_NAMES: Mapping[str, str] = {
    "ORDER_ID": "order ID",
    "AMOUNT": "amount",
    "ACTUAL_AMOUNT": "charge hui amount",
    "PAYMENT_METHOD": "payment method",
    "PAYMENT_TYPE": "payment type",
    "PRODUCT_NAME": "product",
    "PRODUCT_TYPE": "product type",
    "ITEM": "item",
    "COURIER": "courier",
    "DATE": "",                 # "5 din" speaks for itself
    "PROMO_CODE": "promo code",
    "CONTACT": "contact number",
    "REASON": "reason",
    "ISSUE": "issue",
}

# Negative replies carry their own apology, worded for the intent - a generic
# "Sorry, aapko pareshani hui." glued to every reply is what makes a bot sound
# like a bot. This extra line is added only when the priority engine saw the
# customer repeating themselves, and it deliberately contains no second "sorry".
FRUSTRATED_OPENER = "Aapko baar baar bolna pad raha hai, ye bilkul theek nahi hai."

# Rules from the priority engine that mean "this customer is repeating themselves".
FRUSTRATION_RULES = frozenset({"repeated_intent", "sustained_negative"})

# Below this sentiment confidence the tone falls back to neutral: apologising
# to someone who was not upset reads worse than not apologising.
MIN_SENTIMENT_CONFIDENCE = 0.6

DEFAULT_SENTIMENT = "NEUTRAL"


@dataclass(frozen=True)
class IntentStrategy:
    """How one intent is answered, in Hinglish.

    Attributes:
        replies: sentiment -> one short line saying what support will do.
            Phrased as capability ("check karwata hoon"), never as a completed
            outcome ("refund ho gaya"). Missing sentiments fall back to NEUTRAL.
        response_type: the disposition when no override applies.
        helpful_slots: slot types that would let support act. When none of these
            are available - freshly extracted or carried in from context - the
            reply asks for them.
        ask: the question to ask when `helpful_slots` are missing.
    """

    replies: Mapping[str, str]
    response_type: str
    helpful_slots: Tuple[str, ...] = ()
    ask: str = ""

    def line(self, sentiment: str) -> str:
        return self.replies.get(sentiment) or self.replies[DEFAULT_SENTIMENT]


# One strategy per intent label in model_metadata2.json. No template contains a
# digit, a date, a status or a policy statement - see test_no_template_contains_a_digit.
INTENT_STRATEGIES: Mapping[str, IntentStrategy] = {
    "COMPLAINT_BILLING": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, galat charge dekh ke gussa aana bilkul banta hai. Main abhi aapki billing check karta hoon.",
            "NEUTRAL": "Ji, main aapke charges ek baar check kar leta hoon.",
            "POSITIVE": "Bataane ke liye thanks! Main aapke charges ek baar check kar leta hoon.",
        },
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "AMOUNT", "ACTUAL_AMOUNT"),
        ask="Order ID ya invoice number bhej denge?",
    ),
    "COMPLAINT_DELIVERY": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, itna wait karna sach mein irritating hai. Main abhi courier se pata karta hoon ki delivery kahan atki hai.",
            "NEUTRAL": "Ji, main courier se aapki delivery ka pata karta hoon.",
            "POSITIVE": "Bataane ke liye thanks! Main courier se aapki delivery ka pata karta hoon.",
        },
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID",),
        ask="Apna order ID bata denge?",
    ),
    "COMPLAINT_PRODUCT": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, paise dekar kharab product milna bahut bura lagta hai. Replacement chahiye ya return, bataiye, main help karta hoon.",
            "NEUTRAL": "Achha, replacement chahiye ya return? Main dono mein help kar sakta hoon.",
            "POSITIVE": "Bataane ke liye thanks! Replacement chahiye ya return? Main dono mein help kar sakta hoon.",
        },
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "PRODUCT_NAME", "ITEM"),
        ask="Order ID aur item ki ek photo bhej denge?",
    ),
    "COMPLAINT_SERVICE": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, aapko jaisa support milna chahiye tha waisa nahi mila. Ab ye case main khud dekh raha hoon.",
            "NEUTRAL": "Ji, main dekhta hoon ki aapka case kaise handle hua tha.",
            "POSITIVE": "Feedback ke liye shukriya! Main aapka case ek baar review karta hoon.",
        },
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID",),
        ask="Pichhli complaint ka koi reference ho to share kar denge?",
    ),
    "FEEDBACK_NEGATIVE": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, aapka experience itna kharab raha. Aapka feedback main seedha team tak pahunchata hoon.",
            "NEUTRAL": "Aapka feedback note kar liya hai, main ise team tak pahunchata hoon.",
            "POSITIVE": "Honest feedback ke liye thanks, isse humein sudharne mein madad milti hai.",
        },
        response_type="APOLOGY",
        helpful_slots=("ORDER_ID",),
        ask="Kisi particular order ki baat hai to uska order ID bata denge?",
    ),
    "FEEDBACK_POSITIVE": IntentStrategy(
        replies={
            "NEGATIVE": "Feedback ke liye thank you. Kuch aur theek karna ho to bataiye, main yahin hoon.",
            "NEUTRAL": "Thank you! Aapka feedback main team tak pahuncha dunga.",
            "POSITIVE": "Thank you so much! Aapka feedback sunke bahut achha laga, main team ko zaroor bataunga.",
        },
        response_type="APPRECIATION",
    ),
    "PAYMENT_ISSUE": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, paise atak jaayein to tension hona normal hai. Main abhi aapki payment trace karta hoon.",
            "NEUTRAL": "Ji, main aapki payment aur order dono check kar leta hoon.",
            "POSITIVE": "Bilkul, main aapki payment check kar leta hoon.",
        },
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "AMOUNT", "PAYMENT_METHOD"),
        ask="Order ID aur UPI ya bank reference bhej denge?",
    ),
    "QUERY_ORDER_STATUS": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, aapko wait karna pad raha hai. Main abhi aapka order check karta hoon.",
            "NEUTRAL": "Ji, main aapka order status check kar leta hoon.",
            "POSITIVE": "Bilkul! Main aapka order status check kar leta hoon.",
        },
        response_type="INFORMATION",
        helpful_slots=("ORDER_ID",),
        ask="Order ID bata denge?",
    ),
    "QUERY_PRODUCT_INFO": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry agar pehle confusion hua. Main us product ki sahi details check karke batata hoon.",
            "NEUTRAL": "Ji, main aapko us product ki details bata deta hoon.",
            "POSITIVE": "Bilkul! Main aapko us product ki details bata deta hoon.",
        },
        response_type="INFORMATION",
        helpful_slots=("PRODUCT_NAME", "PRODUCT_TYPE", "ITEM"),
        ask="Aap kaunsa product dekh rahe hain?",
    ),
    "QUERY_REFUND_STATUS": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, refund ka intezaar sach mein frustrating hota hai. Main abhi check karta hoon ki wo kahan tak pahuncha.",
            "NEUTRAL": "Ji, main check karta hoon ki aapka refund kahan tak pahuncha.",
            "POSITIVE": "Bilkul, main check karta hoon ki aapka refund kahan tak pahuncha.",
        },
        response_type="INFORMATION",
        helpful_slots=("ORDER_ID",),
        ask="Jis order ka refund hai, uska order ID bata denge?",
    ),
    "REQUEST_CALLBACK": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, samajh sakta hoon ki baat karke solve karna aasan rahega. Aapki callback request team ko bhej di hai.",
            "NEUTRAL": "Ji, aapki callback request team ko bhej di hai.",
            "POSITIVE": "Bilkul! Aapki callback request team ko bhej di hai.",
        },
        response_type="CALLBACK",
        helpful_slots=("CONTACT",),
        ask="Kis number par aur kis time call karna aapke liye theek rahega?",
    ),
    "REQUEST_CANCEL": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry ki aapko cancel karna pad raha hai. Main aapki cancel request aage badhata hoon.",
            "NEUTRAL": "Theek hai, main aapki cancel request aage badhata hoon.",
            "POSITIVE": "Bilkul, main aapki cancel request aage badhata hoon.",
        },
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID",),
        ask="Kaunsa order cancel karna hai? Uska order ID bata denge?",
    ),
    "REQUEST_ESCALATE": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry, aapko escalate karna pada. Main ise abhi senior team ko escalate kar raha hoon.",
            "NEUTRAL": "Ji, main ise senior team ko escalate kar raha hoon.",
            "POSITIVE": "Theek hai, main ise senior team ko escalate kar raha hoon.",
        },
        response_type="ESCALATION",
        helpful_slots=("ORDER_ID",),
        ask="Order ID ya complaint reference ho to bhej denge?",
    ),
    "REQUEST_REFUND": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry for the trouble. Main aapki refund request abhi review ke liye aage badhata hoon.",
            "NEUTRAL": "Ji, main aapki refund request review ke liye aage badhata hoon.",
            "POSITIVE": "Bilkul, main aapki refund request review ke liye aage badhata hoon.",
        },
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID", "AMOUNT"),
        ask="Kis order ka refund hai? Uska order ID aur wajah bata denge?",
    ),
    "RETURN_REQUEST": IntentStrategy(
        replies={
            "NEGATIVE": "Sorry ki aapko return karna pad raha hai. Main return aur pickup mein aapki help karta hoon.",
            "NEUTRAL": "Theek hai, main return aur pickup mein aapki help karta hoon.",
            "POSITIVE": "Bilkul, main return aur pickup mein aapki help karta hoon.",
        },
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID", "PRODUCT_NAME", "ITEM"),
        ask="Order ID aur item mein kya problem hai, bata denge?",
    ),
}

FALLBACK_STRATEGY = IntentStrategy(
    replies={"NEUTRAL": "Ji, main aapki help ke liye yahin hoon."},
    response_type="FALLBACK",
    ask="Thoda detail mein bataiye, kis cheez mein help chahiye?",
)


class ResponseGenerator:
    """Turns a pipeline result into a Hinglish reply.

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
        sentiment = self._tone(nlu)
        priority_label = (priority.get("label") or "MEDIUM").upper()
        routing = priority.get("routing") or ""

        strategy = self.strategies.get(intent, FALLBACK_STRATEGY)

        own_slots = self._referenceable(nlu.get("slots") or [])
        # Only carried-over details this intent actually uses - thanking someone
        # while repeating their old order ID back at them reads like a bot.
        inherited = [
            s for s in self._referenceable(context.get("inherited_slots") or [])
            if s.get("type") in strategy.helpful_slots
        ]
        context_used = bool(context.get("used"))

        needs_detail = bool(strategy.ask) and not self._has_helpful_slot(
            strategy, own_slots, inherited if context_used else []
        )
        requires_human = self._requires_human(intent, priority_label, routing)

        return {
            "response": self._compose(
                strategy=strategy,
                sentiment=sentiment,
                frustrated=self._frustrated(priority),
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
        return f"{name} {slot['value']}" if name else str(slot["value"])

    @classmethod
    def _describe_all(cls, slots: Sequence[Mapping[str, Any]]) -> str:
        """Comma-separated, joined with "aur" before the last."""
        described = [cls._describe(s) for s in slots]
        if len(described) == 1:
            return described[0]
        return ", ".join(described[:-1]) + f" aur {described[-1]}"

    # -------------------------------------------------------------- assembly

    @staticmethod
    def _tone(nlu: Mapping[str, Any]) -> str:
        """Sentiment to reply in; an unsure prediction is treated as neutral."""
        sentiment = nlu.get("sentiment") or DEFAULT_SENTIMENT
        confidence = nlu.get("sentiment_confidence")
        if confidence is not None and float(confidence) < MIN_SENTIMENT_CONFIDENCE:
            return DEFAULT_SENTIMENT
        return sentiment

    @staticmethod
    def _frustrated(priority: Mapping[str, Any]) -> bool:
        """The priority engine saw this customer repeating themselves."""
        return any(
            r.get("rule") in FRUSTRATION_RULES for r in priority.get("rules_fired") or []
        )

    def _compose(
        self,
        strategy: IntentStrategy,
        sentiment: str,
        frustrated: bool,
        priority_label: str,
        own_slots: Sequence[Mapping[str, Any]],
        inherited: Sequence[Mapping[str, Any]],
        needs_detail: bool,
        requires_human: bool,
    ) -> str:
        parts = []
        if sentiment == "NEGATIVE" and frustrated:
            parts.append(FRUSTRATED_OPENER)
        parts.append(strategy.line(sentiment))

        if own_slots:
            described = self._describe_all(own_slots)
            parts.append(f"{described[0].upper()}{described[1:]} note kar liya hai.")
        if inherited:
            # Context is an inference, so confirm it the way a person would.
            parts.append(f"Ye {self._describe_all(inherited)} ke baare mein hai na?")

        if needs_detail and strategy.ask:
            parts.append(strategy.ask)

        parts.append(self._closer(priority_label, requires_human))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _closer(priority_label: str, requires_human: bool) -> str:
        """Only when a person is involved - otherwise the reply just ends.

        Routing language only: the priority engine really did mark the case and
        pick a routing target. Nothing claims a ticket was raised or an agent
        assigned, because the system does neither.
        """
        if priority_label == "HIGH":
            return "Maine ise urgent mark karke team ko bhej diya hai, koi team member khud dekhega."
        if requires_human:
            return "Maine ise team ko bhej diya hai, koi team member khud dekhega."
        return ""

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
