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
    "DATE": "timeframe",
    "PROMO_CODE": "promo code",
    "CONTACT": "contact number",
    "REASON": "reason",
    "ISSUE": "issue",
}

# Empathy line, used only where it belongs. It is a standalone sentence rather
# than a comma-prefix because "Sorry, aap refund request karna chahte hain" reads
# wrong - the apology is about the situation, not about what the user asked for.
# Neutral and positive turns need no separate opener; their lead word carries the
# tone. Exactly one apology per reply - the brief warns against piling them up.
SENTIMENT_OPENERS: Mapping[str, str] = {
    "NEGATIVE": "Sorry, aapko is wajah se pareshani hui.",
    "NEUTRAL": "",
    "POSITIVE": "",
}

# Lead word for the acknowledgement sentence. The acknowledgement itself is a
# lower-case clause written to follow this and a comma.
ACKNOWLEDGEMENT_LEADS: Mapping[str, str] = {
    "NEGATIVE": "Samajh gaya",
    "NEUTRAL": "Samajh gaya",     # direct and professional
    "POSITIVE": "Thank you",      # warm
}

DEFAULT_SENTIMENT = "NEUTRAL"


@dataclass(frozen=True)
class IntentStrategy:
    """How one intent is answered, in Hinglish.

    Attributes:
        acknowledgement: a lower-case clause completing "<prefix>, ...". States
            what the user said back to them, never a claim about system state.
        action: what support can do next. Phrased as capability ("check karne
            mein help karta hoon"), never as a completed outcome ("aapka refund
            ho gaya hai").
        response_type: the disposition when no override applies.
        helpful_slots: slot types that would let support act. When none of these
            are available - freshly extracted or carried in from context - the
            reply asks for them.
        ask: the question to ask when `helpful_slots` are missing.
    """

    acknowledgement: str
    action: str
    response_type: str
    helpful_slots: Tuple[str, ...] = ()
    ask: str = ""


# One strategy per intent label in model_metadata2.json. No template contains a
# digit, a date, a status or a policy statement - see test_no_template_contains_a_digit.
INTENT_STRATEGIES: Mapping[str, IntentStrategy] = {
    "COMPLAINT_BILLING": IntentStrategy(
        acknowledgement="aapki billing mein kuch galat charge hua hai",
        action="Main aapke account ke charges review karke billing error correct "
               "karwane mein aapki help karta hoon.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "AMOUNT", "ACTUAL_AMOUNT"),
        ask="Please apna order ID ya invoice number aur jo amount charge hui hai "
            "wo share kar dijiye.",
    ),
    "COMPLAINT_DELIVERY": IntentStrategy(
        acknowledgement="aapka order abhi tak deliver nahi hua hai",
        action="Main courier ke saath shipment check karke delivery follow up "
               "karne mein aapki help karta hoon.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID",),
        ask="Please apna order ID share kar dijiye taaki hum shipment track kar sakein.",
    ),
    "COMPLAINT_PRODUCT": IntentStrategy(
        acknowledgement="jo product aapko mila wo theek nahi nikla",
        action="Main item review karwa ke replacement ya return arrange karne mein "
               "aapki help karta hoon.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "PRODUCT_NAME", "ITEM"),
        ask="Please apna order ID aur item ki ek photo share kar dijiye.",
    ),
    "COMPLAINT_SERVICE": IntentStrategy(
        acknowledgement="aapko ab tak jo support mila wo theek nahi tha",
        action="Main check karwata hoon ki ye case kaise handle hua, aur isse "
               "sahi tarike se aage badhwata hoon.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID",),
        ask="Agar aapke paas pichhli complaint ya ticket reference hai to please "
            "share kar dijiye, usse trace karna aasan ho jayega.",
    ),
    "FEEDBACK_NEGATIVE": IntentStrategy(
        acknowledgement="aapka experience humare saath achha nahi raha",
        action="Aapka feedback note kar liya gaya hai aur related team tak "
               "pahunchaya jayega.",
        response_type="APOLOGY",
        helpful_slots=("ORDER_ID",),
        ask="Agar iske peeche koi specific order ya interaction hai to bataiye, "
            "main usse check karwata hoon.",
    ),
    "FEEDBACK_POSITIVE": IntentStrategy(
        acknowledgement="aapka feedback humare liye bahut important hai",
        action="Main ye team tak zaroor pahunchata hoon - unhe sunkar bahut achha lagega.",
        response_type="APPRECIATION",
    ),
    "PAYMENT_ISSUE": IntentStrategy(
        acknowledgement="aapki payment theek se process nahi hui",
        action="Main payment aur order status dono check karne mein aapki help "
               "karta hoon, aur deduct hui ya fail hui payment trace karwa sakta hoon.",
        response_type="ACKNOWLEDGEMENT",
        helpful_slots=("ORDER_ID", "AMOUNT", "PAYMENT_METHOD"),
        ask="Please apna order ID aur bank ka payment ya UPI reference share kar dijiye.",
    ),
    "QUERY_ORDER_STATUS": IntentStrategy(
        acknowledgement="aap apne order ka update chahte hain",
        action="Main aapke order ka current status check karne mein help karta hoon.",
        response_type="INFORMATION",
        helpful_slots=("ORDER_ID",),
        ask="Please apna order ID share kar dijiye.",
    ),
    "QUERY_PRODUCT_INFO": IntentStrategy(
        acknowledgement="aap kisi product ke baare mein details chahte hain",
        action="Main aapko us product ki specifications, availability aur pricing "
               "bata sakta hoon jo humare record mein hai.",
        response_type="INFORMATION",
        helpful_slots=("PRODUCT_NAME", "PRODUCT_TYPE", "ITEM"),
        ask="Please bataiye aap kaunse product ke baare mein jaanna chahte hain.",
    ),
    "QUERY_REFUND_STATUS": IntentStrategy(
        acknowledgement="aap jaanna chahte hain ki aapka refund kahan tak pahuncha hai",
        action="Main aapke order ke against refund check karke bata sakta hoon ki "
               "wo abhi kis stage par hai.",
        response_type="INFORMATION",
        helpful_slots=("ORDER_ID",),
        ask="Please us order ka order ID share kar dijiye jiska refund hai.",
    ),
    "REQUEST_CALLBACK": IntentStrategy(
        acknowledgement="aap chahte hain ki humari team aapko call kare",
        action="Aapki callback request note kar li gayi hai aur support team ke "
               "liye route kar di gayi hai.",
        response_type="CALLBACK",
        helpful_slots=("CONTACT",),
        ask="Please confirm kar dijiye ki kis number par call karna hai aur kaunsa "
            "time aapke liye convenient rahega.",
    ),
    "REQUEST_CANCEL": IntentStrategy(
        acknowledgement="aap apna order cancel karna chahte hain",
        action="Order confirm hote hi hum cancellation request process kar sakte hain.",
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID",),
        ask="Please wo order ID share kar dijiye jise aap cancel karna chahte hain.",
    ),
    "REQUEST_ESCALATE": IntentStrategy(
        acknowledgement="aapko is issue ko aage escalate karna pada",
        action="Main ise senior support specialist ke liye escalate kar raha hoon.",
        response_type="ESCALATION",
        helpful_slots=("ORDER_ID",),
        ask="Agar aapke paas order ID ya pehle ki koi complaint reference hai to "
            "please share kar dijiye, isse kaafi tezi aa jayegi.",
    ),
    "REQUEST_REFUND": IntentStrategy(
        acknowledgement="aap refund request karna chahte hain",
        action="Main refund request raise karke usse aapke order ke against review "
               "karwane mein help karta hoon.",
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID", "AMOUNT"),
        ask="Please apna order ID aur refund ki reason share kar dijiye.",
    ),
    "RETURN_REQUEST": IntentStrategy(
        acknowledgement="aap product return karna chahte hain",
        action="Order confirm hote hi hum return start karke pickup arrange kar sakte hain.",
        response_type="ACTION_CONFIRMATION",
        helpful_slots=("ORDER_ID", "PRODUCT_NAME", "ITEM"),
        ask="Please apna order ID aur item mein kya problem hai wo bata dijiye.",
    ),
}

FALLBACK_STRATEGY = IntentStrategy(
    acknowledgement="aapko kisi cheez mein help chahiye",
    action="Main chahta hoon ki aapki ye request sahi jagah pahunche.",
    response_type="FALLBACK",
    ask="Please thoda aur detail mein bataiye ki aapko kis cheez mein help chahiye.",
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
        """Comma-separated, joined with "aur" before the last."""
        described = [cls._describe(s) for s in slots]
        if len(described) == 1:
            return described[0]
        return ", ".join(described[:-1]) + f" aur {described[-1]}"

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
        lead = ACKNOWLEDGEMENT_LEADS.get(sentiment, ACKNOWLEDGEMENT_LEADS[DEFAULT_SENTIMENT])
        parts = [
            SENTIMENT_OPENERS.get(sentiment, ""),
            f"{lead}, {strategy.acknowledgement}.",
        ]

        if own_slots:
            parts.append(f"Maine {self._describe_all(own_slots)} note kar liya hai.")

        if inherited:
            parts.append(
                f"Main aapke pichhle message se {self._describe_all(inherited)} "
                "aage use kar raha hoon."
            )

        parts.append(strategy.action)

        if needs_detail and strategy.ask:
            parts.append(strategy.ask)

        parts.append(self._closer(priority_label, requires_human))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _closer(priority_label: str, requires_human: bool) -> str:
        """Closing line.

        Routing language only: the priority engine really did mark the case and
        pick a routing target, so saying so is accurate. Nothing here claims a
        ticket was raised or an agent assigned, because the system does neither.
        """
        if priority_label == "HIGH":
            return ("Aapke is issue ko high priority par mark kiya gaya hai aur "
                    "human support agent ke liye route kiya gaya hai.")
        if requires_human:
            return "Ye request human support agent ke liye route ki gayi hai."
        if priority_label == "LOW":
            return "Aur kuch chahiye to bataiye, main yahin help kar deta hoon."
        return "Main is issue ko support team ke follow-up ke liye note kar raha hoon."

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
