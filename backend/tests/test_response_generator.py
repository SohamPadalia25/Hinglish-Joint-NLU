"""Hinglish response generation: the required cases plus anti-hallucination guards.

The generator answers in Roman-script Hinglish. These tests check the language is
actually Hinglish (not English, not Devanagari, not stilted formal Hindi) while
pinning the decision logic - `response_type`, `requires_human`, slot filtering -
which the language change must not have touched.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from backend.priority.priority_engine import predict_priority
from backend.response.response_generator import (
    INTENT_STRATEGIES,
    FRUSTRATED_OPENER,
    RESPONSE_TYPES,
    ResponseGenerator,
)

GENERATOR = ResponseGenerator()

# Function words that appear in ordinary spoken Hinglish. A reply carrying none
# of these has drifted back to English.
HINGLISH_MARKERS = (
    "aap", "main", "hai", "hoon", "kar", "ke", "ki", "ka", "mein", "please",
    "raha", "gaya", "dijiye", "sakta", "hum",
)


def make_result(
    intent: str,
    sentiment: str,
    slots: Optional[List[Dict[str, Any]]] = None,
    context_used: bool = False,
    inherited: Optional[List[Dict[str, Any]]] = None,
    message: str = "test message",
    sentiment_confidence: float = 0.99,
    rules_fired: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a pipeline-shaped result, scoring priority with the real engine."""
    slots = slots or []
    inherited = inherited or []
    effective = slots + (inherited if context_used else [])
    priority = predict_priority(intent, sentiment, effective, None)
    return {
        "session_id": "test-session",
        "message": message,
        "nlu": {
            "intent": intent,
            "intent_confidence": 0.99,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_confidence,
            "slots": slots,
        },
        "context": {
            "used": context_used,
            "reason": "test",
            "inherited_slots": inherited,
        },
        "priority": {
            "label": priority["priority"],
            "score": priority["priority_score"],
            "routing": priority["routing"],
            "reason": priority["reason"],
            "rules_fired": priority["rules_fired"] if rules_fired is None else rules_fired,
        },
    }


def slot(slot_type: str, value: str, confidence: float = 0.95) -> Dict[str, Any]:
    return {"type": slot_type, "value": value, "confidence": confidence}


def assert_natural_hinglish(response: str) -> None:
    """Roman-script Hinglish, conversational rather than formal."""
    assert response and response.strip() == response
    assert "  " not in response

    lowered = response.lower()
    found = [marker for marker in HINGLISH_MARKERS if marker in lowered]
    assert len(found) >= 3, f"reply does not read as Hinglish: {response!r}"

    # Devanagari is explicitly out of scope - Roman script only.
    assert not re.search(r"[ऀ-ॿ]", response), f"Devanagari in: {response!r}"

    # Stilted formal Hindi the brief rules out.
    for banned in ("mahoday", "bhugtan", "kripya", "samasya", "dhanyavaad", "nivedan"):
        assert banned not in lowered, f"overly formal Hindi {banned!r} in: {response!r}"


# --- the required intent cases ----------------------------------------------

def test_case_1_payment_issue():
    result = make_result("PAYMENT_ISSUE", "NEGATIVE", [slot("AMOUNT", "799 rupaye")])
    reply = GENERATOR.generate(result)

    assert result["priority"]["label"] == "HIGH"
    assert reply["requires_human"] is True
    assert reply["response_type"] == "ESCALATION"
    assert_natural_hinglish(reply["response"])
    assert "payment" in reply["response"].lower()
    assert "799 rupaye" in reply["response"]      # quoted because it is complete


def test_case_2_complaint_delivery():
    reply = GENERATOR.generate(
        make_result("COMPLAINT_DELIVERY", "NEGATIVE", [slot("ORDER_ID", "#45231")])
    )

    assert reply["requires_human"] is True        # COMPLAINT_DELIVERY scores HIGH
    assert_natural_hinglish(reply["response"])
    assert "#45231" in reply["response"]
    assert "sorry" in reply["response"].lower()
    assert "deliver" in reply["response"].lower()


def test_case_3_query_refund_status():
    reply = GENERATOR.generate(make_result("QUERY_REFUND_STATUS", "NEUTRAL"))

    assert reply["requires_human"] is False
    assert reply["response_type"] == "CLARIFICATION"   # no order id to work with
    assert_natural_hinglish(reply["response"])
    assert "order ID" in reply["response"]
    assert "refund" in reply["response"].lower()


def test_case_4_request_cancel():
    reply = GENERATOR.generate(
        make_result("REQUEST_CANCEL", "NEUTRAL", [slot("ORDER_ID", "#45231")])
    )

    assert reply["response_type"] == "ACTION_CONFIRMATION"
    assert reply["requires_human"] is False
    assert_natural_hinglish(reply["response"])
    assert "#45231" in reply["response"]
    assert "cancel" in reply["response"].lower()


def test_case_5_request_refund():
    reply = GENERATOR.generate(make_result("REQUEST_REFUND", "NEUTRAL"))

    assert reply["response_type"] == "CLARIFICATION"
    assert_natural_hinglish(reply["response"])
    assert "order ID" in reply["response"]
    assert "refund" in reply["response"].lower()


def test_case_6_request_escalate():
    reply = GENERATOR.generate(make_result("REQUEST_ESCALATE", "NEGATIVE"))

    assert reply["response_type"] == "ESCALATION"
    assert reply["requires_human"] is True
    assert_natural_hinglish(reply["response"])
    assert "escalate" in reply["response"].lower()


def test_case_7_request_callback():
    reply = GENERATOR.generate(make_result("REQUEST_CALLBACK", "NEUTRAL"))

    # CALLBACK wins over ESCALATION even though routing forces a human.
    assert reply["response_type"] == "CALLBACK"
    assert reply["requires_human"] is True
    assert_natural_hinglish(reply["response"])
    assert "call" in reply["response"].lower()


def test_case_8_feedback_positive():
    reply = GENERATOR.generate(make_result("FEEDBACK_POSITIVE", "POSITIVE"))

    assert reply["response_type"] == "APPRECIATION"
    assert reply["requires_human"] is False
    assert_natural_hinglish(reply["response"])
    assert "thank you" in reply["response"].lower()
    assert "feedback" in reply["response"].lower()


def test_case_9_feedback_negative():
    result = make_result("FEEDBACK_NEGATIVE", "NEGATIVE")
    reply = GENERATOR.generate(result)

    assert result["priority"]["label"] == "HIGH"
    assert reply["requires_human"] is True
    assert_natural_hinglish(reply["response"])
    assert "sorry" in reply["response"].lower()
    assert "feedback" in reply["response"].lower()


def test_case_10_query_product_info():
    reply = GENERATOR.generate(
        make_result("QUERY_PRODUCT_INFO", "NEUTRAL", [slot("PRODUCT_NAME", "iPhone 15")])
    )

    assert reply["response_type"] == "INFORMATION"
    assert reply["requires_human"] is False
    assert_natural_hinglish(reply["response"])
    assert "iPhone 15" in reply["response"]
    assert "product" in reply["response"].lower()


def test_case_11_context_dependent_refund_query():
    """Turn 2 of the cancel -> refund conversation, with the order id carried in."""
    reply = GENERATOR.generate(make_result(
        "QUERY_REFUND_STATUS", "NEUTRAL",
        slots=[],
        context_used=True,
        inherited=[{"type": "ORDER_ID", "value": "#45231", "source": "context",
                    "from_turn": 1}],
        message="Refund kab milega?",
    ))

    assert_natural_hinglish(reply["response"])
    assert "#45231" in reply["response"]
    assert "hai na?" in reply["response"]          # confirmed, not assumed
    assert reply["response_type"] == "INFORMATION"   # nothing left to clarify


def test_case_12_high_priority():
    reply = GENERATOR.generate(make_result("COMPLAINT_DELIVERY", "NEGATIVE"))

    assert reply["requires_human"] is True
    assert_natural_hinglish(reply["response"])
    assert "urgent" in reply["response"].lower()
    assert "koi team member" in reply["response"].lower()


def test_case_13_medium_priority():
    result = make_result("REQUEST_CANCEL", "NEUTRAL", [slot("ORDER_ID", "#45231")])
    reply = GENERATOR.generate(result)

    assert result["priority"]["label"] == "MEDIUM"
    assert reply["requires_human"] is False
    assert_natural_hinglish(reply["response"])
    assert "urgent" not in reply["response"].lower()
    assert "team ko bhej" not in reply["response"].lower()   # no closer when no human


def test_case_14_low_priority():
    result = make_result("FEEDBACK_POSITIVE", "POSITIVE")
    reply = GENERATOR.generate(result)

    assert result["priority"]["label"] == "LOW"
    assert reply["requires_human"] is False
    assert_natural_hinglish(reply["response"])
    assert "team ko bhej" not in reply["response"].lower()


def test_case_15_partial_slot():
    """The model's known failure mode: AMOUNT = 'rupaye' with the number dropped."""
    reply = GENERATOR.generate(make_result(
        "PAYMENT_ISSUE", "NEGATIVE", [slot("AMOUNT", "rupaye")],
        message="Payment 799 rupaye cut ho gayi but order show nahi ho raha",
    ))

    assert "799" not in reply["response"]          # never repaired from the message
    assert "amount rupaye" not in reply["response"].lower()
    assert_natural_hinglish(reply["response"])
    assert "payment" in reply["response"].lower()  # still useful
    assert len(reply["response"]) > 60


# --- coverage and shape ------------------------------------------------------

def test_all_fifteen_intents_have_a_strategy(nlu):
    assert set(INTENT_STRATEGIES) == set(nlu.metadata.intent_labels)
    assert len(INTENT_STRATEGIES) == 15


@pytest.mark.parametrize("intent", sorted(INTENT_STRATEGIES))
@pytest.mark.parametrize("sentiment", ["NEGATIVE", "NEUTRAL", "POSITIVE"])
def test_every_intent_sentiment_pair_produces_a_valid_reply(intent, sentiment):
    reply = GENERATOR.generate(make_result(intent, sentiment))

    assert set(reply) == {"response", "response_type", "requires_human"}
    assert isinstance(reply["response"], str) and len(reply["response"]) > 30
    assert reply["response_type"] in RESPONSE_TYPES
    assert isinstance(reply["requires_human"], bool)
    assert_natural_hinglish(reply["response"])


def test_generation_is_deterministic():
    result = make_result("COMPLAINT_DELIVERY", "NEGATIVE", [slot("ORDER_ID", "#45231")])
    assert GENERATOR.generate(result) == GENERATOR.generate(result)


def test_unknown_intent_falls_back():
    reply = GENERATOR.generate(make_result("NOT_A_REAL_INTENT", "NEUTRAL"))

    assert reply["response_type"] in ("FALLBACK", "CLARIFICATION", "ESCALATION")
    assert_natural_hinglish(reply["response"])


def test_missing_blocks_do_not_crash():
    assert GENERATOR.generate({})["response"]
    assert GENERATOR.generate({"nlu": {}, "context": {}, "priority": {}})["response"]


# --- tone --------------------------------------------------------------------

def test_tone_tracks_sentiment():
    negative = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "NEGATIVE"))["response"]
    neutral = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "NEUTRAL"))["response"]
    positive = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "POSITIVE"))["response"]

    assert negative.lower().startswith("sorry")
    assert len({negative, neutral, positive}) == 3

    # Neutral and positive turns carry no apology at all.
    assert "sorry" not in neutral.lower()
    assert "sorry" not in positive.lower()


@pytest.mark.parametrize("intent", sorted(INTENT_STRATEGIES))
def test_each_intent_words_each_sentiment_differently(intent):
    replies = INTENT_STRATEGIES[intent].replies
    assert set(replies) == {"NEGATIVE", "NEUTRAL", "POSITIVE"}
    assert len(set(replies.values())) == 3


def test_repeat_frustration_gets_stronger_empathy():
    result = make_result("QUERY_REFUND_STATUS", "NEGATIVE",
                         rules_fired=[{"rule": "sustained_negative"}])
    reply = GENERATOR.generate(result)["response"]
    assert reply.startswith(FRUSTRATED_OPENER)
    assert reply.lower().count("sorry") == 1
    assert_natural_hinglish(reply)


def test_unsure_sentiment_replies_neutrally():
    reply = GENERATOR.generate(
        make_result("QUERY_ORDER_STATUS", "NEGATIVE", sentiment_confidence=0.4)
    )["response"]
    assert "sorry" not in reply.lower()


def test_unrelated_carried_over_slots_are_not_mentioned():
    reply = GENERATOR.generate(make_result(
        "FEEDBACK_POSITIVE", "POSITIVE", context_used=True,
        inherited=[{"type": "ORDER_ID", "value": "#45231", "source": "context"}],
    ))["response"]
    assert "#45231" not in reply


@pytest.mark.parametrize("intent", sorted(INTENT_STRATEGIES))
def test_replies_are_short(intent):
    """Worst case - negative, HIGH, needs a detail - still stays brief."""
    reply = GENERATOR.generate(make_result(intent, "NEGATIVE"))["response"]
    assert len(reply) <= 260, f"{intent}: {len(reply)} chars: {reply}"


def test_apologies_are_not_overused():
    """One "sorry" per reply, not a pile of them."""
    for intent in INTENT_STRATEGIES:
        response = GENERATOR.generate(make_result(intent, "NEGATIVE"))["response"]
        assert response.lower().count("sorry") <= 1, response


# --- anti-hallucination ------------------------------------------------------

def test_partial_order_id_slot_is_not_quoted():
    reply = GENERATOR.generate(make_result("COMPLAINT_DELIVERY", "NEGATIVE",
                                           [slot("ORDER_ID", "order")]))

    assert "order ID order" not in reply["response"]
    assert INTENT_STRATEGIES["COMPLAINT_DELIVERY"].ask in reply["response"]


def test_complete_slots_are_quoted_verbatim():
    reply = GENERATOR.generate(make_result(
        "COMPLAINT_DELIVERY", "NEGATIVE",
        [slot("ORDER_ID", "#AB12CD"), slot("DATE", "5 din")],
    ))

    assert "#AB12CD" in reply["response"]
    assert "5 din" in reply["response"]
    assert "aur" in reply["response"]   # two slots joined naturally


def test_no_template_contains_a_digit():
    """Any number in a reply must have come from a slot, not from a template."""
    templates = [FRUSTRATED_OPENER]
    for strategy in INTENT_STRATEGIES.values():
        templates += list(strategy.replies.values()) + [strategy.ask]
    for closer_args in (("HIGH", True), ("MEDIUM", True), ("MEDIUM", False), ("LOW", False)):
        templates.append(ResponseGenerator._closer(*closer_args))

    offenders = [t for t in templates if re.search(r"\d", t or "")]
    assert offenders == [], f"templates must not contain digits: {offenders}"


@pytest.mark.parametrize("intent", sorted(INTENT_STRATEGIES))
def test_no_invented_numbers(intent):
    """Every digit run in a reply must trace back to a slot value."""
    slots = [slot("ORDER_ID", "#45231"), slot("AMOUNT", "799 rupaye")]
    reply = GENERATOR.generate(make_result(intent, "NEGATIVE", slots))

    allowed = " ".join(s["value"] for s in slots)
    for number in re.findall(r"\d+", reply["response"]):
        assert number in allowed, f"{intent}: invented number {number!r}"


@pytest.mark.parametrize("intent", sorted(INTENT_STRATEGIES))
def test_never_states_status_dates_or_outcomes_in_english(intent):
    forbidden = re.compile(
        r"\b(delivered on|will arrive|arriving on|has been refunded|refund of|"
        r"credited to|tracking (id|number) is|your balance|policy states|"
        r"within \w+ (hours|days)|business days)\b",
        re.I,
    )
    for sentiment in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
        response = GENERATOR.generate(make_result(intent, sentiment))["response"]
        assert not forbidden.search(response), f"{intent}/{sentiment}: {response}"


@pytest.mark.parametrize("intent", sorted(INTENT_STRATEGIES))
def test_never_states_status_dates_or_outcomes_in_hinglish(intent):
    """The Hinglish equivalents of every claim the English guard blocks.

    Switching language must not open a hole in the safety net: a template could
    now promise an outcome in Hindi that the English regex would never catch.
    """
    forbidden = re.compile(
        r"(refund (ho gaya|ho chuka|aa gaya|mil gaya|kar diya)|"
        r"deliver (ho gaya|ho jayega|ho chuka)|"
        r"credit (ho gaya|ho jayega|kar diya)|"
        r"paise (aa jayenge|aa gaye|mil jayenge)|"
        r"cancel (ho gaya|ho chuka|kar diya gaya)|"
        r"\bkal tak\b|\baaj tak\b|\bghante mein\b|\bdin mein\b|\bhafte mein\b|"
        r"\bticket (raise|create) (kar|ho)|"
        r"\bagent assign\b|\bpolicy ke hisaab se\b)",
        re.I,
    )
    for sentiment in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
        response = GENERATOR.generate(make_result(intent, sentiment))["response"]
        assert not forbidden.search(response), f"{intent}/{sentiment}: {response}"


def test_routing_language_does_not_claim_a_ticket():
    """The system routes; it does not raise tickets. Replies must not say it does."""
    for intent in INTENT_STRATEGIES:
        for sentiment in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
            response = GENERATOR.generate(make_result(intent, sentiment))["response"].lower()
            assert "ticket raise" not in response
            assert "ticket create" not in response
            assert "ticket number" not in response
            assert "agent assign" not in response


def test_asks_for_detail_when_nothing_useful_was_extracted():
    reply = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "NEUTRAL"))

    assert reply["response_type"] == "CLARIFICATION"
    assert "?" in reply["response"] or "please" in reply["response"].lower()


def test_does_not_ask_when_the_detail_is_already_present():
    reply = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "NEUTRAL",
                                           [slot("ORDER_ID", "#45231")]))

    assert reply["response_type"] == "INFORMATION"
    assert INTENT_STRATEGIES["QUERY_ORDER_STATUS"].ask not in reply["response"]


def test_inherited_slots_only_count_when_context_was_used():
    inherited = [{"type": "ORDER_ID", "value": "#45231", "source": "context"}]

    used = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "NEUTRAL",
                                          context_used=True, inherited=inherited))
    unused = GENERATOR.generate(make_result("QUERY_ORDER_STATUS", "NEUTRAL",
                                            context_used=False, inherited=inherited))

    assert "#45231" in used["response"]
    assert "#45231" not in unused["response"]
    assert unused["response_type"] == "CLARIFICATION"


# --- integration with the real model ----------------------------------------

def test_end_to_end_on_the_real_pipeline(pipeline, session_id):
    result = pipeline.process_message(
        session_id, "Payment 799 rupaye cut ho gayi but order show nahi ho raha"
    )
    reply = GENERATOR.generate(result)

    assert result["nlu"]["intent"] == "PAYMENT_ISSUE"
    assert reply["requires_human"] is True
    assert reply["response_type"] == "ESCALATION"
    assert_natural_hinglish(reply["response"])
    # The model returns AMOUNT="rupaye" here, so the number must not appear.
    assert "799" not in reply["response"]


def test_end_to_end_context_conversation(pipeline, session_id):
    pipeline.process_message(session_id, "Maine order #45231 cancel kiya tha")
    turn_2 = pipeline.process_message(session_id, "Refund kab milega?")
    reply = GENERATOR.generate(turn_2)

    assert turn_2["context"]["used"] is True
    assert_natural_hinglish(reply["response"])
    assert "#45231" in reply["response"]
    assert reply["requires_human"] is False


def test_end_to_end_english_input_still_answers_in_hinglish(pipeline, session_id):
    """Input language does not switch the reply language - it stays Hinglish."""
    result = pipeline.process_message(session_id, "Can you tell me my order status?")
    reply = GENERATOR.generate(result)

    assert_natural_hinglish(reply["response"])
