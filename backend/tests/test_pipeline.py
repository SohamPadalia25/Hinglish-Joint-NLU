"""End-to-end pipeline behaviour, including the Phase 12 context case."""

from __future__ import annotations

import pytest


def test_result_schema(pipeline, session_id):
    result = pipeline.process_message(session_id, "Mera order #45231 5 din se deliver nahi hua")

    assert set(result) >= {"session_id", "message", "nlu", "context", "priority"}
    assert set(result["nlu"]) == {
        "intent", "intent_confidence", "sentiment", "sentiment_confidence", "slots",
    }
    assert set(result["context"]) >= {"used", "reason"}
    assert set(result["priority"]) >= {"label", "score", "routing", "reason"}
    assert result["session_id"] == session_id


def test_first_turn_never_uses_context(pipeline, session_id):
    result = pipeline.process_message(session_id, "Mera order #45231 5 din se deliver nahi hua")

    assert result["context"]["used"] is False
    assert "No prior turns" in result["context"]["reason"]
    assert result["nlu"]["intent"] in pipeline.nlu.metadata.intent_labels


def test_case_5_context_carries_across_turns(pipeline, session_id):
    """Phase 12 case 5.

    Turn 1: "Maine order cancel kiya tha."   (self-contained)
    Turn 2: "Refund kab milega?"             (short follow-up -> uses context)
    """
    turn_1 = pipeline.process_message(session_id, "Maine order cancel kiya tha.")
    turn_2 = pipeline.process_message(session_id, "Refund kab milega?")

    assert turn_1["context"]["used"] is False

    assert turn_2["context"]["used"] is True
    assert turn_2["context"]["previous_intent"] == turn_1["nlu"]["intent"]
    assert turn_2["context"]["reason"]
    assert len(turn_2["context"]["relevant_history"]) >= 1
    assert turn_2["context"]["relevant_history"][0]["user_message"] == "Maine order cancel kiya tha."


def test_history_accumulates_in_the_session(pipeline, session_id):
    pipeline.process_message(session_id, "Mera order #45231 5 din se deliver nahi hua")
    pipeline.process_message(session_id, "Koi update mila kya")

    info = pipeline.get_session(session_id)
    assert info["total_turns"] == 2
    assert [t["user_message"] for t in info["history"]][0].startswith("Mera order")


def test_repeated_complaints_escalate_priority(pipeline, session_id):
    first = pipeline.process_message(session_id, "Service bahut kharab hai koi sun nahi raha")
    second = pipeline.process_message(session_id, "Service bahut kharab hai koi sun nahi raha")

    assert second["priority"]["score"] >= first["priority"]["score"]


def test_stateless_predict_has_no_context_block(pipeline):
    result = pipeline.predict("Payment 799 rupaye cut ho gayi but order show nahi ho raha")

    assert "context" not in result
    assert result["nlu"]["intent"] == "PAYMENT_ISSUE"
    assert result["priority"]["label"] == "HIGH"


def test_sessions_do_not_leak_into_each_other(pipeline):
    a = pipeline.create_session()
    b = pipeline.create_session()
    try:
        pipeline.process_message(a, "Maine order cancel kiya tha.")
        result = pipeline.process_message(b, "Refund kab milega?")

        assert result["context"]["used"] is False  # session b has no history
    finally:
        pipeline.clear_session(a)
        pipeline.clear_session(b)


def test_empty_message_is_rejected(pipeline, session_id):
    with pytest.raises(ValueError):
        pipeline.process_message(session_id, "   ")


def test_session_is_created_when_omitted(pipeline):
    result = pipeline.process_message(None, "Thanks for quick help")
    try:
        assert result["session_id"]
        assert pipeline.get_session(result["session_id"]) is not None
    finally:
        pipeline.clear_session(result["session_id"])
