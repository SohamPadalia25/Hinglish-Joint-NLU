"""Conversation memory and context-aware interpretation."""

from __future__ import annotations

from backend.context.context_interpreter import ContextInterpreter
from backend.context.context_manager import ConversationContext

NLU_CANCEL = {
    "intent": "REQUEST_CANCEL", "intent_confidence": 0.97,
    "sentiment": "NEUTRAL", "sentiment_confidence": 0.9,
    "slots": [{"type": "ORDER_ID", "value": "#45231", "confidence": 0.98}],
}
NLU_FOLLOW_UP = {
    "intent": "QUERY_REFUND_STATUS", "intent_confidence": 0.99,
    "sentiment": "NEUTRAL", "sentiment_confidence": 0.95,
    "slots": [],
}


# --- ConversationContext -----------------------------------------------------

def test_history_is_bounded():
    ctx = ConversationContext(max_turns=3)
    sid = ctx.create_session()
    for i in range(6):
        ctx.add_turn(sid, f"message {i}", NLU_CANCEL)

    history = ctx.get_history(sid)
    assert len(history) == 3
    assert [t["user_message"] for t in history] == ["message 3", "message 4", "message 5"]
    assert [t["turn_index"] for t in history] == [4, 5, 6]  # counter keeps counting


def test_clear_session():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "hi", NLU_CANCEL)

    assert ctx.clear_session(sid) is True
    assert ctx.get_history(sid) == []
    assert ctx.clear_session(sid) is False  # already gone


def test_sessions_are_isolated():
    ctx = ConversationContext()
    a, b = ctx.create_session(), ctx.create_session()
    ctx.add_turn(a, "only in a", NLU_CANCEL)

    assert len(ctx.get_history(a)) == 1
    assert ctx.get_history(b) == []


def test_relevant_context_is_empty_for_a_new_session():
    ctx = ConversationContext()
    relevant = ctx.get_relevant_context(ctx.create_session())

    assert relevant["turn_count"] == 0
    assert relevant["previous_intent"] is None
    assert relevant["known_slots"] == []


def test_relevant_context_summarises_history():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "Maine order cancel kiya tha", NLU_CANCEL)

    relevant = ctx.get_relevant_context(sid)
    assert relevant["previous_intent"] == "REQUEST_CANCEL"
    assert relevant["turn_count"] == 1
    assert [s["type"] for s in relevant["known_slots"]] == ["ORDER_ID"]


def test_newer_slot_value_wins():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "order 1", NLU_CANCEL)
    ctx.add_turn(sid, "order 2", {
        **NLU_CANCEL,
        "slots": [{"type": "ORDER_ID", "value": "#99999", "confidence": 0.9}],
    })

    known = ctx.get_relevant_context(sid)["known_slots"]
    assert [s["value"] for s in known] == ["#99999"]


# --- ContextInterpreter ------------------------------------------------------

def test_no_context_on_the_first_turn():
    ctx = ConversationContext()
    sid = ctx.create_session()
    result = ContextInterpreter().interpret(
        "Maine order cancel kiya tha", NLU_CANCEL, ctx.get_relevant_context(sid),
    )

    assert result["context_used"] is False
    assert "No prior turns" in result["context_reason"]


def test_self_contained_query_ignores_history():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "Maine order cancel kiya tha", NLU_CANCEL)

    long_confident = {
        "intent": "PAYMENT_ISSUE", "intent_confidence": 0.99,
        "sentiment": "NEGATIVE", "sentiment_confidence": 0.99,
        "slots": [{"type": "AMOUNT", "value": "799 rupaye", "confidence": 0.95}],
    }
    result = ContextInterpreter().interpret(
        "Payment 799 rupaye cut ho gayi lekin order show nahi ho raha hai",
        long_confident, ctx.get_relevant_context(sid),
    )

    assert result["context_used"] is False
    assert "self-contained" in result["context_reason"]


def test_short_follow_up_uses_context():
    """The Phase 12 case 5 scenario, at the interpreter level."""
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "Maine order cancel kiya tha.", NLU_CANCEL)

    result = ContextInterpreter().interpret(
        "Refund kab milega?", NLU_FOLLOW_UP, ctx.get_relevant_context(sid),
    )

    assert result["context_used"] is True
    assert result["previous_intent"] == "REQUEST_CANCEL"
    assert len(result["relevant_history"]) == 1

    signals = {s["signal"] for s in result["signals"]}
    assert "short_query" in signals
    assert "follow_up_marker" in signals


def test_inherited_slots_are_marked_as_coming_from_context():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "Maine order #45231 cancel kiya tha", NLU_CANCEL)

    result = ContextInterpreter().interpret(
        "Refund kab milega?", NLU_FOLLOW_UP, ctx.get_relevant_context(sid),
    )

    inherited = result["inherited_slots"]
    assert [s["type"] for s in inherited] == ["ORDER_ID"]
    assert inherited[0]["value"] == "#45231"
    assert inherited[0]["source"] == "context"  # never presented as freshly extracted


def test_slots_present_in_the_query_are_not_inherited():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "Maine order #45231 cancel kiya tha", NLU_CANCEL)

    restated = {**NLU_FOLLOW_UP,
                "slots": [{"type": "ORDER_ID", "value": "#78342", "confidence": 0.9}]}
    result = ContextInterpreter().interpret("Aur #78342 ka kya", restated,
                                            ctx.get_relevant_context(sid))

    assert all(s["type"] != "ORDER_ID" for s in result["inherited_slots"])


def test_low_confidence_triggers_context():
    ctx = ConversationContext()
    sid = ctx.create_session()
    ctx.add_turn(sid, "Maine order cancel kiya tha", NLU_CANCEL)

    unsure = {**NLU_FOLLOW_UP, "intent_confidence": 0.31}
    result = ContextInterpreter().interpret(
        "kuch samajh nahi aa raha mujhe iske baare mein bilkul bhi", unsure,
        ctx.get_relevant_context(sid),
    )

    assert result["context_used"] is True
    assert "low_intent_confidence" in {s["signal"] for s in result["signals"]}
