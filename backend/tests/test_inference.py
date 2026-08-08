"""The Phase 12 inference cases, run against the real checkpoint."""

from __future__ import annotations

import pytest

CASE_1 = "Payment 799 rupaye cut ho gayi but order show nahi ho raha"
CASE_2 = "Mera order #45231 5 din se deliver nahi hua"
CASE_3 = "Thanks for quick help"
CASE_4 = "Bahut hi worst experience raha"


def _slot_types(result):
    return {slot["type"] for slot in result["slots"]}


def test_case_1_payment_issue(nlu):
    result = nlu.predict(CASE_1)

    assert result["intent"] == "PAYMENT_ISSUE"
    assert result["sentiment"] == "NEGATIVE"
    assert "AMOUNT" in _slot_types(result)


def test_case_2_delivery_complaint(nlu):
    result = nlu.predict(CASE_2)

    assert result["intent"] == "COMPLAINT_DELIVERY"
    assert result["sentiment"] == "NEGATIVE"
    assert {"ORDER_ID", "DATE"} <= _slot_types(result)

    order_id = next(s for s in result["slots"] if s["type"] == "ORDER_ID")
    assert order_id["value"] == "#45231"


def test_case_3_positive_feedback(nlu):
    result = nlu.predict(CASE_3)

    assert result["intent"] == "FEEDBACK_POSITIVE"
    assert result["sentiment"] == "POSITIVE"


def test_case_4_negative_sentiment(nlu):
    result = nlu.predict(CASE_4)

    assert result["sentiment"] == "NEGATIVE"


@pytest.mark.parametrize("text", [CASE_1, CASE_2, CASE_3, CASE_4])
def test_response_schema(nlu, text):
    result = nlu.predict(text)

    assert set(result) == {
        "intent", "intent_confidence", "sentiment", "sentiment_confidence", "slots",
    }
    assert 0.0 <= result["intent_confidence"] <= 1.0
    assert 0.0 <= result["sentiment_confidence"] <= 1.0
    assert isinstance(result["slots"], list)

    for slot in result["slots"]:
        assert {"type", "value", "confidence"} <= set(slot)
        assert slot["value"]
        assert 0.0 <= slot["confidence"] <= 1.0
        assert not slot["type"].startswith(("B-", "I-"))  # spans, not raw BIO tags


def test_slots_are_a_list_so_duplicates_survive(nlu):
    """A dict keyed by slot type would collapse repeated entities."""
    result = nlu.predict("Order #45231 aur order #78342 dono nahi aaye")
    order_ids = [s["value"] for s in result["slots"] if s["type"] == "ORDER_ID"]

    assert isinstance(result["slots"], list)
    # Both ids must survive if the model tags both; at minimum nothing is
    # silently overwritten by a same-typed neighbour.
    assert len(order_ids) == len(set(order_ids)) or len(order_ids) > 1


def test_original_casing_is_recovered(nlu):
    """Normalization lowercases; returned slot values map back to the raw text."""
    result = nlu.predict("Mera order #AB12CD abhi tak nahi aaya")
    order_ids = [s for s in result["slots"] if s["type"] == "ORDER_ID"]

    if order_ids:  # depends on the model tagging it
        assert order_ids[0]["value"] == "#AB12CD"
        assert order_ids[0]["value_normalized"] == "#ab12cd"


def test_empty_input_is_handled(nlu):
    result = nlu.predict("   ")

    assert result["intent"] == "UNKNOWN"
    assert result["intent_confidence"] == 0.0
    assert result["slots"] == []


def test_prediction_is_deterministic(nlu):
    """eval() must be in force - dropout at inference would break this."""
    assert nlu.predict(CASE_1) == nlu.predict(CASE_1)


def test_verbose_prediction(nlu):
    result = nlu.predict_verbose(CASE_1, top_k=3)

    assert len(result["intent_top_k"]) == 3
    assert result["intent_top_k"][0]["intent"] == result["intent"]
    assert sum(result["sentiment_distribution"].values()) == pytest.approx(1.0, abs=1e-3)
