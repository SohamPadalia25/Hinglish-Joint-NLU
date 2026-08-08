"""The rule-based priority / routing layer.

These tests pin the rules to the dataset they were derived from, so retuning a
weight without re-checking the consequences fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.priority.priority_engine import PriorityEngine, predict_priority

DATASET = Path(__file__).resolve().parents[2] / "hinglish_support_dataset_expanded_final.tsv"

# What the engine should output for each intent paired with its modal sentiment
# in the dataset. All of these match the dataset's own majority priority label
# except REQUEST_REFUND - see test_matches_dataset_majority below.
EXPECTED = {
    ("COMPLAINT_BILLING", "NEGATIVE"): "HIGH",
    ("COMPLAINT_DELIVERY", "NEGATIVE"): "HIGH",
    ("COMPLAINT_PRODUCT", "NEGATIVE"): "HIGH",
    ("COMPLAINT_SERVICE", "NEGATIVE"): "HIGH",
    ("FEEDBACK_NEGATIVE", "NEGATIVE"): "HIGH",
    ("FEEDBACK_POSITIVE", "POSITIVE"): "LOW",
    ("PAYMENT_ISSUE", "NEGATIVE"): "HIGH",
    ("QUERY_ORDER_STATUS", "NEUTRAL"): "MEDIUM",
    ("QUERY_PRODUCT_INFO", "NEUTRAL"): "LOW",
    ("QUERY_REFUND_STATUS", "NEUTRAL"): "MEDIUM",
    ("REQUEST_CALLBACK", "NEUTRAL"): "MEDIUM",
    ("REQUEST_CANCEL", "NEUTRAL"): "MEDIUM",
    ("REQUEST_ESCALATE", "NEGATIVE"): "HIGH",
    # Deliberate deviation: the dataset splits this 74 HIGH / 100 MEDIUM, and the
    # engine escalates an *angry* refund request. Neutral ones stay MEDIUM.
    ("REQUEST_REFUND", "NEGATIVE"): "HIGH",
    ("RETURN_REQUEST", "NEGATIVE"): "HIGH",
}


@pytest.mark.parametrize("key, expected", sorted(EXPECTED.items()))
def test_intent_sentiment_pairs(key, expected):
    intent, sentiment = key
    assert predict_priority(intent, sentiment, [], None)["priority"] == expected


def test_matches_dataset_majority_for_at_least_14_of_15_intents():
    """The base scores come from this file; verify they still reproduce it."""
    pandas = pytest.importorskip("pandas")
    if not DATASET.is_file():
        pytest.skip("dataset not present")

    df = pandas.read_csv(DATASET, sep="\t")
    agree, disagree = 0, []
    for intent, sub in df.groupby("intent"):
        modal_sentiment = sub["sentiment"].mode()[0]
        majority_priority = sub["priority"].mode()[0]
        predicted = predict_priority(intent, modal_sentiment, [], None)["priority"]
        if predicted == majority_priority:
            agree += 1
        else:
            disagree.append((intent, majority_priority, predicted))

    assert agree >= 14, f"only {agree}/15 intents agree; mismatches: {disagree}"
    assert [d[0] for d in disagree] in ([], ["REQUEST_REFUND"])


def test_neutral_refund_request_stays_medium():
    assert predict_priority("REQUEST_REFUND", "NEUTRAL", [], None)["priority"] == "MEDIUM"


def test_escalation_always_routes_to_a_human():
    result = predict_priority("REQUEST_ESCALATE", "NEGATIVE", [], None)

    assert result["priority"] == "HIGH"
    assert result["routing"] == "HUMAN_AGENT"
    assert result["priority_score"] == 1.0


def test_callback_routes_to_a_human_despite_medium_priority():
    result = predict_priority("REQUEST_CALLBACK", "NEUTRAL", [], None)

    assert result["priority"] == "MEDIUM"
    assert result["routing"] == "HUMAN_AGENT"
    assert "force_human_routing" in {r["rule"] for r in result["rules_fired"]}


def test_routing_by_priority():
    assert predict_priority("COMPLAINT_DELIVERY", "NEGATIVE")["routing"] == "HUMAN_AGENT"
    assert predict_priority("QUERY_ORDER_STATUS", "NEUTRAL")["routing"] == "ASSISTED_BOT"
    assert predict_priority("FEEDBACK_POSITIVE", "POSITIVE")["routing"] == "AUTO_BOT"


def test_urgency_slots_raise_the_score():
    plain = predict_priority("QUERY_ORDER_STATUS", "NEUTRAL", [])
    urgent = predict_priority(
        "QUERY_ORDER_STATUS", "NEUTRAL",
        [{"type": "URGENCY", "value": "urgent"}, {"type": "SLA", "value": "48 hours"}],
    )

    assert urgent["priority_score"] > plain["priority_score"]
    assert "urgency_slots" in {r["rule"] for r in urgent["rules_fired"]}


def test_repeating_an_intent_raises_the_score():
    context = {"recent_intents": ["COMPLAINT_SERVICE"], "recent_sentiments": ["NEGATIVE"]}
    repeated = predict_priority("COMPLAINT_SERVICE", "NEGATIVE", [], context)

    assert "repeated_intent" in {r["rule"] for r in repeated["rules_fired"]}


def test_sustained_negativity_raises_the_score():
    calm = predict_priority("QUERY_REFUND_STATUS", "NEUTRAL", [], {})
    frustrated = predict_priority(
        "QUERY_REFUND_STATUS", "NEUTRAL", [],
        {"recent_intents": [], "recent_sentiments": ["NEGATIVE", "NEGATIVE", "NEGATIVE"]},
    )

    assert frustrated["priority_score"] > calm["priority_score"]
    assert "sustained_negative" in {r["rule"] for r in frustrated["rules_fired"]}


def test_score_is_clamped_to_the_unit_interval():
    hot = predict_priority(
        "REQUEST_ESCALATE", "NEGATIVE",
        [{"type": "THREAT", "value": "consumer court"}, {"type": "SLA", "value": "48h"}],
        {"recent_intents": ["REQUEST_ESCALATE"], "recent_sentiments": ["NEGATIVE"] * 3},
    )
    cold = predict_priority("FEEDBACK_POSITIVE", "POSITIVE", [], {})

    assert hot["priority_score"] == 1.0
    assert cold["priority_score"] == 0.0


def test_unknown_intent_falls_back_without_crashing():
    result = predict_priority("NOT_A_REAL_INTENT", "NEUTRAL", [], None)

    assert result["priority"] == "MEDIUM"
    assert "unrecognised intent" in result["reason"]


def test_every_decision_is_explained():
    result = predict_priority("COMPLAINT_DELIVERY", "NEGATIVE", [], None)

    assert result["reason"]
    assert result["rules_fired"]
    assert set(result) >= {"priority", "priority_score", "routing", "reason"}


def test_rules_are_configurable():
    from backend.priority.priority_engine import PriorityRules

    strict = PriorityEngine(PriorityRules(high_threshold=0.20, medium_threshold=0.10))
    assert strict.predict_priority("QUERY_PRODUCT_INFO", "NEUTRAL", [], None)["priority"] == "HIGH"
