"""HTTP surface, exercised through FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app


@pytest.fixture(scope="module")
def client(pipeline):  # `pipeline` first so the model is loaded before startup
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["model_name"] == "xlm-roberta-large"
    assert body["model"]["parameters"] == 560_078_007


def test_predict(client):
    response = client.post(
        "/api/predict",
        json={"text": "Payment 799 rupaye cut ho gayi but order show nahi ho raha"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["nlu"]["intent"] == "PAYMENT_ISSUE"
    assert body["nlu"]["sentiment"] == "NEGATIVE"
    assert "AMOUNT" in {s["type"] for s in body["nlu"]["slots"]}
    assert body["priority"]["label"] == "HIGH"
    assert body["priority"]["routing"] == "HUMAN_AGENT"


def test_predict_verbose(client):
    response = client.post(
        "/api/predict", json={"text": "Mera order #45231 5 din se deliver nahi hua", "verbose": True},
    )

    assert response.status_code == 200
    assert len(response.json()["nlu"]["intent_top_k"]) == 3


def test_predict_rejects_empty_text(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 422


def test_session_lifecycle(client):
    created = client.post("/api/session", json={})
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    fetched = client.get(f"/api/session/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["total_turns"] == 0

    deleted = client.delete(f"/api/session/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json()["cleared"] is True

    assert client.get(f"/api/session/{session_id}").status_code == 404
    assert client.delete(f"/api/session/{session_id}").status_code == 404


def test_chat_creates_a_session_when_omitted(client):
    response = client.post("/api/chat", json={"message": "Thanks for quick help"})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["nlu"]["intent"] == "FEEDBACK_POSITIVE"
    assert body["priority"]["label"] == "LOW"

    client.delete(f"/api/session/{body['session_id']}")


def test_chat_is_context_aware_across_turns(client):
    session_id = client.post("/api/session", json={}).json()["session_id"]
    try:
        turn_1 = client.post(
            "/api/chat", json={"session_id": session_id, "message": "Maine order cancel kiya tha."},
        ).json()
        turn_2 = client.post(
            "/api/chat", json={"session_id": session_id, "message": "Refund kab milega?"},
        ).json()

        assert turn_1["context"]["used"] is False
        assert turn_2["context"]["used"] is True
        assert turn_2["context"]["previous_intent"] == turn_1["nlu"]["intent"]

        history = client.get(f"/api/session/{session_id}").json()
        assert history["total_turns"] == 2
    finally:
        client.delete(f"/api/session/{session_id}")


def test_unknown_session_returns_404(client):
    assert client.get("/api/session/does-not-exist").status_code == 404


# --- response-generation fields on /api/chat --------------------------------

def test_chat_includes_the_generated_response(client):
    response = client.post(
        "/api/chat",
        json={"message": "Payment 799 rupaye cut ho gayi but order show nahi ho raha"},
    )

    assert response.status_code == 200
    body = response.json()

    # New fields.
    assert isinstance(body["response"], str) and body["response"]
    assert body["response_type"] == "ESCALATION"
    assert body["requires_human"] is True

    # Existing fields still intact.
    assert set(body) >= {"session_id", "message", "nlu", "context", "priority"}
    assert body["nlu"]["intent"] == "PAYMENT_ISSUE"
    assert body["priority"]["label"] == "HIGH"

    client.delete(f"/api/session/{body['session_id']}")


def test_chat_records_the_reply_in_session_history(client):
    session_id = client.post("/api/session", json={}).json()["session_id"]
    try:
        chat = client.post(
            "/api/chat", json={"session_id": session_id, "message": "Thanks for quick help"},
        ).json()

        history = client.get(f"/api/session/{session_id}").json()["history"]
        assert history[-1]["bot_response"] == chat["response"]
    finally:
        client.delete(f"/api/session/{session_id}")


def test_predict_is_unchanged_by_the_response_layer(client):
    """Only /api/chat gained the reply fields."""
    body = client.post("/api/predict", json={"text": "Thanks for quick help"}).json()

    assert set(body) == {"message", "nlu", "priority"}
    assert "response" not in body
