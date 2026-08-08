"""Console demo: loads the checkpoint and prints real predictions.

    python demo.py

Covers the Phase 12 cases plus a two-turn context conversation.
Output is ASCII-only so it renders on the Windows console (cp1252).
"""

from __future__ import annotations

import logging
import time

from backend.pipeline import NLUPipeline

logging.basicConfig(level=logging.WARNING)

SINGLE_CASES = [
    "Payment 799 rupaye cut ho gayi but order show nahi ho raha",
    "Mera order #45231 5 din se deliver nahi hua",
    "Thanks for quick help",
    "Bahut hi worst experience raha",
    "200 rupaye zyada charge kar liye please refund karo",
    "Complaint 2 hafte se pending hai koi sun nahi raha escalate karo",
]

CONVERSATION = [
    "Maine order cancel kiya tha.",
    "Refund kab milega?",
]


def rule(char: str = "-", width: int = 78) -> None:
    print(char * width)


def show_nlu(result: dict) -> None:
    nlu = result["nlu"]
    print(f"  intent    : {nlu['intent']:22s} ({nlu['intent_confidence']:.3f})")
    print(f"  sentiment : {nlu['sentiment']:22s} ({nlu['sentiment_confidence']:.3f})")
    if nlu["slots"]:
        print("  slots     :")
        for slot in nlu["slots"]:
            print(f"                {slot['type']:20s} = {slot['value']!r} ({slot['confidence']:.3f})")
    else:
        print("  slots     : (none)")
    priority = result["priority"]
    print(f"  priority  : {priority['label']} (score {priority['score']:.2f}) -> {priority['routing']}")


def main() -> None:
    rule("=")
    print("  HINGLISH JOINT NLU - DEMO")
    rule("=")

    started = time.perf_counter()
    pipeline = NLUPipeline()
    print(f"\nModel loaded in {time.perf_counter() - started:.1f}s")
    for key, value in pipeline.nlu.info().items():
        print(f"   {key:14s}: {value}")

    print("\n")
    rule("=")
    print("  SINGLE-QUERY NLU  (POST /api/predict - no conversation context)")
    rule("=")
    for text in SINGLE_CASES:
        started = time.perf_counter()
        result = pipeline.predict(text)
        print(f"\nINPUT: {text}   [{time.perf_counter() - started:.2f}s]")
        show_nlu(result)

    print("\n")
    rule("=")
    print("  CONTEXT-AWARE CONVERSATION  (POST /api/chat)")
    rule("=")
    session_id = pipeline.create_session()
    for turn_number, message in enumerate(CONVERSATION, 1):
        result = pipeline.process_message(session_id, message)
        print(f"\nTURN {turn_number}: {message}")
        show_nlu(result)
        context = result["context"]
        print(f"  context   : used={context['used']}")
        print(f"              {context['reason']}")
        if context["inherited_slots"]:
            print(f"              inherited: {context['inherited_slots']}")

    print("\nSession history:")
    for turn in pipeline.get_session(session_id)["history"]:
        print(f"   turn {turn['turn_index']}: {turn['user_message']!r} -> "
              f"{turn['intent']} / {turn['sentiment']} / {turn['priority']}")
    pipeline.clear_session(session_id)

    print()
    rule("=")
    print("  DONE")
    rule("=")


if __name__ == "__main__":
    main()
