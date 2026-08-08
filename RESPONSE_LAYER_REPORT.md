# Response Layer — Report

**Project:** Hinglish Joint NLU (XLM-RoBERTa Large — Intent + Sentiment + Slot Filling)
**Date:** 2026-08-08
**Scope:** Response-generation layer only. Frontend not started.

**Constraints honoured — none of the following were modified:**
`best_model.pt` · the XLM-R architecture · preprocessing · label mappings · inference logic ·
context manager · priority engine · `backend/pipeline.py` · `POST /api/predict`.

The layer consumes the existing `NLUPipeline.process_message` output and adds nothing to the
model path. It is deterministic and template-based: no model call, no external LLM, no
randomness. The same input always produces the same output.

---

## 1. Files created

| File | Contents |
|---|---|
| `backend/response/response_generator.py` | `ResponseGenerator`, `IntentStrategy`, 15 intent strategies, `RESPONSE_TYPES` |
| `backend/response/__init__.py` | package exports |
| `backend/tests/test_response_generator.py` | 101 tests |

## 2. Files modified

| File | Change |
|---|---|
| `backend/api.py` | `/api/chat` **only** — adds `response`, `response_type`, `requires_human`; module-level `_responder` |
| `backend/tests/test_api.py` | +3 tests for the new fields |
| `README.md` | new §6a documenting the layer; updated file tree and test count |

`POST /api/predict` was verified unchanged — it still returns exactly `message`, `nlu`,
`priority`, with no response fields.

### One deviation from the literal spec, flagged for review

`/api/chat` now calls `pipeline.context.set_bot_response(...)` so the generated reply is
recorded against the turn the pipeline just stored. The `Turn` dataclass already carried a
`bot_response` field that was permanently `null`.

* This uses an **existing public method**; the context manager itself was not modified.
* It does change what `GET /api/session/{id}` shows — `bot_response` is now populated.
* It is a one-line revert if the empty field is preferred.

---

## 3. Design

### Per-intent strategies

All 15 intents have their own `IntentStrategy`, holding four things:

| Field | Meaning |
|---|---|
| `summary` | what we understood, phrased as a restatement of the user's own message — never a claim about system state |
| `action` | what support can do next, phrased as capability ("we can check"), never as a completed outcome ("your refund is processed") |
| `helpful_slots` | slot types that would let support act |
| `ask` | the question to ask when none of `helpful_slots` are available |

Covered intents: `COMPLAINT_BILLING`, `COMPLAINT_DELIVERY`, `COMPLAINT_PRODUCT`,
`COMPLAINT_SERVICE`, `FEEDBACK_NEGATIVE`, `FEEDBACK_POSITIVE`, `PAYMENT_ISSUE`,
`QUERY_ORDER_STATUS`, `QUERY_PRODUCT_INFO`, `QUERY_REFUND_STATUS`, `REQUEST_CALLBACK`,
`REQUEST_CANCEL`, `REQUEST_ESCALATE`, `REQUEST_REFUND`, `RETURN_REQUEST`.

A test asserts the strategy keys exactly equal the 15 intent labels in `model_metadata2.json`.

### Tone from sentiment

| Sentiment | Opener |
|---|---|
| `NEGATIVE` | "I'm sorry for the trouble this has caused." — empathetic, solution-oriented |
| `NEUTRAL` | "Thanks for getting in touch." — professional and direct |
| `POSITIVE` | "Thank you for the kind words!" — friendly and appreciative |

At `HIGH` priority the opener switches to an urgent variant
("I'm really sorry about this, and I understand it needs urgent attention.").

### Closing line from priority

| Priority | Closer |
|---|---|
| `HIGH` | "I've marked this as high priority and a support specialist will pick it up." |
| `MEDIUM` + `requires_human` | "I've passed this to our support team so someone can get back to you." |
| `MEDIUM` | "Our support team will follow this up with you." |
| `LOW` | "Happy to help with anything else right here." |

### Reply assembly

```
[sentiment / urgency opener]
[I understand that {summary}.]
[I've noted the {complete slots}.]                    (only if any survive the filter)
[I'm carrying forward the {inherited slots} from your earlier message.]   (only if context used)
[action]
[ask]                                                  (only if no helpful slot is available)
[priority closer]
```

### `response_type`

One of `ESCALATION`, `CALLBACK`, `CLARIFICATION`, `ACTION_CONFIRMATION`, `ACKNOWLEDGEMENT`,
`INFORMATION`, `APPRECIATION`, `APOLOGY`, `FALLBACK`.

Precedence: **callback → escalation → clarification → the intent's default.**

This matches the specified example where `PAYMENT_ISSUE` + `NEGATIVE` + `HIGH` yields
`ESCALATION` rather than the intent's own type. A consequence worth noting: because the
priority engine scores `COMPLAINT_DELIVERY` and `FEEDBACK_NEGATIVE` as HIGH, those also
surface as `ESCALATION`.

### `requires_human`

`true` when the intent is `REQUEST_ESCALATE` or `REQUEST_CALLBACK`, **or** priority is `HIGH`,
**or** routing is `HUMAN_AGENT`. Otherwise `false`.

---

## 4. Anti-hallucination

The generator may only restate what the pipeline actually extracted. It never states an order
status, a refund or delivery date, an amount, a transaction outcome, an account detail or a
policy. Two mechanisms enforce this, both under test.

### a) No template contains a digit

Every template string is fact-free and digit-free, so any number appearing in a reply
provably came from a slot value, verbatim.

* `test_no_template_contains_a_digit` — scans all openers, closers, summaries, actions and
  asks.
* `test_no_invented_numbers` — for all 15 intents, asserts every digit run in a generated
  reply traces back to a slot value.

### b) Partial slots are dropped, never repaired

The model routinely emits partial spans. Identifier- and quantity-like slot types
(`ORDER_ID`, `AMOUNT`, `ACTUAL_AMOUNT`, `QUANTITY`, `ORDER_COUNT`, `COMPLAINT_COUNT`,
`TAX_ID`, `CONTACT`, `RATING`, `THRESHOLD`, `BUDGET`, `DISCOUNT`, `CHARGE`, `WAIT_TIME`)
must contain a digit to be quoted at all.

When the model returns `AMOUNT = "rupaye"` for *"Payment 799 rupaye cut ho gayi"*, the reply
mentions no amount and asks for the payment reference instead. **"799" is never reconstructed
from the raw message.** A dropped slot also does not count toward satisfying `helpful_slots`,
so the reply still asks for what it needs.

### c) Banned-pattern scan

`test_never_states_status_dates_or_outcomes` runs all 45 intent × sentiment combinations and
asserts none of these appear:

```
delivered on · will arrive · arriving on · has been refunded · refund of ·
credited to · tracking id/number is · your balance · policy states ·
within N hours/days · business days
```

Additional targeted guards:

* `REQUEST_CANCEL` never claims the cancellation already happened
  (`\b(has been|is) cancelled\b`).
* `QUERY_REFUND_STATUS` never states where the refund is
  (`refunded|credited|processed on|will be credited`).
* The context-dependent refund query never invents a timeframe
  (`days|weeks|hours|tomorrow|by <weekday>`).

Where a needed detail is missing, the reply asks for it and the type becomes `CLARIFICATION`.

---

## 5. Tests

**198 total, all passing** (94 before → 198; **104 new**).

```
101  test_response_generator.py     27  test_priority_engine.py     11  test_api.py
 18  test_preprocessing.py          13  test_inference.py            9  test_pipeline.py
 12  test_context.py                 7  test_checkpoint.py
```

```
198 passed in 29.78s
```

The 94 pre-existing tests continue to pass unchanged.

### The ten required cases

| # | Case | `response_type` | `requires_human` |
|---|---|---|---|
| 1 | PAYMENT_ISSUE + NEGATIVE + HIGH | ESCALATION | true |
| 2 | COMPLAINT_DELIVERY | ESCALATION (scores HIGH) | true |
| 3 | QUERY_REFUND_STATUS | CLARIFICATION | false |
| 4 | REQUEST_CANCEL | ACTION_CONFIRMATION | false |
| 5 | REQUEST_REFUND | CLARIFICATION | false |
| 6 | REQUEST_ESCALATE | ESCALATION | true |
| 7 | FEEDBACK_POSITIVE | APPRECIATION | false |
| 8 | FEEDBACK_NEGATIVE | ESCALATION (scores HIGH) | true |
| 9 | REQUEST_CALLBACK | CALLBACK | true |
| 10 | Context-dependent refund query | INFORMATION | false |

### Beyond the required ten

* 15 intents × 3 sentiments — output shape, non-empty reply, valid `response_type`, no double
  spaces, no leading/trailing whitespace.
* Determinism — same input, identical output.
* Tolerance of missing `nlu` / `context` / `priority` blocks and of unrecognised intents.
* Tone tracks sentiment; `HIGH` uses the urgent opener and mentions escalation; `LOW` offers
  self-service.
* Inherited slots are only used when `context.used` is `true`.
* Two end-to-end tests through the real checkpoint.

---

## 6. Example outputs (real model, real pipeline)

### Partial slot — the case that matters most

**Input:** `Payment 799 rupaye cut ho gayi but order show nahi ho raha`
**Model slots:** `AMOUNT = "rupaye"` (the number was dropped by the slot head)

> I'm really sorry about this, and I understand it needs urgent attention. I understand that
> your payment didn't go through the way it should have. We can help you check the payment
> and the order status, and get a deducted or failed payment traced. Could you share the
> order ID and the payment or UPI reference from your bank? I've marked this as high
> priority and a support specialist will pick it up.

`response_type: ESCALATION` · `requires_human: true` · **"799" never appears.**

### Complete slots quoted verbatim

**Input:** `Mera order #45231 5 din se deliver nahi hua`

> I'm really sorry about this, and I understand it needs urgent attention. I understand that
> your order hasn't reached you as expected. **I've noted the order ID #45231 and the
> timeframe 5 din.** We can check the shipment with the courier and chase up the delivery.
> I've marked this as high priority and a support specialist will pick it up.

`response_type: ESCALATION` · `requires_human: true`

### Positive feedback

**Input:** `Thanks for quick help`

> Thank you for the kind words! I understand that you're happy with how things went. I'll
> pass this on to the team - it genuinely means a lot to them. Happy to help with anything
> else right here.

`response_type: APPRECIATION` · `requires_human: false`

### Negative feedback

**Input:** `Bahut hi worst experience raha`

> I'm really sorry about this, and I understand it needs urgent attention. I understand that
> your experience with us hasn't been a good one. Your feedback is noted and passed on to
> the team responsible. If there's a particular order or interaction behind this, tell me
> and I'll have it looked into. I've marked this as high priority and a support specialist
> will pick it up.

`response_type: ESCALATION` · `requires_human: true`

### Escalation

**Input:** `Complaint 2 hafte se pending hai koi sun nahi raha escalate karo`

> I'm really sorry about this, and I understand it needs urgent attention. I understand that
> you'd like this taken further. I've noted the timeframe 2 hafte. I'm escalating this to a
> senior support specialist. If you have an order ID or an earlier complaint reference,
> sharing it will speed this up. I've marked this as high priority and a support specialist
> will pick it up.

`response_type: ESCALATION` · `requires_human: true`

### Callback

**Input:** `Mujhe callback chahiye`

> Thanks for getting in touch. I understand that you'd like someone from our team to call
> you. I've noted your callback request and passed it to the support team. Could you confirm
> the number to call and a time window that suits you? I've passed this to our support team
> so someone can get back to you.

`response_type: CALLBACK` (wins over ESCALATION) · `requires_human: true`

### Product query

**Input:** `iPhone 15 ka price kya hai`

> Thanks for getting in touch. I understand that you'd like more details about a product.
> I've noted the product iPhone 15. I can share the specifications, availability and pricing
> we have on record. Happy to help with anything else right here.

`response_type: INFORMATION` · `requires_human: false`

### Context conversation — verified over live HTTP

**Turn 1** — `Maine order #45231 cancel kiya tha`

> Thanks for getting in touch. I understand that you'd like to cancel your order. I've noted
> the order ID #45231. **We can put a cancellation request through once the order is
> confirmed.** Our support team will follow this up with you.

`response_type: ACTION_CONFIRMATION` · `requires_human: false` · `context.used: false`
Takes the request down; never claims the cancellation already happened.

**Turn 2** — `Refund kab milega?`

> Thanks for getting in touch. I understand that you'd like to know where your refund has
> reached. **I'm carrying forward the order ID #45231 from your earlier message.** I can get
> the refund checked against your order and tell you where it stands. Our support team will
> follow this up with you.

`response_type: INFORMATION` · `requires_human: false` · `context.used: true`
References the remembered order; invents no refund date and no amount.

---

## 7. API integration

`POST /api/chat` returns the full pipeline result **plus** three fields. Nothing existing was
removed or renamed.

```json
{
  "session_id": "5f2ff7a5e8064e7a878abffaa1236dbb",
  "message": "Maine order #45231 cancel kiya tha",
  "nlu": {
    "intent": "REQUEST_CANCEL",
    "intent_confidence": 0.9726,
    "sentiment": "NEUTRAL",
    "sentiment_confidence": 0.7828,
    "slots": [
      {"type": "ORDER_ID", "value": "#45231", "confidence": 0.9939, "value_normalized": "#45231"}
    ]
  },
  "context": {
    "used": false,
    "reason": "No prior turns in this session; nothing to interpret against.",
    "previous_intent": null,
    "inherited_slots": [],
    "signals": [],
    "relevant_history": []
  },
  "priority": {
    "label": "MEDIUM",
    "score": 0.515,
    "routing": "ASSISTED_BOT",
    "reason": "MEDIUM (score 0.52, thresholds HIGH>=0.75 MEDIUM>=0.35) -> ASSISTED_BOT. intent=REQUEST_CANCEL base=0.515 (+0.52)"
  },

  "response": "Thanks for getting in touch. I understand that you'd like to cancel your order. I've noted the order ID #45231. We can put a cancellation request through once the order is confirmed. Our support team will follow this up with you.",
  "response_type": "ACTION_CONFIRMATION",
  "requires_human": false
}
```

`POST /api/predict` is untouched and still returns only `message`, `nlu`, `priority` —
confirmed over live HTTP.

---

## 8. Verification performed

| Check | Result |
|---|---|
| Full test suite | 198 passed |
| 94 pre-existing tests | all still passing |
| API starts | PASS — real uvicorn, port 8124 |
| `POST /api/chat` returns the three new fields | PASS |
| Existing `/api/chat` fields intact | PASS |
| `POST /api/predict` unchanged | PASS — `['message', 'nlu', 'priority']` |
| `bot_response` recorded in session history | PASS |
| Context carried across turns in a live session | PASS |
| No invented numbers across all 15 intents | PASS |
| No status / date / outcome claims across 45 combinations | PASS |

Server stopped after verification.

---

## 9. Notes and limitations

1. **Deterministic by design.** No LLM is used, optional or otherwise. Replies are fixed
   templates assembled from pipeline output, so they are repetitive across similar messages —
   the tradeoff taken for verifiability and zero hallucination risk.
2. **`response_type` is influenced by priority**, per the specified example. Because the
   priority engine scores `COMPLAINT_DELIVERY`, `COMPLAINT_*`, `PAYMENT_ISSUE` and
   `FEEDBACK_NEGATIVE` as HIGH, all of them surface as `ESCALATION` rather than their
   intent-specific type. `requires_human` carries the same information.
3. **Slot quality bounds reply quality.** With slot F1 ≈ 0.55, many messages yield no
   quotable slot, so the reply asks for the order ID instead of using one. This is correct
   behaviour, but it means a noticeable share of replies are `CLARIFICATION`.
4. **Replies make routing promises, not outcome promises.** "A support specialist will pick
   it up" reflects `routing = HUMAN_AGENT`. There is no actual ticketing integration behind
   it — wiring that up is a separate piece of work.
5. **English-only replies.** The model reads Hinglish; the generator answers in English.
   Hinglish reply templates would be a straightforward addition if wanted.

---

## Status

Response-generation layer complete and tested. 198/198 tests passing. Frontend not started,
as instructed. Awaiting next instruction.
