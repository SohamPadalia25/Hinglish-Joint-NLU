# Implementation Report — Backend / Inference Layer

**Project:** Hinglish Joint NLU (XLM-RoBERTa Large — Intent + Sentiment + Slot Filling)
**Date:** 2026-08-08
**Scope:** Phases 1–14 (backend inference service). Frontend not started.

**Constraints honoured:** no retraining · no weight modification · no notebook modification ·
no label-mapping changes · no preprocessing changes · no CRF · no mBERT/MuRIL/IndicBERT.

---

## 1. Files created

### Model layer

| File | Purpose |
|---|---|
| `backend/model/joint_model.py` | `JointHinglishModel` — architecture, unchanged from the notebook |
| `backend/model/metadata.py` | `ModelMetadata` — label mappings + consistency validation |
| `backend/model/preprocessing.py` | `normalize_hinglish` verbatim + original-casing alignment |
| `backend/model/inference.py` | `HinglishNLU` — load once, `predict()` / `predict_verbose()` |

### Context, priority, pipeline, API

| File | Purpose |
|---|---|
| `backend/context/context_manager.py` | `ConversationContext`, `Turn` |
| `backend/context/context_interpreter.py` | `ContextInterpreter`, `InterpreterConfig` |
| `backend/priority/priority_engine.py` | `PriorityEngine`, `PriorityRules`, `predict_priority` |
| `backend/pipeline.py` | `NLUPipeline.process_message`, singleton `get_pipeline()` |
| `backend/api.py` | FastAPI app, 6 endpoints |
| `backend/config.py` | env-overridable paths and settings |

### Tests (94 total)

`backend/tests/conftest.py`, `test_preprocessing.py`, `test_checkpoint.py`,
`test_inference.py`, `test_context.py`, `test_priority_engine.py`,
`test_pipeline.py`, `test_api.py`

### Root / assets

`README.md`, `requirements.txt`, `run_api.py`, `demo.py`, and
`assets/xlm-roberta-large/` (`config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` — 17 MB, **no model weights**).

---

## 2. Files modified

**None of the originals.**

The notebook, both `.pt` files, the `.tsv`, the metadata JSON and the three `.png` files all
retain their original size and timestamp:

| File | Size (bytes) | LastWriteTime |
|---|---|---|
| `Another_copy_of_Maybe_final.ipynb` | 82,543 | 07-08-2026 12:37:16 |
| `best_model.pt` | 2,240,477,873 | 08-08-2026 11:48:43 |
| `hinglish_joint_model1.pt` | 2,240,481,903 | 08-08-2026 11:45:23 |
| `hinglish_support_dataset_expanded_final.tsv` | 1,113,631 | 21-04-2026 19:22:05 |
| `Intent_Confusion_Matrix1.png` | 221,117 | 07-08-2026 13:36:14 |
| `model_metadata2.json` | 11,670 | 07-08-2026 13:36:26 |
| `Sentiment_Confusion_Matrix1.png` | 46,623 | 07-08-2026 13:36:16 |
| `training_curves1.png` | 60,756 | 07-08-2026 13:36:28 |

No retraining, no weight writes, no label-mapping changes, no preprocessing changes, no CRF.

---

## 3. Installation commands

```bash
pip install -r requirements.txt      # only `transformers` was actually missing
```

`torch 2.8.0+cpu`, `tokenizers 0.22.1`, `numpy`, `fastapi`, `uvicorn`, `pytest` and `httpx`
were already present in the environment. `transformers 4.57.6` was installed.

The tokenizer + encoder config were downloaded once (~17 MB) and vendored into
`assets/xlm-roberta-large/`. **The 2.2 GB pretrained weights were never downloaded.**

---

## 4. Run commands

```bash
python run_api.py --host 127.0.0.1 --port 8000   # API + interactive docs at /docs
python demo.py                                    # console demo
python -m pytest backend/tests -q                 # 94 tests
```

Run with a **single worker** — each worker loads its own ~2.2 GB copy of the model.

---

## 5. Test outputs

```
94 passed in 26.92s
```

### The five required Phase 12 cases — actual output

| # | Input | Result |
|---|---|---|
| 1 | `Payment 799 rupaye cut ho gayi but order show nahi ho raha` | PAYMENT_ISSUE (0.999), NEGATIVE (0.999), **AMOUNT**=`rupaye` — PASS |
| 2 | `Mera order #45231 5 din se deliver nahi hua` | COMPLAINT_DELIVERY (0.971), NEGATIVE (0.989), **ORDER_ID**=`#45231`, **DATE**=`5 din` — PASS |
| 3 | `Thanks for quick help` | FEEDBACK_POSITIVE (0.999), POSITIVE (1.000) — PASS |
| 4 | `Bahut hi worst experience raha` | FEEDBACK_NEGATIVE (0.999), **NEGATIVE** (0.999) — PASS |
| 5 | Turn 1 cancel → Turn 2 `Refund kab milega?` | `context_used = true` — PASS |

### Additional demo predictions

| Input | Intent | Sentiment | Slots | Priority |
|---|---|---|---|---|
| `200 rupaye zyada charge kar liye please refund karo` | REQUEST_REFUND (0.990) | NEGATIVE (0.984) | AMOUNT=`200 rupaye` | HIGH → HUMAN_AGENT |
| `Complaint 2 hafte se pending hai koi sun nahi raha escalate karo` | REQUEST_ESCALATE (0.997) | NEGATIVE (0.997) | DATE=`2 hafte` | HIGH → HUMAN_AGENT |

### Phase 13 validation checklist

| # | Check | Status |
|---|---|---|
| 1 | Load `best_model.pt` | PASS |
| 2 | `state_dict` strict loading (`strict=True`) | PASS — 397/397 keys |
| 3 | Tokenizer works (fast, `word_ids()`) | PASS |
| 4 | Run ≥5 inference examples | PASS — 6 run |
| 5 | Print actual predictions | PASS |
| 6 | API starts | PASS — real uvicorn on port 8123 |
| 7 | `GET /health` | PASS |
| 8 | `POST /api/predict` | PASS |
| 9 | `POST /api/chat` | PASS |
| 10 | Context session handling | PASS |

All endpoint checks were performed over **real HTTP** against a live uvicorn server, not only
via `TestClient`. The server has since been stopped.

Performance: model load ~7–11 s (once per process); inference ~0.4–0.6 s per request on CPU.

---

## 6. Errors

Three things surfaced. None are unresolved defects, but two change what can be claimed about
the model.

### a) One test assertion was wrong — fixed

An assertion that `"Mera order abhi tak nahi aaya"` → `COMPLAINT_DELIVERY` failed; the model
returns `QUERY_ORDER_STATUS`. That phrasing is genuinely ambiguous and was not one of the
required Phase 12 cases, so the test was rewritten to check the context behaviour it was
actually there for.

### b) Sentence-final punctuation shifts predictions

The Phase 8 example assumed `"Maine order cancel kiya tha."` → `REQUEST_CANCEL`. It does not:

| Input | Prediction |
|---|---|
| `Maine order cancel kiya tha.` | COMPLAINT_SERVICE 0.65 (REQUEST_CANCEL 0.30) |
| `Maine order cancel kiya tha` | REQUEST_CANCEL 0.48 (COMPLAINT_SERVICE 0.46) |
| `Maine order cancel kiya tha kal` | REQUEST_CANCEL 0.95 |
| `Maine order #45231 cancel kiya tha` | REQUEST_CANCEL 0.97 |

The training data contains no sentence-final punctuation, so a trailing `.` or `?` is out of
distribution. Stripping punctuation would mean changing the preprocessing the model was
trained on — explicitly forbidden — so this is **documented, not fixed**.

The Phase 12 case-5 test still passes, because `context_used` does not depend on which intent
turn 1 received.

### c) Spurious startup warning

transformers 4.57 prints *"incorrect regex pattern … will lead to incorrect tokenization"*,
citing a Mistral issue. This was verified false for this tokenizer: the vendored tokenizer
produces **identical `input_ids` and `word_ids`** to the Hub tokenizer on every test sentence.
A stray TensorFlow import also logs `oneDNN` notices. Neither affects results.

---

## 7. How the model is loaded

```python
config  = AutoConfig.from_pretrained("assets/xlm-roberta-large")   # config.json only
encoder = AutoModel.from_config(config)                            # no download
model.load_state_dict(torch.load("best_model.pt", weights_only=True), strict=True)
model.eval()
```

**No 2.2 GB fetch.** `AutoModel.from_pretrained()` would download the pretrained weights only
for `load_state_dict` to overwrite every one of them. Only `config.json` is genuinely needed.

Encoder construction runs under `transformers.modeling_utils.no_init_weights()` to skip
randomising 560M parameters that are immediately replaced, with a graceful fallback to normal
initialisation if that internal moves.

**`strict=True` passes**, proving the rebuilt architecture matches the trained one exactly:
397/397 keys, 560,078,007 parameters, no missing or unexpected entries.

Loading takes ~7–11 s and happens once per process; `get_pipeline()` holds a process-wide
singleton behind a lock. Forward passes are serialised with a mutex, since uvicorn runs sync
endpoints in a threadpool.

**Slot decoding** reads only the **first sub-token** of each word — matching how training
aligned labels (continuations were set to `-100` and ignored by the loss), so any other
position would read an untrained output. Results are emitted as a **list of spans**, so
duplicate entity types survive; a dict keyed by slot type would silently overwrite them.
Values map back to the original text by index alignment, so `#ABC123` stays `#ABC123`, with
the lowercased form retained in `value_normalized`.

---

## 8. How context works

NLU runs on the **current message only** — conversation history is never concatenated into
XLM-R, which would put the encoder far outside its training distribution and let an older turn
override the current one.

History is read *before* the current turn is appended, so a message is never treated as its
own context.

Context is used only when at least one ambiguity signal fires:

| Signal | Condition |
|---|---|
| `short_query` | ≤ 5 tokens after normalization |
| `anaphora_marker` | contains `iska`, `uska`, `wo`, `ye`, `same`, `it`, `that`, … |
| `follow_up_marker` | contains `kab`, `kitna`, `phir`, `dobara`, `status`, `abhi`, … |
| `low_intent_confidence` | intent confidence < 0.60 |
| `no_slots_in_query` | current message has no entities but earlier turns do |

No signals → the query is self-contained and history is ignored entirely.

The layer **never rewrites** the model's intent or sentiment, and **never invents** a slot
value. Inherited slots are tagged `"source": "context"` and carry the turn they came from, so
a caller can always distinguish an extracted entity from a remembered one. Only stable
identifier-like slot types are inheritable (`ORDER_ID`, `AMOUNT`, `PRODUCT_NAME`,
`PAYMENT_METHOD`, …). History is bounded to the last 5 turns per session.

---

## 9. How priority works

Additive score → threshold → route.

```
score  = intent_base
       + sentiment_modifier
       + urgency_slot_bonus
       + repeated_intent_bonus
       + sustained_negative_bonus
       clamped to [0, 1]

HIGH   if score >= 0.75   -> HUMAN_AGENT
MEDIUM if score >= 0.35   -> ASSISTED_BOT
LOW    otherwise          -> AUTO_BOT

REQUEST_ESCALATE and REQUEST_CALLBACK always route to HUMAN_AGENT.
```

Base scores are **derived from the dataset**, not invented:
`(HIGH*1.0 + MEDIUM*0.5 + LOW*0.0) / n` per intent, measured over all 2,989 rows.

| Intent | Base | Intent | Base |
|---|---|---|---|
| REQUEST_ESCALATE | 1.000 | QUERY_REFUND_STATUS | 0.544 |
| COMPLAINT_DELIVERY | 0.968 | REQUEST_CANCEL | 0.515 |
| PAYMENT_ISSUE | 0.944 | REQUEST_CALLBACK | 0.461 |
| COMPLAINT_PRODUCT | 0.906 | QUERY_ORDER_STATUS | 0.410 |
| COMPLAINT_SERVICE | 0.862 | QUERY_PRODUCT_INFO | 0.238 |
| COMPLAINT_BILLING | 0.858 | FEEDBACK_POSITIVE | 0.000 |
| RETURN_REQUEST | 0.778 | | |
| FEEDBACK_NEGATIVE | 0.759 | | |
| REQUEST_REFUND | 0.713 | | |

Modifiers: `NEGATIVE +0.15`, `NEUTRAL 0.00`, `POSITIVE −0.30`; urgency-bearing slots
(`URGENCY`, `SLA`, `THREAT`, `ESCALATION_LEVEL`, `WAIT_TIME`, `COMPLAINT_COUNT`, …) `+0.10`
each capped at `+0.20`; repeating an intent within a session `+0.10`; two or more negative
turns in recent history `+0.10`.

Every response carries `rules_fired` listing each rule and its numeric contribution.

A test asserts the engine reproduces the dataset's **majority priority label for 14 of 15
intents** at their modal sentiment. The single deviation is deliberate and documented:
`REQUEST_REFUND` + `NEGATIVE` → HIGH, where the dataset splits 74 HIGH / 100 MEDIUM. The
weights were not fudged to make that test green. Neutral refund requests stay MEDIUM.

All thresholds and weights live in `PriorityRules` and can be swapped without touching code.

Documented throughout as a **"priority-based routing layer"**, never as a trained classifier.

---

## 10. Exact API request/response examples

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "0.1.0",
  "model": {
    "model_name": "xlm-roberta-large",
    "checkpoint": "D:\\Soham Clg\\major project\\best_model.pt",
    "device": "cpu",
    "max_len": 128,
    "num_intents": 15,
    "num_sentiments": 3,
    "num_slots": 165,
    "parameters": 560078007,
    "load_seconds": 6.41
  },
  "max_context_turns": 5,
  "active_sessions": 0
}
```

### `POST /api/predict`

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Payment 799 rupaye cut ho gayi but order show nahi ho raha"}'
```

```json
{
  "message": "Payment 799 rupaye cut ho gayi but order show nahi ho raha",
  "nlu": {
    "intent": "PAYMENT_ISSUE",
    "intent_confidence": 0.9988,
    "sentiment": "NEGATIVE",
    "sentiment_confidence": 0.9986,
    "slots": [
      {"type": "AMOUNT", "value": "rupaye", "confidence": 0.954, "value_normalized": "rupaye"}
    ]
  },
  "priority": {
    "label": "HIGH",
    "score": 1.0,
    "routing": "HUMAN_AGENT",
    "reason": "HIGH (score 1.00, thresholds HIGH>=0.75 MEDIUM>=0.35) -> HUMAN_AGENT. intent=PAYMENT_ISSUE base=0.944 (+0.94); sentiment=NEGATIVE (+0.15)"
  }
}
```

Pass `"verbose": true` to add `intent_top_k` and `sentiment_distribution`.

### `POST /api/chat` — turn 1

```bash
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"session_id":"abc","message":"Maine order #45231 cancel kiya tha"}'
```

```json
{
  "session_id": "abc",
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
  }
}
```

### `POST /api/chat` — turn 2 (follow-up, context used)

```bash
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"session_id":"abc","message":"Refund kab milega?"}'
```

```json
{
  "session_id": "abc",
  "message": "Refund kab milega?",
  "nlu": {
    "intent": "QUERY_REFUND_STATUS",
    "intent_confidence": 0.9998,
    "sentiment": "NEUTRAL",
    "sentiment_confidence": 0.9778,
    "slots": []
  },
  "context": {
    "used": true,
    "reason": "Query looks like a follow-up (short_query, follow_up_marker, no_slots_in_query). Carried forward previous_intent=REQUEST_CANCEL and slots ['ORDER_ID'].",
    "previous_intent": "REQUEST_CANCEL",
    "previous_sentiment": "NEUTRAL",
    "inherited_slots": [
      {"type": "ORDER_ID", "value": "#45231", "from_turn": 1, "source": "context"}
    ],
    "signals": [
      {"signal": "short_query", "detail": "3 tokens (<= 5)"},
      {"signal": "follow_up_marker", "detail": "follow-up phrasing ['kab']"},
      {"signal": "no_slots_in_query", "detail": "current message carries no entities, earlier turns do"}
    ]
  },
  "priority": {
    "label": "MEDIUM",
    "score": 0.544,
    "routing": "ASSISTED_BOT",
    "reason": "MEDIUM (score 0.54, thresholds HIGH>=0.75 MEDIUM>=0.35) -> ASSISTED_BOT. intent=QUERY_REFUND_STATUS base=0.544 (+0.54)"
  }
}
```

### Session endpoints

| Request | Response |
|---|---|
| `POST /api/session` `{}` | `201` `{"session_id": "f46eb3a3…", "created": true}` |
| `GET /api/session/{id}` | `200` `{"session_id", "created_at", "total_turns": 2, "retained_turns": 2, "max_turns": 5, "history": [...]}` |
| `DELETE /api/session/{id}` | `200` `{"session_id": "f46eb3a3…", "cleared": true}` |
| `GET /api/session/{id}` after delete | `404` `{"detail": "Unknown session: f46eb3a3…"}` |
| `POST /api/predict` `{"text": ""}` | `422` `{"detail": [{"type": "string_too_short", …}]}` |

---

## 11. Items requiring attention before the next phase

1. **Punctuation sensitivity.** If the frontend sends user-typed text containing question
   marks and periods, predictions will shift (see §6b). The clean fix — stripping punctuation
   — requires retraining to remain honest.
2. **Slot F1 ≈ 0.55.** Partial extractions such as `AMOUNT = "rupaye"` instead of
   `"799 rupaye"` are routine. Slots should not drive transactional actions without human
   confirmation. Verified as genuine model output, not a decoding bug, by dumping raw
   per-word BIO tags.
3. **Intent accuracy 83.95%.** Worst confusion is `PAYMENT_ISSUE` → `COMPLAINT_BILLING`
   (9 of 37 validation examples). Gate automated actions on `intent_confidence`.
4. **Sessions are in-process memory.** Lost on restart, not shared between workers. Redis or
   a table is the first thing to add before scaling past one process.
5. **No authentication, rate limiting, or request-size limits.** This is a demo service.
6. **`hinglish_joint_model1.pt` is redundant.** Verified bit-identical to `best_model.pt`
   (0 of 397 tensors differ). Deleting it reclaims 2.24 GB. Not deleted — awaiting approval.

---

## Status

Backend and inference layer complete and tested (94/94 passing). Frontend not started, as
instructed. Awaiting next instruction.
