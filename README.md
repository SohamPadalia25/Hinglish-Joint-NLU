# Hinglish Joint NLU — Backend Inference Service

A reusable inference service around an already-trained joint NLU model for Hinglish
customer-support text. **Nothing in this backend trains, fine-tunes, or modifies the
model.** It loads the existing checkpoint and serves it.

> **The NLU model is a fine-tuned XLM-RoBERTa Large joint model with Intent, Sentiment and Slot heads.**
>
> **Priority is currently implemented as a routing layer, not a separately trained neural classifier.**
>
> **Context awareness is implemented as a context-management/inference layer around the trained NLU model.**

---

## 1. Architecture

```
                      message
                         |
              normalize_hinglish()            backend/model/preprocessing.py
                         |
              whitespace tokenization
                         |
        XLMRobertaTokenizerFast (is_split_into_words=True, max_len=128)
                         |
        +----------------------------------+
        |  XLMRobertaModel                 |   24 layers, hidden = 1024
        |  "xlm-roberta-large"             |   250,002 vocab
        +----------------------------------+
                         |
                 last_hidden_state
                         |
     +-------------------+-------------------+
     |                   |                   |
  [:, 0, :]          [:, 0, :]           [:, :, :]
     |                   |                   |
 Dropout(0.3)       Dropout(0.3)        Dropout(0.3)
     |                   |                   |
Linear(1024->15)   Linear(1024->3)    Linear(1024->165)
     |                   |                   |
   INTENT            SENTIMENT          SLOTS (BIO)
     |                   |                   |
     +-------------------+-------------------+
                         |
              ConversationContext             backend/context/context_manager.py
                         |
              ContextInterpreter              backend/context/context_interpreter.py
                         |
              PriorityEngine                  backend/priority/priority_engine.py
                         |
                 structured result
```

Two details are load-bearing and must not drift:

* The sentence-level heads read **`last_hidden_state[:, 0, :]`**, not `pooler_output`.
  The pooler exists in the checkpoint but feeds nothing. `test_checkpoint.py::test_sentence_heads_read_cls_not_pooler`
  perturbs the pooler weights and asserts the intent logits do not move.
* Slot filling is **plain per-token softmax. There is no CRF.**

### Layout

```
backend/
    config.py                       paths + runtime settings (env-overridable)
    pipeline.py                     process_message() — the master pipeline
    api.py                          FastAPI app
    model/
        joint_model.py              JointHinglishModel (architecture)
        metadata.py                 label mappings + validation
        preprocessing.py            normalize_hinglish (verbatim from training)
        inference.py                HinglishNLU — load once, predict many
    context/
        context_manager.py          ConversationContext (session history)
        context_interpreter.py      self-contained vs follow-up decision
    priority/
        priority_engine.py          rule-based priority + routing
    response/
        response_generator.py       template-based reply generation
    tests/                          198 tests
assets/
    xlm-roberta-large/              config.json + tokenizer files ONLY (no weights)
demo.py                             console demo
run_api.py                          server entry point
requirements.txt
```

---

## 2. Model information

| | |
|---|---|
| Encoder | `xlm-roberta-large` — 24 layers, hidden 1024, vocab 250,002, max positions 514 |
| Heads | intent 1024→15, sentiment 1024→3, slots 1024→165 |
| Total parameters | **560,078,007** (559,890,432 encoder + 187,575 heads) |
| Precision | float32 throughout |
| Checkpoint | `best_model.pt` — a flat `OrderedDict` state_dict, 397 tensors |
| Max sequence length | 128 |
| Training | full fine-tune of the entire encoder — no LoRA, no adapters, no frozen layers |

**Validation metrics** (598-row validation split, read from the saved confusion matrices):

| Metric | Value |
|---|---|
| Intent accuracy | 502/598 = **83.95%** |
| Sentiment accuracy | 560/598 = **93.65%** |
| Slot F1 | **≈0.55** (from `training_curves1.png`) |

Training ran 9 of 10 epochs and early-stopped with `PATIENCE=3`, which places the
last val-loss improvement — and therefore the saved weights — at **epoch 6**.

`best_model.pt` and `hinglish_joint_model1.pt` were compared tensor by tensor with
`torch.equal`: **0 of 397 differ**. They are the same weights. This service uses
`best_model.pt`; the other file is a redundant 2.24 GB copy.

---

## 3. How checkpoint loading works

`torch.load()` on either `.pt` returns a **plain `OrderedDict[str, Tensor]`** — not a
wrapper dict, not a training checkpoint. There is no optimizer state, no scheduler, no
epoch counter, no LoRA or adapter weights.

Loading therefore has to rebuild the architecture first and then fill it:

```python
config  = AutoConfig.from_pretrained("assets/xlm-roberta-large")  # config.json only
encoder = AutoModel.from_config(config)                           # random init, no download
model   = JointHinglishModel(config, num_intents=15, num_sentiments=3, num_slots=165)
model.load_state_dict(torch.load("best_model.pt", weights_only=True), strict=True)
model.eval()
```

**Why `from_config` and not `from_pretrained`:** `AutoModel.from_pretrained("xlm-roberta-large")`
downloads ~2.2 GB of pretrained weights, and `load_state_dict` then overwrites **every one
of them** with the fine-tuned values. The download is pure waste. Only `config.json` is
genuinely needed. To skip the pointless random initialisation of 560M parameters that are
immediately replaced, the encoder is built under `transformers.modeling_utils.no_init_weights()`,
with a graceful fallback to normal init.

**`strict=True` is deliberate.** It is the proof that the architecture rebuilt here is
byte-for-byte the one that was trained — every key present, no extras, all shapes matching.
A silent rename or a wrong head width would raise instead of quietly mispredicting.

Loading takes **~7–11 seconds** on CPU and happens once per process. `get_pipeline()` holds
a process-wide singleton.

---

## 3a. Obtaining the checkpoint

**`best_model.pt` is not in this repository.** It is ~2.14 GB, and GitHub rejects any file
over 100 MB — it also exceeds the 2 GiB per-file limit for Release assets, so there is no
configuration under which it could be hosted here. `.gitignore` excludes `*.pt`.

To run the backend you need `best_model.pt` in the project root. It was produced by
`Another_copy_of_Maybe_final.ipynb` (which *is* in this repository) and copied to Google Drive
at the end of training:

```
/content/drive/MyDrive/XLmLarge/best_model1.pt   ->  ./best_model.pt
```

`hinglish_joint_model1.pt` from the same folder is bit-identical (verified tensor by tensor,
0 of 397 differ) and works equally well — rename either one to `best_model.pt`, or point
`HINGLISH_CHECKPOINT` at wherever you keep it:

```bash
set HINGLISH_CHECKPOINT=D:\path\to\best_model.pt     # Windows
export HINGLISH_CHECKPOINT=/path/to/best_model.pt    # macOS / Linux
```

Without the checkpoint, the backend fails at startup with a clear
`FileNotFoundError: Checkpoint not found`, and the frontend shows *"Backend offline"*.

---

## 4. Installation

```bash
pip install -r requirements.txt
```

Only `transformers` was missing from the original environment; `torch`, `tokenizers`,
`numpy`, `fastapi`, `uvicorn`, `pytest` and `httpx` were already present.

The tokenizer and encoder config are already vendored in `assets/xlm-roberta-large/`
(~17 MB). If that directory is ever deleted, `HinglishNLU` re-downloads **the tokenizer and
config only** on first use and caches them back — it never fetches model weights.

Verified equivalent: the vendored tokenizer produces **identical `input_ids` and `word_ids`**
to `AutoTokenizer.from_pretrained("xlm-roberta-large")` on every test sentence.

Required at runtime:

| File | Purpose |
|---|---|
| `best_model.pt` | the fine-tuned weights |
| `model_metadata2.json` | label mappings (15 / 3 / 165) and `max_len` |
| `assets/xlm-roberta-large/config.json` | encoder architecture |
| `assets/xlm-roberta-large/tokenizer.json` | fast tokenizer (needed for `word_ids()`) |

Not needed at runtime: the `.tsv` dataset (used only by one priority test), the notebook
(used only by one preprocessing-drift test), and the `.png` files.

---

## 4a. Frontend

A React + Vite dashboard lives in `frontend/`. It consumes `POST /api/chat` and renders
nothing it did not receive from the backend — no prediction logic exists on the client.

```bash
cd frontend
npm install
cp .env.example .env      # VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev               # http://localhost:5173
```

The backend must be running first. CORS is configured for the Vite dev origin
(`backend/config.py` → `CORS_ORIGINS`); override with `HINGLISH_CORS_ORIGINS` for other hosts.

> Vite binds to `localhost`, which resolves to IPv6 on Windows — open
> `http://localhost:5173`, not `http://127.0.0.1:5173`.

---

## 5. Running the backend

```bash
python run_api.py --host 127.0.0.1 --port 8000
# or
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Interactive API docs at `http://127.0.0.1:8000/docs`.

Console demo without the HTTP layer:

```bash
python demo.py
```

Tests:

```bash
python -m pytest backend/tests -q
```

> Run with a **single worker**. Each worker would load its own ~2.2 GB copy of the model.

---

## 6. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + what is loaded |
| `POST` | `/api/predict` | stateless single-query NLU |
| `POST` | `/api/chat` | context-aware conversational turn |
| `POST` | `/api/session` | create a session |
| `GET` | `/api/session/{id}` | session history |
| `DELETE` | `/api/session/{id}` | forget a session |

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

### `POST /api/chat`

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "Refund kab milega?"}'
```

Omit `session_id` and one is created and returned.

`/api/chat` returns the full pipeline result **plus** three fields from the response layer:

```json
{
  "session_id": "abc", "message": "...", "nlu": {...}, "context": {...}, "priority": {...},

  "response": "I'm really sorry about this, and I understand it needs urgent attention. I understand that your payment didn't go through the way it should have. We can help you check the payment and the order status, and get a deducted or failed payment traced. Could you share the order ID and the payment or UPI reference from your bank? I've marked this as high priority and a support specialist will pick it up.",
  "response_type": "ESCALATION",
  "requires_human": true
}
```

`/api/predict` is deliberately unchanged — it still returns only `message`, `nlu`, `priority`.

---

## 6a. Response generation

`backend/response/response_generator.py` turns a pipeline result into a reply. It is
**deterministic and template-based**: no model call, no external LLM, no randomness. The
same input always produces the same output.

All 15 intents have their own strategy (what we understood, what support can do, which
details would help, what to ask when they are missing). Tone follows sentiment — empathetic
for `NEGATIVE`, professional for `NEUTRAL`, appreciative for `POSITIVE` — and the closing
line follows priority, with `HIGH` acknowledging urgency and naming the handoff.

`response_type` is one of `ESCALATION`, `CALLBACK`, `CLARIFICATION`, `ACTION_CONFIRMATION`,
`ACKNOWLEDGEMENT`, `INFORMATION`, `APPRECIATION`, `APOLOGY`, `FALLBACK`, resolved in that
precedence: callback → escalation → clarification → the intent's default.

`requires_human` is `true` for `REQUEST_ESCALATE`, `REQUEST_CALLBACK`, any `HIGH` priority,
or any `HUMAN_AGENT` routing.

### Anti-hallucination

The generator may only restate what the pipeline actually extracted. It never states an
order status, a refund or delivery date, an amount, a transaction outcome, an account
detail or a policy. Two mechanisms enforce this, both covered by tests:

* **No template contains a digit.** Any number in a reply therefore came from a slot value,
  verbatim. `test_no_invented_numbers` asserts every digit run in a generated reply traces
  back to a slot.
* **Partial slots are dropped, never repaired.** Identifier- and quantity-like slot types
  (`ORDER_ID`, `AMOUNT`, `QUANTITY`, …) must contain a digit to be quoted at all. When the
  model returns `AMOUNT = "rupaye"` for *"Payment 799 rupaye cut ho gayi"*, the reply
  mentions no amount and asks for the payment reference instead — it never reconstructs
  "799" from the raw message.

When a needed detail is missing the reply asks for it, rather than assuming it.

---

## 7. Example predictions

Real output from `python demo.py` on this checkpoint:

| Input | Intent | Sentiment | Slots | Priority |
|---|---|---|---|---|
| `Payment 799 rupaye cut ho gayi but order show nahi ho raha` | PAYMENT_ISSUE (1.00) | NEGATIVE (1.00) | AMOUNT=`rupaye` | HIGH → HUMAN_AGENT |
| `Mera order #45231 5 din se deliver nahi hua` | COMPLAINT_DELIVERY (0.97) | NEGATIVE (0.99) | ORDER_ID=`#45231`, DATE=`5 din` | HIGH → HUMAN_AGENT |
| `Thanks for quick help` | FEEDBACK_POSITIVE (1.00) | POSITIVE (1.00) | — | LOW → AUTO_BOT |
| `Bahut hi worst experience raha` | FEEDBACK_NEGATIVE (1.00) | NEGATIVE (1.00) | — | HIGH → HUMAN_AGENT |
| `200 rupaye zyada charge kar liye please refund karo` | REQUEST_REFUND (0.99) | NEGATIVE (0.98) | AMOUNT=`200 rupaye` | HIGH → HUMAN_AGENT |
| `Complaint 2 hafte se pending hai koi sun nahi raha escalate karo` | REQUEST_ESCALATE (1.00) | NEGATIVE (1.00) | DATE=`2 hafte` | HIGH → HUMAN_AGENT |

Note row 1: the model tags only `rupaye` as `I-AMOUNT` and leaves `799` as `O`. That is the
model's own output, not a decoding bug — verified by dumping the raw per-word BIO tags.
Row 5 shows the same slot type decoded correctly as `B-AMOUNT I-AMOUNT` → `200 rupaye`.
This is consistent with the ~0.55 slot F1.

---

## 8. Context example

The trained model is single-utterance — it never sees conversation history. Rather than
concatenating the transcript and pushing it through XLM-R (which would put the encoder far
outside its training distribution and let an older turn override the current one), this
layer keeps the model's prediction for the **current** message and reports which earlier
facts are relevant.

```
Turn 1: "Maine order #45231 cancel kiya tha"
        intent  = REQUEST_CANCEL (0.97)
        slots   = ORDER_ID "#45231"
        context = used: false — "No prior turns in this session"

Turn 2: "Refund kab milega?"
        intent  = QUERY_REFUND_STATUS (1.00)
        slots   = []
        context = used: true
                  "Query looks like a follow-up (short_query, follow_up_marker,
                   no_slots_in_query). Carried forward previous_intent=REQUEST_CANCEL
                   and slots ['ORDER_ID']."
        inherited_slots = [{"type": "ORDER_ID", "value": "#45231",
                            "from_turn": 1, "source": "context"}]
```

**The decision rule.** A message uses context only when at least one ambiguity signal fires:

| Signal | Condition |
|---|---|
| `short_query` | ≤ 5 tokens after normalization |
| `anaphora_marker` | contains `iska`, `uska`, `wo`, `ye`, `same`, `it`, `that`, … |
| `follow_up_marker` | contains `kab`, `kitna`, `phir`, `dobara`, `status`, `abhi`, … |
| `low_intent_confidence` | intent confidence < 0.60 |
| `no_slots_in_query` | current message has no entities but earlier turns do |

No signals → the query is self-contained and history is ignored entirely.

**What it never does:** it never rewrites the model's intent or sentiment, and it never
invents a slot value. Inherited slots are tagged `"source": "context"` and carry the turn
they came from, so a caller can always tell an extracted entity from a remembered one. Only
stable identifier-like slot types are inheritable (`ORDER_ID`, `AMOUNT`, `PRODUCT_NAME`,
`PAYMENT_METHOD`, …).

History is bounded to the last **5 turns** per session (`HINGLISH_MAX_CONTEXT_TURNS`).

---

## 9. Priority routing

**This is a transparent rule engine, not a trained classifier.** No transformer was trained
on the `priority` column. Every weight is hand-set and every decision returns the list of
rules that fired.

```
score  = intent_base
       + sentiment_modifier
       + urgency_slot_bonus
       + repeated_intent_bonus
       + sustained_negative_bonus
       clamped to [0, 1]

HIGH   if score >= 0.75      -> HUMAN_AGENT
MEDIUM if score >= 0.35      -> ASSISTED_BOT
LOW    otherwise             -> AUTO_BOT

REQUEST_ESCALATE and REQUEST_CALLBACK always route to HUMAN_AGENT.
```

The intent base scores are **not invented** — they are the observed priority mix per intent
across all 2,989 dataset rows, collapsed with
`(HIGH*1.0 + MEDIUM*0.5 + LOW*0.0) / n`:

| Intent | Base | | Intent | Base |
|---|---|---|---|---|
| REQUEST_ESCALATE | 1.000 | | QUERY_REFUND_STATUS | 0.544 |
| COMPLAINT_DELIVERY | 0.968 | | REQUEST_CANCEL | 0.515 |
| PAYMENT_ISSUE | 0.944 | | REQUEST_CALLBACK | 0.461 |
| COMPLAINT_PRODUCT | 0.906 | | QUERY_ORDER_STATUS | 0.410 |
| COMPLAINT_SERVICE | 0.862 | | QUERY_PRODUCT_INFO | 0.238 |
| COMPLAINT_BILLING | 0.858 | | FEEDBACK_POSITIVE | 0.000 |
| RETURN_REQUEST | 0.778 | | | |
| FEEDBACK_NEGATIVE | 0.759 | | | |
| REQUEST_REFUND | 0.713 | | | |

Modifiers: `NEGATIVE +0.15`, `NEUTRAL 0.00`, `POSITIVE −0.30`; urgency-bearing slots
(`URGENCY`, `SLA`, `THREAT`, `ESCALATION_LEVEL`, `WAIT_TIME`, `COMPLAINT_COUNT`, …) `+0.10`
each capped at `+0.20`; repeating an intent within a session `+0.10`; two or more negative
turns in recent history `+0.10`.

`test_priority_engine.py` asserts the engine reproduces the dataset's **majority priority
label for 14 of 15 intents** at their modal sentiment. The one deliberate deviation is
`REQUEST_REFUND` + `NEGATIVE`: the dataset splits it 74 HIGH / 100 MEDIUM, and the engine
escalates an angry refund request to HIGH. Neutral refund requests stay MEDIUM.

All thresholds and weights live in `PriorityRules` and can be swapped without touching code.

---

## 10. Known limitations

**Model quality** — these are properties of the trained checkpoint, not of this backend:

1. **Intent accuracy is 83.95%.** The worst confusion is `PAYMENT_ISSUE` → `COMPLAINT_BILLING`
   (9 of 37 validation examples). Use `intent_confidence` to gate automated actions.
2. **Slot F1 is ≈0.55.** Slots are frequently partial — `AMOUNT` on
   `"Payment 799 rupaye cut ho gayi"` returns `rupaye`, dropping `799`. Do not rely on slot
   extraction for anything transactional without human confirmation.
3. **The model overfit after epoch 6.** Train loss kept falling to 0.35 while val loss rose
   to 1.52. The saved weights are the epoch-6 early-stopping point.
4. **Sentence-final punctuation shifts predictions.** The training data has none, so a
   trailing `.` or `?` is out of distribution: `"Maine order cancel kiya tha."` predicts
   `COMPLAINT_SERVICE` (0.65) while `"Maine order cancel kiya tha"` predicts
   `REQUEST_CANCEL` (0.48). Stripping punctuation would change the preprocessing the model
   was trained on, so it is **documented rather than fixed**. Callers who control input
   formatting should prefer unpunctuated text.
5. **No CRF**, so BIO transitions are not constrained. A stray `I-X` with no opening `B-X`
   is recovered leniently by starting a new span rather than being discarded.

**Backend:**

6. **Sessions are in-process memory.** They are lost on restart and are not shared between
   workers. Redis or a table is the first thing to add before scaling past one process.
7. **CPU inference is ~0.4–0.6 s per request** with `padding="max_length"` (128 tokens),
   matching training exactly. Dynamic padding (`HINGLISH_PAD_TO_MAX_LENGTH=0`) is
   numerically equivalent because padded positions are attention-masked, and is faster for
   short inputs. Batching would help more. A GPU helps most.
8. **No batch endpoint.** One sentence per request.
9. **No authentication, rate limiting, or request-size limits.** This is a demo service.
10. **Original-casing recovery is index-based.** It relies on every normalization
    substitution being word-count preserving (verified by
    `test_word_count_is_preserved_by_normalization`). If that ever stopped holding, slot
    values silently fall back to the lowercased form — `value_normalized` always carries the
    exact string the model actually saw.
11. **Context interpretation is heuristic**, not learned. The token thresholds and marker
    word lists in `InterpreterConfig` were chosen by hand and are not tuned against
    annotated data.
12. **Cosmetic warning on startup.** transformers 4.57 prints
    *"incorrect regex pattern … will lead to incorrect tokenization"* referencing a Mistral
    issue. It is spurious for this tokenizer: the vendored tokenizer was verified to produce
    **identical `input_ids` and `word_ids`** to the Hub tokenizer on every test sentence.
    A stray TensorFlow import also logs `oneDNN` notices; neither affects results.

---

## 11. Configuration

All settings are environment-variable overridable (see `backend/config.py`):

| Variable | Default |
|---|---|
| `HINGLISH_CHECKPOINT` | `./best_model.pt` |
| `HINGLISH_METADATA` | `./model_metadata2.json` |
| `HINGLISH_TOKENIZER_DIR` | `./assets/xlm-roberta-large` |
| `HINGLISH_DEVICE` | `cuda` if available, else `cpu` |
| `HINGLISH_PAD_TO_MAX_LENGTH` | `1` |
| `HINGLISH_MAX_CONTEXT_TURNS` | `5` |
