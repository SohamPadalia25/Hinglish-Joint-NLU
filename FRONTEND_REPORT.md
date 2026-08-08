# Frontend — Report

**Project:** Hinglish Joint NLU (XLM-RoBERTa Large — Intent + Sentiment + Slot Filling)
**Date:** 2026-08-08
**Scope:** React + Vite frontend, connected to the existing backend.

**Constraints honoured — none of the following were modified:**
`best_model.pt` · the XLM-R architecture · preprocessing · `inference.py` · context manager ·
priority engine · response generator · every existing API response structure.

The only backend change is CORS configuration, which the brief explicitly permitted. No ML
logic was touched, and the backend suite was re-run afterwards: **198/198 passing**.

The frontend contains **no prediction logic**. Intent, sentiment, slots, priority, context and
the reply text are all read from `POST /api/chat` and rendered as received. There are no
canned responses, no hardcoded predictions, and no fabricated API payloads anywhere in the
client.

---

## 1. Frontend file structure

```
frontend/
├── index.html                     favicon, meta description, #root
├── package.json                   React 19.2.8 · Vite 8.2.1 · @vitejs/plugin-react 6.0.5
├── package-lock.json
├── vite.config.js                 port 5173, strictPort
├── .env                           VITE_API_BASE_URL=http://127.0.0.1:8000
├── .env.example
├── .gitignore
├── public/
│   └── favicon.svg
└── src/
    ├── main.jsx                   React root
    ├── App.jsx                    session, messages, send / demo / reset orchestration
    ├── App.module.css             2-column grid, stacks at 940px
    ├── services/
    │   └── api.js                 sendMessage · healthCheck · clearSession · ApiError
    ├── utils/
    │   └── format.js              toPercent · humanise · newSessionId · formatParameters
    ├── styles/
    │   └── global.css             CSS custom-property design tokens
    └── components/                12 components, each with its own .module.css
        ├── Header.jsx             title, backend status, session id, New Conversation
        ├── ChatPanel.jsx          scrolling log, auto-scroll, typing indicator
        ├── MessageBubble.jsx      user / bot bubbles, per-turn tags, click to re-analyse
        ├── ChatInput.jsx          textarea, Enter to send, disabled while waiting
        ├── ExampleQueries.jsx     6 example buttons + Try Context Demo
        ├── AnalysisPanel.jsx      the full AI analysis rail
        ├── ConfidenceBar.jsx      labelled 0–1 progress bar
        ├── EntityList.jsx         typed slot spans (never raw BIO)
        ├── ContextCard.jsx        Context Used YES/NO, previous context, reason
        ├── PriorityCard.jsx       HIGH/MEDIUM/LOW + routing + score
        ├── ModelInfo.jsx          collapsible model card
        └── ErrorBanner.jsx        user-facing failure message
```

**No Tailwind.** Styling is CSS Modules over a design-token file (`global.css`).

### Stack

| | |
|---|---|
| Node | v22.13.0 |
| npm | 11.1.0 |
| React | 19.2.8 |
| Vite | 8.2.1 |
| Plugin | @vitejs/plugin-react 6.0.5 |

Production build output:

```
dist/index.html                   0.41 kB │ gzip:  0.27 kB
dist/assets/index-*.css          16.89 kB │ gzip:  4.09 kB
dist/assets/index-*.js          210.42 kB │ gzip: 66.58 kB
✓ built in 264ms
```

---

## 2. Backend files modified

Only CORS. No ML logic, no response-shape changes.

| File | Change |
|---|---|
| `backend/config.py` | added `CORS_ORIGINS` — Vite dev and preview origins, overridable via `HINGLISH_CORS_ORIGINS` |
| `backend/api.py` | added `CORSMiddleware` (import + one `add_middleware` call) |
| `README.md` | new §4a documenting frontend setup |

Everything else in the backend is byte-for-byte as it was.

---

## 3. Installation commands

```bash
# Backend (once)
cd "d:\Soham Clg\major project"
pip install -r requirements.txt

# Frontend (once)
cd "d:\Soham Clg\major project\frontend"
npm install
cp .env.example .env        # VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## 4. Backend start command

```bash
cd "d:\Soham Clg\major project"
python run_api.py --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete` — the 2.2 GB checkpoint loads first (~7–12 s).

---

## 5. Frontend start command

```bash
cd "d:\Soham Clg\major project\frontend"
npm run dev
```

---

## 6. Frontend URL

**http://localhost:5173**

> ⚠️ Use `localhost`, **not** `127.0.0.1`. Vite 8 binds `localhost`, which resolves to IPv6
> on Windows, so `http://127.0.0.1:5173` returns an empty response. Both hostnames are in the
> CORS allow-list, so either works once the page has loaded.

---

## 7. Backend URL

**http://127.0.0.1:8000** — interactive API docs at `/docs`.

Configured in the frontend through `VITE_API_BASE_URL`.

---

## 8. CORS changes

`backend/config.py`:

```python
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "HINGLISH_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]
```

`backend/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)
```

Both dev ports (5173) and preview ports (4173) are listed, under both hostnames, because
`localhost` and `127.0.0.1` are distinct browser origins.

**Verified with real requests:**

```
OPTIONS /api/chat   Origin: http://localhost:5173
  -> 200, access-control-allow-origin: http://localhost:5173
          access-control-allow-methods: GET, POST, DELETE, OPTIONS

POST /api/chat      Origin: http://localhost:5173
  -> 200, access-control-allow-origin: http://localhost:5173
```

---

## 9. Test results

All tests were run **in a real browser** — headless Microsoft Edge driven by `playwright-core`
— against both live servers. Not curl, not mocks. The browser executed the actual React app
and made the actual `fetch` calls.

### The four required tests

| Test | Intent | Sentiment | Entities | Priority / Routing | Result |
|---|---|---|---|---|---|
| **1** `Mera order #45231 5 din se deliver nahi hua` | COMPLAINT_DELIVERY 97.1% | NEGATIVE 98.9% | ORDER_ID `#45231` 98.5%<br>DATE `5 din` 90.3% | **HIGH** 100.0% → HUMAN AGENT | PASS |
| **2** `200 rupaye zyada charge kar liye please refund karo` | REQUEST_REFUND 99.0% | NEGATIVE 98.4% | AMOUNT `200 rupaye` 94.7% | **HIGH** 86.3% → HUMAN AGENT | PASS |
| **3** `Maine order #45231 cancel kiya tha` → `Refund kab milega?` | QUERY_REFUND_STATUS 100.0% | NEUTRAL 97.8% | ORDER_ID `#45231` **FROM CONTEXT** | MEDIUM 54.4% → ASSISTED BOT | PASS — `context.used = true`, previous order displayed |
| **4** `Thanks for quick help` | FEEDBACK_POSITIVE 99.9% | POSITIVE 100.0% | none | **LOW** 0.0% → AUTO BOT | PASS |

Test 3 detail — the analysis panel showed:

```
CONTEXT USED          YES
Current query interpreted using the previous conversation.
PREVIOUS CONTEXT      Order #45231     REQUEST_CANCEL
```

and the entity card was badged **FROM CONTEXT**, visually distinct (amber) from freshly
extracted entities (blue).

### Functional checks

| Check | Result |
|---|---|
| Bot replies rendered exactly as returned by the backend | PASS |
| Analysis panel updates after every assistant response | PASS |
| Confidence shown as `97.1%`, not `0.971` | PASS |
| Entities shown as `ORDER_ID` / `#45231` — never raw BIO tags | PASS |
| Enter to send; Shift+Enter for newline | PASS |
| Send button disabled while waiting | PASS |
| Loading indicator during inference | PASS |
| Auto-scroll to newest message | PASS |
| No page reload on send | PASS |
| One session id generated at startup, reused for all turns | PASS |
| New Conversation clears messages and starts a new session | PASS |
| Try Context Demo sends both turns in the same session | PASS |
| Model Information collapsible, populated from live `/health` | PASS — 560M, CPU, 128 tokens |
| Responsive at 820px (single column) | PASS |
| Error handling with backend unreachable | PASS |

### Error handling

With `/api/chat` blocked at the network layer, the UI displayed:

> **Could not complete that request**
> Cannot reach the backend at http://127.0.0.1:8000. Make sure it is running, then try again.

No Python traceback, no status code, no exception text. `services/api.js` converts every
failure class — offline, timeout, 422, 404, 5xx, malformed JSON — into a plain sentence before
it reaches a component.

### Browser console

**Clean.** The only entry across the whole run is the `net::ERR_FAILED` from the request I
deliberately aborted to test error handling.

### Defects found and fixed during verification

Both surfaced only by running a real browser; neither was visible from curl.

1. **Spurious 404 on New Conversation.** The client generates the session id, but the backend
   only materialises a session on its first message. Clicking New Conversation on an empty
   chat issued `DELETE /api/session/{id}` for a session that did not exist yet. Now skipped
   when there are no messages.
2. **Missing favicon.** `/favicon.ico` 404'd on every page load. Added `public/favicon.svg`
   and a `<link rel="icon">`; removed the two unreferenced scaffold assets (`vite.svg`,
   `icons.svg`).

### Scaffold note

`npm create vite@latest frontend -- --template react` produced a **vanilla TypeScript**
project — create-vite 9 changed its CLI and ignored the `--template react` flag, generating
`src/main.ts` and `src/counter.ts`. The stub was removed and React plus `@vitejs/plugin-react`
were installed manually. Worth knowing if the project is ever re-scaffolded.

---

## 10. Two-minute demo sequence

| Time | Action | What to point at |
|---|---|---|
| 0:00 | Open **http://localhost:5173** | Title, *Context-Aware Code-Switched NLU*, green **Backend connected**, session id |
| 0:10 | Click example → **Delivery complaint** | Reply appears in chat; the whole right rail fills in |
| 0:25 | Walk the analysis rail top to bottom | COMPLAINT_DELIVERY **97.1%** → NEGATIVE **98.9%** → entities **ORDER_ID `#45231`**, **DATE `5 din`** → Context **NO** → **HIGH PRIORITY / HUMAN AGENT** |
| 0:45 | **New Conversation**, then **Refund request** | AMOUNT `200 rupaye` extracted; different intent, same pipeline, still HIGH |
| 1:00 | **New Conversation**, then **Try Context Demo** | Two turns auto-send in one session |
| 1:15 | Turn 1 | `REQUEST_CANCEL`, order id captured, MEDIUM |
| 1:25 | **Turn 2 — `Refund kab milega?` — the key slide** | **Context Used YES** · *"Current query interpreted using the previous conversation"* · **Previous context: Order #45231** · entity badged **FROM CONTEXT**. Three words carry a full order reference forward. |
| 1:45 | **New Conversation**, then **Positive feedback** | FEEDBACK_POSITIVE / POSITIVE / **LOW → AUTO BOT** — priority routing is doing real work, not decoration |
| 1:55 | Expand **Model Information** | XLM-RoBERTa Large, 560M, running on CPU — plus the footnote that priority is a rule layer, not a trained classifier |

**Presenter tips**

* Start the backend a minute early — the checkpoint takes ~7–12 s to load, and the first
  inference is the slowest.
* Each reply takes roughly 0.5 s on CPU; the typing indicator covers it.
* Any bot bubble can be clicked to bring its analysis back into the panel — useful for
  jumping between turn 1 and turn 2 during the context explanation.
* If a demo machine has no GPU, nothing changes; CPU is the tested path.

---

## 11. Design decisions worth defending

1. **The backend is the only source of truth.** `services/api.js` is the sole network module.
   No component computes a label, a confidence or a reply. Removing the backend leaves a UI
   that shows an error, not a demo.
2. **Routing is displayed as returned, not derived from the priority label.** They usually
   agree (HIGH → HUMAN_AGENT), but a callback request is MEDIUM and still routes to a human.
   Deriving it in the UI would have hidden that.
3. **Context reason is shown in plain language, with the raw reason one click away.** The
   backend's own string names internal signals (`short_query`, `follow_up_marker`), which is
   noise during a demo. The card leads with *"Current query interpreted using the previous
   conversation"* and keeps the exact backend text behind **Show detection detail** — nothing
   is hidden, nothing is cluttered.
4. **Colour comes from CSS classes; only bar widths are dynamic.** Priority, sentiment and
   confidence colours are all class-driven, per the brief. The single dynamic value is the
   fill percentage, passed as a CSS custom property.
5. **Inherited slots are visually distinct.** Entities carried from context render amber with
   a **FROM CONTEXT** badge, so a remembered value is never mistaken for one extracted from
   the current message — matching the backend's own `source: "context"` marking.

---

## 12. Known limitations

1. **Sessions are in-process on the backend.** Restarting the API loses all history; the
   frontend keeps its own message list, so the UI and server can drift after a restart. Click
   New Conversation after restarting the backend.
2. **No message persistence in the browser.** A refresh clears the conversation and starts a
   new session.
3. **CPU inference is ~0.5 s per message.** Fine for a demo; the typing indicator covers it.
4. **Single-worker backend only.** Each uvicorn worker would load its own 2.2 GB copy of the
   model, and in-memory sessions are not shared between workers.
5. **No authentication or rate limiting.** This is a demonstration application.
6. **Frontend has no automated test suite.** Verification was done through scripted browser
   runs rather than committed component tests. Adding Vitest + Testing Library would be the
   natural next step if the project needs frontend regression coverage.

---

## Status

Frontend complete and verified end-to-end in a real browser against the live backend.
All four required tests pass. Backend suite still 198/198.
