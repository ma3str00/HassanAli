# IPDR Analyst: project overview

## In one paragraph

IPDR Analyst is a web app that lets telecom analysts ask questions about internet usage records in plain
language (English, Urdu or Roman Urdu) and get a correct answer as a table, a chart and a short written
conclusion. Behind the scenes an AI agent turns the question into ClickHouse SQL, runs it on a
1.5-billion-row table, checks the result with rules written in code, and fixes its own mistakes before
anything is shown. A second module, **Explore**, gives raw records and summaries with filters and no AI,
for when an analyst wants to look up one number, IP or website directly.

**The 30-second version:** "Analysts can't write SQL, and the data is huge and messy. I built a chat app where
they ask 'which apps did Karachi use most last week?' and get a verified table and chart in under a minute. An
LLM writes the SQL, but code checks every answer: dates, places, user counting, empty columns, and the app
names that are really stored. It works with any AI provider, including a free local model."

---

## What IPDR data is

IPDR = **Internet Protocol Detail Record**. Every time a phone or device opens a connection to an online
service, the network records one row: who (subscriber number, IMSI, IMEI), when (start and end time), from
where (client IP, city, cell tower), to what (server IP, website name, app label such as TikTok or WhatsApp)
and how much (bytes and packets each way). Telecom operators keep these records for traffic analysis,
planning and lawful requests.

In this project:

| Fact | Value |
|---|---|
| Table | `wms_dist.ntc_ssl_log` |
| Rows | about 1.57 billion (one month, 11 Oct to 10 Nov 2026) |
| Size on disk | about 93 GiB (compressed) |
| Columns | 60, of which 25 are always empty |
| App labels | about 430 distinct (TikTok alone has 5: TikTok, TikTok_Live, TikTok_Upload…) |
| Locations | about 165 distinct, written as "Karachi Pakistan" |
| Time zone | stored as Unix seconds, shown in Pakistan time (UTC+5) |

Messy parts the app has to handle (good talking points):

- **Unidentified users.** About half of all rows have no real subscriber: the ID holds `<ip>_<port>` instead, which
  is different for nearly every row. A naive `COUNT(DISTINCT subscriber_id)` would count most connections as
  separate people. The app counts users as identified subscribers only:
  `uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0)`.
- **Background traffic.** Labels like SSL, QUIC and GoogleAPI are generic encrypted traffic, not apps a person
  chose. Rankings leave them out unless asked.
- **Stored names differ from spoken names.** WhatsApp is stored as `WhatsAPP`. "Adult sites" or "gambling" are
  not labels at all, so they are matched by groups of labels plus website patterns.
- **ISP vs SIM operator.** The ISP (PTCL, Nayatel, StormFiber, Jazz…) is `client_asn`, filled on 98% of rows; `ipdr/isps.yaml` maps the names people use to the stored names. The SIM operator code (mcc/mnc) exists only for mobile numbers (30%) and is used only when a question says SIM.

---

## Tech stack

| Layer | Technology | Why it was chosen |
|---|---|---|
| Database | **ClickHouse 26.10** (MergeTree engine) | Column store built for billions of rows; scans a month of data in seconds |
| Backend | **Python 3.14** + **FastAPI** + **Uvicorn** | Fast to build, async-friendly, automatic request validation with Pydantic |
| AI (cloud) | **Anthropic Claude** via the official SDK | Best accuracy; prompt caching cuts cost and latency |
| AI (any other) | **OpenAI-compatible HTTP API** | One adapter covers OpenAI, Gemini, Groq, DeepSeek, Mistral, OpenRouter, Together and local Ollama |
| AI (current default) | **Qwen2.5-Coder 14B** on **Ollama**, hosted on a free **Kaggle** GPU, exposed through a **Cloudflare tunnel** | Free, and the data never goes to a commercial AI provider |
| SQL safety | **sqlglot** (SQL parser) | Parses every query into a tree, so the safety checks look at the real structure and not at text |
| App storage | **SQLite** | Chats, answer cache, uploaded dictionaries, team terms and verified examples; no extra server |
| Config | **YAML** files + **.env** (python-dotenv) | Business rules and app groups are editable without touching code |
| Frontend | **Vanilla JavaScript** (ES modules), HTML, CSS | No build step, no framework, loads instantly |
| Charts | **Apache ECharts 5.6** (bundled locally) | 19 chart types, works offline, light and dark themes |
| Voice | Browser speech API, or **Whisper**-class models (OpenAI `gpt-4o-transcribe`, Groq `whisper-large-v3`) | Spoken questions in English or Urdu |
| Excel upload | **openpyxl** | Data dictionaries can be uploaded as Excel |
| Tests | **pytest** + httpx test client | 396 tests, run offline in about 11 seconds (AI and database are faked) |

Size: about 6,700 lines of Python, 3,700 lines of JavaScript and 800 lines of CSS and HTML.

---

## Architecture

```
 Browser (vanilla JS)                     FastAPI server (Python)                         Outside
 ────────────────────                     ───────────────────────                         ───────
 Chat page  ──── question ────────▶  main.py ──▶ small talk / display command? ──▶ instant reply
                                        │
                                        ▼
                                     agent.py  (the AI loop)
                                        │  builds context: rules, dictionary, terms,
                                        │  date periods, resolved names, past examples
                                        ▼
                                     llm/ adapters ─────────────────────────────▶  Claude / Ollama / OpenAI…
                                        │  model calls tools:
                                        │   find_values  (look up real stored names)
                                        │   run_sql      (test a query)
                                        │   submit_answer
                                        ▼
                                     sql_guard.py ── checks SQL is one safe SELECT
                                        ▼
                                     db.py ───────── read-only HTTP query ─────────▶  ClickHouse
                                        ▼
                                     code checks (dates, places, users, names…)
                                        │  wrong? → sent back to the model to fix
                                        ▼
 Answer card  ◀── streamed steps + final answer (table, chart options, SQL)
 (table, chart, conclusion)

 Explore page ──── filters ──▶  explore.py ── builds parameterised SQL (no AI) ──▶  ClickHouse
```

Answers are **streamed**: the browser sees each step ("Looking up app names…", "Running the query…") while it
happens. This also keeps the connection alive through Cloudflare's 100-second timeout.

---

## What each part does

### Backend (`ipdr/`)

| File | What it does |
|---|---|
| `main.py` | The web server: all JSON endpoints (chat, conversations, Explore, dictionary, terms, settings), streaming answers to the browser, audit log |
| `agent.py` | The heart of the app. Runs the question-to-SQL loop with the model, then runs about 20 code checks on each answer and sends it back with a precise note when one fails. Also writes and fact-checks conclusions, and caches answers |
| `context.py` | Everything the model is told: instructions, tool definitions, the data's date range, ready-made periods ("last week" = exact dates), earlier turns of the chat. Also recognises small talk ("thanks"), display commands ("draw a pie chart") and retries ("try again") so they never reach the AI |
| `concepts.py` + `app_concepts.yaml` | About 35 meaning groups (social media, video streaming, VPN, adult content, gambling, dating…) with English, Urdu and Roman Urdu synonyms, mapped to the exact stored app labels. This is how "kon si app adult log zyada use karte hain" is understood |
| `semantic.yaml` | Built-in business rules, measures (users, connections, volume), column meanings and worked example questions with correct SQL |
| `dictionary.py` | Merges the built-in field descriptions, any uploaded data dictionary and the live table columns. Works offline if the database is down |
| `retrieval.py` | Finds the verified examples most similar to a new question (word overlap with domain synonyms) and shows them to the model. Every "Mark as correct" click adds one, so accuracy grows with use |
| `sql_guard.py` | Safety gate: allows exactly one SELECT on the IPDR table, blocks other tables, table functions, `SETTINGS` and `FORMAT`, adds a row limit, and requires a real time filter so nobody scans the whole table by accident |
| `db.py` | ClickHouse client over HTTP: read-only mode, limits on run time and memory, query parameters (user text is never pasted into SQL), 10-minute result cache |
| `explore.py` | Explore module: turns filters (is, contains, starts with, one of, between…) into parameterised SQL; reads typed look-ups ("0325-3047295", "tiktok karachi", "jazz") into filters |
| `charts.py` | Decides which charts fit a result from its column types (time, category, measure, coordinates), from 19 types. Never stacks or shows user counts as shares, since one person uses several apps |
| `llm/` | Provider adapters: `anthropic_llm.py` (SDK, prompt caching, thinking effort, fallback model) and `openai_llm.py` (any OpenAI-compatible API, streaming, recovers tool calls that small models write as plain text) |
| `net.py` | Shared HTTP connection pool with fast failover: skips a dead network address in 3 seconds instead of waiting 21 |
| `store.py` | SQLite: conversations, messages, answer cache, dictionaries, team terms, verified examples |
| `voice.py` | Speech-to-text through Whisper-class APIs, with a hint listing real app and city names |
| `config.py` | Settings from `.env`: database login, query limits, default AI provider |

### Frontend (`ipdr/web/`)

| File | What it does |
|---|---|
| `chat.js` | Chat page: conversations sidebar, sending questions, live progress steps, "Ask again" for interrupted questions |
| `answer.js` | One answer card: "Understood as", assumptions, table, chart on request, conclusion on request, copy, CSV, Mark as correct, the SQL used |
| `charts.js` | Draws the 19 chart types with ECharts, using the theme colours; grouped chart picker |
| `explore.js` | Explore page: look-up bar, time presets, filter cards, records/summary views, click-to-filter, record drawer, saved views, CSV export |
| `table.js` | Result tables with formatted numbers and clickable cells |
| `dictionary.js`, `terms.js`, `settings.js` | Data dictionary upload, the team's own terms ("KHI" = Karachi, "Jazz" = its ISP name), AI provider and theme settings |
| `voice.js` | Microphone button: browser speech or recorded audio sent to Whisper |
| `api.js`, `util.js`, `theme.js` | Server calls, formatting helpers, light/dark/system theme |

---

## How one question is answered (step by step)

Example: *"pichle hafte karachi me sab se zyada kon si apps use hui?"*
(Roman Urdu: "which apps were used most in Karachi last week?")

1. **Shortcut checks (no AI).** Is it small talk, a display command ("draw a chart"), a retry, a request to
   change data (refused), or something the data does not hold? These are answered instantly.
2. **Answer cache.** Has the same question been answered for the same data range? Then it returns in
   milliseconds.
3. **Context building.** The model receives the rules, the data dictionary, team terms, the exact dates for
   "last week" (the data is not live, so "last week" means the last 7 full days in the data), names already
   resolved by code ("Karachi" → `'Karachi Pakistan'`), and the closest verified examples.
4. **The model works.** It rewrites the question in clear English ("Understood as"), writes SQL, and submits
   its answer with a title, assumptions and a chart suggestion.
5. **Safety gate.** `sql_guard.py` parses the SQL and blocks anything unsafe.
6. **Run.** The query runs read-only on ClickHouse.
7. **Code checks.** About 20 rules compare the SQL with the question, for example:
   - the dates match the period asked for;
   - a city the user never named was not added, and a named one was not dropped;
   - users are counted the right way;
   - background traffic is left out of app rankings;
   - no always-empty column is used;
   - an app the user named appears in the result;
   - "all gambling sites" includes the website patterns, not one label.
   A failed check goes back to the model with a precise note, and it fixes the SQL.
8. **Self-check** (for smaller models): the model compares its SQL against its own understood question once
   more.
9. **Shown.** The table appears first. The chart and the written conclusion appear only when the analyst asks.
   Every figure in a conclusion is checked against the table.

---

## Database details (ClickHouse)

- **Engine:** MergeTree, **partitioned by day** of `start_time`, **sorted by** `(start_time, subscriber_id, log_id)`.
- Because of that layout, every query must filter on `start_time`. ClickHouse then reads only the days needed
  (partition pruning). The SQL guard enforces this.
- Summing a whole month by app (1.57 billion rows) takes about 7 seconds. Counting the rows takes 0.01 seconds,
  because ClickHouse keeps counts in its metadata.
- **Access:** the app connects over ClickHouse's HTTP interface (port 8123) through an SSH tunnel to the database
  server. Every query runs with `readonly=2`, a time limit (10 minutes), a memory limit (8 GB) and a result
  limit. There is no limit on the date range.
- **App data** (chats, cache, examples) lives in a separate small SQLite file, so the app never writes to
  ClickHouse.

---

## The AI part

- **Tool use / function calling.** The model does not just write text; it calls tools: `find_values` to look up
  real stored names, `run_sql` to test, `submit_answer` with a fixed JSON schema, and `confirm_answer` for the
  self-check.
- **Model-agnostic.** Works with Claude, GPT, Gemini, Llama, Qwen or anything with an OpenAI-compatible API.
  Each analyst can bring their own key in Settings. Keys stay in the browser and are never saved on the server.
- **Small-model support.** The free 14B model is much weaker than Claude, so code does more of the work: ready-made
  date conditions, names resolved in advance, tool calls recovered from plain text, deterministic fallbacks
  (for example, "how many users visit gambling sites" can be built entirely by code if the model refuses).
- **Grounded conclusions.** The written conclusion is checked figure by figure against the table. A number
  that cannot be found is sent back once; if it is still wrong, the conclusion shows a warning.
- **Learning from use.** "Mark as correct" saves the question and SQL as a verified example, which future
  similar questions are shown.

---

## Performance work (good story to tell)

The first version took **4 to 9 minutes** per answer with the free model. Now typical answers take
**25 to 110 seconds**, and repeated questions return instantly. What was done:

1. **Found a hidden 21-second delay per AI call.** Cloudflare's tunnel address resolved to several IPs and one
   was unreachable from this network; Python waited 21 seconds for it every call. A custom connection layer
   (`net.py`) tries each address for 3 seconds, remembers dead ones, and reuses connections (keep-alive).
   Calls dropped from about 22 seconds to 1–2.
2. **Fewer model turns.** Names and dates are resolved by code and handed to the model, so it rarely needs to
   look them up. The self-check runs only for complex questions.
3. **Answer cache** keyed by the normalised question and the data range.
4. **Query result cache** (10 minutes) for repeated SQL.
5. **Streaming** so the analyst sees progress immediately.

---

## Safety and privacy

- Read-only database access, enforced twice (ClickHouse `readonly=2` and the SQL parser).
- User text never becomes SQL in Explore; values are sent as query parameters (no SQL injection).
- Requests to change data ("delete", "update", "drop") are refused before reaching the AI.
- API keys stay in the browser. Questions and queries are written to an audit log, keys never.
- With the local model, no IPDR data is sent to a commercial AI provider.

---

## Testing

- **396 automated tests** (pytest), fully offline: the AI is replaced by scripted fake replies and the database
  by fixed results. They cover the SQL guard, every code check, chart rules, Explore filters, Urdu and Roman Urdu
  phrasing, small-talk detection and the API.
- **Soak testing:** a script fired hundreds of realistic messy questions at a separate test server and logged
  every wrong answer. Each failure became a new code check and a new test.

---

## Questions you may be asked

**Why not let the AI answer directly from the data?**
The table has 1.5 billion rows; no model can read that. The model writes the query, the database does the
counting, so numbers are exact.

**How do you stop the AI from making things up?**
The numbers always come from the database. The model only writes SQL, and about 20 code checks plus a SQL
parser verify that SQL before it runs and after. Conclusions are checked figure by figure.

**Why ClickHouse and not MySQL or PostgreSQL?**
It is a column store: an aggregate over a month (1.5 billion rows) takes seconds, where a row store would take
minutes or hours. It also compresses the data to about 93 GiB.

**Why vanilla JavaScript and not React?**
The UI is a handful of pages. Without a framework there is no build step, nothing to install, and the page loads
instantly. ES modules keep the code organised.

**Why SQLite for chats?**
It needs no server, is a single file, and the load (a team of analysts) is small. ClickHouse stays read-only.

**What happens if the AI is down?**
Explore keeps working, since it needs no AI. Cached answers still return. Chat shows a clear message
(for example, that the Kaggle tunnel needs restarting).

**How does it understand Urdu?**
Three layers: the model itself, a team terms list (short forms, operators, periods), and the meaning groups in
`app_concepts.yaml` with Urdu and Roman Urdu synonyms.

**What would you add next?**
Logins and roles; a proper read-only database user; a base map for the location chart; scheduled reports.

---

## Run it

```
pip install -r requirements.txt
copy .env.example .env          (database login, default AI provider)
ssh -N -L 8123:localhost:8123 <database-server>     (tunnel to ClickHouse)
python -m ipdr.main
```

Listens on port 8000 on every network interface (`0.0.0.0`). Open http://127.0.0.1:8000 on this machine, or http://<this-machine-ip>:8000 from another machine. Tests: `python -m pytest -q`.
