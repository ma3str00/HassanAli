# IPDR Analyst

A chat app for asking questions about the IPDR records in plain English. The AI writes the
ClickHouse SQL, tests it and runs it. Each answer shows its assumptions, a table, the charts that
fit the result, the SQL (editable), and on request a short written conclusion. Chats are saved.

## Run it

```
pip install -r requirements.txt
copy .env.example .env        (fill in the ClickHouse login; an AI key is optional here)
python -m ipdr.main
```

That listens on port 8000 on every network interface (0.0.0.0). Open http://127.0.0.1:8000 on this machine, or http://<this-machine-ip>:8000 from another machine. Set IPDR_HOST=127.0.0.1 in .env to keep it local only. Go to **Settings**, choose an AI provider and paste your key.

Tests: `python -m pytest -q` (offline: the AI and, where needed, the database are replaced by fakes).

## What is in the app

| Page | What it does |
|---|---|
| Chat | Ask in plain language only; the SQL is written, tested and run behind the scenes. Each answer shows its assumptions and the result table, with **Draw chart** and **Write conclusion** on request, a CSV download, and **Mark as correct**. The query stays available under a collapsed "Query used". Conversations are kept in the sidebar (search, rename, delete), and follow-ups use the earlier questions and results. |
| Explore | Records without the AI, in two panels. On top, **Look up** turns what you type into filters: a phone number in any form, an IP, a website, an IMSI/IMEI, a city, an app, a kind of app or an operator ("0325-3047295", "tiktok karachi", "jazz"); a look-up of one subscriber, device or server searches the last 7 days. On the left: time (presets or exact), filter cards (is, contains, starts with, one of, between, empty…) and switches to leave out unidentified subscribers or background traffic (SSL, GoogleAPI…). On the right: Records or Summary with their own toolbar (fields, sort, rows; group by, measures), which re-runs at once; typed filter values run with Run or Enter. Records show the total that matches. Click any value to filter on it, leave it out, or open the full record (every field, with units); a summary row opens the records behind it. Export up to 200,000 rows as CSV, or save named views. |
| Data dictionary | Upload a file describing each field (CSV, Excel, JSON, YAML, Markdown, text). It is checked against the real table columns. See what the AI is told about every field, and download the current dictionary as a template to edit in Excel. |
| Terms | What the team's own words mean in the data, in plain language: short forms (KHI, YT), ISPs and operators (PTCL, Nayatel, Jazz = their network names in client_asn), periods (night, last week), measures (usage = data volume). The AI reads the list with every question. Starts with a dozen common terms. |
| Settings | AI provider, model, key and answer quality; light, dark or system theme. |

### Voice input

A microphone button in the question box. The spoken question is turned into text and put in the box
for checking (it is sent automatically only if that is switched on in Settings). Engines:

- **Browser** (Chrome, Edge): free, words appear as you speak. Fine for plain English, weaker on names.
  Chrome sends the audio to Google, Edge to Microsoft.
- **Whisper-class via OpenAI** (`gpt-4o-transcribe`), **Groq** (`whisper-large-v3`), or any
  OpenAI-compatible `/audio/transcriptions` endpoint (for example a Whisper server on your own network).
  The audio is recorded in the browser, stops by itself after a pause, and is transcribed with a hint
  listing the app and city names in the data, the chosen language (English or Urdu), and temperature 0.

Both engines apply the same name fixes ("tick tock" → TikTok, "whats app" → WhatsApp, "i m o" → IMO …).
The microphone needs https or localhost.

### AI providers (bring your own key)

- **Anthropic (Claude)** through the official SDK. Default model `claude-opus-5-5`, with prompt
  caching, adjustable thinking effort, and an automatic fallback model if a request is declined.
- **Any OpenAI-compatible API:** OpenAI, Google Gemini, OpenRouter, Groq, DeepSeek, Mistral,
  Together, local Ollama, or a custom base URL.

Keys entered in Settings stay in that browser (local or session storage) and are sent with each
request. The server does not save them. A key in `.env` (`IPDR_API_KEY` or `ANTHROPIC_API_KEY`) is
used only for the server's default provider (`IPDR_PROVIDER`), never for another provider.

## How a question is answered

1. The model gets the instructions, the data dictionary and the team's terms (cacheable), plus the data's
   date range, the last few turns of the chat, and up to four reference queries from answers marked as correct.
   It first rewrites the question, however messy (typos, short forms, Urdu or Roman Urdu, missing details),
   as one clear English question. Every answer shows this as "Understood as".
2. It looks up exact stored names with `find_values` (WhatsApp is stored as `WhatsAPP`; TikTok has
   five labels) and tests SQL with `run_sql`. Tool calls in one turn run in parallel, and repeated
   queries come from a 10-minute cache.
3. It submits the answer with `submit_answer`: title, SQL, explanation, assumptions, chart.
4. The server checks the SQL (`sql_guard.py`) and runs it. A failure goes back to the model to fix.
   An empty result goes back once to be double-checked.
5. Checks in code, independent of the model:
   - **Dates:** common periods ("yesterday", "last week", "pichle hafte", "last 30 days"…) are worked out by
     the server and given to the model. If the question names one of them, or names no period at all (then
     the default of the last 7 full days applies), and the SQL uses other dates, it goes back to be fixed.
   - **Reference queries** are shown with their dates replaced by placeholders, so old dates cannot be copied.
   - **Follow-ups** ("aur lahore me?") build on the previous answer's SQL; reference queries are left out.
   - **Self-check** (all providers except Claude, `IPDR_REVIEW_PASS`): before an answer is shown the model
     compares its SQL with the question it understood, and confirms or fixes it.
6. Charts are chosen by rules from the result's columns (`charts.py`), from 19 kinds: bars, columns,
   Pareto, donut, treemap, stacked and 100% bars, line, area, stacked area, calendar (per day), 24-hour
   clock, heatmap (also weekday × hour), flow (sankey), sunburst, bars-and-line, radar, scatter, bubbles
   and map. The picker groups them by purpose, and a typed "show it as a pie chart" switches the last
   answer's chart. User counts are never stacked, summed or shown as shares (donut, treemap, flow),
   because one subscriber can use several apps.
7. Conclusions: every figure is checked against the table (values, totals, gaps, ratios, shares). A
   figure that cannot be found is sent back once; if it is still wrong, the conclusion shows a warning.

## Safety

- Every query runs with ClickHouse `readonly=2` plus limits on run time, rows read, memory and
  result size. The database user itself can write, so ask for a read-only user before sharing the app.
- The SQL guard allows one SELECT on the configured table only. It blocks other tables, table
  functions, `SETTINGS` and `FORMAT`, and adds a `LIMIT`. It requires a real lower bound on `start_time`
  joined with AND, so `start_time > 0` or `… OR 1=1` cannot scan the whole table.
- Search text reaches the database only as query parameters.
- Questions, queries and conclusions are logged to `logs/audit.jsonl` (keys never).
- What the AI provider receives: the question, the data dictionary, earlier turns of the chat,
  up to 15 rows per test query, and up to 300 rows for a conclusion.

## Counting users

About half of all records have no identified subscriber: `subscriber_id` then holds `<client_ip>_<port>`
(also for IPv6), which is different for nearly every record. Counting `uniq(subscriber_id)` therefore
counts most connections as separate "users". The app counts users as identified subscribers only:
`uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0)` (mobile numbers starting 92 and alphanumeric
account codes). "Unidentified connections" is available as its own measure in Explore.

## Known gaps

- No login or roles. Anyone who can open the page can query everything and see every chat.
- The base URL for OpenAI-compatible providers is called from the server, so only run the app
  where the people using it are trusted.
- The location chart has no base map.

## Files

```
ipdr/
  main.py          FastAPI app and JSON API
  agent.py         question -> tested SQL loop; conclusions
  explore.py       Explore page: filters -> parameterised SQL (no user text ever becomes SQL)
  llm/             provider adapters: anthropic_llm.py (SDK), openai_llm.py (OpenAI-compatible HTTP)
  context.py       instructions, tool definitions, per-question context
  dictionary.py    built-in + uploaded field descriptions, merged with the live table columns
  retrieval.py     finds the verified examples most similar to a question
  store.py         SQLite: chats, uploaded dictionaries, examples (data/ipdr.sqlite3)
  sql_guard.py     SQL checks before anything runs
  db.py            ClickHouse HTTP client (read-only mode, limits, result cache)
  charts.py        which charts fit a result
  semantic.yaml    built-in rules, metrics, column meanings and worked examples
  web/             the page: index.html, static/css, static/js (ES modules, no build step)
tests/
```

#   I P D R  
 