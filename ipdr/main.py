from __future__ import annotations

import csv
import dataclasses
import io
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import agent, catalog, charts, context, dictionary, explore, llm, retrieval, sql_guard, store, voice
from .config import ROOT, settings
from .db import QueryError, db

WEB = Path(__file__).with_name("web")
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("ipdr")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.init()
    retrieval.seed_builtin_examples()
    catalog.warm()
    yield


app = FastAPI(title="IPDR Analyst", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


_range_cache: dict = {"at": 0.0, "value": None}
_ACTIVE: set[str] = set()


def data_range() -> dict:
    if time.time() - _range_cache["at"] > 600 or _range_cache["value"] is None:
        _range_cache["value"] = db.data_range()
        _range_cache["at"] = time.time()
    return _range_cache["value"]


def _audit(**fields) -> None:
    fields["at"] = datetime.now().isoformat(timespec="seconds")
    with open(LOG_DIR / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")


class LLMBody(BaseModel):
    provider: str = ""
    model: str = ""
    base_url: str = ""
    effort: str = "medium"


def llm_config(body: LLMBody | None, key_header: str | None) -> llm.LLMConfig:
    b = body or LLMBody()
    provider = b.provider or settings.default_provider
    key = (key_header or "").strip()
    base_url = b.base_url.strip()
    if provider == settings.default_provider:
        key = key or settings.server_api_key
        base_url = base_url or settings.default_base_url
    model = b.model.strip() or (settings.default_model if provider == settings.default_provider
                                else llm.PRESETS.get(provider, {}).get("default_model", ""))
    effort = b.effort if b.effort in ("low", "medium", "high") else "medium"
    return llm.LLMConfig(provider=provider, model=model, api_key=key, base_url=base_url, effort=effort)


def _result_data(plan: dict, sql_run: str, result, steps: list[str], seconds: float, source: str) -> dict:
    res = result.to_dict()
    return {
        "understood": plan.get("understood", ""),
        "title": plan.get("title", ""),
        "explanation": plan.get("explanation", ""),
        "assumptions": plan.get("assumptions", []),
        "chart_hint": plan.get("chart"),
        "sql": plan.get("sql", "").strip(),
        "sql_run": sql_run,
        "result": res,
        "charts": charts.suggest(result.columns, result.rows, plan.get("chart")),
        "steps": steps,
        "seconds": seconds,
        "source": source,
    }


def _conversation_or_new(cid: str | None, title: str) -> dict:
    if cid:
        conv = store.get_conversation(cid)
        if conv is None:
            raise HTTPException(404, "This conversation no longer exists.")
        return conv
    return store.create_conversation()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/meta")
def meta():
    try:
        rng = data_range()
    except QueryError as e:
        rng = {"error": str(e)}
    return {
        "data_range": rng,
        "table": settings.full_table,
        "max_rows": settings.max_result_rows,
        "defaults": {
            "provider": settings.default_provider,
            "model": settings.default_model,
            "base_url": settings.default_base_url,
            "server_has_key": bool(settings.server_api_key),
        },
        "providers": {k: {kk: v[kk] for kk in ("label", "kind", "base_url", "needs_key", "default_model", "models", "key_hint")}
                      for k, v in llm.PRESETS.items()},
    }


class TitleBody(BaseModel):
    title: str = Field(max_length=200)


@app.get("/api/conversations")
def conversations(q: str = ""):
    return store.list_conversations(q.strip())


@app.post("/api/conversations")
def create_conversation():
    return store.create_conversation()


@app.get("/api/conversations/{cid}")
def conversation(cid: str):
    conv = store.get_conversation(cid)
    if conv is None:
        raise HTTPException(404, "This conversation no longer exists.")
    msgs = conv.get("messages", [])
    for i, m in enumerate(msgs):
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if m["role"] != "user" or (nxt is not None and nxt["role"] != "user"):
            continue
        if m["id"] in _ACTIVE:
            m["unanswered"] = "working"
            continue
        later = [x for x in msgs[i + 1:] if x["role"] == "user"]
        asked_again = any(x["content"].strip().lower() == m["content"].strip().lower() or context.is_retry(x["content"])
                          for x in later)
        if not asked_again:
            m["unanswered"] = "interrupted"
    return conv


@app.patch("/api/conversations/{cid}")
def rename(cid: str, body: TitleBody):
    if not store.rename_conversation(cid, body.title.strip()):
        raise HTTPException(404, "This conversation no longer exists.")
    return {"ok": True}


class RewindBody(BaseModel):
    message_id: str


@app.post("/api/conversations/{cid}/rewind")
def rewind(cid: str, body: RewindBody):
    msg = store.get_message(body.message_id)
    if msg is None or msg["conversation_id"] != cid:
        raise HTTPException(404, "That message no longer exists.")
    if msg["role"] != "user":
        raise HTTPException(400, "Only your own questions can be edited.")
    removed = store.delete_from(cid, body.message_id)
    _audit(event="edit_question", conversation=cid, removed=removed)
    return {"removed": removed}


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    store.delete_conversation(cid)
    return {"ok": True}


class ChatBody(BaseModel):
    conversation_id: str | None = None
    question: str = Field(min_length=2, max_length=4000)
    llm: LLMBody | None = None

    @field_validator("question")
    @classmethod
    def _real_question(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("Type a question first.")
        return v


def _without_small_talk(msgs: list[dict]) -> list[dict]:
    out = []
    for i, m in enumerate(msgs):
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if m["kind"] == "reply" or (m["role"] == "user" and nxt is not None and nxt["kind"] == "reply"):
            continue
        out.append(m)
    return out


@app.post("/api/chat")
def chat(body: ChatBody, x_llm_key: str | None = Header(default=None)):
    cfg = llm_config(body.llm, x_llm_key)
    try:
        llm.check_config(cfg)
    except llm.LLMError as e:
        raise HTTPException(400, str(e)) from e

    conv = _conversation_or_new(body.conversation_id, body.question)
    cid = conv["id"]
    turns = _without_small_talk(store.recent_turns(cid, limit=6)) if body.conversation_id else []
    user_msg = store.add_message(cid, "user", "question", body.question.strip())
    store.touch_conversation(cid, title_if_empty=body.question.strip())
    _ACTIVE.add(user_msg["id"])
    talk = context.small_talk(body.question, turns)
    command = None if talk else context.app_command(body.question, turns)
    earlier = None if talk or command else context.retry_target(body.question, turns)
    question = earlier or body.question

    events: queue.Queue = queue.Queue()
    cancel = threading.Event()
    source = f"{llm.PRESETS.get(cfg.provider, {}).get('label', cfg.provider)} · {cfg.model}"

    def work():
        started = time.time()
        if talk or command:
            try:
                if command:
                    msg = store.add_message(cid, "assistant", "reply", command["reply"],
                                            {"command": True, "actions": command["actions"], "target": command["target"]})
                else:
                    msg = store.add_message(cid, "assistant", "reply", context.SMALL_TALK_REPLIES[talk], {"small_talk": talk})
                store.touch_conversation(cid)
                events.put({"type": "message", "message": msg})
            finally:
                _ACTIVE.discard(user_msg["id"])
                events.put(None)
            return
        try:
            try:
                rng = data_range()
            except QueryError as e:
                raise agent.AgentError(f"The database is not reachable: {e}") from e
            if earlier:
                events.put({"type": "step", "text": f"Answering your earlier question: “{earlier}”"})
            ans = agent.answer(question, turns, cfg, rng,
                               lambda text: events.put({"type": "step", "text": text}), cancel)
            if earlier:
                ans.steps.insert(0, f"Answering your earlier question: “{earlier}”")
            seconds = agent.elapsed(started)
            plan = ans.plan
            if ans.result is None:
                kind = plan["status"]
                msg = store.add_message(cid, "assistant", kind, plan.get("message", ""),
                                        {"title": plan.get("title", ""), "understood": plan.get("understood", ""),
                                         "steps": ans.steps, "seconds": seconds, "source": source})
                _audit(event="chat", conversation=cid, question=body.question, status=kind, provider=cfg.provider,
                       model=cfg.model, usage=ans.usage)
            else:
                data = _result_data(plan, ans.sql, ans.result, ans.steps, seconds, source)
                data["usage"] = ans.usage
                msg = store.add_message(cid, "assistant", "result", plan.get("title", ""), data)
                store.touch_conversation(cid, title_if_empty=plan.get("title", ""))
                _audit(event="chat", conversation=cid, question=body.question, status="ok", sql=data["sql"],
                       rows=len(ans.result.rows), rows_read=ans.result.rows_read, seconds=seconds,
                       provider=cfg.provider, model=cfg.model, usage=ans.usage)
            store.touch_conversation(cid)
            events.put({"type": "message", "message": msg})
        except agent.Cancelled:
            msg = store.add_message(cid, "assistant", "error", "Stopped.", {"stopped": True})
            events.put({"type": "message", "message": msg})
        except agent.AgentError as e:
            msg = store.add_message(cid, "assistant", "error", str(e), {"source": source})
            _audit(event="chat", conversation=cid, question=body.question, status="error", message=str(e))
            events.put({"type": "message", "message": msg})
        except Exception:
            log.exception("chat failed")
            msg = store.add_message(cid, "assistant", "error",
                                    "Something went wrong on the server. The details are in the server log.")
            events.put({"type": "message", "message": msg})
        finally:
            _ACTIVE.discard(user_msg["id"])
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        try:
            yield json.dumps({"type": "conversation", "id": cid, "title": conv.get("title") or body.question[:80]}) + "\n"
            yield json.dumps({"type": "user_message", "message": user_msg}, ensure_ascii=False) + "\n"
            while (item := events.get()) is not None:
                yield json.dumps(item, ensure_ascii=False, default=str) + "\n"
        finally:
            cancel.set()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class SqlBody(BaseModel):
    conversation_id: str | None = None
    sql: str = Field(min_length=1, max_length=50_000)
    title: str = ""


@app.post("/api/sql")
def run_sql(body: SqlBody):
    conv = _conversation_or_new(body.conversation_id, "Custom query")
    cid = conv["id"]
    user_msg = store.add_message(cid, "user", "sql", body.sql.strip())
    store.touch_conversation(cid, title_if_empty=body.title or "Custom query")
    started = time.time()
    try:
        checked = sql_guard.check(body.sql)
        result = db.cached_query(checked, ttl=60)
    except (sql_guard.GuardError, QueryError) as e:
        msg = store.add_message(cid, "assistant", "error", str(e))
        _audit(event="sql", conversation=cid, sql=body.sql, status="error", message=str(e))
        return {"conversation_id": cid, "messages": [user_msg, msg]}
    plan = {"title": body.title or "Custom query", "sql": body.sql}
    data = _result_data(plan, checked, result, ["Running the query"], agent.elapsed(started), "Your SQL")
    msg = store.add_message(cid, "assistant", "result", plan["title"], data)
    _audit(event="sql", conversation=cid, sql=body.sql, status="ok", rows=len(result.rows), rows_read=result.rows_read)
    return {"conversation_id": cid, "messages": [user_msg, msg]}


def _result_message(mid: str) -> dict:
    msg = store.get_message(mid)
    if msg is None or msg["kind"] != "result":
        raise HTTPException(404, "That result no longer exists.")
    return msg


def _question_for(msg: dict) -> str:
    conv = store.get_conversation(msg["conversation_id"]) or {"messages": []}
    prev = None
    for m in conv["messages"]:
        if m["id"] == msg["id"]:
            break
        if m["role"] == "user":
            prev = m
    return prev["content"] if prev and prev["kind"] == "question" else ""


class ConcludeBody(BaseModel):
    llm: LLMBody | None = None


@app.post("/api/messages/{mid}/conclude")
def conclude(mid: str, body: ConcludeBody, x_llm_key: str | None = Header(default=None)):
    msg = _result_message(mid)
    cfg = llm_config(body.llm, x_llm_key)
    try:
        llm.check_config(cfg)
        out = agent.conclude(msg["data"], _question_for(msg), cfg)
    except (llm.LLMError, agent.AgentError) as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    store.update_message_data(mid, conclusion=out)
    _audit(event="conclude", message=mid, headline=out.get("headline"), provider=cfg.provider, model=cfg.model)
    return out


class VerifyBody(BaseModel):
    question: str = ""
    notes: str = ""


@app.post("/api/messages/{mid}/verify")
def verify(mid: str, body: VerifyBody):
    msg = _result_message(mid)
    question = body.question.strip() or _question_for(msg) or msg["data"].get("title", "")
    ex = store.add_example(question, msg["data"]["sql"], body.notes, source="user")
    store.update_message_data(mid, verified=ex["id"])
    return ex


class MessageView(BaseModel):
    chart: int | None = None
    chart_hidden: bool | None = None
    conclusion_hidden: bool | None = None


@app.patch("/api/messages/{mid}")
def set_message_view(mid: str, body: MessageView):
    _result_message(mid)
    fields = {}
    if body.chart is not None:
        fields["chart_choice"] = body.chart
    if body.chart_hidden is not None:
        fields["chart_hidden"] = body.chart_hidden
    if body.conclusion_hidden is not None:
        fields["conclusion_hidden"] = body.conclusion_hidden
    if fields:
        store.update_message_data(mid, **fields)
    return {"ok": True}


@app.get("/api/messages/{mid}/export.csv")
def export_csv(mid: str):
    msg = _result_message(mid)
    res = msg["data"]["result"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([c["name"] for c in res["columns"]])
    w.writerows(res["rows"])
    title = msg["data"].get("title") or "result"
    from urllib.parse import quote
    ascii_name = "".join(ch if (ch.isascii() and ch.isalnum()) else "_" for ch in title)[:60].strip("_") or "result"
    full = "".join(ch if ch.isalnum() else "_" for ch in title)[:60].strip("_") or "result"
    return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{ascii_name}.csv"; '
                                                             f"filename*=UTF-8''{quote(full)}.csv"})


@app.get("/api/dictionary")
def get_dictionary():
    eff = dictionary.effective()
    prompt = dictionary.render_for_prompt()
    cols = eff["columns"]
    return {
        "columns": cols,
        "notes": eff["notes"],
        "files": store.list_dictionaries(),
        "summary": {
            "columns": len(cols),
            "described": sum(1 for c in cols if c["description"] and not c["empty"]),
            "empty": sum(1 for c in cols if c["empty"]),
            "missing": [c["name"] for c in cols if not c["description"] and not c["empty"]],
            "prompt_chars": len(prompt),
        },
    }


@app.get("/api/dictionary/prompt", response_class=PlainTextResponse)
def dictionary_prompt():
    return dictionary.render_for_prompt()


@app.get("/api/dictionary/template.csv")
def dictionary_template():
    return StreamingResponse(iter(["﻿" + dictionary.template_csv()]), media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="data_dictionary_template.csv"'})


@app.post("/api/dictionary")
async def upload_dictionary(file: UploadFile = File(...), name: str = Form("")):
    raw = await file.read()
    if len(raw) > 5_000_000:
        raise HTTPException(413, "The file is larger than 5 MB.")
    try:
        d = dictionary.save_upload(file.filename or "upload.txt", raw, name.strip())
    except dictionary.DictionaryError as e:
        raise HTTPException(400, str(e)) from e
    _audit(event="dictionary_upload", file=file.filename, fields=len(d["fields"]))
    return d


class ActiveBody(BaseModel):
    active: bool


@app.patch("/api/dictionary/{did}")
def toggle_dictionary(did: str, body: ActiveBody):
    if not store.set_dictionary_active(did, body.active):
        raise HTTPException(404, "That file no longer exists.")
    return {"ok": True}


@app.delete("/api/dictionary/{did}")
def remove_dictionary(did: str):
    store.delete_dictionary(did)
    return {"ok": True}


class TermBody(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    meaning: str = Field(min_length=1, max_length=1000)

    @field_validator("term", "meaning")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Fill in both the term and what it means.")
        return v


@app.get("/api/terms")
def terms():
    return store.list_terms()


@app.post("/api/terms")
def add_term(body: TermBody):
    return store.add_term(body.term, body.meaning)


@app.patch("/api/terms/{tid}")
def update_term(tid: str, body: TermBody):
    if not store.update_term(tid, body.term, body.meaning):
        raise HTTPException(404, "That term no longer exists.")
    return {"ok": True}


@app.delete("/api/terms/{tid}")
def delete_term(tid: str):
    store.delete_term(tid)
    return {"ok": True}


class ExampleBody(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    sql: str = Field(min_length=10, max_length=50_000)
    notes: str = ""


@app.get("/api/examples")
def examples():
    return store.list_examples()


@app.post("/api/examples")
def add_example(body: ExampleBody):
    try:
        checked = sql_guard.check(body.sql, max_rows=5)
    except sql_guard.GuardError as e:
        raise HTTPException(400, f"The SQL is not allowed: {e}") from e
    try:
        db.query(checked, max_rows=5)
    except QueryError as e:
        raise HTTPException(400, f"The SQL did not run: {e}") from e
    return store.add_example(body.question, body.sql, body.notes, source="user")


@app.delete("/api/examples/{eid}")
def delete_example(eid: str):
    store.delete_example(eid)
    return {"ok": True}


class LLMOnly(BaseModel):
    llm: LLMBody


@app.post("/api/llm/test")
def llm_test(body: LLMOnly, x_llm_key: str | None = Header(default=None)):
    cfg = llm_config(body.llm, x_llm_key)
    try:
        return {"ok": True, "message": llm.test(cfg)}
    except llm.LLMError as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=400)


@app.post("/api/llm/models")
def llm_models(body: LLMOnly, x_llm_key: str | None = Header(default=None)):
    cfg = llm_config(body.llm, x_llm_key)
    try:
        return {"models": llm.list_models(cfg)}
    except llm.LLMError as e:
        return JSONResponse({"models": [], "message": str(e)}, status_code=400)


MAX_EXPORT_ROWS = 200_000


@app.get("/api/explore/fields")
def explore_fields():
    try:
        rng = data_range()
    except QueryError as e:
        rng = {"error": str(e)}
    return {
        "fields": explore.fields(),
        "ops": {k: [{"op": o, "label": label} for o, label in v] for k, v in explore.OPS.items()},
        "measures": [{"key": k, "label": v[0]} for k, v in explore.MEASURES.items()],
        "time_groups": [{"key": k, "label": v[0]} for k, v in explore.TIME_GROUPS.items()],
        "default_columns": explore.DEFAULT_COLUMNS,
        "max_page": explore.MAX_PAGE,
        "max_days": explore.MAX_DAYS,
        "max_export": MAX_EXPORT_ROWS,
        "data_range": rng,
    }


def _explore_error(e: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.post("/api/explore/run")
def explore_run(q: explore.ExploreQuery):
    try:
        built = explore.build(q)
        res = db.cached_query(built.sql, max_rows=explore.MAX_PAGE, params=built.params, ttl=120)
    except (explore.ExploreError, QueryError) as e:
        return _explore_error(e)
    _audit(event="explore", mode=q.mode, filters=[f.model_dump() for f in q.filters], rows=len(res.rows),
           rows_read=res.rows_read)
    keys = None
    if built.key_cols:
        k = built.key_cols
        keys = [r[-k:] for r in res.rows]
        res = dataclasses.replace(res, columns=res.columns[:-k], rows=[r[:-k] for r in res.rows])
    out = {"sql": built.display_sql, "result": res.to_dict(), "keys": keys}
    if q.mode == "summary":
        hint = None
        if q.group_by:
            hint = {"type": "line" if q.group_by[0] in explore.TIME_GROUPS else "bar", "x": q.group_by[0],
                    "y": (q.measures or ["connections"])[0], "series": q.group_by[1] if len(q.group_by) > 1 else ""}
        out["charts"] = charts.suggest(res.columns, res.rows, hint)
    return out


@app.post("/api/explore/count")
def explore_count(q: explore.ExploreQuery):
    try:
        built = explore.build(q, count=True)
        res = db.cached_query(built.sql, max_rows=1, params=built.params, ttl=300)
    except (explore.ExploreError, QueryError) as e:
        return _explore_error(e)
    return {"count": res.rows[0][0] if res.rows else 0, "elapsed_s": round(res.elapsed_s, 2), "rows_read": res.rows_read}


class RecordBody(BaseModel):
    t: str = Field(max_length=32)
    id: str = Field(min_length=1, max_length=200)


@app.post("/api/explore/record")
def explore_record(body: RecordBody):
    try:
        sql, params = explore.record_sql(body.t, body.id)
        res = db.cached_query(sql, max_rows=1, params=params, ttl=600)
    except (explore.ExploreError, QueryError) as e:
        return _explore_error(e)
    if not res.rows:
        raise HTTPException(404, "That record was not found. Run the search again.")
    desc = {f["name"]: f.get("description", "") for f in explore.fields()}
    row = res.rows[0]
    return {"fields": [{"name": c["name"], "type": c["type"], "value": row[i], "description": desc.get(c["name"], "")}
                       for i, c in enumerate(res.columns)]}


@app.get("/api/explore/interpret")
def explore_interpret(q: str = ""):
    return explore.interpret(q[:300])


@app.get("/api/explore/values")
def explore_values(field: str, q: str = ""):
    try:
        day = data_range().get("last_full_day")
        return {"values": explore.values(field, q.strip(), day, db)}
    except QueryError:
        return {"values": []}


@app.get("/api/explore/export")
def explore_export(q: str):
    try:
        query = explore.ExploreQuery.model_validate_json(q)
        built = explore.build(query, export_limit=MAX_EXPORT_ROWS)
        stream = db.stream_csv(built.sql, params=built.params, max_rows=MAX_EXPORT_ROWS)
    except explore.ExploreError as e:
        raise HTTPException(400, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, "This export link is not valid. Run the query again and export from the page.") from e
    except QueryError as e:
        raise HTTPException(400, str(e)) from e
    _audit(event="explore_export", mode=query.mode, filters=[f.model_dump() for f in query.filters])
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return StreamingResponse(stream, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="ipdr_{query.mode}_{stamp}.csv"'})


class ViewBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    config: dict

    @field_validator("name")
    @classmethod
    def _real_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Give the view a name.")
        return v.strip()


@app.get("/api/views")
def views():
    return store.list_views()


@app.post("/api/views")
def save_view(body: ViewBody):
    return store.save_view(body.name, body.config)


@app.delete("/api/views/{vid}")
def delete_view(vid: str):
    store.delete_view(vid)
    return {"ok": True}


@app.get("/api/voice/options")
def voice_options():
    return {"providers": {k: {"label": v["label"], "base_url": v["base_url"], "model": v["model"]}
                          for k, v in voice.PROVIDERS.items()}}


@app.post("/api/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...), provider: str = Form("openai"), model: str = Form(""),
                           language: str = Form(""), base_url: str = Form(""),
                           x_llm_key: str | None = Header(default=None)):
    raw = await audio.read()
    try:
        text = voice.transcribe(raw, audio.filename or "speech.webm", audio.content_type or "",
                                provider, (x_llm_key or "").strip(), model.strip(), language.strip(), base_url.strip())
    except voice.VoiceError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    return {"text": text}


@app.post("/api/voice/clean")
def voice_clean(body: dict):
    return {"text": voice.clean(str(body.get("text", "")))}


@app.get("/api/busy")
def busy():
    return {"answering": len(_ACTIVE)}


@app.get("/api/health")
def health():
    ok_db = True
    try:
        db.query("SELECT 1", max_rows=1)
    except QueryError:
        ok_db = False
    return {"database": ok_db, "today": date.today().isoformat()}


def main() -> None:
    import uvicorn

    uvicorn.run("ipdr.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
