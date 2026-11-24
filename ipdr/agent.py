from __future__ import annotations

import functools
import json
import logging
import os
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from . import catalog, concepts, context, dictionary, llm, retrieval, sql_guard, store, urdu
from .config import ROOT, settings
from .db import QueryError, QueryResult, db
from .llm import LLMConfig, LLMError, ToolCall


class AgentError(Exception):
    pass


class Cancelled(Exception):
    pass


@dataclass
class Answer:
    plan: dict
    sql: str | None = None
    result: QueryResult | None = None
    steps: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    tested: bool = False


StepFn = Callable[[str], None]
log = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=6)


def _tool_find_values(column: str, search: str, data_range: dict) -> str:
    allowed = dictionary.string_columns()
    if column not in allowed:
        return f"Column {column} cannot be searched. Searchable: {', '.join(allowed)}."
    day = data_range.get("last_full_day") or date.today().isoformat()
    ids = [e for e in _safe(catalog.entities, search) or [] if e.kind in ("imsi", "imei", "ip")
           or (e.kind == "subscriber" and column != "subscriber_id")]
    if ids:
        e = ids[0]
        return (f"'{e.name}' is {'an IMSI (SIM)' if e.kind == 'imsi' else 'an IMEI (handset)' if e.kind == 'imei' else 'an IP address' if e.kind == 'ip' else 'a subscriber ID'}"
                f". Do not search for it: filter on it directly over the question's period: {e.sql}"
                + (f" ({e.note})" if e.note else "") + ". For 'which number / who has it': SELECT subscriber_id, "
                "count() AS connections ... WHERE <period> AND that filter GROUP BY subscriber_id ORDER BY connections DESC.")
    phone = context.phones(search) if column == "subscriber_id" else []
    if phone:
        search = phone[0][1][2:]
    pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    sql = f"""
        SELECT {column} AS value, count() AS rows
        FROM {settings.full_table}
        WHERE start_time >= toUnixTimestamp('{day} 00:00:00', '{settings.timezone}')
          AND start_time <  toUnixTimestamp('{day} 00:00:00', '{settings.timezone}') + 86400
          AND {column} != '' AND {column} ILIKE {{pattern:String}}
        GROUP BY value ORDER BY rows DESC LIMIT 25"""
    res = db.cached_query(sql, max_rows=25, params={"pattern": pattern}, ttl=1800, timeout=_QUERY_TIMEOUT)
    if not res.rows and phone:
        return (f"{phone[0][0]} has no records on {day}, the only day searched here. It may have records on other "
                f"days: query the question's period with {context.phone_sql(phone)} and answer from that.")
    if not res.rows and (alike := catalog.values_like(column, search)):
        return (f"Values of {column} matching '{search}' (accents and punctuation ignored): "
                + ", ".join(repr(v) for v in alike) + ". Use the exact stored value.")
    if not res.rows:
        hint = concepts.hint_for(search)
        if hint:
            return f"No values of {column} contain '{search}' on {day}. {hint} Use that list."
        elsewhere = _find_elsewhere(column, search, day, allowed)
        if elsewhere:
            return f"No values of {column} contain '{search}' on {day}. {elsewhere}"
        return (f"No values of {column} contain '{search}' on {day}. Try a shorter or different spelling, "
                "another column, or a concept from app_concepts if the word describes a kind of app.")
    lines = [f"Values of {column} containing '{search}' on {day} (value: rows):"]
    lines += [f"{v!r}: {n}" for v, n in res.rows]
    if len(res.rows) == 25:
        lines.append("(Only the 25 most common are shown. Search with more specific text to narrow it.)")
    return "\n".join(lines)


_OTHER_COLUMNS = ["client_location", "client_asn", "app_label", "app_category", "loc", "server_asn", "requested_server",
                  "server_location"]


def _find_elsewhere(column: str, search: str, day: str, allowed: list[str]) -> str:
    city = concepts.city_in(search)
    if city:
        return (f"'{search}' is a city: filter with client_location = '{city} Pakistan' (city level), or "
                f"loc ILIKE '%{city}%' for cell-site addresses.")
    pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    for other in _OTHER_COLUMNS:
        if other == column or other not in allowed:
            continue
        sql = f"""
            SELECT {other} AS value, count() AS rows
            FROM {settings.full_table}
            WHERE start_time >= toUnixTimestamp('{day} 00:00:00', '{settings.timezone}')
              AND start_time <  toUnixTimestamp('{day} 00:00:00', '{settings.timezone}') + 86400
              AND {other} != '' AND {other} ILIKE {{pattern:String}}
            GROUP BY value ORDER BY rows DESC LIMIT 5"""
        try:
            res = db.cached_query(sql, max_rows=5, params={"pattern": pattern}, ttl=1800, timeout=30)
        except QueryError:
            continue
        if res.rows:
            values = ", ".join(f"{v!r} ({n} rows)" for v, n in res.rows)
            return f"Found in {other} instead: {values}. Use that column."
    return ""


def _tool_run_sql(sql: str) -> str:
    fixed = repair_measures(sql)
    checked = sql_guard.check(fixed or sql, max_rows=1000)
    res = db.cached_query(checked, max_rows=1000, timeout=_QUERY_TIMEOUT)
    preview = {
        "columns": [f"{c['name']} {c['type']}" for c in res.columns],
        "row_count": len(res.rows),
        "cut_off_at_limit": res.truncated,
        "first_rows": res.rows[:15],
        "seconds": round(res.elapsed_s, 2),
        "rows_read": res.rows_read,
    }
    if not res.rows:
        preview["note"] = "No rows. Check the names (find_values), the period and the filters."
    if fixed:
        preview["fixed_sql"] = fixed
        preview["note_on_measures"] = ("users, connections and volume_gb are not columns. The system wrote them out for "
                                       "you: use fixed_sql from now on.")
    return json.dumps(preview, default=str)


def _run_tool(call: ToolCall, data_range: dict) -> tuple[ToolCall, str, bool]:
    if call.bad_args:
        return call, f"Error: the arguments were not valid JSON: {call.bad_args}", True
    try:
        if call.name == "find_values":
            return call, _tool_find_values(str(call.args.get("column", "")), str(call.args.get("search", "")), data_range), False
        if call.name == "run_sql":
            return call, _tool_run_sql(str(call.args.get("sql", ""))), False
        return call, f"Error: unknown tool {call.name}.", True
    except (sql_guard.GuardError, QueryError) as e:
        return call, f"Error: {e}", True


def _describe(call: ToolCall) -> str:
    if call.name == "find_values":
        search = str(call.args.get("search", "")).strip()
        column = call.args.get("column", "")
        return f"Looking up “{search}” in {column}" if search else f"Listing the most common values of {column}"
    if call.name == "run_sql":
        return "Testing the query"
    return "Working"


_QUERY_TIMEOUT = 90

def _too_heavy(err) -> bool:
    text = str(err).lower()
    return "too heavy" in text or "too long" in text or "timeout" in text


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
ANSWER_CACHE = os.environ.get("IPDR_ANSWER_CACHE", "1") != "0"
REVIEW_TURNS = 3


_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(SELECT\b.*?)```", re.I | re.S)
_SQL_FIELD = re.compile(r"^\s*\**sql\**\s*:\s*(SELECT\b.*?)(?=^\s*\**(?:explanation|assumptions|chart|message|status|"
                        r"title|understood)\**\s*:|\Z)", re.I | re.M | re.S)
_TEXT_FIELD = re.compile(r"^\s*[-*]?\s*\**(understood|title|explanation|message|status)\**\s*:\s*(.+?)\s*$", re.I | re.M)


def _plan_from_text(text: str) -> dict | None:
    text = text or ""
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "status" in data:
                return data
        except json.JSONDecodeError:
            pass
    sql = _SQL_BLOCK.search(text) or _SQL_FIELD.search(text)
    if not sql:
        return None
    fields = {k.lower(): v.strip().strip("\"'") for k, v in _TEXT_FIELD.findall(text)}
    return {"status": fields.get("status", "ok").lower() if fields.get("status", "ok").lower() in ("ok", "clarify", "cannot_answer") else "ok",
            "understood": fields.get("understood", ""), "title": fields.get("title", ""),
            "explanation": fields.get("explanation", ""), "message": fields.get("message", ""),
            "sql": sql.group(1).strip(), "assumptions": [], "chart": {}}


_MEASURE_SQL = {
    "users": "uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0)",
    "connections": "count()",
    "volume_gb": "round(sum(c2s_byte_num + s2c_byte_num) / 1e9, 2)",
    "unidentified_connections": "countIf(position(subscriber_id, '_') > 0)",
}


def _split_top(text: str) -> list[str]:
    out, depth, quote, start = [], 0, "", 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"`":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def _fix_multiif(sql: str) -> str | None:
    out, i, changed = [], 0, False
    low = sql.lower()
    while True:
        j = low.find("multiif(", i)
        if j < 0:
            out.append(sql[i:])
            break
        start = j + len("multiif(")
        depth, quote, k = 1, "", start
        while k < len(sql) and depth:
            ch = sql[k]
            if quote:
                if ch == quote:
                    quote = ""
            elif ch in "'\"`":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            k += 1
        if depth:
            return None
        inner = sql[start:k - 1]
        args = _split_top(inner)
        if len(args) >= 2 and len(args) % 2 == 0:
            col = re.search(r"\b(app_label|app_category|client_location|server_location|requested_server)\b", inner)
            inner = inner.rstrip() + f", {col.group(1) if col else repr('Other')}"
            changed = True
        out.append(sql[i:start] + inner + ")")
        i = k
    return "".join(out) if changed else None


_LOOSE_OR = re.compile(r"\bAND\s+(app_label\s+IN\s*\([^()]*\))\s+OR\s+(match\(\s*requested_server\s*,\s*'(?:[^'\\]|\\.)*'\s*\))",
                       re.I)


_TIME_HEAD = re.compile(r"^(\s*start_time\s*>=?\s*toUnixTimestamp\([^()]*\)\s+AND\s+start_time\s*<=?\s*"
                        r"toUnixTimestamp\([^()]*\)\s+AND\s+)(.+)$", re.I | re.S)
_WHERE_END = re.compile(r"\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|SETTINGS)\b", re.I)


def _has_top_or(text: str) -> bool:
    depth, quote, i, low = 0, "", 0, text.lower()
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "'\"`":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and low.startswith(" or ", i):
            return True
        i += 1
    return False


def _fix_loose_or(sql: str) -> str | None:
    fixed = _LOOSE_OR.sub(lambda m: f"AND ({m.group(1)} OR {m.group(2)})", sql)
    w = re.search(r"\bWHERE\b", fixed, re.I)
    if w:
        tail = fixed[w.end():]
        end = _WHERE_END.search(tail)
        body, rest = (tail[:end.start()], tail[end.start():]) if end else (tail, "")
        head = _TIME_HEAD.match(body)
        if head and _has_top_or(" " + head.group(2)):
            filters = head.group(2).rstrip()
            fixed = fixed[:w.end()] + head.group(1) + "(" + filters + ")\n" + rest
    return fixed if fixed != sql else None


def _fix_one_arg_if(sql: str) -> str | None:
    out, i, changed = [], 0, False
    for m in re.finditer(r"\bsumIf\s*\(", sql):
        depth, j = 1, m.end()
        top_commas = 0
        while j < len(sql) and depth:
            ch = sql[j]
            if ch == "'":
                j = sql.find("'", j + 1)
                if j < 0:
                    return None
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 1:
                top_commas += 1
            j += 1
        if top_commas == 0 and m.start() >= i:
            out.append(sql[i:m.start()] + "countIf(")
            i = m.end()
            changed = True
    if not changed:
        return None
    out.append(sql[i:])
    return "".join(out)


_PERIOD_COND = re.compile(r"start_time\s*>=?\s*toUnixTimestamp\([^()]*\)\s+AND\s+start_time\s*<=?\s*toUnixTimestamp\([^()]*\)",
                          re.I)


def _fix_subquery_period(sql: str) -> str | None:
    m = _PERIOD_COND.search(sql)
    if not m:
        return None
    period = m.group(0)
    table = re.escape(settings.full_table)
    out, last, changed = [], 0, False
    for t in re.finditer(rf"\bFROM\s+{table}\b", sql, re.I):
        depth, j = 0, t.end()
        while j < len(sql):
            ch = sql[j]
            if ch == "'":
                k = sql.find("'", j + 1)
                j = k if k >= 0 else len(sql)
            elif ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        body = sql[t.end():j]
        if re.search(r"(?i)\bstart_time\b", body) or j >= len(sql):
            continue
        w = re.match(r"(?is)(\s+WHERE\s+)", body)
        if w:
            tail = re.search(r"(?i)\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|SETTINGS)\b", body[w.end():])
            cut = w.end() + (tail.start() if tail else len(body) - w.end())
            new_body = body[:w.end()] + f"{period} AND ({body[w.end():cut].rstrip()})" + (" " if tail else "") + body[cut:]
        else:
            tail = re.search(r"(?i)\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b", body)
            cut = tail.start() if tail else len(body)
            new_body = body[:cut].rstrip() + f" WHERE {period} " + body[cut:]
        out.append(sql[last:t.end()] + new_body)
        last = j
        changed = True
    if not changed:
        return None
    out.append(sql[last:])
    return "".join(out)


def _fix_lookup_exclusions(sql: str) -> str | None:
    if not re.search(r"subscriber_id\s*(=\s*'|IN\s*\(\s*')", sql, re.I) or re.search(r"(?i)\bGROUP\s+BY\s+[^;]*client_location", sql):
        return None
    new = re.sub(r"(?i)\s+AND\s+client_location\s+NOT\s+IN\s*\([^()]*\)", "", sql)
    new = re.sub(r"(?i)\s+AND\s+client_location\s*(!=|<>)\s*''", "", new)
    return new if new != sql else None


def repair_measures(sql: str) -> str | None:
    orig = sql or ""
    sql = _fix_one_arg_if(orig) or orig
    sql = _fix_lookup_exclusions(sql) or sql
    sql = _fix_subquery_period(sql) or sql
    sql = re.sub(r"(?i)\bsum\s*\(\s*block_type\s*\)", "count()", sql)
    if sql != orig:
        return repair_measures(sql) or sql
    loose = _fix_loose_or(sql or "")
    sql = loose or sql
    fixed_if = _fix_multiif(sql or "") or loose
    sql = fixed_if or sql
    m = re.search(r"^\s*SELECT\s+(.*?)\s+FROM\s+(\(?)", sql or "", re.I | re.S)
    if not m or m.group(2) == "(":
        return fixed_if
    defined = {a.lower() for a in re.findall(r"(?i)\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql)}
    items, changed = [], False
    for item in _split_top(m.group(1)):
        name = item.strip()
        if name.lower() in _MEASURE_SQL and name.lower() not in defined:
            items.append(f"{_MEASURE_SQL[name.lower()]} AS {name}")
            changed = True
        else:
            items.append(name)
    if changed:
        sql = sql[:m.start(1)] + ",\n       ".join(items) + sql[m.end(1):]
    tail = _fix_order_measures(sql)
    if tail:
        return tail
    return sql if changed else fixed_if


def _fix_order_measures(sql: str) -> str | None:
    defined = {a.lower() for a in re.findall(r"(?i)\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql)}
    m = re.search(r"(?is)\b(ORDER\s+BY|HAVING)\b(.*?)(\bLIMIT\b|\bSETTINGS\b|$)", sql)
    if not m:
        return None
    clause = m.group(2)
    new = clause
    for name, expr in _MEASURE_SQL.items():
        if name not in defined:
            new = re.sub(rf"(?<![A-Za-z0-9_.]){name}(?![A-Za-z0-9_(])", expr, new)
    if new == clause:
        return None
    return sql[:m.start(2)] + new + sql[m.end(2):]


def fix_rounded_order(sql: str) -> str | None:
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql or "", read="clickhouse")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    rounded = {a.alias: a.this.this.sql("clickhouse") for a in tree.expressions
               if isinstance(a, exp.Alias) and isinstance(a.this, exp.Round) and a.this.this is not None
               and not a.this.this.find(exp.Window)}
    if not rounded:
        return None
    last = None
    for m in re.finditer(r"(?i)\bORDER\s+BY\b", sql):
        before = sql[:m.start()]
        if before.count("(") == before.count(")"):
            last = m
    if last is None:
        return None
    end = re.search(r"(?i)\bLIMIT\b|;|$", sql[last.end():])
    clause = sql[last.end(): last.end() + end.start()]
    new = clause
    for alias, inner in rounded.items():
        new = re.sub(rf"(?<![\w.]){re.escape(alias)}(?![\w(])", f"({inner})", new)
    if new == clause:
        return None
    return sql[:last.end()] + new + sql[last.end() + end.start():]


_ASK_VOLUME = re.compile(r"(?<![a-z])(volume|volumne|usage|data|traffic|consumption|consume[sd]?|bandwidth|gb|mb|kb|"
                         r"bytes|downloads?|downloaded|uploads?|uploaded|kitna data)(?![a-z])")
_ASK_CONNECTIONS = re.compile(r"(?<![a-z])(connections?|times|hits|visits|visited|accessed|opened|sessions|requests|"
                              r"how often|kitni baar|baar)(?![a-z])")


def _safe(check, *args):
    try:
        return check(*args)
    except Exception as e:
        log.exception("check %s failed", getattr(check, "__name__", check))
        _debug("(check)", check_error=getattr(check, "__name__", str(check)), message=str(e)[:300])
        return None
_ASK_DURATION = re.compile(r"(?<![a-z])(time spent|duration|how long|longest|kitni der|"
                           r"spen[dt]s? (the )?(most |more |least )?time|hours spent|minutes spent)(?![a-z])")
_NUMBER_WORDS = {"one": 1, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "fifty": 50, "hundred": 100}


def _measure_kind(text: str) -> str:
    t = text.lower()
    if re.search(r"byte|volume|upload|download|_gb|_mb|_kb", t):
        return "volume"
    if re.search(r"duration|hours|seconds", t):
        return "duration"
    if re.search(r"uniq|users", t):
        return "users"
    if re.search(r"count\(|connections", t):
        return "connections"
    return ""


def measure_or_limit_mismatch(question: str, sql: str, turns: list[dict] | None = None) -> str | None:
    q = (question or "").lower()
    s = sql or ""
    rank_context = q
    if turns and len(q.split()) <= 6:
        rank_context += " " + next((m.get("content", "").lower() for m in reversed(turns) if m.get("role") == "user"), "")
    if not (_RANKING.search(rank_context) or wants_subscriber_list(question)):
        return None
    problems = []
    asked = [k for k, rx in (("volume", _ASK_VOLUME), ("connections", _ASK_CONNECTIONS), ("duration", _ASK_DURATION))
             if rx.search(q)]
    people = r"(users?|people|subscribers|log|logon)"
    if not wants_subscriber_list(question) and re.search(
            rf"(?<![a-z])(by|most|more|number of|count of|how many|kitne|ziyada|zyada)\s+{people}(?![a-z])"
            rf"|(?<![a-z]){people}\s+(and|vs|versus|or)(?![a-z])|(?<![a-z])(and|vs|versus|or)\s+{people}(?![a-z])", q):
        asked.append("users")
    order = None
    for m in re.finditer(r"(?i)\bORDER\s+BY\b", s):
        before = s[:m.start()]
        if before.count("(") == before.count(")"):
            order = s[m.end():]
    if len(asked) == 1 and order:
        first = _split_top(re.split(r"(?i)\bLIMIT\b|;", order)[0])[0]
        kind = _measure_kind(first)
        if kind and kind != asked[0]:
            want = {"volume": "data volume: ORDER BY sum(c2s_byte_num + s2c_byte_num) DESC (upload: sum(c2s_byte_num), "
                              "download: sum(s2c_byte_num))",
                    "connections": "connections: ORDER BY count() DESC",
                    "duration": "time: ORDER BY sum(duration) DESC",
                    "users": "users: ORDER BY uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0) DESC"}[asked[0]]
            problems.append(f"the question asks for the ranking by {want}, but the SQL orders by {first.strip()}")
    n = re.search(r"(?<![a-z])(?:top|first|pehle|ٹاپ)\s+(\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")(?![a-z0-9])", q)
    lim = re.search(r"(?i)\bLIMIT\s+(\d+)\s*;?\s*$", s.strip())
    if n and lim:
        want_n = int(n.group(1)) if n.group(1).isdigit() else _NUMBER_WORDS[n.group(1)]
        if int(lim.group(1)) != want_n:
            problems.append(f"the question asks for the top {want_n}, but the SQL returns {lim.group(1)} rows: use "
                            f"LIMIT {want_n}")
    if not problems:
        return None
    return ("Not accepted yet: " + "; ".join(problems) + ". Fix it, keep everything else, state the measure and the "
            "number in the understood question and title, test, and submit again.")


_DEFAULT_RANK_THINGS = re.compile(r"(?<![a-z])(apps?|applications?|vpns?|isps?|operators?|networks?|cit(?:y|ies)|"
                                  r"categor(?:y|ies)|games?|messengers?)(?![a-z])")
_NATURAL_MEASURE = re.compile(r"(?<![a-z])(blocked|block|websites?|sites?|domains?|hosts?|hours?|time|when|peak|"
                              r"busiest|quietest|days?|daily|servers?|countr(?:y|ies))(?![a-z])")


def default_ranking_measure(question: str, sql: str, turns: list[dict] | None = None) -> str | None:
    q = (question or "").lower()
    names_nothing = not (_DEFAULT_RANK_THINGS.search(q) or concepts.match(q) or concepts.apps_in(q)
                         or _NATURAL_MEASURE.search(q)) and len(q.split()) <= 7
    if turns and (is_follow_up(question) or names_nothing):
        prev = next((m.get("content", "") for m in reversed(turns) if m.get("role") == "user"), "")
        q = f"{q} {prev.lower()}"
        question = f"{question} {prev}"
    if not (_RANKING.search(q) or "used the most" in q) or wants_subscriber_list(question) or _NATURAL_MEASURE.search(q):
        return None
    if not (_DEFAULT_RANK_THINGS.search(q) or concepts.match(q) or concepts.apps_in(q)):
        return None
    asked = [rx for rx in (_ASK_VOLUME, _ASK_CONNECTIONS, _ASK_DURATION) if rx.search(q)]
    if asked or re.search(r"(?<![a-z])(users?|people|subscribers)(?![a-z])", q):
        return None
    s = sql or ""
    order = None
    for m in re.finditer(r"(?i)\bORDER\s+BY\b", s):
        before = s[:m.start()]
        if before.count("(") == before.count(")"):
            order = s[m.end():]
    if not order or not re.search(r"(?i)\bGROUP\s+BY\b", s):
        return None
    first = _split_top(re.split(r"(?i)\bLIMIT\b|;", order)[0])[0]
    one_person = bool(re.search(r"subscriber_id\s*(IN\s*\(|=\s*')", s, re.I))
    want = "connections" if one_person else "users"
    if _measure_kind(first) in (want, "") or (one_person and _measure_kind(first) == "users"):
        return None
    how = ("ORDER BY count() DESC (one subscriber: users is always 1)" if one_person else
           "ORDER BY uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0) DESC, count() DESC")
    return ("Not accepted yet: the question names no measure, so 'most used' follows the team's default: " + how +
            f". The SQL orders by {first.strip()}. Keep volume as an extra column if useful, fix the order, test, and "
            "submit again.")


_SMALL_UNIT = re.compile(r"(gb|mb|kb|volume|upload|download|hours|bytes)", re.I)


def rounded_to_zero(result) -> str | None:
    if result is None or not result.rows:
        return None
    for i, c in enumerate(result.columns):
        if not _SMALL_UNIT.search(c["name"]) or not any(t in str(c.get("type", "")) for t in ("Float", "Int", "Decimal")):
            continue
        vals = [r[i] for r in result.rows if isinstance(r[i], (int, float))]
        name = c["name"].lower()
        too_big_unit = re.search(r"(^|_)(gb|mb|hours)($|_)", name) and vals and max(abs(v) for v in vals) < 1
        if vals and (too_big_unit or sum(1 for v in vals if v == 0) >= len(vals) / 2):
            unit = ("minutes (sum(duration) / 6e4)" if "hours" in name else "KB (/ 1e3)" if "mb" in name
                    else "MB (/ 1e6), or KB (/ 1e3) if MB is still below 1")
            return (f"Not accepted yet: the values in {c['name']} are all below 1 (or 0), so they read as ties. Show "
                    f"them in {unit} with a matching name (volume_mb, volume_kb) and 2 decimals. Change only the unit: "
                    "keep the same rows, filters and ranking measure (if the ranking is by volume, ORDER BY the "
                    "unrounded sum). Do not mention the unit change in the understood question. Test, and submit again.")
    return None


def _normalize_plan(p: dict) -> dict:
    chart = p.get("chart") if isinstance(p.get("chart"), dict) else {}
    assumptions = [a for a in (p.get("assumptions") or []) if isinstance(a, dict) and a.get("label")]
    return {
        "understood": str(p.get("understood") or "").strip(),
        "status": p.get("status") if p.get("status") in ("ok", "clarify", "cannot_answer") else "ok",
        "title": str(p.get("title") or "").strip(),
        "sql": str(p.get("sql") or "").strip(),
        "explanation": str(p.get("explanation") or "").strip(),
        "assumptions": [{"label": str(a["label"]), "value": str(a.get("value", ""))} for a in assumptions],
        "chart": {k: str(chart.get(k) or "") for k in ("type", "x", "y", "series")},
        "message": str(p.get("message") or "").strip(),
    }


_SHOW_WORDS = {"show", "display", "give", "see", "something", "anything", "interesting", "overview", "insight",
               "insights", "stats", "statistics", "report", "dashboard", "dikhao", "dikha", "batao", "kuch", "dekhna"}


def _is_vague(question: str) -> bool:
    q = question.lower()
    words = re.findall(r"[a-z']+", q)
    if not set(words) & _SHOW_WORDS:
        return False
    return (0 < len(words) <= 6 and not re.search(r"\d", q) and not _SUBJECT.search(q) and not _MEASURE_WORDS.search(q)
            and not concepts.match(q) and not concepts.apps_in(q) and not concepts.cities_in(q)
            and not context.operators(q) and not _UNAVAILABLE.search(q))


def _default_overview(question: str, data_range: dict, steps: list, usage: dict, say) -> Answer | None:
    from datetime import timedelta

    ps = context.periods(data_range)
    if "7d" not in ps:
        return None
    _, start, end = ps["7d"]
    generic = dictionary.base().get("value_lists", {}).get("generic_labels", {}).get("list", [])
    not_generic = ("\n  AND app_label NOT IN (" + ", ".join("'" + g.replace("'", "\\'") + "'" for g in generic) + ")"
                   if generic else "")
    tz = settings.timezone
    sql = (f"SELECT app_label AS app,\n       uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0) AS users,\n"
           f"       count() AS connections,\n       round(sum(c2s_byte_num + s2c_byte_num) / 1e9, 2) AS volume_gb\n"
           f"FROM {settings.full_table}\n"
           f"WHERE start_time >= toUnixTimestamp('{start.isoformat()} 00:00:00', '{tz}')\n"
           f"  AND start_time <  toUnixTimestamp('{(end + timedelta(days=1)).isoformat()} 00:00:00', '{tz}')"
           f"{not_generic}\nGROUP BY app\nORDER BY users DESC, connections DESC\nLIMIT 20")
    say("Showing an overview: the most used apps")
    try:
        checked = sql_guard.check(sql)
        result = db.cached_query(checked, timeout=_QUERY_TIMEOUT)
    except (sql_guard.GuardError, QueryError):
        return None
    if not result.rows:
        return None
    plan = {"understood": f"No specific question was asked, so this is an overview: the most used apps in Pakistan "
                          f"from {_fmt_period(start, end)}.",
            "status": "ok", "title": "Most used apps, last 7 days", "sql": sql,
            "explanation": "An overview, because the request named no app, place, period or measure. Ask about "
                           "anything more specific in your own words.",
            "assumptions": [{"label": "Measure", "value": "Identified users, then connections"},
                            {"label": "Excluded", "value": "background traffic"}],
            "chart": {"type": "bar", "x": "app", "y": "users", "series": ""}, "message": ""}
    return Answer(plan=align_period_text(plan, sql), sql=checked, result=result, steps=steps, usage=usage, tested=True)


def _fallback_understood(question: str, turns: list[dict]) -> str:
    q = " ".join(question.split())
    prev = next(((m.get("data") or {}).get("understood") for m in reversed(turns or [])
                 if m.get("role") == "assistant" and m.get("kind") == "result" and (m.get("data") or {}).get("understood")),
                None)
    if prev and is_follow_up(question):
        return f"{prev.rstrip('?. ')}, changed to: {q}"
    return q


DEBUG_LOG = ROOT / "logs" / "agent_debug.jsonl"


def _debug(question: str, **fields) -> None:
    try:
        DEBUG_LOG.parent.mkdir(exist_ok=True)
        entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question[:200],
                 **{k: v for k, v in fields.items() if v is not None}}
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


_WRITE_REQUEST = re.compile(
    r"(?<![a-z])(drop|truncate)\s+(table|database|the|all)(?![a-z])|(?<![a-z])delete\s+(all|from|the|every|records|data|rows)"
    r"(?![a-z])|(?<![a-z])insert\s+into(?![a-z])|(?<![a-z])update\s+[a-z_.]+\s+set(?![a-z])|(?<![a-z])alter\s+table(?![a-z])|"
    r"(?<![a-z])ignore\s+(your|the|all|previous|these)\s+(instructions|rules|prompt)")


_REFUSAL_TEXT = re.compile(r"(?i)^(i\s*(am|'m)\s*sorry|sorry|i\s*can(no|')t|i\s*cannot|i\s*am\s*unable|i'm\s*unable|"
                           r"unable to|the (question|request) asks to|this request|that request)")


_META_UNDERSTOOD = re.compile(r"(?i)\b(the sql|the query|should (filter|use|be)|not accepted|the previous quer|"
                              r"specifically asks|is distinct from|the check|resubmit|submitted)\b")


@functools.lru_cache(maxsize=1)
def _code_version() -> str:
    import hashlib

    h = hashlib.sha256()
    for name in ("agent.py", "context.py", "catalog.py", "concepts.py", "sql_guard.py", "urdu.py"):
        try:
            h.update((ROOT / "ipdr" / name).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:16]


def cache_key(question: str, data_range: dict) -> str:
    import hashlib

    q = " ".join(re.sub(r"[^\w\s']", " ", question.lower()).split())
    basis = json.dumps([q, data_range.get("last_full_day"), data_range.get("last", "")[:13],
                        dictionary.render_for_prompt(), _code_version()], ensure_ascii=False)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def answer(question: str, turns: list[dict], cfg: LLMConfig, data_range: dict,
           step: StepFn, cancel: threading.Event | None = None) -> Answer:
    question = urdu.with_latin(question)
    if _WRITE_REQUEST.search(question.lower()):
        step("Reading the question")
        return Answer(plan={"understood": f"Request: {' '.join(question.split())}", "status": "cannot_answer",
                            "title": "", "sql": "", "explanation": "", "assumptions": [], "chart": {},
                            "message": "This app can only read the network records. It cannot delete, change or add "
                                       "data, and its instructions cannot be changed from the chat. Ask a question about "
                                       "the data instead, for example \"top apps in Karachi last week\"."},
                      steps=["Reading the question"])
    q_low = question.lower()
    missing = missing_in(q_low)
    if missing and not turns and not (concepts.match(q_low) or concepts.apps_in(q_low)):
        step("Reading the question")
        what = " or ".join(missing)
        return Answer(plan={"understood": f"Request: {' '.join(question.split())}", "status": "cannot_answer",
                            "title": "", "sql": "", "explanation": "", "assumptions": [], "chart": {},
                            "message": f"This data has no {what} information, so this cannot be answered. It can answer "
                                       "questions about apps, places, times, servers and subscribers, for example "
                                       "\"how many people used TikTok in Karachi last week\"."},
                      steps=["Reading the question"])
    key = None if turns or not ANSWER_CACHE else cache_key(question, data_range)
    if key:
        hit = _from_cache(key, question, step)
        if hit is not None:
            return hit
    ans = _answer(question, turns, cfg, data_range, step, cancel)
    ans.plan = _safe(note_partial_days, ans.plan, ans.result, data_range) or ans.plan
    understood = (ans.plan.get("understood") or "").strip()
    if ans.result is None and (not understood or _REFUSAL_TEXT.search(understood)):
        ans.plan["understood"] = f"Request: {' '.join(question.split())}"
    elif ans.result is not None and _META_UNDERSTOOD.search(understood):
        ans.plan["understood"] = " ".join(question.split())
        ans.plan = align_period_text(ans.plan, ans.plan.get("sql") or "")
    if ans.result is None and not (ans.plan.get("message") or "").strip():
        ans.plan["message"] = ("Could you say a little more about what you want to see? For example the app, the place "
                               "or the period." if ans.plan.get("status") == "clarify" else
                               "This could not be answered from the network records. Ask about app usage, places, "
                               "times, servers or subscribers, for example \"top apps in Karachi last week\".")
    if key and ans.result is not None and ans.result.rows and ans.sql:
        try:
            store.save_answer(key, question, {**ans.plan, "sql": ans.plan.get("sql") or ans.sql})
        except Exception:
            pass
    return ans


def _from_cache(key: str, question: str, step: StepFn) -> Answer | None:
    try:
        plan = store.cached_answer(key)
    except Exception:
        return None
    if not plan or not plan.get("sql"):
        return None
    steps = ["Same question as before: using its checked query"]
    step(steps[0])
    try:
        checked = sql_guard.check(plan["sql"])
        result = db.cached_query(checked, timeout=_QUERY_TIMEOUT)
    except (sql_guard.GuardError, QueryError):
        store.forget_answer(key)
        return None
    if not result.rows:
        store.forget_answer(key)
        return None
    step("Running the final query")
    return Answer(plan=plan, sql=checked, result=result, steps=steps + ["Running the final query"], tested=True)


def _answer(question: str, turns: list[dict], cfg: LLMConfig, data_range: dict,
            step: StepFn, cancel: threading.Event | None = None) -> Answer:
    cancel = cancel or threading.Event()
    steps: list[str] = []

    def say(text: str) -> None:
        if not steps or steps[-1] != text:
            steps.append(text)
            step(text)

    system = context.sql_system_prompt(dictionary.render_for_prompt())
    if turns and (is_follow_up(question) or context.references(question, turns)):
        examples = []
    else:
        examples = retrieval.similar(question, k=4)
    user = context.sql_user_message(question, data_range, date.today().isoformat(), turns, examples,
                                    resolved_names(question, data_range, turns))
    tools = context.sql_tools(dictionary.string_columns())

    say("Reading the question")
    try:
        session = llm.open_session(cfg, system, tools, user)
    except LLMError as e:
        raise AgentError(str(e)) from e

    usage = {"input": 0, "output": 0, "cache_read": 0, "turns": 0}
    tested_sql: set[str] = set()
    empty_checked: set[str] = set()
    max_turns = settings.max_agent_turns
    review = review_enabled(cfg) and (needs_review(question) or (settings.review_pass or "").lower() in ("on", "always", "true", "1"))
    reviewed = False
    pending = None
    period_flagged: set[str] = set()
    concept_pushed = False
    concept_checked = False
    clarify_pushed = False
    place_checked = False
    unasked_checked = False
    old_place_checked = False
    unknown_city_checked = False
    grouped_checked = False
    shape_checked = False
    generic_checked = False
    bucket_checked = False
    phone_pushed = False
    refusal_pushed = False
    empty_col_checked = False
    last_plan = None
    apps_checked = False
    cities_checked = False
    operator_pushed = False
    ref_pushed = False
    isp_checked = False
    subs_checked = False
    zero_checked = False
    measure_checked = False
    outside_checked = False
    negation_checked = False
    app_ignored_checked = False
    per_user_checked = False
    field_checked: set[str] = set()
    average_checked = False
    unasked_kind_checked = False
    default_rank_checked = False
    follow_filters_checked = 0
    placeholder_checked = False
    people_count_checked = False
    follow_period_checked = False
    empty_group_checked = False
    total_checked = False
    brand_checked = False
    labels_checked = False
    share_checked = False
    identified_checked = False
    ref_checked = False

    review_turn = None
    failures: dict[str, int] = {}
    heavy = 0
    ran = None
    for turn_no in range(max_turns):
        if cancel.is_set():
            raise Cancelled()
        if pending is not None and review_turn is not None and turn_no - review_turn > REVIEW_TURNS:
            break
        if turn_no:
            say("Working on the query")
        try:
            turn = session.send()
        except LLMError as e:
            raise AgentError(str(e)) from e
        usage["turns"] += 1
        for k in ("input", "output", "cache_read"):
            usage[k] += turn.usage.get(k, 0)
        if cancel.is_set():
            raise Cancelled()
        if turn.stop == "refusal":
            raise AgentError("The AI declined to answer this request.")
        if turn.stop == "max_tokens":
            raise AgentError("The AI answer was cut off. Try a narrower question.")

        _debug(question, turn=turn_no, calls=[{"tool": c.name, "sql": str(c.args.get("sql", ""))[:400] or None,
                                               "search": c.args.get("search")} for c in turn.tool_calls]
               or None, text=(turn.text or "")[:300] if not turn.tool_calls else None)
        confirms = [c for c in turn.tool_calls if c.name == "confirm_answer"]
        if confirms and pending is not None:
            say("Answer confirmed")
            p_plan, p_sql, p_result, p_key = pending
            return Answer(plan=note_missing(align_period_text(p_plan, p_plan.get("sql", "")), question), sql=p_sql, result=p_result, steps=steps,
                          usage=usage, tested=p_key in tested_sql)
        submits = [c for c in turn.tool_calls if c.name == "submit_answer"]
        others = [c for c in turn.tool_calls if c.name not in ("submit_answer", "confirm_answer")]

        results: list[tuple[ToolCall, str, bool]] = [
            (c, "Nothing to confirm yet. Submit your answer with submit_answer.", True) for c in confirms]
        if others:
            for c in others:
                say(_describe(c))
            results += list(_pool.map(lambda c: _run_tool(c, data_range), others))
            for c, text, err in results:
                if c.name == "run_sql" and not err:
                    tested_sql.add(_norm_sql(str(c.args.get("sql", ""))))
                if err:
                    _debug(question, turn=turn_no, tool_error=c.name, message=str(text)[:400])
                    if _too_heavy(text):
                        heavy += 1
            if heavy >= 2:
                raise AgentError(
                    "The query was stopped because it ran too long. The database is up. "
                    "Ask for a shorter period or a simpler breakdown."
                )

        plan_raw = None
        submit_call = None
        if submits:
            submit_call = submits[0]
            plan_raw = submit_call.args
            for extra in submits[1:]:
                results.append((extra, "Ignored: submit only one answer.", True))
        elif not turn.tool_calls:
            plan_raw = _plan_from_text(turn.text)
            if plan_raw is None and pending is not None:
                say("Answer confirmed")
                p_plan, p_sql, p_result, p_key = pending
                return Answer(plan=note_missing(align_period_text(p_plan, p_plan.get("sql", "")), question), sql=p_sql,
                              result=p_result, steps=steps, usage=usage, tested=p_key in tested_sql)
            if plan_raw is None and _REFUSAL_TEXT.search((turn.text or "").strip()):
                plan_raw = {"status": "cannot_answer", "message": turn.text.strip()[:500], "understood": "", "sql": ""}
            if plan_raw is None:
                session.add_user("Call the submit_answer tool with your answer. Do not answer in plain text.")
                continue

        if plan_raw is not None:
            plan = _normalize_plan(plan_raw)
            if last_plan is not None:
                for k in ("understood", "title", "explanation", "assumptions"):
                    if not plan[k]:
                        plan[k] = last_plan[k]
                if not plan["chart"].get("type"):
                    plan["chart"] = last_plan["chart"]
            if plan["status"] == "ok" and not plan["understood"]:
                plan["understood"] = _fallback_understood(question, turns)
            if plan["status"] == "ok":
                last_plan = plan
            feedback = None
            asks_missing = bool(_UNAVAILABLE.search(question.lower())) and not _STORED_IDS.search(question.lower())
            concept_hint = (concepts.hint_for(question)
                            if plan["status"] != "ok" and not concept_pushed and not asks_missing else "")
            fallback = concept_fallback(question, data_range) if plan["status"] != "ok" else None
            if concept_hint:
                concept_pushed = True
                say("Reconsidering the meaning of the question")
                ready = (f" A query that answers it (adapt it if the question asks for more):\n{fallback[4]}"
                         if fallback else "")
                if (city := concepts.city_in(question)):
                    ready += f" '{city}' is a city: filter with client_location = '{city} Pakistan'."
                feedback = (f"Not accepted yet. {concept_hint} The data can answer the question with these apps, so "
                            "write the query using the concept's app_label list (state your interpretation as an "
                            f"assumption), test it, and submit it.{ready} Only if it truly cannot be answered, submit "
                            f"{plan['status']} again with the reason and what can be answered instead.")
            elif plan["status"] != "ok" and not ref_pushed and (refs := context.references(question, turns)):
                ref_pushed = True
                say("Using the rows of the earlier answer")
                feedback = ("Not accepted yet: the question can be answered without asking the user. " +
                            " ".join(context.reference_lines(refs, wants_profile(question))) + " " + " ".join(shape_hints(question)) +
                            " Write the query, test it, and submit it.")
            elif plan["status"] == "clarify" and not clarify_pushed and not asks_missing and _lazy_clarify(question, plan):
                clarify_pushed = True
                say("Looking again before asking you")
                city = concepts.city_in(question)
                where = f" '{city}' is a city: use client_location = '{city} Pakistan'." if city else ""
                feedback = ("Not accepted yet: the question can be answered without asking the user." + where +
                            " Look names up in the right column (cities in client_location, neighbourhoods and "
                            "roads in loc, apps in app_label or app_concepts), use the defaults for anything not "
                            "stated, write the query, test it, and submit it. Ask the user only about a real "
                            "ambiguity, such as a place that exists in several cities.")
            elif plan["status"] != "ok" and not phone_pushed and (numbers := context.phones(question)):
                phone_pushed = True
                say("Looking the number up in its stored form")
                feedback = (f"Not accepted yet: a mobile number is stored in subscriber_id as 92XXXXXXXXXX or as "
                            f"03XXXXXXXXX. Query the question's period with {context.phone_sql(numbers)}, test it, and "
                            "submit it. Say the number has no records only if that query returns no rows.")
            elif plan["status"] != "ok" and not operator_pushed and (op_note := context.operator_note(question)):
                operator_pushed = True
                say("Using the operator codes")
                feedback = (f"Not accepted yet: the data can answer this. {op_note} Write the query with that filter, "
                            "test it, and submit it.")
            elif (plan["status"] == "cannot_answer" and not refusal_pushed and not asks_missing
                  and not (fallback and concept_pushed) and not _CHANGE_DATA.search(question.lower())):
                refusal_pushed = True
                say("Answering with the defaults")
                understood = (plan.get("understood") or "").strip()
                feedback = ("Not accepted yet: the data can answer this. Use the data dictionary's defaults for anything "
                            "the question does not state (all of Pakistan when no place is named, the last 7 full days "
                            "when no period is named, users with connections and volume_gb when no measure is named)"
                            + (f" and answer the question as you understood it: \"{understood}\"" if understood else "")
                            + ". Write the SQL and submit it. Submit cannot_answer again only if the data truly does not "
                            "hold what is asked, and then say exactly what is missing.")
            elif plan["status"] != "ok" and fallback and concept_pushed:
                ans = _answer_with_fallback(fallback, plan, steps, usage, say)
                if ans is not None:
                    return ans
                return Answer(plan=plan, steps=steps, usage=usage)
            elif plan["status"] != "ok":
                if (refusal_pushed or clarify_pushed) and _is_vague(question):
                    overview = _default_overview(question, data_range, steps, usage, say)
                    if overview is not None:
                        return overview
                if refusal_pushed and plan["status"] == "cannot_answer" and not missing_in(question.lower()):
                    fallback = _safe(profile_fallback, question, turns, data_range)
                    if fallback is not None:
                        say("Looking up the stored details directly")
                        f_plan, f_sql, f_result = fallback
                        return Answer(plan=f_plan, sql=f_sql, result=f_result, steps=steps, usage=usage, tested=True)
                return Answer(plan=plan, steps=steps, usage=usage)
            elif not plan["sql"]:
                feedback = "The answer has status ok but no SQL. Submit again with the tested SQL."
            else:
                say("Checking the answer")
                submitted_key = _norm_sql(plan["sql"])
                if (fixed := repair_measures(plan["sql"])):
                    plan["sql"] = fixed
                if (fixed := _safe(fix_rounded_order, plan["sql"])):
                    plan["sql"] = fixed
                if (fixed := _safe(fix_brand_values, plan["sql"])):
                    plan["sql"] = fixed
                if (fixed := _safe(fix_isp_groups, plan["sql"])):
                    plan["sql"] = fixed
                if (fixed := _safe(fix_app_groups, plan["sql"], question)):
                    plan["sql"] = fixed
                if (fixed := _safe(fix_user_count, plan["sql"])):
                    plan["sql"] = fixed
                if (fixed := _safe(fix_limit_one, plan["sql"], question)):
                    plan["sql"] = fixed
                if submitted_key in tested_sql:
                    tested_sql.add(_norm_sql(plan["sql"]))
                try:
                    checked = sql_guard.check(plan["sql"])
                    say("Running the final query")
                    result = db.cached_query(checked, timeout=_QUERY_TIMEOUT)
                except (sql_guard.GuardError, QueryError) as e:
                    say("Fixing the query")
                    if _too_heavy(e):
                        heavy += 1
                        if heavy >= 2:
                            raise AgentError(
                                "The query was stopped because it ran too long. The database is up. "
                                "Ask for a shorter period or a simpler breakdown."
                            )
                        feedback = ("Not accepted. The query was stopped because it ran too long. The database is up. "
                                    "Keep the same start_time range, use one GROUP BY, no subquery, and LIMIT 10. "
                                    "Test that simpler query with run_sql, then submit it.")
                    else:
                        failed_key = _norm_sql(plan["sql"])
                        failures[failed_key] = failures.get(failed_key, 0) + 1
                        if failures[failed_key] > 1:
                            feedback = (f"Not accepted. You submitted the SAME SQL again and it fails the same way: {e}. "
                                        "Do not resend it. Write a simpler query: no subquery unless it is needed, and select "
                                        "the table's columns directly (a column renamed inside a subquery is only known by "
                                        "its new name outside it). Test it with run_sql, then submit.")
                        else:
                            feedback = f"Not accepted. The SQL failed when run: {e}. Fix it, test it with run_sql, then submit again."
                else:
                    key = _norm_sql(plan["sql"])
                    empty_col = uses_empty_column(plan["sql"])
                    if result.rows and not empty_col:
                        ran = (plan, checked, result, key)
                    if empty_col and empty_col_checked:
                        cols = ", ".join(c for c in (dictionary.base().get("empty_columns") or {}).get("list", [])
                                         if c in plan["sql"])
                        return Answer(plan={**plan, "status": "cannot_answer", "sql": "",
                                            "message": f"This data has no {cols.replace('_', ' ')} information (the "
                                                       "column is empty in every record), so this cannot be answered. "
                                                       "Ask about apps, places, times, servers or subscribers instead."},
                                      steps=steps, usage=usage)
                    if empty_col:
                        empty_col_checked = True
                        say("Checking the columns used")
                        feedback = empty_col
                    elif not result.rows and key not in empty_checked:
                        empty_checked.add(key)
                        say("Double-checking an empty result")
                        feedback = ("Not accepted yet: the query returned no rows. Check the names with find_values, "
                                    "the period and the filters. If no rows is truly the right answer, submit the same "
                                    "SQL again and say so in the explanation.")
                    elif (wrong := period_mismatch(question, plan["sql"], data_range)) and key not in period_flagged:
                        period_flagged.add(key)
                        say("Fixing the dates")
                        feedback = wrong
                    elif not ref_checked and (dropped := _safe(reference_dropped, question, plan["sql"], turns)):
                        ref_checked = True
                        say("Using the rows of the earlier answer")
                        feedback = dropped
                    elif not isp_checked and (no_name := _safe(entity_missing, question, plan["sql"], turns)):
                        isp_checked = True
                        say("Adding the names the question gives")
                        feedback = no_name
                    elif not labels_checked and (gaps := _safe(app_labels_incomplete, question, plan["sql"])):
                        labels_checked = True
                        say("Including every label of the app")
                        feedback = gaps
                    elif not brand_checked and (brands := _safe(isp_split_in_ranking, result)):
                        brand_checked = True
                        say("One row per ISP brand")
                        feedback = brands
                    elif not total_checked and (one := _safe(total_as_list, question, result, turns)):
                        total_checked = True
                        say("Returning one total")
                        feedback = one
                    elif not share_checked and (whole := _safe(share_of_itself, question, plan["sql"], result)):
                        share_checked = True
                        say("Dividing by the whole")
                        feedback = whole
                    elif not identified_checked and (alltraffic := _safe(identified_filter_on_traffic, question, plan["sql"])):
                        identified_checked = True
                        say("Counting all traffic")
                        feedback = alltraffic
                    elif not people_count_checked and (rows_people := _safe(rows_counted_as_people, question, plan["sql"])):
                        people_count_checked = True
                        say("Counting people, not connections")
                        feedback = rows_people
                    elif not follow_period_checked and (fperiod := _safe(follow_up_period_changed, question, plan["sql"], turns)):
                        follow_period_checked = True
                        say("Keeping the earlier period")
                        feedback = fperiod
                    elif not placeholder_checked and (holder := _safe(placeholder_in_sql, plan["sql"])):
                        placeholder_checked = True
                        say("Using real values")
                        feedback = holder
                    elif follow_filters_checked < 2 and (kept := _safe(follow_up_dropped_filters, question, plan["sql"], turns)):
                        follow_filters_checked += 1
                        say("Keeping the filters of the earlier answer")
                        feedback = kept
                    elif not unasked_kind_checked and (kind_extra := _safe(unasked_app_filter, question, plan["sql"], turns)):
                        unasked_kind_checked = True
                        say("Removing filters the question did not ask for")
                        feedback = kind_extra
                    elif not app_ignored_checked and (ignored := _safe(app_ignored, question, plan["sql"])):
                        app_ignored_checked = True
                        say("Adding the app the question names")
                        feedback = ignored
                    elif not average_checked and (avg_missing := _safe(average_missing, question, plan["sql"])):
                        average_checked = True
                        say("Computing the average per user")
                        feedback = avg_missing
                    elif not per_user_checked and (per_user := _safe(per_user_with_unidentified, question, plan["sql"])):
                        per_user_checked = True
                        say("Averaging over identified subscribers")
                        feedback = per_user
                    elif (fc := _safe(field_checks, question, plan["sql"], field_checked)):
                        field_checked.add(fc[0])
                        say(fc[1])
                        feedback = fc[2]
                    elif not negation_checked and (neg := _safe(negation_as_row_filter, question, plan["sql"])):
                        negation_checked = True
                        say("Checking who did not use it, person by person")
                        feedback = neg
                    elif not outside_checked and (outside := _safe(period_outside_data, plan["sql"], data_range)):
                        outside_checked = True
                        say("Using dates inside the data")
                        feedback = outside
                    elif not default_rank_checked and (default_rank := _safe(default_ranking_measure, question, plan["sql"], turns)):
                        default_rank_checked = True
                        say("Ranking by the default measure")
                        feedback = default_rank
                    elif not measure_checked and (asked := _safe(measure_or_limit_mismatch, question, plan["sql"], turns)):
                        measure_checked = True
                        say("Ranking by the measure the question asks for")
                        feedback = asked
                    elif not zero_checked and (zeros := _safe(rounded_to_zero, result)):
                        zero_checked = True
                        say("Showing small amounts in a smaller unit")
                        feedback = zeros
                    elif not subs_checked and (subs := _safe(subscriber_shape, question, plan["sql"], result)):
                        subs_checked = True
                        say("Listing individual subscribers")
                        feedback = subs
                    elif not place_checked and (wrong_place := place_mismatch(question, plan["sql"])):
                        place_checked = True
                        say("Fixing the place")
                        feedback = wrong_place
                    elif not shape_checked and (shape := ranking_as_total(question, result)
                                                or unasked_app_breakdown(question, result)):
                        shape_checked = True
                        say("Ranking instead of one total")
                        feedback = shape
                    elif not grouped_checked and (split := labels_not_grouped(question, result)):
                        grouped_checked = True
                        say("Showing one row per app")
                        feedback = split
                    elif not empty_group_checked and (empty := _safe(empty_group_in_ranking, result, plan["sql"])):
                        empty_group_checked = True
                        say("Leaving out records with no value")
                        feedback = empty
                    elif not unknown_city_checked and (unknown := unknown_city_in_ranking(result)):
                        unknown_city_checked = True
                        say("Leaving out records with no known city")
                        feedback = unknown
                    elif not old_place_checked and (old_place := follow_up_kept_old_place(question, plan["sql"], turns)):
                        old_place_checked = True
                        say("Switching to the new place")
                        feedback = old_place
                    elif not unasked_checked and (extra_place := unasked_place(question, plan["sql"], turns)):
                        unasked_checked = True
                        say("Removing places the question did not name")
                        feedback = extra_place
                    elif not generic_checked and (noise := generic_in_ranking(question, plan["sql"], result)):
                        generic_checked = True
                        say("Leaving out background traffic")
                        feedback = noise
                    elif not bucket_checked and (extra := extra_other_bucket(question, result)):
                        bucket_checked = True
                        say("Leaving out groups the question did not ask for")
                        feedback = extra
                    elif not apps_checked and (gone := named_app_missing(question, plan["sql"])):
                        apps_checked = True
                        say("Adding the apps the question names")
                        feedback = gone
                    elif not cities_checked and (merged := cities_merged(question, plan["sql"])):
                        cities_checked = True
                        say("Showing each city separately")
                        feedback = merged
                    elif not concept_checked and (gap := concept_incomplete(question, plan["sql"])):
                        concept_checked = True
                        say("Checking the app group is complete")
                        feedback = gap
                    elif review and not reviewed and turn_no < max_turns - 2:
                        reviewed = True
                        review_turn = turn_no
                        pending = (plan, checked, result, key)
                        say("Checking the query against the question")
                        feedback = _review_request(plan, result, context.date_helpers(data_range))
                    else:
                        return Answer(plan=note_missing(align_period_text(plan, plan["sql"]), question), sql=checked, result=result, steps=steps, usage=usage,
                                      tested=key in tested_sql)
            _debug(question, turn=turn_no, status=plan.get("status"), feedback=(feedback or "")[:400],
                   sql=(plan.get("sql") or "")[:2000] or None)
            if submit_call is not None:
                results.append((submit_call, feedback, True))
            else:
                session.add_user(feedback)
                continue

        if results:
            left = max_turns - turn_no - 1
            if left <= 2:
                call, text, err = results[-1]
                results[-1] = (call, text + f"\n\n(Only {left} turn(s) left: submit your best tested answer now, or call confirm_answer if the answer under review is right.)", err)
            session.add_tool_results(results)

    if pending is not None:
        say("Using the answer that passed the checks")
        p_plan, p_sql, p_result, p_key = pending
        return Answer(plan=note_missing(align_period_text(p_plan, p_plan.get("sql", "")), question), sql=p_sql,
                      result=p_result, steps=steps, usage=usage, tested=p_key in tested_sql)
    if ran is not None:
        say("Showing the last answer that ran")
        r_plan, r_sql, r_result, r_key = ran
        r_plan = dict(r_plan)
        r_plan["explanation"] = ((r_plan.get("explanation") or "").rstrip() + " Note: the AI ran out of time while "
                                 "improving this answer, so check the period and filters shown before relying on it.").strip()
        return Answer(plan=note_missing(align_period_text(r_plan, r_plan.get("sql", "")), question), sql=r_sql,
                      result=r_result, steps=steps, usage=usage, tested=r_key in tested_sql)
    fallback = _safe(profile_fallback, question, turns, data_range)
    if fallback is not None:
        say("Looking up the stored details directly")
        f_plan, f_sql, f_result = fallback
        return Answer(plan=f_plan, sql=f_sql, result=f_result, steps=steps, usage=usage, tested=True)
    raise AgentError("The AI could not produce a working query for this question. Try rephrasing it, or be more specific.")


_PROFILE_DEFAULT = ("arrayStringConcat(arraySort(groupUniqArrayIf(loc, loc != '')), ' | ') AS address, arrayStringConcat(arraySort(groupUniqArrayIf(client_asn, client_asn != '')), ' | ') AS isp, "
                    "arrayStringConcat(arraySort(groupUniqArrayIf(client_location, client_location != '')), ' | ') AS city")


def profile_fallback(question: str, turns: list[dict] | None, data_range: dict):
    from datetime import timedelta
    refs = context.references(question, turns) if turns else []
    if not refs or refs[0]["column"] != "subscriber_id" or not wants_profile(question):
        return identifier_fallback(question, turns, data_range)
    ids = context._stored_values("subscriber_id", refs[0]["values"])
    hint = shape_hints(question)
    m = re.search(r"SELECT subscriber_id, (.*?) \.\.\. GROUP BY", " ".join(hint))
    cols = m.group(1) if m else _PROFILE_DEFAULT
    span = follow_up_period(question, turns) or (lambda p: (p[1], p[2]) if p else None)(question_period(question, data_range))
    if not span:
        return None
    start, end = span
    tz = settings.timezone
    id_list = ", ".join("'" + i.replace("'", "") + "'" for i in ids)
    sql = (f"SELECT subscriber_id, {cols}\nFROM {settings.full_table}\n"
           f"WHERE start_time >= toUnixTimestamp('{start.isoformat()} 00:00:00', '{tz}')\n"
           f"  AND start_time <  toUnixTimestamp('{(end + timedelta(days=1)).isoformat()} 00:00:00', '{tz}')\n"
           f"  AND subscriber_id IN ({id_list})\nGROUP BY subscriber_id")
    checked = sql_guard.check(sql)
    result = db.cached_query(checked, timeout=_QUERY_TIMEOUT)
    if not result.rows:
        return None
    plan = {"understood": f"Stored details of the {len(ids)} subscriber(s) from the earlier answer, {_fmt_period(start, end)}.",
            "status": "ok", "title": "Subscriber details", "sql": sql,
            "explanation": "The stored details (address, ISP, city and what else was asked) of the subscribers in the "
                           "earlier answer, looked up directly.",
            "assumptions": [], "chart": {"type": "table", "x": "", "y": "", "series": ""}, "message": ""}
    return align_period_text(plan, sql), checked, result


def identifier_fallback(question: str, turns: list[dict] | None, data_range: dict):
    from datetime import timedelta
    ents = catalog.entities(question)
    subs = [e for e in ents if e.kind == "subscriber" and "_" not in e.name]
    keys = [e for e in ents if e.kind in ("imsi", "imei", "ip")]
    if not subs and not keys:
        return None
    span = follow_up_period(question, turns) or (lambda p: (p[1], p[2]) if p else None)(question_period(question, data_range))
    if not span:
        return None
    start, end = span
    tz = settings.timezone
    period = (f"start_time >= toUnixTimestamp('{start.isoformat()} 00:00:00', '{tz}')\n"
              f"  AND start_time <  toUnixTimestamp('{(end + timedelta(days=1)).isoformat()} 00:00:00', '{tz}')")
    if keys and not subs:
        e = keys[0]
        sql = (f"SELECT subscriber_id, count() AS connections, anyIf(client_asn, client_asn != '') AS isp, "
               f"any(client_location) AS city\nFROM {settings.full_table}\nWHERE {period}\n  AND {e.sql}\n"
               f"GROUP BY subscriber_id\nORDER BY connections DESC\nLIMIT 50")
        title, what = f"Who has {e.kind.upper() if e.kind != 'ip' else 'IP'} {e.name}", f"the subscribers seen with {e.name}"
    elif subs and wants_profile(question):
        hint = shape_hints(question)
        m = re.search(r"SELECT subscriber_id, (.*?) \.\.\. GROUP BY", " ".join(hint))
        cols = m.group(1) if m else _PROFILE_DEFAULT
        flt = " OR ".join(f"({e.sql})" for e in subs)
        sql = (f"SELECT subscriber_id, {cols}\nFROM {settings.full_table}\nWHERE {period}\n  AND ({flt})\n"
               f"GROUP BY subscriber_id")
        title, what = "Subscriber details", "the stored details of " + ", ".join(e.name for e in subs)
    else:
        return None
    checked = sql_guard.check(sql)
    result = db.cached_query(checked, timeout=_QUERY_TIMEOUT)
    if not result.rows:
        return None
    plan = {"understood": f"{title}, {_fmt_period(start, end)}.", "status": "ok", "title": title, "sql": sql,
            "explanation": f"Looked up directly: {what}, {_fmt_period(start, end)}.",
            "assumptions": [], "chart": {"type": "table", "x": "", "y": "", "series": ""}, "message": ""}
    return align_period_text(plan, sql), checked, result


_PERIOD_WORDS = [
    (re.compile(r"\b(day before yesterday|parso|parson)\b"), "day_before"),
    (re.compile(r"\b(yesterday|yday|kal|today|aaj|aj|last night|tonight)\b"), "day"),
    (re.compile(r"\b(last|past|pichl[aeiy]*)\s*(3|three|teen)\s*(days?|din)\b"), "3d"),
    (re.compile(r"\b(last|past|pichl[aeiy]*)\s*(7|seven|saat)\s*(days?|din)\b|\b(last|past|pichl[aeiy]*)\s*(week|wk|hafta|hafte|haftay)\b"), "7d"),
    (re.compile(r"\b(last|past)\s*(10|ten)\s*days?\b"), "10d"),
    (re.compile(r"\b(last|past)\s*(14\s*days?|2\s*weeks?|two\s*weeks?)\b"), "14d"),
    (re.compile(r"\b(last|past)\s*30\s*days?\b"), "30d"),
    (re.compile(r"\b(this month|is mahine|is mahinay)\b"), "this_month"),
    (re.compile(r"\b(last|past)\s*24\s*(hours?|hrs?)\b"), "day"),
    (re.compile(r"\b(last|past|previous)\s*fortnight\b"), "14d"),
    (re.compile(r"\b(previous|prior)\s*(week|7 days)\b"), "7d"),
    (re.compile(r"\bthis\s*(week|hafta|hafte)\b"), "this_week"),
    (re.compile(r"\b(last|previous|prior|pichl[aeiy]*)\s*(month|mahin[ae])\b|\b(in|during|for|of)\s+october\b(?!\s*\d)"),
     "prev_month"),
    (re.compile(r"\b(whole|entire|full|all the|all)\s+(data|period|dataset|records)\b|\ball[- ]time\b|"
                r"\bsince the (start|beginning)\b"), "all"),
    (re.compile(r"\b(last|this|past|over the|on the|at the|previous)\s*weekend\b"), "weekend"),
]
_TIME_REF = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b20\d\d\b|\b\d{1,2}[/.-]\d{1,2}\b"
    r"|\b\d{1,2}(st|nd|rd|th)\b|\b(today|yesterday|yday|tomorrow|kal|aaj|aj|parso|last night|tonight|week|weeks|wk|"
    r"weekly|month|months|monthly|mahine|mahina|hafta|hafte|haftay|days?|din|"
    r"weekend|all time|all data|overall|ever|total period|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"somwar|mangal|budh|jumerat|jumma|juma|itwar)\b"
    r"|\b(since|between|from|until|till|during|after|before)\s+(the\s+)?((?!\d{1,2}(:\d\d)?\s*(am|pm|baje|o'?clock|:|h\b|"
    r"hrs|hours))\d|last|this|past|previous|yesterday|today|kal|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|mon|tue|wed|thu|fri|sat|sun)")
_FOLLOW_UP = re.compile(r"^(and|aur|or|also|now|only|same|what about|how about|ab|sirf|phir)\b")
_SUBJECT = re.compile(r"(?<![a-z])(apps?|users?|people|ppl|log|data|traffic|usage|volume|connections?|sites?|servers?|"
                      r"subscribers?|records?|hour|peak|cities|city)(?![a-z])")


def is_follow_up(question: str) -> bool:
    q = question.strip().lower()
    if len(q.split()) > 6:
        return False
    if _FOLLOW_UP.search(q):
        return True
    if not q.rstrip().endswith("?"):
        return False
    return not (_SUBJECT.search(q) or concepts.match(q) or concepts.apps_in(q))


def _answer_with_fallback(fallback, model_plan: dict, steps: list, usage: dict, say) -> Answer | None:
    c, start, end, city, sql, total = fallback
    name = c["key"].replace("_", " ")
    say(f"Answering with the standard {'count' if total else 'ranking'} of {name} apps")
    try:
        checked = sql_guard.check(sql)
        result = db.cached_query(checked, timeout=_QUERY_TIMEOUT)
    except (sql_guard.GuardError, QueryError):
        return None
    if not result.rows:
        return None
    where = f" in {city}" if city else ""
    when = f" on {_fmt_period(start, end)}" if start == end else f" from {_fmt_period(start, end)}"
    plan = {
        "understood": f"Which {name} apps were used the most{where}{when}?",
        "status": "ok",
        "title": f"Most used {name} apps{', ' + city if city else ''}",
        "sql": sql,
        "explanation": (f"Ranks every app in the '{name}' group ({c['description']}) by identified users, then by "
                        "connections. Produced by the system's standard ranking, because the AI did not complete this "
                        "question."),
        "assumptions": [{"label": "Group", "value": f"{name}: {', '.join(c['labels'][:8])}{'…' if len(c['labels']) > 8 else ''}"},
                        {"label": "Measure", "value": "Identified users, then connections"}]
                       + ([{"label": "Area", "value": f"{city} (city level)"}] if city else []),
        "chart": {"type": "bar", "x": "app", "y": "users", "series": ""},
        "message": "",
    }
    if total:
        one = c["labels"][0] if len(c["labels"]) == 1 or name.lower() in c["labels"][0].lower() else ""
        what = one or f"{name} apps"
        plan.update({
            "understood": f"How many people used {what}{where}{when}?",
            "title": f"People using {what}{', ' + city if city else ''}",
            "explanation": (f"Counts the identified users of every app in the '{name}' group ({c['description']}) "
                            "together, each person once, with their connections and data volume. Produced by the "
                            "system's standard count, because the AI did not complete this question."),
            "chart": {"type": "table", "x": "", "y": "", "series": ""},
        })
        plan["assumptions"][1] = {"label": "Measure", "value": "Identified users (each counted once), connections, volume"}
    return Answer(plan=align_period_text(plan, sql), sql=checked, result=result, steps=steps, usage=usage, tested=True)


def labels_not_grouped(question: str, result) -> str | None:
    names = concepts.apps_in(question)
    if len(names) < 2 or result is None or not result.rows:
        return None
    cat = concepts.catalogue()
    for i, c in enumerate(result.columns):
        if c["name"].lower() not in ("app", "app_label", "application"):
            continue
        values = {str(r[i]) for r in result.rows}
        split = [n for n in names if len(values & set(cat[n]["labels"] if n in cat else concepts.app_names()[n])) > 1]
        if split:
            groups = ", ".join(
                f"'{n}' for app_label IN (" + ", ".join(f"'{x}'" for x in (cat[n]["labels"] if n in cat else concepts.app_names()[n])) + ")"
                for n in names)
            return (f"Not accepted yet: the question compares apps, but {', '.join(split)} "
                    f"{'is' if len(split) == 1 else 'are'} split over several rows, one per label. Show one row per app "
                    f"(and per city if the question names several): name each app with multiIf ({groups}) and group by "
                    "that name, so uniqIf counts each person once per app. Test, and submit again.")
    return None


def ranking_as_total(question: str, result) -> str | None:
    q = question.lower()
    if result is None or len(result.rows) != 1 or concepts.wants_total(q) or not concepts._RANK_INTENT.search(q):
        return None
    if _AVERAGE.search(q) or _TOTAL_WORDS.search(q):
        return None
    if any("String" in str(c.get("type", "")) for c in result.columns):
        return None
    return ("Not accepted yet: the question asks which is used the most, so it needs a ranking, but the result is one "
            "total. List the apps (or places, or hours) the question is about with their users, connections and "
            "volume_gb, ordered by the team's default, test, and submit again.")


_TOTAL_WORDS = re.compile(r"(?<![a-z])(total|overall|altogether|in all|kul|how much|kitna|kitni|how many|kitne|count|"
                          r"count of|number of|sum of|average|avg|mean|ratio|times more|kitne guna|guna)(?![a-z])")
_BREAKDOWN_WORDS = re.compile(
    r"(?<![a-z])(per(?!\s+(user|person|subscriber|head|customer|connection|session|people))|each(?!\s+(user|person|subscriber))|"
    r"every(?!\s+(user|person|subscriber))|wise|breakdown|split|which|top|most|least|rank|ranking|list|vs|versus|compare|kon|kaun|"
    r"konsi|kaunsi|trend|daily|hourly|by\s+(apps?|city|cities|isps?|operators?|networks?|days?|dates?|hours?|websites?|"
    r"sites?|domains?|categor(?:y|ies)|areas?|users?|subscribers?|countr(?:y|ies)|servers?|weekday))(?![a-z])")


def isp_split_in_ranking(result) -> str | None:
    if result is None or len(result.rows) < 2:
        return None
    for i, c in enumerate(result.columns):
        if "String" not in str(c.get("type", "String")):
            continue
        split = concepts.isp_brands_split([str(r[i]) for r in result.rows])
        if split:
            return (f"Not accepted yet: {', '.join(split)} appear{'s' if len(split) == 1 else ''} on several rows, one per "
                    "stored name, so the brand's total and rank are wrong. Group by brand instead: "
                    f"{concepts.isp_brand_sql()} AS isp ... GROUP BY isp. Test, and submit again.")
    return None


_FILTER_COLS = ("client_location", "client_asn", "app_label", "app_category", "requested_server", "subscriber_id",
                "server_location", "server_asn", "imei", "imsi", "client_ip", "server_ip", "mnc")
_ENTITY_COLUMN = {"city": "client_location", "region": "client_location", "place": "client_location",
                  "isp": "client_asn", "sim": "mnc", "website": "requested_server", "area": "loc", "road": "loc",
                  "subscriber": "subscriber_id", "server_country": "server_location", "server_org": "server_asn",
                  "imei": "imei", "imsi": "imsi", "ip": "client_ip", "category": "app_category"}
_CONTINUES = re.compile(r"^\s*(and|also|same|now|only|just|what about|how about|then|aur|sirf|ab)(?![a-z])")


def _where_filters(sql: str) -> list[tuple[str, list[str], str]]:
    where = re.split(r"(?i)\bWHERE\b", sql or "", maxsplit=1)
    if len(where) < 2:
        return []
    w = re.split(r"(?i)\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b", where[1])[0]
    out = []
    cols = "|".join(_FILTER_COLS)
    for m in re.finditer(rf"(?<![\w.])(NOT\s+)?({cols})\s*(=|IN)\s*(\([^)]*\)|'[^']*')", w, re.I):
        if m.group(1):
            continue
        vals = re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(4))
        if vals:
            out.append((m.group(2).lower(), vals, m.group(0)))
    for m in re.finditer(r"\bloc\s+I?LIKE\s+'%?([^'%]+)%?'", w, re.I):
        out.append(("loc", [m.group(1)], m.group(0)))
    for m in re.finditer(r"endsWith\(\s*server_location\s*,\s*'([^']+)'\s*\)", w, re.I):
        out.append(("server_location", [m.group(1)], m.group(0)))
    if re.search(r"block_type\s*(>|!=|<>)\s*0", w, re.I):
        out.append(("block_type", ["block_type"], "block_type > 0"))
    return out


_HOW_MANY_PEOPLE = re.compile(r"(?<![a-z])(how many|number of|count of|kitne|kitni)\s+(\w+\s+){0,3}?"
                              r"(users?|people|persons?|subscribers?|customers?|of them|of these|of those|numbers?|log)(?![a-z])")


def rows_counted_as_people(question: str, sql: str) -> str | None:
    q = (question or "").lower()
    if not _HOW_MANY_PEOPLE.search(q):
        return None
    select = re.split(r"(?i)\bFROM\b", sql or "", maxsplit=1)[0]
    if re.search(r"(?i)uniq\w*\(\s*subscriber_id|count\s*\(\s*DISTINCT", select) or re.search(r"(?i)\bFROM\s*\(", sql or ""):
        return None
    if not re.search(r"(?i)\bcount(?:If)?\s*\(", select):
        return None
    return ("Not accepted yet: the question asks how many PEOPLE, but count() / countIf() counts connections (rows). "
            "Count people (for a split, uniqExactIf(subscriber_id, <condition> AND position(subscriber_id, '_') = 0) per "
            "part, or GROUP BY the part) "
            "with uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0) AS users (or, for given subscribers, "
            "uniq(subscriber_id)), keep every filter, test, and submit again.")


def follow_up_period(question: str, turns: list[dict] | None):
    if not turns:
        return None
    q = (question or "").lower()
    own = {key for rx, key in _PERIOD_WORDS if rx.search(q)} or _TIME_REF.search(q) or any(rx.search(q) for rx in _WEEKDAYS)
    if own:
        return None
    if not (context.refers_back(question) or is_follow_up(question) or _CONTINUES.search(q)
            or (len(q.split()) <= 6 and not catalog.entities(question))):
        return None
    prev = next((m for m in reversed(turns) if m.get("role") == "assistant" and m.get("kind") == "result"
                 and (m.get("data") or {}).get("sql")), None)
    return sql_period(prev["data"]["sql"]) if prev else None


def follow_up_period_changed(question: str, sql: str, turns: list[dict] | None) -> str | None:
    want = follow_up_period(question, turns)
    got = sql_period(sql or "")
    if not want or not got or want == got:
        return None
    from datetime import timedelta
    s, e = want
    return (f"Not accepted yet: this follow-up names no period, so it keeps the earlier answer's period, {_fmt_period(s, e)}, "
            f"but the SQL uses {_fmt_period(*got)}. Use start_time >= toUnixTimestamp('{s.isoformat()} 00:00:00', "
            f"'{settings.timezone}') AND start_time < toUnixTimestamp('{(e + timedelta(days=1)).isoformat()} 00:00:00', "
            f"'{settings.timezone}'), test, and submit again.")


def placeholder_in_sql(sql: str) -> str | None:
    found = re.findall(r"'(<[^'<>]{2,60}>|\{[^'{}]{2,60}\}|\[[A-Za-z_ ]{2,40}\])'", sql or "")
    if not found:
        return None
    return ("Not accepted yet: the SQL holds placeholder text instead of real values: " + ", ".join(found)
            + ". Use the exact values from <resolved> or from the earlier answer's rows (or look them up), test, and "
              "submit again.")


def follow_up_dropped_filters(question: str, sql: str, turns: list[dict] | None) -> str | None:
    if not turns:
        return None
    prev = None
    for m in reversed(turns):
        if m.get("role") == "user":
            if prev is not None and not wants_profile(m.get("content", "")):
                break
            if prev is not None:
                prev = None
            continue
        if m.get("kind") == "result" and (m.get("data") or {}).get("sql") and prev is None:
            prev = m
    if prev is None:
        return None
    q = (question or "").lower()
    mine = _safe(catalog.entities, question) or []
    own_kinds = {_ENTITY_COLUMN.get(e.kind) for e in mine}
    if concepts.apps_in(q) or concepts.match(q):
        own_kinds.add("app_label")
    continues = (context.refers_back(question) or is_follow_up(question) or _CONTINUES.search(q)
                 or (len(q.split()) <= 6 and not own_kinds))
    if not continues:
        return None
    s = sql or ""
    profile = wants_profile(question)
    dropped, kept_old = [], []
    for col, vals, text in _where_filters(prev["data"]["sql"]):
        if profile and col != "subscriber_id":
            continue
        present = any(f"'{v}'" in s or (col == "loc" and v.lower() in s.lower()) for v in vals) or \
            (col == "block_type" and "block_type" in s)
        if col in own_kinds:
            replaces = re.search(r"(?<![a-z])(same|now|only|just|instead|rather|sirf)(?![a-z])", q)
            if present and col != "client_location" and replaces and not _ADDS_PLACE.search(q):
                kept_old.append(text)
            continue
        if not present:
            dropped.append(text)
    if not dropped and not kept_old:
        return None
    parts = []
    if dropped:
        parts.append("this follow-up continues the previous answer, but the SQL drops its filters: "
                     + "; ".join(dropped) + ". Keep them (they define who 'they' / 'them' / 'his' are)")
    if kept_old:
        parts.append("the follow-up replaces " + "; ".join(kept_old) + " with what it names, but the SQL still keeps "
                     "the old value")
    return ("Not accepted yet: " + ". Also, ".join(parts) + ". Fix the WHERE clause, keep the earlier period unless the "
            "question changes it, state the full question in the understood field, test, and submit again. If the "
            "question really starts afresh, submit the same SQL again.")


def unasked_app_filter(question: str, sql: str, turns: list[dict] | None = None) -> str | None:
    s = sql or ""
    said = " ".join([question or ""] + [m.get("content", "") for m in (turns or []) if m.get("role") == "user"])
    said_kinds = {c["key"] for c in concepts.match(said)}
    said_apps = set(concepts.apps_in(said))
    for m in (turns or []):
        for v in str(((m.get("data") or {}).get("sql") or "")).split("'"):
            said_apps |= {n for n, labs in concepts.app_names().items() if v in labs}
    cat = concepts.catalogue()
    where = re.split(r"(?i)\bWHERE\b", s, maxsplit=1)[-1]
    if re.search(r"(?i)app_label\s+NOT\s+IN", where) and not re.search(r"(?i)app_label\s+(IN|=)", where.replace("NOT IN", "")):
        return None
    extra = []
    covered = {lab for k in said_kinds | said_apps if k in cat for lab in cat[k]["labels"]}
    covered |= {lab for n in said_apps for lab in concepts.app_names().get(n, [])}
    for key, c in cat.items():
        if key in said_kinds or key in said_apps or len(c["labels"]) < 3 or c["includes"]:
            continue
        used = [lab for lab in c["labels"] if f"'{lab}'" in where and lab not in covered]
        own = {n for n, labs in concepts.app_names().items() if set(labs) & set(used)}
        if len(used) >= 3 and not (own & said_apps):
            extra.append(key.replace("_", " "))
    if not extra:
        return None
    return ("Not accepted yet: the SQL filters on " + ", ".join(extra) + " apps, but the question does not mention "
            + ("it" if len(extra) == 1 else "them") + " (a filter copied from an example?). Answer exactly the question "
            "asked: remove that filter, keep only what the question names, update the understood question, test, and "
            "submit again.")


_EXCLUDING = re.compile(r"(?<![a-z])(leaving out|leave out|leave|apart from|except( for)?|excluding|exclude|besides|other "
                        r"than|not counting|without|but not|ignoring|ignore)\s+(.+?)(?=[,?.;:]|$|\s(whose|which|who|what|how|"
                        r"where|when|top|in|on|for|by|from|last|yesterday)\s)")


def excluded_words(question: str) -> set[str]:
    out = set()
    for m in _EXCLUDING.finditer((question or "").lower()):
        out |= {w for w in re.split(r"[\s,&/]+|\band\b|\bor\b", m.group(3)) if w}
    return out


def app_ignored(question: str, sql: str) -> str | None:
    s = sql or ""
    left_out = excluded_words(question)
    names = [n for n in concepts.apps_in(question) if n not in left_out and n not in s.lower()]
    if not names:
        return None
    if re.search(r"app_label\s*(=|IN\b|I?LIKE\b)|match\(\s*app_label|startsWith\(\s*app_label", s, re.I):
        return None
    if re.search(r"(?<![a-z])(top|most|popular|rank|which apps?|kon si apps?)(?![a-z])", (question or "").lower()) \
            and re.search(r"GROUP\s+BY[^;]*app", s, re.I):
        return None
    cat = concepts.catalogue()
    calls = concepts.call_labels(question)
    lines = []
    for n in names:
        labels = calls.get(n) or (cat[n]["labels"] if n in cat else concepts.app_names()[n])
        lines.append(f"{n}: app_label IN (" + ", ".join(f"'{x}'" for x in labels) + ")")
    return ("Not accepted yet: the question is about " + ", ".join(names) + ", but the SQL has no filter on "
            + ("it" if len(names) == 1 else "them") + ", so it answers for all apps. Add " + "; ".join(lines)
            + " to the WHERE clause, answer exactly what is asked about it (a ratio is sum(a) / sum(b)), update the "
              "understood question, test, and submit again.")


def per_user_with_unidentified(question: str, sql: str) -> str | None:
    s = sql or ""
    if not re.search(r"/\s*(uniq\w*\(\s*subscriber_id|users\b)", s, re.I):
        return None
    if re.search(r"position\(\s*subscriber_id\s*,\s*'_'\s*\)\s*=\s*0", re.split(r"(?i)\bWHERE\b", s, maxsplit=1)[-1].split(
            "GROUP BY")[0], re.I) or re.search(r"sumIf\([^)]*position\(\s*subscriber_id", s, re.I):
        return None
    if not re.search(r"sum\(\s*(c2s_byte_num|s2c_byte_num|duration|\(?\s*c2s)", s, re.I):
        return None
    return ("Not accepted yet: the average divides ALL traffic (including the half with no identified subscriber) by "
            "identified users, so it is about twice too big. Per-user values use identified subscribers' traffic only: "
            "sumIf(c2s_byte_num + s2c_byte_num, position(subscriber_id, '_') = 0) / uniqExactIf(subscriber_id, "
            "position(subscriber_id, '_') = 0) (or add position(subscriber_id, '_') = 0 to the WHERE). Test, and "
            "submit again.")


_NAMED_PORTS = [(re.compile(r"(?<![a-z])(apple push|apns|ios push|iphone push)"), 5223, "Apple push"),
                (re.compile(r"(?<![a-z])(android push|google push|fcm|firebase cloud messaging|gcm)(?![a-z])"), 5228,
                 "Android push"),
                (re.compile(r"(?<![a-z])(dns over tls|dns-over-tls|dot dns)(?![a-z])"), 853, "DNS over TLS"),
                (re.compile(r"(?<![a-z])(imap over ssl|imaps|imap ssl)(?![a-z])"), 993, "IMAP over SSL")]
_PORT_NUMBER = re.compile(r"(?<![a-z])(source |client |server |destination )?port\s*(?:number\s*)?[:=]?\s*(\d{2,5})(?!\d)|"
                          r"(?<!\d)(\d{2,5})\s+port(?![a-z])")


def ports_in(question: str) -> list[tuple[str, int, str]]:
    q = (question or "").lower()
    out = []
    for m in _PORT_NUMBER.finditer(q):
        side, num = m.group(1), int(m.group(2) or m.group(3))
        out.append(("client_port" if side in ("source ", "client ") else "server_port", num, f"port {num}"))
    for rx, num, name in _NAMED_PORTS:
        if rx.search(q) and not any(p == num for _, p, _ in out):
            out.append(("server_port", num, name))
    return out


_LAT_WORDS = re.compile(r"(?<![a-z])(latitude|lat|north of|south of|northern|southern|northernmost|southernmost)(?![a-z])")
_LON_WORDS = re.compile(r"(?<![a-z])(longitude|lon|lng|east of|west of|eastern|western|easternmost|westernmost)(?![a-z])")
_OPEN_END = re.compile(r"(?<![a-z])(since|onwards|onward|so far|until now|till now|up to now|to date|all time|"
                       r"whole data|entire data|all data|all the data)(?![a-z])")


def field_checks(question: str, sql: str, done: set) -> tuple[str, str, str] | None:
    q = (question or "").lower()
    s = sql or ""
    low = s.lower()
    select = re.split(r"(?i)\bFROM\b", s, maxsplit=1)[0]
    where = re.split(r"(?i)\bGROUP\s+BY\b|\bORDER\s+BY\b", re.split(r"(?i)\bWHERE\b", s, maxsplit=1)[-1])[0]

    def first(key, step, text):
        return None if key in done else (key, step, text)

    if (re.search(r"/\s*uniq\w*\(\s*subscriber_id", select, re.I) and not _SHARE_WORDS.search(q)
            and not re.search(r"(?<![a-z])(average|avg|mean|per user|per person|per subscriber|per head|each user|"
                              r"typical|ausat|har user|fi user)(?![a-z])", q)
            and re.search(r"(?<![a-z])(total|in total|altogether|overall|in all|combined|sum)(?![a-z])", q)):
        if (r := first("per_user_unasked", "Returning the total, not a per-user value",
                       "Not accepted yet: the question asks for a TOTAL, but the SQL divides by the number of users, "
                       "which gives a per-user value. Return the total itself (for time: sum(duration) / 3.6e6 AS "
                       "hours over all rows; for data: sum(c2s_byte_num + s2c_byte_num) / 1e9 AS volume_gb), keep every "
                       "filter, test, and submit again.")):
            return r

    geo_part = re.findall(r"splitByChar\(\s*','\s*,\s*(?:any\w*\(\s*)?(?:client_|server_)?geo\s*\)?\s*\)\s*\[\s*([12])\s*\]",
                          s, re.I)
    if geo_part:
        lat_q, lon_q = bool(_LAT_WORDS.search(q)), bool(_LON_WORDS.search(q))
        if (lat_q and not lon_q and set(geo_part) == {"1"}) or (lon_q and not lat_q and set(geo_part) == {"2"}):
            if (r := first("geo_axis", "Reading latitude and longitude the right way round",
                           "Not accepted yet: coordinates are stored as 'lon,lat', so splitByChar(',', geo)[1] is the "
                           "LONGITUDE and splitByChar(',', geo)[2] is the LATITUDE. The SQL uses the wrong part for what "
                           "the question asks. Fix it, test, and submit again.")):
                return r

    if (re.search(r"(?<![a-z])ipv[46](?![a-z0-9])|(?<![a-z])ip ?v ?[46] ", q) and "server" not in q
            and not re.search(r"client_ip\s*(=|in\s*\()", low)
            and not any(e.kind == "ip" for e in catalog.entities(question))):
        if not re.search(r"client_ip\s*,\s*':'|client_ip\s+(not\s+)?i?like\s+'%:%'|isipv[46]string\(\s*client_ip"
                         r"|client_ip\s*,\s*'\.'", low):
            if (r := first("ip_version", "Testing the subscriber's IP version",
                           "Not accepted yet: IPv6 means the SUBSCRIBER's address client_ip is IPv6: position(client_ip, "
                           "':') > 0 (IPv4: position(client_ip, ':') = 0). server_ip is the far end and does not say "
                           "how the subscriber connected. Use that condition, test, and submit again.")):
                return r

    if re.search(r"(?<![a-z])(same|shared?|one|common)\s+(phone|handset|device|mobile|imei|set|mobile set)(?![a-z])", q) \
            and "imei" not in low:
        subs = [e.name for e in catalog.entities(question) if e.kind == "subscriber"]
        who = (f"subscriber_id IN ({', '.join(repr(f) for f in context.phone_forms(subs[0]))})" if subs and subs[0].isdigit()
               else f"subscriber_id = '{subs[0]}'" if subs else "<the person>")
        if (r := first("same_handset", "Matching the handset (IMEI)",
                       "Not accepted yet: 'the same phone' is the same handset, i.e. the same imei. Find the person's "
                       f"IMEIs and then everyone else on them: ... WHERE imei IN (SELECT DISTINCT imei FROM "
                       f"wms_dist.ntc_ssl_log WHERE <same period> AND {who} AND imei != '') AND NOT ({who}) GROUP BY "
                       "subscriber_id. Test, and submit again.")):
            return r

    if (re.search(r"start_time\s*>=?", where, re.I) and not re.search(r"start_time\s*<|start_time\s+BETWEEN", where, re.I)
            and not _OPEN_END.search(q)):
        if (r := first("open_period", "Closing the period",
                       "Not accepted yet: the WHERE clause has a start for the period (start_time >= ...) but no end, so "
                       "it runs to the end of the data. Add start_time < toUnixTimestamp('<day after the last day> "
                       "00:00:00', 'Asia/Karachi') for the period the question names. Test, and submit again.")):
            return r

    if _SHARE_WORDS.search(q) and re.search(r"(?<![a-z])(connections?|times|requests?|sessions?|flows?)(?![a-z])", q) \
            and not re.search(r"(?<![a-z])(data|volume|traffic|bytes|gb|mb|usage|bandwidth)(?![a-z])", q) \
            and re.search(r"byte_num|duration", select) and not re.search(r"(?i)count(If)?\s*\(", select):
        if (r := first("share_measure", "Taking the share of connections",
                       "Not accepted yet: the question asks for a share of CONNECTIONS, but the SQL divides data volume "
                       "or time. Use 100 * countIf(<the part>) / count() AS share_pct. Test, and submit again.")):
            return r

    if re.search(r"(?<![a-z])(upstream|upload(ed|s)?|uplink)(?![a-z])", q) and _SHARE_WORDS.search(q) \
            and re.search(r"(?<![a-z])(their|its|own)(?![a-z])", q) and re.search(r"(?i)\bOVER\s*\(", select):
        if (r := first("own_share", "Dividing by each group's own traffic",
                       "Not accepted yet: the question asks what share of EACH group's own traffic is upload, but the "
                       "SQL divides by the total of all groups (OVER ()). Use sum(c2s_byte_num) / sum(c2s_byte_num + "
                       "s2c_byte_num) per group, order by it, test, and submit again.")):
            return r

    for col, num, name in ports_in(question):
        if not re.search(rf"(?<!\d){num}(?!\d)", s):
            if (r := first(f"port_{num}", "Using the port the question names",
                           f"Not accepted yet: the question names {name}, but the SQL does not filter on it. Add "
                           f"{col} = {num} to the WHERE clause. The port defines the traffic by itself: do not also "
                           "limit it to app labels unless the question names an app. Keep the other filters (place, "
                           "period). Test, and submit again.")):
                return r

    if re.search(r"only (the )?province|province[- ]only|province is known|province level", q) and \
            re.search(r"client_location\s+NOT\s+IN\s*\([^)]*Punjab Pakistan", s, re.I):
        if (r := first("province_only", "Keeping the province-only records",
                       "Not accepted yet: records where only the province is known ARE the values 'Punjab Pakistan', "
                       "'Sindh Pakistan', 'Khyber Pakhtunkhwa Pakistan' (and 'Balochistan Pakistan' if present). Keep "
                       "them with client_location IN (...) instead of leaving them out. Test, and submit again.")):
            return r

    if re.search(r"(?<![a-z])(come|came|coming|comes|connect(ed|ing)?|connections?|users?|subscribers?|people) from(?![a-z])",
                 q) and "server" not in q and "server_location" in low and "client_location" not in low:
        if (r := first("from_place", "Using where the subscribers are",
                       "Not accepted yet: where connections come FROM is the subscriber's place, client_location "
                       "(foreign places: client_location != '' AND NOT endsWith(client_location, 'Pakistan')). "
                       "server_location is where the servers are. Use client_location, test, and submit again.")):
            return r

    if (re.search(r"(?<![a-z])(foreign|abroad|outside (of )?pakistan|other countries|overseas|international|"
                  r"not in pakistan)(?![a-z])", q) and "server" not in q and "client_location" in low
            and not re.search(r"not\s*\(?\s*endswith\(\s*client_location\s*,\s*'\s*pakistan'|client_location\s+not\s+i?like\s+"
                              r"'%pakistan'|not\s+match\(\s*client_location", low)):
        if (r := first("foreign_place", "Keeping only places outside Pakistan",
                       "Not accepted yet: the question is about places OUTSIDE Pakistan, but the SQL keeps Pakistani "
                       "cities. Add client_location != '' AND NOT endsWith(client_location, 'Pakistan') to the WHERE "
                       "clause (this leaves the foreign values such as 'Kabul Afghanistan', 'United States'). Test, and "
                       "submit again.")):
            return r

    if re.search(r"(?<![a-z])account (codes?|ids?|numbers?)(?![a-z])", q) and "match(subscriber_id" not in low.replace(" ", ""):
        if (r := first("account_codes", "Telling account codes from phone numbers",
                       "Not accepted yet: phone numbers are stored as 92 + 10 digits or 03 + 9 digits; every other "
                       "identified subscriber_id is an account code. Use position(subscriber_id, '_') = 0 AND NOT "
                       "match(subscriber_id, '^(92[0-9]{10}|03[0-9]{9})$') and count people with uniqExact(subscriber_id). "
                       "Test, and submit again.")):
            return r

    if re.search(r"(?<![a-z])categor(y|ies)(?![a-z])", q) and "app_category" not in low:
        cats = [e for e in catalog.entities(question) if e.kind == "category"]
        if cats:
            if (r := first("category", "Using the app category",
                           f"Not accepted yet: the question names the category {cats[0].name}; filter on the stored "
                           f"category itself: {cats[0].sql} (every app in it, none left out). Test, and submit again.")):
                return r
    return None


def total_as_list(question: str, result, turns: list[dict] | None = None) -> str | None:
    q = (question or "").lower()
    if result is None or len(result.rows) < 2 or not _TOTAL_WORDS.search(q) or _BREAKDOWN_WORDS.search(q):
        return None
    if wants_subscriber_list(question):
        return None
    refs = context.references(question, turns) if turns else []
    if refs and len(refs[0]["values"]) > 1:
        return None
    if _AVERAGE.search(q):
        return ("Not accepted yet: the question asks for one average, but the result is a list of "
                f"{len(result.rows)} rows. Return one row: {average_formula(question)} (keep every filter). Test, and "
                "submit again.")
    return ("Not accepted yet: the question asks for one total, but the result is a list of "
            f"{len(result.rows)} rows. Return one row with the total (keep every filter). To count people who meet a "
            "condition (used both apps, more than N connections), keep the per-subscriber query as a subquery and "
            "count its rows: SELECT count() AS users FROM (SELECT subscriber_id FROM ... GROUP BY subscriber_id "
            "HAVING ...). Test, and submit again.")


_AVERAGE = re.compile(r"(?<![a-z])(average|avg|the mean|mean value|mean of|ausat|per user|per person|per subscriber|"
                      r"per head|each user|do people spend|does a user spend|do users spend|people spend|users spend|"
                      r"typical user|a user spends?)(?![a-z])")


def average_formula(question: str) -> str:
    q = (question or "").lower()
    users = "uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0)"
    if re.search(r"(?<![a-z])(time|duration|watch|hours|minutes|kitni der|spend|spent)(?![a-z])", q):
        what, unit = "sumIf(duration, position(subscriber_id, '_') = 0)", (" / 6e4 AS minutes_per_user" if "minute" in q
                                                                           else " / 3.6e6 AS hours_per_user")
    elif re.search(r"(?<![a-z])(upload|uploads|uplink)(?![a-z])", q):
        what, unit = "sumIf(c2s_byte_num, position(subscriber_id, '_') = 0)", " / 1e6 AS upload_mb_per_user"
    elif re.search(r"(?<![a-z])(download|downloads|downlink)(?![a-z])", q):
        what, unit = "sumIf(s2c_byte_num, position(subscriber_id, '_') = 0)", " / 1e6 AS download_mb_per_user"
    elif re.search(r"(?<![a-z])(connections?|sessions?|times|visits?)(?![a-z])", q):
        what, unit = "countIf(position(subscriber_id, '_') = 0)", " AS connections_per_user"
    else:
        what, unit = "sumIf(c2s_byte_num + s2c_byte_num, position(subscriber_id, '_') = 0)", " / 1e6 AS mb_per_user"
    per_day = " / <days in the period>" if re.search(r"per day|daily|a day|roz", q) else ""
    return f"round({what} / {users}{per_day}{unit.split(' AS ')[0]}, 2) AS {unit.split(' AS ')[1]}"


def average_missing(question: str, sql: str) -> str | None:
    q = (question or "").lower()
    select = re.split(r"(?i)\bFROM\b", sql or "", maxsplit=1)[0]
    items = _split_top(re.sub(r"(?is)^\s*SELECT\s+(DISTINCT\s+)?", "", select))
    for item in items:
        m = re.search(r"(?is)^(.*)\s+AS\s+(\w*per_(?:user|person|subscriber|head)\w*)\s*$", item.strip())
        if m and not re.search(r"(?i)/\s*(uniq|users|count|\(?\s*SELECT)|/\s*\w+_users\b|\bavg\s*\(", m.group(1)):
            return (f"Not accepted yet: the column {m.group(2)} is named per user, but it is a total: nothing is divided "
                    f"by the number of users. Return {average_formula(question)} (identified subscribers' total divided "
                    "by identified users; keep every filter). Test, and submit again.")
    if not _AVERAGE.search(q) or wants_subscriber_list(question):
        return None
    if re.search(r"(?i)\bavg\s*\(|/\s*(uniq|users|count|\(?\s*SELECT)|/\s*\w+_users\b", select):
        return None
    return ("Not accepted yet: the question asks for an average per user, but the SQL only returns totals, so nothing "
            f"is divided. Return {average_formula(question)} (identified subscribers' traffic divided by identified "
            "users; keep every filter). Test, and submit again.")


_SHARE_WORDS = re.compile(r"(?<![a-z])(share|percent|percentage|proportion|fraction|portion|hissa|hisa|contribution)(?![a-z])|%")


def share_of_itself(question: str, sql: str, result) -> str | None:
    if result is None or not result.rows or not _SHARE_WORDS.search((question or "").lower()):
        return None
    nums = [v for r in result.rows for v in r if isinstance(v, (int, float))]
    if len(result.rows) == 1 and any(abs(v - 100) < 1e-9 for v in nums):
        return ("Not accepted yet: the share is 100% because the WHERE clause keeps only the thing being measured, so "
                "it is divided by itself. Keep the WHERE to the whole (period, place) and put the part in the numerator: "
                "100 * sumIf(c2s_byte_num + s2c_byte_num, <the part's filter>) / sum(c2s_byte_num + s2c_byte_num) AS "
                "share_pct (or countIf(...) / count() for connections). Test, and submit again.")
    return None


_PER_PERSON = re.compile(r"(?<![a-z])(per user|per person|per subscriber|each user|per head|average user|avg user|"
                         r"per customer|fi user|har user|do people spend|does a user spend|do users spend|people spend|"
                         r"time spent per|average|avg)(?![a-z])")


def identified_filter_on_traffic(question: str, sql: str) -> str | None:
    s = sql or ""
    where = re.split(r"(?i)\bGROUP\s+BY\b|\bORDER\s+BY\b", re.split(r"(?i)\bWHERE\b", s, maxsplit=1)[-1])[0]
    if not re.search(r"position\(\s*subscriber_id\s*,\s*'_'\s*\)\s*=\s*0", where, re.I):
        return None
    q = (question or "").lower()
    per_person = _PER_PERSON.search(q) and not (
        re.search(r"(?<![a-z])(total|in total|altogether|overall|in all|combined)(?![a-z])", q)
        and not re.search(r"(?<![a-z])(average|avg|per user|per person|per subscriber|each user|per head)(?![a-z])", q))
    if (_GROUPS_BY_SUB.search(s) or re.search(r"subscriber_id\s*(IN\s*\(|=\s*')", s, re.I)
            or wants_subscriber_list(question) or wants_profile(question) or per_person):
        return None
    select = re.split(r"(?i)\bFROM\b", s, maxsplit=1)[0]
    if not re.search(r"byte_num|count\(|duration", select, re.I):
        return None
    return ("Not accepted yet: the WHERE clause has position(subscriber_id, '_') = 0, which drops about half of all traffic "
            "(records with no identified subscriber). Volume and connections of apps, places, ISPs, sites or times include "
            "every record; only the users measure leaves unidentified records out, and uniqIf already does that. Remove "
            "that condition from the WHERE clause, test, and submit again.")


def uses_empty_column(sql: str) -> str | None:
    empty = (dictionary.base().get("empty_columns") or {}).get("list", [])
    s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql or "")
    aliases = {a.lower() for a in re.findall(r"(?i)\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", s)}
    used = [c for c in empty if c.lower() not in aliases
            and re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", s)]
    if not used:
        return None
    return (f"Not accepted: {', '.join(used)} {'is' if len(used) == 1 else 'are'} empty in every record of this table, so "
            "an answer that uses it is false (it can only give zero or nothing). If the question needs it, submit "
            "cannot_answer and say plainly that the data has no such information, and what can be answered instead. "
            "Otherwise remove it from the SQL, test, and submit again.")


_MEASURE_WORDS = re.compile(r"(?<![a-z])(users?|people|ppl|log|subscribers?|data|traffic|usage|volume|connections?|"
                            r"sessions?|duration|kitne|kitna|kitni)(?![a-z])")
_APP_WORDS = re.compile(r"(?<![a-z])(apps?|applications?|platforms?|services?|sites?|websites?|category|categories|"
                        r"labels?|which|kon|kaun|konsi|kaunsi|breakdown|split|each)(?![a-z])")


_APP_RANK = re.compile(r"(?<![a-z])(which|top|most|popular|rank|ranking|best|leading|highest|biggest|zyada|ziada|"
                       r"kon|kaun|konsi|kaunsi)(?![a-z])")


def unasked_app_breakdown(question: str, result) -> str | None:
    q = question.lower()
    if result is None or len(result.rows) < 2 or not _MEASURE_WORDS.search(q):
        return None
    if _APP_WORDS.search(q) or _APP_RANK.search(q) or concepts.apps_in(q):
        return None
    if not any(c["name"].lower() in ("app", "app_label", "application") for c in result.columns):
        return None
    return ("Not accepted yet: the question does not ask about apps, but the result is split by app. Answer at the "
            "level the question asks: one total for the place and period (or one row per day, hour or city when the "
            "question asks for that), with users, connections and volume_gb. Test, and submit again.")


_UNKNOWN_CITY = {"pakistan", "punjab pakistan", "punjab", "sindh pakistan", "sindh", "khyber pakhtunkhwa pakistan",
                 "khyber pakhtunkhwa", "balochistan pakistan", "balochistan", ""}


def unknown_city_in_ranking(result) -> str | None:
    if result is None or not result.rows:
        return None
    for i, c in enumerate(result.columns):
        if c["name"].lower() not in ("city", "client_location", "location", "area"):
            continue
        found = sorted({str(r[i]) for r in result.rows if str(r[i]).strip().lower() in _UNKNOWN_CITY})
        if found:
            return (f"Not accepted yet: the rows include {', '.join(repr(v) for v in found)}, which mean the city is "
                    "unknown, not a city. Leave them out with client_location NOT IN ('Pakistan', 'Punjab Pakistan', "
                    "'Sindh Pakistan', 'Khyber Pakhtunkhwa Pakistan', ''), "
                    "test, and submit again.")
    return None


def empty_group_in_ranking(result, sql: str = "") -> str | None:
    if result is None or len(result.rows) < 2:
        return None
    if _GROUPS_BY_SUB.search(sql or "") or re.search(r"subscriber_id\s*(IN\s*\(|=\s*')", sql or "", re.I):
        return None
    for i, c in enumerate(result.columns):
        if "String" not in str(c.get("type", "String")) or c["name"].lower() in ("subscriber_id", "imei", "imsi"):
            continue
        if any(r[i] == "" for r in result.rows):
            return (f"Not accepted yet: the result has a row whose {c['name']} is empty (''): those are records with no "
                    f"value, not a {c['name']}. Leave them out with <column> != '' in the WHERE clause (for websites "
                    "requested_server != ''), test, and submit again.")
        break
    return None


def fix_brand_values(sql: str) -> str | None:
    s = sql or ""
    if "client_asn" not in s:
        return None
    brands = {c["name"].lower(): c["asn"] for c in concepts.isp_catalogue().values() if c["asn"]}
    stored = {a.lower() for asns in brands.values() for a in asns}

    def expand(m):
        body = m.group(2)
        items = re.findall(r"'((?:[^'\\]|\\.)*)'", body)
        out, changed = [], False
        for it in items:
            low = it.lower()
            if low in brands and low not in stored:
                out += brands[low]
                changed = True
            else:
                out.append(it)
        if not changed:
            return m.group(0)
        vals = ", ".join("'" + v.replace("'", "\\'") + "'" for v in dict.fromkeys(out))
        return f"{m.group(1)}IN ({vals})"

    new = re.sub(r"(client_asn\s+(?:NOT\s+)?)IN\s*\(([^()]*)\)", expand, s, flags=re.I)
    new = re.sub(r"(client_asn\s*(?:!=|=)\s*)'([^']+)'", lambda m: (
        f"client_asn {'NOT ' if '!' in m.group(1) else ''}IN (" + ", ".join(f"'{v}'" for v in brands[m.group(2).lower()]) + ")"
        if m.group(2).lower() in brands and m.group(2).lower() not in stored else m.group(0)), new, flags=re.I)
    return new if new != s else None


def fix_isp_groups(sql: str) -> str | None:
    brand = concepts.isp_brand_sql()
    return None if brand == "client_asn" else _regroup(sql, "client_asn", brand, "isp")


def _regroup(sql: str, column: str, expr: str, default_alias: str) -> str | None:
    s = sql or ""
    if len(re.findall(r"(?i)\bSELECT\b", s)) != 1 or re.search(r"(?i)\bOVER\s*\(", s):
        return None
    m = re.search(r"(?is)\bGROUP\s+BY\b(.*?)(?=\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|\bSETTINGS\b|;|$)", s)
    parts = re.split(r"(?i)\bFROM\b", s, maxsplit=1)
    if not m or len(parts) != 2 or "subscriber_id" in m.group(1):
        return None
    head, rest = parts
    alias_m = re.search(rf"(?i)(?<![\w.]){column}\s+AS\s+(\w+)", head)
    grouped = re.search(rf"(?<![\w.]){column}(?![\w])", m.group(1)) or (
        alias_m and re.search(rf"(?<![\w.]){re.escape(alias_m.group(1))}(?![\w])", m.group(1)))
    if not grouped:
        return None
    if alias_m:
        alias = alias_m.group(1)
        if alias.lower() == column:
            return None
        head = head[:alias_m.start()] + f"{expr} AS {alias}" + head[alias_m.end():]
    else:
        bare = re.search(rf"(?i)(?<![\w.'(]){column}(?=\s*,|\s*$)", head)
        if not bare:
            return None
        alias = default_alias
        head = head[:bare.start()] + f"{expr} AS {alias}" + head[bare.end():]

    def regroup(mm):
        return re.sub(rf"(?<![\w.']){column}(?![\w'])", alias, mm.group(0))
    rest = re.sub(r"(?is)\bGROUP\s+BY\b.*?(?=\bHAVING\b|\bLIMIT\b|\bSETTINGS\b|;|$)", regroup, rest)
    return head + "FROM" + rest


def fix_app_groups(sql: str, question: str = "") -> str | None:
    q = (question or "").lower()
    if re.search(r"(?<![a-z])(labels?|types?|kinds?|parts?|breakdown of|split of|components?)(?![a-z])", q):
        return None
    families = concepts.app_families()
    expr = concepts.app_family_sql()
    if expr == "app_label":
        return None
    where = re.split(r"(?i)\bGROUP\s+BY\b", re.split(r"(?i)\bWHERE\b", sql or "", maxsplit=1)[-1])[0]
    for m in re.finditer(r"(?i)\bapp_label\s+IN\s*\(([^()]*)\)", where):
        vals = set(re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1)))
        if vals and any(vals <= set(labs) for labs in families.values()):
            return None
    return _regroup(sql, "app_label", expr, "app")


_RAW_USER_COUNT = re.compile(r"\b(?:count\s*\(\s*DISTINCT\s+subscriber_id\s*\)|uniq(?:Exact|Combined|HLL12)?\s*\(\s*subscriber_id\s*\))",
                             re.I)


def fix_user_count(sql: str) -> str | None:
    orig = sql or ""
    s = re.sub(r"(?i)\b(subscriber_id)\s+NOT\s+I?LIKE\s+'%_%'", r"position(\1, '_') = 0", orig)
    s = re.sub(r"(?i)\b(subscriber_id)\s+I?LIKE\s+'%_%'", r"position(\1, '_') > 0", s)
    s = re.sub(r"\buniq(?:Combined|HLL12|Theta)?If\s*\(\s*subscriber_id\s*,", "uniqExactIf(subscriber_id,", s)
    s = re.sub(r"(?i)\bcount\s*\(\s*subscriber_id\s*\)(?!\s+AS\s+\w*(connection|hit|record|row|time|visit|session|request))",
               "uniqExact(subscriber_id)", s)
    if not _RAW_USER_COUNT.search(s):
        return s if s != orig else None
    if re.search(r"position\(\s*subscriber_id\s*,\s*'_'\s*\)\s*=\s*0|subscriber_id\s*(IN\s*\(|=\s*')", s, re.I):
        s = re.sub(r"\buniq(?:Combined|HLL12|Theta)?\s*\(\s*subscriber_id\s*\)", "uniqExact(subscriber_id)", s)
        return s if s != orig else None
    return _RAW_USER_COUNT.sub("uniqExactIf(subscriber_id, position(subscriber_id, '_') = 0)", s)


def fix_limit_one(sql: str, question: str = "") -> str | None:
    s = sql or ""
    if re.search(r"(?<![a-z])(top 1|top one|only one|just one|single|sirf ek|only the top)(?![a-z0-9])",
                 (question or "").lower()):
        return None
    if not re.search(r"(?i)\bLIMIT\s+1\s*;?\s*$", s) or not re.search(r"(?i)\bGROUP\s+BY\b", s):
        return None
    tail = re.split(r"(?i)\bORDER\s+BY\b", s)[-1] if re.search(r"(?i)\bORDER\s+BY\b", s) else ""
    if not re.search(r"(?i)\bDESC\b", tail):
        return None
    return re.sub(r"(?i)\bLIMIT\s+1(\s*;?\s*)$", r"LIMIT 10\1", s.rstrip())


def note_partial_days(plan: dict, result, data_range: dict) -> dict:
    if result is None or not result.rows:
        return plan
    first, last = (data_range.get("first") or ""), (data_range.get("last") or "")
    partial = []
    if first[11:16] not in ("", "00:00"):
        partial.append((first[:10], f"starts at {first[11:16]}"))
    if last[:10] and last[:10] != data_range.get("last_full_day"):
        partial.append((last[:10], f"ends at {last[11:16]}"))
    days = {str(v)[:10] for c_i, c in enumerate(result.columns) if "Date" in str(c.get("type", ""))
            for v in (r[c_i] for r in result.rows)}
    hit = [f"{d} (data {why})" for d, why in partial if d in days]
    if hit:
        note = f"Partly recorded day{'s' if len(hit) > 1 else ''}: {', '.join(hit)}; lower numbers there are not a real drop."
        expl = plan.get("explanation") or ""
        if "Partly recorded" not in expl:
            plan = {**plan, "explanation": f"{expl} {note}".strip()}
    return plan


_ADDS_PLACE = re.compile(r"(?<![a-z])(bhi|dono|both|also|too|together|plus|compare|comparison|vs|versus|sath|saath)(?![a-z])")


def follow_up_kept_old_place(question: str, sql: str, turns: list[dict]) -> str | None:
    if not turns or not is_follow_up(question) or _ADDS_PLACE.search(question.lower()):
        return None
    new = set(concepts.cities_in(question))
    if not new:
        return None
    earlier = set()
    for m in turns:
        if m.get("role") == "user":
            earlier |= set(concepts.cities_in(m.get("content", "")))
    s = sql or ""
    kept = [c for c in sorted(earlier - new)
            if f"'{c} Pakistan'" in s or re.search(rf"loc\s+I?LIKE\s+'%{c}%'", s, re.I)]
    if not kept:
        return None
    return (f"Not accepted yet: this follow-up asks the same question for {', '.join(sorted(new))} instead of "
            f"{', '.join(kept)}, but the SQL still includes {', '.join(kept)}, so the numbers mix the cities. Remove "
            f"{', '.join(kept)} from the filter and keep everything else (measure, period, other filters); update the "
            "understood question and title. Test, and submit again. If the question really asks for both, submit the "
            "same SQL again.")


def unasked_place(question: str, sql: str, turns: list[dict]) -> str | None:
    s = sql or ""
    filtered = [c for c in concepts.CITIES if f"'{c} Pakistan'" in s or re.search(rf"loc\s+I?LIKE\s+'%{c}%'", s, re.I)]
    if not filtered:
        return None
    said = " ".join([question] + [m.get("content", "") for m in turns or [] if m.get("role") == "user"])
    areas = [a for a in re.findall(r"loc\s+I?LIKE\s+'%([^%']+)%'", s, re.I)
             if a.strip().lower() not in {c.lower() for c in concepts.CITIES} and a.strip().lower() in said.lower()]
    if areas:
        return None
    named = set(concepts.cities_in(said))
    extra = [c for c in filtered if c not in named]
    if not extra:
        return None
    where = (f"the question is about {', '.join(sorted(named))} only" if named
             else "the question names no city, so it is about all of Pakistan")
    return (f"Not accepted yet: the SQL filters on {', '.join(extra)}, but {where}. Remove "
            f"{'that city' if len(extra) == 1 else 'those cities'} from the WHERE clause (and from the understood "
            "question, title and assumptions). Keep every place the question does name, such as a neighbourhood. "
            "Test, and submit again.")


def place_mismatch(question: str, sql: str) -> str | None:
    city = concepts.city_in(question)
    s = (sql or "").lower().replace("asia/karachi", "")
    if not city or city.lower() in s:
        return None
    return (f"Not accepted yet: the question is about {city}, but the SQL does not filter on it. For a follow-up, keep "
            f"everything else from the previous query and change only the place. Use client_location = '{city} Pakistan' "
            f"(city level) or loc ILIKE '%{city}%' (cell-site addresses); update the understood question, title and assumptions to {city} too; test, and submit again.")


_RANKING = re.compile(r"\b(top|most|popular|highest|biggest|zyada|ziada|sab se|sb se|leading|best)\b")


def generic_in_ranking(question: str, sql: str, result) -> str | None:
    if not _RANKING.search(question.lower()) or result is None or not result.rows:
        return None
    generic = set(dictionary.base().get("value_lists", {}).get("generic_labels", {}).get("list", []))
    names = {c["name"] for c in result.columns}
    if not generic or not ({"app", "app_label"} & names):
        return None
    i = next(k for k, c in enumerate(result.columns) if c["name"] in ("app", "app_label"))
    found = [r[i] for r in result.rows[:5] if r[i] in generic]
    if not found or re.search(r"NOT\s+IN", sql or "", re.I) and all(f"'{g}'" in sql for g in found):
        return None
    for c in concepts.match(question)[:1]:
        if c["hosts"] and "requested_server" in (sql or ""):
            name = c["key"].replace("_", " ")
            return (f"Not accepted yet: the rows named {', '.join(found)} are {name} sites found by requested_server "
                    f"that the classifier left unlabelled. Do not exclude them. Name them in the SELECT with "
                    f"if(app_label IN {concepts.sql_list(c['key'])}, app_label, 'Other {name} sites') AS app, group by "
                    "that, test, and submit again.")
    return (f"Not accepted yet: the top rows include background traffic ({', '.join(found)}), which is not an app people "
            "choose to use. Exclude the generic labels listed in the data dictionary with app_label NOT IN (...), test, "
            "and submit again. If the question is really about all traffic, submit the same answer again.")


_ASKS_OTHER = re.compile(r"\b(other(?!\s+than)|others|rest|remaining|else|baqi|baki|digar|share|percent|percentage|%)")


_PEOPLE_NEGATION = re.compile(
    r"(?<![a-z])(did ?n[o']t|do ?n[o']t|does ?n[o']t|not|never|without|haven'?t|hasn'?t|no longer|nahi|nhi|na|bina)\s+"
    r"(\w+\s+){0,3}?(use|used|using|open|opened|visit|visited|access|accessed|chala|chalai|chalate|istemal|kiya)(?![a-z])"
    r"|(?<![a-z])(non[- ]?users?|never used|did not use any)(?![a-z])")


def negation_as_row_filter(question: str, sql: str) -> str | None:
    q = (question or "").lower()
    s = sql or ""
    if not _PEOPLE_NEGATION.search(q):
        return None
    if not re.search(r"(app_label|requested_server|app_category|client_asn)\s+NOT\s+(IN|LIKE|ILIKE)|!=\s*'", s, re.I):
        return None
    if re.search(r"HAVING[^;]*(countIf|sumIf)\([^;]*=\s*0", s, re.I) or re.search(r"NOT\s+IN\s*\(\s*SELECT", s, re.I):
        return None
    return ("Not accepted yet: 'did not use X' is about people, so it cannot be a row filter: app_label NOT IN (...) keeps "
            "every person who used anything else. Decide it per subscriber: SELECT count() AS users FROM (SELECT "
            "subscriber_id FROM ... WHERE <period, place> AND position(subscriber_id, '_') = 0 GROUP BY subscriber_id "
            "HAVING countIf(<X's filter>) = 0) (for a list, select subscriber_id from that subquery instead). Test, and "
            "submit again.")


def extra_other_bucket(question: str, result) -> str | None:
    if result is None or not result.rows or _ASKS_OTHER.search(question.lower()):
        return None
    text_cols = [i for i, c in enumerate(result.columns) if "String" in str(c.get("type", "String"))]
    if not any(isinstance(r[i], str) and r[i].strip().lower() in ("other", "others", "rest")
               for r in result.rows for i in text_cols):
        return None
    return ("Not accepted yet: the result has an 'Other' group the question did not ask for. Filter the WHERE clause to "
            "only the apps (or places) the question names, so each row is one of them, test, and submit again. If "
            "the 'Other' group is really needed, submit the same answer again.")


_TECH_CLARIFY = re.compile(r"(?<![a-z])(metric|column|sql|query|syntax|identifier|function|error|failed|not directly available|uniqif|subscriber_id|app_label|client_location|unidentified|identified|records)(?![a-z])")


def app_labels_incomplete(question: str, sql: str) -> str | None:
    s = sql or ""
    cat = concepts.catalogue()
    gaps = []
    calls = concepts.call_labels(question)
    for name in concepts.apps_in(question):
        if name in calls:
            continue
        labels = cat[name]["labels"] if name in cat else concepts.app_names().get(name, [])
        used = [lab for lab in labels if f"'{lab}'" in s]
        if not used or len(used) == len(labels):
            continue
        if re.search(rf"app_label\s+(I?LIKE)\s+'{re.escape(labels[0].split('_')[0])}", s, re.I):
            continue
        gaps.append((name, [lab for lab in labels if lab not in used]))
    if not gaps:
        return None
    return ("Not accepted yet: " + "; ".join(f"{n} also has the labels {', '.join(repr(x) for x in miss)}" for n, miss in gaps)
            + ". An app means all of its labels: add them to the app filter (and to the app's group in multiIf), test, "
              "and submit again.")


def named_app_missing(question: str, sql: str) -> str | None:
    s = (sql or "").lower()
    if not re.search(r"app_label\s*(in\b|=|like\b|ilike\b)", s):
        return None
    missing = [name for name in concepts.apps_in(question) if name not in s]
    if not missing:
        return None
    labels = "; ".join(f"{name}: " + ", ".join(f"'{lab}'" for lab in concepts.app_names()[name]) for name in missing)
    return (f"Not accepted yet: the question names {', '.join(missing)}, but the SQL leaves {'it' if len(missing) == 1 else 'them'} "
            f"out. Their app_labels: {labels}. Include every app the question names (one row per app, grouping an "
            "app's labels under one name with multiIf), test, and submit again.")


_TOGETHER = re.compile(r"(?<![a-z])(combined|together|total|overall|kul|mila ?kar|sum)(?![a-z])")


def cities_merged(question: str, sql: str) -> str | None:
    cities = concepts.cities_in(question)
    s = (sql or "").lower()
    if len(cities) < 2 or _TOGETHER.search(question.lower()) or not all(c.lower() in s for c in cities):
        return None
    select_part = s.split(" from ", 1)[0] if " from " in s else s.split("\nfrom", 1)[0]
    if "client_location" in select_part or "city" in select_part:
        return None
    return (f"Not accepted yet: the question asks about {' and '.join(cities)} separately, but the SQL adds them "
            "together. Add the city as a column (replaceOne(client_location, ' Pakistan', '') AS city) and group by "
            "it, so each city has its own rows, test, and submit again.")


_PERIOD_CLARIFY = re.compile(r"(?<![a-z])(time period|period|time range|date range|timeframe|time frame|which dates|"
                             r"which days|what dates|how many days|which month|which week)(?![a-z])")


def _lazy_clarify(question: str, plan: dict) -> bool:
    msg = (plan.get("message") or "").lower()
    if "no values of" in msg or "try a shorter" in msg or "different spelling" in msg:
        return True
    if _TECH_CLARIFY.search(msg):
        return True
    if "default" in (plan.get("understood") or "").lower():
        return True
    if _PERIOD_CLARIFY.search(msg):
        return True
    if _is_vague(question):
        return True
    places = [e for e in (_safe(catalog.entities, question) or [])
              if e.kind in ("city", "region", "area", "road", "server_country")]
    if not places and re.search(r"(?<![a-z])(city|cities|place|location|area|region|province|karachi|lahore|islamabad)"
                                r"(?![a-z])", msg):
        return True
    city = concepts.city_in(question)
    return bool(city) and city.lower() not in msg


def concept_incomplete(question: str, sql: str) -> str | None:
    if concepts.call_labels(question):
        return None
    cs = concepts.match(question)
    if not cs:
        return None
    c = cs[0]
    left_out = excluded_words(question)
    if left_out & set(c["key"].split("_")) or any(syn in left_out for syn in c["synonyms"]):
        return None
    if len(c["labels"]) < 2 and not c["hosts"]:
        return None
    name = c["key"].replace("_", " ")
    used = [lab for lab in c["labels"] if f"'{lab}'" in (sql or "")]
    if not used:
        if _LABEL_PATTERN.search(sql or ""):
            return None
        return (f"Not accepted yet: the question is about {name} apps ({c['description']}), but the SQL does not use "
                f"that group. A word in the question is not a value to search for in other columns. Filter with "
                f"app_label IN {concepts.sql_list(c['key'])}, test, and submit again. If the question really means "
                "something else, submit the same answer again and say why in the explanation.")
    if len(used) == len(c["labels"]):
        if c["hosts"] and "requested_server" not in (sql or ""):
            return (f"Check: {name} also has sites the classifier does not label (they appear as SSL or other apps). "
                    f"Include them: add OR match(requested_server, '{c['hosts']}') to the app filter and name them in "
                    f"the SELECT with if(app_label IN {concepts.sql_list(c['key'])}, app_label, 'Other {name} sites'). "
                    "Test, and submit again.")
        return None
    missing = [lab for lab in c["labels"] if lab not in used]
    shown = ", ".join(missing[:30]) + (f" and {len(missing) - 30} more" if len(missing) > 30 else "")
    return (f"Check: the question is about {name}, which has {len(c['labels'])} app_labels, but "
            f"the SQL uses {len(used)}. Missing: {shown}. Add them unless leaving them out is right for "
            "this question; if it is, submit the same answer again (or confirm it).")


_LABEL_PATTERN = re.compile(r"app_label\s+(NOT\s+)?(I?LIKE)\s|match\(\s*app_label|startsWith\(\s*app_label", re.I)


_WEEKDAYS = [re.compile(rf"(?<![a-z])({w})(?![a-z])") for w in (
    "monday|mon|pir|somwar|somvar", "tuesday|tue|tues|mangal|mangalwar", "wednesday|wed|budh|budhwar",
    "thursday|thu|thurs|jumerat|jumeraat", "friday|fri|jumma|juma|jummah", "saturday|sat|sanichar",
    "sunday|sun|itwar|itwaar|itvar")]


def period_outside_data(sql: str, data_range: dict) -> str | None:
    found = sql_period(sql or "")
    if not found:
        return None
    start, end = found
    try:
        first = data_range["first"][:10]
        last = data_range["last"][:10]
    except (KeyError, TypeError):
        return None
    if start.isoformat() > last or end.isoformat() < first:
        return (f"Not accepted yet: the SQL's dates ({start.isoformat()} to {end.isoformat()}) are outside the data, which "
                f"runs from {first} to {last} (last full day {data_range.get('last_full_day')}). Relative days such as "
                "'last Saturday' or 'yesterday' count back from the last full day of the data, not from today's "
                "calendar date. Use the period from <resolved>, test, and submit again.")
    return None


def question_period(question: str, data_range: dict):
    q = question.lower()
    for rx in (catalog._IP_PORT, catalog._IPV4, catalog._IPV6, catalog._DOMAIN, catalog._FIFTEEN, catalog._ACCOUNT):
        q = rx.sub(" ", q)
    q = context._PHONE.sub(" ", q)
    days = [i for i, rx in enumerate(_WEEKDAYS) if rx.search(q)]
    if len(days) == 1 and not re.search(r"\d{1,2}\s*(st|nd|rd|th)?\s*(nov|oct)|(nov|oct)\w*\s*\d", q):
        from datetime import date, timedelta
        try:
            last = date.fromisoformat(data_range["last_full_day"])
        except (KeyError, ValueError, TypeError):
            return None
        back = (last.weekday() - days[0]) % 7
        if re.search(r"(?<![a-z])(before last|pichle se pichle)(?![a-z])", q):
            back += 7
        d = last - timedelta(days=back)
        return f"{d.strftime('%A')} (the most recent one in the data)", d, d
    keys = {key for rx, key in _PERIOD_WORDS if rx.search(q)}
    if "day_before" in keys:
        keys.discard("day")
    default = False
    if not keys and not _TIME_REF.search(q) and not is_follow_up(question) and not context.refers_back(question):
        keys, default = {"7d"}, True
    if len(keys) != 1:
        return None
    ps = context.periods(data_range)
    key = keys.pop()
    if key not in ps:
        return None
    label, start, end = ps[key]
    if default:
        label = "no period, so the default: the last 7 full days"
    return label, start, end


_CHANGE_DATA = re.compile(r"(?<![a-z])(delete|drop|insert|update|truncate|alter|create|grant|remove all|ignore your|"
                          r"ignore the|instructions|mita|hata)(?![a-z])")

_DEVICE_MODEL = (r"\b((device|phone|handset|mobile|smartphone|set)s? ?(models?|brands?|makes?|vendors?|manufacturers?|"
                 r"companies|company|types?|os)|models? of (the )?(phones?|devices?|handsets?|mobiles?)|iphones?|"
                 r"samsung (phones?|mobiles?|devices?|handsets?|users)|xiaomi|oppo|vivo|infinix|tecno|"
                 r"android (version|users)|ios (version|users)|operating systems?|which phones?|kaunsa phone|konsa phone)\b")
_UNAVAILABLE = re.compile(r"\b(age|ages|aged|age group|years old|gender|male|female|men|women|boys|girls|income|salary|"
                          r"roaming|religion|education level|umar|aurat|mard)\b|" + _DEVICE_MODEL)


def concept_fallback(question: str, data_range: dict):
    from datetime import timedelta

    if _UNAVAILABLE.search(question.lower()):
        return None
    cs = concepts.match(question)
    found = question_period(question, data_range)
    if not cs or not found:
        return None
    c = cs[0]
    label, start, end = found
    city = concepts.city_in(question)
    total = concepts.wants_total(question)
    build = concepts.total_sql if total else concepts.ranking_sql
    sql = build(c["key"], start.isoformat(), (end + timedelta(days=1)).isoformat(), city,
                settings.timezone, settings.full_table)
    return c, start, end, city, sql, total


def period_mismatch(question: str, sql: str, data_range: dict) -> str | None:
    from datetime import timedelta

    found = question_period(question, data_range)
    if not found:
        return None
    label, start, end = found
    dates = {m[:10] for m in context.SQL_DATE.findall(sql)}
    if not dates:
        return None
    want_start, want_end = start.isoformat(), (end + timedelta(days=1)).isoformat()
    if want_start in dates and want_end in dates:
        return None
    return (f"Not accepted yet: the question asks for '{label}', which is {start.isoformat()} to {end.isoformat()}. "
            f"Use exactly: start_time >= toUnixTimestamp('{want_start} 00:00:00', 'Asia/Karachi') AND start_time < "
            f"toUnixTimestamp('{want_end} 00:00:00', 'Asia/Karachi'). Your SQL uses other dates "
            f"({', '.join(sorted(dates))}). Fix the dates, and the understood question, title and assumptions to match, test, and submit again. "
            "If the question really meant other dates, submit the same SQL again.")


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_ONE_DATE = rf"(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MON}(?:,?\s+20\d\d)?|{_MON}\s+\d{{1,2}}(?!\d)(?:st|nd|rd|th)?(?:,?\s+20\d\d)?)"
_DATE_PHRASE = re.compile(
    rf"\s*\b(?:from|between|on|for|during|in)?\s*{_ONE_DATE}(?:\s*(?:to|-|–|and|until|till|through)\s*{_ONE_DATE})?", re.I)

_MONTH_PHRASE = re.compile(rf"\s*\b(?:in|during|for|of)?\s*(?:this month|last month)?\s*\(?\s*{_MON}\s+20\d\d\s*\)?|\s*\b(?:in|during|for|of)?\s*(?:this|last) month\b", re.I)


def _parse_date(text: str, year: int):
    from datetime import date as _date

    t = text.lower().strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
            return _date.fromisoformat(t)
        m = re.search(r"(\d{1,2})", t)
        mon = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t)
        y = re.search(r"(20\d\d)", t)
        if m and mon:
            return _date(int(y.group(1)) if y else year, _MONTHS[mon.group(1)], int(m.group(1)))
    except ValueError:
        return None
    return None


def sql_period(sql: str):
    from datetime import date as _date, timedelta

    found = context.SQL_DATE.findall(sql or "")
    if len(found) < 2:
        return None
    try:
        days = sorted({_date.fromisoformat(f[:10]) for f in found})
    except ValueError:
        return None
    start, end_excl = days[0], days[-1]
    if end_excl <= start:
        return None
    return start, end_excl - timedelta(days=1)


def _fmt_period(start, end) -> str:
    def f(d):
        return f"{d.day} {d.strftime('%b')}"
    return f"{f(start)} {start.year}" if start == end else f"{f(start)} to {f(end)} {end.year}"


_RELATIVE_PERIOD = re.compile(
    r"\b(?:(?:in|during|over|for|from|of)\s+)?(?:the\s+)?(?:last|past|previous)\s+(?:7|seven|full\s+7|7\s+full)?\s*"
    r"(?:full\s+)?(?:days?|week)\b"
    r"|\b(?:(?:in|during|for|from|on|of)\s+)?(?:yesterday|today)\b"
    r"|\b(?:(?:in|during|for|of)\s+)?(?:this|the current)\s+week\b",
    re.I)
_TOP_N = re.compile(r"\b(top|the\s+top)\s+(\d+|ten|five|twenty|three)?\s*", re.I)


def _exact_understood(text: str, sql: str, start, end, period: str) -> str:
    if not text:
        return text
    t = text
    when = f"on {period}" if start == end else f"from {period}"
    has_date = bool(re.search(_ONE_DATE, t, re.I) or re.search(rf"{_MON}\s+20\d\d", t, re.I))
    if _RELATIVE_PERIOD.search(t):
        if has_date:
            t = _RELATIVE_PERIOD.sub(" ", t)
        else:
            t = _RELATIVE_PERIOD.sub(f" {when} ", t, count=1)
            t = _RELATIVE_PERIOD.sub(" ", t)
        t = re.sub(r"\(\s*\)", " ", t)
        has_date = True
    if not has_date:
        body = t.rstrip(" ?.")
        t = f"{body} {when}" + ("?" if text.rstrip().endswith("?") else "")
    lim = re.search(r"(?i)\bLIMIT\s+(\d+)\s*;?\s*$", (sql or "").strip())
    ranked = re.search(r"(?i)\bORDER\s+BY\b[^;]*\bDESC\b", sql or "")
    if lim and ranked and 1 < int(lim.group(1)) <= 100:
        n = lim.group(1)
        m = _TOP_N.search(t)
        nxt = re.match(r"\s*([A-Za-z][\w-]*)(?:\s+([A-Za-z][\w-]*))?", t[m.end():]) if m else None
        plural = bool(nxt) and any(w and w.lower().endswith("s") and w.lower() not in ("is", "was", "has", "its", "this")
                                   for w in nxt.groups())
        if m and plural and not (m.group(2) or "").isdigit():
            t = t[:m.start()] + f"{m.group(1)} {n} " + t[m.end():]
    return re.sub(r"\s+([?.,!])", r"\1", re.sub(r"\s{2,}", " ", t)).strip()


def align_period_text(plan: dict, sql: str) -> dict:
    span = sql_period(sql)
    if not span:
        return plan
    start, end = span
    period = _fmt_period(start, end)
    other_dates = set(re.findall(r"(?:end_time|recv_time)\s*[<>=]+\s*toUnixTimestamp\(\s*'(\d{4}-\d{2}-\d{2})", sql or ""))
    for field in ("understood", "title"):
        text = plan.get(field) or ""
        mentioned = [_parse_date(m.group(0), end.year) for m in re.finditer(_ONE_DATE, text, re.I)]
        if any(d and not (start <= d <= end) and d.isoformat() not in other_dates for d in mentioned):
            cleaned = _DATE_PHRASE.sub("", text)
            cleaned = _MONTH_PHRASE.sub("", cleaned)
            cleaned = re.sub(r"\(\s*\)", "", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().rstrip("?.,").strip()
            sep = ", " if field == "title" else " "
            plan[field] = f"{cleaned}{sep}{'from ' if field == 'understood' and start != end else ('on ' if field == 'understood' else '')}{period}" + ("?" if field == "understood" else "")
    plan["understood"] = _exact_understood(plan.get("understood") or "", sql, start, end, period)
    assumptions = [a for a in plan.get("assumptions", []) if a.get("label", "").strip().lower() not in ("period", "time", "dates", "date range")]
    plan["assumptions"] = [{"label": "Period", "value": f"{period} (Pakistan time)"}] + _true_assumptions(assumptions, sql)
    return plan


_ASSUMPTION_NEEDS = {
    "coverage": re.compile(r"\b(loc|geo|cid|lac|mcc|mnc|usr_loc_info)\b", re.I),
    "excluded": re.compile(r"\bNOT\s+IN\b|!=|<>|\bNOT\s+has\s*\(|\bNOT\s+(I?LIKE|match)\b", re.I),
    "area": re.compile(r"\b(client_location|loc|server_location|client_geo|geo|cid|lac)\b", re.I),
}


_MISSING_NAMES = [
    ("age", r"\b(age|ages|aged|age group|years old|umar)\b"),
    ("gender", r"\b(gender|male|female|men|women|boys|girls|aurat|mard)\b"),
    ("device model", _DEVICE_MODEL),
    ("income", r"\b(income|salary)\b"),
    ("roaming", r"\broaming\b"),
    ("religion", r"\breligion\b"),
    ("education level", r"\beducation level\b"),
]
_STORED_IDS = re.compile(r"\b(imei|imsi|msisdn)\b|(?<!\d)\d{15}(?!\d)")


def missing_in(question: str) -> list[str]:
    q = (question or "").lower()
    out = [name for name, rx in _MISSING_NAMES if re.search(rx, q)]
    if _STORED_IDS.search(q):
        out = [n for n in out if n != "device model"]
    return out


def drop_unasked_city(plan: dict, question: str) -> dict:
    sql = (plan.get("sql") or "").lower().replace("asia/karachi", "")
    asked = set(concepts.cities_in(question))
    for city in concepts.CITIES:
        if city in asked or city.lower() in sql:
            continue
        for field in ("understood", "title"):
            text = plan.get(field) or ""
            if city.lower() not in text.lower():
                continue
            text = re.sub(rf"\s*,\s*{city}(?![a-z])|\s+(in|for|of|across)\s+{city}(\s+Pakistan)?(?![a-z])", "",
                          text, flags=re.I)
            plan[field] = re.sub(r"\s{2,}", " ", text).strip()
    plan["assumptions"] = [a for a in plan.get("assumptions", [])
                           if not (a.get("label", "").lower() in ("area", "place", "city", "location")
                                   and any(c.lower() in str(a.get("value", "")).lower() for c in concepts.CITIES
                                           if c not in asked and c.lower() not in sql))]
    return plan


def short_assumptions(plan: dict) -> dict:
    out = []
    for a in plan.get("assumptions", []):
        value = str(a.get("value", ""))
        items = [x.strip() for x in value.split(",") if x.strip()]
        if len(items) > 5 and len(value) > 80:
            generic = set(dictionary.base().get("value_lists", {}).get("generic_labels", {}).get("list", []))
            head = "background traffic: " if generic and sum(x in generic for x in items) >= len(items) * 0.6 else ""
            value = f"{head}{', '.join(items[:3])} and {len(items) - 3} more"
        out.append({**a, "value": value})
    plan["assumptions"] = out
    return plan


def note_missing(plan: dict, question: str) -> dict:
    plan = short_assumptions(drop_unasked_city(plan, question))
    q = question.lower()
    missing = missing_in(q)
    if not missing:
        return plan
    what = ", ".join(missing)
    note = f"{what}: this data has no {what} information, so that part of the question is not answered"
    plan["assumptions"] = [a for a in plan.get("assumptions", []) if a.get("label") != "Not in the data"]
    plan["assumptions"].append({"label": "Not in the data", "value": note})
    expl = plan.get("explanation") or ""
    if "no " + missing[0] not in expl.lower():
        plan["explanation"] = f"The data has no {what} information, so this shows what can be answered. {expl}".strip()
    return plan


def _true_assumptions(assumptions: list[dict], sql: str) -> list[dict]:
    out = []
    for a in assumptions:
        need = _ASSUMPTION_NEEDS.get(a.get("label", "").strip().lower())
        if need is not None and not need.search(sql or ""):
            continue
        out.append(a)
    return out


def resolved_names(question: str, data_range: dict, turns: list[dict] | None = None) -> list[str]:
    from datetime import timedelta

    out = context.reference_lines(context.references(question, turns), wants_profile(question))
    kinds_of_app = concepts.match(question)
    for e in (_safe(catalog.entities, question) or []):
        if e.kind == "category" and kinds_of_app:
            continue
        out.append(e.line() + ("" if e.sure else " (possible meaning; use it only if it fits the question)"))
    out += shape_hints(question)
    cat = concepts.catalogue()
    names = concepts.apps_in(question)
    calls = concepts.call_labels(question)
    for n in names:
        if n in calls:
            out.append(f"{n} calls (the app's call label only): app_label IN (" + ", ".join(f"'{x}'" for x in calls[n]) + ")")
            continue
        labels = cat[n]["labels"] if n in cat else concepts.app_names()[n]
        out.append(f"{n} (app, every label of it): app_label IN (" + ", ".join(f"'{x}'" for x in labels) + ")")
    for c in concepts.match(question)[:2]:
        if c["key"] in names or (calls and c["key"] == "calls"):
            continue
        line = f"{c['key'].replace('_', ' ')} (kind of app: {c['description']}): app_label IN {concepts.sql_list(c['key'])}"
        if c["hosts"]:
            line += (f"; unlabelled sites of this kind: match(requested_server, '{c['hosts']}'), named "
                     f"'Other {c['key'].replace('_', ' ')} sites'")
        out.append(line)
    found = question_period(question, data_range)
    if not found and (earlier := _safe(follow_up_period, question, turns)):
        found = ("the same period as the earlier answer", earlier[0], earlier[1])
    if found:
        label, start, end = found
        out.append(f"Period, {label}: {_fmt_period(start, end)}: start_time >= toUnixTimestamp('{start.isoformat()} "
                   f"00:00:00', '{settings.timezone}') AND start_time < toUnixTimestamp('"
                   f"{(end + timedelta(days=1)).isoformat()} 00:00:00', '{settings.timezone}')")
    return out


_PEOPLE = (r"(users?|subscribers?|customers?|numbers?|msisdns?|people|persons?|consumers?|accounts?|log|logon|bande|"
           r"downloaders?|uploaders?|streamers?|gamers?|viewers?|visitors?|watchers?|players?|callers?|fans?|"
           r"members?|clients?|talkers?|chatters?|listeners?|readers?|shoppers?|surfers?|subs|individuals?|anyone|anybody|someone)")
_NOT_PEOPLE = re.compile(r"(?<![a-z])(apps?|applications?|cities|city|sites?|websites?|domains?|servers?|isps?|"
                         r"operators?|networks?|categories|category|hours?|days?|areas?|locations?|by|per|of)(?![a-z])")
_LIST_HEAD = re.compile(rf"(?<![a-z])(top|heaviest|heavy|biggest|main|leading|most active|highest|power|bulk)\s+"
                        rf"((?:[\w.-]+\s+){{0,6}}?){_PEOPLE}(?![a-z])")
_LIST_OTHER = re.compile(
    rf"(?<![a-z])(which|what|list( of)?|show( me)?( the)?|kon|kaun|konse|kaunse|kon se|kaun se|kis|kin)\s+{_PEOPLE}"
    rf"(?![a-z])|(?<![a-z])(who|kisne|kis ne|kin logon ne|kaun log|kon log)\s+(is |are |was |were |has |have )?"
    r"(use|used|uses|using|access|accessed|accesses|visit|visited|visits|open|opened|download|downloaded|watch|"
    r"watched|chala|chalate|chalaya|istemal|spent|spends|spend|stayed|stays|played|plays|called|calls|uploaded|"
    r"uploads|streamed|consumed|consumes|tried|made|sent|received)"
    r"|(?<![a-z])who\s+(is|are|was|were|has|had|have)\s+(\w+\s+){0,3}?(most|longest|highest|biggest|largest|maximum|max)"
    r"(?![a-z])"
    r"|(?<![a-z])(data|bandwidth)\s+hogs?(?![a-z])"
    r"|(?<![a-z])who\s*\?|(?<![a-z])(anyone|anybody|someone)\s+(who\s+)?(go|went|goes|cross|crossed|use|used|exceed|exceeded|"
    r"download|downloaded|reach|reached)(?![a-z])"
    r"|(?<![a-z])(kis ne|kisne|kin logon ne|kin ne|kaun log|kon log|konse log|kaunse log)(?![a-z])"
    rf"|(?<![a-z]){_PEOPLE}\s+(that|who|which)\s+(\w+\s+)?(used|use|uses|visited|visit|accessed|opened|watched|"
    r"downloaded|uploaded|made|called|played|spent|tried)(?![a-z])")


def wants_subscriber_list(question: str) -> bool:
    q = (question or "").lower()
    m = _LIST_HEAD.search(q)
    if m and not _NOT_PEOPLE.search(m.group(2) or ""):
        return True
    if re.search(r"(?<![a-z])(how many|kitne|kitni|count|number of)(?![a-z])", q):
        return False
    return bool(_LIST_OTHER.search(q))


_PROFILE = re.compile(r"(?<![a-z])(address|addresses|addr|location|locations|ghar|pata|isp|isps|network|provider|"
                      r"operator|imsi|imei|device|handset|phone number|mobile number|msisdn|city|cities|area|areas|cell|cells|tower|towers|geo|networks|providers|operators|imsis|imeis|devices|handsets|"
                      r"coordinates|lat|lon|details|detail|profile|info|information|live|lives|living|reside|resides|"
                      r"rehta|rehti|rehte|kahan|where)(?![a-z])")


def wants_profile(question: str) -> bool:
    q = (question or "").lower()
    return bool(_PROFILE.search(q)) and bool(
        context.refers_back(q) or context.phones(q) or _STORED_IDS.search(q) or catalog._IPV4.search(q)
        or re.search(r"(?<![a-z0-9])([A-Z]{2,4}\d{6,}|\d{5,}[a-z]{3,})(?![a-z0-9])", question or ""))


_SUB_VOLUME = "round(sum(c2s_byte_num + s2c_byte_num) / 1e6, 2) AS volume_mb"


def _list_measure(question: str) -> tuple[str, str, str]:
    q = (question or "").lower()
    if re.search(r"(?<![a-z])(upload|uploads|uploaders?|uploading|uplink|sent|send|sends|bhejta|bhejte)(?![a-z])", q):
        return "upload_mb", "round(sum(c2s_byte_num) / 1e6, 2) AS upload_mb", "sum(c2s_byte_num)"
    if re.search(r"(?<![a-z])(downloads?|downloaders?|downloading|downlink|received)(?![a-z])", q):
        return "download_mb", "round(sum(s2c_byte_num) / 1e6, 2) AS download_mb", "sum(s2c_byte_num)"
    if re.search(r"(?<![a-z])(connections?|sessions?|hits?|requests?|times|visits?|visited|visiting|accessed|"
                 r"access|accesses|opened|opens|baar|khola|kholi|calls|called|calling)(?![a-z])", q):
        return "connections", "count() AS connections", "connections"
    if re.search(r"(?<![a-z])(duration|time spent|longest|hours spent|minutes|hours|spent (the )?(most |more )?time|"
                 r"spends (the )?(most |more )?time|spend (the )?(most |more )?time|online|kitni der)(?![a-z])", q):
        return "hours", "round(sum(duration) / 3.6e6, 2) AS hours", "sum(duration)"
    return "volume_mb", _SUB_VOLUME, "sum(c2s_byte_num + s2c_byte_num)"


def shape_hints(question: str) -> list[str]:
    out = []
    if wants_subscriber_list(question):
        name, expr, order = _list_measure(question)
        extra = ", count() AS connections" if name != "connections" else f", {_SUB_VOLUME}"
        out.append(f"Answer shape: a list of individual subscribers. SELECT subscriber_id, {expr}{extra} ... WHERE "
                   f"position(subscriber_id, '_') = 0 (identified subscribers only) AND <the question's filters> GROUP "
                   f"BY subscriber_id ORDER BY {order} DESC LIMIT <the number asked; 10 for 'the top user', so ties "
                   f"show; else 20>. Never count users per subscriber (always 1). If the amounts are tiny (one "
                   f"website), use KB (/ 1e3) instead of MB.")
    if wants_profile(question):
        q = (question or "").lower()
        asked = [f for f, rx in (("arrayStringConcat(arraySort(groupUniqArrayIf(loc, loc != '')), ' | ') AS address", r"address|ghar|pata|location|live|rehta|where|kahan|area"),
                                 ("arrayStringConcat(arraySort(groupUniqArrayIf(client_asn, client_asn != '')), ' | ') AS isp", r"isp|network|provider|operator"),
                                 ("arrayStringConcat(arraySort(groupUniqArrayIf(client_location, client_location != '')), ' | ') AS city", r"city|shehar|location|where|kahan"),
                                 ("arrayStringConcat(arraySort(groupUniqArrayIf(imsi, imsi != '')), ' | ') AS imsi", r"imsi|sim"), ("arrayStringConcat(arraySort(groupUniqArrayIf(imei, imei != '')), ' | ') AS imei", r"imei|device|handset|phone set"),
                                 ("arrayStringConcat(arraySort(groupUniqArrayIf(geo, geo != '')), ' | ') AS geo", r"geo|coordinates|lat|lon|map"),
                                 ("arrayStringConcat(arraySort(groupUniqArrayIf(client_ip, client_ip != '')), ' | ') AS ip", r"(?<![a-z])ip(?![a-z])"),
                                 ("anyIf(cid, cid != '') AS cell_id, anyIf(lac, lac != '') AS lac", r"cell|tower"))
                 if re.search(rx, q)] or ["arrayStringConcat(arraySort(groupUniqArrayIf(loc, loc != '')), ' | ') AS address", "arrayStringConcat(arraySort(groupUniqArrayIf(client_asn, client_asn != '')), ' | ') AS isp", "arrayStringConcat(arraySort(groupUniqArrayIf(client_location, client_location != '')), ' | ') AS city"]
        out.append("Answer shape: stored details of the given subscriber(s), one row each: SELECT subscriber_id, "
                   + ", ".join(asked) + " ... GROUP BY subscriber_id. Return only these details (plus a measure the "
                   "earlier answer had, if the question asks to keep it). loc is the house address or serving cell site; "
                   "client_asn is the ISP.")
    return out


_GROUPS_BY_SUB = re.compile(r"GROUP\s+BY\s+(?:(?!\bHAVING\b|\bORDER\b|\bLIMIT\b)[^;()])*?(?<![\w.])subscriber_id\b"
                            r"(?!\s*,\s*'_')", re.I)


def subscriber_shape(question: str, sql: str, result=None) -> str | None:
    s = sql or ""
    grouped = bool(_GROUPS_BY_SUB.search(s))
    if grouped and re.search(r"uniq\w*\(\s*subscriber_id", s, re.I):
        return ("Not accepted yet: the SQL groups by subscriber_id and counts users inside each group, which is always 1, "
                "so the order means nothing. For a list of subscribers, measure each one by data volume "
                f"({_SUB_VOLUME}) and connections (count() AS connections), ORDER BY the unrounded measure the question "
                "asks for (sum(c2s_byte_num + s2c_byte_num) by default), test, and submit again.")
    shown = result is None or any("_" in str(r[i]) for i, c in enumerate(result.columns)
                                  if c["name"].lower() in ("subscriber_id", "user", "user_id", "subscriber", "number")
                                  for r in result.rows)
    if grouped and shown and not re.search(r"position\(\s*subscriber_id\s*,\s*'_'\s*\)\s*=\s*0|subscriber_id\s+NOT\s+LIKE\s+'%\\?_%'"
                                           r"|subscriber_id\s+IN\s*\(|subscriber_id\s*=\s*'", s, re.I):
        return ("Not accepted yet: the list includes unidentified records, whose subscriber_id is '<ip>_<port>' and not a "
                "person. Add AND position(subscriber_id, '_') = 0 to the WHERE clause, test, and submit again.")
    if not grouped and wants_subscriber_list(question) and not re.search(r"subscriber_id\s*(IN\s*\(|=\s*')", s, re.I):
        name, expr, order = _list_measure(question)
        return (f"Not accepted yet: the question asks for individual subscribers (a list of people), but the SQL does "
                f"not return one row per subscriber. SELECT subscriber_id, {expr}, keep every filter the question "
                f"names, add position(subscriber_id, '_') = 0, GROUP BY subscriber_id ORDER BY {order} DESC with the "
                "number asked as LIMIT, update the understood question and title, test, and submit again.")
    return None


def entity_missing(question: str, sql: str, turns: list[dict] | None = None) -> str | None:
    s = (sql or "").lower()
    missing = []
    one_city = concepts.city_in(question)
    for e in catalog.entities(question):
        if not e.sure or not e.values:
            continue
        if e.kind == "city" and e.name == one_city:
            continue
        if any(v.lower() in s for v in e.values) and not (
                e.kind in ("server_place", "server_country") and not re.search(
                    r"server_location\s*(=|in\s*\()\s*'?[^)]*" + re.escape(e.values[0].lower())
                    + r"|endswith\(\s*server_location\s*,\s*'\s*" + re.escape(e.values[0].lower()), s)):
            continue
        if e.kind == "city" and re.search(rf"loc\s+i?like\s+'%{re.escape(e.name.lower())}", s):
            continue
        missing.append(e)
    if not missing:
        return None
    return ("Not accepted yet: the question names " + "; ".join(f"{e.name} ({e.kind.replace('_', ' ')})" for e in missing)
            + ", but the SQL does not filter on " + ("it" if len(missing) == 1 else "them") + ". Add to the WHERE clause: "
            + " AND ".join(e.sql for e in missing) + ". Keep every other filter, name it in the understood question and "
            "title, test, and submit again. If the name is really meant as something else, submit the same SQL again "
            "and say so in the explanation.")


def isp_missing(question: str, sql: str) -> str | None:
    s = sql or ""
    missing = [(n, f) for n, f in context.operators(question)
               if not any(v in s for v in re.findall(r"'([^']+)'", f) if v != "410")]
    if not missing:
        return None
    sim = any("mnc" in f for _, f in missing)
    return ("Not accepted yet: the question names " + ", ".join(n for n, _ in missing) + ", but the SQL does not "
            "filter on it. Add " + " / ".join(f"{n}: {f}" for n, f in missing) + " to the WHERE clause (client_asn "
            "is the subscriber's ISP" + ("; mnc is the SIM operator code" if sim else "") + "), keep every other "
            "filter, put it in the understood question and title, test, and submit again.")


def reference_dropped(question: str, sql: str, turns: list[dict]) -> str | None:
    refs = context.references(question, turns)
    if not refs:
        return None
    r = refs[0]
    stored = context._stored_values(r["column"], r["values"])
    if any(f"'{v}'" in (sql or "") for v in list(r["values"]) + stored):
        return None
    return ("Not accepted yet: the question is about rows of an earlier answer, but the SQL does not use them. " +
            context.reference_lines(refs, wants_profile(question))[0] + " Update the understood question, test, and submit again.")


_COMPLEX = re.compile(r"(?<![a-z])(vs|versus|compare|comparison|also|both|share|percent|percentage|ratio|growth|"
                      r"change|difference|except|without|excluding|between)(?![a-z])")


def needs_review(question: str) -> bool:
    q = question.lower()
    return bool(_COMPLEX.search(q)) or len(concepts.cities_in(q)) > 1 or len(concepts.apps_in(q)) > 1


def review_enabled(cfg: LLMConfig) -> bool:
    mode = (settings.review_pass or "auto").lower()
    if mode in ("on", "always", "true", "1"):
        return True
    if mode in ("off", "never", "false", "0"):
        return False
    return cfg.provider != "anthropic"


def _review_request(plan: dict, result: QueryResult, periods: list[str]) -> str:
    cols = ", ".join(c["name"] for c in result.columns)
    rows = [("  " + ", ".join(str(v) for v in r)) for r in result.rows[:5]]
    assumptions = "; ".join(f"{a['label']}: {a['value']}" for a in plan.get("assumptions", [])) or "none"
    lines = [
        "Not shown yet. Check your answer once against the question before it is shown.",
        f"Understood question: {plan.get('understood') or '(missing: write it)'}",
        f"Your assumptions: {assumptions}",
        f"Result ({len(result.rows)} rows): {cols}",
        *rows,
        "",
        *periods,
        "",
        "Checklist:",
        "1. The dates in the WHERE clause are exactly the ready-made period above that matches the question's wording "
        "(for example 'yesterday' is the last full day of data, not the day before it).",
        "2. Every filter and exclusion in your assumptions is really in the SQL. If you say background traffic is "
        "excluded, the SQL must have app_label NOT IN (...the generic labels...).",
        "3. The measure, grouping and sort order answer the question, and the result looks plausible.",
        "4. Rankings return enough rows to compare: the top 10 for 'the most / the top one', the top 20 for lists, "
        "unless the question gives a number.",
        "5. The understood question, title and explanation describe what the SQL actually does.",
        "Check silently. Reply with a tool call only, no text: if everything is right, call confirm_answer (it takes "
        "no arguments; do not send the answer again). If anything is wrong, call submit_answer with the corrected "
        "answer. Do not repeat this checklist or the question.",
    ]
    return chr(10).join(lines)


def _norm_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split()).lower()


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return None


def summarize(columns: list[dict], rows: list[list]) -> dict:
    out = {}
    for i, col in enumerate(columns):
        vals = [x for x in (_num(r[i]) for r in rows) if x is not None]
        if not vals or len(vals) < len(rows) * 0.8:
            continue
        total = sum(vals)
        s = {"sum": round(total, 3), "min": min(vals), "max": max(vals), "mean": round(total / len(vals), 3)}
        if total:
            s["first_row_share_of_sum_pct"] = round(vals[0] / total * 100, 2)
        out[col["name"]] = s
    return out


def conclude(data: dict, question: str, cfg: LLMConfig) -> dict:
    res = data.get("result") or {}
    columns, rows = res.get("columns", []), res.get("rows", [])
    names = [c["name"] for c in columns]
    shown = rows[:300]
    csv_lines = [",".join(names)] + [",".join("" if v is None else str(v) for v in row) for row in shown]
    note = ""
    if len(rows) > len(shown):
        note = f"Only the first {len(shown)} of {len(rows)} rows are listed below. The summary covers all rows."
    if res.get("truncated"):
        note += " The result was cut off at the row limit, so totals cover the returned rows only."
    assumptions = "\n".join(f"- {a['label']}: {a['value']}" for a in data.get("assumptions", []))
    user = (
        f"<question>{question or data.get('title', '')}</question>\n"
        f"<title>{data.get('title', '')}</title>\n"
        f"<assumptions>\n{assumptions or '- none stated'}\n</assumptions>\n"
        f"<sql>\n{data.get('sql', '')}\n</sql>\n"
        f"<summary>{json.dumps(summarize(columns, rows))}</summary>\n"
        f"<result rows=\"{len(rows)}\">{note}\n" + "\n".join(csv_lines) + "\n</result>"
    )
    cfg = LLMConfig(**{**cfg.__dict__, "effort": "low"})
    candidates = _derivable_numbers(columns, rows)
    checked_once = False
    try:
        session = llm.open_session(cfg, context.CONCLUSION_INSTRUCTIONS, context.CONCLUSION_TOOLS, user)
        for _ in range(4):
            turn = session.send()
            if turn.stop == "refusal":
                raise AgentError("The AI declined to write a conclusion.")
            call = next((c for c in turn.tool_calls if c.name == "submit_conclusion"), None)
            data_out = call.args if call else _json_from_text(turn.text)
            if data_out and data_out.get("headline"):
                out = {
                    "headline": str(data_out["headline"]),
                    "findings": [str(x) for x in data_out.get("findings") or []],
                    "caveats": [str(x) for x in data_out.get("caveats") or []],
                }
                text = " ".join([out["headline"], *out["findings"], *out["caveats"]])
                bad = unverified_numbers(text, candidates)
                if bad and not checked_once:
                    checked_once = True
                    fix = (f"These figures do not match the result or simple arithmetic on it: {', '.join(bad)}. "
                           "Recalculate them from the rows (or remove them) and call submit_conclusion again. "
                           "When you give a share, say what it is a share of (for example 'of the top 5 shown').")
                    if call:
                        session.add_tool_results([(call, fix, True)])
                    else:
                        session.add_user(fix)
                    continue
                if bad:
                    out["unverified"] = bad
                return out
            if turn.tool_calls:
                session.add_tool_results([(c, "Call submit_conclusion only.", True) for c in turn.tool_calls])
            elif turn.text.strip():
                return {"headline": turn.text.strip().splitlines()[0], "findings": turn.text.strip().splitlines()[1:6], "caveats": []}
            else:
                session.add_user("Call submit_conclusion with the conclusion.")
    except LLMError as e:
        raise AgentError(str(e)) from e
    raise AgentError("The AI did not return a conclusion. Try again.")


_NUMBER = re.compile(r"(?<![\w.\-/:])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|percent|KB|GB|MB|TB|[KkMmBb](?![A-Za-z]))?")


def _derivable_numbers(columns: list[dict], rows: list[list]) -> list[float]:
    out: set[float] = set()
    for i, _ in enumerate(columns):
        vals = [v for v in (_num(r[i]) for r in rows) if v is not None]
        if not vals:
            continue
        total = sum(vals)
        out.update(vals)
        out.update({total, total / len(vals), float(len(vals))})
        top = vals[:25]
        for a in top:
            if total:
                out.add(a / total * 100)
            for b in top:
                if a is b:
                    continue
                out.add(abs(a - b))
                if b:
                    out.add(a / b)
                    out.add(a / b * 100)
                    out.add((a - b) / b * 100)
        if len(vals) >= 2 and total:
            out.add(sum(vals[:2]) / total * 100)
            out.add(sum(vals[:3]) / total * 100)
    out.add(float(len(rows)))
    return sorted(out)


def unverified_numbers(text: str, candidates: list[float]) -> list[str]:
    import bisect

    bad = []
    for m in _NUMBER.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            x = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit in ("k",):
            x *= 1e3
        elif unit in ("m",):
            x *= 1e6
        elif unit in ("b",):
            x *= 1e9
        if not unit and x.is_integer() and (x <= 31 or 1900 <= x <= 2100):
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get(unit, 1)
        factors = (1, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9) if unit in ("kb", "mb", "gb", "tb") else (1,)
        found = False
        for f in factors:
            y = x * f
            tol = max(abs(y) * 0.01, 0.5 * 10 ** -decimals * scale * f) + 1e-9
            j = bisect.bisect_left(candidates, y - tol)
            if j < len(candidates) and candidates[j] <= y + tol:
                found = True
                break
        if not found:
            bad.append(m.group(0).strip())
    return list(dict.fromkeys(bad))


def _json_from_text(text: str) -> dict | None:
    m = _JSON_BLOCK.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def elapsed(start: float) -> float:
    return round(time.time() - start, 1)
