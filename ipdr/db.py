from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import requests

from .config import settings


class QueryError(Exception):
    pass


_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
# Stored clocks sometimes run decades ahead (a max of July 2104). Rows inside
# 2010–2030 are used as the real range. If nothing is in that window, displayed
# dates are shifted back and this many years are added again before a query runs.
_clock_ahead_years = 0
_SANE_LO = 1262304000  # 2010-01-01 UTC
_SANE_HI = 1924992000  # 2031-01-01 UTC
_TS_LITERAL = re.compile(
    r"(toUnixTimestamp\(\s*')(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)(')"
)


@dataclass
class QueryResult:
    columns: list[dict]
    rows: list[list]
    elapsed_s: float
    rows_read: int
    bytes_read: int
    truncated: bool = False
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": len(self.rows),
            "elapsed_s": round(self.elapsed_s, 3),
            "rows_read": self.rows_read,
            "bytes_read": self.bytes_read,
            "truncated": self.truncated,
        }


_ERR_RE = re.compile(r"Code:\s*(\d+)\.\s*DB::Exception:\s*(.*?)(?:\s*\(([A-Z_]+)\))?\s*\(version [^)]*\)", re.S)


def _clean_error(text: str) -> str:
    m = _ERR_RE.search(text)
    if not m:
        return text.strip().splitlines()[0][:600] if text.strip() else "Unknown database error"
    code, msg, name = m.group(1), m.group(2).strip(), m.group(3)
    msg = re.sub(r"\s+", " ", msg)
    msg = re.sub(r"^[\w.]+: ", "", msg)
    if len(msg) > 600:
        msg = msg[:600] + "..."
    return f"{msg} [{name or 'code ' + code}]"


class ClickHouse:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.auth = (settings.ch_user, settings.ch_password)

    def _settings(self, max_rows: int | None) -> dict:
        return {
            "readonly": "2",
            "database": settings.ch_database,
            "max_execution_time": str(settings.max_exec_seconds),
            "max_rows_to_read": str(settings.max_rows_to_read),
            "max_memory_usage": str(settings.max_memory_bytes),
            "max_result_rows": str((max_rows or settings.max_result_rows) + 1),
            "result_overflow_mode": "break",
            "output_format_json_quote_64bit_integers": "0",
            "output_format_json_quote_denormals": "0",
        }

    def query(self, sql: str, *, max_rows: int | None = None, params: dict | None = None,
              timeout: float | None = None) -> QueryResult:
        started = time.perf_counter()
        http_params = self._settings(max_rows)
        for k, v in (params or {}).items():
            http_params[f"param_{k}"] = str(v)
        try:
            resp = self.session.post(
                settings.ch_url,
                params=http_params,
                data=(_prepare_sql(sql).rstrip().rstrip(";") + "\nFORMAT JSONCompact").encode("utf-8"),
                timeout=timeout or settings.max_exec_seconds + 15,
            )
        except requests.RequestException as e:
            raise QueryError(f"Cannot reach the database: {e.__class__.__name__}") from e
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            raise QueryError(_clean_error(resp.text))
        body = resp.json()
        stats = body.get("statistics", {})
        limit = max_rows or settings.max_result_rows
        rows = body.get("data", [])
        truncated = len(rows) > limit
        return QueryResult(
            columns=body.get("meta", []),
            rows=rows[:limit],
            elapsed_s=stats.get("elapsed", elapsed),
            rows_read=int(stats.get("rows_read", 0)),
            bytes_read=int(stats.get("bytes_read", 0)),
            truncated=truncated,
        )

    def cached_query(self, sql: str, *, max_rows: int | None = None, params: dict | None = None,
                     ttl: float = 600, timeout: float | None = None) -> QueryResult:
        key = (sql, max_rows, tuple(sorted((params or {}).items())))
        now = time.time()
        with _cache_lock:
            hit = _cache.get(key)
            if hit and now - hit[0] < ttl:
                _cache.move_to_end(key)
                return hit[1]
        res = self.query(sql, max_rows=max_rows, params=params, timeout=timeout)
        with _cache_lock:
            _cache[key] = (now, res)
            while len(_cache) > 200:
                _cache.popitem(last=False)
        return res

    def stream_csv(self, sql: str, *, params: dict | None = None, max_rows: int):
        http_params = self._settings(max_rows)
        for k, v in (params or {}).items():
            http_params[f"param_{k}"] = str(v)
        try:
            resp = self.session.post(
                settings.ch_url, params=http_params, stream=True,
                data=(_prepare_sql(sql).rstrip().rstrip(";") + "\nFORMAT CSVWithNames").encode("utf-8"),
                timeout=settings.max_exec_seconds + 15,
            )
        except requests.RequestException as e:
            raise QueryError(f"Cannot reach the database: {e.__class__.__name__}") from e
        if resp.status_code != 200:
            raise QueryError(_clean_error(resp.text))

        def chunks():
            try:
                yield b"\xef\xbb\xbf"
                yield from resp.iter_content(chunk_size=65536)
            finally:
                resp.close()

        return chunks()

    def estimate_rows(self, sql: str) -> int | None:
        try:
            res = self.query("EXPLAIN ESTIMATE " + sql.rstrip().rstrip(";"), max_rows=100)
        except QueryError:
            return None
        names = [c["name"] for c in res.columns]
        if "rows" not in names:
            return None
        i = names.index("rows")
        return sum(int(r[i]) for r in res.rows)

    def data_range(self) -> dict:
        global _clock_ahead_years
        table = settings.full_table
        # min/max alone follows the sort key. A scalar subquery in the same SELECT
        # was re-run across the whole table and the page waited on it.
        res = self.query(
            f"SELECT toUInt32(min(start_time)), toUInt32(max(start_time)) FROM {table}",
            max_rows=1,
            timeout=8,
        )
        if not res.rows or res.rows[0][0] is None or res.rows[0][1] is None:
            raise QueryError("The table has no start_time values.")
        raw_lo, raw_hi = res.rows[0]
        if _usable(raw_lo) and _usable(raw_hi):
            _clock_ahead_years = 0
            first_v, last_v = raw_lo, raw_hi
        else:
            capped = self.query(
                f"SELECT toUInt32(min(start_time)), toUInt32(max(start_time)) FROM {table} "
                f"WHERE start_time >= {_SANE_LO} AND start_time < {_SANE_HI}",
                max_rows=1,
                timeout=8,
            )
            sane_lo, sane_hi = capped.rows[0] if capped.rows else (None, None)
            if _usable(sane_lo) and _usable(sane_hi):
                _clock_ahead_years = 0
                first_v, last_v = sane_lo, sane_hi
            else:
                _clock_ahead_years = _years_ahead(raw_hi)
                first_v, last_v = raw_lo, raw_hi
        first = _show_time(first_v)
        last = _show_time(last_v)
        last_day = last[:10]
        complete = last[11:13] >= "23"
        return {
            "first": first,
            "last": last,
            "last_full_day": last_day if complete else _prev_day(last_day),
        }


def _usable(value) -> bool:
    if value is None:
        return False
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    return _SANE_LO <= number < _SANE_HI


def _years_ahead(value) -> int:
    """How many years to subtract so a far-future clock lands on or before today."""
    from datetime import date

    seen = _as_karachi(value)
    today = date.today()
    if seen.year <= today.year + 1:
        return 0
    years = seen.year - today.year
    shifted = _add_years(seen, -years)
    if shifted.date() > today:
        years += 1
    return years


def _show_time(value) -> str:
    return _add_years(_as_karachi(value), -_clock_ahead_years).strftime("%Y-%m-%d %H:%M:%S")


def _add_years(moment, years: int):
    if not years:
        return moment
    try:
        return moment.replace(year=moment.year + years)
    except ValueError:
        return moment.replace(year=moment.year + years, day=28)


def _prepare_sql(sql: str) -> str:
    if not _clock_ahead_years:
        return sql

    def repl(match):
        raw = match.group(2).replace("T", " ")
        from datetime import datetime
        if len(raw) == 10:
            moment = datetime.strptime(raw, "%Y-%m-%d")
            shown = _add_years(moment, _clock_ahead_years).strftime("%Y-%m-%d")
        else:
            moment = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
            shown = _add_years(moment, _clock_ahead_years).strftime("%Y-%m-%d %H:%M:%S")
        return match.group(1) + shown + match.group(3)

    return _TS_LITERAL.sub(repl, sql)


def _as_karachi(value):
    """Unix seconds (UTC) as a Pakistan-local datetime."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if isinstance(value, str) and len(value) >= 10 and value[4] == "-":
        return datetime.fromisoformat(value[:19])
    return datetime.fromtimestamp(int(value), ZoneInfo(settings.timezone))


def _prev_day(day: str) -> str:
    from datetime import date, timedelta

    return (date.fromisoformat(day) - timedelta(days=1)).isoformat()


db = ClickHouse()
