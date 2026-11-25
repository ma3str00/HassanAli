from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).with_name("app_concepts.yaml")


@lru_cache(maxsize=1)
def catalogue() -> dict:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8")) or {}
    out = {}
    for key, c in data.items():
        out[key] = {
            "key": key,
            "description": str(c.get("description", "")),
            "synonyms": [str(s).lower() for s in c.get("synonyms", [])],
            "labels": [str(x) for x in c.get("labels", [])],
            "hosts": str(c.get("hosts") or ""),
            "includes": [str(x) for x in c.get("includes", [])],
        }
    for c in out.values():
        for sub in c["includes"]:
            part = out[sub]
            c["labels"] += [x for x in part["labels"] if x not in c["labels"]]
            if part["hosts"]:
                c["hosts"] = f"{c['hosts']}|{part['hosts']}" if c["hosts"] else part["hosts"]
    return out


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9+.' ]+", " ", (text or "").lower()) + " "


def _stored_categories() -> set:
    from . import catalog
    return set(catalog._index()["categories"])


def match(text: str) -> list[dict]:
    t = _norm(text)
    found = []
    for c in catalogue().values():
        hits = [s for s in c["synonyms"] if f" {s} " in t or f" {s.replace(' ', '')} " in t]
        key_words = c["key"].replace("_", " ")
        if f" {key_words} " in t:
            hits.append(key_words)
        hits = [h for h in hits if not (f" {h} categor" in t and h in _stored_categories())]
        apps = _app_name_set()
        if hits and key_words not in hits and all(h in apps or h.replace(" ", "") in apps for h in hits):
            continue
        if hits:
            found.append((max(len(h) for h in hits), hits, c))
    found.sort(key=lambda x: (-x[0], len(x[2]["labels"])))
    taken: list[str] = []
    out = []
    for _, hits, c in found:
        if taken and all(any(f" {h} " in f" {t} " for t in taken) for h in hits):
            continue
        out.append(c)
        taken += hits
    return out


@lru_cache(maxsize=1)
def _known_words() -> frozenset:
    from . import context
    words = set(context._DATA_WORDS) | set(context._QUESTION_WORDS)
    for c in catalogue().values():
        for s in c["synonyms"] + [c["key"].replace("_", " ")]:
            words.update(s.split())
    words |= {"videos", "video", "movies", "music", "songs", "games", "gaming", "calls", "messages", "photos",
              "sites", "website", "websites", "streaming", "shopping", "dating", "people", "number", "numbers"}
    return frozenset(words)


def _app_name_set() -> set:
    try:
        return set(app_names())
    except RecursionError:
        return set()


def labels(key: str) -> list[str]:
    c = catalogue().get(key)
    return list(c["labels"]) if c else []


def sql_list(key: str) -> str:
    return "(" + ", ".join("'" + x.replace("'", "\\'") + "'" for x in labels(key)) + ")"


def describe(c: dict, with_labels: bool = True) -> str:
    s = f"{c['key']}: {c['description']}"
    if with_labels:
        s += f" app_label IN {sql_list(c['key'])}"
    if c["hosts"]:
        s += f" Unlabelled sites of this kind: match(requested_server, '{c['hosts']}')."
    return s


def prompt_block() -> str:
    lines = [
        "app_concepts:  # What kind of app each app_label is. Use these for any question about a KIND of app,",
        "               # service or content. Words people use for each concept are in brackets.",
    ]
    for c in catalogue().values():
        words = ", ".join(c["synonyms"][:12])
        lines.append(f"  - {c['key']} [{words}]: {c['description']}")
        if c["includes"]:
            lines.append(f"      app_label IN the labels of {', '.join(c['includes'])} together")
            continue
        lines.append(f"      app_label IN {sql_list(c['key'])}")
        if c["hosts"]:
            lines.append(f"      unlabelled sites: match(requested_server, '{c['hosts']}'). When you include them, "
                         f"name them with if(app_label IN {sql_list(c['key'])}, app_label, 'Other "
                         f"{c['key'].replace('_', ' ')} sites') so generic labels such as SSL never appear.")
    return "\n".join(lines)


CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Quetta", "Gujranwala",
          "Sialkot", "Hyderabad", "Sargodha", "Bahawalpur", "Sukkur", "Abbottabad", "Mardan", "Gujrat", "Sahiwal",
          "Larkana", "Sheikhupura", "Jhang", "Rahim Yar Khan", "Okara", "Dera Ghazi Khan", "Mirpur Khas", "Kasur"]
_CITY_SHORT = {"khi": "Karachi", "lhr": "Lahore", "isb": "Islamabad", "rwp": "Rawalpindi", "pindi": "Rawalpindi",
               "ryk": "Rahim Yar Khan", "dgk": "Dera Ghazi Khan", "dg khan": "Dera Ghazi Khan", "di khan": "Dera Ismail Khan",
               "d g khan": "Dera Ghazi Khan", "d i khan": "Dera Ismail Khan", "dik": "Dera Ismail Khan",
               "r y khan": "Rahim Yar Khan",
               "fsd": "Faisalabad", "mux": "Multan", "pew": "Peshawar", "isl": "Islamabad"}


def cities_in(text: str) -> list[str]:
    t = _norm(text)
    t = t + " ".join(t.replace(".", " ").split()) + " "
    found = {c for c in CITIES if f" {c.lower()} " in t}
    found |= {full for short, full in _CITY_SHORT.items() if f" {short} " in t}
    return [c for c in CITIES if c in found] + sorted(c for c in found if c not in CITIES)


def city_in(text: str) -> str | None:
    found = cities_in(text)
    return found[0] if len(found) == 1 else None


ISP_PATH = Path(__file__).with_name("isps.yaml")
_SIM_WORDS = re.compile(r"(?<![a-z])(sim|sims|mnc|operator code|network code|home network|home operator)(?![a-z])")


@lru_cache(maxsize=1)
def isp_catalogue() -> dict:
    data = yaml.safe_load(ISP_PATH.read_text(encoding="utf-8")) or {}
    return {name: {"name": name, "words": [str(w).lower() for w in c.get("words", [])],
                   "asn": [str(x) for x in c.get("asn") or []], "mnc": [str(x) for x in c.get("mnc") or []]}
            for name, c in data.items()}


def isps_in(text: str) -> list[dict]:
    t = _norm(text)
    return [c for c in isp_catalogue().values()
            if any(f" {w} " in t or (" " in w and f" {w.replace(' ', '')} " in t) for w in c["words"])]


def wants_sim_operator(text: str) -> bool:
    return bool(_SIM_WORDS.search((text or "").lower()))


def isp_filter(c: dict, sim: bool = False) -> str:
    if c["mnc"] and (sim or not c["asn"]):
        return f"mcc = '410' AND mnc IN ({', '.join(repr(m) for m in c['mnc'])})"
    return "client_asn IN (" + ", ".join("'" + a.replace("'", "\\'") + "'" for a in c["asn"]) + ")"


def isp_brand_sql() -> str:
    parts = []
    for c in isp_catalogue().values():
        if len(c["asn"]) > 1:
            parts.append("client_asn IN (" + ", ".join("'" + a.replace("'", "\\'") + "'" for a in c["asn"]) + f"), '{c['name']}'")
    return "multiIf(" + ", ".join(parts) + ", client_asn)" if parts else "client_asn"


def isp_brands_split(values: list[str]) -> list[str]:
    vals = set(values)
    return [c["name"] for c in isp_catalogue().values() if len(vals & set(c["asn"])) > 1]


def isp_prompt_block() -> str:
    lines = ["isp_names:  # ISP / operator brand [words people use] -> exact client_asn values (the network). Any other",
             "            # provider: find_values on client_asn. mnc only when the question says SIM / mnc / operator code."]
    for c in isp_catalogue().values():
        target = (", ".join(repr(a) for a in c["asn"]) if c["asn"]
                  else f"no client_asn entry: mnc IN ({', '.join(repr(m) for m in c['mnc'])})")
        sim = f"; SIM operator code mnc {'/'.join(c['mnc'])}" if c["mnc"] and c["asn"] else ""
        lines.append(f"  - {c['name']} [{', '.join(c['words'][:5])}]: {target}{sim}")
    lines.append(f"  breakdown by ISP (one row per brand): {isp_brand_sql()} AS isp ... GROUP BY isp, AND client_asn != ''")
    return "\n".join(lines)


_NOT_APP_NAMES = {"adjust", "find", "grab", "hotspot", "kick", "like", "line", "outfit", "ring", "riot", "signal",
                  "steam", "trend", "weather", "booking", "emirates", "windows", "office", "google", "microsoft",
                  "apple", "amazon", "mail", "imap", "smtp", "pop3", "x224", "vivo", "zoom", "nike", "sayhi"}


_SHORT_APP_NAMES = {"imo", "olx", "bbc", "cnn", "pes", "rdp"}


@lru_cache(maxsize=1)
def app_names() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in catalogue().values():
        for lab in c["labels"]:
            base = lab.split("_")[0].lower()
            if ((len(base) >= 4 or base in _SHORT_APP_NAMES) and (base not in _NOT_APP_NAMES or base in _CONTEXT_APPS)
                    and lab not in out.setdefault(base, [])):
                out[base].append(lab)
    return {k: v for k, v in out.items() if v}


_CONTEXT_APPS = {"zoom", "signal", "steam", "kick"}
_APP_CONTEXT = (r"(?<![a-z])((on|use|uses|used|using|via|over|of|through|opened|open|visited) {n}|{n} (users?|calls?|"
                r"meetings?|app|application|traffic|data|connections?|usage|messenger|video|games?|streams?|streaming|"
                r"sessions?|time|minutes|hours|volume))(?![a-z])")


def _app_context(name: str, text: str) -> bool:
    return name not in _CONTEXT_APPS or bool(re.search(_APP_CONTEXT.format(n=name), text))


_APP_FAMILY_NAMES = {"youtube": "YouTube", "whatsapp": "WhatsApp", "imo": "IMO", "xbox": "Xbox", "pubg": "PUBG",
                     "googlesearch": "Google Search", "googleplay": "Google Play", "appstore": "App Store",
                     "dailymotion": "Dailymotion", "amazoninstantvideo": "Amazon Prime Video", "bigo": "Bigo"}


@lru_cache(maxsize=1)
def app_families() -> dict[str, list[str]]:
    out = {}
    cat = catalogue()
    for base, labs in app_names().items():
        if base in cat:
            labs = cat[base]["labels"]
        if len(labs) > 1 and base not in ("webmail", "adobe"):
            out[_APP_FAMILY_NAMES.get(base) or next((x for x in labs if x.lower() == base), labs[0].split("_")[0])] = labs
    return out


def app_family_sql() -> str:
    parts = [f"app_label IN ({', '.join(repr(x) for x in labs)}), '{name}'" for name, labs in app_families().items()]
    return "multiIf(" + ", ".join(parts) + ", app_label)" if parts else "app_label"


_CALL_WORDS = re.compile(r"(?<![a-z])(calls?|calling|voice|video calls?|voip|audio|minutes of calls|call minutes)(?![a-z])")


def call_labels(text: str) -> dict[str, list[str]]:
    if not _CALL_WORDS.search((text or "").lower()):
        return {}
    out = {}
    for name in apps_in(text):
        voip = [lab for lab in app_names().get(name, []) if "voip" in lab.lower()]
        if voip:
            out[name] = voip
    return out


_APP_SPLIT_MIN = 6


@lru_cache(maxsize=1)
def _app_spellings() -> dict[str, str]:
    out = {}
    for name in app_names():
        if len(name) >= _APP_SPLIT_MIN:
            for i in range(3, len(name) - 2):
                out[f"{name[:i]} {name[i:]}"] = name
    return out


def apps_in(text: str) -> list[str]:
    return list(_apps_in(_norm(text)))


@lru_cache(maxsize=4096)
def _apps_in(t: str) -> tuple:
    import difflib

    found = [name for name in app_names() if f" {name} " in t and _app_context(name, t)]
    for spaced, name in _app_spellings().items():
        if name not in found and f" {spaced} " in t:
            found.append(name)
    long_names = [n for n in app_names() if len(n) >= 6]
    known = _known_words()
    for w in t.split():
        if len(w) < 6 or w in app_names() or w in known or any(w in n or n in w for n in found):
            continue
        hit = difflib.get_close_matches(w, long_names, n=1, cutoff=0.88)
        if hit and w in hit[0] and not hit[0].startswith(w):
            continue
        if hit and hit[0] not in found:
            found.append(hit[0])
    return tuple(n for n in app_names() if n in found)


def ranking_sql(key: str, start: str, end_exclusive: str, city: str | None = None, timezone: str = "Asia/Karachi",
                table: str = "wms_dist.ntc_ssl_log") -> str:
    c = catalogue()[key]
    name = c["key"].replace("_", " ")
    app = (f"if(app_label IN {sql_list(key)}, app_label, 'Other {name} sites')" if c["hosts"] else "app_label")
    match_ = f"(app_label IN {sql_list(key)}" + (f" OR match(requested_server, '{c['hosts']}'))" if c["hosts"] else ")")
    where_city = f"\n  AND client_location = '{city} Pakistan'" if city else ""
    return (f"SELECT {app} AS app,\n"
            f"       uniqCombined64If(subscriber_id, position(subscriber_id, '_') = 0) AS users,\n"
            f"       count() AS connections,\n"
            f"       round(sum(c2s_byte_num + s2c_byte_num) / 1e9, 2) AS volume_gb\n"
            f"FROM {table}\n"
            f"WHERE start_time >= toUnixTimestamp('{start} 00:00:00', '{timezone}')\n"
            f"  AND start_time <  toUnixTimestamp('{end_exclusive} 00:00:00', '{timezone}')\n"
            f"  AND {match_}{where_city}\n"
            f"GROUP BY app\nORDER BY users DESC, connections DESC\nLIMIT 20")


def total_sql(key: str, start: str, end_exclusive: str, city: str | None = None, timezone: str = "Asia/Karachi",
              table: str = "wms_dist.ntc_ssl_log") -> str:
    c = catalogue()[key]
    name = c["key"].replace("_", " ")
    match_ = f"(app_label IN {sql_list(key)}" + (f" OR match(requested_server, '{c['hosts']}'))" if c["hosts"] else ")")
    where_city = f"\n  AND client_location = '{city} Pakistan'" if city else ""
    return (f"SELECT '{name}' AS app_group,\n"
            f"       uniqCombined64If(subscriber_id, position(subscriber_id, '_') = 0) AS users,\n"
            f"       count() AS connections,\n"
            f"       round(sum(c2s_byte_num + s2c_byte_num) / 1e9, 2) AS volume_gb\n"
            f"FROM {table}\n"
            f"WHERE start_time >= toUnixTimestamp('{start} 00:00:00', '{timezone}')\n"
            f"  AND start_time <  toUnixTimestamp('{end_exclusive} 00:00:00', '{timezone}')\n"
            f"  AND {match_}{where_city}")


_COUNT_INTENT = re.compile(r"\b(how many|how much|kitne|kitni|kitna|count|number of|total|overall)\b")
_RANK_INTENT = re.compile(r"\b(which|top|most|popular|rank|ranking|best|leading|highest|biggest|zyada|ziada|kon si|konsi|kaun si|sb se|sab se|vs|versus|compare|each|per|by)\b")


def wants_total(text: str) -> bool:
    t = (text or "").lower()
    return bool(_COUNT_INTENT.search(t)) and not _RANK_INTENT.search(t)


def hint_for(text: str) -> str:
    cs = match(text)
    if not cs:
        return ""
    parts = [describe(c) for c in cs[:3]]
    return "This refers to a kind of app, not a name inside app_label. " + " ".join(parts)
