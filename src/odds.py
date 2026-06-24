"""
Fetch NRL match + player-prop odds from Sportsbet and Ladbrokes/Neds public
JSON APIs, normalise to one schema, and match player markets to our playerIds.

Player props (tackles / run metres / tries / fantasy) only open ~1-2 days before
kickoff; this fetcher is defensive and simply returns whatever is live. Run it on
the 6-hourly schedule so edges appear as soon as the books post lines.

Output:
  reports/odds_snapshot.parquet   one row per (book, event, market, player, line)
  reports/odds_snapshot.json      same, for the static site

Schema per row:
  book, event_name, home, away, start_iso, category('player'|'match'),
  stat (canonical key), player, line, over, under, single,
  market_raw, selection_raw, fetched_at
"""
import re, json, time, html, unicodedata, datetime as dt
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

import nrl_meta as M

# -----------------------------------------------------------------------------
# canonical stat keys (align with model targets where possible)
# model targets: runsHitup, runs, runMetres, postContactMetres, tackles, perf_points
# Each entry: canonical key -> regex that the *stat phrase after the player name*
# must match. Player props are detected only when a market title is
# "<Proper Name> <stat phrase>", which excludes the thousands of team/match/try
# combination markets.
# Industry market names (Sportsbet / Ladbrokes / Dabble): "<Player> <stat phrase>",
# e.g. "Nicho Hynes Performance Points", "James Tedesco Most Metres", "Reece Walsh
# Run Metres", "Cody Walker Player Points", "Mitchell Moses Kicker Points".
# We locate the stat phrase anywhere in the title and take the player as the prefix.
# Order matters: specific phrases (performance/kicker) before the generic "points".
STAT_PHRASES = [
    ("performance_points", r"\bperf(ormance)?\s*p(oin)?ts?\b"),
    ("kicker_points",      r"\b(goal\s*)?kicker\s*p(oin)?ts?\b"),
    ("post_contact_metres", r"\bpost[\s-]*contact\s*met(re|er)s?\b"),
    ("kick_metres",        r"\bkick(ing)?\s*met(re|er)s?\b"),
    ("run_metres",         r"\b(run(ning)?|all\s*run)?\s*met(re|er)s?\b"),
    ("tackle_breaks",      r"\btackle\s*(busts?|breaks?)\b"),
    ("line_breaks",        r"\bline\s*breaks?\b"),
    ("tackles",            r"\btackles?\b"),
    ("runs",               r"\b(hit[\s-]*ups?|carries)\b"),
    ("offloads",           r"\boffloads?\b"),
    ("fantasy",            r"\bfantasy\b"),
    ("goals",              r"\bgoals?\b"),
    ("points",             r"\bp(oin)?ts?\b"),   # "Points", "Points Scored", "Pts"
]
# stat -> model target / pricing route
STAT_TO_TARGET = {
    "tackles": "tackles", "run_metres": "runMetres",
    "post_contact_metres": "postContactMetres", "runs": "runs",
    "performance_points": "perf_points", "fantasy": "perf_points",
}

# A market title that begins with a 2-3 token proper name, then the stat phrase.
PLAYER_PROP_RE = re.compile(
    r"^(?P<player>[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){1,2})\s+(?P<rest>.+)$")
# Clean single try-scorer markets we keep as informational (1-way, player = selection).
# Anchored to avoid the many "… / Margin Double", "… (1st Half)" combination markets.
# Covers Ladbrokes ("Anytime/First Try Scorer") and Sportsbet ("1+ Try", "2+ Tries").
TRYSCORER_RE = re.compile(
    r"^(anytime|first) try ?scorer$|^player to score (2\+|3\+) tries$"
    r"|^\d\+\s*tr(y|ies)$|^to score \d\+\s*tries?$|^to score \d or more tries?$"
    r"|^to score a hat[\s-]?trick$", re.I)

# Leading tokens that mean the title is a team/match market, not a player.
NON_PLAYER_LEAD = {
    "total", "alternative", "alternate", "match", "both", "either", "first", "last",
    "highest", "team", "home", "away", "winning", "half", "full", "exact", "most", "any",
    "race", "next", "anytime", "player", "1st", "2nd", "combined", "to",
}

OVER_RE = re.compile(r"\bover\b", re.I)
UNDER_RE = re.compile(r"\bunder\b", re.I)
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
PLUS_RE = re.compile(r"(\d+)\s*\+")


def _try_kind(market_raw):
    """anytime / first / 2+ / 3+ from a try market title (keeps first-try out of anytime)."""
    mr = (market_raw or "").lower()
    if "first" in mr:
        return "first"
    if "hat" in mr or "3+" in mr or "3 or more" in mr:
        return "3+"
    if "2+" in mr or "2 or more" in mr:
        return "2+"
    return "anytime"


def _looks_like_player(prefix, team_keys):
    """True if `prefix` looks like a player name (2-3 capitalised tokens, not a team)."""
    prefix = re.sub(r"\s+(most|total|player)$", "", prefix.strip(), flags=re.I)
    toks = prefix.split()
    if not (2 <= len(toks) <= 3):
        return False, prefix
    if toks[0].lower() in NON_PLAYER_LEAD:
        return False, prefix
    if not all(re.match(r"[A-Z][\w'’.\-]*$", t) for t in toks):
        return False, prefix
    pk = norm_team(prefix)
    if any(pk and (pk in t or t in pk) for t in team_keys if t):
        return False, prefix
    return True, prefix


def classify_player_prop(market_name, team_keys=()):
    """Return (player, stat) if the market is a clean player stat prop, else (None, None).

    Finds the stat phrase anywhere in the title and treats the text before it as the
    player; rejects team-total markets ("Parramatta Eels Total Points") and generic
    market names.
    """
    mn = (market_name or "").strip()
    low = mn.lower()
    for key, pat in STAT_PHRASES:
        m = re.search(pat, low)
        if not m:
            continue
        prefix = mn[:m.start()].strip(" -–:·")
        ok, player = _looks_like_player(prefix, team_keys)
        if ok:
            return player, key
    return None, None


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def norm_name(s):
    s = html.unescape(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


# Canonical NRL nicknames so every book's team naming aligns (Souths == South Sydney
# Rabbitohs == Rabbitohs). Order/keys chosen to avoid the "Sydney" collision between
# South Sydney (rabbitohs) and Sydney Roosters.
NICKNAMES = {
    "broncos": ["broncos", "brisbane"], "raiders": ["raiders", "canberra"],
    "bulldogs": ["bulldogs", "canterbury", "bankstown"], "sharks": ["sharks", "cronulla"],
    "dolphins": ["dolphins"], "titans": ["titans", "gold coast"],
    "sea eagles": ["sea eagles", "seaeagles", "manly"], "storm": ["storm", "melbourne"],
    "knights": ["knights", "newcastle"],
    "cowboys": ["cowboys", "north queensland", "nth queensland", "townsville"],
    "eels": ["eels", "parramatta", "parra"], "panthers": ["panthers", "penrith"],
    "rabbitohs": ["rabbitohs", "souths", "south sydney"],
    "dragons": ["dragons", "st george", "illawarra"],
    "roosters": ["roosters", "easts", "sydney roosters"],
    "warriors": ["warriors", "new zealand"],
    "tigers": ["tigers", "wests", "west tigers"],
}


def norm_team(s):
    """Canonical NRL nickname for any book's team string; '' if not a team."""
    low = (s or "").lower()
    for canon, keys in NICKNAMES.items():
        if any(k in low for k in keys):
            return canon
    return re.sub(r"[^a-z ]", "", low).strip()


def _get(url, headers, retries=3, timeout=40):
    last = None
    for i in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.0 * (i + 1))
    print(f"  [odds] giving up on {url[:80]}… ({last!r})")
    return None


def decimal(num, den):
    try:
        return round(1 + float(num) / float(den), 2)
    except Exception:
        return None


# ----------------------------------------------------------------------------- Sportsbet
SB = "https://www.sportsbet.com.au/apigw"
SB_HDR = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SB_NRL_COMP = 3436


def sportsbet_events():
    d = _get(f"{SB}/sportsbook-sports/Sportsbook/Sports/Competitions/{SB_NRL_COMP}"
             "?displayType=default&eventFilter=matches", SB_HDR)
    out = []
    if not d:
        return out
    for e in d.get("events", []):
        out.append({"eventId": e["id"], "name": e.get("name"),
                    "home": e.get("participant1"), "away": e.get("participant2"),
                    "start": e.get("startTime")})
    return out


def _sb_pair_overunder(market):
    """Return (line, over, under) if a market looks like a 2-way over/under."""
    sels = market.get("selections", [])
    over = under = line = None
    for s in sels:
        nm = s.get("name", "")
        price = (s.get("price") or {}).get("winPrice")
        num = NUM_RE.search(nm)
        if OVER_RE.search(nm):
            over = price
            if num: line = float(num.group(1))
        elif UNDER_RE.search(nm):
            under = price
            if num: line = float(num.group(1))
    return line, over, under


def sportsbet_event_rows(ev):
    d = _get(f"{SB}/sportsbook-sports/Sportsbook/Sports/Events/{ev['eventId']}/Markets", SB_HDR)
    rows = []
    if not isinstance(d, list):
        d = (d or {}).get("list") or []
    fetched = now_iso()
    team_keys = (norm_team(ev.get("home")), norm_team(ev.get("away")))
    for m in d:
        mname = m.get("name", "")
        player, stat = classify_player_prop(mname, team_keys)
        base = {"book": "sportsbet", "event_name": ev["name"], "home": ev["home"],
                "away": ev["away"], "start_iso": _sb_iso(ev.get("start")),
                "market_raw": mname, "fetched_at": fetched}
        if stat:  # player stat over/under prop
            line, over, under = _sb_pair_overunder(m)
            if over is None and under is None:
                continue
            rows.append({**base, "category": "player", "stat": stat, "player": player,
                         "line": line, "over": over, "under": under, "single": None,
                         "selection_raw": ""})
        elif TRYSCORER_RE.match(mname.strip()):  # clean anytime/first try scorer
            for s in m.get("selections", []):
                pr = (s.get("price") or {}).get("winPrice")
                snm = s.get("name", "")
                if pr is None or "no " in snm.lower():
                    continue
                rows.append({**base, "category": "player", "stat": "tries",
                             "kind": _try_kind(mname),
                             "player": _strip_team(snm), "line": _plus_line(mname),
                             "over": None, "under": None, "single": pr,
                             "selection_raw": s.get("name", "")})
    return rows


def _sb_iso(epoch):
    try:
        return dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc).replace(
            microsecond=0).isoformat()
    except Exception:
        return None


def _plus_line(name):
    m = PLUS_RE.search(name or "")
    return float(m.group(1)) - 0.5 if m else None


def fetch_sportsbet():
    rows = []
    evs = sportsbet_events()
    print(f"  [sportsbet] {len(evs)} NRL events")
    for ev in evs:
        r = sportsbet_event_rows(ev)
        rows.extend(r)
    print(f"  [sportsbet] {len(rows)} market rows")
    return rows


# ----------------------------------------------------------------------------- Ladbrokes / Neds
LAD = "https://api.ladbrokes.com.au"
LAD_HDR = {"User-Agent": "Mozilla/5.0", "Origin": "https://www.ladbrokes.com.au",
           "Referer": "https://www.ladbrokes.com.au/", "Content-Type": "application/json"}
RUGBY_LEAGUE_CAT = "608a1803-45bc-465a-8471-c89dcb68a27d"


def _lad_decimal(price):
    if not price:
        return None
    odds = price.get("odds") or {}
    if "decimal" in odds:
        return round(float(odds["decimal"]), 2)
    return decimal(odds.get("numerator"), odds.get("denominator"))


def ladbrokes_events():
    """Enumerate NRL events (uuid + names) via the hash-free event-request endpoint."""
    u = f'{LAD}/v2/sport/event-request?category_ids=["{RUGBY_LEAGUE_CAT}"]'
    d = _get(u, LAD_HDR)
    out = []
    if not d:
        return out
    parts = d.get("event_participants", {})
    for eid, e in d.get("events", {}).items():
        nm = e.get("name", "")
        if " vs " not in nm and " v " not in nm:
            continue  # skip futures/outrights
        # only AU NRL: heuristic on competition or known team names
        out.append({"eventId": eid, "name": nm,
                    "start": e.get("event_start") or e.get("advertised_start"),
                    "competition_id": e.get("competition_id")})
    # keep the NRL competition (the one most events share) — filter to the modal comp
    if out:
        from collections import Counter
        modal = Counter(o["competition_id"] for o in out).most_common(1)[0][0]
        out = [o for o in out if o["competition_id"] == modal]
    return out


def ladbrokes_event_rows(ev):
    d = _get(f"{LAD}/v2/sport/event-card?id={ev['eventId']}", LAD_HDR)
    rows = []
    if not d:
        return rows
    markets = d.get("markets", {})
    entrants = d.get("entrants", {})
    prices = d.get("prices", {})
    fetched = now_iso()
    home = away = None
    ev_name = ev["name"]
    if " vs " in ev_name:
        home, away = ev_name.split(" vs ", 1)
    elif " v " in ev_name:
        home, away = ev_name.split(" v ", 1)

    def entrant_price(ent_id):
        for k, v in prices.items():
            if k.startswith(ent_id + ":"):
                return _lad_decimal(v)
        return None

    # group entrants by market
    by_market = {}
    for ent in entrants.values():
        by_market.setdefault(ent.get("market_id"), []).append(ent)

    team_keys = (norm_team(home), norm_team(away))
    for mid, market in markets.items():
        mname = market.get("name", "")
        # Ladbrokes player props are titled "<Player> - <Stat>"; normalise the dash.
        mtitle = mname.replace(" - ", " ")
        player, stat = classify_player_prop(mtitle, team_keys)
        base = {"book": "ladbrokes", "event_name": ev_name, "home": home, "away": away,
                "start_iso": _lad_iso(ev.get("start")), "market_raw": mname,
                "fetched_at": fetched}
        ents = by_market.get(mid, [])
        if stat:
            over = under = line = None
            for e in ents:
                enm = e.get("name", "")
                pr = entrant_price(e["id"])
                num = NUM_RE.search(enm) or NUM_RE.search(mname)
                if OVER_RE.search(enm):
                    over = pr
                    if num: line = float(num.group(1))
                elif UNDER_RE.search(enm):
                    under = pr
                    if num: line = float(num.group(1))
            if over is None and under is None:
                continue
            rows.append({**base, "category": "player", "stat": stat, "player": player,
                         "line": line, "over": over, "under": under, "single": None,
                         "selection_raw": ""})
        elif TRYSCORER_RE.match(mname.strip()):
            for e in ents:
                pr = entrant_price(e["id"])
                enm = e.get("name", "")
                if pr is None or "no scorer" in enm.lower() or "no try" in enm.lower():
                    continue
                rows.append({**base, "category": "player", "stat": "tries",
                             "kind": _try_kind(mname),
                             "player": _strip_team(enm), "line": _plus_line(mname),
                             "over": None, "under": None, "single": pr,
                             "selection_raw": enm})
    return rows


def _strip_team(name):
    """'Alex Johnston (South Sydney Rabbitohs)' -> 'Alex Johnston'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name or "").strip()


def _lad_iso(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(
            microsecond=0).isoformat()
    except Exception:
        return None


def fetch_ladbrokes():
    rows = []
    evs = ladbrokes_events()
    print(f"  [ladbrokes] {len(evs)} NRL events")
    for ev in evs:
        rows.extend(ladbrokes_event_rows(ev))
    print(f"  [ladbrokes] {len(rows)} market rows")
    return rows


# ----------------------------------------------------------------------------- Dabble
# Dabble is the native iOS app's backend (Cloudflare-fronted). It authenticates with
# a Cognito Bearer token — capture it from the app with Charles and pass it in env:
#   DABBLE_AUTH  -> "Bearer eyJ..."   (required)
#   DABBLE_DEVICE_ID, DABBLE_UA, DABBLE_COOKIE  (optional; sensible defaults below)
# Tokens are short-lived (~1h); refresh by re-capturing. curl_cffi safari_ios TLS.
DAB = "https://api.dabble.com.au"
DAB_RUGBY_SPORT = "Rugby League"


def _dab_session():
    try:
        from curl_cffi import requests as creq
    except Exception:
        print("  [dabble] curl_cffi not installed — skipping")
        return None, None
    import os
    auth = os.environ.get("DABBLE_AUTH", "").strip()
    if not auth:
        print("  [dabble] no DABBLE_AUTH (Bearer token) — skipping. Capture it from the "
              "iOS app with Charles (see README).")
        return None, None
    if not auth.lower().startswith("bearer "):
        auth = "Bearer " + auth
    headers = {"authorization": auth, "accept": "application/json",
               "x-device-id": os.environ.get("DABBLE_DEVICE_ID", "00000000-0000-0000-0000-000000000000"),
               "user-agent": os.environ.get("DABBLE_UA", "Dabble/1000041710 CFNetwork/3826.600.41.2.1 Darwin/24.6.0"),
               "x-app-version": os.environ.get("DABBLE_APP_VERSION", "4.17.10+019ededb"),
               "accept-language": "en-AU,en;q=0.9"}
    if os.environ.get("DABBLE_COOKIE", "").strip():
        headers["cookie"] = os.environ["DABBLE_COOKIE"].strip()
    return creq, headers


def _dab_get(creq, headers, path):
    try:
        r = creq.get(DAB + path, headers=headers, impersonate="safari_ios", timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"  [dabble] {path[:60]}… HTTP {r.status_code}")
    except Exception as e:
        print(f"  [dabble] {path[:50]}… {e!r}")
    return None


def _dab_nrl_competition(creq, headers):
    d = _dab_get(creq, headers, "/competitions")
    comps = (d.get("data", d) if isinstance(d, dict) else d) or []
    rl = [c for c in comps if "rugby league" in str(c.get("sportName", "")).lower()] or comps
    for c in comps:
        if str(c.get("name", "")).strip().lower() == "nrl":
            return c
    return None


def dabble_fixture_rows(creq, headers, fixture):
    """Pull full markets for one fixture via the details endpoint and parse them."""
    fid = fixture.get("id")
    detail = _dab_get(creq, headers, f"/frontend-api/sport-fixtures/details/{fid}")
    if not detail:
        return []
    sfd = detail.get("sportFixtureDetail") or detail.get("data", {}).get("sportFixtureDetail") or {}
    markets = sfd.get("markets") or []
    sel_name = {s["id"]: s.get("name", "") for s in sfd.get("selections", [])}
    price_by_mkt = {}
    for p in sfd.get("prices", []):
        price_by_mkt.setdefault(p.get("marketId"), []).append(
            (sel_name.get(p.get("selectionId")), p.get("price")))
    teams = [t.get("name") for t in (sfd.get("teams") or fixture.get("teams") or [])]
    name = fixture.get("name", sfd.get("name", ""))
    home = away = None
    if " v " in name:
        home, away = name.split(" v ", 1)
    elif len(teams) == 2:
        home, away = teams[0], teams[1]
    team_keys = (norm_team(home), norm_team(away))
    fetched = now_iso()
    rows = []
    for m in markets:
        # Dabble runs a Pick'em (multiplier/parlay) product alongside its sportsbook —
        # "pickem_*" markets are flat even-money picks, NOT traditional fixed odds, so
        # exclude them from the odds/EV comparison. Same for same-game-multi bundles.
        rt = (m.get("resultingType") or "").lower()
        if rt.startswith("pickem") or rt == "player_sgm":
            continue
        mname = (m.get("name") or "").strip()
        outs = [(nm, pr) for nm, pr in price_by_mkt.get(m.get("id"), []) if nm and pr]
        base = {"book": "dabble", "event_name": name, "home": home, "away": away,
                "start_iso": fixture.get("advertisedStart"), "market_raw": mname,
                "fetched_at": fetched}
        # try-scorer markets (selections are players)
        if TRYSCORER_RE.match(mname):
            for nm, pr in outs:
                if "no " in nm.lower():
                    continue
                rows.append({**base, "category": "player", "stat": "tries",
                             "kind": _try_kind(mname), "player": _strip_team(nm),
                             "line": None, "over": None, "under": None,
                             "single": float(pr), "selection_raw": nm})
            continue
        # player stat over/under markets ("<Player> points 7.5", "<Player> Tackles", ...)
        player, stat = classify_player_prop(mname, team_keys)
        if stat:
            over = under = line = None
            mnum = NUM_RE.search(mname)
            for nm, pr in outs:
                num = NUM_RE.search(nm) or mnum
                if OVER_RE.search(nm):
                    over = float(pr); line = float(num.group(1)) if num else line
                elif UNDER_RE.search(nm):
                    under = float(pr); line = float(num.group(1)) if num else line
            if over is not None or under is not None:
                rows.append({**base, "category": "player", "stat": stat, "kind": None,
                             "player": player, "line": line, "over": over, "under": under,
                             "single": None, "selection_raw": ""})
    return rows


def fetch_dabble():
    creq, headers = _dab_session()
    if creq is None:
        return []
    comp = _dab_nrl_competition(creq, headers)
    if not comp:
        print("  [dabble] NRL competition not found (token expired?)")
        return []
    fx = _dab_get(creq, headers,
                  f"/frontend-api/competitions/{comp['id']}/sport-fixtures"
                  "?includeInPlay=false&exclude%5B%5D=none")
    flist = (fx.get("data", fx) if isinstance(fx, dict) else fx) or []
    rows = []
    for fixture in flist:
        if fixture.get("id"):
            rows.extend(dabble_fixture_rows(creq, headers, fixture))
    print(f"  [dabble] {len(flist)} fixtures, {len(rows)} market rows")
    return rows


# ----------------------------------------------------------------------------- PointsBet
PB_V2 = "https://api.au.pointsbet.com/api/v2"
PB_MES = "https://api.au.pointsbet.com/api/mes/v3"
PB_HDR = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
          "Origin": "https://pointsbet.com.au"}
PB_TRY = {"anytime tryscorer": "anytime", "to score 2+ tries": "2+",
          "to score 3+ tries": "3+", "first tryscorer": "first"}


def _cffi_get(url, headers, params=None):
    try:
        from curl_cffi import requests as creq
    except Exception:
        return None
    try:
        r = creq.get(url, headers=headers, params=params, impersonate="chrome", timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"  [http] {url[:70]}… {e!r}")
        return None


def pointsbet_nrl_key():
    d = _cffi_get(f"{PB_V2}/sports/list/", PB_HDR)
    if not d:
        return None
    sports = d.get("sports", d) if isinstance(d, dict) else d
    for s in sports:
        if "rugby league" in str(s.get("name", "")).lower():
            for c in s.get("competitions", []):
                if str(c.get("name", "")).strip().lower() == "nrl":
                    return c.get("key") or c.get("competitionKey") or c.get("id")
    return None


def fetch_pointsbet():
    key = pointsbet_nrl_key()
    if not key:
        print("  [pointsbet] NRL competition not found")
        return []
    feat = _cffi_get(f"{PB_MES}/events/featured/competition/{key}", PB_HDR)
    evs = (feat.get("events", []) if isinstance(feat, dict) else feat) or []
    rows, fetched = [], now_iso()
    for ev in evs:
        eid = ev.get("key") or ev.get("eventId") or ev.get("id")
        home, away = ev.get("homeTeam"), ev.get("awayTeam")
        name = ev.get("name") or f"{home} v {away}"
        team_keys = (norm_team(home), norm_team(away))
        det = _cffi_get(f"{PB_MES}/events/{eid}", PB_HDR)
        if not det:
            continue
        markets = det.get("fixedOddsMarkets") or det.get("markets") or []
        base0 = {"book": "pointsbet", "event_name": name, "home": home, "away": away,
                 "start_iso": ev.get("startsAt"), "fetched_at": fetched}
        for m in markets:
            mname = re.sub(r"\s*\([^)]*\)\s*$", "", m.get("name", "")).strip()
            outs = m.get("outcomes") or []
            kind = PB_TRY.get(mname.lower())
            base = {**base0, "market_raw": m.get("name", "")}
            if kind:
                for o in outs:
                    pr = o.get("price")
                    nm = o.get("name", "")
                    if not pr or "no " in nm.lower():
                        continue
                    rows.append({**base, "category": "player", "stat": "tries", "kind": kind,
                                 "player": _strip_team(nm), "line": _plus_line(mname),
                                 "over": None, "under": None, "single": float(pr),
                                 "selection_raw": nm})
                continue
            player, stat = classify_player_prop(mname, team_keys)
            if stat:  # player O/U stat market
                over = under = line = None
                for o in outs:
                    nm, pr = o.get("name", ""), o.get("price")
                    ln = o.get("points")
                    if pr is None:
                        continue
                    if OVER_RE.search(nm):
                        over = float(pr); line = float(ln) if ln is not None else line
                    elif UNDER_RE.search(nm):
                        under = float(pr); line = float(ln) if ln is not None else line
                if over is not None or under is not None:
                    rows.append({**base, "category": "player", "stat": stat, "kind": None,
                                 "player": player, "line": line, "over": over, "under": under,
                                 "single": None, "selection_raw": ""})
    print(f"  [pointsbet] {len(evs)} NRL events, {len(rows)} market rows")
    return rows


# ----------------------------------------------------------------------------- TAB
TAB_BASE = "https://api.beta.tab.com.au/v1/tab-info-service"
TAB_TOKEN_URL = "https://api.beta.tab.com.au/oauth/token"
TAB_TRY = {"to score a try": "anytime", "to score 2+ tries": "2+",
           "to score a hat trick": "3+", "1st try scorer": "first",
           "last try scorer": "last"}


def _tab_token():
    """Get a TAB access token. Prefer minting a FRESH one via the client_credentials
    grant (so it never expires between runs); fall back to a static TAB_ACCESS_TOKEN."""
    import os
    cid = os.environ.get("TAB_CLIENT_ID", "").strip()
    csec = os.environ.get("TAB_CLIENT_SECRET", "").strip()
    if cid and csec:
        try:
            from curl_cffi import requests as creq
            r = creq.post(TAB_TOKEN_URL,
                          data={"grant_type": "client_credentials",
                                "client_id": cid, "client_secret": csec},
                          headers={"Accept": "application/json"},
                          impersonate="chrome", timeout=15)
            if r.status_code == 200 and r.json().get("access_token"):
                print(f"  [tab] minted fresh token (expires in "
                      f"{r.json().get('expires_in', '?')}s)")
                return r.json()["access_token"]
            print(f"  [tab] token refresh failed: HTTP {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(f"  [tab] token refresh error: {e!r}")
    tok = os.environ.get("TAB_ACCESS_TOKEN", "").strip()
    if tok:
        return tok
    return None


def fetch_tab():
    tok = _tab_token()
    if not tok:
        print("  [tab] no TAB_ACCESS_TOKEN / TAB_CLIENT_ID+SECRET — skipping")
        return []
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/json",
           "User-Agent": "Mozilla/5.0"}
    d = _cffi_get(f"{TAB_BASE}/sports/Rugby%20League/competitions/NRL"
                  "?jurisdiction=VIC&homeState=VIC", hdr)
    if not d:
        print("  [tab] NRL competition fetch failed (token expired?)")
        return []
    rows, fetched = [], now_iso()
    for match in d.get("matches", []):
        cons = match.get("contestants") or []
        home = next((c["name"] for c in cons if c.get("isHome")), None)
        away = next((c["name"] for c in cons if not c.get("isHome")), None)
        name = match.get("name", f"{home} v {away}")
        for mk in match.get("markets", []):
            bo = (mk.get("betOption") or "").strip()
            props = mk.get("propositions", [])
            base = {"book": "tab", "event_name": name, "home": home, "away": away,
                    "start_iso": match.get("startTime"), "market_raw": bo, "fetched_at": fetched}
            kind = TAB_TRY.get(bo.lower())
            if kind in ("anytime", "2+", "3+"):
                for p in props:
                    pr = p.get("returnWin")
                    nm = p.get("name", "")
                    if not pr or "no try" in nm.lower():
                        continue
                    rows.append({**base, "category": "player", "stat": "tries", "kind": kind,
                                 "player": _strip_team(nm), "line": None, "over": None,
                                 "under": None, "single": float(pr), "selection_raw": nm})
    print(f"  [tab] {len(d.get('matches', []))} matches, {len(rows)} market rows")
    return rows


# ----------------------------------------------------------------------------- matching
def split_name(s):
    """Return (first_initial, surname_key). Handles 'D.Cherry-Evans' and 'Daly Cherry-Evans'."""
    s = html.unescape(s or "").strip()
    if not s:
        return "", ""
    if "." in s and " " not in s.split(".")[0]:
        init, sur = s.split(".", 1)
        return init[:1].lower(), norm_name(sur)
    toks = s.split()
    if len(toks) == 1:
        return "", norm_name(toks[0])
    return toks[0][:1].lower(), norm_name(toks[-1])


def attach_player_ids(odds_df, preds_df):
    """Match player-prop rows to our predicted playerIds by (initial, surname) + team.

    Our predictions carry Champion-Data display names ('D.Cherry-Evans'); the books
    use full names ('Daly Cherry-Evans'). We key on first-initial + surname and
    disambiguate same-surname collisions by the event's two teams.
    """
    if odds_df.empty:
        odds_df = odds_df.copy()
        odds_df["playerId"] = pd.Series(dtype="object")
        odds_df["matched_team"] = pd.Series(dtype="object")
        return odds_df
    lut = {}  # (initial, surname) -> [(playerId, team, norm_team)]
    for _, p in preds_df.iterrows():
        nm = p.get("name")
        if not isinstance(nm, str) or not nm.strip():
            continue
        lut.setdefault(split_name(nm), []).append(
            (p["playerId"], p.get("team"), norm_team(p.get("team"))))

    pids, teams = [], []
    for _, r in odds_df.iterrows():
        pid = team = None
        pname = r.get("player")
        if isinstance(pname, str) and pname.strip():
            key = split_name(pname)
            cands = lut.get(key, [])
            if not cands:  # surname-only fallback (different initial styling)
                cands = [c for k, v in lut.items() if k[1] == key[1] for c in v]
            ev_teams = [t for t in (norm_team(r.get("home")), norm_team(r.get("away"))) if t]
            # require team agreement with one of the event's two teams
            team_ok = [c for c in cands
                       if c[2] and any(c[2] in t or t in c[2] for t in ev_teams)]
            if len(team_ok) >= 1:
                pid, team, _ = team_ok[0]
            elif len(cands) == 1 and not ev_teams:
                pid, team, _ = cands[0]  # no team context to check against
        pids.append(pid)
        teams.append(team)
    odds_df = odds_df.copy()
    odds_df["playerId"] = pids
    odds_df["matched_team"] = teams
    return odds_df


def main():
    rows = []
    try:
        rows += fetch_sportsbet()
    except Exception as e:
        print("  [sportsbet] ERROR", repr(e))
    try:
        rows += fetch_ladbrokes()
    except Exception as e:
        print("  [ladbrokes] ERROR", repr(e))
    try:
        rows += fetch_dabble()
    except Exception as e:
        print("  [dabble] ERROR", repr(e))
    try:
        rows += fetch_pointsbet()
    except Exception as e:
        print("  [pointsbet] ERROR", repr(e))
    try:
        rows += fetch_tab()
    except Exception as e:
        print("  [tab] ERROR", repr(e))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No odds rows fetched (props may not be open yet).")
        df = pd.DataFrame(columns=["book", "event_name", "home", "away", "start_iso",
                                   "category", "stat", "kind", "player", "line", "over",
                                   "under", "single", "market_raw", "selection_raw",
                                   "fetched_at"])
    if "kind" not in df.columns:
        df["kind"] = None

    try:
        preds = pd.read_parquet("reports/round_predictions.parquet")
        df = attach_player_ids(df, preds)
    except Exception as e:
        print("  [match] skipped:", repr(e))
        df["playerId"] = None
        df["matched_team"] = None

    df.to_parquet("reports/odds_snapshot.parquet", index=False)
    df.to_json("reports/odds_snapshot.json", orient="records")
    n_player = int((df["category"] == "player").sum()) if len(df) else 0
    n_matched = int(df["playerId"].notna().sum()) if "playerId" in df else 0
    print(f"\nWrote reports/odds_snapshot.* : {len(df)} rows "
          f"({n_player} player markets, {n_matched} matched to playerIds)")
    if len(df):
        print("by book/stat:\n", df.groupby(["book", "stat"]).size().to_string())


if __name__ == "__main__":
    main()
