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
STAT_PATTERNS = [
    ("post_contact_metres", r"^post[\s-]*contact\s*met(re|er)s?\b"),
    ("kick_metres",         r"^kick(ing)?\s*met(re|er)s?\b"),
    ("run_metres",          r"^(all\s*run\s*|run(ning)?\s*)?met(re|er)s?\b|^run\s*met(re|er)s?\b"),
    ("tackle_breaks",       r"^tackle\s*(busts?|breaks?)\b"),
    ("tackles",             r"^tackles?\b"),
    ("runs",                r"^(runs|hit[\s-]*ups?|carries)\b"),
    ("line_breaks",         r"^line\s*breaks?\b"),
    ("offloads",            r"^offloads?\b"),
    ("fantasy",             r"^fantasy\b"),
    ("points",              r"^points\b"),
]
# stat -> model target column (None = informational / no model prediction)
STAT_TO_TARGET = {
    "tackles": "tackles", "run_metres": "runMetres",
    "post_contact_metres": "postContactMetres", "runs": "runs",
    "fantasy": "perf_points",  # approximate — books' fantasy != our perf_points formula
}

# A market title that begins with a 2-3 token proper name, then the stat phrase.
PLAYER_PROP_RE = re.compile(
    r"^(?P<player>[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){1,2})\s+(?P<rest>.+)$")
# Clean single try-scorer markets we keep as informational (1-way, player = selection).
# Anchored to avoid the many "… / Margin Double", "… (1st Half)" combination markets.
# Covers Ladbrokes ("Anytime/First Try Scorer") and Sportsbet ("1+ Try", "2+ Tries").
TRYSCORER_RE = re.compile(
    r"^(anytime|first) try ?scorer$|^player to score (2\+|3\+) tries$"
    r"|^\d\+\s*tr(y|ies)$", re.I)

# Leading tokens that mean the title is a team/match market, not a player.
NON_PLAYER_LEAD = {
    "total", "alternative", "match", "both", "either", "first", "last", "highest",
    "team", "home", "away", "winning", "half", "full", "exact", "most", "any",
    "race", "next", "anytime", "player", "1st", "2nd",
}

OVER_RE = re.compile(r"\bover\b", re.I)
UNDER_RE = re.compile(r"\bunder\b", re.I)
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
PLUS_RE = re.compile(r"(\d+)\s*\+")


def classify_player_prop(market_name, team_keys=()):
    """Return (player, stat) if the market is a clean player stat prop, else (None, None).

    team_keys: normalised team names for the event, used to reject team-total markets
    like "Parramatta Eels Total Points" that superficially look like a player title.
    """
    m = PLAYER_PROP_RE.match((market_name or "").strip())
    if not m:
        return None, None
    player = m.group("player").strip()
    if player.split()[0].lower() in NON_PLAYER_LEAD:
        return None, None
    pk = norm_team(player)
    if any(pk and (pk in t or t in pk) for t in team_keys if t):
        return None, None
    rest = m.group("rest").lower().strip()
    for key, pat in STAT_PATTERNS:
        if re.search(pat, rest):
            return player, key
    return None, None


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def norm_name(s):
    s = html.unescape(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def norm_team(s):
    """Loose team key so Sportsbet/Ladbrokes/ChampionData names align."""
    s = (s or "").lower()
    s = re.sub(r"\b(nrl|rugby league|fc)\b", "", s)
    # keep a distinctive token (last word usually the nickname)
    s = re.sub(r"[^a-z ]", "", s)
    toks = [t for t in s.split() if t not in
            {"the", "of", "and", "v", "vs"}]
    return " ".join(toks)


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

    df = pd.DataFrame(rows)
    if df.empty:
        print("No odds rows fetched (props may not be open yet).")
        df = pd.DataFrame(columns=["book", "event_name", "home", "away", "start_iso",
                                   "category", "stat", "player", "line", "over",
                                   "under", "single", "market_raw", "selection_raw",
                                   "fetched_at"])

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
