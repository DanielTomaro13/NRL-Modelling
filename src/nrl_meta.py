"""
Shared NRL metadata helpers used by the daily/6-hourly automation.

  - current_competition(): latest Champion Data men's NRL Premiership comp id
  - next_round(comp):      the next round with at least one non-complete match
  - fixture(comp):         the comp's fixture (list of match dicts)
  - round_matches(comp, r) round r's matches with squad names + UTC times
  - find_teamlist_url(r):  best-effort discovery of the nrl.com team-list article
  - squad_name_map(comp):  squadId -> display name (from the fixture)

Nothing here is leakage-relevant; it only resolves "which comp / round / page".
"""
import re, json, datetime as dt
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import tracks as T

BASE = "https://mc.championdata.com"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*"}
MENS_RE = re.compile(r"\bNRL (Premiership|Finals)\b", re.I)


def _get_json(url, retries=4):
    for i in range(retries):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            if i == retries - 1:
                raise
            import time
            time.sleep(1.5 * (i + 1))


def _get_text(url, retries=3):
    for i in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except (HTTPError, URLError, TimeoutError):
            if i == retries - 1:
                raise
            import time
            time.sleep(1.5 * (i + 1))


def competitions_for(track=None):
    """All of a track's competitions, oldest first (default = active TRACK)."""
    track = track or T.current()
    d = _get_json(f"{BASE}/data/competitions.json")
    comps = d["competitionDetails"]["competition"]
    return T.select_competitions(comps, track)


def mens_competitions():
    """All men's NRL Premiership/Finals comps, oldest first (back-compat helper)."""
    return competitions_for(T.TRACKS["nrl"])


def current_competition(track=None, prefer_regular=True):
    """Latest-season competition id for the track (regular season preferred over Finals).

    When the track defines a narrower `target_include` (Origin tracks ingest club
    history but predict only the Origin fixture), restrict to those comps.
    """
    track = track or T.current()
    comps = competitions_for(track)
    if track.target_include is not None:
        comps = [c for c in comps if track.target_include.search(c["name"])]
    if not comps:
        raise SystemExit(f"no competitions for track {track.name!r}")
    season = max(c["season"] for c in comps)
    cur = [c for c in comps if c["season"] == season]
    if prefer_regular:
        reg = [c for c in cur if "final" not in c["name"].lower()]
        cur = reg or cur
    chosen = cur[-1]
    return int(chosen["id"]), chosen


def fixture(comp):
    return _get_json(f"{BASE}/data/{comp}/fixture.json")["fixture"]["match"]


def next_round(comp, fx=None):
    """Next round number that still has a non-complete match; else the last round."""
    fx = fx if fx is not None else fixture(comp)
    by_round = {}
    for m in fx:
        by_round.setdefault(m["roundNumber"], []).append(m.get("matchStatus"))
    incomplete = sorted(r for r, st in by_round.items()
                        if any(s != "complete" for s in st))
    if incomplete:
        return incomplete[0]
    return max(by_round) if by_round else 1


def round_matches(comp, rnd, fx=None):
    fx = fx if fx is not None else fixture(comp)
    return [m for m in fx if m["roundNumber"] == rnd]


def squad_name_map(comp, fx=None):
    fx = fx if fx is not None else fixture(comp)
    nm = {}
    for m in fx:
        nm[m["homeSquadId"]] = m.get("homeSquadName")
        nm[m["awaySquadId"]] = m.get("awaySquadName")
    return nm


def find_teamlist_url(rnd, matches=None, season=None):
    """
    Best-effort discovery of the nrl.com 'NRL Team Lists - Round N' article URL.

    nrl.com publishes team lists at
        /news/<yyyy>/<mm>/<dd>/nrl-team-lists-round-<N>/
    The date varies (Tuesday of game week), so we scan the news landing page and
    a few candidate dates around the round's first kickoff. Returns the URL or
    None (caller then falls back to the most-recent-XVII proxy).
    """
    slug_re = re.compile(rf"/news/\d{{4}}/\d{{2}}/\d{{2}}/nrl-team-lists-round-{rnd}\b[\w-]*/")

    # 1) scan the news index for the published article
    for idx in ("https://www.nrl.com/news/", "https://www.nrl.com/news/?competition=111"):
        try:
            html = _get_text(idx)
        except Exception:
            continue
        m = slug_re.search(html)
        if m:
            return "https://www.nrl.com" + m.group(0)

    # 2) brute-force candidate dates: the Tue–Thu before the round's first match
    anchor = None
    if matches:
        try:
            anchor = min(dt.datetime.fromisoformat(m["utcStartTime"].replace("Z", "+00:00"))
                         for m in matches)
        except Exception:
            anchor = None
    if anchor is None:
        anchor = dt.datetime.now(dt.timezone.utc)
    for back in range(2, 8):  # team lists drop ~Tue, games Thu–Sun
        d = (anchor - dt.timedelta(days=back)).astimezone(
            dt.timezone(dt.timedelta(hours=10)))  # AEST
        url = (f"https://www.nrl.com/news/{d.year}/{d.month:02d}/{d.day:02d}/"
               f"nrl-team-lists-round-{rnd}/")
        try:
            html = _get_text(url)
            if "team-list-profile" in html:
                return url
        except Exception:
            continue
    return None


if __name__ == "__main__":
    comp, meta = current_competition()
    fx = fixture(comp)
    r = next_round(comp, fx)
    ms = round_matches(comp, r, fx)
    print(f"comp={comp} ({meta['name']})  next_round={r}  matches={len(ms)}")
    for m in ms:
        print(f"  R{r}: {m.get('homeSquadName')} v {m.get('awaySquadName')}  {m.get('utcStartTime')}")
    print("teamlist url:", find_teamlist_url(r, ms, meta.get("season")))
