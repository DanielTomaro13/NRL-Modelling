"""
Scrape official NRL team lists from nrl.com and turn them into a lineup table
(confirmed positions + jerseys) that predict.py can use instead of the
'most-recent XVII' proxy.

Usage:
  python src/scrape_teamlists.py <url> <competitionId> <round>
  e.g. python src/scrape_teamlists.py \
       https://www.nrl.com/news/2026/06/16/nrl-team-lists-round-16/ 12999 16

Output: data/processed/lineups_r{round}.parquet
        columns: matchId, squadId, oppSquadId, isHome, playerId, name,
                 position, jumperNumber, jersey
"""
import sys, re, json, glob, html as htmllib, unicodedata, urllib.request
import pandas as pd


def norm(s):
    """lowercase, strip accents/apostrophes/hyphens/spaces for robust name matching."""
    s = htmllib.unescape(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())

# nrl.com position label -> our dataset 'position' value
POS_MAP = {
    "Fullback": "Fullback", "Winger": "Wing", "Centre": "Centre",
    "Five-Eighth": "Five-Eighth", "Halfback": "Halfback", "Hooker": "Hooker",
    "Prop": "Prop", "2nd Row": "Second Row", "Lock": "Lock",
    "Interchange": "Interchange", "Reserve": "Interchange",
}

PLAYER_RE = re.compile(
    r'team-list-profile--(?P<side>home|away)"[^>]*>\s*'
    r'<div class="team-list-profile__name">\s*'
    r'<span class="u-visually-hidden">\s*(?P<pos>.*?)\s+for\s+(?P<team>.*?)\s+is number\s+(?P<num>\d+)\s*</span>\s*'
    r'(?P<first>[^<]*?)\s*<span[^>]*>\s*(?P<sur>.*?)\s*</span>',
    re.S)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def squad_lookup(comp):
    """nickname/name -> squadId, and round match context, from the fixture."""
    fx = json.loads(fetch(f"https://mc.championdata.com/data/{comp}/fixture.json"))["fixture"]["match"]
    nick2id, ctx = {}, {}
    for m in fx:
        for side in ("home", "away"):
            sid = m[f"{side}SquadId"]
            for key in (m.get(f"{side}SquadNickname"), m.get(f"{side}SquadName")):
                if key:
                    nick2id[key.lower()] = sid
    return nick2id, fx


def round_context(fx, rnd):
    """squadId -> (matchId, oppSquadId, isHome) for a given round."""
    ctx = {}
    for m in fx:
        if m["roundNumber"] != rnd:
            continue
        ctx[m["homeSquadId"]] = (m["matchId"], m["awaySquadId"], 1)
        ctx[m["awaySquadId"]] = (m["matchId"], m["homeSquadId"], 0)
    return ctx


def player_lookup():
    """(surname_lower, squadId) -> list[(playerId, firstname_lower)] from history."""
    info = {}
    for fp in glob.glob("data/raw/*/*.json"):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for p in d.get("matchStats", {}).get("playerInfo", {}).get("player", []):
            info[p["playerId"]] = (p.get("firstname", ""), p.get("surname", ""),
                                   p.get("displayName", ""))
    pm = pd.read_parquet("data/processed/player_match.parquet")
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    latest = pm.sort_values("utcStartTime").groupby("playerId").tail(1).set_index("playerId")["squadId"]
    by_sur_sq, by_sur = {}, {}
    for pid, (fn, sur, disp) in info.items():
        sq = latest.get(pid)
        by_sur.setdefault(norm(sur), []).append((pid, norm(fn), sq))
        if sq is not None:
            by_sur_sq.setdefault((norm(sur), int(sq)), []).append((pid, norm(fn), disp))
    return by_sur_sq, by_sur, info


def resolve(first, sur, squad_id, by_sur_sq, by_sur):
    first, surl = norm(first), norm(sur)
    cands = by_sur_sq.get((surl, squad_id), [])
    if cands:
        exact = [c for c in cands if c[1] and (c[1] == first or c[1][0] == first[:1])]
        return (exact or cands)[0][0]
    # fallback: surname anywhere (e.g. new signing with no history at this club)
    cands = by_sur.get(surl, [])
    exact = [c for c in cands if c[1] and c[1][0] == first[:1]]
    return (exact or cands)[0][0] if (exact or cands) else None


def scrape(url, comp, rnd):
    html = fetch(url)
    nick2id, fx = squad_lookup(comp)
    ctx = round_context(fx, rnd)
    by_sur_sq, by_sur, info = player_lookup()

    rows, unmapped = [], []
    for m in PLAYER_RE.finditer(html):
        g = m.groupdict()
        team = g["team"].strip()
        squad_id = nick2id.get(team.lower())
        if squad_id is None or squad_id not in ctx:
            continue
        pid = resolve(g["first"], g["sur"], squad_id, by_sur_sq, by_sur)
        jersey = int(g["num"])
        if pid is None:
            unmapped.append((g["first"], g["sur"], team, jersey))
            continue
        matchId, opp, home = ctx[squad_id]
        rows.append({
            "matchId": matchId, "squadId": squad_id, "oppSquadId": opp, "isHome": home,
            "playerId": pid,
            "name": htmllib.unescape(f"{g['first']} {g['sur']}").strip(),
            "position": POS_MAP.get(g["pos"], "Interchange"),
            "jumperNumber": jersey, "jersey": jersey,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["matchId", "playerId"])
    return df, unmapped


def main():
    url = sys.argv[1]
    comp = int(sys.argv[2])
    rnd = int(sys.argv[3])
    df, unmapped = scrape(url, comp, rnd)
    out = f"data/processed/lineups_r{rnd}.parquet"
    df.to_parquet(out, index=False)
    print(f"Scraped {len(df)} players across {df['matchId'].nunique()} matches -> {out}")
    print("players per squad:\n", df.groupby("squadId").size().to_string())
    if unmapped:
        print(f"\n{len(unmapped)} unmapped (no history match):")
        for u in unmapped[:30]:
            print("   ", u)


if __name__ == "__main__":
    main()
