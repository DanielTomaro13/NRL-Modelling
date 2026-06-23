"""
Ingest NRL men's Premiership + Finals data from the Champion Data match-centre feed.

Pipeline:
  competitions.json  -> select men's NRL Premiership/Finals competitions
  {comp}/fixture.json -> match ids + context (round, venue, home/away, date)
  {comp}/{match}.json -> per-player stat lines  (cached to data/raw/)

Outputs:
  data/processed/player_match.parquet  (one row per player per match, raw stats + context)
"""
import json, os, re, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

BASE = "https://mc.championdata.com"
RAW = "data/raw"
OUT = "data/processed/player_match.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*"}

# men's NRL Premiership or Finals, exclude NRLW
MENS_RE = re.compile(r"\bNRL (Premiership|Finals)\b", re.I)

# raw player stat fields we keep
STAT_FIELDS = [
    "runsHitup", "runs", "runMetres", "postContactMetres", "tackles",          # targets-ish
    "points", "tryAssists", "lineBreaks",                                       # perf-points inputs
    "possessions", "metresGained", "tackleBreaks", "missedTackles", "offloads",
    "passes", "runsHitupMetres", "tackleds", "tacklesIneffective",
    "kicksGeneralPlay", "kickMetres", "errors", "handlingErrors",
    "runsDummyHalf", "runsKickReturn", "runsNormal", "tries", "kicksCaught",
    # goal-kicking (for the kicker-points + player-points models)
    "conversions", "conversionAttempts", "penaltyGoals", "penaltyGoalAttempts",
    "fieldGoals",
]


def fetch_json(url, retries=4):
    for i in range(retries):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def select_competitions():
    d = fetch_json(f"{BASE}/data/competitions.json")
    comps = d["competitionDetails"]["competition"]
    mens = [c for c in comps if MENS_RE.search(c["name"]) and "NRLW" not in c["name"]]
    mens.sort(key=lambda c: (c["season"], c["id"]))
    return mens


def cache_match(comp_id, match_id):
    path = f"{RAW}/{comp_id}/{match_id}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = fetch_json(f"{BASE}/data/{comp_id}/{match_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def parse_match(comp, match_meta, mj):
    ms = mj.get("matchStats", {})
    info = ms.get("matchInfo", {})
    if info.get("matchStatus") != "complete":
        return []
    players = ms.get("playerStats", {}).get("player", [])
    if not players:
        return []
    home, away = info.get("homeSquadId"), info.get("awaySquadId")

    # per-period presence (minutes proxy): did the player record possessions in each half?
    pres = {}  # playerId -> set of periods with activity
    for pp in ms.get("playerPeriodStats", {}).get("player", []):
        if (pp.get("possessions", 0) or pp.get("tackles", 0) or pp.get("tackleds", 0)):
            pres.setdefault(pp["playerId"], set()).add(pp.get("period"))

    rows = []
    for p in players:
        pid = p["playerId"]
        sq = p["squadId"]
        activity = p.get("possessions", 0) + p.get("tackles", 0) + p.get("tackleds", 0)
        if activity == 0:
            continue  # DNP / unused interchange
        opp = away if sq == home else home
        row = {
            "season": comp["season"], "competitionId": comp["id"], "compName": comp["name"],
            "matchId": info.get("matchId"), "roundNumber": info.get("roundNumber"),
            "utcStartTime": match_meta.get("utcStartTime"),
            "venueId": info.get("venueId"),
            "playerId": pid, "squadId": sq, "oppSquadId": opp,
            "isHome": int(sq == home),
            "position": p.get("position"), "jumperNumber": p.get("jumperNumber"),
            "p1": int(1 in pres.get(pid, set())), "p2": int(2 in pres.get(pid, set())),
            "activity": activity,
        }
        for f in STAT_FIELDS:
            row[f] = p.get(f, 0)
        rows.append(row)
    return rows


def main():
    comps = select_competitions()
    print(f"Selected {len(comps)} men's NRL competitions "
          f"({comps[0]['season']}-{comps[-1]['season']})", flush=True)

    # gather (comp, match_meta) for all matches in all comps
    tasks = []
    for c in comps:
        fx = fetch_json(f"{BASE}/data/{c['id']}/fixture.json")
        for m in fx.get("fixture", {}).get("match", []):
            tasks.append((c, m))
    print(f"{len(tasks)} matches listed across fixtures", flush=True)

    all_rows = []
    done = 0
    def work(task):
        c, m = task
        try:
            mj = cache_match(c["id"], m["matchId"])
            return parse_match(c, m, mj)
        except Exception as e:
            return ("ERR", c["id"], m.get("matchId"), repr(e))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fu in as_completed(futs):
            res = fu.result()
            done += 1
            if isinstance(res, tuple) and res and res[0] == "ERR":
                print("  err", res[1:], flush=True)
            else:
                all_rows.extend(res)
            if done % 200 == 0:
                print(f"  processed {done}/{len(tasks)} matches, {len(all_rows)} player-rows", flush=True)

    df = pd.DataFrame(all_rows)
    df["utcStartTime"] = pd.to_datetime(df["utcStartTime"], errors="coerce", utc=True)
    df = df.sort_values(["utcStartTime", "matchId", "squadId", "jumperNumber"]).reset_index(drop=True)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}: {len(df)} rows, {df['playerId'].nunique()} players, "
          f"{df['matchId'].nunique()} matches, seasons {df['season'].min()}-{df['season'].max()}")
    print("rows per season:\n", df.groupby("season").size())


if __name__ == "__main__":
    main()
