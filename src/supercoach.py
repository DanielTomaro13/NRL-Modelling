"""
NRL SuperCoach player data -> reports/site/supercoach.json

Pulls SuperCoach's public (no-auth) JSON: every player's price, scoring averages,
projection, ownership, position(s), availability, news and matchup context, plus a
per-round score series for std/sparklines. nrl24-0.com fetches the committed bundle
and renders the /supercoach pages. Mirrors the AFL SuperCoach feed on afl23-0.com.

Source: https://www.supercoach.com.au/<year>/api/nrl/classic/v1/...   (anonymous)
"""
import json, os, datetime, statistics, time
from curl_cffi import requests as creq

YEAR = datetime.date.today().year
BASE = f"https://www.supercoach.com.au/{YEAR}/api/nrl/classic/v1"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 Chrome/126 Safari/537.36", "Accept": "application/json"}
OUT = "reports/site/supercoach.json"


def get(url, tries=4):
    for i in range(tries):
        try:
            r = creq.get(url, headers=HDR, impersonate="chrome", timeout=40)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"  [sc] {url[:70]}… {e!r}")
        time.sleep(0.8 * (i + 1))
    return None


def num(v):
    return v if isinstance(v, (int, float)) and v == v else 0


def r1(v):
    return round(v * 10) / 10


def main():
    settings = get(f"{BASE}/settings?min=false") or {}
    comp = settings.get("competition", {})
    completed = int(num(comp.get("current_round")))
    rnd = int(num(comp.get("next_round")) or completed + 1)

    raw = get(f"{BASE}/players-cf?embed=positions%2Cplayer_stats%2Cnotes&round={rnd}")
    if not raw:
        print("[supercoach] no players returned — aborting (feed unchanged)")
        raise SystemExit(1)

    # per-round score series -> std / consistency / sparkline
    series = {}
    for rd in range(1, completed + 1):
        rows = get(f"{BASE}/players-cf?embed=player_match_stats&round={rd}")
        if not rows:
            continue
        for p in rows:
            ms = (p.get("player_match_stats") or [None])[0]
            if not ms or num(ms.get("games")) < 1:
                continue
            series.setdefault(p["id"], []).append({"round": rd, "pts": num(ms.get("points"))})
        time.sleep(0.12)

    players = []
    for p in raw:
        ps = (p.get("player_stats") or [{}])[0] or {}
        scores = sorted(series.get(p["id"], []), key=lambda s: s["round"])
        vals = [s["pts"] for s in scores]
        mean = statistics.mean(vals) if vals else num(ps.get("avg"))
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        positions = [x.get("position") for x in (p.get("positions") or []) if x.get("position")]
        note = (p.get("notes") or [None])[0]
        nxt = []
        for ab, h in [(ps.get("opp1"), ps.get("opp1h")), (ps.get("opp2"), ps.get("opp2h")), (ps.get("opp3"), ps.get("opp3h"))]:
            if ab and ab.get("abbrev"):
                nxt.append({"opp": ab["abbrev"], "home": bool(num(h))})
        ven = ps.get("ven") or {}
        players.append({
            "id": p["id"],
            "name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "team": (p.get("team") or {}).get("name", ""),
            "teamAbbr": (p.get("team") or {}).get("abbrev", ""),
            "positions": positions, "dpp": len(positions) > 1,
            "price": int(num(ps.get("price"))),
            "priceChange": int(num(ps.get("price_change"))),
            "totalPriceChange": int(num(ps.get("total_price_change"))),
            "avg": r1(num(ps.get("avg"))), "avg3": r1(num(ps.get("avg3"))), "avg5": r1(num(ps.get("avg5"))),
            # ppts1 = SuperCoach's real next-round projection (ppts is erratic — do not use)
            "proj": r1(num(ps.get("ppts1")) or num(ps.get("avg"))),
            "games": int(num(ps.get("total_games"))), "totalPoints": int(num(ps.get("total_points"))),
            "std": r1(std), "consistency": r1(100 * (1 - std / mean)) if mean > 0 else 0,
            "scores": scores,
            "owned": r1(num(ps.get("owned"))), "ppm": r1(num(ps.get("total_points_per_min"))),
            "status": (p.get("played_status") or {}).get("status", ""),
            "statusText": p.get("injury_suspension_status_text"),
            "note": note.get("note") if note else None,
            "noteDate": note.get("created_on") if note else None,
            "opp": (ps.get("opp") or {}).get("abbrev"), "oppHome": bool(num(ps.get("opph"))),
            "oppAvg": r1(num(ps.get("oppavg"))),
            "ven": ven.get("display_name") or ven.get("short_name") or ven.get("name"),
            "venAvg": r1(num(ps.get("venavg"))),
            "next": nxt,
        })

    players.sort(key=lambda x: -x["avg"])
    out = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "season": YEAR, "round": rnd, "n_players": len(players), "players": players}
    os.makedirs("reports/site", exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"Wrote {OUT}: {len(players)} players, round {rnd}, {len(series)} with score history")


if __name__ == "__main__":
    main()
