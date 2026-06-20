"""
Predict the six targets for an upcoming round.

For each scheduled match we build synthetic 'upcoming' rows for BOTH squads' likely
lineups (each squad's most-recent completed XVII as a proxy for the team list),
append them to history, re-run the identical leakage-safe feature pipeline, then
predict with the saved models. Reading predictions off the named players lets us
compare to bookmaker/industry performance-point lines.

Usage: python src/predict.py <competitionId> <roundNumber>
"""
import sys, json, glob, joblib
import numpy as np
import pandas as pd
import features as F   # reuse the exact feature functions

PM = "data/processed/player_match.parquet"


def name_map():
    nm = {}
    for fp in glob.glob("data/raw/*/*.json"):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for p in d.get("matchStats", {}).get("playerInfo", {}).get("player", []):
            nm[p["playerId"]] = p.get("displayName") or p.get("surname")
    return nm


def fixture(comp):
    import urllib.request
    url = f"https://mc.championdata.com/data/{comp}/fixture.json"
    return json.loads(urllib.request.urlopen(url, timeout=30).read())["fixture"]["match"]


def build_upcoming(pm, comp, rnd, lineups=None):
    matches = [m for m in fixture(comp) if m["roundNumber"] == rnd]
    utc_by_match = {m["matchId"]: pd.to_datetime(m["utcStartTime"], utc=True) for m in matches}
    venue_by_match = {m["matchId"]: m["venueId"] for m in matches}
    pm = pm.sort_values("utcStartTime")
    rows = []

    if lineups is not None:
        # CONFIRMED team lists: named 1-17 only (18+ are emergency reserves)
        lu = lineups[lineups["jersey"] <= 17]
        for _, p in lu.iterrows():
            r = {c: 0 for c in pm.columns}
            r.update({
                "season": 2026, "competitionId": comp, "compName": "upcoming",
                "matchId": p["matchId"], "roundNumber": rnd,
                "utcStartTime": utc_by_match.get(p["matchId"]),
                "venueId": venue_by_match.get(p["matchId"]),
                "playerId": p["playerId"], "squadId": p["squadId"],
                "oppSquadId": p["oppSquadId"], "isHome": p["isHome"],
                "position": p["position"], "jumperNumber": p["jumperNumber"],
            })
            rows.append(r)
    else:
        # PROXY: each squad's most-recent completed XVII
        for m in matches:
            utc = utc_by_match[m["matchId"]]
            for sq, opp, home in [(m["homeSquadId"], m["awaySquadId"], 1),
                                  (m["awaySquadId"], m["homeSquadId"], 0)]:
                hist = pm[pm.squadId == sq]
                if hist.empty:
                    continue
                lineup = hist[hist.matchId == hist.iloc[-1]["matchId"]]
                for _, p in lineup.iterrows():
                    r = {c: 0 for c in pm.columns}
                    r.update({
                        "season": 2026, "competitionId": comp, "compName": "upcoming",
                        "matchId": m["matchId"], "roundNumber": rnd, "utcStartTime": utc,
                        "venueId": m["venueId"], "playerId": p["playerId"], "squadId": sq,
                        "oppSquadId": opp, "isHome": home, "position": p["position"],
                        "jumperNumber": p["jumperNumber"],
                    })
                    rows.append(r)

    up = pd.DataFrame(rows)
    up["utcStartTime"] = pd.to_datetime(up["utcStartTime"], utc=True)
    return up, matches


def main():
    comp = int(sys.argv[1]) if len(sys.argv) > 1 else 12999
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    lineups_path = sys.argv[3] if len(sys.argv) > 3 else None

    pm = pd.read_parquet(PM)
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    lineups = pd.read_parquet(lineups_path) if lineups_path else None
    src = "CONFIRMED team lists" if lineups is not None else "most-recent-XVII proxy"
    print(f"Lineup source: {src}")
    up, matches = build_upcoming(pm, comp, rnd, lineups)

    # augment history with upcoming rows, recompute features identically
    full = pd.concat([pm, up], ignore_index=True)
    full = F.add_perf_points(full)
    full = F.player_rolling(full)
    full = F.team_context(full)

    feats = json.load(open("data/processed/feature_cols.json"))["features"]
    pred_df = full[(full.compName == "upcoming")].copy()
    X = pred_df[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")

    bundle = joblib.load("models/nrl_models.joblib")
    for t, mdl in bundle["models"].items():
        pred_df[f"pred_{t}"] = np.clip(mdl.predict(X), 0, None)

    nm = name_map()
    pred_df["name"] = pred_df["playerId"].map(nm)
    sidname = {}
    for m in matches:
        sidname[m["homeSquadId"]] = m["homeSquadName"]; sidname[m["awaySquadId"]] = m["awaySquadName"]
    pred_df["team"] = pred_df["squadId"].map(sidname)
    pred_df["opp"] = pred_df["oppSquadId"].map(sidname)
    pred_df.to_parquet("reports/round_predictions.parquet", index=False)

    cols = ["name", "team", "opp", "position", "pred_runsHitup", "pred_runs",
            "pred_runMetres", "pred_postContactMetres", "pred_tackles", "pred_perf_points"]
    out = pred_df[cols].rename(columns=lambda c: c.replace("pred_", ""))
    out = out.sort_values("perf_points", ascending=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 400)
    print(out.round(1).to_string(index=False))

    # ---- compare to industry lines ----
    lines = {995889: ("Isaako", 68.5), 1011024: ("Ford(Warriors)", 65.6),
             1028652: ("T.May(prop)", 62.5), 1024143: ("T.May(centre)", None),
             1011034: ("Tupouniua", 56.5), 1023703: ("Luki", 56.5),
             1030037: ("Lisati", 53.5), 1007443: ("I.Papali'i(Pen)", 52.5),
             31677: ("J.Papali'i(Can)", None), 997794: ("Tracey", 27.5),
             1023893: ("Turuva", 28.5), 31932: ("Molo", 28.5), 1031174: ("Tuaupiki", 30.5)}
    rows = []
    pdict = pred_df.set_index("playerId")
    for pid, (lab, line) in lines.items():
        if pid in pdict.index:
            pp = float(pdict.loc[pid, "pred_perf_points"]) if np.ndim(pdict.loc[pid, "pred_perf_points"]) == 0 else float(pdict.loc[pid, "pred_perf_points"].iloc[0])
            rows.append({"player": lab, "team": pdict.loc[pid].get("team") if np.ndim(pdict.loc[pid, "pred_perf_points"]) == 0 else pdict.loc[pid, "team"].iloc[0],
                         "model_perf_pts": round(pp, 1), "industry_line": line,
                         "diff": round(pp - line, 1) if line else None})
        else:
            rows.append({"player": lab, "team": None, "model_perf_pts": None,
                         "industry_line": line, "diff": None})
    cmp = pd.DataFrame(rows)
    print("\n=== Model vs industry performance-point lines (round %d) ===" % rnd)
    print(cmp.to_string(index=False))
    valid = cmp.dropna(subset=["model_perf_pts", "industry_line"])
    if len(valid):
        print(f"\nMAE vs industry lines: {valid['diff'].abs().mean():.2f}  "
              f"(n={len(valid)}, mean line={valid['industry_line'].mean():.1f})")


if __name__ == "__main__":
    main()
