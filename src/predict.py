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
import tracks as T


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
    # Only project games that haven't been played. Normally every game in the
    # upcoming round is still scheduled, but State of Origin lists all three games
    # under roundNumber 1, so without this filter each player would be projected
    # once per completed game as well as the next one.
    matches = [m for m in fixture(comp)
               if m["roundNumber"] == rnd and m.get("matchStatus") != "complete"]
    utc_by_match = {m["matchId"]: pd.to_datetime(m["utcStartTime"], utc=True) for m in matches}
    venue_by_match = {m["matchId"]: m["venueId"] for m in matches}
    season_by_match = {mid: (utc.year if pd.notna(utc) else None)
                       for mid, utc in utc_by_match.items()}
    pm = pm.sort_values("utcStartTime")
    rows = []

    if lineups is not None:
        # CONFIRMED team lists: named 1-17 only (18+ are emergency reserves)
        lu = lineups[lineups["jersey"] <= 17]
        for _, p in lu.iterrows():
            r = {c: 0 for c in pm.columns}
            r.update({
                "season": season_by_match.get(p["matchId"]),
                "competitionId": comp, "compName": "upcoming",
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
                        "season": season_by_match.get(m["matchId"]),
                        "competitionId": comp, "compName": "upcoming",
                        "matchId": m["matchId"], "roundNumber": rnd, "utcStartTime": utc,
                        "venueId": m["venueId"], "playerId": p["playerId"], "squadId": sq,
                        "oppSquadId": opp, "isHome": home, "position": p["position"],
                        "jumperNumber": p["jumperNumber"],
                    })
                    rows.append(r)

    # no upcoming games (e.g. a rep series whose games are all played) -> return an
    # empty frame with the right columns so downstream concat/feature code is a no-op
    up = pd.DataFrame(rows, columns=list(pm.columns)) if not rows else pd.DataFrame(rows)
    up["utcStartTime"] = pd.to_datetime(up["utcStartTime"], utc=True)
    return up, matches


def run(comp, rnd, lineups_path=None, track=None):
    track = track or T.current()
    pm = pd.read_parquet(T.proc("player_match.parquet", track))
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    lineups = pd.read_parquet(lineups_path) if lineups_path else None
    src = "CONFIRMED team lists" if lineups is not None else "most-recent-XVII proxy"
    print(f"[{track.name}] comp {comp} round {rnd} — lineup source: {src}")
    up, matches = build_upcoming(pm, comp, rnd, lineups)

    # augment history with upcoming rows, recompute features identically
    full = pd.concat([pm, up], ignore_index=True)
    full = F.add_perf_points(full)
    full = F.player_rolling(full)
    full = F.team_context(full)

    # use the (possibly reused) model's own feature list — authoritative, and means
    # Origin tracks don't need their own feature_cols.json
    bundle = joblib.load(T.model_for_prediction("nrl_models.joblib", track))
    feats = bundle.get("features") or json.load(open(T.proc("feature_cols.json", track)))["features"]
    pred_df = full[(full.compName == "upcoming")].copy()
    if pred_df.empty:
        # nothing to project (e.g. a rep series that has finished) — write empty
        T.ensure_dirs(track)
        pred_df.to_parquet(T.report("round_predictions.parquet", track), index=False)
        print(f"[{track.name}] no upcoming games for round {rnd} — wrote empty predictions")
        return pred_df, matches, rnd
    X = pred_df[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")

    prov = {t for t, m in bundle.get("targets_meta", {}).items() if m.get("provisional")}
    for t, mdl in bundle["models"].items():
        pred_df[f"pred_{t}"] = np.clip(mdl.predict(X), 0, None)
    pred_df.attrs["provisional_targets"] = sorted(prov)

    nm = name_map()
    pred_df["name"] = pred_df["playerId"].map(nm)
    sidname = {}
    for m in matches:
        sidname[m["homeSquadId"]] = m["homeSquadName"]; sidname[m["awaySquadId"]] = m["awaySquadName"]
    pred_df["team"] = pred_df["squadId"].map(sidname)
    pred_df["opp"] = pred_df["oppSquadId"].map(sidname)
    T.ensure_dirs(track)
    pred_df.to_parquet(T.report("round_predictions.parquet", track), index=False)

    avail = [f"pred_{t}" for t in ["runsHitup", "runs", "runMetres",
             "postContactMetres", "tackles", "perf_points"] if f"pred_{t}" in pred_df.columns]
    out = pred_df[["name", "team", "opp", "position"] + avail].rename(
        columns=lambda c: c.replace("pred_", ""))
    sort_col = "perf_points" if "perf_points" in out.columns else avail[0].replace("pred_", "")
    out = out.sort_values(sort_col, ascending=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 400)
    print(out.round(1).head(40).to_string(index=False))
    if prov:
        print(f"\n(provisional targets, low sample: {', '.join(sorted(prov))})")
    return pred_df, matches, rnd


def main():
    comp = int(sys.argv[1]) if len(sys.argv) > 1 else 12999
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    lineups_path = sys.argv[3] if len(sys.argv) > 3 else None
    track = T.current()
    pred_df, matches, rnd = run(comp, rnd, lineups_path, track)

    # ---- compare to industry lines (men's NRL debug aid only) ----
    if track.name != "nrl":
        return
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
