"""
Derived analytics for the site: Champion Data insights, model backtest, probability
calibration of the distribution-pricing layer, and feature importance.

Writes:
  reports/analysis.json     everything the site renders (tables + chart data)
  docs/data/model.json      compact payload for the interactive Model Lab (JS)

Run from repo root:  python src/analysis.py
"""
import json, glob, os
import numpy as np
import pandas as pd
from scipy.stats import norm

import pricing as P
import nrl_meta as M

FEAT = "data/processed/features.parquet"
META = "data/processed/feature_cols.json"
MODEL = "models/nrl_models.joblib"
PM = "data/processed/player_match.parquet"
TARGETS = P.TARGETS
HOLDOUTS = (2023, 2024, 2025)
TARGET_LABEL = {"runsHitup": "Hit-ups", "runs": "Runs", "runMetres": "Run metres",
                "postContactMetres": "Post-contact m", "tackles": "Tackles",
                "perf_points": "Performance pts"}


# --------------------------------------------------------------------------- helpers
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


def squad_names():
    try:
        comp, _ = M.current_competition()
        return M.squad_name_map(comp)
    except Exception:
        return {}


# --------------------------------------------------------------------------- backtest
def backtest(df, feats, bundle, disp):
    te = df[df.season.isin(HOLDOUTS)].copy()
    X = te[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")

    season_mae, summary, calibration, residuals = [], [], {}, {}
    reliability = {}
    pooled_pred_p, pooled_real = [], []

    for t in TARGETS:
        pred = np.clip(bundle["models"][t].predict(X), 0, None)
        actual = te[t].values
        resid = actual - pred
        # per-season MAE vs trailing-5 baseline
        for s in HOLDOUTS:
            m = te.season.values == s
            if m.sum() == 0:
                continue
            base = np.clip(te[f"{t}_r5"].fillna(te[t].mean()).values, 0, None)
            mae_m = float(np.abs(resid[m]).mean())
            mae_b = float(np.abs(actual[m] - base[m]).mean())
            season_mae.append({"target": t, "label": TARGET_LABEL[t], "holdout": int(s),
                               "MAE_model": round(mae_m, 2), "MAE_base_r5": round(mae_b, 2),
                               "gain_pct": round(100 * (mae_b - mae_m) / mae_b, 1)})
        base = np.clip(te[f"{t}_r5"].fillna(te[t].mean()).values, 0, None)
        mae_m = float(np.abs(resid).mean())
        mae_b = float(np.abs(actual - base).mean())
        summary.append({"target": t, "label": TARGET_LABEL[t],
                        "MAE_model": round(mae_m, 2), "MAE_base_r5": round(mae_b, 2),
                        "gain_pct": round(100 * (mae_b - mae_m) / mae_b, 1),
                        "n": int(len(te))})

        # predicted-vs-actual calibration (decile means)
        order = np.argsort(pred)
        bins = np.array_split(order, 10)
        calibration[t] = {
            "pred": [round(float(pred[b].mean()), 2) for b in bins if len(b)],
            "actual": [round(float(actual[b].mean()), 2) for b in bins if len(b)]}

        # residual histogram
        lo, hi = np.percentile(resid, [1, 99])
        edges = np.linspace(lo, hi, 26)
        counts, _ = np.histogram(np.clip(resid, lo, hi), bins=edges)
        residuals[t] = {"edges": [round(float(e), 2) for e in edges],
                        "counts": [int(c) for c in counts],
                        "mean": round(float(resid.mean()), 2),
                        "sd": round(float(resid.std()), 2)}

        # over/under probability reliability: for each row, place lines at
        # mean + k*sigma and check whether the calibrated P(over) matches reality.
        sig = np.array([P.sigma_for(t, m, disp) for m in pred])
        pp, rr = [], []
        for k in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
            line = pred + k * sig
            p_over = 1 - norm.cdf(k)            # = P(actual > pred + k*sigma) under model
            real = (actual > line).astype(float)
            pp.append(np.full(len(real), p_over))
            rr.append(real)
        pp = np.concatenate(pp); rr = np.concatenate(rr)
        # bin into 10 probability buckets
        bins_idx = np.clip((pp * 10).astype(int), 0, 9)
        rel_pred, rel_emp, rel_n = [], [], []
        for b in range(10):
            m = bins_idx == b
            if m.sum() >= 30:
                rel_pred.append(round(float(pp[m].mean()), 3))
                rel_emp.append(round(float(rr[m].mean()), 3))
                rel_n.append(int(m.sum()))
        reliability[t] = {"pred": rel_pred, "emp": rel_emp, "n": rel_n}
        pooled_pred_p.append(pp); pooled_real.append(rr)

    pp = np.concatenate(pooled_pred_p); rr = np.concatenate(pooled_real)
    bins_idx = np.clip((pp * 10).astype(int), 0, 9)
    pooled = {"pred": [], "emp": [], "n": []}
    for b in range(10):
        m = bins_idx == b
        if m.sum() >= 50:
            pooled["pred"].append(round(float(pp[m].mean()), 3))
            pooled["emp"].append(round(float(rr[m].mean()), 3))
            pooled["n"].append(int(m.sum()))
    cal_err = round(float(np.mean([abs(a - b) for a, b in zip(pooled["pred"], pooled["emp"])])), 3)

    return {"season_mae": season_mae, "summary": summary, "calibration": calibration,
            "residuals": residuals, "reliability": reliability, "reliability_pooled": pooled,
            "calibration_error": cal_err, "n_test": int(len(te)),
            "holdouts": list(HOLDOUTS)}


# --------------------------------------------------------------------------- champion data
def champion_insights(pm, nm, snames, season=None):
    season = season or int(pm.season.max())
    cur = pm[pm.season == season].copy()
    # perf_points is a derived target (not in raw player_match) — apply the formula
    cur["perf_points"] = (4 * cur["points"] + 10 * cur["tryAssists"]
                          + 5 * cur["lineBreaks"] + cur["tackles"]
                          + np.floor(cur["runMetres"] / 10))
    cur["name"] = cur.playerId.map(nm)
    cur["team"] = cur.squadId.map(snames)
    cur["opp"] = cur.oppSquadId.map(snames)

    # position profiles (mean six stats), exclude tiny/unknown buckets
    prof = (cur[cur.activity > 0]
            .groupby("position")[TARGETS + ["activity"]].mean())
    counts = cur.groupby("position").size()
    profiles = []
    for pos, row in prof.iterrows():
        if pos in ("Unknown",) or counts.get(pos, 0) < 30:
            continue
        profiles.append({"position": pos, "n": int(counts[pos]),
                         **{t: round(float(row[t]), 1) for t in TARGETS}})
    profiles.sort(key=lambda r: -r["tackles"])

    # team defence: stats conceded (grouped by the opponent the player faced)
    dfc = (cur.groupby("oppSquadId")
           .agg(games=("matchId", "nunique"),
                runMetres=("runMetres", "sum"),
                tackleBreaks=("tackleBreaks", "sum"),
                lineBreaks=("lineBreaks", "sum"),
                perf=("perf_points", "sum")).reset_index())
    team_def = []
    for _, r in dfc.iterrows():
        g = max(int(r["games"]), 1)
        team_def.append({"team": snames.get(r["oppSquadId"], str(r["oppSquadId"])),
                         "games": g,
                         "runMetres_conceded": round(r["runMetres"] / g),
                         "tackleBreaks_conceded": round(r["tackleBreaks"] / g, 1),
                         "lineBreaks_conceded": round(r["lineBreaks"] / g, 1),
                         "perf_conceded": round(r["perf"] / g)})
    team_def.sort(key=lambda r: -r["runMetres_conceded"])

    # leaders (season per-game average, min games)
    min_games = 4
    g = cur.groupby(["playerId"])
    base = g.agg(games=("matchId", "nunique")).reset_index()
    leaders = {}
    for t in ("tackles", "runMetres", "perf_points", "postContactMetres"):
        avg = g[t].mean().reset_index().merge(base, on="playerId")
        avg = avg[avg.games >= min_games].copy()
        avg["name"] = avg.playerId.map(nm)
        # team = latest squad
        latest = cur.sort_values("utcStartTime").groupby("playerId").tail(1).set_index("playerId")["team"]
        avg["team"] = avg.playerId.map(latest)
        avg = avg.sort_values(t, ascending=False).head(12)
        leaders[t] = [{"name": r["name"], "team": r["team"],
                       "avg": round(float(r[t]), 1), "games": int(r["games"])}
                      for _, r in avg.iterrows() if r["name"]]

    return {"season": season, "position_profiles": profiles,
            "team_defence": team_def, "leaders": leaders,
            "n_players": int(cur.playerId.nunique()),
            "n_matches": int(cur.matchId.nunique())}


# --------------------------------------------------------------------------- feature importance
PRETTY_FEATURE = [
    ("_r5", " (5-game avg)"), ("_r3", " (3-game avg)"), ("_r10", " (10-game avg)"),
    ("_career", " (career)"), ("opp_", "opp def: "), ("own_", "team: "),
    ("Allowed", " conceded"), ("_", " "),
]


def prettify(f):
    s = f
    for a, b in PRETTY_FEATURE:
        s = s.replace(a, b)
    return s.strip()


def feature_importance(df, feats, bundle, n_sample=1500, n_repeats=4):
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error
    te = df[df.season.isin(HOLDOUTS)]
    te = te.sample(min(n_sample, len(te)), random_state=0)
    X = te[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")
    imp = {}
    for t in TARGETS:
        mdl = bundle["models"][t]
        r = permutation_importance(mdl, X, te[t].values, n_repeats=n_repeats,
                                   random_state=0, scoring="neg_mean_absolute_error")
        idx = np.argsort(r.importances_mean)[::-1][:10]
        cols = feats + ["position"]
        imp[t] = [{"feature": prettify(cols[i]),
                   "importance": round(float(r.importances_mean[i]), 3)}
                  for i in idx if r.importances_mean[i] > 0]
    return imp


# --------------------------------------------------------------------------- main
def main():
    import joblib
    df = pd.read_parquet(FEAT)
    feats = json.load(open(META))["features"]
    bundle = joblib.load(MODEL)
    disp = P.load_dispersion() or P.calibrate_dispersion()

    print("backtest…")
    bt = backtest(df, feats, bundle, disp)
    print(f"  calibration error (pooled |pred-emp|) = {bt['calibration_error']}")

    print("champion data insights…")
    pm = pd.read_parquet(PM)
    nm = name_map()
    ci = champion_insights(pm, nm, squad_names())

    print("feature importance…")
    try:
        imp = feature_importance(df, feats, bundle)
    except Exception as e:
        print("  importance skipped:", repr(e))
        imp = {}

    out = {"generated": pd.Timestamp.now('UTC').isoformat(),
           "targets": TARGETS, "target_label": TARGET_LABEL,
           "backtest": bt, "champion": ci, "importance": imp, "dispersion": disp}
    os.makedirs("reports", exist_ok=True)
    json.dump(out, open("reports/analysis.json", "w"))
    print("wrote reports/analysis.json")

    # compact payload for the JS Model Lab: dispersion + per-target typical mean
    os.makedirs("docs/data", exist_ok=True)
    lab = {"dispersion": disp, "target_label": TARGET_LABEL,
           "typical_mean": {t: round(float(df[df.season.isin(HOLDOUTS)][t].mean()), 1)
                            for t in TARGETS},
           "calibration_error": bt["calibration_error"]}
    json.dump(lab, open("docs/data/model.json", "w"))
    print("wrote docs/data/model.json")


if __name__ == "__main__":
    main()
