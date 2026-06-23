"""
Try-scorer model.

A Poisson model for a player's expected tries in a match (reusing the same
leakage-safe feature machinery as the stat models, plus tries-specific rollups and
opponent tries-conceded). From the expected-tries rate lambda we derive the prices
the books actually offer:

    P(anytime / 1+) = 1 - e^-lambda
    P(2+)           = 1 - e^-lambda (1 + lambda)
    P(3+)           = 1 - e^-lambda (1 + lambda + lambda^2/2)
    E[tries]        = lambda

Backtest treats "did the player score (1+)" as the binary outcome and reports Brier
score, log loss and AUC against naive baselines, plus a probability-calibration curve.

CLI:
  python src/tryscorer.py            # train + backtest + predict the current round
  python src/tryscorer.py backtest   # train + backtest only
"""
import sys, json, math, glob, os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import features as F
import predict as PR
import nrl_meta as M

PM = "data/processed/player_match.parquet"
MODEL = "models/try_model.joblib"
MIN_SEASON = 2021
HOLDOUTS = (2023, 2024, 2025)


# --------------------------------------------------------------------------- features
def build(df):
    """Run the leakage-safe pipeline with tries added to the rolling/team stats."""
    if "tries" not in F.ROLL_STATS:
        F.ROLL_STATS = F.ROLL_STATS + ["tries"]
    if "tries" not in F.TEAM_STATS:
        F.TEAM_STATS = F.TEAM_STATS + ["tries"]
    df = F.add_perf_points(df)
    df = F.player_rolling(df)
    df = F.team_context(df)
    df["position"] = df["position"].replace("-", "Unknown").fillna("Unknown")
    return df


def feature_cols(df):
    cols = (["isHome", "games_prior", "days_rest", "jumperNumber"]
            + [c for c in df.columns if c.endswith(("_r3", "_r5", "_r10", "_career"))]
            + [c for c in df.columns if c.startswith(("own_", "opp_"))])
    return sorted(set(cols))


def make_model():
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25,
        categorical_features=["position"], random_state=0)


def prep_X(df, feats):
    X = df[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")
    return X


# probability helpers
def p_at_least(lam, k):
    lam = np.asarray(lam, float)
    if k == 1:
        return 1 - np.exp(-lam)
    if k == 2:
        return 1 - np.exp(-lam) * (1 + lam)
    if k == 3:
        return 1 - np.exp(-lam) * (1 + lam + lam ** 2 / 2)
    return 1 - np.exp(-lam)


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


# --------------------------------------------------------------------------- backtest
def backtest(df, feats):
    rows_pred, rows_real, rows_pos = [], [], []
    base_pos, base_trail = [], []
    # position base rate learned on training portion only (per holdout)
    for s in HOLDOUTS:
        tr = df[(df.season >= MIN_SEASON) & (df.season < s)]
        te = df[df.season == s]
        if len(te) == 0:
            continue
        m = make_model().fit(prep_X(tr, feats), tr["tries"].values)
        lam = np.clip(m.predict(prep_X(te, feats)), 0, None)
        p = p_at_least(lam, 1)
        scored = (te["tries"].values >= 1).astype(int)
        rows_pred.append(p); rows_real.append(scored)
        # baselines
        pos_rate = tr.assign(sc=(tr["tries"] >= 1).astype(int)).groupby("position")["sc"].mean()
        base_pos.append(te["position"].map(pos_rate).fillna(scored.mean()).values)
        trail = np.clip(te["tries_r5"].fillna(te["tries"].mean()).values, 0, None)
        base_trail.append(1 - np.exp(-trail))

    p = np.concatenate(rows_pred); y = np.concatenate(rows_real)
    bp = np.clip(np.concatenate(base_pos), 1e-6, 1 - 1e-6)
    bt = np.clip(np.concatenate(base_trail), 1e-6, 1 - 1e-6)
    pc = np.clip(p, 1e-6, 1 - 1e-6)

    def metrics(prob):
        return {"brier": round(float(brier_score_loss(y, prob)), 4),
                "logloss": round(float(log_loss(y, prob)), 4),
                "auc": round(float(roc_auc_score(y, prob)), 4)}

    # calibration reliability (decile bins by predicted prob)
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    rel = {"pred": [], "emp": [], "n": []}
    for b in bins:
        if len(b):
            rel["pred"].append(round(float(p[b].mean()), 3))
            rel["emp"].append(round(float(y[b].mean()), 3))
            rel["n"].append(int(len(b)))
    cal_err = round(float(np.mean([abs(a - e) for a, e in zip(rel["pred"], rel["emp"])])), 3)

    return {"n_test": int(len(y)), "base_rate": round(float(y.mean()), 3),
            "model": metrics(pc), "baseline_position": metrics(bp),
            "baseline_trailing5": metrics(bt),
            "reliability": rel, "calibration_error": cal_err,
            "holdouts": list(HOLDOUTS)}


# --------------------------------------------------------------------------- predict the round
def predict_round(df, feats, model):
    comp, meta = M.current_competition()
    fx = M.fixture(comp)
    rnd = M.next_round(comp, fx)
    lineups_path = f"data/processed/lineups_r{rnd}.parquet"
    lineups = pd.read_parquet(lineups_path) if os.path.exists(lineups_path) else None
    pm = pd.read_parquet(PM)
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    up, matches = PR.build_upcoming(pm, comp, rnd, lineups)
    full = build(pd.concat([pm, up], ignore_index=True))
    pred_df = full[full.compName == "upcoming"].copy()
    lam = np.clip(model.predict(prep_X(pred_df, feats)), 0, None)
    pred_df["lambda"] = lam
    pred_df["p_anytime"] = p_at_least(lam, 1)
    pred_df["p_2plus"] = p_at_least(lam, 2)
    pred_df["exp_tries"] = lam
    nm = name_map()
    pred_df["name"] = pred_df["playerId"].map(nm)
    sidname = {}
    for mm in matches:
        sidname[mm["homeSquadId"]] = mm["homeSquadName"]
        sidname[mm["awaySquadId"]] = mm["awaySquadName"]
    pred_df["team"] = pred_df["squadId"].map(sidname)
    pred_df["opp"] = pred_df["oppSquadId"].map(sidname)
    out = pred_df[["playerId", "name", "team", "opp", "position", "matchId",
                   "lambda", "p_anytime", "p_2plus", "exp_tries"]].copy()
    out = out.sort_values("p_anytime", ascending=False)
    out.to_parquet("reports/tryscorer_predictions.parquet", index=False)
    return out, rnd


# --------------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    pm = pd.read_parquet(PM)
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    df = build(pm)
    feats = feature_cols(df)
    lab = df[df.season >= MIN_SEASON]
    print(f"try model: {len(feats)} features, {len(lab)} labelled rows "
          f"(season>={MIN_SEASON}), base score rate {(lab['tries']>=1).mean():.3f}")

    bt = backtest(df, feats)
    print(f"  backtest: model Brier {bt['model']['brier']} AUC {bt['model']['auc']} "
          f"logloss {bt['model']['logloss']}  | calib err {bt['calibration_error']}")
    print(f"  vs position-rate Brier {bt['baseline_position']['brier']} / "
          f"trailing5 Brier {bt['baseline_trailing5']['brier']}")

    # final model on all post-2021 data
    final = make_model().fit(prep_X(lab, feats), lab["tries"].values)
    joblib.dump({"model": final, "features": feats}, MODEL)
    print(f"  saved {MODEL}")

    # permutation feature importance (for the Model Lab)
    importance = []
    try:
        from sklearn.inspection import permutation_importance
        te = df[df.season.isin(HOLDOUTS)].sample(min(1500, int((df.season.isin(HOLDOUTS)).sum())),
                                                 random_state=0)
        r = permutation_importance(final, prep_X(te, feats), te["tries"].values,
                                   n_repeats=4, random_state=0,
                                   scoring="neg_mean_absolute_error")
        cols = feats + ["position"]
        for i in np.argsort(r.importances_mean)[::-1][:10]:
            if r.importances_mean[i] > 0:
                f = cols[i]
                for a, b in [("_r5", " (5-game)"), ("_r3", " (3-game)"), ("_r10", " (10-game)"),
                             ("_career", " (career)"), ("opp_", "opp def: "), ("own_", "team: "),
                             ("Allowed", " conceded"), ("_", " ")]:
                    f = f.replace(a, b)
                importance.append({"feature": f.strip(), "importance": round(float(r.importances_mean[i]), 3)})
    except Exception as e:
        print("  importance skipped:", repr(e))

    # season leaders by model expected tries (this season actuals for context)
    cur = pm[pm.season == int(pm.season.max())].copy()
    nm = name_map()
    g = cur.groupby("playerId")
    base = g.agg(games=("matchId", "nunique"), tries=("tries", "sum")).reset_index()
    base = base[base.games >= 4]
    base["name"] = base.playerId.map(nm)
    base["per_game"] = (base.tries / base.games).round(2)
    leaders = base.sort_values("per_game", ascending=False).head(12)
    leaders = [{"name": r["name"], "tries": int(r["tries"]), "games": int(r["games"]),
                "per_game": float(r["per_game"])} for _, r in leaders.iterrows() if r["name"]]

    payload = {"generated": pd.Timestamp.now('UTC').isoformat(), "backtest": bt,
               "leaders": leaders, "importance": importance, "season": int(pm.season.max())}

    if cmd != "backtest":
        out, rnd = predict_round(df, feats, final)
        payload["round"] = int(rnd)
        payload["top_chances"] = [
            {"name": r["name"], "team": r["team"], "opp": r["opp"],
             "p_anytime": round(float(r["p_anytime"]), 3),
             "p_2plus": round(float(r["p_2plus"]), 3)}
            for _, r in out.head(15).iterrows() if r["name"]]
        print(f"  predicted round {rnd}: {len(out)} players; "
              f"top chance {out.iloc[0]['name']} {out.iloc[0]['p_anytime']:.2f}")

    json.dump(payload, open("reports/tryscorer.json", "w"))
    print("  wrote reports/tryscorer.json")


if __name__ == "__main__":
    main()
