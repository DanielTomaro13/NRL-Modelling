"""
Kicker-points model.

Goal kicking is concentrated in one designated kicker per team. We model a player's
expected successful goals (conversions + penalty goals) as a Poisson rate, reusing the
leakage-safe feature machinery plus kicking-history rollups (so the model learns who the
kicker is and how busy the team's attack is). Field goals are rare and taken from rolling
history. Then:

    kicker_points = 2*goals + 1*fieldGoals
    E[kicker_points] = 2*lambda_goals + lambda_fg

Outputs:
  models/kicker_model.joblib
  reports/kicker_predictions.parquet   playerId,name,team,opp,lambda_goals,lambda_fg,exp_kicker_points
  reports/kicker.json                  backtest + leaders + round top kickers

CLI: python src/kicker.py [backtest]
"""
import sys, json, glob, os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import features as F
import predict as PR
import nrl_meta as M

PM = "data/processed/player_match.parquet"
MODEL = "models/kicker_model.joblib"
MIN_SEASON = 2021
HOLDOUTS = (2023, 2024, 2025)
# kicking stats we add to the rolling history
KICK_STATS = ["goals", "kicker_points", "fieldGoals", "conversions",
              "conversionAttempts", "penaltyGoals", "penaltyGoalAttempts"]


def add_kicking(df):
    df["goals"] = df["conversions"] + df["penaltyGoals"]
    df["kicker_points"] = 2 * df["goals"] + df["fieldGoals"]
    return df


def build(df):
    df = add_kicking(df)
    for s in KICK_STATS:
        if s not in F.ROLL_STATS:
            F.ROLL_STATS = F.ROLL_STATS + [s]
    for s in ["goals", "kicker_points"]:
        if s not in F.TEAM_STATS:
            F.TEAM_STATS = F.TEAM_STATS + [s]
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


def backtest(df, feats):
    kp_pred, kp_real, base, lam_all, goal_real = [], [], [], [], []
    for s in HOLDOUTS:
        tr = df[(df.season >= MIN_SEASON) & (df.season < s)]
        te = df[df.season == s]
        if len(te) == 0:
            continue
        mg = make_model().fit(prep_X(tr, feats), tr["goals"].values)
        lam_g = np.clip(mg.predict(prep_X(te, feats)), 0, None)
        lam_fg = np.clip(te["fieldGoals_r10"].fillna(0).values, 0, None)
        kp_pred.append(2 * lam_g + lam_fg)
        kp_real.append(te["kicker_points"].values)
        base.append(np.clip(te["kicker_points_r5"].fillna(0).values, 0, None))
        lam_all.append(lam_g)
        goal_real.append((te["goals"].values >= 1).astype(int))
    p = np.concatenate(kp_pred); y = np.concatenate(kp_real); b = np.concatenate(base)
    lam = np.concatenate(lam_all); gy = np.concatenate(goal_real)
    kk = y > 0

    # P(player kicks >=1 goal) calibration  (1 - e^-lambda_goals)
    pg = 1 - np.exp(-lam)
    order = np.argsort(pg)
    rel = {"pred": [], "emp": [], "n": []}
    for bb in np.array_split(order, 10):
        if len(bb):
            rel["pred"].append(round(float(pg[bb].mean()), 3))
            rel["emp"].append(round(float(gy[bb].mean()), 3))
            rel["n"].append(int(len(bb)))
    cal_err = round(float(np.mean([abs(a - e) for a, e in zip(rel["pred"], rel["emp"])])), 3)

    return {"n_test": int(len(y)),
            "mae_model": round(float(mean_absolute_error(y, p)), 3),
            "mae_baseline": round(float(mean_absolute_error(y, b)), 3),
            "mae_kickers_only": round(float(mean_absolute_error(y[kk], p[kk])), 3),
            "n_kickers": int(kk.sum()),
            "mean_kicker_points": round(float(y.mean()), 2),
            "goal_reliability": rel, "goal_calibration_error": cal_err,
            "holdouts": list(HOLDOUTS)}


def predict_round(df, feats, model):
    comp, meta = M.current_competition()
    fx = M.fixture(comp)
    rnd = M.next_round(comp, fx)
    lp = f"data/processed/lineups_r{rnd}.parquet"
    lineups = pd.read_parquet(lp) if os.path.exists(lp) else None
    pm = pd.read_parquet(PM)
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    up, matches = PR.build_upcoming(pm, comp, rnd, lineups)
    full = build(pd.concat([pm, up], ignore_index=True))
    pred = full[full.compName == "upcoming"].copy()
    pred["lambda_goals"] = np.clip(model.predict(prep_X(pred, feats)), 0, None)
    pred["lambda_fg"] = np.clip(pred["fieldGoals_r10"].fillna(0).values, 0, None)
    pred["exp_kicker_points"] = 2 * pred["lambda_goals"] + pred["lambda_fg"]
    nm = name_map()
    pred["name"] = pred["playerId"].map(nm)
    sid = {}
    for mm in matches:
        sid[mm["homeSquadId"]] = mm["homeSquadName"]
        sid[mm["awaySquadId"]] = mm["awaySquadName"]
    pred["team"] = pred["squadId"].map(sid)
    pred["opp"] = pred["oppSquadId"].map(sid)
    out = pred[["playerId", "name", "team", "opp", "position", "matchId",
                "lambda_goals", "lambda_fg", "exp_kicker_points"]].copy()
    out = out.sort_values("exp_kicker_points", ascending=False)
    out.to_parquet("reports/kicker_predictions.parquet", index=False)
    return out, rnd


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    pm = pd.read_parquet(PM)
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    df = build(pm)
    feats = feature_cols(df)
    lab = df[df.season >= MIN_SEASON]
    print(f"kicker model: {len(feats)} features, {len(lab)} rows, "
          f"mean kicker pts {lab['kicker_points'].mean():.2f}")

    bt = backtest(df, feats)
    print(f"  backtest: MAE {bt['mae_model']} vs baseline {bt['mae_baseline']} "
          f"(kickers-only MAE {bt['mae_kickers_only']}, n={bt['n_kickers']})")

    final = make_model().fit(prep_X(lab, feats), lab["goals"].values)
    joblib.dump({"model": final, "features": feats}, MODEL)
    print(f"  saved {MODEL}")

    cur = pm[pm.season == int(pm.season.max())].copy()
    cur = add_kicking(cur)
    nm = name_map()
    g = cur.groupby("playerId").agg(games=("matchId", "nunique"),
                                    kp=("kicker_points", "sum")).reset_index()
    g = g[g.games >= 4]
    g["name"] = g.playerId.map(nm)
    g["per_game"] = (g.kp / g.games).round(1)
    leaders = g.sort_values("per_game", ascending=False).head(12)
    leaders = [{"name": r["name"], "per_game": float(r["per_game"]), "games": int(r["games"])}
               for _, r in leaders.iterrows() if r["name"]]

    payload = {"generated": pd.Timestamp.now("UTC").isoformat(), "backtest": bt,
               "leaders": leaders, "season": int(pm.season.max())}
    if cmd != "backtest":
        out, rnd = predict_round(df, feats, final)
        payload["round"] = int(rnd)
        payload["top_kickers"] = [
            {"name": r["name"], "team": r["team"],
             "exp_kicker_points": round(float(r["exp_kicker_points"]), 1)}
            for _, r in out.head(12).iterrows() if r["name"] and r["exp_kicker_points"] > 0.5]
        print(f"  predicted round {rnd}: top kicker {out.iloc[0]['name']} "
              f"{out.iloc[0]['exp_kicker_points']:.1f} pts")
    json.dump(payload, open("reports/kicker.json", "w"))
    print("  wrote reports/kicker.json")


if __name__ == "__main__":
    main()
