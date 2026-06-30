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
import tracks as T

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


def backtest(df, feats, track):
    kp_pred, kp_real, base, lam_all, goal_real = [], [], [], [], []
    for s in track.holdouts:
        tr = df[(df.season >= track.min_season) & (df.season < s)]
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
            "holdouts": list(track.holdouts)}


def predict_round(df, feats, model, track):
    comp, meta = M.current_competition(track)
    fx = M.fixture(comp)
    rnd = M.next_round(comp, fx)
    lp = T.proc(f"lineups_r{rnd}.parquet", track)
    lineups = pd.read_parquet(lp) if os.path.exists(lp) else None
    pm = pd.read_parquet(T.proc("player_match.parquet", track))
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
    T.ensure_dirs(track)
    out.to_parquet(T.report("kicker_predictions.parquet", track), index=False)
    return out, rnd


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    track = T.current()
    T.ensure_dirs(track)
    pm = pd.read_parquet(T.proc("player_match.parquet", track))
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    df = build(pm)

    # Origin tracks reuse the club track's kicker model (predict-only).
    if track.name != track.model_track:
        mb = joblib.load(T.model("kicker_model.joblib", T.TRACKS[track.model_track]))
        out, rnd = predict_round(df, mb["features"], mb["model"], track)
        payload = {"generated": pd.Timestamp.now("UTC").isoformat(),
                   "reused_model": track.model_track, "round": int(rnd),
                   "top_kickers": [{"name": r["name"], "team": r["team"],
                                    "exp_kicker_points": round(float(r["exp_kicker_points"]), 1)}
                                   for _, r in out.head(12).iterrows()
                                   if r["name"] and r["exp_kicker_points"] > 0.5]}
        json.dump(payload, open(T.report("kicker.json", track), "w"))
        print(f"[{track.name}] reused {track.model_track} kicker model; predicted round {rnd}")
        return

    feats = feature_cols(df)
    lab = df[df.season >= track.min_season]
    print(f"[{track.name}] kicker model: {len(feats)} features, {len(lab)} rows, "
          f"mean kicker pts {lab['kicker_points'].mean():.2f}")

    bt = backtest(df, feats, track)
    print(f"  backtest: MAE {bt['mae_model']} vs baseline {bt['mae_baseline']} "
          f"(kickers-only MAE {bt['mae_kickers_only']}, n={bt['n_kickers']})")

    final = make_model().fit(prep_X(lab, feats), lab["goals"].values)
    MODEL = T.model("kicker_model.joblib", track)
    joblib.dump({"model": final, "features": feats, "track": track.name}, MODEL)
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
        out, rnd = predict_round(df, feats, final, track)
        payload["round"] = int(rnd)
        payload["top_kickers"] = [
            {"name": r["name"], "team": r["team"],
             "exp_kicker_points": round(float(r["exp_kicker_points"]), 1)}
            for _, r in out.head(12).iterrows() if r["name"] and r["exp_kicker_points"] > 0.5]
        print(f"  predicted round {rnd}: top kicker {out.iloc[0]['name']} "
              f"{out.iloc[0]['exp_kicker_points']:.1f} pts")
    dest = T.report("kicker.json", track)
    json.dump(payload, open(dest, "w"))
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
