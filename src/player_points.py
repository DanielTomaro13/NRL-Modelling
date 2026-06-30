"""
Player-points model = try model + kicker model.

A player's points decompose exactly into try points and goal-kicking points:

    points = 4*tries + 2*goals + 1*fieldGoals

We already model each piece as a Poisson rate (tries -> src/tryscorer.py,
goals/field goals -> src/kicker.py). Treating them as independent Poissons, the points
distribution is the convolution of 4*Pois(lt) + 2*Pois(lg) + 1*Pois(lfg). From it we get
the expected points and P(points >= line) for any posted player-points line.

Outputs:
  reports/player_points_predictions.parquet  playerId,name,team,opp,lt,lg,lfg,exp_points
  reports/player_points.json                  backtest (MAE + over/under reliability) + leaders

CLI: python src/player_points.py [backtest]
"""
import sys, json, math, os
import numpy as np
import pandas as pd
from scipy.stats import poisson

import features as F
import predict as PR
import nrl_meta as M
import tryscorer as TS
import kicker as KK
import tracks as T


# --------------------------------------------------------------------------- points distribution
def points_pmf(lt, lg, lfg, kmax_t=8, kmax_g=14, kmax_f=4):
    """PMF of points = 4*tries + 2*goals + fieldGoals for independent Poissons."""
    t = poisson.pmf(np.arange(kmax_t + 1), max(lt, 1e-9))
    g = poisson.pmf(np.arange(kmax_g + 1), max(lg, 1e-9))
    f = poisson.pmf(np.arange(kmax_f + 1), max(lfg, 1e-9))
    pmf = {}
    for ti, pt in enumerate(t):
        for gi, pg in enumerate(g):
            base = 4 * ti + 2 * gi
            ptg = pt * pg
            for fi, pf in enumerate(f):
                v = base + fi
                pmf[v] = pmf.get(v, 0.0) + ptg * pf
    return pmf


def p_over_under(lt, lg, lfg, line):
    """Return (p_over, p_under, p_push) for a player-points line."""
    pmf = points_pmf(lt, lg, lfg)
    p_under = sum(p for v, p in pmf.items() if v < math.ceil(line) and v < line)
    p_push = sum(p for v, p in pmf.items() if abs(v - line) < 1e-9)
    p_over = sum(p for v, p in pmf.items() if v > line)
    s = p_over + p_under + p_push
    if s > 0:
        p_over, p_under, p_push = p_over / s, p_under / s, p_push / s
    return p_over, p_under, p_push


def expected_points(lt, lg, lfg):
    return 4 * lt + 2 * lg + lfg


# --------------------------------------------------------------------------- backtest
def backtest(track=None):
    """Train tries + goals on each holdout's past, combine, score player points."""
    track = track or T.current()
    pm = pd.read_parquet(T.proc("player_match.parquet", track))
    pm["utcStartTime"] = pd.to_datetime(pm["utcStartTime"], utc=True)
    # one combined leakage-safe build with BOTH tries and goal-kicking rollups
    pm = KK.add_kicking(pm)
    for s in KK.KICK_STATS + ["tries"]:
        if s not in F.ROLL_STATS:
            F.ROLL_STATS = F.ROLL_STATS + [s]
    for s in ["goals", "kicker_points", "tries"]:
        if s not in F.TEAM_STATS:
            F.TEAM_STATS = F.TEAM_STATS + [s]
    df = F.add_perf_points(pm)
    df = F.player_rolling(df)
    df = F.team_context(df)
    df["position"] = df["position"].replace("-", "Unknown").fillna("Unknown")
    feats = [c for c in sorted(set(KK.feature_cols(df)) | set(TS.feature_cols(df))) if c in df.columns]

    pred_pts, real_pts, lt_all, lg_all, lfg_all, real_t, real_g = [], [], [], [], [], [], []
    for s in track.holdouts:
        tr = df[(df.season >= track.min_season) & (df.season < s)]
        te = df[df.season == s]
        if len(te) == 0:
            continue
        mt = TS.make_model().fit(TS.prep_X(tr, feats), tr["tries"].values)
        mg = KK.make_model().fit(KK.prep_X(tr, feats), tr["goals"].values)
        lt = np.clip(mt.predict(TS.prep_X(te, feats)), 0, None)
        lg = np.clip(mg.predict(KK.prep_X(te, feats)), 0, None)
        lfg = np.clip(te["fieldGoals_r10"].fillna(0).values, 0, None)
        pred_pts.append(4 * lt + 2 * lg + lfg)
        real_pts.append(te["points"].values)
        lt_all.append(lt); lg_all.append(lg); lfg_all.append(lfg)
    P = np.concatenate(pred_pts); Y = np.concatenate(real_pts)
    lt = np.concatenate(lt_all); lg = np.concatenate(lg_all); lfg = np.concatenate(lfg_all)
    base = None
    mae = round(float(np.abs(P - Y).mean()), 3)

    # over/under reliability: place lines at expected +/- offsets, check P(over) vs reality
    # (sample to keep the convolution cost reasonable)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(Y), size=min(6000, len(Y)), replace=False)
    pp, rr = [], []
    for i in idx:
        exp = expected_points(lt[i], lg[i], lfg[i])
        for off in (-6, -3, -1, 2, 5):
            line = max(0.5, round(exp + off) - 0.5)
            po, pu, _ = p_over_under(lt[i], lg[i], lfg[i], line)
            pp.append(po); rr.append(1.0 if Y[i] > line else 0.0)
    pp = np.array(pp); rr = np.array(rr)
    bins = np.clip((pp * 10).astype(int), 0, 9)
    rel = {"pred": [], "emp": [], "n": []}
    for b in range(10):
        m = bins == b
        if m.sum() >= 50:
            rel["pred"].append(round(float(pp[m].mean()), 3))
            rel["emp"].append(round(float(rr[m].mean()), 3))
            rel["n"].append(int(m.sum()))
    cal_err = round(float(np.mean([abs(a - e) for a, e in zip(rel["pred"], rel["emp"])])), 3)
    return {"n_test": int(len(Y)), "mae_points": mae,
            "mean_points": round(float(Y.mean()), 2),
            "reliability": rel, "calibration_error": cal_err, "holdouts": list(track.holdouts)}


# --------------------------------------------------------------------------- predict round
def anchor_goalkickers(m, match_path=None,
                       window=10, min_kick_games=2, rate_cap=5.5):
    """Concentrate goals on each team's CURRENT primary kicker.

    The per-player goal model lags when kicking duty changes hands (a player who
    just took over the tee has a diluted rolling average, so he's under-rated, while
    a back-up who kicked a few keeps a stray goal rate). Fix it with the team list +
    recent form: within each named lineup, the kicker is whoever has kicked the most
    over the team's last `window` games; give him his rate in games he actually kicked
    (so a kicker returning from injury isn't dragged to zero by missed games) and zero
    everyone else. Falls back to the model's lg if no clear kicker.
    """
    match_path = match_path or T.proc("player_match.parquet")
    try:
        pm = pd.read_parquet(match_path, columns=["playerId", "season", "roundNumber",
                                                  "conversions", "penaltyGoals"])
    except Exception:
        return m  # no history available — leave the model's goal rates as-is
    pm["goals"] = pm["conversions"].fillna(0) + pm["penaltyGoals"].fillna(0)
    pm["order"] = pm["season"] * 100 + pm["roundNumber"].fillna(0)
    recent = pm.sort_values("order").groupby("playerId").tail(window)
    kicked = recent[recent["goals"] > 0]
    tot = recent.groupby("playerId")["goals"].sum()
    kick_games = kicked.groupby("playerId")["goals"].size()
    kick_rate = kicked.groupby("playerId")["goals"].mean()   # goals per *kicking* game

    m = m.copy()
    m["_tot"] = m["playerId"].map(tot).fillna(0.0)
    m["_kg"] = m["playerId"].map(kick_games).fillna(0).astype(int)
    m["_rate"] = m["playerId"].map(kick_rate).fillna(0.0).clip(upper=rate_cap)

    grp = "matchId" if "matchId" in m.columns else "opp"
    m["lg_model"] = m["lg"]
    m["lg"] = m["lg"].clip(upper=0.05)          # default: ~no goals for non-kickers
    for _, idx in m.groupby([grp, "team"]).groups.items():
        sub = m.loc[idx]
        top = sub["_tot"].idxmax()              # most goals over the window = the kicker
        if sub.loc[top, "_kg"] >= min_kick_games:
            m.loc[top, "lg"] = max(float(sub.loc[top, "_rate"]),
                                   float(sub.loc[top, "lg_model"]))
    return m.drop(columns=["_tot", "_kg", "_rate", "lg_model"])


def predict_round(track=None):
    track = track or T.current()
    try:
        tdf = pd.read_parquet(T.report("tryscorer_predictions.parquet", track))
        kdf = pd.read_parquet(T.report("kicker_predictions.parquet", track))
    except FileNotFoundError as e:
        print("need try + kicker predictions first:", e)
        return pd.DataFrame()
    tcols = ["playerId", "name", "team", "opp", "position", "lt"]
    if "matchId" in tdf.columns:
        tcols.append("matchId")
    t = tdf.rename(columns={"lambda": "lt"})[tcols]
    k = kdf.rename(columns={"lambda_goals": "lg", "lambda_fg": "lfg"})[["playerId", "lg", "lfg"]]
    m = t.merge(k, on="playerId", how="left").fillna({"lg": 0.0, "lfg": 0.0})
    m = anchor_goalkickers(m, match_path=T.proc("player_match.parquet", track))
    m["exp_points"] = expected_points(m["lt"], m["lg"], m["lfg"])
    m["exp_tries"] = m["lt"]
    m["exp_kicker_points"] = 2 * m["lg"] + m["lfg"]
    m = m.sort_values("exp_points", ascending=False)
    T.ensure_dirs(track)
    m.to_parquet(T.report("player_points_predictions.parquet", track), index=False)
    return m


def price_snapshot(odds_path=None, pred_path=None, out_parquet=None, out_json=None):
    """Value live player-points and goals over/under markets against the model."""
    odds_path = odds_path or T.report("odds_snapshot.parquet")
    pred_path = pred_path or T.report("player_points_predictions.parquet")
    out_parquet = out_parquet or T.report("points_edges.parquet")
    out_json = out_json or T.report("points_edges.json")
    import pricing as PRC
    try:
        odds = pd.read_parquet(odds_path)
        preds = pd.read_parquet(pred_path)
    except FileNotFoundError as e:
        print("points pricing skipped:", e)
        return pd.DataFrame()
    pb = preds.set_index("playerId")
    ou = odds[(odds["stat"].isin(["points", "goals", "kicker_points"]))
              & (odds["over"].notna() | odds["under"].notna())
              & odds["playerId"].notna()]
    rows = []
    for _, r in ou.iterrows():
        pid = r["playerId"]
        if pid not in pb.index or r.get("line") is None:
            continue
        pr = pb.loc[pid]
        if getattr(pr, "ndim", 1) > 1:
            pr = pr.iloc[0]
        lt, lg, lfg = float(pr["lt"]), float(pr["lg"]), float(pr["lfg"])
        line = float(r["line"])
        if r["stat"] == "points":
            p_over, p_under, p_push = p_over_under(lt, lg, lfg, line)
            model_mean = expected_points(lt, lg, lfg)
        elif r["stat"] == "kicker_points":  # 2*goals + field goals  (set tries rate to 0)
            p_over, p_under, p_push = p_over_under(0.0, lg, lfg, line)
            model_mean = 2 * lg + lfg
        else:  # goals (Poisson on lambda_goals)
            p_under = float(poisson.cdf(math.floor(line), lg))
            p_over = 1 - p_under
            p_push = 0.0
            model_mean = lg
        m_over, m_under = PRC.devig_two_way(r.get("over"), r.get("under"))
        ev_over = PRC.ev_per_dollar(p_over, p_push, r.get("over"))
        ev_under = PRC.ev_per_dollar(p_under, p_push, r.get("under"))
        cands = []
        if r.get("over") is not None:
            cands.append(("over", ev_over, r["over"], p_over, m_over))
        if r.get("under") is not None:
            cands.append(("under", ev_under, r["under"], p_under, m_under))
        if not cands:
            continue
        side, ev, price, mp, mkt = max(cands, key=lambda c: (c[1] if c[1] is not None else -9))
        rows.append({"book": r["book"], "player": pr["name"], "playerId": pid,
                     "team": pr["team"], "stat": r["stat"], "line": line,
                     "model_mean": round(model_mean, 1), "model_p_over": round(p_over, 3),
                     "best_side": side, "best_price": price,
                     "my_price": PRC.fair_odds(mp), "mkt_p": round(mkt, 3) if mkt else None,
                     "ev_pct": round(ev * 100, 1) if ev is not None else None,
                     "event_name": r.get("event_name"), "fetched_at": r.get("fetched_at")})

    # genuine fixed-odds "To Score N+ Points" alt-lines (1-way over, single price)
    nplus = odds[(odds["stat"] == "points") & (odds.get("kind") == "pts_plus")
                 & odds["single"].notna() & odds["playerId"].notna()]
    for _, r in nplus.iterrows():
        pid = r["playerId"]
        if pid not in pb.index or r.get("line") is None:
            continue
        pr = pb.loc[pid]
        if getattr(pr, "ndim", 1) > 1:
            pr = pr.iloc[0]
        lt, lg, lfg = float(pr["lt"]), float(pr["lg"]), float(pr["lfg"])
        line, price = float(r["line"]), float(r["single"])
        pmf = points_pmf(lt, lg, lfg)
        p_over = sum(p for v, p in pmf.items() if v > line)
        ev = p_over * price - 1
        rows.append({"book": r["book"], "player": pr["name"], "playerId": pid,
                     "team": pr["team"], "stat": "points", "line": int(line + 0.5),
                     "model_mean": round(expected_points(lt, lg, lfg), 1),
                     "model_p_over": round(p_over, 3), "best_side": f"{int(line+0.5)}+",
                     "best_price": price, "my_price": PRC.fair_odds(p_over), "mkt_p": None,
                     "ev_pct": round(ev * 100, 1),
                     "event_name": r.get("event_name"), "fetched_at": r.get("fetched_at")})
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values("ev_pct", ascending=False, na_position="last")
    edges.to_parquet(out_parquet, index=False)
    edges.to_json(out_json, orient="records")
    pos = int((edges["ev_pct"] > 0).sum()) if len(edges) else 0
    print(f"Wrote {out_parquet}: {len(edges)} points/goals markets, {pos} +EV")
    return edges


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    track = T.current()
    T.ensure_dirs(track)
    if cmd == "price":
        price_snapshot()
        return
    # Origin tracks reuse the club models upstream (try/kicker); points just combine
    # them, so skip the club-sized backtest and go straight to the round.
    if track.name != track.model_track:
        payload = {"generated": pd.Timestamp.now("UTC").isoformat(),
                   "reused_model": track.model_track}
    else:
        print(f"[{track.name}] player-points backtest…")
        bt = backtest(track)
        print(f"  points MAE {bt['mae_points']} (mean {bt['mean_points']}); "
              f"over/under calib err {bt['calibration_error']} over {bt['n_test']:,} rows")
        payload = {"generated": pd.Timestamp.now("UTC").isoformat(), "backtest": bt}

    if cmd != "backtest":
        m = predict_round(track)
        if not m.empty:
            payload["leaders"] = [
                {"name": r["name"], "team": r["team"], "opp": r["opp"],
                 "exp_points": round(float(r["exp_points"]), 1),
                 "exp_tries": round(float(r["exp_tries"]), 2),
                 "exp_kicker_points": round(float(r["exp_kicker_points"]), 1)}
                for _, r in m.head(20).iterrows() if r["name"]]
            print(f"  predicted {len(m)} players; top {m.iloc[0]['name']} "
                  f"{m.iloc[0]['exp_points']:.1f} pts")
    dest = T.report("player_points.json", track)
    json.dump(payload, open(dest, "w"))
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
