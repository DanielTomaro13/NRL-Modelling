"""
Export a compact, stable JSON bundle for the NRL-24-0 site to consume.

This repo stays the data engine: the Python model pipeline runs here and writes
reports/*.json. The nrl24-0.com Next.js app fetches the small bundle below (NOT the
1.8 MB odds_snapshot) at build time and renders native pages from it.

Writes reports/site/:
  meta.json         { round, updated, generated }
  predictions.json  per-match player projections (tries / points / kicker)
  compare.json      odds comparison (reuse of comparison.json)
  pickem.json       model-vs-line Pick'em rows + dist params (reuse of pickem.json)
  scoring.json      player-points & try-scorer leaders with model price + best book

CLI: python src/export_site_data.py
"""
import datetime as dt
import json
import os
import shutil
import glob
import pandas as pd
import tracks as T

AEST = dt.timezone(dt.timedelta(hours=10))


def now_aest():
    return dt.datetime.now(AEST)


def _load(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _round(x, n=2):
    return round(float(x), n) if pd.notna(x) else None


def build_predictions(track):
    """Per-match player rows with the headline model projections."""
    tdf = _load(T.report("tryscorer_predictions.parquet", track))
    pdf = _load(T.report("player_points_predictions.parquet", track))
    if not len(tdf) and not len(pdf):
        return {"matches": []}
    cols_t = ["playerId", "name", "team", "opp", "position", "matchId",
              "p_anytime", "exp_tries"]
    t = tdf[[c for c in cols_t if c in tdf]] if len(tdf) else pd.DataFrame()
    p = (pdf[["playerId", "exp_points", "exp_kicker_points"]]
         if len(pdf) else pd.DataFrame(columns=["playerId"]))
    df = t.merge(p, on="playerId", how="outer") if len(t) else pdf
    # expected performance (fantasy) points from the round predictions
    rdf = _load(T.report("round_predictions.parquet", track))
    if len(rdf) and "pred_perf_points" in rdf:
        df = df.merge(rdf[["playerId", "pred_perf_points"]], on="playerId", how="left")
    matches = []
    for mid, g in df.groupby("matchId"):
        g0 = g.iloc[0]
        players = []
        for _, r in g.sort_values("exp_points", ascending=False, na_position="last").iterrows():
            players.append({
                "playerId": int(r["playerId"]) if pd.notna(r.get("playerId")) else None,
                "name": r.get("name"), "team": r.get("team"), "pos": r.get("position"),
                "p_anytime": _round(r.get("p_anytime"), 3),
                "exp_tries": _round(r.get("exp_tries"), 2),
                "exp_points": _round(r.get("exp_points"), 1),
                "exp_kicker": _round(r.get("exp_kicker_points"), 1),
                "exp_perf": _round(r.get("pred_perf_points"), 1),
            })
        matches.append({"matchId": str(mid), "team": g0.get("team"), "opp": g0.get("opp"),
                        "event": f'{g0.get("team")} vs {g0.get("opp")}', "players": players})
    matches.sort(key=lambda m: m["event"])
    return {"matches": matches}


def build_team_markets(track):
    """Model fair odds for H2H / line / total per match (match-outcome model)."""
    tm = _load(T.report("team_edges.parquet", track))
    if not len(tm):
        return {"matches": []}
    matches = []
    for _, r in tm.iterrows():
        row = {
            "matchId": str(r.get("matchId")), "round": int(r["round"]) if pd.notna(r.get("round")) else None,
            "home": r.get("home"), "away": r.get("away"), "start": r.get("start_iso"),
            "pred_margin": _round(r.get("pred_margin"), 1), "pred_total": _round(r.get("pred_total"), 1),
            "p_home": _round(r.get("p_home"), 3), "p_away": _round(r.get("p_away"), 3),
            "fair_home": _round(r.get("fair_home"), 2), "fair_away": _round(r.get("fair_away"), 2),
            "line_home": _round(r.get("line_home"), 1),
            "total_line": _round(r.get("total_line"), 1),
            "fair_over": _round(r.get("fair_over"), 2), "fair_under": _round(r.get("fair_under"), 2),
        }
        # best live book prices + EV, when the odds snapshot carried this match
        # (men's cron only for now — fair-odds-only rows just omit these keys)
        for k, nd in [("book_home", None), ("book_home_price", 2), ("ev_home", 1),
                      ("book_away", None), ("book_away_price", 2), ("ev_away", 1),
                      ("book_over", None), ("book_over_price", 2), ("book_total_line", 1), ("ev_over", 1),
                      ("book_under", None), ("book_under_price", 2), ("book_under_line", 1), ("ev_under", 1),
                      ("book_line", None), ("book_line_side", None), ("book_line_hcap", 1),
                      ("book_line_price", 2), ("ev_line", 1)]:
            v = r.get(k)
            if v is not None and (not isinstance(v, float) or pd.notna(v)):
                row[k] = _round(v, nd) if nd is not None else v
        matches.append(row)
    matches.sort(key=lambda m: m.get("start") or "")
    return {"matches": matches}


def build_scoring(track):
    """Player-points edges (model price + best book) + top try scorers."""
    pe = _load(T.report("points_edges.parquet", track))
    points = []
    if len(pe):
        pe = pe[pe["stat"] == "points"] if "stat" in pe else pe
        # best (lowest line surfaces strongest) — keep one row per player: best EV
        seen = set()
        for _, e in pe.sort_values("ev_pct", ascending=False, na_position="last").iterrows():
            pid = e.get("playerId")
            if pid in seen:
                continue
            seen.add(pid)
            points.append({
                "player": e.get("player"), "team": e.get("team"),
                "line": _round(e.get("line"), 1), "model_mean": _round(e.get("model_mean"), 1),
                "my_price": _round(e.get("my_price"), 2), "book": e.get("book"),
                "best_price": _round(e.get("best_price"), 2), "ev": _round(e.get("ev_pct"), 1),
            })
    tdf = _load(T.report("tryscorer_predictions.parquet", track))
    tries = []
    if len(tdf):
        for _, r in tdf.sort_values("p_anytime", ascending=False).head(60).iterrows():
            tries.append({"player": r.get("name"), "team": r.get("team"),
                          "p_anytime": _round(r.get("p_anytime"), 3),
                          "exp_tries": _round(r.get("exp_tries"), 2)})
    return {"points": points, "tries": tries}


def build_lineups(track):
    """Confirmed team lists per match, with the designated goal kicker flagged."""
    files = sorted(glob.glob(T.proc("lineups_r*.parquet", track)))
    if not files:
        return {"matches": []}
    lu = pd.read_parquet(files[-1])   # latest round's confirmed lists
    pp = _load(T.report("player_points_predictions.parquet", track))
    team_of, lg_of = {}, {}
    if len(pp):
        team_of = dict(zip(pp["playerId"], pp["team"]))
        lg_of = dict(zip(pp["playerId"], pp["lg"]))
    # kicker per (matchId, squadId) = highest goal rate (>1) in that squad
    kicker_pid = {}
    for (mid, sid), g in lu.groupby(["matchId", "squadId"]):
        cand = [(lg_of.get(p, 0.0), p) for p in g["playerId"]]
        best = max(cand, default=(0, None))
        if best[0] > 1.0:
            kicker_pid[(mid, sid)] = best[1]

    def team_name(g):
        names = [team_of.get(p) for p in g["playerId"] if team_of.get(p)]
        return max(set(names), key=names.count) if names else None

    def players(g, mid, sid):
        out = []
        for _, r in g.sort_values("jumperNumber").iterrows():
            out.append({
                "playerId": int(r["playerId"]) if pd.notna(r.get("playerId")) else None,
                "name": r.get("name"), "position": r.get("position"),
                "jumper": int(r["jumperNumber"]) if pd.notna(r.get("jumperNumber")) else None,
                "kicker": kicker_pid.get((mid, sid)) == r.get("playerId"),
            })
        return out

    matches = []
    for mid, mg in lu.groupby("matchId"):
        sides = {}
        for sid, g in mg.groupby("squadId"):
            home = bool(g["isHome"].iloc[0])
            sides["home" if home else "away"] = {"team": team_name(g),
                                                 "players": players(g, mid, sid)}
        if "home" in sides and "away" in sides:
            matches.append({"matchId": str(mid),
                            "event": f'{sides["home"]["team"]} vs {sides["away"]["team"]}',
                            **sides})
    matches.sort(key=lambda m: m["event"])
    return {"matches": matches}


def _read_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def build_backtest(track):
    """Out-of-sample accuracy: try-scorer classification + per-stat regression MAE."""
    analysis = _read_json(T.report("analysis.json", track))
    ts = _read_json(T.report("tryscorer.json", track))
    abt = analysis.get("backtest", {})
    tbt = ts.get("backtest", {})

    tries = None
    if tbt:
        m = tbt.get("model", {})
        rel = tbt.get("reliability", {})
        tries = {
            "n_test": tbt.get("n_test"), "base_rate": tbt.get("base_rate"),
            "auc": m.get("auc"), "brier": m.get("brier"), "logloss": m.get("logloss"),
            "auc_baseline": (tbt.get("baseline_trailing5") or {}).get("auc"),
            "calibration_error": tbt.get("calibration_error"),
            "reliability": {"pred": rel.get("pred", []), "emp": rel.get("emp", [])},
        }
    # per-stat regression: model MAE vs trailing-5 baseline + improvement
    regression = []
    for r in abt.get("summary", []):
        regression.append({
            "target": r.get("target"), "label": r.get("label"),
            "mae_model": r.get("MAE_model"), "mae_base": r.get("MAE_base_r5"),
            "gain_pct": r.get("gain_pct"), "n": r.get("n"),
        })
    return {
        "holdouts": abt.get("holdouts") or tbt.get("holdouts") or [],
        "n_test": abt.get("n_test") or tbt.get("n_test"),
        "tries": tries,
        "regression": regression,
        "generated": analysis.get("generated"),
    }


def main():
    track = T.current()
    OUT = T.site_dir(track)
    if os.path.isdir(OUT):
        # The men's legacy site dir is the SHARED ROOT (reports/site) and the other
        # tracks' bundles live in subdirectories of it — rmtree here nukes the
        # NRLW/Origin bundles every 3-hourly cron export. Only remove this track's
        # own files; leave subdirectories alone.
        for f in os.listdir(OUT):
            p = os.path.join(OUT, f)
            # supercoach.json is produced by supercoach.py on its own cadence —
            # not ours to delete
            if os.path.isfile(p) and f != "supercoach.json":
                os.unlink(p)
    os.makedirs(OUT, exist_ok=True)

    rnd = None
    try:
        preds = pd.read_parquet(T.report("round_predictions.parquet", track))
        if "roundNumber" in preds:
            rnd = int(preds["roundNumber"].iloc[0])
    except Exception:
        pass
    # which player-prop targets are provisional (low-sample) for this track
    provisional = []
    try:
        import joblib
        bundle = joblib.load(T.model_for_prediction("nrl_models.joblib", track))
        provisional = sorted(t for t, m in bundle.get("targets_meta", {}).items()
                             if m.get("provisional"))
    except Exception:
        pass
    meta = {"comp": track.name, "label": track.label, "round": rnd,
            "provisional_targets": provisional,
            "updated": now_aest().strftime("%a %d %b %Y, %I:%M%p AEST"),
            "generated": pd.Timestamp.now("UTC").isoformat()}
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    json.dump(build_predictions(track), open(f"{OUT}/predictions.json", "w"))
    json.dump(build_scoring(track), open(f"{OUT}/scoring.json", "w"))
    json.dump(build_backtest(track), open(f"{OUT}/backtest.json", "w"))
    json.dump(build_lineups(track), open(f"{OUT}/lineups.json", "w"))
    json.dump(build_team_markets(track), open(f"{OUT}/team.json", "w"))

    # reuse the rich JSON already produced by compare.py / pickem.py verbatim
    for name in ("comparison", "pickem"):
        src = T.report(f"{name}.json", track)
        if os.path.exists(src):
            dst = f"{OUT}/{'compare' if name == 'comparison' else name}.json"
            shutil.copyfile(src, dst)

    sizes = {f: os.path.getsize(f"{OUT}/{f}") for f in sorted(os.listdir(OUT))}
    print(f"[{track.name}] wrote {OUT}/ :")
    for f, s in sizes.items():
        print(f"  {s:>9,d}  {f}")


if __name__ == "__main__":
    main()
