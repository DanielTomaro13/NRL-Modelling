"""
Distribution-based pricing + edge detection.

The model predicts the MEAN of each player target. A bookmaker prices the whole
DISTRIBUTION. To compare and find value we:

  1. Turn each model mean into a calibrated Normal(mean, sigma) predictive
     distribution. sigma scales with the mean (heteroscedastic counts), fitted as
     sigma = alpha + beta*mean from out-of-time model residuals.
  2. Price any posted over/under line off the Normal CDF, with the bookmaker's
     integer-line push band (+-0.5) and quarter-line 50/50 split handling.
  3. De-vig the book's over+under pair to the market-implied probabilities.
  4. Compute EV for each side and flag value.

This is an original Python implementation of the distribution-pricing method; the
push-band and quarter-line conventions follow standard bookmaker practice.

CLI:
  python src/pricing.py calibrate   # fit models/dispersion.json from residuals
  python src/pricing.py price        # write reports/edges.{parquet,json} from odds + preds
  python src/pricing.py selftest     # sanity-check the maths
"""
import sys, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm
import tracks as T

TARGETS = ["runsHitup", "runs", "runMetres", "postContactMetres", "tackles", "perf_points"]
STAT_TO_TARGET = {
    "tackles": "tackles", "run_metres": "runMetres",
    "post_contact_metres": "postContactMetres", "runs": "runs",
    "performance_points": "perf_points",  # the industry "Performance Points" market
    "fantasy": "perf_points",
}
DISP_PATH = "models/dispersion.json"
# minimum sigma floor per target (avoid absurd certainty on tiny means)
SIGMA_FLOOR = {"runsHitup": 1.0, "runs": 1.8, "runMetres": 15.0,
               "postContactMetres": 6.0, "tackles": 2.5, "perf_points": 8.0}


# --------------------------------------------------------------------------- calibration
def calibrate_dispersion(track=None, features=None, model=None, meta=None, out=None):
    """Fit sigma(mu) = alpha + beta*mu per target from out-of-time residuals.

    Preferred residual source: the walk-forward OOS predictions train.py stacks
    into <track>/oos_predictions.parquet (each holdout season predicted by a
    model trained strictly before it). Falls back to re-predicting the holdout
    seasons with the production bundle — but note that bundle TRAINS through
    those seasons, so fallback residuals are in-sample and sigma runs low.
    """
    import joblib
    track = track or T.current()
    features = features or T.proc("features.parquet", track)
    model = model or T.model("nrl_models.joblib", track)
    meta = meta or T.proc("feature_cols.json", track)
    out = out or T.model("dispersion.json", track)

    oos = None
    try:
        oos = pd.read_parquet(T.proc("oos_predictions.parquet", track))
        print(f"[{track.name}] dispersion from {len(oos):,} walk-forward OOS predictions")
    except Exception:
        print(f"[{track.name}] WARNING: no oos_predictions.parquet — falling back "
              f"to in-sample residuals (sigma will run low); re-run train.py")

    df = pd.read_parquet(features)
    feats = json.load(open(meta))["features"]
    bundle = joblib.load(model)
    avail = set(bundle["models"])

    disp = {}
    for t in TARGETS:
        if t not in avail:
            continue  # target not modelled for this track
        if oos is not None and (oos["target"] == t).any():
            sub = oos[oos["target"] == t]
            pred = np.clip(sub["pred"].to_numpy(float), 0, None)
            resid = sub["y"].to_numpy(float) - pred
        else:
            seasons = (track.train_max,) if t in track.provisional_targets else track.holdouts
            te = df[df.season.isin(seasons)].copy()
            X = te[feats + ["position"]].copy()
            X["position"] = X["position"].astype("category")
            pred = np.clip(bundle["models"][t].predict(X), 0, None)
            resid = te[t].values - pred
        # bin by predicted decile; robust per-bin residual std; fit sigma ~ mu
        qs = np.quantile(pred, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        mus, sds = [], []
        for i in range(len(qs) - 1):
            msk = (pred >= qs[i]) & (pred <= qs[i + 1])
            if msk.sum() >= 50:
                mus.append(float(pred[msk].mean()))
                sds.append(float(resid[msk].std()))
        mus, sds = np.array(mus), np.array(sds)
        if len(mus) >= 2:
            beta, alpha = np.polyfit(mus, sds, 1)
            beta = max(beta, 0.0)              # sigma must not shrink with mean
            alpha = max(alpha, 0.0)
        else:
            alpha, beta = float(resid.std()), 0.0
        # Provisional targets calibrate on a single season's round-holdout — genuinely
        # out-of-sample now, but one short season is a small residual sample, so keep
        # a safety widening until multi-season data exists.
        infl = 1.25 if t in track.provisional_targets else 1.0
        alpha *= infl; beta *= infl
        disp[t] = {"alpha": round(float(alpha), 4), "beta": round(float(beta), 4),
                   "sigma_floor": round(SIGMA_FLOOR.get(t, 1.0) * infl, 3),
                   "global_sd": round(float(resid.std()) * infl, 3),
                   "provisional": t in track.provisional_targets,
                   "mean_pred": round(float(pred.mean()), 2)}
    json.dump(disp, open(out, "w"), indent=2)
    print(f"Wrote {out}")
    for t, d in disp.items():
        print(f"  {t:20s} sigma = {d['alpha']:.2f} + {d['beta']:.3f}*mu "
              f"(floor {d['sigma_floor']}, global_sd {d['global_sd']})")
    return disp


def load_dispersion(path=None):
    path = path or T.model("dispersion.json")
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def sigma_for(target, mu, disp):
    d = disp[target]
    return max(d["alpha"] + d["beta"] * float(mu), d["sigma_floor"])


# --------------------------------------------------------------------------- pricing maths
def _frac(x):
    return x - math.floor(x)


def over_under_probs(mu, sigma, line):
    """Return (p_over, p_under, p_push) for an actual-vs-line over/under.

    Half-point line: no push. Integer line: push band [line-0.5, line+0.5].
    Quarter/three-quarter line: 50/50 average of the two adjacent half/integer lines.
    """
    sigma = max(float(sigma), 1e-6)
    f = round(_frac(line), 4)
    if f in (0.25, 0.75):  # quarter line -> average the two adjacent lines
        a = over_under_probs(mu, sigma, line - 0.25)
        b = over_under_probs(mu, sigma, line + 0.25)
        return tuple((a[i] + b[i]) / 2 for i in range(3))
    z = lambda x: norm.cdf((x - mu) / sigma)
    if abs(f - 0.5) < 1e-6:  # half-point line: clean, no push
        p_under = z(line)
        return 1 - p_under, p_under, 0.0
    # integer line: exact value is a push
    lo, hi = z(line - 0.5), z(line + 0.5)
    p_under = lo
    p_push = hi - lo
    p_over = 1 - hi
    return p_over, p_under, p_push


def fair_odds(p):
    return round(1.0 / p, 2) if p and p > 1e-9 else None


def devig_two_way(over_price, under_price):
    """Market-implied (p_over, p_under) after removing the bookmaker margin."""
    if not over_price or not under_price:
        return None, None
    io, iu = 1.0 / over_price, 1.0 / under_price
    s = io + iu
    return io / s, iu / s


def ev_per_dollar(p_win, p_push, price):
    """EV per $1 staked on a side priced at `price` (push returns the stake)."""
    if price is None:
        return None
    return p_win * price + p_push * 1.0 - 1.0


# --------------------------------------------------------------------------- price a snapshot
def price_snapshot(odds_path=None, preds_path=None, disp_path=None,
                   out_parquet=None, out_json=None):
    odds_path = odds_path or T.report("odds_snapshot.parquet")
    preds_path = preds_path or T.report("round_predictions.parquet")
    out_parquet = out_parquet or T.report("edges.parquet")
    out_json = out_json or T.report("edges.json")
    odds = pd.read_parquet(odds_path)
    preds = pd.read_parquet(preds_path)
    disp = load_dispersion(disp_path)
    if disp is None:
        print("No dispersion file — run `python src/pricing.py calibrate` first.")
        disp = {t: {"alpha": SIGMA_FLOOR[t], "beta": 0.0,
                    "sigma_floor": SIGMA_FLOOR[t]} for t in TARGETS}

    pred_by_pid = {}
    for _, p in preds.iterrows():
        pred_by_pid[p["playerId"]] = p

    # only price player over/under markets whose stat maps to a model target
    rows = []
    ou = odds[(odds["category"] == "player") & (odds["over"].notna() | odds["under"].notna())]
    for _, r in ou.iterrows():
        stat = r["stat"]
        target = STAT_TO_TARGET.get(stat)
        pid = r.get("playerId")
        if target is None or pid is None or pid not in pred_by_pid or r.get("line") is None:
            continue
        mu = float(pred_by_pid[pid][f"pred_{target}"])
        sigma = sigma_for(target, mu, disp)
        line = float(r["line"])
        p_over, p_under, p_push = over_under_probs(mu, sigma, line)
        m_over, m_under = devig_two_way(r.get("over"), r.get("under"))
        ev_over = ev_per_dollar(p_over, p_push, r.get("over"))
        ev_under = ev_per_dollar(p_under, p_push, r.get("under"))
        # pick the +EV side (if any)
        side, ev, price, p_win, mkt_p, model_p = None, None, None, None, None, None
        cands = []
        if r.get("over") is not None:
            cands.append(("over", ev_over, r["over"], p_over, m_over))
        if r.get("under") is not None:
            cands.append(("under", ev_under, r["under"], p_under, m_under))
        if cands:
            side, ev, price, p_win, mkt_p = max(cands, key=lambda c: (c[1] if c[1] is not None else -9))
            model_p = p_win
        rows.append({
            "book": r["book"], "player": r.get("player"), "playerId": pid,
            "team": pred_by_pid[pid].get("team"), "opp": pred_by_pid[pid].get("opp"),
            "event_name": r.get("event_name"), "stat": stat, "target": target,
            "model_mean": round(mu, 1), "sigma": round(sigma, 2), "line": line,
            "over_price": r.get("over"), "under_price": r.get("under"),
            "model_p_over": round(p_over, 3), "model_p_under": round(p_under, 3),
            "p_push": round(p_push, 3),
            "fair_over": fair_odds(p_over), "fair_under": fair_odds(p_under),
            "mkt_p_over": round(m_over, 3) if m_over else None,
            "best_side": side, "best_price": price,
            "edge_pct": round((model_p - mkt_p) * 100, 1) if (model_p is not None and mkt_p is not None) else None,
            "ev_pct": round(ev * 100, 1) if ev is not None else None,
            "start_iso": r.get("start_iso"), "fetched_at": r.get("fetched_at"),
        })
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values("ev_pct", ascending=False, na_position="last")
    edges.to_parquet(out_parquet, index=False)
    edges.to_json(out_json, orient="records")
    pos = int((edges["ev_pct"] > 0).sum()) if len(edges) else 0
    print(f"Wrote {out_parquet}/{out_json}: {len(edges)} priced player markets, {pos} +EV")
    if len(edges):
        cols = ["book", "player", "stat", "line", "model_mean", "best_side",
                "best_price", "fair_over", "edge_pct", "ev_pct"]
        print(edges[cols].head(20).to_string(index=False))
    return edges


# --------------------------------------------------------------------------- match markets (H2H / line / total)
def _norm(s):
    import re as _re
    return _re.sub(r"[^a-z]", "", (s or "").lower())


TOTAL_LINE_MIN = 20.0    # an NRL total below this is a parse artifact, not a market
HCAP_MAX = 40.0          # |handicap| beyond this is junk


def _best_match_odds(mk, home, away):
    """From a book match-market snapshot for one event, return best available
    head-to-head prices per side, line rows, and total over/under — each total
    side kept WITH its own line+book so EV is always priced at the line the
    price was posted for (books post different totals). Rows whose line/handicap
    failed to parse (None / 0 / implausible) are dropped."""
    hk, ak = _norm(home), _norm(away)
    best = {"home": (None, None), "away": (None, None),          # (price, book)
            "line": [],                                          # (kind, hcap, price, book)
            "total_over": (None, None, None),                    # (price, book, line)
            "total_under": (None, None, None)}
    for _, r in mk.iterrows():
        stat, kind, price = r.get("stat"), r.get("kind"), r.get("single")
        if stat == "head_to_head" and price and kind in ("home", "away"):
            if best[kind][0] is None or price > best[kind][0]:
                best[kind] = (float(price), r.get("book"))
        elif stat == "line" and price and kind in ("home", "away"):
            h = r.get("line")
            if h is not None and pd.notna(h) and abs(float(h)) <= HCAP_MAX:
                best["line"].append((kind, float(h), float(price), r.get("book")))
        elif stat == "total":
            ln = r.get("line")
            if ln is None or pd.isna(ln) or float(ln) < TOTAL_LINE_MIN:
                continue
            ln = float(ln)
            if r.get("over") and (best["total_over"][0] is None or r["over"] > best["total_over"][0]):
                best["total_over"] = (float(r["over"]), r.get("book"), ln)
            if r.get("under") and (best["total_under"][0] is None or r["under"] > best["total_under"][0]):
                best["total_under"] = (float(r["under"]), r.get("book"), ln)
    return best


def team_markets(track=None):
    """Model fair odds for head-to-head, line (handicap) and total, from the
    match-outcome model's per-match margin/total predictions, plus EV against the
    live book match markets in the odds snapshot (category=="match") if present.

    margin ~ Normal(mu_m, sd_m): P(home win) = Phi(mu_m / sd_m); line cover at
    handicap h (home -h) = P(margin > h). total ~ Normal(mu_t, sd_t): priced with
    the same push-band CDF as player props.
    """
    track = track or T.current()
    tp_path = T.report("team_predictions.parquet", track)
    try:
        tp = pd.read_parquet(tp_path)
    except FileNotFoundError:
        print(f"no {tp_path} — run team_model.py predict first")
        return pd.DataFrame()

    # optional live book match markets
    try:
        odds = pd.read_parquet(T.report("odds_snapshot.parquet", track))
        odds = odds[odds.get("category") == "match"] if "category" in odds else odds.iloc[0:0]
    except Exception:
        odds = pd.DataFrame()

    rows = []
    for _, r in tp.iterrows():
        mu_m, sd_m = float(r["pred_margin"]), max(float(r["sigma_margin"]), 1e-6)
        mu_t, sd_t = float(r["pred_total"]), max(float(r["sigma_total"]), 1e-6)
        p_home = float(norm.cdf(mu_m / sd_m))           # P(margin > 0)
        line = round(mu_m * 2) / 2                       # model's fair handicap (home -line)
        total_line = round(mu_t * 2) / 2
        p_over, p_under, _ = over_under_probs(mu_t, sd_t, total_line)
        row = {
            "matchId": r["matchId"], "round": int(r["roundNumber"]),
            "home": r["home"], "away": r["away"], "start_iso": r.get("start_iso"),
            "pred_margin": round(mu_m, 1), "pred_total": round(mu_t, 1),
            "p_home": round(p_home, 3), "p_away": round(1 - p_home, 3),
            "fair_home": fair_odds(p_home), "fair_away": fair_odds(1 - p_home),
            "line_home": -line, "line_away": line,
            "total_line": total_line, "p_over": round(p_over, 3), "p_under": round(p_under, 3),
            "fair_over": fair_odds(p_over), "fair_under": fair_odds(p_under),
        }
        # match this event's book rows by team names and compute EV on best prices
        if len(odds):
            hk, ak = _norm(r["home"]), _norm(r["away"])
            mk = odds[odds.apply(lambda x: {_norm(x.get("home")), _norm(x.get("away"))}
                                 == {hk, ak}, axis=1)]
            if len(mk):
                b = _best_match_odds(mk, r["home"], r["away"])
                if b["home"][0]:
                    row.update(book_home=b["home"][1], book_home_price=b["home"][0],
                               ev_home=round((p_home * b["home"][0] - 1) * 100, 1))
                if b["away"][0]:
                    row.update(book_away=b["away"][1], book_away_price=b["away"][0],
                               ev_away=round(((1 - p_home) * b["away"][0] - 1) * 100, 1))
                if b["total_over"][0]:
                    price, book, ln = b["total_over"]
                    po, _, _ = over_under_probs(mu_t, sd_t, ln)
                    row.update(book_over=book, book_over_price=price,
                               book_total_line=ln,
                               ev_over=round((po * price - 1) * 100, 1))
                if b["total_under"][0]:
                    price, book, ln = b["total_under"]
                    _, pu, _ = over_under_probs(mu_t, sd_t, ln)
                    row.update(book_under=book, book_under_price=price,
                               book_under_line=ln,
                               ev_under=round((pu * price - 1) * 100, 1))
                # best line: cover prob vs each posted side's SIGNED handicap h
                # (home -4.5 covers iff margin > +4.5 = -h; away +4.5 covers iff
                # margin < +4.5 = h). All five books post per-side signed lines.
                best_line_ev = None
                for kind, hcap, price, book in b["line"]:
                    if hcap is None:
                        continue
                    p = (1 - float(norm.cdf((-hcap - mu_m) / sd_m))) if kind == "home" \
                        else float(norm.cdf((hcap - mu_m) / sd_m))
                    ev = (p * price - 1) * 100
                    if best_line_ev is None or ev > best_line_ev["ev_line"]:
                        best_line_ev = {"book_line": book, "book_line_side": kind,
                                        "book_line_hcap": hcap, "book_line_price": price,
                                        "ev_line": round(ev, 1)}
                if best_line_ev:
                    row.update(best_line_ev)
        rows.append(row)
    out = pd.DataFrame(rows)

    T.ensure_dirs(track)
    out.to_parquet(T.report("team_edges.parquet", track), index=False)
    out.to_json(T.report("team_markets.json", track), orient="records")
    n_ev = int(out["ev_home"].notna().sum()) if "ev_home" in out else 0
    print(f"[{track.name}] wrote {T.report('team_markets.json', track)}: {len(out)} matches, "
          f"{n_ev} with live H2H odds")
    if len(out):
        show = [c for c in ["home", "away", "p_home", "fair_home", "book_home_price",
                            "ev_home", "fair_over", "fair_under"] if c in out]
        print(out[show].to_string(index=False))
    return out


# --------------------------------------------------------------------------- try-scorer pricing
def _try_market(market_raw, line):
    """Classify a try odds row into anytime / 2+ / 3+ / first (skip first)."""
    mr = (market_raw or "").lower()
    if "first" in mr:
        return "first"
    if line is not None:
        if line <= 0.75:
            return "anytime"
        if line <= 1.75:
            return "2+"
        if line <= 2.75:
            return "3+"
    if "anytime" in mr or "1+" in mr:
        return "anytime"
    return "anytime"


def price_tries(odds_path=None, pred_path=None, out_parquet=None, out_json=None):
    """Value live try-scorer odds against the try model's calibrated probabilities."""
    odds_path = odds_path or T.report("odds_snapshot.parquet")
    pred_path = pred_path or T.report("tryscorer_predictions.parquet")
    out_parquet = out_parquet or T.report("try_edges.parquet")
    out_json = out_json or T.report("try_edges.json")
    try:
        odds = pd.read_parquet(odds_path)
        preds = pd.read_parquet(pred_path)
    except FileNotFoundError as e:
        print("try pricing skipped:", e)
        return pd.DataFrame()
    pb = preds.set_index("playerId")
    tr = odds[(odds["stat"] == "tries") & odds["single"].notna() & odds["playerId"].notna()]
    best = {}  # (playerId, market) -> row dict, keep best (highest) price
    for _, r in tr.iterrows():
        pid = r["playerId"]
        if pid not in pb.index:
            continue
        # prefer the explicit try kind; fall back to inferring from name/line
        mkt = r.get("kind") if isinstance(r.get("kind"), str) else _try_market(r.get("market_raw"), r.get("line"))
        if mkt == "first":  # we don't model first-try-scorer (needs game script)
            continue
        prow = pb.loc[pid]
        if hasattr(prow, "iloc") and getattr(prow, "ndim", 1) > 1:
            prow = prow.iloc[0]
        lam = float(prow["lambda"])
        if mkt == "anytime":
            p = float(prow["p_anytime"])
        elif mkt == "2+":
            p = float(prow["p_2plus"])
        elif mkt == "3+":
            p = 1 - math.exp(-lam) * (1 + lam + lam ** 2 / 2)
        else:
            continue
        price = float(r["single"])
        ev = p * price - 1.0
        key = (pid, mkt)
        if key not in best or price > best[key]["price"]:
            best[key] = {"playerId": pid, "player": prow["name"], "team": prow["team"],
                         "opp": prow["opp"], "market": mkt, "book": r["book"],
                         "price": price, "model_p": round(p, 3),
                         "fair": fair_odds(p), "ev_pct": round(ev * 100, 1),
                         "event_name": r.get("event_name"), "fetched_at": r.get("fetched_at")}
    edges = pd.DataFrame(list(best.values()))
    if not edges.empty:
        edges = edges.sort_values("ev_pct", ascending=False)
    edges.to_parquet(out_parquet, index=False)
    edges.to_json(out_json, orient="records")
    pos = int((edges["ev_pct"] > 0).sum()) if len(edges) else 0
    print(f"Wrote {out_parquet}/{out_json}: {len(edges)} try markets priced, {pos} +EV")
    if len(edges):
        print(edges[["player", "market", "book", "price", "model_p", "fair", "ev_pct"]]
              .head(15).to_string(index=False))
    return edges


# --------------------------------------------------------------------------- self test
def selftest():
    # half-point line: P(over 20.5) with mu=22, sigma=5
    po, pu, pp = over_under_probs(22, 5, 20.5)
    assert abs((po + pu) - 1) < 1e-9 and pp == 0.0
    assert abs(po - (1 - norm.cdf((20.5 - 22) / 5))) < 1e-9
    # integer line has a push band
    po, pu, pp = over_under_probs(22, 5, 20)
    assert pp > 0 and abs(po + pu + pp - 1) < 1e-9
    # quarter line averages neighbours
    po25, _, _ = over_under_probs(22, 5, 20.25)
    a, _, _ = over_under_probs(22, 5, 20.0)
    b, _, _ = over_under_probs(22, 5, 20.5)
    assert abs(po25 - (a + b) / 2) < 1e-9
    # de-vig sums to 1
    mo, mu_ = devig_two_way(1.90, 1.90)
    assert abs(mo + mu_ - 1) < 1e-9 and abs(mo - 0.5) < 1e-9
    # EV: a fair-priced bet with no margin and matching prob -> ~0 EV
    p_over, _, push = over_under_probs(22, 5, 20.5)
    assert abs(ev_per_dollar(p_over, push, 1 / p_over)) < 1e-9
    # worked example
    mu, sigma, line, over, under = 22.0, 5.1, 19.5, 1.80, 1.95
    po, pu, pp = over_under_probs(mu, sigma, line)
    mo, mu2 = devig_two_way(over, under)
    print(f"example: tackles mu={mu} line={line}  model P(over)={po:.3f} "
          f"fair_over={fair_odds(po)}  book over={over} (devig {mo:.3f})  "
          f"EV_over={ev_per_dollar(po, pp, over)*100:+.1f}%")
    print("selftest OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "calibrate":
        calibrate_dispersion()
    elif cmd == "price":
        price_snapshot()
    elif cmd == "tries":
        price_tries()
    elif cmd == "team":
        team_markets()
    else:
        selftest()
