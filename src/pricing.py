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

TARGETS = ["runsHitup", "runs", "runMetres", "postContactMetres", "tackles", "perf_points"]
STAT_TO_TARGET = {
    "tackles": "tackles", "run_metres": "runMetres",
    "post_contact_metres": "postContactMetres", "runs": "runs",
    "fantasy": "perf_points",
}
DISP_PATH = "models/dispersion.json"
# minimum sigma floor per target (avoid absurd certainty on tiny means)
SIGMA_FLOOR = {"runsHitup": 1.0, "runs": 1.8, "runMetres": 15.0,
               "postContactMetres": 6.0, "tackles": 2.5, "perf_points": 8.0}


# --------------------------------------------------------------------------- calibration
def calibrate_dispersion(features="data/processed/features.parquet",
                         model="models/nrl_models.joblib",
                         meta="data/processed/feature_cols.json",
                         out=DISP_PATH, holdout_seasons=(2023, 2024, 2025)):
    """Fit sigma(mu) = alpha + beta*mu per target from out-of-time residuals."""
    import joblib
    df = pd.read_parquet(features)
    feats = json.load(open(meta))["features"]
    bundle = joblib.load(model)
    te = df[df.season.isin(holdout_seasons)].copy()
    X = te[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")

    disp = {}
    for t in TARGETS:
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
        disp[t] = {"alpha": round(float(alpha), 4), "beta": round(float(beta), 4),
                   "sigma_floor": SIGMA_FLOOR.get(t, 1.0),
                   "global_sd": round(float(resid.std()), 3),
                   "mean_pred": round(float(pred.mean()), 2)}
    json.dump(disp, open(out, "w"), indent=2)
    print(f"Wrote {out}")
    for t, d in disp.items():
        print(f"  {t:20s} sigma = {d['alpha']:.2f} + {d['beta']:.3f}*mu "
              f"(floor {d['sigma_floor']}, global_sd {d['global_sd']})")
    return disp


def load_dispersion(path=DISP_PATH):
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
def price_snapshot(odds_path="reports/odds_snapshot.parquet",
                   preds_path="reports/round_predictions.parquet",
                   disp_path=DISP_PATH,
                   out_parquet="reports/edges.parquet", out_json="reports/edges.json"):
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


def price_tries(odds_path="reports/odds_snapshot.parquet",
                pred_path="reports/tryscorer_predictions.parquet",
                out_parquet="reports/try_edges.parquet", out_json="reports/try_edges.json"):
    """Value live try-scorer odds against the try model's calibrated probabilities."""
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
        mkt = _try_market(r.get("market_raw"), r.get("line"))
        if mkt in ("first", "3+"):  # we model anytime + 2+ cleanly
            if mkt == "first":
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
    else:
        selftest()
