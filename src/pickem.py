"""
Dabble Pick'em — model vs line.

Pick'em is a multiplier/parlay product, not a priced two-way market: you pick players
to go over/under a line, min 2 legs, fixed multipliers (2 legs x3.2, 3 x6.5, 4 x12,
5 x25). There's no single-bet price to de-vig, so instead we treat Dabble's line as
their projection and let the model judge it:

  - model projection of the stat
  - P(over the line) / P(under)   (from the same calibrated models the site uses)
  - model fair odds (1 / P) and a lean

A parlay is +EV when  multiplier_N * product(P(your side)) > 1, so pick the legs where
the model most strongly beats Dabble's line.

Output: reports/pickem.json   CLI: python src/pickem.py
"""
import json, math
import numpy as np
import pandas as pd
from scipy.stats import poisson

import pricing as PRC
import player_points as PP

STAT_LABEL = {"points": "Player Points", "tries": "Tries",
              "performance_points": "Performance Points", "tackles": "Tackles",
              "run_metres": "Run Metres", "tackle_breaks": "Tackle Breaks",
              "kicker_points": "Kicker Points"}
MULTIPLIERS = {2: 3.2, 3: 6.5, 4: 12.0, 5: 25.0}   # Dabble Power Play (1 leg not allowed)


def _p_over(stat, line, tries_row, pts_row, rp_row, disp):
    """Model P(actual > line) for a Pick'em stat."""
    if stat == "tries" and tries_row is not None:
        lam = float(tries_row["lambda"])
        return float(1 - poisson.cdf(math.floor(line), lam))    # P(tries >= floor+1)
    if stat == "points" and pts_row is not None:
        pmf = PP.points_pmf(float(pts_row["lt"]), float(pts_row["lg"]), float(pts_row["lfg"]))
        return float(sum(p for v, p in pmf.items() if v > line))
    target = PRC.STAT_TO_TARGET.get(stat)               # perf_points / tackles / runMetres…
    if target and rp_row is not None and f"pred_{target}" in rp_row:
        mu = float(rp_row[f"pred_{target}"])
        sigma = PRC.sigma_for(target, mu, disp)
        return float(PRC.over_under_probs(mu, sigma, float(line))[0])
    return None


def _proj(stat, tries_row, pts_row, rp_row):
    if stat == "tries" and tries_row is not None:
        return round(float(tries_row["lambda"]), 2)
    if stat == "points" and pts_row is not None:
        return round(PP.expected_points(float(pts_row["lt"]), float(pts_row["lg"]),
                                        float(pts_row["lfg"])), 1)
    target = PRC.STAT_TO_TARGET.get(stat)
    if target and rp_row is not None and f"pred_{target}" in rp_row:
        return round(float(rp_row[f"pred_{target}"]), 1)
    return None


def main():
    od = pd.read_parquet("reports/odds_snapshot.parquet")
    pk = od[(od.get("category") == "pickem") & od.playerId.notna()] if "category" in od else od.iloc[:0]
    disp = PRC.load_dispersion() or {}
    tdf = _load("reports/tryscorer_predictions.parquet")
    pdf = _load("reports/player_points_predictions.parquet")
    rdf = _load("reports/round_predictions.parquet")
    ti = tdf.set_index("playerId") if len(tdf) else tdf
    pi = pdf.set_index("playerId") if len(pdf) else pdf
    ri = rdf.set_index("playerId") if len(rdf) else rdf

    # one row per (player, stat, line) carrying BOTH over and under + the sides Dabble offers
    groups = {}
    for _, r in pk.iterrows():
        key = (r["playerId"], r["stat"], float(r["line"]))
        g = groups.setdefault(key, {"player": r["player"], "event": r.get("event_name"),
                                    "sides": set()})
        g["sides"].add((r.get("kind") or "over").lower())
    rows = []
    for (pid, stat, line), g in groups.items():
        tr = _one(ti.loc[pid]) if (len(ti) and pid in ti.index) else None
        pr = _one(pi.loc[pid]) if (len(pi) and pid in pi.index) else None
        rr = _one(ri.loc[pid]) if (len(ri) and pid in ri.index) else None
        p_over = _p_over(stat, line, tr, pr, rr, disp)
        if p_over is None:
            continue
        p_under = 1 - p_over
        proj = _proj(stat, tr, pr, rr)
        team = None
        for src in (pr, tr, rr):
            if src is not None and src.get("team"):
                team = src.get("team"); break
        rows.append({"player": g["player"], "team": team, "stat": stat,
                     "stat_label": STAT_LABEL.get(stat, stat), "event": g["event"],
                     "line": line, "offered": sorted(g["sides"]),
                     "model_proj": proj,
                     "p_over": round(p_over, 3), "p_under": round(p_under, 3),
                     "fair_over": round(1 / p_over, 2) if p_over > 1e-9 else None,
                     "fair_under": round(1 / p_under, 2) if p_under > 1e-9 else None,
                     "lean": "OVER" if p_over >= 0.5 else "UNDER",
                     "lean_p": round(max(p_over, p_under), 3)})
    # sort by player, then stat, then line (so a player's lines are grouped)
    rows.sort(key=lambda r: (str(r["player"]).lower(), r["stat_label"], r["line"]))
    out = {"generated": pd.Timestamp.now("UTC").isoformat(), "multipliers": MULTIPLIERS,
           "rows": rows, "stats": sorted(set(r["stat_label"] for r in rows))}
    json.dump(out, open("reports/pickem.json", "w"))
    n_strong = sum(1 for r in rows if r["lean_p"] >= 0.6)
    print(f"Wrote reports/pickem.json: {len(rows)} Pick'em lines "
          f"({n_strong} with a model lean >=60%)")
    by = {}
    for r in rows:
        by[r["stat_label"]] = by.get(r["stat_label"], 0) + 1
    print("  by stat:", by)


def _load(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _one(row):
    if row is None:
        return None
    return row.iloc[0] if getattr(row, "ndim", 1) > 1 else row


if __name__ == "__main__":
    main()
