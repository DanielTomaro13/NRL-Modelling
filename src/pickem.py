"""
Dabble Pick'em — model vs line, with manual line entry.

Pick'em is a multiplier/parlay product, not a priced two-way market: you pick players
to go over/under a line, min 2 legs, fixed multipliers (2 legs x3.2, 3 x6.5, 4 x12,
5 x25). There's no single-bet price to de-vig, so the model judges the *line* instead.

Dabble is iOS-only, so we often can't pull its lines automatically. To stay useful, this
emits a MODEL ROW for every projected player across the Pick'em stat set, each carrying the
model's distribution parameters. The site computes P(over) for whatever line you type in —
prefilled with Dabble's posted line when we do have it. So you can paste a line off your
phone and get the model's read (P over/under, fair odds, lean) with zero scraping.

  - tries:  Poisson(lambda)                              -> {"k":"pois","lam":..}
  - points: 4*Pois(lt) + 2*Pois(lg) + Pois(lfg) (convol) -> {"k":"conv","lt":..,"lg":..,"lfg":..}
  - perf points / tackles / run metres: Normal(mu, sigma)-> {"k":"norm","mu":..,"sg":..}

Output: reports/pickem.json   CLI: python src/pickem.py
"""
import json
import numpy as np
import pandas as pd

import pricing as PRC

# stat key -> (display label, which Normal target in dispersion, prediction column)
STAT_LABEL = {"points": "Player Points", "tries": "Tries",
              "performance_points": "Performance Points", "tackles": "Tackles",
              "run_metres": "Run Metres"}
# Normal-distributed stats we can model from round_predictions + dispersion
NORM_STATS = {"performance_points": ("perf_points", "pred_perf_points"),
              "tackles": ("tackles", "pred_tackles"),
              "run_metres": ("runMetres", "pred_runMetres")}
# only surface a row when the projection is meaningful (keeps the board navigable)
MIN_PROJ = {"tries": 0.06, "points": 1.0, "performance_points": 6.0,
            "tackles": 5.0, "run_metres": 25.0}
MULTIPLIERS = {2: 3.2, 3: 6.5, 4: 12.0, 5: 25.0}   # Dabble Power Play (1 leg not allowed)


def _suggest_line(stat, proj):
    """A sensible default line to prefill (near the projection, half-point so no push)."""
    if stat == "tries":
        return 0.5
    return max(0.5, round(proj) - 0.5)


def _dabble_lines():
    """Map (playerId, stat) -> {'line': x, 'offered': [sides]} from any live Dabble Pick'em."""
    try:
        od = pd.read_parquet("reports/odds_snapshot.parquet")
    except Exception:
        return {}
    if "category" not in od:
        return {}
    pk = od[(od["category"] == "pickem") & od["playerId"].notna()]
    out = {}
    for (pid, stat), g in pk.groupby(["playerId", "stat"]):
        line = float(g["line"].mode().iloc[0]) if g["line"].notna().any() else None
        sides = sorted(set((k or "over").lower() for k in g.get("kind", [])))
        out[(pid, stat)] = {"line": line, "offered": sides}
    return out


def main():
    disp = PRC.load_dispersion() or {}
    tdf = _load("reports/tryscorer_predictions.parquet")
    pdf = _load("reports/player_points_predictions.parquet")
    rdf = _load("reports/round_predictions.parquet")
    dab = _dabble_lines()

    # base identity (name/team/opp/matchId) keyed by playerId, from whichever file has it
    info = {}
    for df in (pdf, tdf, rdf):
        if not len(df):
            continue
        for _, r in df.iterrows():
            pid = r.get("playerId")
            if pid is None or pid in info:
                continue
            opp = r.get("opp")
            info[pid] = {"player": r.get("name"), "team": r.get("team"),
                         "matchId": r.get("matchId"),
                         "event": f'{r.get("team")} vs {opp}' if opp else r.get("team")}

    ti = tdf.set_index("playerId") if len(tdf) else tdf
    pi = pdf.set_index("playerId") if len(pdf) else pdf
    ri = rdf.set_index("playerId") if len(rdf) else rdf

    rows = []
    for pid, meta in info.items():
        specs = []   # (stat, proj, dist)
        if len(ti) and pid in ti.index:
            lam = float(_one(ti.loc[pid])["lambda"])
            specs.append(("tries", lam, {"k": "pois", "lam": round(lam, 4)}))
        if len(pi) and pid in pi.index:
            pr = _one(pi.loc[pid])
            lt, lg, lfg = float(pr["lt"]), float(pr["lg"]), float(pr["lfg"])
            proj = round(4 * lt + 2 * lg + lfg, 1)
            specs.append(("points", proj, {"k": "conv", "lt": round(lt, 4),
                                           "lg": round(lg, 4), "lfg": round(lfg, 4)}))
        if len(ri) and pid in ri.index:
            rr = _one(ri.loc[pid])
            for stat, (target, col) in NORM_STATS.items():
                if col not in rr or pd.isna(rr[col]) or target not in disp:
                    continue
                mu = float(rr[col])
                sg = PRC.sigma_for(target, mu, disp)
                specs.append((stat, mu, {"k": "norm", "mu": round(mu, 2), "sg": round(sg, 2)}))

        for stat, proj, dist in specs:
            if proj is None or proj < MIN_PROJ.get(stat, 0):
                continue
            d = dab.get((pid, stat), {})
            dab_line = d.get("line")
            rows.append({
                "player": meta["player"], "team": meta["team"], "matchId": meta.get("matchId"),
                "event": meta["event"], "stat": stat, "stat_label": STAT_LABEL.get(stat, stat),
                "proj": round(float(proj), 1), "dist": dist,
                "dab_line": dab_line, "offered": d.get("offered") or [],
                "line": dab_line if dab_line is not None else _suggest_line(stat, proj)})

    rows.sort(key=lambda r: (str(r["player"]).lower(), r["stat_label"]))
    matches = sorted(set(r["event"] for r in rows if r.get("event")))
    out = {"generated": pd.Timestamp.now("UTC").isoformat(), "multipliers": MULTIPLIERS,
           "rows": rows, "stats": sorted(set(r["stat_label"] for r in rows)),
           "matches": matches, "n_dabble": sum(1 for r in rows if r["dab_line"] is not None)}
    json.dump(out, open("reports/pickem.json", "w"))
    by = {}
    for r in rows:
        by[r["stat_label"]] = by.get(r["stat_label"], 0) + 1
    print(f"Wrote reports/pickem.json: {len(rows)} model rows "
          f"({out['n_dabble']} prefilled from live Dabble lines)")
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
