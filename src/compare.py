"""
Build the odds-comparison dataset: for every player market we can price, emit a row
with the model's fair price ("my price") alongside each book's live price (Sportsbet,
Ladbrokes, Dabble), the best available, and the model's edge.

Output: reports/comparison.json  (consumed by the Compare dashboard)

CLI: python src/compare.py
"""
import json, math
import numpy as np
import pandas as pd
from scipy.stats import poisson

import pricing as PRC
import player_points as PP

# market key -> (display label, group) ; group drives the dashboard's market filter
MARKET_LABEL = {
    "try_anytime": "Anytime Try", "try_2+": "2+ Tries", "try_3+": "3+ Tries",
    "points": "Player Points", "kicker_points": "Kicker Points",
    "performance_points": "Performance Points", "tackles": "Tackles",
    "run_metres": "Run Metres", "post_contact_metres": "Post-Contact M",
    "runs": "Runs", "fantasy": "Fantasy", "goals": "Goals",
}
OU_TO_TARGET = PRC.STAT_TO_TARGET  # stat O/U markets priced off the Normal stat models


def _best(books):
    if not books:
        return None, None
    book, price = max(books.items(), key=lambda kv: kv[1])
    return book, price


def main():
    od = pd.read_parquet("reports/odds_snapshot.parquet")
    try:
        tdf = pd.read_parquet("reports/tryscorer_predictions.parquet").set_index("playerId")
    except Exception:
        tdf = pd.DataFrame()
    try:
        ppdf = pd.read_parquet("reports/player_points_predictions.parquet").set_index("playerId")
    except Exception:
        ppdf = pd.DataFrame()
    try:
        rp = pd.read_parquet("reports/round_predictions.parquet").set_index("playerId")
    except Exception:
        rp = pd.DataFrame()
    disp = PRC.load_dispersion() or {}

    def match_of(pid):
        for src in (tdf, ppdf, rp):
            if len(src) and pid in src.index:
                r = src.loc[pid]
                if getattr(r, "ndim", 1) > 1:
                    r = r.iloc[0]
                t, o = r.get("team"), r.get("opp")
                if t and o:
                    return " vs ".join(sorted([str(t), str(o)])), t  # canonical (one per game)
        return "", ""

    rows = []

    # ---- try markets (1-way: back the scorer) ----
    if len(tdf):
        tri = od[(od.stat == "tries") & od.single.notna() & od.playerId.notna()]
        for pid, g in tri.groupby("playerId"):
            if pid not in tdf.index:
                continue
            pr = tdf.loc[pid]
            if getattr(pr, "ndim", 1) > 1:
                pr = pr.iloc[0]
            lam = float(pr["lambda"])
            pmap = {"anytime": float(pr["p_anytime"]), "2+": float(pr["p_2plus"]),
                    "3+": 1 - math.exp(-lam) * (1 + lam + lam ** 2 / 2)}
            for kind, sub in g.groupby("kind"):
                if kind not in pmap:
                    continue
                books = {}
                for _, r in sub.iterrows():
                    b, p = r["book"], float(r["single"])
                    if b not in books or p > books[b]:
                        books[b] = p
                mp = pmap[kind]
                bk, bp = _best(books)
                match, team = match_of(pid)
                rows.append({"match": match, "player": pr.get("name"), "team": team,
                             "mkey": f"try_{kind}", "market": MARKET_LABEL[f"try_{kind}"],
                             "line": None, "my_p": round(mp, 3),
                             "my_fair": round(1 / mp, 2) if mp > 1e-9 else None,
                             "sportsbet": books.get("sportsbet"), "ladbrokes": books.get("ladbrokes"),
                             "dabble": books.get("dabble"), "best_book": bk, "best": bp,
                             "ev": round((mp * bp - 1) * 100, 1) if bp else None})

    # ---- over/under markets (show the OVER selection) ----
    ou = od[(od.category == "player") & od.over.notna() & od.playerId.notna() & od.line.notna()]
    for (pid, stat, line), g in ou.groupby(["playerId", "stat", "line"]):
        books = {r["book"]: float(r["over"]) for _, r in g.iterrows() if pd.notna(r["over"])}
        if not books:
            continue
        mp = None
        target = OU_TO_TARGET.get(stat)
        if target and len(rp) and pid in rp.index and target in [c.replace("pred_", "") for c in rp.columns if c.startswith("pred_")]:
            mu = float(rp.loc[pid, f"pred_{target}"])
            sigma = PRC.sigma_for(target, mu, disp) if disp else None
            if sigma:
                mp = PRC.over_under_probs(mu, sigma, float(line))[0]
        elif stat in ("points", "kicker_points", "goals") and len(ppdf) and pid in ppdf.index:
            pr = ppdf.loc[pid]
            lt, lg, lfg = float(pr["lt"]), float(pr["lg"]), float(pr["lfg"])
            if stat == "points":
                mp = PP.p_over_under(lt, lg, lfg, float(line))[0]
            elif stat == "kicker_points":
                mp = PP.p_over_under(0.0, lg, lfg, float(line))[0]
            else:
                mp = 1 - float(poisson.cdf(math.floor(line), lg))
        if mp is None:
            continue
        bk, bp = _best(books)
        match, team = match_of(pid)
        pname = (tdf.loc[pid]["name"] if len(tdf) and pid in tdf.index else
                 (ppdf.loc[pid]["name"] if len(ppdf) and pid in ppdf.index else None))
        rows.append({"match": match, "player": pname, "team": team,
                     "mkey": stat, "market": MARKET_LABEL.get(stat, stat),
                     "line": float(line), "my_p": round(mp, 3),
                     "my_fair": round(1 / mp, 2) if mp > 1e-9 else None,
                     "sportsbet": books.get("sportsbet"), "ladbrokes": books.get("ladbrokes"),
                     "dabble": books.get("dabble"), "best_book": bk, "best": bp,
                     "ev": round((mp * bp - 1) * 100, 1) if bp else None})

    rows = [r for r in rows if r["player"]]
    rows.sort(key=lambda r: (r["ev"] if r["ev"] is not None else -999), reverse=True)
    out = {"generated": pd.Timestamp.now("UTC").isoformat(),
           "markets": sorted(set(r["market"] for r in rows)),
           "matches": sorted(set(r["match"] for r in rows if r["match"])),
           "rows": rows}
    json.dump(out, open("reports/comparison.json", "w"))
    n_ev = sum(1 for r in rows if (r["ev"] or 0) > 0)
    print(f"Wrote reports/comparison.json: {len(rows)} market rows across "
          f"{len(out['matches'])} matches, {n_ev} +EV")
    by = {}
    for r in rows:
        by[r["market"]] = by.get(r["market"], 0) + 1
    print("  by market:", by)


if __name__ == "__main__":
    main()
