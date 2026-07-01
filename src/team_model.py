"""
Match-outcome model — prices the headline team markets (head-to-head, line,
total) that the player-prop engine doesn't cover. Built for NRLW + Origin where
these are the main markets, but track-agnostic (works for men's NRL too).

Method (deliberately simple + robust on short histories):
  1. Margin-informed Elo ratings updated chronologically across the track's
     history, with between-season regression to the mean. Home-field advantage
     is FITTED, not assumed (it is ~0 for NRLW neutral double-headers).
  2. A ridge margin model on [elo_diff(+HFA), recent points-diff form] and a
     ridge total model on both teams' recent scoring/conceding rates.
  3. Residual sigmas (margin, total) calibrated on the holdout seasons.
  4. Pricing: margin ~ Normal(mu_m, sd_m) -> H2H = P(margin>0) and line cover
     = P(margin > -handicap); total ~ Normal(mu_t, sd_t) -> over/under via the
     same Normal-CDF push-band conventions as player props (pricing.py).

CLI:
  TRACK=nrlw python src/team_model.py train      # fit + save models/<track>/team_model.joblib
  TRACK=nrlw python src/team_model.py predict <comp> <round>   # -> reports/<track>/team_predictions.parquet
"""
import sys, json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import tracks as T

FORM_WIN = 5          # rolling window for team form
ELO_K = 20.0
ELO_HOME = 0.0        # additive home rating bump is folded into the fitted model
SEASON_REGRESS = 0.75  # carry 75% of last season's rating deviation into the next


# --------------------------------------------------------------------------- data
def team_matches(track=None):
    """One row per (match, squad): score for/against, opponent, home flag, time."""
    track = track or T.current()
    df = pd.read_parquet(T.proc("player_match.parquet", track))
    df["utcStartTime"] = pd.to_datetime(df["utcStartTime"], utc=True)
    ts = (df.groupby(["matchId", "squadId", "oppSquadId", "season", "roundNumber",
                      "isHome", "utcStartTime"], as_index=False)["points"].sum()
            .rename(columns={"points": "score"}))
    opp = ts[["matchId", "squadId", "score"]].rename(
        columns={"squadId": "oppSquadId", "score": "oppScore"})
    ts = ts.merge(opp, on=["matchId", "oppSquadId"], how="left")
    ts = ts.dropna(subset=["oppScore"])
    return ts.sort_values(["utcStartTime", "matchId"]).reset_index(drop=True)


def add_elo(ts):
    """Pre-match Elo for each squad (margin-of-victory weighted, season-regressed).

    Single chronological pass: record each squad's CURRENT rating as its pre-match
    value, then apply the post-match update once per match (rows come in home/away
    pairs, so gate the update on an unseen matchId to avoid double-counting).
    """
    rating, last_season, seen = {}, {}, set()
    pre_for, pre_against = [], []
    for _, r in ts.iterrows():
        mid, s, h = r["matchId"], r["squadId"], r["oppSquadId"]
        for sq in (s, h):
            rating.setdefault(sq, 1500.0)
            last_season.setdefault(sq, r["season"])
            if last_season[sq] != r["season"]:  # regress to mean between seasons
                rating[sq] = 1500.0 + SEASON_REGRESS * (rating[sq] - 1500.0)
                last_season[sq] = r["season"]
        pre_for.append(rating[s])       # this squad's pre-match rating
        pre_against.append(rating[h])   # opponent's pre-match rating
        if mid in seen:
            continue                    # update the match's ratings only once
        seen.add(mid)
        Rs, Rh = rating[s], rating[h]
        exp_s = 1.0 / (1.0 + 10 ** (-(Rs - Rh) / 400.0))
        res_s = 1.0 if r["score"] > r["oppScore"] else (0.5 if r["score"] == r["oppScore"] else 0.0)
        mov = np.log1p(abs(r["score"] - r["oppScore"]))
        delta = ELO_K * mov * (res_s - exp_s)
        rating[s] = Rs + delta
        rating[h] = Rh - delta
    ts = ts.copy()
    ts["elo_for"], ts["elo_against"] = pre_for, pre_against
    # `rating` now holds each squad's CURRENT (post-all-updates) rating — the one
    # upcoming fixtures should be predicted with. (Taking the last row's elo_for
    # instead would exclude every team's most recent result.)
    ts.attrs["final_ratings"] = {k: float(v) for k, v in rating.items()}
    return ts


def add_form(ts):
    ts = ts.sort_values(["squadId", "utcStartTime"]).reset_index(drop=True)
    g = ts.groupby("squadId", sort=False)
    ts["form_for"] = (g["score"].shift(1).groupby(ts["squadId"], sort=False)
                      .rolling(FORM_WIN, min_periods=1).mean().reset_index(level=0, drop=True))
    ts["form_against"] = (g["oppScore"].shift(1).groupby(ts["squadId"], sort=False)
                          .rolling(FORM_WIN, min_periods=1).mean().reset_index(level=0, drop=True))
    return ts


def build(track=None):
    ts = team_matches(track)
    ts = add_elo(ts)
    ts = add_form(ts)
    # collapse to one row per match from the HOME team's perspective
    home = ts[ts.isHome == 1].copy()
    away = ts[ts.isHome == 0][["matchId", "elo_for", "form_for", "form_against"]].rename(
        columns={"elo_for": "elo_away", "form_for": "form_for_away",
                 "form_against": "form_against_away"})
    m = home.merge(away, on="matchId", how="inner")
    # first-ever appearance has no prior form -> fall back to the league mean score
    la = float(ts["score"].mean())
    for c in ["form_for", "form_against", "form_for_away", "form_against_away"]:
        m[c] = m[c].fillna(la)
    m["margin"] = m["score"] - m["oppScore"]          # home - away
    m["total"] = m["score"] + m["oppScore"]
    m["elo_diff"] = m["elo_for"] - m["elo_away"]
    m["form_diff"] = (m["form_for"] - m["form_against"]) - (m["form_for_away"] - m["form_against_away"])
    m["att_sum"] = m["form_for"] + m["form_for_away"]
    m["def_sum"] = m["form_against"] + m["form_against_away"]
    return m.dropna(subset=["elo_diff"]).reset_index(drop=True)


MARGIN_FEATS = ["elo_diff", "form_diff"]
TOTAL_FEATS = ["att_sum", "def_sum"]


# --------------------------------------------------------------------------- train
def train(track=None):
    track = track or T.current()
    m = build(track)
    T.ensure_dirs(track)

    def fit_eval(feats, target):
        maes, base_maes, accs, base_accs = [], [], [], []
        oos_resid, oos_pred, oos_y = [], [], []
        for season in track.holdouts:
            tr = m[(m.season >= track.min_season) & (m.season < season)]
            te = m[m.season == season]
            if len(tr) < 20 or len(te) == 0:
                continue
            mdl = Ridge(alpha=5.0).fit(tr[feats], tr[target])
            pred = mdl.predict(te[feats])
            oos_resid.extend((te[target].values - pred).tolist())
            oos_pred.extend(pred.tolist()); oos_y.extend(te[target].values.tolist())
            maes.append(mean_absolute_error(te[target], pred))
            base = np.full(len(te), tr[target].mean())
            base_maes.append(mean_absolute_error(te[target], base))
            if target == "margin":  # H2H hit-rate vs "home always wins"
                accs.append(float(((pred > 0) == (te[target] > 0)).mean()))
                base_accs.append(float((te[target] > 0).mean()))
        final = Ridge(alpha=5.0).fit(
            m[m.season.between(track.min_season, track.train_max)][feats],
            m[m.season.between(track.min_season, track.train_max)][target])
        # sigma from WALK-FORWARD holdout residuals (each season predicted by a
        # model trained strictly before it). In-sample residuals of the final
        # model run ~10-20% low, which inflates every H2H/line/total probability.
        if oos_resid:
            sigma = float(np.std(oos_resid))
        else:
            sigma = float((m[target] - final.predict(m[feats])).std())
        metrics = {
            "mae": round(float(np.mean(maes)), 2) if maes else None,
            "base_mae": round(float(np.mean(base_maes)), 2) if base_maes else None,
            "sigma": round(sigma, 2),
            "h2h_acc": round(float(np.mean(accs)), 3) if accs else None,
            "h2h_base_acc": round(float(np.mean(base_accs)), 3) if base_accs else None,
        }
        if target == "margin" and oos_pred:
            # H2H probability calibration on the stacked holdouts, using this sigma
            from scipy.stats import norm as _n
            p = _n.cdf(np.array(oos_pred) / sigma)
            y = (np.array(oos_y) > 0).astype(float)
            metrics["h2h_brier"] = round(float(np.mean((p - y) ** 2)), 4)
            bins = np.clip((p * 5).astype(int), 0, 4)  # 5 reliability bins
            ece = sum(np.mean(bins == b) * abs(p[bins == b].mean() - y[bins == b].mean())
                      for b in range(5) if (bins == b).sum() >= 10)
            metrics["h2h_cal_err"] = round(float(ece), 4)
        return final, metrics

    margin_mdl, margin_m = fit_eval(MARGIN_FEATS, "margin")
    total_mdl, total_m = fit_eval(TOTAL_FEATS, "total")

    # CURRENT Elo per squad for upcoming fixtures: the post-update rating after
    # every played game (the last row's elo_for is the PRE-match rating of the
    # most recent game, i.e. one result stale for every team).
    ts = add_elo(team_matches(track))
    latest_elo = ts.attrs["final_ratings"]
    # current form likewise INCLUDES each team's most recent game: rolling mean
    # over the last FORM_WIN played games, unshifted (the shift(1) in add_form
    # is for leakage-safe training rows, not for the "as of now" snapshot).
    fts = team_matches(track).sort_values(["squadId", "utcStartTime"])
    cur = fts.groupby("squadId").agg(
        form_for=("score", lambda s: s.tail(FORM_WIN).mean()),
        form_against=("oppScore", lambda s: s.tail(FORM_WIN).mean()))
    latest_form = cur

    bundle = {
        "track": track.name,
        "margin_model": margin_mdl, "total_model": total_mdl,
        "margin_feats": MARGIN_FEATS, "total_feats": TOTAL_FEATS,
        "sigma_margin": margin_m["sigma"], "sigma_total": total_m["sigma"],
        "metrics": {"margin": margin_m, "total": total_m},
        "latest_elo": {int(k): float(v) for k, v in latest_elo.items()},
        "latest_form": {int(k): {"for": float(r.form_for) if pd.notna(r.form_for) else None,
                                 "against": float(r.form_against) if pd.notna(r.form_against) else None}
                        for k, r in latest_form.iterrows()},
        "league_avg_score": round(float(team_matches(track)["score"].mean()), 1),
    }
    out = T.model("team_model.joblib", track)
    joblib.dump(bundle, out)
    print(f"[{track.name}] saved {out}")
    print(f"  margin: holdout MAE {margin_m['mae']} (base {margin_m['base_mae']}), "
          f"sigma {margin_m['sigma']}, H2H acc {margin_m['h2h_acc']} (base {margin_m['h2h_base_acc']}), "
          f"brier {margin_m.get('h2h_brier')}, cal_err {margin_m.get('h2h_cal_err')}")
    print(f"  total : holdout MAE {total_m['mae']} (base {total_m['base_mae']}), sigma {total_m['sigma']}")
    return bundle


# --------------------------------------------------------------------------- predict
def _feat_row(bundle, home_sq, away_sq):
    elo = bundle["latest_elo"]; form = bundle["latest_form"]; la = bundle["league_avg_score"]
    ff = lambda sq, k: (form.get(sq, {}) or {}).get(k) or la
    eh, ea = elo.get(home_sq, 1500.0), elo.get(away_sq, 1500.0)
    return {
        "elo_diff": eh - ea,
        "form_diff": (ff(home_sq, "for") - ff(home_sq, "against"))
                     - (ff(away_sq, "for") - ff(away_sq, "against")),
        "att_sum": ff(home_sq, "for") + ff(away_sq, "for"),
        "def_sum": ff(home_sq, "against") + ff(away_sq, "against"),
    }


def predict(comp, rnd, track=None):
    import nrl_meta as M
    track = track or T.current()
    bundle = joblib.load(T.model("team_model.joblib", track))
    fx = M.fixture(comp)
    matches = [x for x in fx if x["roundNumber"] == int(rnd)
               and x.get("matchStatus") != "complete"]
    rows = []
    for x in matches:
        h, a = x["homeSquadId"], x["awaySquadId"]
        fr = _feat_row(bundle, h, a)
        mu_m = float(bundle["margin_model"].predict(pd.DataFrame([fr])[bundle["margin_feats"]])[0])
        mu_t = float(bundle["total_model"].predict(pd.DataFrame([fr])[bundle["total_feats"]])[0])
        rows.append({
            "matchId": x["matchId"], "roundNumber": int(rnd),
            "home": x.get("homeSquadName"), "away": x.get("awaySquadName"),
            "homeSquadId": h, "awaySquadId": a, "start_iso": x.get("utcStartTime"),
            "pred_margin": round(mu_m, 1), "sigma_margin": bundle["sigma_margin"],
            "pred_total": round(mu_t, 1), "sigma_total": bundle["sigma_total"],
            "pred_home_score": round((mu_t + mu_m) / 2, 1),
            "pred_away_score": round((mu_t - mu_m) / 2, 1),
        })
    out = pd.DataFrame(rows)
    dest = T.report("team_predictions.parquet", track)
    T.ensure_dirs(track)
    out.to_parquet(dest, index=False)
    out.to_json(T.report("team_predictions.json", track), orient="records")
    print(f"[{track.name}] wrote {dest}: {len(out)} matches for round {rnd}")
    if len(out):
        print(out[["home", "away", "pred_margin", "pred_total",
                   "pred_home_score", "pred_away_score"]].to_string(index=False))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "predict":
        predict(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit("usage: team_model.py train | predict <comp> <round>")
