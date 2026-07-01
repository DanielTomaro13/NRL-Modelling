"""
Train per-target models with strict out-of-time (season-holdout) evaluation,
benchmarked against naive baselines (trailing-5-game avg, career avg).

Count targets use Poisson loss (non-negative); perf_points uses squared error.
Final models are retrained on ALL [min_season .. train_max] data and saved.

Track-aware (TRACK env, default "nrl"):
  - full-history targets train across the track's seasons with season holdouts;
  - provisional targets (e.g. NRLW run metres / PCM, only captured from 2025)
    train on that single season and validate by a within-season round holdout,
    and are flagged provisional in the saved bundle so the site can label them.
"""
import json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tracks as T

TARGETS = ["runsHitup", "runs", "runMetres", "postContactMetres", "tackles", "perf_points"]
POISSON = {"runsHitup", "runs", "runMetres", "postContactMetres", "tackles"}
BASELINE_R5 = {t: f"{t}_r5" for t in TARGETS}
BASELINE_CAREER = {t: f"{t}_career" for t in TARGETS}


def make_model(target):
    loss = "poisson" if target in POISSON else "squared_error"
    return HistGradientBoostingRegressor(
        loss=loss, max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0, max_depth=None,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        categorical_features=["position"], random_state=0)


def prep_X(df, feats):
    X = df[feats + ["position"]].copy()
    X["position"] = X["position"].astype("category")
    return X


def _round_cutoff(df, frac=0.7):
    """Round number splitting a single season ~frac/(1-frac) train/test."""
    rounds = np.sort(df["roundNumber"].dropna().unique())
    if len(rounds) < 4:
        return None
    return float(rounds[int(len(rounds) * frac)])


def evaluate(df, feats, track):
    """Per-target out-of-sample MAE vs baselines. Full targets use season
    holdouts; provisional targets use a within-train_max round holdout."""
    rows = []
    full = [t for t in TARGETS if t not in track.provisional_targets]
    prov = [t for t in TARGETS if t in track.provisional_targets]

    for season in track.holdouts:
        tr = df[(df.season >= track.min_season) & (df.season < season)]
        te = df[df.season == season]
        if len(te) == 0 or len(tr) == 0:
            continue
        Xtr, Xte = prep_X(tr, feats), prep_X(te, feats)
        for t in full:
            m = make_model(t).fit(Xtr, tr[t].values)
            pred = np.clip(m.predict(Xte), 0, None)
            b5 = np.clip(te[BASELINE_R5[t]].fillna(te[t].mean()).values, 0, None)
            bc = np.clip(te[BASELINE_CAREER[t]].fillna(te[t].mean()).values, 0, None)
            rows.append({"holdout": str(season), "kind": "season", "target": t, "n_test": len(te),
                         "MAE_model": mean_absolute_error(te[t].values, pred),
                         "MAE_base_r5": mean_absolute_error(te[t].values, b5),
                         "MAE_base_career": mean_absolute_error(te[t].values, bc),
                         "RMSE_model": mean_squared_error(te[t].values, pred) ** 0.5,
                         "RMSE_base_r5": mean_squared_error(te[t].values, b5) ** 0.5})

    if prov:
        s = df[df.season == track.train_max]
        cut = _round_cutoff(s)
        if cut is not None:
            tr, te = s[s.roundNumber <= cut], s[s.roundNumber > cut]
            if len(tr) and len(te):
                Xtr, Xte = prep_X(tr, feats), prep_X(te, feats)
                for t in prov:
                    m = make_model(t).fit(Xtr, tr[t].values)
                    pred = np.clip(m.predict(Xte), 0, None)
                    b5 = np.clip(te[BASELINE_R5[t]].fillna(te[t].mean()).values, 0, None)
                    bc = np.clip(te[BASELINE_CAREER[t]].fillna(te[t].mean()).values, 0, None)
                    rows.append({"holdout": f"{track.train_max} R>{cut:.0f}", "kind": "round",
                                 "target": t, "n_test": len(te),
                                 "MAE_model": mean_absolute_error(te[t].values, pred),
                                 "MAE_base_r5": mean_absolute_error(te[t].values, b5),
                                 "MAE_base_career": mean_absolute_error(te[t].values, bc),
                                 "RMSE_model": mean_squared_error(te[t].values, pred) ** 0.5,
                                 "RMSE_base_r5": mean_squared_error(te[t].values, b5) ** 0.5})
    return pd.DataFrame(rows)


def main():
    track = T.current()
    df = pd.read_parquet(T.proc("features.parquet", track))
    feats = json.load(open(T.proc("feature_cols.json", track)))["features"]
    T.ensure_dirs(track)

    res = evaluate(df, feats, track)
    if not res.empty:
        res["MAE_gain_vs_r5_%"] = (100 * (res.MAE_base_r5 - res.MAE_model) / res.MAE_base_r5).round(1)
        pd.set_option("display.width", 200, "display.max_columns", 30)
        print(f"=== [{track.name}] walk-forward MAE: model vs naive baselines ===")
        print(res.round(2).to_string(index=False))
        print("\n=== Mean across holdouts ===")
        summ = (res.groupby("target")[["MAE_model", "MAE_base_r5", "MAE_base_career",
                                       "MAE_gain_vs_r5_%"]].mean().round(2))
        print(summ.to_string())
        res.to_csv(T.report("holdout_metrics.csv", track), index=False)
        summ.to_csv(T.report("holdout_summary.csv", track))

    # ---- retrain final models on each target's full window and save ----
    models, targets_meta = {}, {}
    for t in TARGETS:
        if t in track.provisional_targets:
            full = df[df.season == track.train_max]
            window = [track.train_max, track.train_max]
        else:
            full = df[df.season.between(track.min_season, track.train_max)]
            window = [track.min_season, track.train_max]
        full = full[full[t].notna()]
        if len(full) < 50:
            print(f"  skip {t}: only {len(full)} labelled rows")
            continue
        models[t] = make_model(t).fit(prep_X(full, feats), full[t].values)
        mae = res.loc[res.target == t, "MAE_model"].mean() if not res.empty else None
        targets_meta[t] = {
            "provisional": t in track.provisional_targets,
            "train_seasons": window, "n_train": int(len(full)),
            "holdout_mae": None if mae is None or pd.isna(mae) else round(float(mae), 2),
        }

    out = T.model("nrl_models.joblib", track)
    joblib.dump({"models": models, "features": feats, "track": track.name,
                 "targets_meta": targets_meta}, out)
    print(f"\nSaved {len(models)} models -> {out}")
    for t, m in targets_meta.items():
        tag = " [provisional]" if m["provisional"] else ""
        print(f"  {t:18s} seasons {m['train_seasons']}  n={m['n_train']}  "
              f"holdout_MAE={m['holdout_mae']}{tag}")

    # calibrate predictive dispersion (sigma(mu) per target) for distribution pricing
    try:
        import pricing
        pricing.calibrate_dispersion(track=track)
    except Exception as e:
        print("dispersion calibration skipped:", repr(e))


if __name__ == "__main__":
    main()
