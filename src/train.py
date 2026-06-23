"""
Train per-target models with strict out-of-time (season-holdout) evaluation,
benchmarked against naive baselines (trailing-5-game avg, career avg).

Count targets use Poisson loss (non-negative); perf_points uses squared error.
Final models are retrained on ALL 2021-2025 data and saved for prediction.
"""
import json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEAT = "data/processed/features.parquet"
META = "data/processed/feature_cols.json"

TARGETS = ["runsHitup", "runs", "runMetres", "postContactMetres", "tackles", "perf_points"]
POISSON = {"runsHitup", "runs", "runMetres", "postContactMetres", "tackles"}
# naive baseline = this target's trailing-5 rolling mean (already a feature)
BASELINE_R5 = {t: f"{t}_r5" for t in TARGETS}
BASELINE_CAREER = {t: f"{t}_career" for t in TARGETS}

MIN_SEASON = 2021          # PCM + modern stats consistent from here
HOLDOUTS = [2023, 2024, 2025]


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


def evaluate(df, feats):
    rows = []
    for season in HOLDOUTS:
        tr = df[(df.season >= MIN_SEASON) & (df.season < season)]
        te = df[df.season == season]
        if len(te) == 0:
            continue
        Xtr = prep_X(tr, feats)
        Xte = prep_X(te, feats)
        for t in TARGETS:
            ytr, yte = tr[t].values, te[t].values
            m = make_model(t).fit(Xtr, ytr)
            pred = np.clip(m.predict(Xte), 0, None)
            b5 = np.clip(te[BASELINE_R5[t]].fillna(te[t].mean()).values, 0, None)
            bc = np.clip(te[BASELINE_CAREER[t]].fillna(te[t].mean()).values, 0, None)
            rows.append({
                "holdout": season, "target": t, "n_test": len(te),
                "MAE_model": mean_absolute_error(yte, pred),
                "MAE_base_r5": mean_absolute_error(yte, b5),
                "MAE_base_career": mean_absolute_error(yte, bc),
                "RMSE_model": mean_squared_error(yte, pred) ** 0.5,
                "RMSE_base_r5": mean_squared_error(yte, b5) ** 0.5,
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(FEAT)
    feats = json.load(open(META))["features"]

    res = evaluate(df, feats)
    res["MAE_gain_vs_r5_%"] = (100 * (res.MAE_base_r5 - res.MAE_model) / res.MAE_base_r5).round(1)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== Walk-forward (season holdout) MAE: model vs naive baselines ===")
    print(res.round(2).to_string(index=False))

    print("\n=== Mean across holdouts ===")
    summ = (res.groupby("target")[["MAE_model", "MAE_base_r5", "MAE_base_career",
                                    "MAE_gain_vs_r5_%"]].mean().round(2))
    print(summ.to_string())
    res.to_csv("reports/holdout_metrics.csv", index=False)
    summ.to_csv("reports/holdout_summary.csv")

    # ---- retrain final models on ALL 2021-2025 and save ----
    full = df[df.season.between(MIN_SEASON, 2025)]
    Xf = prep_X(full, feats)
    models = {}
    for t in TARGETS:
        models[t] = make_model(t).fit(Xf, full[t].values)
    joblib.dump({"models": models, "features": feats}, "models/nrl_models.joblib")
    print("\nSaved final models -> models/nrl_models.joblib (trained on 2021-2025, "
          f"{len(full)} rows)")

    # calibrate predictive dispersion (sigma(mu) per target) for distribution pricing
    try:
        import pricing
        pricing.calibrate_dispersion()
    except Exception as e:
        print("dispersion calibration skipped:", repr(e))


if __name__ == "__main__":
    main()
