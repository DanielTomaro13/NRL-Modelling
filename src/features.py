"""
Leakage-safe feature engineering.

ONE ROW = one player in one match. Every feature uses ONLY information available
BEFORE kickoff: all rolling/expanding stats are shifted by one game so the current
match is excluded. Current-match presence (p1/p2/activity) is NEVER a feature
(it describes the match we're predicting) -- instead we use its rolling average as
an "expected workload / minutes proxy".

Outputs: data/processed/features.parquet
"""
import numpy as np
import pandas as pd
import tracks as T

TARGETS = ["runsHitup", "runs", "runMetres", "postContactMetres", "tackles", "perf_points"]

# stats we build player rolling history for (drivers of the targets)
ROLL_STATS = [
    "runsHitup", "runs", "runMetres", "postContactMetres", "tackles", "perf_points",
    "points", "tryAssists", "lineBreaks", "possessions", "metresGained", "tackleBreaks",
    "missedTackles", "offloads", "passes", "tackleds", "runsKickReturn",
    "kicksGeneralPlay", "runsHitupMetres", "activity", "halves",
]
WINDOWS = [3, 5, 10]

# team-level stats for own-attack and opponent-defence context
TEAM_STATS = ["runMetres", "postContactMetres", "tackles", "metresGained",
              "tackleBreaks", "missedTackles", "lineBreaks", "runs", "possessions", "perf_points"]
TEAM_WIN = 5


def add_perf_points(df):
    df["runsHitup"] = df["runsHitup"].clip(lower=0)
    df["perf_points"] = (4 * df["points"] + 10 * df["tryAssists"] + 5 * df["lineBreaks"]
                         + df["tackles"] + (df["runMetres"] // 10)).astype(float)
    df["halves"] = df["p1"] + df["p2"]
    return df


def player_rolling(df):
    df = df.sort_values(["playerId", "utcStartTime", "matchId"]).reset_index(drop=True)
    g = df.groupby("playerId", sort=False)
    # games played so far, days rest
    df["games_prior"] = g.cumcount()
    df["days_rest"] = g["utcStartTime"].diff().dt.total_seconds() / 86400.0
    for stat in ROLL_STATS:
        prev = g[stat].shift(1)  # exclude current match
        pg = prev.groupby(df["playerId"], sort=False)
        df[f"{stat}_career"] = pg.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        for w in WINDOWS:
            df[f"{stat}_r{w}"] = pg.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def team_context(df):
    # team "for": sum of a squad's player stats in a match
    tf = (df.groupby(["matchId", "squadId", "utcStartTime"], as_index=False)[TEAM_STATS]
            .sum())
    # team "allowed" = opponent's "for" in the same match
    opp = tf.rename(columns={"squadId": "oppSquadId",
                             **{s: f"{s}_allowed" for s in TEAM_STATS}})
    tm = tf.merge(df[["matchId", "squadId", "oppSquadId"]].drop_duplicates(),
                  on=["matchId", "squadId"])
    tm = tm.merge(opp[["matchId", "oppSquadId"] + [f"{s}_allowed" for s in TEAM_STATS]],
                  on=["matchId", "oppSquadId"])
    tm = tm.sort_values(["squadId", "utcStartTime", "matchId"]).reset_index(drop=True)
    gs = tm.groupby("squadId", sort=False)
    roll_cols = {}
    for s in TEAM_STATS:
        for col in [s, f"{s}_allowed"]:
            prev = gs[col].shift(1)
            roll_cols[f"tm_{col}_r{TEAM_WIN}"] = (
                prev.groupby(tm["squadId"], sort=False)
                    .rolling(TEAM_WIN, min_periods=1).mean().reset_index(level=0, drop=True))
    tm = pd.concat([tm, pd.DataFrame(roll_cols)], axis=1)

    own_cols = [c for c in tm.columns if c.startswith("tm_") and "allowed" not in c]
    allow_cols = [c for c in tm.columns if c.startswith("tm_") and "allowed" in c]

    # own team form: join on (matchId, squadId)
    own = tm[["matchId", "squadId"] + own_cols].rename(
        columns={c: c.replace("tm_", "own_") for c in own_cols})
    # opponent defence: take opponent's ALLOWED rolling -> join player's oppSquadId to that squad's row
    oppd = tm[["matchId", "squadId"] + allow_cols].rename(
        columns={"squadId": "oppSquadId",
                 **{c: c.replace("tm_", "opp_").replace("_allowed", "Allowed") for c in allow_cols}})
    df = df.merge(own, on=["matchId", "squadId"], how="left")
    df = df.merge(oppd, on=["matchId", "oppSquadId"], how="left")
    return df


def main():
    track = T.current()
    IN = T.proc("player_match.parquet", track)
    OUT = T.proc("features.parquet", track)
    df = pd.read_parquet(IN)
    df["utcStartTime"] = pd.to_datetime(df["utcStartTime"], utc=True)
    df = add_perf_points(df)
    df = player_rolling(df)
    df = team_context(df)

    feat_cols = (["isHome", "roundNumber", "games_prior", "days_rest", "jumperNumber"]
                 + [c for c in df.columns if c.endswith(tuple(f"_r{w}" for w in WINDOWS))
                    or c.endswith("_career")]
                 + [c for c in df.columns if c.startswith(("own_", "opp_"))])
    feat_cols = sorted(set(feat_cols))
    df["position"] = df["position"].replace("-", "Unknown").fillna("Unknown")

    keep = (["season", "matchId", "utcStartTime", "playerId", "squadId", "oppSquadId",
             "position"] + feat_cols + TARGETS)
    out = df[keep].copy()
    out.attrs = {}
    # persist the feature list
    import json
    T.ensure_dirs(track)
    with open(T.proc("feature_cols.json", track), "w") as f:
        json.dump({"features": feat_cols, "categorical": ["position"], "targets": TARGETS}, f, indent=2)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} rows, {len(feat_cols)} numeric features + position")
    print("seasons:", sorted(out['season'].unique()))
    print("labelled rows season>=2021:", (out['season'] >= 2021).sum())


if __name__ == "__main__":
    main()
