# NRL Player-Stat Prediction

Predicts six per-player, per-match quantities for NRL men's matches from the
Champion Data match-centre feeds:

**Hit-ups · Runs · Run Metres · Post-Contact Metres · Tackles · Performance Points**

Performance points:
```
perf_points = 4*points + 10*tryAssists + 5*lineBreaks + 1*tackles + floor(runMetres/10)
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Ingest | `src/ingest.py` | `data/processed/player_match.parquet` (85,795 player-match rows, 2014–2026) |
| 2. Features | `src/features.py` | `data/processed/features.parquet` (109 leakage-safe features) |
| 3. Train + evaluate | `src/train.py` | `models/nrl_models.joblib`, `reports/holdout_metrics.csv` |
| 4. Scrape team lists | `src/scrape_teamlists.py <url> <compId> <round>` | `data/processed/lineups_r{round}.parquet` |
| 5. Predict a round | `src/predict.py <compId> <round> [lineups.parquet]` | `reports/round_predictions.parquet` |

Run all:
```bash
.venv/bin/python src/ingest.py
.venv/bin/python src/features.py
.venv/bin/python src/train.py
# confirmed lineups (recommended) -> scrape then predict
.venv/bin/python src/scrape_teamlists.py "https://www.nrl.com/news/2026/06/16/nrl-team-lists-round-16/" 12999 16
.venv/bin/python src/predict.py 12999 16 data/processed/lineups_r16.parquet
# or proxy lineups (most-recent XVII), omit the lineups arg:
.venv/bin/python src/predict.py 12999 16
```

`scrape_teamlists.py` parses each player's confirmed position/jersey/side from nrl.com's
team-list page, maps team nickname -> `squadId` and name -> `playerId` (HTML-entity and
accent normalised; 292/293 mapped for R16, the miss being a jersey-18 reserve), and
restricts the predicted set to the named 1-17. Using confirmed lineups improved the
round-16 line MAE from 5.89 (proxy) to 5.38 and caught that a player on the industry
sheet (Terrell May, prop) was not actually named — the Tigers' named "May" is the
centre, Taylan.

## Published site & automation

The model is published as a static **GitHub Pages** site, rebuilt automatically.

| Step | Script | Output |
|---|---|---|
| 6. Detect comp/round + scrape lineups + predict | `src/run_round.py` | `reports/round_predictions.parquet` |
| 7. Analysis + backtest + feature importance | `src/analysis.py` | `reports/analysis.json`, `docs/data/model.json` |
| 8. Try-scorer / goal-kicking / player-points models | `src/tryscorer.py`, `src/kicker.py`, `src/player_points.py` | `reports/{tryscorer,kicker,player_points}*` |
| 9. Fetch odds (Sportsbet + Ladbrokes) | `src/odds.py` | `reports/odds_snapshot.{parquet,json}` |
| 10. Pricing + value edges (stats, tries, points) | `src/pricing.py price`/`tries`, `src/player_points.py price` | `reports/{edges,try_edges,points_edges}.*` |
| 11. Render site | `src/build_site.py` | `docs/` (Pages root) |

The site has six pages: **Predictions** (per-match player projections + odds), **Value**
(model-vs-market edges), **Scoring** (try-scorer, goal-kicking and player-points models vs live
prices), **Analysis** (Champion Data — defences leaking metres, season leaders, position profiles),
**Backtest** (out-of-sample accuracy + probability-calibration reliability for every model — the
trust page), and an interactive **Model Lab** (a live pricing explorer + permutation feature
importance). Charts are dependency-free SVG (`src/charts.py`); the explorer is vanilla JS
(`docs/app.js`) mirroring `src/pricing.py`.

### Try-scorer model (`src/tryscorer.py`)
A Poisson model for a player's expected tries, reusing the leakage-safe feature machinery plus
tries-specific rollups and opponent tries-conceded. From the rate λ it derives the prices books
offer — P(anytime)=1−e^−λ, P(2+)=1−e^−λ(1+λ), P(3+), E[tries] — then `pricing.py tries` values
them against the **live** Sportsbet/Ladbrokes try markets (these open early, unlike stat props).
Out of sample it ranks scorers at AUC ≈ 0.73 with calibration error ≈ 0.008, beating the position
base-rate and trailing-5 baselines on Brier/log-loss. Implausibly large edges are flagged on the
Scoring page as likely lineup/name mismatches rather than real value.

### Goal-kicking + player-points models (`src/kicker.py`, `src/player_points.py`)
The feed carries `conversions`, `penaltyGoals`, `fieldGoals`, so a player's points decompose
exactly: `points = 4·tries + 2·goals + fieldGoals`. `kicker.py` is a Poisson model for expected
goals (kicking is a persistent role, so it matches the trailing-average baseline on points error
while staying well calibrated on *who* kicks — goal-probability error ≈ 0.014). `player_points.py`
convolves the try and kicker Poissons into the full points distribution → expected points and
P(points ≥ line) for any posted player-points line (out-of-sample over/under calibration ≈ 0.06,
with mild overconfidence on big favourites). `player_points.py price` values live player-points
and goals over/unders; both run daily (models) and 6-hourly (pricing).

Two GitHub Actions workflows keep it live:
- **`.github/workflows/model.yml`** — daily: ingest → features → train (+ calibrate
  dispersion) → `run_round.py` → odds → pricing → site.
- **`.github/workflows/odds.yml`** — every 6 hours: odds → pricing → site only
  (odds move fast; the model doesn't). They share a concurrency group so commits never race.

Enable Pages once in repo **Settings → Pages → Source: deploy from branch `main`, folder `/docs`**.

### Bookmakers & markets
Odds are pulled from **Sportsbet**, **Ladbrokes/Neds**, **TAB**, **PointsBet** and **Dabble**, and
the **Compare odds** dashboard shows every book's price side by side (best highlighted) next to the
model's fair "my price", with match/market/+EV/longshot filters. The fetcher recognises the player
markets the industry posts — **Try Scorer, Performance Points, Kicker Points, Player Points,
(Most) Run Metres, (Most) Tackles** — and routes each to the right model. Team names are
canonicalised to NRL nicknames so a player lines up across books (e.g. "Souths" == "South Sydney
Rabbitohs"). All books use `curl_cffi` (browser-TLS) and run from `src/odds.py`.

> **TAB** uses OAuth2. **Recommended:** set `TAB_CLIENT_ID` + `TAB_CLIENT_SECRET` — `src/odds.py`
> mints a **fresh access token every run** via the `client_credentials` grant
> (`https://api.beta.tab.com.au/oauth/token`), so it never expires in CI. A static
> `TAB_ACCESS_TOKEN` also works but lapses in ~3h. **PointsBet** needs no auth.
> **Dabble** is the iOS app's backend — set `DABBLE_AUTH="Bearer <token>"` captured from the app
> with Charles (see below; ~1h expiry). AU bookmaker APIs may geo-block some CI regions; running
> `python src/odds.py` from an AU connection always works.

#### Capturing the Dabble token (Charles, iOS)
Dabble has no web API; the token comes from the app. With Charles proxying the iPhone (SSL proxy
`*.dabble.com.au`), open the app → an NRL game → player markets, find a request to
`api.dabble.com.au`, and copy the **`authorization: Bearer …`** header. Then:
```bash
export DABBLE_AUTH='Bearer eyJ...'
python src/odds.py     # pulls Dabble tries + player-points markets
```
`src/odds.py` discovers the NRL competition, lists fixtures via
`/frontend-api/competitions/{id}/sport-fixtures`, and reads full markets from
`/frontend-api/sport-fixtures/details/{fixtureId}` (`sportFixtureDetail.{markets,selections,prices}`).
The token is short-lived — re-capture when it lapses, or add it as the `DABBLE_AUTH` GitHub secret.

> **Pick'em is excluded.** Dabble runs a Pick'em (multiplier/parlay) product alongside its
> sportsbook; those markets (`resultingType` starting `pickem_*`) are flat even-money picks, not
> traditional fixed odds, so the fetcher drops them. Only genuine fixed-odds markets (try-scorer
> etc.) are used for the EV comparison.

> **Dabble** is behind Cloudflare bot-protection that blocks datacenter IPs / plain TLS clients,
> so it 403s from GitHub Actions. `src/odds.py` uses `curl_cffi` (browser-TLS impersonation) and an
> optional `DABBLE_COOKIE` env var — paste a `cf_clearance` cookie from a logged-in browser (or add
> it as a GitHub Actions secret) to enable Dabble. Without it, Dabble is skipped gracefully and the
> other two books still work. To refresh odds locally: `python src/odds.py && python src/pricing.py
> price && python src/pricing.py tries && python src/player_points.py price && python src/build_site.py`.

### Odds → value (distribution pricing)
`src/pricing.py` turns each point prediction into a calibrated `Normal(mean, σ)`
(σ fitted as `α + β·mean` from out-of-time residuals), prices the posted line off the
normal CDF (integer-line push band ±0.5, quarter-line 50/50 split), **de-vigs** the
book's two-way price, and reports edge / EV per side. Player tackle / run-metre / fantasy
lines open ~1–2 days before kickoff; until then the site shows predictions + try-scorer odds
and the value board fills in automatically. `python src/pricing.py selftest` checks the maths.

> Auto-detection: `src/nrl_meta.py` resolves the latest men's NRL competition id, the next
> round to predict, and the nrl.com team-list URL — no hard-coded round each week.

## Method notes
- **Data**: men's Premiership + Finals only. `postContactMetres` only exists from
  2021, so labelled training/eval is restricted to **2021–2026** (~39k rows); earlier
  seasons still seed each player's rolling history.
- **No minutes field** exists (and `playerSubs` is empty league-wide), so time-on-field
  is proxied by rolling **half-presence** (`halves`) and **activity**
  (`possessions+tackles+tackleds`). Current-match presence is never a feature (leakage).
- **Leakage-safe features**: every rolling/career stat is shifted one game; opponent
  *defence* (metres/missed-tackles/etc. conceded) and own-team form are 5-game rollups
  joined as-of the match.
- **Models**: per-target `HistGradientBoostingRegressor`; Poisson loss for the count
  targets, squared-error for perf_points. Regularisation tuned on out-of-time holdouts
  (lighter settings tested and rejected — they worsen generalisation).
- **Validation**: season holdouts (2023/24/25). The model beats both naive baselines
  (trailing-5 avg, career avg) on every target, every season.

## Headline accuracy (mean over 2023/24/25 holdouts)

| Target | Model MAE | Trailing-5 MAE | Gain |
|---|---|---|---|
| Tackles | 5.23 | 5.89 | +11.2% |
| Performance Points | 11.68 | 12.76 | +8.5% |
| Run Metres | 27.66 | 29.93 | +7.6% |
| Runs | 2.74 | 2.95 | +7.2% |
| Post-Contact Metres | 11.41 | 12.12 | +5.8% |
| Hit-ups | 2.04 | 2.11 | +3.6% |

Against bookmaker performance-point lines for Round 16 2026 (11 players): **MAE 5.89**.
The model is more conservative than the market on in-form stars (it regresses recent
form toward career; the market does not) — this is correct for predicting *actual*
outcomes but explains the gap to the lines.

## Limitations / next levers
- **Lineups for forward prediction** use each squad's most-recent completed XVII as a
  proxy. Production should scrape the official team lists
  (e.g. `https://www.nrl.com/news/.../nrl-team-lists-round-16/`) for confirmed
  starters/bench and late changes, mapping names → `playerId`.
- No injury / rest / weather signal, and no betting-market feature.
- A "derived" perf-points route (predict components, apply the formula) is a possible
  enhancement; the direct model already matches it within noise.
# NRL-Modelling
