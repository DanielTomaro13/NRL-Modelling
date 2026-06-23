"""
Render the static GitHub Pages site into docs/ from the model + odds + edge outputs.

Pages (all server-rendered HTML, no client framework):
  docs/index.html        round predictions per match, with odds + value highlights
  docs/value.html        the value board — every +EV model-vs-market edge
  docs/accuracy.html     holdout accuracy vs naive baselines
  docs/methodology.html  how the model + distribution pricing work
  docs/style.css

Inputs (all optional — the site degrades gracefully if a file is missing):
  reports/round_predictions.parquet, reports/odds_snapshot.json,
  reports/edges.parquet, reports/holdout_summary.csv

Usage: python src/build_site.py [roundNumber]
"""
import sys, os, json, html, datetime as dt
import pandas as pd

import nrl_meta as M
import charts as C

DOCS = "docs"
AEST = dt.timezone(dt.timedelta(hours=10))
STAT_COLS = [("runsHitup", "Hit-ups"), ("runs", "Runs"), ("runMetres", "Run m"),
             ("postContactMetres", "PCM"), ("tackles", "Tackles"), ("perf_points", "Perf pts")]


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def now_aest():
    return dt.datetime.now(AEST)


def fmt_dt(iso_or_epoch):
    try:
        if isinstance(iso_or_epoch, (int, float)):
            d = dt.datetime.fromtimestamp(iso_or_epoch, dt.timezone.utc)
        else:
            d = dt.datetime.fromisoformat(str(iso_or_epoch).replace("Z", "+00:00"))
        return d.astimezone(AEST).strftime("%a %d %b, %I:%M%p AEST")
    except Exception:
        return ""


# --------------------------------------------------------------------------- load
def load_inputs():
    preds = pd.read_parquet("reports/round_predictions.parquet")
    try:
        odds = pd.read_json("reports/odds_snapshot.json")
    except Exception:
        odds = pd.DataFrame()
    try:
        edges = pd.read_parquet("reports/edges.parquet")
    except Exception:
        edges = pd.DataFrame()
    try:
        analysis = json.load(open("reports/analysis.json"))
    except Exception:
        analysis = {}
    try:
        tries = pd.read_parquet("reports/tryscorer_predictions.parquet")
    except Exception:
        tries = pd.DataFrame()
    try:
        try_edges = pd.read_parquet("reports/try_edges.parquet")
    except Exception:
        try_edges = pd.DataFrame()
    try:
        tryinfo = json.load(open("reports/tryscorer.json"))
    except Exception:
        tryinfo = {}
    sc = {}
    for key, path in [("ppoints", "reports/player_points_predictions.parquet"),
                      ("points_edges", "reports/points_edges.parquet")]:
        try:
            sc[key] = pd.read_parquet(path)
        except Exception:
            sc[key] = pd.DataFrame()
    for key, path in [("ppinfo", "reports/player_points.json"),
                      ("kinfo", "reports/kicker.json")]:
        try:
            sc[key] = json.load(open(path))
        except Exception:
            sc[key] = {}
    try:
        sc["comparison"] = json.load(open("reports/comparison.json"))
    except Exception:
        sc["comparison"] = {}
    return preds, odds, edges, analysis, tries, try_edges, tryinfo, sc


BOOKS = [("sportsbet", "SB"), ("ladbrokes", "LAD"), ("tab", "TAB"),
         ("pointsbet", "PB"), ("dabble", "DAB")]
BOOK_ABBR = {"sportsbet": "SB", "ladbrokes": "LAD", "tab": "TAB",
             "pointsbet": "PB", "dabble": "DAB"}


def best_try_price(odds, pid):
    """Best (shortest) anytime try price across books for a player."""
    bb = anytime_by_book(odds, pid)
    if not bb:
        return None
    book, price = min(bb.items(), key=lambda kv: kv[1])
    return {"price": price, "book": book}


def anytime_by_book(odds, pid):
    """{book: price} for a player's anytime try market across books."""
    if odds.empty or "playerId" not in odds or "kind" not in odds:
        return {}
    sub = odds[(odds["playerId"] == pid) & (odds["stat"] == "tries")
               & (odds["kind"] == "anytime") & (odds["single"].notna())]
    out = {}
    for _, r in sub.iterrows():
        b = r["book"]
        p = float(r["single"])
        if b not in out or p > out[b]:   # keep best (longest) price per book
            out[b] = p
    return out


def book_cells(by_book, model_p):
    """Render one <td> per book with its price; best (highest) highlighted; + an EV note."""
    if by_book:
        best_price = max(by_book.values())
    cells = []
    for key, _lbl in BOOKS:
        if key in by_book:
            p = by_book[key]
            cls = "pos" if (by_book and p == best_price) else "mut"
            cells.append(f'<td class="{cls}">{p:.2f}</td>')
        else:
            cells.append('<td class="mut">–</td>')
    ev = ""
    if by_book and model_p:
        e = model_p * max(by_book.values()) - 1
        ev = f'<td class="{"pos" if e>0 else ""}">{e*100:+.0f}%</td>'
    else:
        ev = '<td>–</td>'
    return "".join(cells) + ev


def edges_for_pid(edges, pid):
    if edges.empty or "playerId" not in edges:
        return pd.DataFrame()
    return edges[edges.playerId == pid]


# --------------------------------------------------------------------------- HTML chunks
def page(title, body, active, updated):
    nav = "".join(
        f'<a class="{ "on" if k==active else "" }" href="{href}">{label}</a>'
        for k, href, label in [("index", "index.html", "Predictions"),
                               ("compare", "compare.html", "Compare odds"),
                               ("scoring", "scoring.html", "Scoring"),
                               ("analysis", "analysis.html", "Analysis"),
                               ("backtest", "backtest.html", "Backtest"),
                               ("lab", "lab.html", "Model Lab")])
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><link rel="stylesheet" href="style.css">
<meta name="description" content="Machine-learning predictions for NRL player tackles, run metres, hit-ups, post-contact metres and performance points, with live bookmaker odds and value edges.">
</head><body>
<header><div class="wrap">
<a class="brand" href="index.html">NRL <span>Player Projections</span></a>
<nav>{nav}</nav></div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap">
<p>Updated {esc(updated)}. Predictions are model estimates for informational purposes only — not betting advice.</p>
<p class="rg">Gamble responsibly. 18+. For free &amp; confidential support call 1800 858 858 or visit
<a href="https://www.gamblinghelponline.org.au/">gamblinghelponline.org.au</a>.</p>
<p class="src">Model: per-target HistGradientBoosting on Champion Data match feeds. Odds: Sportsbet &amp; Ladbrokes public APIs. Not affiliated with the NRL or any wpbookmaker.</p>
</div></footer></body></html>"""


def stat_badge(edge_row):
    """Small odds/edge badge for a player's best O/U market."""
    ev = edge_row.get("ev_pct")
    side = edge_row.get("best_side")
    price = edge_row.get("best_price")
    line = edge_row.get("line")
    stat = edge_row.get("stat")
    cls = "edge pos" if (ev is not None and ev > 0) else "edge"
    mm = edge_row.get("model_mean")
    title = f"model {mm} vs line {line}"
    ev_txt = f"{ev:+.0f}% EV" if ev is not None else ""
    return (f'<span class="{cls}" title="{title}">'
            f'{esc(stat)} {esc(side)} {line} @ {price} <b>{ev_txt}</b></span>')


def match_section(preds, odds, edges, match_rows):
    home_pid_rows = match_rows.sort_values("pred_perf_points", ascending=False)
    teams = match_rows[["team", "opp", "isHome"]].drop_duplicates()
    title = match_rows.iloc[0]["team"] + " vs " + match_rows.iloc[0]["opp"]
    kickoff = fmt_dt(match_rows.iloc[0].get("utcStartTime"))

    head = "".join(f"<th>{lbl}</th>" for _, lbl in STAT_COLS)
    body_rows = []
    for _, p in home_pid_rows.iterrows():
        pid = p["playerId"]
        cells = "".join(
            f'<td>{p[f"pred_{c}"]:.0f}</td>' if c != "runMetres" else f'<td>{p[f"pred_{c}"]:.0f}</td>'
            for c, _ in STAT_COLS)
        # odds / edge chips
        chips = []
        pe = edges_for_pid(edges, pid)
        for _, er in pe.iterrows():
            chips.append(stat_badge(er))
        tp = best_try_price(odds, pid)
        if tp:
            chips.append(f'<span class="try">TS ${tp["price"]:.2f} <i>{BOOK_ABBR.get(tp["book"], esc(tp["book"]))}</i></span>')
        chip_html = " ".join(chips)
        body_rows.append(
            f'<tr><td class="pl"><b>{esc(p["name"])}</b><span class="pos">{esc(p["position"])}</span>'
            f'<span class="tm">{esc(p["team"])}</span></td>{cells}'
            f'<td class="ch">{chip_html}</td></tr>')
    return f"""<section class="match" data-match="{esc(title)}">
<h3>{esc(title)} <span class="ko">{esc(kickoff)}</span></h3>
<div class="tablewrap"><table>
<thead><tr><th class="pl">Player</th>{head}<th>Odds / value</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></div></section>"""


def build_index(preds, odds, edges, rnd, updated):
    # value summary banner
    n_odds = int((odds.get("playerId").notna().sum())) if (len(odds) and "playerId" in odds) else 0
    banner = (f'<a class="banner pos" href="compare.html">Compare every market against '
              f'Sportsbet, Ladbrokes &amp; Dabble on the odds dashboard &rarr;</a>' if n_odds else
              '<div class="banner">Player prop odds open ~1–2 days before kickoff; '
              'odds &amp; value populate automatically.</div>')

    secs, match_labels = [], []
    for mid, g in preds.groupby("matchId"):
        secs.append(match_section(preds, odds, edges, g))
        match_labels.append(g.iloc[0]["team"] + " vs " + g.iloc[0]["opp"])
    match_opts = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in match_labels)
    body = f"""<div class="hero"><h1>Round {rnd} player projections</h1>
<p>Six per-player quantities for every named NRL player — hit-ups, runs, run metres,
post-contact metres, tackles and performance points — from a leakage-safe gradient-boosting
model, with live bookmaker odds and model-vs-market value.</p></div>
{banner}
<div class="filters"><label>Jump to match <select onchange="scFilter(this.value)">
<option value="all">All matches</option>{match_opts}</select></label></div>
{''.join(secs)}
<script src="app.js"></script>"""
    return page(f"NRL Round {rnd} player projections", body, "index", updated)


def build_compare(comparison, updated):
    rows = (comparison or {}).get("rows", [])
    if not rows:
        body = """<div class="hero"><h1>Compare odds</h1></div>
<div class="banner">No live markets to compare yet. Try-scorer prices are usually up first;
tackle / metre / points lines open closer to kickoff. This dashboard fills automatically.</div>"""
        return page("Compare odds", body, "compare", updated)
    matches = comparison.get("matches", [])
    markets = comparison.get("markets", [])
    match_opts = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in matches)
    market_opts = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in markets)

    def price_cell(r, book):
        v = (r.get("books") or {}).get(book)
        if v is None:
            return '<td class="mut">–</td>'
        best = (r.get("best_book") == book)
        return f'<td class="{"pos" if best else ""}">{v:.2f}</td>'

    book_head = "".join(f"<th>{lbl}</th>" for _, lbl in BOOKS)
    trs = []
    for r in rows:
        ev = r.get("ev")
        ev_txt = "" if ev is None else f"{ev:+.0f}%"
        ev_cls = "pos" if (ev is not None and 0 < ev <= 40) else ("warn" if (ev or 0) > 40 else "")
        line = "" if r.get("line") is None else f'{r["line"]:g}'
        cells = "".join(price_cell(r, k) for k, _ in BOOKS)
        trs.append(
            f'<tr data-match="{esc(r.get("match",""))}" data-market="{esc(r.get("market",""))}" '
            f'data-ev="{ev if ev is not None else ""}">'
            f'<td class="pl"><b>{esc(r.get("player"))}</b><span class="tm">{esc(r.get("team"))}</span></td>'
            f'<td>{esc(r.get("market"))}</td><td class="mut">{line}</td>'
            f'<td><b>{r.get("my_fair","–")}</b></td>{cells}'
            f'<td class="{ev_cls}"><b>{ev_txt}</b></td></tr>')
    body = f"""<div class="hero"><h1>Compare odds</h1>
<p>Every player market we can price, with <b>my price</b> (the model's fair odds) next to each
book's live price — Sportsbet, Ladbrokes, TAB, PointsBet and Dabble — best highlighted. Positive EV
(green) means the best available price is longer than the model thinks it should be.</p></div>

<div class="filters" id="cmpf">
  <label>Match <select id="f-match" onchange="cmpFilter()"><option value="all">All matches</option>{match_opts}</select></label>
  <label>Market <select id="f-market" onchange="cmpFilter()"><option value="all">All markets</option>{market_opts}</select></label>
  <label class="chk"><input type="checkbox" id="f-ev" onchange="cmpFilter()"> +EV only</label>
  <label class="chk"><input type="checkbox" id="f-cred" checked onchange="cmpFilter()"> hide longshots</label>
  <span class="count" id="f-count"></span>
</div>
<div class="tablewrap"><table id="cmp"><thead><tr>
<th class="pl">Player</th><th>Market</th><th>Line</th><th>My price</th>
{book_head}<th>Best EV</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<p class="disclaim">“My price” is the model's fair odds (1 ÷ model probability), no margin. Very large EV
usually means a team-list or name mismatch — “hide longshots” filters those out by default.</p>
<script src="app.js"></script>"""
    return page("Compare odds", body, "compare", updated)


def build_value(edges, updated):
    if edges.empty:
        body = """<div class="hero"><h1>Value board</h1></div>
<div class="banner">No player over/under markets priced yet. Tackle, run-metre and
fantasy lines open ~1–2 days before kickoff; this board fills automatically (it refreshes
every 6 hours) and ranks every market by model expected value.</div>"""
        return page("Value board", body, "value", updated)
    rows = []
    for _, e in edges.iterrows():
        cls = "pos" if (e.get("ev_pct") or 0) > 0 else ""
        rows.append(
            f'<tr class="{cls}"><td><b>{esc(e["player"])}</b><span class="tm">{esc(e.get("team"))}</span></td>'
            f'<td>{esc(e["stat"])}</td><td>{e["model_mean"]:.1f}</td><td>{e["line"]}</td>'
            f'<td>{esc(e["best_side"])}</td><td>{esc(e["book"])}</td><td>{e["best_price"]}</td>'
            f'<td>{e["fair_over"] if e["best_side"]=="over" else e["fair_under"]}</td>'
            f'<td>{("%+.1f" % e["edge_pct"]) if pd.notna(e.get("edge_pct")) else ""}</td>'
            f'<td><b>{("%+.1f%%" % e["ev_pct"]) if pd.notna(e.get("ev_pct")) else ""}</b></td></tr>')
    body = f"""<div class="hero"><h1>Value board</h1>
<p>Every player over/under market where we have a model projection, ranked by expected value.
Model probability comes from a calibrated Normal(mean, σ) around the prediction; the market
probability is de-vigged from the two-way price. Positive EV = the model thinks the price is too long.</p></div>
<div class="tablewrap"><table class="value">
<thead><tr><th>Player</th><th>Stat</th><th>Model</th><th>Line</th><th>Side</th><th>Book</th>
<th>Price</th><th>Fair</th><th>Edge</th><th>EV</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""
    return page("Value board", body, "value", updated)


def _stat_cards(analysis):
    """Per-target backtest cards: MAE/gain + predicted-vs-actual + residuals."""
    bt = analysis.get("backtest", {})
    labels = analysis.get("target_label", {})
    summ = {r["target"]: r for r in bt.get("summary", [])}
    cards = []
    for t in analysis.get("targets", []):
        s = summ.get(t, {})
        cal = bt.get("calibration", {}).get(t, {})
        res = bt.get("residuals", {}).get(t, {})
        cal_pts = list(zip(cal.get("pred", []), cal.get("actual", [])))
        lo = min([p for p, a in cal_pts] + [a for p, a in cal_pts] + [0]) if cal_pts else 0
        hi = max([p for p, a in cal_pts] + [a for p, a in cal_pts] + [1]) if cal_pts else 1
        cal_svg = C.line_chart([{"name": "model", "color": C.ACC, "points": cal_pts}],
                               (lo, hi), (lo, hi), width=300, height=210, diagonal=True,
                               x_label="predicted", y_label="actual", dots=True) if cal_pts else ""
        hist_svg = C.histogram(res.get("edges", [0, 1]), res.get("counts", [0]),
                               color=C.POS, width=300, height=170,
                               x_label="error (actual − predicted)",
                               mean=res.get("mean")) if res else ""
        cards.append(f"""<div class="bcard">
<h4>{esc(labels.get(t, t))}</h4>
<div class="kpis"><span class="kpi"><b>{s.get('MAE_model','–')}</b><i>model MAE</i></span>
<span class="kpi"><b>{s.get('MAE_base_r5','–')}</b><i>baseline MAE</i></span>
<span class="kpi pos"><b>{('+%.1f%%'%s['gain_pct']) if s.get('gain_pct') is not None else '–'}</b><i>better</i></span></div>
<div class="duo"><figure>{cal_svg}<figcaption>Predicted vs actual — points on the dashed line = unbiased.</figcaption></figure>
<figure>{hist_svg}<figcaption>Error spread (σ = {res.get('sd','–')}), centred near zero.</figcaption></figure></div>
</div>""")
    return "".join(cards)


def build_try_panel(tryinfo):
    bt = tryinfo.get("backtest", {})
    if not bt:
        return ""
    rel = bt.get("reliability", {})
    rel_pts = list(zip(rel.get("pred", []), rel.get("emp", [])))
    rel_svg = C.line_chart([{"name": "model", "color": C.ACC, "points": rel_pts}],
                           (0, 1), (0, 1), width=520, height=340, diagonal=True,
                           x_label="model probability the player scores",
                           y_label="how often they actually scored") if rel_pts else ""
    m = bt.get("model", {}); bpos = bt.get("baseline_position", {}); btr = bt.get("baseline_trailing5", {})
    def cmp_row(label, d):
        return (f'<tr><td class="pl">{label}</td><td>{d.get("brier","–")}</td>'
                f'<td>{d.get("logloss","–")}</td><td>{d.get("auc","–")}</td></tr>')
    return f"""<section class="panel">
<h3>Try-scorer model <span class="tag">classification</span></h3>
<p class="lead">A Poisson tries model, scored on whether a player got over the line (1+).
Predicting tries is genuinely hard — but the probabilities are well calibrated and rank
scorers better than the position base rate or recent form.</p>
<div class="split"><div>{rel_svg}</div>
<div class="note"><p>Reliability of the anytime-try probability across {bt.get('n_test',0):,}
out-of-sample player-matches (base score rate {bt.get('base_rate','–')}).</p>
<p class="big">{bt.get('calibration_error','–')}</p>
<p class="sub">mean gap between predicted and actual scoring rate.</p></div></div>
<div class="tablewrap"><table><thead><tr><th class="pl">Model</th><th>Brier ↓</th>
<th>Log loss ↓</th><th>AUC ↑</th></tr></thead><tbody>
{cmp_row('Try model', m)}{cmp_row('Position base rate', bpos)}{cmp_row('Trailing-5 form', btr)}
</tbody></table></div></section>"""


def build_points_panel(sc):
    ppinfo = (sc or {}).get("ppinfo", {})
    kinfo = (sc or {}).get("kinfo", {})
    out = ""
    pb = ppinfo.get("backtest", {})
    if pb:
        rel = pb.get("reliability", {})
        pts = list(zip(rel.get("pred", []), rel.get("emp", [])))
        svg = C.line_chart([{"name": "model", "color": C.ACC, "points": pts}],
                           (0, 1), (0, 1), width=520, height=340, diagonal=True,
                           x_label="model probability over the points line",
                           y_label="how often it went over") if pts else ""
        out += f"""<section class="panel">
<h3>Player-points model <span class="tag">try + kicker combined</span></h3>
<p class="lead">Points = 4·tries + 2·goals + field goals, built by convolving the try and kicker
Poisson models. Reliability of the over/under probability across {pb.get('n_test',0):,}
out-of-sample player-matches.</p>
<div class="split"><div>{svg}</div>
<div class="note"><p>Predicting exact points is lumpy (each try is four points), but the implied
over/under probabilities are well calibrated through the bulk, with mild overconfidence on big
favourites.</p><p class="big">{pb.get('calibration_error','–')}</p>
<p class="sub">mean gap between predicted and actual over-rate · points MAE {pb.get('mae_points','–')}</p></div></div></section>"""
    kb = kinfo.get("backtest", {})
    if kb:
        out += f"""<section class="panel">
<h3>Goal-kicking model</h3>
<p class="lead">Expected goals (conversions + penalties) as a Poisson rate. Goal-kicking is a
persistent role, so the trailing average is a strong baseline — the model matches it on points
error while staying well calibrated on <em>who</em> kicks (goal-probability error
{kb.get('goal_calibration_error','–')}).</p>
<div class="tablewrap"><table><thead><tr><th class="pl">Metric</th><th>Model</th>
<th>Trailing-5 baseline</th></tr></thead><tbody>
<tr><td class="pl">Kicker-points MAE (all players)</td><td>{kb.get('mae_model','–')}</td><td>{kb.get('mae_baseline','–')}</td></tr>
<tr><td class="pl">MAE on actual kickers</td><td>{kb.get('mae_kickers_only','–')}</td><td>—</td></tr>
</tbody></table></div></section>"""
    return out


def build_backtest(analysis, updated, tryinfo=None, sc=None):
    bt = analysis.get("backtest", {})
    if not bt:
        return page("Backtest", "<div class='hero'><h1>Backtest</h1></div>"
                    "<p>Run <code>python src/analysis.py</code> to populate.</p>", "backtest", updated)
    pooled = bt.get("reliability_pooled", {})
    rel_pts = list(zip(pooled.get("pred", []), pooled.get("emp", [])))
    rel_svg = C.line_chart([{"name": "model", "color": C.ACC, "points": rel_pts}],
                           (0, 1), (0, 1), width=520, height=360, diagonal=True,
                           x_label="model probability the player goes OVER",
                           y_label="how often they actually did") if rel_pts else ""
    cal_err = bt.get("calibration_error")
    # accuracy table
    rows = "".join(
        f'<tr><td class="pl">{esc(r["label"])}</td><td>{r["MAE_model"]}</td>'
        f'<td>{r["MAE_base_r5"]}</td><td class="pos">+{r["gain_pct"]}%</td></tr>'
        for r in bt.get("summary", []))
    holdouts = ", ".join(str(h) for h in bt.get("holdouts", []))
    body = f"""<div class="hero"><h1>Does it actually work?</h1>
<p>Everything here is measured on seasons the model never saw in training ({holdouts}) —
{bt.get('n_test',0):,} player-matches. No cherry-picking: these are out-of-sample results.</p></div>

<section class="panel">
<h3>Probability calibration <span class="tag">the one that matters for betting</span></h3>
<div class="split"><div>{rel_svg}</div>
<div class="note"><p>When the model says a player has a <b>given chance</b> of going over a line,
does it happen that often in reality? The dots track the diagonal almost perfectly — when the
model says 70%, it lands ~70% of the time.</p>
<p class="big">{cal_err:.3f}</p><p class="sub">mean gap between predicted and actual probability
(0 = perfect). This is what makes the value/EV numbers trustworthy rather than wishful.</p></div></div>
</section>

<section class="panel">
<h3>Accuracy vs the naive baseline</h3>
<p class="lead">Mean absolute error per prediction, against the standard punter heuristic
(a player's trailing-5-game average). Lower is better; the model wins on every stat.</p>
<div class="tablewrap"><table><thead><tr><th class="pl">Stat</th><th>Model MAE</th>
<th>Trailing-5 MAE</th><th>Improvement</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>

<section class="panel"><h3>Per-stat detail</h3>
<div class="grid2">{_stat_cards(analysis)}</div></section>

{build_try_panel(tryinfo or {})}

{build_points_panel(sc)}

<p class="disclaim">Honest caveat: we don't have a historical archive of bookmaker prices, so this
is a forecast-accuracy and probability-calibration backtest — not a profit/ROI claim. It shows the
projections and their implied probabilities are sound; whether a given price is value still depends
on the live market.</p>"""
    return page("Backtest & accuracy", body, "backtest", updated)


def build_analysis(analysis, tryinfo, kinfo, updated):
    ci = analysis.get("champion", {})
    if not ci:
        return page("Analysis", "<div class='hero'><h1>Analysis</h1></div>"
                    "<p>Run <code>python src/analysis.py</code> to populate.</p>", "analysis", updated)
    season = ci.get("season")
    # weakest defences (run metres conceded)
    td = ci.get("team_defence", [])[:16]
    def_svg = C.hbars([(d["team"], d["runMetres_conceded"]) for d in td],
                      color=C.WARN, value_fmt="{:.0f}", width=560, label_w=190, unit=" m")
    # leaders
    def leader_block(key, title, unit=""):
        items = ci.get("leaders", {}).get(key, [])[:12]
        svg = C.hbars([(f'{l["name"]}', l["avg"]) for l in items],
                      color=C.ACC, value_fmt="{:.1f}", width=540, label_w=150, unit=unit)
        return f'<div class="lcard"><h4>{title}</h4>{svg}</div>'
    # position profiles table
    prof = ci.get("position_profiles", [])
    phead = "".join(f"<th>{lbl}</th>" for _, lbl in STAT_COLS)
    prows = "".join(
        "<tr><td class='pl'>" + esc(p["position"]) + f"</td>" +
        "".join(f"<td>{p.get(c,0):.0f}</td>" if c != 'runsHitup' else f"<td>{p.get(c,0):.1f}</td>"
                for c, _ in STAT_COLS) + "</tr>" for p in prof)
    body = f"""<div class="hero"><h1>Champion Data, {season}</h1>
<p>What the underlying match data says this season — {ci.get('n_players',0)} players across
{ci.get('n_matches',0)} matches. Useful context before you back a line: who's racking up the
volume, and which defences are leaking it.</p></div>

<section class="panel"><h3>Which defences leak metres? <span class="tag">attack-side value</span></h3>
<p class="lead">Run metres conceded per game (all opponents). Players facing the teams at the top
have the friendliest match-ups for run-metre and post-contact overs.</p>{def_svg}</section>

<section class="panel"><h3>Season leaders (per-game average, min 4 games)</h3>
<div class="grid2">
{leader_block('tackles','Most tackles')}
{leader_block('runMetres','Most run metres',' m')}
{leader_block('perf_points','Most performance points')}
{leader_block('postContactMetres','Most post-contact metres',' m')}
</div></section>

<section class="panel"><h3>Scoring leaders <span class="tag">try + kicker models</span></h3>
<p class="lead">Who finds the line and who slots the goals — the inputs to the player-points model.</p>
<div class="grid2">
<div class="lcard"><h4>Most tries / game</h4>{C.hbars(
    [(l["name"], l["per_game"]) for l in (tryinfo or {}).get("leaders", [])[:12]],
    color=C.POS, value_fmt="{:.2f}", width=540, label_w=150)}</div>
<div class="lcard"><h4>Most kicker points / game</h4>{C.hbars(
    [(l["name"], l["per_game"]) for l in (kinfo or {}).get("leaders", [])[:12]],
    color=C.ACC, value_fmt="{:.1f}", width=540, label_w=150)}</div>
</div></section>

<section class="panel"><h3>Average output by position</h3>
<p class="lead">Roles shape the stat line — props pile up run metres and post-contact, locks and
hookers tackle, halves drive performance points. The model conditions on position throughout.</p>
<div class="tablewrap"><table><thead><tr><th class="pl">Position</th>{phead}</tr></thead>
<tbody>{prows}</tbody></table></div></section>"""
    return page(f"Analysis — {season}", body, "analysis", updated)


def fmt_odds_cell(p):
    return f"${1/p:.2f}" if p and p > 1e-9 else "–"


def _try_section(tries, odds):
    book_head = "".join(f"<th>{lbl}</th>" for _, lbl in BOOKS)
    secs = []
    for mid, g in tries.groupby("matchId"):
        g = g.sort_values("p_anytime", ascending=False)
        teams = f'{g.iloc[0]["team"]} vs {g.iloc[0]["opp"]}'
        rows = []
        for _, p in g.head(7).iterrows():
            by_book = anytime_by_book(odds, p["playerId"])
            rows.append(
                f'<tr><td class="pl"><b>{esc(p["name"])}</b><span class="pos">{esc(p["position"])}</span>'
                f'<span class="tm">{esc(p["team"])}</span></td>'
                f'<td><b>{p["p_anytime"]*100:.0f}%</b></td>'
                f'<td class="mut">{fmt_odds_cell(p["p_anytime"])}</td>'
                f'{book_cells(by_book, float(p["p_anytime"]))}</tr>')
        secs.append(f"""<section class="match" data-match="{esc(teams)}"><h3>{esc(teams)}</h3>
<div class="tablewrap"><table><thead><tr><th class="pl">Player</th><th>Anytime</th>
<th>Fair</th>{book_head}<th>Best EV</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>""")
    return "".join(secs)


def _kicker_section(ppoints, odds):
    """Top expected goal-kickers per match, with expected goals + any live kicker-points odds."""
    if ppoints.empty or "exp_kicker_points" not in ppoints:
        return "<p class='mut'>Run the kicker model to populate.</p>"
    book_head = "".join(f"<th>{lbl}</th>" for _, lbl in BOOKS)
    secs = []
    grp = ppoints.groupby("matchId") if "matchId" in ppoints else [(0, ppoints)]
    for mid, g in grp:
        kk = g[g["exp_kicker_points"] > 0.5].sort_values("exp_kicker_points", ascending=False)
        if kk.empty:
            continue
        teams = f'{g.iloc[0]["team"]} vs {g.iloc[0]["opp"]}'
        rows = []
        for _, p in kk.head(4).iterrows():
            by_book = ou_by_book(odds, p["playerId"], "kicker_points")
            rows.append(
                f'<tr><td class="pl"><b>{esc(p["name"])}</b><span class="tm">{esc(p["team"])}</span></td>'
                f'<td><b>{p["exp_kicker_points"]:.1f}</b></td><td class="mut">{p["lg"]:.1f}</td>'
                f'{book_cells(by_book, None)}</tr>')
        secs.append(f"""<section class="match" data-match="{esc(teams)}"><h3>{esc(teams)}</h3>
<div class="tablewrap"><table><thead><tr><th class="pl">Goal kicker</th><th>Exp pts</th>
<th>Exp goals</th>{book_head}<th>Best EV</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>""")
    return "".join(secs) or "<p class='mut'>No goal-kickers identified for this round yet.</p>"


def ou_by_book(odds, pid, stat):
    """{book: over-price} for a player's over/under market of a given stat."""
    if odds.empty or "playerId" not in odds:
        return {}
    sub = odds[(odds["playerId"] == pid) & (odds["stat"] == stat) & odds["over"].notna()]
    out = {}
    for _, r in sub.iterrows():
        b = r["book"]
        p = float(r["over"])
        if b not in out or p > out[b]:
            out[b] = p
    return out


def _points_section(ppoints, points_edges):
    if ppoints.empty:
        return "<p class='mut'>Run the points model to populate.</p>"
    ev_by = {}
    if not points_edges.empty:
        for _, e in points_edges[points_edges.stat == "points"].iterrows():
            ev_by[e["playerId"]] = e
    secs = []
    grp = ppoints.groupby("matchId") if "matchId" in ppoints else [(0, ppoints)]
    for mid, g in grp:
        g = g.sort_values("exp_points", ascending=False)
        teams = f'{g.iloc[0]["team"]} vs {g.iloc[0]["opp"]}'
        rows = []
        for _, p in g.head(6).iterrows():
            e = ev_by.get(p["playerId"])
            if e is not None and e.get("ev_pct") is not None:
                ev = e["ev_pct"]; credible = 0 < ev <= 40
                cls = "pos" if credible else ""
                odds_cell = (f'<td>{e["line"]}</td><td class="{cls}">${e["best_price"]:.2f} '
                             f'<i>{BOOK_ABBR.get(e["book"], esc(e["book"]))}</i></td>'
                             f'<td class="{cls}"><b>{ev:+.0f}%</b></td>')
            else:
                odds_cell = '<td>–</td><td>–</td><td>–</td>'
            rows.append(
                f'<tr><td class="pl"><b>{esc(p["name"])}</b><span class="pos">{esc(p["position"])}</span>'
                f'<span class="tm">{esc(p["team"])}</span></td>'
                f'<td><b>{p["exp_points"]:.1f}</b></td><td class="mut">{p["exp_tries"]*4:.1f}</td>'
                f'<td class="mut">{p["exp_kicker_points"]:.1f}</td>{odds_cell}</tr>')
        secs.append(f"""<section class="match" data-match="{esc(teams)}"><h3>{esc(teams)}</h3>
<div class="tablewrap"><table><thead><tr><th class="pl">Player</th><th>Exp pts</th>
<th>from tries</th><th>from kicking</th><th>Line</th><th>Best price</th><th>EV</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>""")
    return "".join(secs)


def build_scoring(tries, try_edges, tryinfo, sc, odds, updated):
    ppoints = sc.get("ppoints", pd.DataFrame())
    points_edges = sc.get("points_edges", pd.DataFrame())
    if tries.empty and ppoints.empty:
        body = """<div class="hero"><h1>Scoring</h1></div>
<div class="banner">Run the try, kicker and points models to populate this page.</div>"""
        return page("Scoring", body, "scoring", updated)
    auc = tryinfo.get("backtest", {}).get("model", {}).get("auc")

    # match filter options (from whichever predictions we have)
    match_set = []
    src = ppoints if not ppoints.empty else tries
    if "matchId" in src:
        for _, g in src.groupby("matchId"):
            match_set.append(f'{g.iloc[0]["team"]} vs {g.iloc[0]["opp"]}')
    match_opts = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in sorted(set(match_set)))

    tabs = [("points", "Player points"), ("kicker", "Kicker points"), ("tries", "Try scorers")]
    tab_btns = "".join(
        f'<button data-tabgroup="sc" data-tab="{k}" class="{"on" if i==0 else ""}" '
        f'onclick="showTab(\'sc\',\'{k}\')">{lbl}</button>' for i, (k, lbl) in enumerate(tabs))
    panes = {
        "points": f'<p class="lead">Expected points per player, split into try points and '
                  f'goal-kicking points; the model edge shows where a book has posted a line.</p>'
                  + _points_section(ppoints, points_edges),
        "kicker": f'<p class="lead">The designated goal-kickers and their expected kicking points '
                  f'(2 per goal + field goals), with any live kicker-points lines.</p>'
                  + _kicker_section(ppoints, odds),
        "tries": f'<p class="lead">Model anytime-try probability and fair price next to every '
                 f'book\'s live price (best highlighted). Best EV = edge at the best price.</p>'
                 + _try_section(tries, odds),
    }
    pane_html = "".join(
        f'<div class="tabpane {"on" if i==0 else ""}" data-pane="sc" data-pane-name="{k}">{panes[k]}</div>'
        for i, (k, _l) in enumerate(tabs))

    body = f"""<div class="hero"><h1>Scoring &amp; points</h1>
<p>Three linked models — a try-scorer model, a goal-kicking model, and player points built by
combining them (points = 4·tries + 2·goals + field goals), priced against the live books. See the
<a href="backtest.html">backtest</a> for calibration{f' (try model AUC {auc})' if auc else ''}.</p></div>
<div class="filters">
  <div class="tabs">{tab_btns}</div>
  <label style="margin-left:auto">Match <select onchange="scFilter(this.value)">
    <option value="all">All matches</option>{match_opts}</select></label>
</div>
{pane_html}
<p class="disclaim">Very large EV usually means a team-list or name mismatch, not real value —
check the lineup. <a href="compare.html">Compare odds →</a></p>
<script src="app.js"></script>"""
    return page("Scoring & points", body, "scoring", updated)


def build_lab(analysis, updated):
    imp = analysis.get("importance", {})
    labels = analysis.get("target_label", {})
    imp_cards = []
    for t in analysis.get("targets", []):
        items = imp.get(t, [])[:8]
        if not items:
            continue
        svg = C.hbars([(i["feature"], i["importance"]) for i in items],
                      color=C.ACC, value_fmt="{:.2f}", width=520, label_w=210)
        imp_cards.append(f'<div class="lcard"><h4>{esc(labels.get(t,t))}</h4>{svg}</div>')
    imp_html = "".join(imp_cards)
    body = f"""<div class="hero"><h1>Model Lab</h1>
<p>Play with the exact maths the site uses to turn a projection into a price and an edge.
Change the numbers and watch the probability, fair odds and expected value update live.</p></div>

<section class="panel"><h3>Pricing explorer <span class="tag">interactive</span></h3>
<p class="lead">Pick a stat, set the model's projected mean and a bookmaker line, then enter the
over/under price a book is offering. The tool builds the calibrated <code>Normal(mean, σ)</code>,
reads the probability off the curve (with the integer-line push band), de-vigs the market and
computes your edge — the same pipeline that fills the value board.</p>
<div id="lab" class="lab">
  <div class="controls">
    <label>Stat
      <select id="lab-stat"></select></label>
    <label>Model projected mean <output id="lab-mean-v"></output>
      <input id="lab-mean" type="range" min="0" max="60" step="0.5"></label>
    <label>Bookmaker line <output id="lab-line-v"></output>
      <input id="lab-line" type="range" min="0" max="60" step="0.5"></label>
    <div class="prices"><label>Over price <input id="lab-over" type="number" step="0.01" value="1.90"></label>
    <label>Under price <input id="lab-under" type="number" step="0.01" value="1.90"></label></div>
  </div>
  <div class="readout"><div id="lab-curve"></div><div id="lab-out" class="out"></div></div>
</div></section>

<section class="panel"><h3>What drives each prediction?</h3>
<p class="lead">Permutation importance on held-out data — how much each input matters to each stat.
Recent form and role dominate; opponent-defence features add the match-up signal.</p>
<div class="grid2">{imp_html}</div></section>

<section class="panel"><h3>Six stat models + three scoring models</h3>
<p class="lead">The explorer above covers the six continuous stat markets (tackles, run metres,
hit-ups, post-contact metres, performance points, runs). Scoring is handled by three more models —
a <b>try-scorer</b> (Poisson on tries), a <b>goal-kicking</b> model (Poisson on goals), and
<b>player points</b> built by convolving them (points = 4·tries + 2·goals + field goals). Their
out-of-sample calibration is on the <a href="backtest.html">backtest</a>, their projections on
<a href="scoring.html">Scoring</a>, and every market lines up against the books on
<a href="compare.html">Compare odds</a>.</p></section>

<section class="prose panel">
<h3>The method, in four steps</h3>
<p><b>1. Predict the mean.</b> A separate gradient-boosting model per stat, trained on Champion Data
feeds with strictly pre-match features (rolling form shifted one game, opponent defence conceded,
team context). Counts use Poisson loss.</p>
<p><b>2. Build a distribution.</b> The prediction is the mean; σ is calibrated from out-of-sample
residuals as <code>σ = α + β·mean</code> (spread grows with volume). That gives a full
<code>Normal(mean, σ)</code>, not just a point.</p>
<p><b>3. Price the line.</b> Probability of going over = the normal tail above the line, with a
±0.5 push band on whole-number lines and a 50/50 split on quarter lines — standard book convention.</p>
<p><b>4. Find the edge.</b> De-vig the book's two prices to a market probability, compare to the
model, and compute EV per dollar (a push returns the stake). Positive EV = the price looks too long.</p>
<p class="disclaim">Informational only — not betting advice. See the
<a href="backtest.html">backtest</a> for how calibrated these probabilities actually are.</p>
</section>
<script src="app.js"></script>"""
    return page("Model Lab", body, "lab", updated)


CSS = """:root{--bg:#0b0e14;--card:#141a24;--line:#222c3a;--ink:#e6edf3;--mut:#8aa0b2;
--acc:#4cc2ff;--pos:#39d98a;--posbg:#0f2c20;--chip:#1b2430}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 18px}
header{border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(11,14,20,.92);
backdrop-filter:blur(8px);z-index:5}header .wrap{display:flex;align-items:center;gap:20px;height:58px}
.brand{font-weight:700;color:var(--ink);text-decoration:none;font-size:18px}
.brand span{color:var(--acc);font-weight:600}
nav{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
nav a{color:var(--mut);text-decoration:none;padding:6px 11px;border-radius:8px;font-size:14px}
nav a:hover{color:var(--ink);background:var(--chip)}nav a.on{color:var(--ink);background:var(--chip)}
.hero{padding:26px 0 10px}.hero h1{margin:0 0 8px;font-size:26px;letter-spacing:-.02em}
.hero p{color:var(--mut);max-width:760px}
.banner{display:block;margin:14px 0;padding:12px 16px;border:1px solid var(--line);
border-radius:12px;background:var(--card);color:var(--mut);text-decoration:none}
.banner.pos{border-color:#1c6b4a;background:var(--posbg);color:var(--pos);font-weight:600}
.match{margin:22px 0;border:1px solid var(--line);border-radius:14px;background:var(--card);overflow:hidden}
.match h3{margin:0;padding:13px 16px;border-bottom:1px solid var(--line);font-size:16px;
display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.match h3 .ko{color:var(--mut);font-weight:500;font-size:13px}
.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 10px;text-align:right;white-space:nowrap}
th{color:var(--mut);font-weight:600;border-bottom:1px solid var(--line);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td{border-bottom:1px solid #1a2230}tr:last-child td{border-bottom:0}
th.pl,td.pl{text-align:left}td.pl b{font-weight:600}
td.pl .pos{color:var(--mut);margin-left:8px;font-size:12px}
td.pl .tm{display:block;color:var(--mut);font-size:11px}
td.ch{text-align:left;white-space:normal}
.edge{display:inline-block;margin:2px 4px 2px 0;padding:2px 7px;border-radius:7px;background:var(--chip);
font-size:12px;color:var(--mut)}.edge.pos{background:var(--posbg);color:var(--pos)}
.edge b{color:inherit}.try{display:inline-block;padding:2px 7px;border-radius:7px;background:var(--chip);
font-size:12px;color:var(--mut)}.try i{color:var(--acc);font-style:normal}.try.pos{background:var(--posbg);color:var(--pos)}
td.pos{color:var(--pos)}td.mut{color:var(--mut)}td i{color:var(--acc);font-style:normal;font-size:11px}
table.value td:first-child{text-align:left}table.value tr.pos td b{color:var(--pos)}
.prose{max-width:780px}.prose h3{margin:22px 0 6px;font-size:17px}.prose p{color:#c4d2de}
.prose code{background:var(--chip);padding:1px 5px;border-radius:5px;font-size:13px}
footer{border-top:1px solid var(--line);margin-top:40px;padding:22px 0;color:var(--mut);font-size:12.5px}
footer p{margin:5px 0}footer .rg{color:#b58}footer a{color:var(--acc)}
/* analysis / backtest / lab */
.panel{margin:20px 0;padding:18px 18px 22px;border:1px solid var(--line);border-radius:14px;background:var(--card)}
.panel h3{margin:0 0 4px;font-size:17px;display:flex;align-items:center;gap:10px}
.panel .lead{color:var(--mut);margin:.3em 0 14px;max-width:760px}
.tag{font-size:11px;font-weight:600;color:var(--acc);background:#10243150;border:1px solid #1d3a4a;
padding:2px 8px;border-radius:20px;text-transform:uppercase;letter-spacing:.04em}
.chart{width:100%;height:auto;display:block}
.split{display:grid;grid-template-columns:1.1fr .9fr;gap:20px;align-items:center}
.split .note p{color:#c4d2de}.note .big{font-size:40px;font-weight:700;color:var(--pos);margin:6px 0 0}
.note .sub{color:var(--mut);font-size:13px;margin:.2em 0 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.bcard,.lcard{border:1px solid var(--line);border-radius:12px;padding:13px 14px;background:#0f141c}
.bcard h4,.lcard h4{margin:0 0 8px;font-size:14.5px}
.kpis{display:flex;gap:16px;margin-bottom:8px}.kpi{display:flex;flex-direction:column}
.kpi b{font-size:18px}.kpi i{color:var(--mut);font-size:11px;font-style:normal}.kpi.pos b{color:var(--pos)}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:10px}
figure{margin:0}figcaption{color:var(--mut);font-size:11px;margin-top:2px}
.disclaim{color:var(--mut);font-size:12.5px;border-left:3px solid var(--line);padding:2px 0 2px 12px;margin:18px 0}
.lab{display:grid;grid-template-columns:300px 1fr;gap:20px;align-items:start}
.controls label{display:block;color:var(--mut);font-size:13px;margin:0 0 14px}
.controls input[type=range]{width:100%;margin-top:6px;accent-color:var(--acc)}
.controls output{color:var(--ink);font-weight:600;float:right}
.controls select,.controls input[type=number]{width:100%;margin-top:5px;background:#0f141c;color:var(--ink);
border:1px solid var(--line);border-radius:8px;padding:7px 9px;font-size:14px}
.prices{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.readout .out{margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ostat{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#0f141c}
.ostat b{display:block;font-size:21px}.ostat span{color:var(--mut);font-size:11.5px}
.ostat.win b{color:var(--pos)}.ostat.lose b{color:#e0605f}
.verdict{grid-column:1/-1;border-radius:10px;padding:11px 14px;font-weight:600}
.verdict.win{background:var(--posbg);color:var(--pos);border:1px solid #1c6b4a}
.verdict.lose{background:#2a1414;color:#e0958f;border:1px solid #5a2b2b}
.prose.panel{max-width:none}.prose.panel p{color:#c4d2de;margin:.5em 0}
/* filters + tabs */
.filters{position:sticky;top:58px;z-index:4;display:flex;flex-wrap:wrap;gap:12px;align-items:center;
margin:6px 0 14px;padding:11px 14px;border:1px solid var(--line);border-radius:12px;background:rgba(20,26,36,.96);backdrop-filter:blur(8px)}
.filters label{color:var(--mut);font-size:13px;display:flex;align-items:center;gap:7px}
.filters select{background:#0f141c;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:13.5px}
.filters .chk{cursor:pointer}.filters .count{margin-left:auto;color:var(--mut);font-size:12.5px}
td.warn{color:var(--warn)}.warn{color:var(--warn)}
.tabs{display:flex;gap:6px;margin:4px 0 0;flex-wrap:wrap}
.tabs button{background:var(--chip);color:var(--mut);border:1px solid var(--line);border-radius:9px;
padding:7px 14px;font-size:13.5px;cursor:pointer;font-weight:600}
.tabs button.on{background:var(--posbg);color:var(--ink);border-color:#1c6b4a}
.tabpane{display:none}.tabpane.on{display:block}
@media(max-width:760px){.split,.grid2,.duo,.lab{grid-template-columns:1fr}}
@media(max-width:640px){.hero h1{font-size:22px}nav a{padding:6px 8px}.filters{top:54px}}
"""

# ---------------------------------------------------------------- interactive Model Lab JS
APP_JS = r"""// Pricing explorer — mirrors src/pricing.py (Normal CDF, push band, quarter-line, de-vig, EV).
function erf(x){var s=x<0?-1:1;x=Math.abs(x);var t=1/(1+0.3275911*x);
var y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
return s*y;}
function cdf(x,m,sd){return 0.5*(1+erf((x-m)/(sd*Math.SQRT2)));}
function frac(x){return x-Math.floor(x);}
function overUnder(mu,sd,line){sd=Math.max(sd,1e-6);var f=Math.round(frac(line)*10000)/10000;
 if(f===0.25||f===0.75){var a=overUnder(mu,sd,line-0.25),b=overUnder(mu,sd,line+0.25);
   return[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2];}
 if(Math.abs(f-0.5)<1e-6){var pu=cdf(line,mu,sd);return[1-pu,pu,0];}
 var lo=cdf(line-0.5,mu,sd),hi=cdf(line+0.5,mu,sd);return[1-hi,lo,hi-lo];}
function devig(o,u){if(!o||!u)return[null,null];var io=1/o,iu=1/u,s=io+iu;return[io/s,iu/s];}
function fmtOdds(p){return p>1e-9?(1/p).toFixed(2):'–';}

var MODEL=null;
fetch('data/model.json').then(r=>r.json()).then(d=>{MODEL=d;initLab();}).catch(()=>{});

function sigmaFor(t,mu){var d=MODEL.dispersion[t];return Math.max(d.alpha+d.beta*mu,d.sigma_floor);}

function initLab(){
 var sel=document.getElementById('lab-stat');if(!sel)return;
 var ts=Object.keys(MODEL.dispersion);
 ts.forEach(function(t){var o=document.createElement('option');o.value=t;
   o.textContent=MODEL.target_label[t]||t;sel.appendChild(o);});
 sel.value=ts.indexOf('tackles')>=0?'tackles':ts[0];
 ['lab-stat','lab-mean','lab-line','lab-over','lab-under'].forEach(function(id){
   document.getElementById(id).addEventListener('input',function(e){if(id==='lab-stat')presetFor(sel.value);render();});});
 presetFor(sel.value);render();
}
function presetFor(t){
 var mean=MODEL.typical_mean[t]||10, max=Math.max(8,Math.ceil(mean*2.2));
 var mu=document.getElementById('lab-mean'),li=document.getElementById('lab-line');
 mu.max=max;li.max=max;mu.value=mean;li.value=Math.max(0,Math.round((mean-1.5)*2)/2);
}
function render(){
 var t=document.getElementById('lab-stat').value;
 var mu=parseFloat(document.getElementById('lab-mean').value);
 var line=parseFloat(document.getElementById('lab-line').value);
 var over=parseFloat(document.getElementById('lab-over').value)||null;
 var under=parseFloat(document.getElementById('lab-under').value)||null;
 var sd=sigmaFor(t,mu);
 document.getElementById('lab-mean-v').textContent=mu.toFixed(1);
 document.getElementById('lab-line-v').textContent=line.toFixed(1);
 var pr=overUnder(mu,sd,line),pOver=pr[0],pUnder=pr[1],push=pr[2];
 var mk=devig(over,under),mOver=mk[0];
 var evOver=over?(pOver*over+push-1):null, evUnder=under?(pUnder*under+push-1):null;
 var best=(evOver||-9)>=(evUnder||-9)?['OVER',evOver,pOver,over]:['UNDER',evUnder,pUnder,under];
 drawCurve(mu,sd,line);
 var out=document.getElementById('lab-out');
 function stat(cls,v,l){return '<div class="ostat '+cls+'"><b>'+v+'</b><span>'+l+'</span></div>';}
 var edge=(mOver!=null)?((pOver-mOver)*100):null;
 var verdict;
 if(best[1]==null){verdict='<div class="verdict lose">Enter a book price to see edge & EV.</div>';}
 else if(best[1]>0.001){verdict='<div class="verdict win">Model sees value on the '+best[0]+
   ' @ '+best[3].toFixed(2)+' — '+(best[1]*100).toFixed(1)+'% EV.</div>';}
 else{verdict='<div class="verdict lose">No edge at these prices ('+best[0]+' EV '+
   (best[1]*100).toFixed(1)+'%).</div>';}
 out.innerHTML=
   stat('', (pOver*100).toFixed(1)+'%','model P(over '+line+')')+
   stat('', fmtOdds(pOver),'fair over odds')+
   stat('', (mOver!=null?(mOver*100).toFixed(1)+'%':'–'),'market P(over), de-vigged')+
   stat('', (push>0.001?(push*100).toFixed(1)+'%':'0%'),'push chance')+
   stat(best[0]==='OVER'?'win':'', (evOver!=null?(evOver*100).toFixed(1)+'%':'–'),'EV backing over')+
   stat(best[0]==='UNDER'?'win':'', (evUnder!=null?(evUnder*100).toFixed(1)+'%':'–'),'EV backing under')+
   verdict;
}
function drawCurve(mu,sd,line){
 var W=520,H=240,ml=8,mr=8,mt=10,mb=22,pw=W-ml-mr,ph=H-mt-mb;
 var x0=mu-3.5*sd,x1=mu+3.5*sd;
 var sx=function(x){return ml+(x-x0)/(x1-x0)*pw;};
 var pdf=function(x){return Math.exp(-0.5*Math.pow((x-mu)/sd,2));};
 var top=pdf(mu),sy=function(v){return mt+(1-v/top)*ph;};
 var N=120,pts=[],fill=[];
 for(var i=0;i<=N;i++){var x=x0+(x1-x0)*i/N;pts.push([sx(x),sy(pdf(x))]);}
 for(var j=0;j<=N;j++){var x=Math.max(line,x0)+(x1-Math.max(line,x0))*j/N;fill.push([sx(x),sy(pdf(x))]);}
 var path='M'+pts.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L');
 var area='M'+sx(Math.max(line,x0)).toFixed(1)+' '+(mt+ph)+' L'+
   fill.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L')+' L'+sx(x1).toFixed(1)+' '+(mt+ph)+' Z';
 var lx=sx(line);
 var svg='<svg viewBox="0 0 '+W+' '+H+'" class="chart">'+
  '<path d="'+area+'" fill="#39d98a" opacity="0.18"/>'+
  '<path d="'+path+'" fill="none" stroke="#4cc2ff" stroke-width="2"/>'+
  '<line x1="'+lx.toFixed(1)+'" y1="'+mt+'" x2="'+lx.toFixed(1)+'" y2="'+(mt+ph)+'" stroke="#f0a35e" stroke-width="1.5" stroke-dasharray="4 3"/>'+
  '<text x="'+lx.toFixed(1)+'" y="'+(mt+ph+15)+'" text-anchor="middle" fill="#f0a35e" font-size="11">line '+line.toFixed(1)+'</text>'+
  '<text x="'+sx(mu).toFixed(1)+'" y="'+(mt+12)+'" text-anchor="middle" fill="#8aa0b2" font-size="11">model '+mu.toFixed(1)+'</text>'+
  '<text x="'+(ml+pw-4)+'" y="'+(mt+ph-6)+'" text-anchor="end" fill="#39d98a" font-size="11">over →</text></svg>';
 document.getElementById('lab-curve').innerHTML=svg;
}

// ---- Compare dashboard filters ----
function cmpFilter(){
 var tbl=document.getElementById('cmp'); if(!tbl)return;
 var match=(document.getElementById('f-match')||{}).value||'all';
 var market=(document.getElementById('f-market')||{}).value||'all';
 var evonly=(document.getElementById('f-ev')||{}).checked;
 var cred=(document.getElementById('f-cred')||{}).checked;
 var shown=0;
 tbl.querySelectorAll('tbody tr').forEach(function(tr){
   var ok=true, ev=parseFloat(tr.dataset.ev);
   if(match!=='all' && tr.dataset.match!==match) ok=false;
   if(market!=='all' && tr.dataset.market!==market) ok=false;
   if(evonly && !(ev>0)) ok=false;
   if(cred && (!isNaN(ev) && (ev>40||ev<-95))) ok=false;  // hide implausible longshots
   tr.style.display=ok?'':'none'; if(ok)shown++;
 });
 var c=document.getElementById('f-count'); if(c)c.textContent=shown+' markets';
}
// ---- Scoring match filter ----
function scFilter(match){
 document.querySelectorAll('section.match').forEach(function(s){
   s.style.display=(match==='all'||s.dataset.match===match)?'':'none';});
}
// ---- Tabs (Scoring) ----
function showTab(group,name){
 document.querySelectorAll('[data-tabgroup="'+group+'"]').forEach(function(b){
   b.classList.toggle('on', b.dataset.tab===name);});
 document.querySelectorAll('[data-pane="'+group+'"]').forEach(function(p){
   p.classList.toggle('on', p.dataset.paneName===name);});
}
document.addEventListener('DOMContentLoaded', function(){ if(document.getElementById('cmp')) cmpFilter(); });
"""


def main():
    rnd = int(sys.argv[1]) if len(sys.argv) > 1 else None
    preds, odds, edges, analysis, tries, try_edges, tryinfo, sc = load_inputs()
    if rnd is None:
        rnd = int(preds["roundNumber"].iloc[0]) if "roundNumber" in preds else 0
    updated = now_aest().strftime("%a %d %b %Y, %I:%M%p AEST")

    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(f"{DOCS}/data", exist_ok=True)
    open(f"{DOCS}/style.css", "w").write(CSS)
    open(f"{DOCS}/app.js", "w").write(APP_JS)
    open(f"{DOCS}/index.html", "w").write(build_index(preds, odds, edges, rnd, updated))
    open(f"{DOCS}/compare.html", "w").write(build_compare(sc.get("comparison", {}), updated))
    open(f"{DOCS}/scoring.html", "w").write(build_scoring(tries, try_edges, tryinfo, sc, odds, updated))
    open(f"{DOCS}/analysis.html", "w").write(
        build_analysis(analysis, tryinfo, sc.get("kinfo", {}), updated))
    open(f"{DOCS}/backtest.html", "w").write(build_backtest(analysis, updated, tryinfo, sc))
    open(f"{DOCS}/lab.html", "w").write(build_lab(analysis, updated))
    open(f"{DOCS}/.nojekyll", "w").write("")  # serve files verbatim
    n_matches = preds["matchId"].nunique()
    print(f"Built {DOCS}/ — round {rnd}, {n_matches} matches, "
          f"{len(edges)} priced markets, updated {updated}")


if __name__ == "__main__":
    main()
