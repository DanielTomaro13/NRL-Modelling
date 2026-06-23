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
        acc = pd.read_csv("reports/holdout_summary.csv")
    except Exception:
        acc = pd.DataFrame()
    return preds, odds, edges, acc


def best_try_price(odds, pid):
    """Best (shortest) anytime/1+ try price across books for a player."""
    if odds.empty or "playerId" not in odds:
        return None
    sub = odds[(odds.get("playerId") == pid) & (odds.get("stat") == "tries")
               & (odds.get("single").notna())]
    # 1+ / anytime only (line ~0.5)
    sub = sub[sub["line"].fillna(0.5) <= 0.5]
    if sub.empty:
        return None
    row = sub.loc[sub["single"].idxmin()]
    return {"price": float(row["single"]), "book": row["book"]}


def edges_for_pid(edges, pid):
    if edges.empty or "playerId" not in edges:
        return pd.DataFrame()
    return edges[edges.playerId == pid]


# --------------------------------------------------------------------------- HTML chunks
def page(title, body, active, updated):
    nav = "".join(
        f'<a class="{ "on" if k==active else "" }" href="{href}">{label}</a>'
        for k, href, label in [("index", "index.html", "Predictions"),
                               ("value", "value.html", "Value board"),
                               ("accuracy", "accuracy.html", "Accuracy"),
                               ("method", "methodology.html", "Method")])
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
            chips.append(f'<span class="try">TS ${tp["price"]:.2f} <i>{esc(tp["book"])[:3]}</i></span>')
        chip_html = " ".join(chips)
        body_rows.append(
            f'<tr><td class="pl"><b>{esc(p["name"])}</b><span class="pos">{esc(p["position"])}</span>'
            f'<span class="tm">{esc(p["team"])}</span></td>{cells}'
            f'<td class="ch">{chip_html}</td></tr>')
    return f"""<section class="match">
<h3>{esc(title)} <span class="ko">{esc(kickoff)}</span></h3>
<div class="tablewrap"><table>
<thead><tr><th class="pl">Player</th>{head}<th>Odds / value</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></div></section>"""


def build_index(preds, odds, edges, rnd, updated):
    # value summary banner
    n_edges = int((edges["ev_pct"] > 0).sum()) if len(edges) else 0
    n_odds = int((odds.get("playerId").notna().sum())) if (len(odds) and "playerId" in odds) else 0
    if n_edges:
        banner = (f'<a class="banner pos" href="value.html">{n_edges} positive-EV edges '
                  f'found vs the market &rarr;</a>')
    elif n_odds:
        banner = (f'<div class="banner">{n_odds} player markets live (try-scorer). '
                  f'Tackle / run-metre lines open closer to kickoff — value edges will appear here.</div>')
    else:
        banner = ('<div class="banner">Player prop odds open ~1–2 days before kickoff. '
                  'Predictions below; odds &amp; value populate automatically.</div>')

    secs = []
    for mid, g in preds.groupby("matchId"):
        secs.append(match_section(preds, odds, edges, g))
    body = f"""<div class="hero"><h1>Round {rnd} player projections</h1>
<p>Six per-player quantities for every named NRL player — hit-ups, runs, run metres,
post-contact metres, tackles and performance points — from a leakage-safe gradient-boosting
model, with live bookmaker odds and model-vs-market value.</p></div>
{banner}
{''.join(secs)}"""
    return page(f"NRL Round {rnd} player projections", body, "index", updated)


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


def build_accuracy(acc, updated):
    if acc.empty:
        body = "<div class='hero'><h1>Accuracy</h1></div><p>Run training to populate.</p>"
        return page("Accuracy", body, "accuracy", updated)
    head = "".join(f"<th>{esc(c)}</th>" for c in acc.columns)
    rows = "".join("<tr>" + "".join(
        f"<td>{v:.2f}</td>" if isinstance(v, float) else f"<td>{esc(v)}</td>"
        for v in r) + "</tr>" for r in acc.itertuples(index=False))
    body = f"""<div class="hero"><h1>Holdout accuracy</h1>
<p>Mean absolute error on out-of-time season holdouts (2023–25), versus naive baselines
(trailing-5-game average). The model beats both baselines on every target.</p></div>
<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"""
    return page("Accuracy", body, "accuracy", updated)


def build_method(updated):
    body = """<div class="hero"><h1>How it works</h1></div>
<section class="prose">
<h3>1. The model</h3>
<p>For each of six targets — hit-ups, runs, run metres, post-contact metres, tackles and
performance points — a separate <code>HistGradientBoostingRegressor</code> is trained on
Champion Data men's NRL match feeds (2021–present; earlier seasons seed each player's rolling
history). Count targets use Poisson loss. Every feature is shifted one game so nothing from the
match being predicted leaks in: rolling player form (3/5/10-game and career), own-team form, and
opponent <em>defence</em> conceded, joined as-of kickoff. Performance points =
<code>4·points + 10·tryAssists + 5·lineBreaks + 1·tackles + ⌊runMetres/10⌋</code>.</p>

<h3>2. From a prediction to a price</h3>
<p>The model predicts the <em>mean</em> of each stat; a bookmaker prices the whole
<em>distribution</em>. We turn each prediction into a calibrated <code>Normal(mean, σ)</code>,
where σ grows with the mean (<code>σ = α + β·mean</code>, fitted from out-of-time residuals).
Any posted line is then priced off the normal CDF, using the standard integer-line push band
(±0.5) and quarter-line 50/50 split. That yields a fair price for over and under.</p>

<h3>3. Finding value</h3>
<p>The bookmaker's two-way price is <strong>de-vigged</strong> (margin removed) to a market-implied
probability. We compare it to the model probability and compute expected value per dollar,
treating a push as a returned stake. A positive EV means the model thinks the offered price is
too long. Odds are pulled from the Sportsbet and Ladbrokes public APIs and refreshed every six
hours; player tackle / run-metre / fantasy lines typically open one to two days before kickoff.</p>

<h3>4. Lineups</h3>
<p>Predictions use the confirmed NRL team lists (named 1–17) scraped from nrl.com, falling back
to each squad's most-recent line-up if the official list is not yet published.</p>
</section>"""
    return page("Methodology", body, "method", updated)


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
font-size:12px;color:var(--mut)}.try i{color:var(--acc);font-style:normal}
table.value td:first-child{text-align:left}table.value tr.pos td b{color:var(--pos)}
.prose{max-width:780px}.prose h3{margin:22px 0 6px;font-size:17px}.prose p{color:#c4d2de}
.prose code{background:var(--chip);padding:1px 5px;border-radius:5px;font-size:13px}
footer{border-top:1px solid var(--line);margin-top:40px;padding:22px 0;color:var(--mut);font-size:12.5px}
footer p{margin:5px 0}footer .rg{color:#b58}footer a{color:var(--acc)}
@media(max-width:640px){.hero h1{font-size:22px}nav a{padding:6px 8px}}
"""


def main():
    rnd = int(sys.argv[1]) if len(sys.argv) > 1 else None
    preds, odds, edges, acc = load_inputs()
    if rnd is None:
        rnd = int(preds["roundNumber"].iloc[0]) if "roundNumber" in preds else 0
    updated = now_aest().strftime("%a %d %b %Y, %I:%M%p AEST")

    os.makedirs(DOCS, exist_ok=True)
    open(f"{DOCS}/style.css", "w").write(CSS)
    open(f"{DOCS}/index.html", "w").write(build_index(preds, odds, edges, rnd, updated))
    open(f"{DOCS}/value.html", "w").write(build_value(edges, updated))
    open(f"{DOCS}/accuracy.html", "w").write(build_accuracy(acc, updated))
    open(f"{DOCS}/methodology.html", "w").write(build_method(updated))
    open(f"{DOCS}/.nojekyll", "w").write("")  # serve files verbatim
    n_matches = preds["matchId"].nunique()
    print(f"Built {DOCS}/ — round {rnd}, {n_matches} matches, "
          f"{len(edges)} priced markets, updated {updated}")


if __name__ == "__main__":
    main()
