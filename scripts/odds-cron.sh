#!/bin/bash
# NRL odds refresh — runs from an AU IP because Sportsbet + TAB geo-block GitHub's US
# runners (curl_cffi gets Ladbrokes/Dabble/PointsBet through, but Sportsbet/TAB are
# IP-blocked, so CI could only carry forward stale snapshots). This scrapes all five
# books fresh, re-prices against the committed predictions, rebuilds the site bundle,
# commits, pushes, and triggers the nrl24-0.com redeploy.
#
# Driven by a launchd agent every 3h. Model TRAINING stays in CI (model.yml, daily).
set -uo pipefail

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
BOT=/Users/danieltomaro/sports-bots
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$BOT/nrl-venv/bin/python"
cd "$REPO" || exit 1
LOG="$REPO/scripts/odds-cron.log"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
exec >>"$LOG" 2>&1
echo "===== $(ts) nrl odds-cron start ($REPO) ====="

# Shared bookmaker creds (TAB_*, DABBLE_* …) — same file AFL uses.
set -a; [ -f "$BOT/secrets.env" ] && . "$BOT/secrets.env"; set +a

git pull --rebase --autostash origin main || { echo "$(ts) pull failed"; exit 1; }

# Odds → price → site. odds.py is best-effort per book (carry-forward inside), the
# rest must succeed to produce a coherent priced site.
"$PY" src/odds.py            || echo "$(ts) odds.py nonzero (continuing)"
"$PY" src/pricing.py price   || { echo "$(ts) pricing price failed"; exit 1; }
"$PY" src/pricing.py tries   || echo "$(ts) pricing tries nonzero"
"$PY" src/pricing.py team    || echo "$(ts) pricing team nonzero"
"$PY" src/player_points.py price || echo "$(ts) player_points price nonzero"
"$PY" src/pickem.py          || echo "$(ts) pickem nonzero"
"$PY" src/compare.py         || echo "$(ts) compare nonzero"
"$PY" src/export_site_data.py || echo "$(ts) export nonzero"
"$PY" src/build_site.py      || { echo "$(ts) build_site failed"; exit 1; }

# SuperCoach feed for nrl24-0.com — refresh at most once a day (prices/news move
# weekly-ish; it pulls a per-round score series, heavier than the odds scrape).
SC=reports/site/supercoach.json
if [ ! -f "$SC" ] || [ -n "$(find "$SC" -mmin +1200 2>/dev/null)" ]; then
  echo "$(ts) refreshing supercoach…"
  "$PY" src/supercoach.py || echo "$(ts) supercoach scrape failed (keeping previous)"
fi

git add -A reports docs
if git diff --cached --quiet; then
  echo "$(ts) no changes — nothing to push"; exit 0
fi
git config user.name  "nrl-odds-bot"
git config user.email "nrl-odds-bot@localhost"
git commit -m "Refresh NRL odds + site (local AU scrape $(date -u +%Y-%m-%dT%H:%MZ))"
for i in 1 2 3 4 5; do
  if git push origin HEAD:main; then echo "$(ts) pushed (attempt $i)"; break; fi
  echo "$(ts) push rejected (attempt $i) — rebasing..."
  git pull --rebase --autostash origin main || { echo "$(ts) rebase failed"; exit 1; }
  [ "$i" = 5 ] && { echo "$(ts) failed to push"; exit 1; }
done

# Trigger nrl24-0.com redeploy (uses the git PAT from the credential store).
TOKEN=$(sed -E 's#^https?://[^:]*:([^@]+)@.*#\1#' <<<"$(grep github.com ~/.git-credentials | head -1)")
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/DanielTomaro13/NRL-24-0/actions/workflows/deploy.yml/dispatches \
  -d '{"ref":"main"}')
echo "$(ts) nrl24-0 redeploy dispatch -> $code"
