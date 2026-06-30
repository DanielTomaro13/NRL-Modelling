#!/usr/bin/env bash
# Run one competition track's daily model pipeline. TRACK selects paths + comps
# (see src/tracks.py). Call order matters: nrl/nrlw must run BEFORE soo/soow,
# which reuse the men's/NRLW player-prop, try-scorer and kicker models.
#
#   scripts/run_track.sh nrl|nrlw|soo|soow
#
# nrl  : full train; site export is done by the local AU odds cron (needs odds).
# nrlw : full train + match-outcome model + site export (no AU-book odds needed).
# soo/soow: reuse the club models (predict-only) + site export.
set -euo pipefail

TRACK="${1:?usage: run_track.sh <nrl|nrlw|soo|soow>}"
export TRACK
PY="${PYTHON:-python}"
run() { echo "::group::[$TRACK] $1"; "$PY" "src/$1.py" "${@:2}"; echo "::endgroup::"; }

case "$TRACK" in
  nrl)
    run ingest; run features; run train; run run_round
    run analysis; run tryscorer; run kicker; run player_points
    ;;
  nrlw)
    run ingest; run features; run train
    run team_model train
    run run_round                 # predicts props + match markets for the round
    run tryscorer; run kicker; run player_points
    run export_site_data
    ;;
  soo|soow)
    run ingest                    # club + Origin history (club matches are cached)
    run run_round                 # reuses the club model via model_for_prediction
    run tryscorer; run kicker; run player_points   # reuse-model predict-only branch
    run export_site_data
    ;;
  *)
    echo "unknown track: $TRACK" >&2; exit 2 ;;
esac
echo "[$TRACK] done."
