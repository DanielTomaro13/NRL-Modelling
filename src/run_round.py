"""
CI driver: detect the current competition + round, fetch confirmed team lists
(falling back to the most-recent-XVII proxy), and predict the round.

Run from the repo root:  python src/run_round.py
"""
import sys, subprocess, os
import nrl_meta as M
import tracks as T

PY = sys.executable


def main():
    track = T.current()
    comp, meta = M.current_competition(track)
    fx = M.fixture(comp)
    rnd = M.next_round(comp, fx)
    matches = M.round_matches(comp, rnd, fx)
    print(f"[{track.name}] competition {comp} ({meta['name']}) — predicting round {rnd} "
          f"({len(matches)} matches)")

    lineups = T.proc(f"lineups_r{rnd}.parquet", track)
    predict_args = [PY, "src/predict.py", str(comp), str(rnd)]

    # confirmed nrl.com team lists are men's-NRL-only (slug + format differ for NRLW /
    # Origin); other tracks use the most-recent-XVIII proxy until track-specific
    # scrapers exist.
    url = M.find_teamlist_url(rnd, matches, meta.get("season")) if track.name == "nrl" else None
    if url:
        print(f"Confirmed team lists: {url}")
        try:
            subprocess.run([PY, "src/scrape_teamlists.py", url, str(comp), str(rnd)],
                           check=True)
            if os.path.exists(lineups):
                predict_args.append(lineups)
        except subprocess.CalledProcessError as e:
            print(f"team-list scrape failed ({e}); falling back to XVIII proxy")
    else:
        print("Using most-recent-XVIII proxy lineups")

    env = {**os.environ, "TRACK": track.name}
    subprocess.run(predict_args, check=True, env=env)
    # match-outcome markets for the same round
    subprocess.run([PY, "src/team_model.py", "predict", str(comp), str(rnd)],
                   check=False, env=env)
    T.ensure_dirs(track)
    open(T.report("current_round.txt", track), "w").write(str(rnd))
    print(f"[{track.name}] predicted round {rnd}.")


if __name__ == "__main__":
    main()
