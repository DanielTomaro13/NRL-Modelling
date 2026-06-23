"""
CI driver: detect the current competition + round, fetch confirmed team lists
(falling back to the most-recent-XVII proxy), and predict the round.

Run from the repo root:  python src/run_round.py
"""
import sys, subprocess, os
import nrl_meta as M

PY = sys.executable


def main():
    comp, meta = M.current_competition()
    fx = M.fixture(comp)
    rnd = M.next_round(comp, fx)
    matches = M.round_matches(comp, rnd, fx)
    print(f"Competition {comp} ({meta['name']}) — predicting round {rnd} "
          f"({len(matches)} matches)")

    url = M.find_teamlist_url(rnd, matches, meta.get("season"))
    lineups = f"data/processed/lineups_r{rnd}.parquet"
    predict_args = [PY, "src/predict.py", str(comp), str(rnd)]

    if url:
        print(f"Confirmed team lists: {url}")
        try:
            subprocess.run([PY, "src/scrape_teamlists.py", url, str(comp), str(rnd)],
                           check=True)
            if os.path.exists(lineups):
                predict_args.append(lineups)
        except subprocess.CalledProcessError as e:
            print(f"team-list scrape failed ({e}); falling back to XVII proxy")
    else:
        print("No published team-list URL found yet; using most-recent-XVII proxy")

    subprocess.run(predict_args, check=True)
    os.makedirs("reports", exist_ok=True)
    open("reports/current_round.txt", "w").write(str(rnd))
    print(f"Predicted round {rnd}.")


if __name__ == "__main__":
    main()
