#!/usr/bin/env python3
"""
SmartShuffle pipeline runner — collect → score → train → recommend.

Usage:
    python run.py                                   # full pipeline, default playlist
    python run.py --playlist Chill --scores         # specific playlist with score detail
    python run.py --playlist Chill --play           # generate + push to Spotify
    python run.py --skip-collect                    # skip data collection
    python run.py --skip-collect --skip-model       # score + queue only (fastest)

All unrecognised flags pass through to recommend.py:
    python run.py --playlist Hype --context morning --k 2 --scores
"""
import argparse
import os
import sqlite3
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
PY  = sys.executable
sys.path.insert(0, os.path.join(DIR, "src"))


def step(n: int, total: int, label: str, cmd: list) -> bool:
    print(f"\n{'═' * 56}")
    print(f"  [{n}/{total}]  {label}")
    print(f"{'═' * 56}\n")
    return subprocess.run(cmd).returncode == 0


def _start_watcher():
    """Spawn the session watcher daemon and return its stop_event."""
    from recommend import current_time_bucket, load_engagement_baselines, DB_PATH
    from watcher import start_watcher

    bucket = current_time_bucket()
    try:
        conn      = sqlite3.connect(DB_PATH)
        baselines = load_engagement_baselines(conn)
        conn.close()
        baseline_rate = baselines.get(bucket, {}).get("baseline_skip_rate", 0.0)
    except Exception:
        baseline_rate = 0.0

    return start_watcher(bucket, baseline_rate)


def main():
    parser = argparse.ArgumentParser(
        description="SmartShuffle pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip collect.py (if already run recently)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip train.py retraining")
    parser.add_argument("--play", action="store_true",
                        help="Push the generated SmartShuffle queue to Spotify after generation")
    known, phase34_args = parser.parse_known_args()

    pipeline = []
    if not known.skip_collect:
        pipeline.append(("Collecting plays",     [PY, f"{DIR}/src/collect.py"]))
    pipeline.append(("Behavioral modeling",       [PY, f"{DIR}/src/score.py"]))
    if not known.skip_model:
        pipeline.append(("Training model",        [PY, f"{DIR}/src/train.py"]))
    pipeline.append(("Generating queue",          [PY, f"{DIR}/src/recommend.py"] + phase34_args))
    if known.play:
        pipeline.append(("Playing on Spotify",    [PY, f"{DIR}/src/push.py"]))

    total      = len(pipeline)
    stop_event = None

    try:
        for n, (label, cmd) in enumerate(pipeline, 1):
            # Start the watcher just before handing off to Spotify so it's live
            # during the actual listening session, not during pipeline setup.
            if label == "Playing on Spotify":
                stop_event = _start_watcher()

            if not step(n, total, label, cmd):
                print(f"\n  Pipeline stopped: {label}")
                sys.exit(1)
    finally:
        if stop_event is not None:
            stop_event.set()


if __name__ == "__main__":
    main()
