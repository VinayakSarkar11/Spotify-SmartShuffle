"""
One-off migration: re-run combined LIS skip inference for all multi-push
rolling sessions. Historical sessions were processed per-push (10 songs
at a time), missing cross-batch skips. This replaces those stale results
with the full-session combined inference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
from src.collect import _infer_rolling_session_skips

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/smartshuffle.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    sessions = conn.execute("""
        SELECT COALESCE(rolling_session_id, push_id) AS session_id,
               COUNT(*) AS num_pushes,
               MIN(pushed_at) AS first_push
        FROM queue_pushes
        WHERE mode = 'rolling'
        GROUP BY session_id
        HAVING num_pushes > 1
        ORDER BY session_id DESC
    """).fetchall()

    print(f"Found {len(sessions)} multi-push rolling sessions to reprocess.\n")

    for session_id, num_pushes, first_push in sessions:
        before = conn.execute(
            "SELECT COUNT(*) FROM queue_skips WHERE push_id IN "
            "(SELECT push_id FROM queue_pushes WHERE COALESCE(rolling_session_id, push_id) = ?)",
            (session_id,)
        ).fetchone()[0]

        _infer_rolling_session_skips(conn, session_id)

        after = conn.execute(
            "SELECT COUNT(*) FROM queue_skips WHERE push_id IN "
            "(SELECT push_id FROM queue_pushes WHERE COALESCE(rolling_session_id, push_id) = ?)",
            (session_id,)
        ).fetchone()[0]

        date = first_push[:10]
        delta = after - before
        sign  = "+" if delta >= 0 else ""
        print(f"  session {session_id:4d}  ({num_pushes:2d} pushes, {date})  "
              f"skips: {before} → {after}  ({sign}{delta})")

    print(f"\nDone. Commit changes? [y/N] ", end="")
    if input().strip().lower() == "y":
        conn.commit()
        print("Committed.")
    else:
        conn.rollback()
        print("Rolled back.")

    conn.close()


if __name__ == "__main__":
    main()
