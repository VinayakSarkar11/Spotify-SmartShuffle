"""
One-off migration: fix plays stuck at inferred_skip='unknown'.

Every time collect.py ran during an active session, the last play it fetched
was inserted with inferred_skip='unknown' (no successor yet). INSERT OR IGNORE
meant it was never updated on later runs. This script finds all such plays that
now have a successor in the DB, computes the correct gap-based skip status, and
updates them.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
from src.collect import infer_skip

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/smartshuffle.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # Find unknown plays that have a successor in the DB
    rows = conn.execute("""
        SELECT p1.played_at, p1.duration_ms,
               p2.played_at AS next_played_at
        FROM plays p1
        JOIN plays p2 ON p2.played_at = (
            SELECT p3.played_at FROM plays p3
            WHERE p3.played_at > p1.played_at
            ORDER BY p3.played_at
            LIMIT 1
        )
        WHERE p1.inferred_skip = 'unknown'
        ORDER BY p1.played_at
    """).fetchall()

    print(f"Found {len(rows)} unknown plays with a known successor.\n")

    updated = 0
    for played_at, duration_ms, next_played_at in rows:
        from datetime import datetime, timezone
        t1 = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(next_played_at.replace("Z", "+00:00"))
        gap_ms = int((t2 - t1).total_seconds() * 1000)
        play_duration_ms = min(gap_ms, duration_ms) if duration_ms else gap_ms
        new_skip = infer_skip(play_duration_ms, duration_ms)

        if new_skip != "unknown":
            conn.execute(
                "UPDATE plays SET play_duration_ms = ?, inferred_skip = ? WHERE played_at = ?",
                (play_duration_ms, new_skip, played_at)
            )
            updated += 1

    print(f"Updated {updated} plays (remainder had gap < 1s or missing duration → stays unknown).")
    print(f"\nCommit? [y/N] ", end="")
    if input().strip().lower() == "y":
        conn.commit()
        print("Committed.")
    else:
        conn.rollback()
        print("Rolled back.")

    conn.close()


if __name__ == "__main__":
    main()
