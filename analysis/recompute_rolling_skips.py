#!/usr/bin/env python3
"""
recompute_rolling_skips.py
==========================
Retroactively re-infers queue skips for all rolling sessions by treating
each session's full sequence of pushes as one combined queue.

Before: each 10-song push was processed independently. A user jumping
        from batch-1 song 8 to batch-2 song 3 was invisible — neither
        batch saw a confirmed skip across the boundary.

After:  all pushes in a rolling session are concatenated in push order,
        and LIS runs once across the combined ~40-song sequence. Cross-push
        skips are now correctly detected.

All resulting queue_skips are stored under push_id = rolling_session_id
(the first push in the session).  Per-push results are replaced.
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import sqlite3
from collect import _infer_rolling_session_skips

DB_PATH = os.path.join(_ROOT, "data", "smartshuffle.db")
conn    = sqlite3.connect(DB_PATH)

# ── Show baseline before recompute ────────────────────────────────────────────
def skip_rates(conn):
    rows = conn.execute("""
        WITH play_push AS (
            SELECT p.play_source, p.inferred_skip,
                   (SELECT qp2.push_id FROM queue_pushes qp2
                    WHERE qp2.algorithm || '_queued' = p.play_source
                      AND qp2.pushed_at <= p.played_at
                    ORDER BY qp2.pushed_at DESC LIMIT 1) AS push_id
            FROM plays p
            WHERE p.play_source IN ('smartshuffle_queued','random_baseline_queued')
              AND p.inferred_skip IN ('skip','partial','full')
        ),
        valid_pushes AS (
            SELECT push_id FROM play_push GROUP BY push_id HAVING COUNT(*) >= 4
        ),
        per_push AS (
            SELECT pp.push_id, qp.algorithm, qp.mode,
                   COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS plays_n,
                   SUM(pp.inferred_skip='skip') AS dur_skips
            FROM play_push pp
            JOIN valid_pushes vp ON vp.push_id = pp.push_id
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            GROUP BY pp.push_id
        ),
        qs AS (
            SELECT qs.push_id, COUNT(*) AS raw_qs
            FROM queue_skips qs
            JOIN valid_pushes vp ON vp.push_id = qs.push_id
            GROUP BY qs.push_id
        )
        SELECT pp.algorithm, pp.mode,
               COUNT(DISTINCT pp.session_id) AS sessions,
               SUM(pp.plays_n) AS plays,
               SUM(pp.dur_skips) AS dur_skips,
               COALESCE(SUM(qs.raw_qs),0) AS queue_skips,
               ROUND(CAST(SUM(pp.dur_skips)+COALESCE(SUM(qs.raw_qs),0) AS REAL)
                     / (SUM(pp.plays_n)+COALESCE(SUM(qs.raw_qs),0)), 3) AS combined_rate
        FROM per_push pp
        LEFT JOIN qs ON qs.push_id = pp.push_id
        GROUP BY pp.algorithm, pp.mode
        ORDER BY pp.algorithm, pp.mode
    """).fetchall()
    return rows

def print_rates(rows, label):
    print(f"\n  {label}")
    print(f"  {'algorithm':<20} {'mode':<8} {'sessions':>8} {'plays':>6} "
          f"{'dur_sk':>6} {'q_sk':>5} {'combined':>9}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r[0]:<20} {r[1]:<8} {r[2]:>8} {r[3]:>6} "
              f"{r[4]:>6} {r[5]:>5} {r[6]:>9.1%}")

print("=" * 72)
print("  recompute_rolling_skips.py")
print("=" * 72)
before = skip_rates(conn)
print_rates(before, "BEFORE  (per-push inference)")

# ── Find all rolling sessions ──────────────────────────────────────────────────
sessions = conn.execute("""
    SELECT COALESCE(rolling_session_id, push_id) AS session_id,
           COUNT(*) AS n_pushes,
           algorithm,
           MIN(pushed_at) AS first_push,
           MAX(pushed_at) AS last_push
    FROM queue_pushes
    WHERE mode = 'rolling'
    GROUP BY session_id
    ORDER BY first_push
""").fetchall()

print(f"\n  Rolling sessions to recompute: {len(sessions)}")
for sid, n_pushes, alg, first, last in sessions:
    print(f"    session {sid:>5}  {alg:<20}  {n_pushes} pushes  {first[:16]} → {last[:16]}")

# ── Reset skips_inferred_at so collect.py won't skip them ─────────────────────
# (Not needed here since we run inference directly, but good hygiene.)
all_session_ids = [s[0] for s in sessions]

# ── Recompute ─────────────────────────────────────────────────────────────────
print()
n_ok = 0
for sid, n_pushes, alg, first, last in sessions:
    _infer_rolling_session_skips(conn, sid)
    n_ok += 1
    print(f"  [{n_ok:>2}/{len(sessions)}] session {sid:>5} ({alg}, {n_pushes} pushes) recomputed")

# ── Show after ────────────────────────────────────────────────────────────────
after = skip_rates(conn)
print_rates(after, "AFTER   (combined session inference)")

# ── Delta ─────────────────────────────────────────────────────────────────────
print("\n  Change in combined skip rate:")
before_map = {(r[0], r[1]): r for r in before}
for r in after:
    key = (r[0], r[1])
    b = before_map.get(key)
    if b:
        delta = r[6] - b[6]
        arrow = "↓" if delta < 0 else "↑" if delta > 0 else "="
        print(f"    {r[0]:<20} {r[1]:<8}  {b[6]:.1%} → {r[6]:.1%}  ({arrow}{abs(delta):.1%})")

conn.close()
print()
