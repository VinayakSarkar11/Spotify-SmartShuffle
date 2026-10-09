"""
Compute README stats from the SmartShuffle DB.
Run from the repo root: python scripts/readme_stats.py
"""
import os, json, sqlite3
from scipy.stats import chi2_contingency

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "smartshuffle.db")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

QUALIFY_PLAYS = 4  # minimum plays for a session to count

# ── A/B table ─────────────────────────────────────────────────────────────────
# Skip rate = (play_skips + queue_skips) / (plays + queue_skips)
# play_skips: attributed plays with inferred_skip = 'skip'
# queue_skips: songs never played at all (hard skipped)
# Sessions filtered to >= QUALIFY_PLAYS attributed plays.

results = {}

for algo in ("smartshuffle", "random_baseline"):
    play_source = f"{algo}_queued"

    session_rows = conn.execute(f"""
        WITH session_plays AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS plays_n,
                   SUM(CASE WHEN p.inferred_skip = 'skip' THEN 1 ELSE 0 END) AS skip_n
            FROM plays p
            JOIN queue_pushes qp ON qp.push_id = (
                SELECT qp2.push_id FROM queue_pushes qp2
                WHERE qp2.algorithm = ?
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1
            )
            WHERE p.play_source = ?
              AND p.inferred_skip IN ('skip', 'partial', 'full')
              AND qp.mode = 'rolling'
            GROUP BY session_id
            HAVING plays_n >= {QUALIFY_PLAYS}
        ),
        session_qs AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS qs_n
            FROM queue_skips qs
            JOIN queue_pushes qp ON qp.push_id = qs.push_id
            WHERE qp.algorithm = ?
              AND qp.mode = 'rolling'
              AND qs.queue_position IS NOT NULL
            GROUP BY session_id
        )
        SELECT
            COUNT(DISTINCT sp.session_id)          AS sessions,
            SUM(sp.plays_n)                        AS plays_n,
            SUM(sp.skip_n)                         AS play_skips,
            SUM(COALESCE(sqs.qs_n, 0))             AS qs_n,
            AVG(sp.plays_n)                        AS avg_plays
        FROM session_plays sp
        LEFT JOIN session_qs sqs ON sqs.session_id = sp.session_id
    """, (algo, play_source, algo)).fetchone()

    r = dict(session_rows)
    total_skips = (r["play_skips"] or 0) + (r["qs_n"] or 0)
    total_plays = (r["plays_n"] or 0) + (r["qs_n"] or 0)
    r["skip_rate"]   = total_skips / total_plays if total_plays else 0
    r["total_skips"] = total_skips
    r["total_for_rate"] = total_plays
    results[algo] = r

print("\n=== A/B TABLE ===")
ss = results["smartshuffle"]
rb = results["random_baseline"]

for label, algo, r in [("SmartShuffle (rolling)", "smartshuffle", ss),
                        ("Random baseline",        "random_baseline", rb)]:
    print(f"  {label:28s}  sessions={r['sessions']}  plays={r['plays_n']}"
          f"  skip_rate={r['skip_rate']:.1%}  avg_plays={r['avg_plays']:.1f}")

pp_diff       = rb["skip_rate"] - ss["skip_rate"]
pct_more_plays = (ss["avg_plays"] - rb["avg_plays"]) / rb["avg_plays"]
print(f"\n  Skip rate diff : {pp_diff*100:.1f}pp lower for SS")
print(f"  Plays/session  : SS averages {pct_more_plays:.0%} more plays")

# Chi-squared
ss_complete = ss["total_for_rate"] - ss["total_skips"]
rb_complete = rb["total_for_rate"] - rb["total_skips"]
contingency  = [[int(ss["total_skips"]), int(ss_complete)],
                [int(rb["total_skips"]), int(rb_complete)]]
chi2, p, *_ = chi2_contingency(contingency)
p_str = f"{p:.4g}" if p >= 0.0001 else "p<0.0001"
print(f"  χ²={chi2:.2f}, {p_str}")

# ── Song coverage ──────────────────────────────────────────────────────────────
print("\n=== SONG COVERAGE ===")
cov = conn.execute("""
    WITH all_songs AS (
        SELECT DISTINCT song_id FROM playlist_tracks
    ),
    ss_played AS (
        SELECT DISTINCT p.song_id
        FROM plays p
        WHERE p.play_source = 'smartshuffle_queued'
    ),
    rb_played AS (
        SELECT DISTINCT p.song_id
        FROM plays p
        WHERE p.play_source = 'random_baseline_queued'
    )
    SELECT
        COUNT(DISTINCT a.song_id)              AS total_songs,
        COUNT(DISTINCT ss.song_id)             AS ss_songs,
        COUNT(DISTINCT rb.song_id)             AS rb_songs
    FROM all_songs a
    LEFT JOIN ss_played ss ON ss.song_id = a.song_id
    LEFT JOIN rb_played rb ON rb.song_id = a.song_id
""").fetchone()

ss_pct = cov["ss_songs"] / cov["total_songs"] if cov["total_songs"] else 0
rb_pct = cov["rb_songs"] / cov["total_songs"] if cov["total_songs"] else 0
print(f"  Total unique playlist songs : {cov['total_songs']}")
print(f"  SS   songs played ≥1×       : {cov['ss_songs']} ({ss_pct:.1%})")
print(f"  RB   songs played ≥1×       : {cov['rb_songs']} ({rb_pct:.1%})")

conn.close()
print()
