#!/usr/bin/env python3
"""
Evaluate vibe scores using the negative oracle from behavioral data.

Logic:
  Within a session, when song A completes and song B is the immediate next play:
  - If B is skipped early → A and B are probably vibe-incompatible
  - If B completes       → A and B are probably vibe-compatible

  We can't prove two songs BELONG together, but a consistent skip right after A
  is strong evidence B doesn't fit the vibe A was setting.

  Evaluation: skipped-after pairs should have higher vibe distance than
  completed-after pairs. If our vibe scores are good, we'll see this gap.

Usage:
  python evaluate_vibes.py                        # all playlists
  python evaluate_vibes.py --playlist Chill       # filter to Chill playlist songs
  python evaluate_vibes.py --playlist <id>        # by playlist ID
"""
import argparse
import math
import os
import sqlite3
from collections import Counter

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "smartshuffle.db")
CHILL_ID  = "3QotZUtXE5w5aLeR3v9PmI"

parser = argparse.ArgumentParser()
parser.add_argument("--playlist", default=None,
                    help="Playlist name (substring) or ID to restrict analysis to")
args = parser.parse_args()

conn = sqlite3.connect(DB_PATH)

# ── Resolve playlist filter ────────────────────────────────────────────────────
playlist_id   = None
playlist_name = "all playlists"

if args.playlist:
    # Try exact ID match first, then name substring
    row = conn.execute(
        "SELECT playlist_id, playlist_name FROM playlists WHERE playlist_id = ?",
        (args.playlist,)
    ).fetchone()
    if not row:
        # Prefer shortest name match so "Chill" doesn't grab "Chillly"
        candidates = conn.execute(
            "SELECT playlist_id, playlist_name FROM playlists "
            "WHERE LOWER(playlist_name) LIKE LOWER(?)",
            (f"%{args.playlist}%",)
        ).fetchall()
        if candidates:
            row = min(candidates, key=lambda r: len(r[1]))
    if row:
        playlist_id   = row[0]
        playlist_name = row[1].strip()
    else:
        print(f"Playlist not found: {args.playlist!r}")
        conn.close()
        exit(1)

n_scored = conn.execute(
    "SELECT COUNT(*) FROM songs WHERE vibe_content IS NOT NULL"
).fetchone()[0]
print(f"Songs with vibe scores: {n_scored}")
if n_scored == 0:
    print("Run score_vibes.py first.")
    conn.close()
    exit(1)

if playlist_id:
    n_pl = conn.execute(
        "SELECT COUNT(*) FROM playlist_tracks pt "
        "JOIN songs s ON s.song_id = pt.song_id "
        "WHERE pt.playlist_id = ? AND s.vibe_content IS NOT NULL",
        (playlist_id,)
    ).fetchone()[0]
    print(f"Playlist: {playlist_name}  ({n_pl} scored songs)\n")
else:
    print()

# ── Build consecutive-play pairs within sessions ───────────────────────────────
# Optional playlist filter: both A and B must be in the playlist
playlist_join = ""
playlist_where = ""
if playlist_id:
    playlist_join = """
        JOIN playlist_tracks pt_a ON pt_a.song_id = sp.prev_song_id
          AND pt_a.playlist_id = '{pid}'
        JOIN playlist_tracks pt_b ON pt_b.song_id = sp.song_id
          AND pt_b.playlist_id = '{pid}'
    """.format(pid=playlist_id)

query = f"""
    WITH session_plays AS (
        SELECT
            s.session_id,
            p.song_id,
            p.played_at,
            p.inferred_skip,
            LAG(p.song_id)       OVER w AS prev_song_id,
            LAG(p.inferred_skip) OVER w AS prev_skip
        FROM sessions s
        JOIN plays p
          ON p.played_at >= s.start_time
         AND p.played_at <= s.end_time
        WHERE p.play_source NOT IN ('jam_excluded')
        WINDOW w AS (PARTITION BY s.session_id ORDER BY p.played_at)
    )
    SELECT
        sp.prev_song_id  AS song_a,
        sp.song_id       AS song_b,
        sp.inferred_skip AS b_skip,
        sa.vibe_content  AS a_content,
        sa.vibe_melodic  AS a_melodic,
        sa.vibe_bpm      AS a_bpm,
        sb.vibe_content  AS b_content,
        sb.vibe_melodic  AS b_melodic,
        sb.vibe_bpm      AS b_bpm,
        sa.song_name     AS a_name,
        sb.song_name     AS b_name
    FROM session_plays sp
    JOIN songs sa ON sa.song_id = sp.prev_song_id
    JOIN songs sb ON sb.song_id = sp.song_id
    {playlist_join}
    WHERE sp.prev_song_id IS NOT NULL
      AND sp.prev_skip IN ('full', 'partial')
      AND sa.vibe_content IS NOT NULL
      AND sb.vibe_content IS NOT NULL
"""

pairs = conn.execute(query).fetchall()
conn.close()

print(f"Consecutive pairs analysed: {len(pairs)}")
if not pairs:
    print("No pairs found — not enough session data with vibe-scored songs.")
    exit(0)

# ── Compute vibe distance ──────────────────────────────────────────────────────
def vibe_dist(row) -> float:
    dc = (row[3] - row[6]) ** 2
    dm = (row[4] - row[7]) ** 2
    db = (row[5] - row[8]) ** 2
    return math.sqrt(dc + dm + db)

skipped_dists   = []
completed_dists = []

for row in pairs:
    d = vibe_dist(row)
    if row[2] == "skip":
        skipped_dists.append((d, row[9], row[10]))
    elif row[2] in ("full", "partial"):
        completed_dists.append((d, row[9], row[10]))

# ── Stats ──────────────────────────────────────────────────────────────────────
def stats(dists):
    vals = sorted(d for d, _, _ in dists)
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0}
    n = len(vals)
    return {
        "n":      n,
        "mean":   sum(vals) / n,
        "median": vals[n // 2],
        "p75":    vals[int(n * 0.75)],
        "p90":    vals[int(n * 0.90)],
    }

sk = stats(skipped_dists)
cp = stats(completed_dists)

print(f"\n{'':30} {'skipped-after':>14} {'completed-after':>16}")
print("-" * 62)
for k in ("n", "mean", "median", "p75", "p90"):
    if k == "n":
        print(f"  {k:<28} {sk[k]:>14d} {cp[k]:>16d}")
    else:
        print(f"  {k:<28} {sk[k]:>14.3f} {cp[k]:>16.3f}")

if sk["mean"] > 0 and cp["mean"] > 0:
    lift = (sk["mean"] - cp["mean"]) / cp["mean"] * 100
    print(f"\n  Vibe distance lift (skip vs complete): {lift:+.1f}%")
    if lift > 10:
        print("  ✓  Skipped pairs are more vibe-distant — scores are predictive.")
    elif lift > 0:
        print("  ~  Weak positive signal — scores partially predict skips.")
    else:
        print("  ✗  No signal yet — need more sessions or better scores.")

# ── Top completed pairs with high vibe distance (unexpected completions) ───────
print(f"\nTop 10 highest-distance COMPLETED pairs (vibe-distant but user stayed):")
completed_dists.sort(reverse=True)
print(f"  {'Song A':<28} {'Song B':<28} {'dist':>6}")
print("  " + "-" * 66)
for d, a, b in completed_dists[:10]:
    print(f"  {(a or '?')[:27]:<28} {(b or '?')[:27]:<28} {d:>6.3f}")

# ── Top skipped pairs with high vibe distance (oracle confirmed) ───────────────
print(f"\nTop 10 highest-distance SKIPPED pairs (oracle confirmed mismatch):")
skipped_dists.sort(reverse=True)
print(f"  {'Song A':<28} {'Song B':<28} {'dist':>6}")
print("  " + "-" * 66)
for d, a, b in skipped_dists[:10]:
    print(f"  {(a or '?')[:27]:<28} {(b or '?')[:27]:<28} {d:>6.3f}")

# ── Consistent A→B skips (≥2 occurrences) ─────────────────────────────────────
print(f"\nConsistent A→B skips (negative oracle, ≥2 occurrences):")
skip_counter = Counter((a, b) for _, a, b in skipped_dists)
consistent = sorted(
    [(cnt, a, b) for (a, b), cnt in skip_counter.items() if cnt >= 2],
    reverse=True
)
if consistent:
    print(f"  {'Song A':<28} {'Song B':<28} {'skips':>6}")
    print("  " + "-" * 66)
    for cnt, a, b in consistent[:15]:
        print(f"  {(a or '?')[:27]:<28} {(b or '?')[:27]:<28} {cnt:>6}")
else:
    print("  None yet — need more sessions for repeat patterns.")
