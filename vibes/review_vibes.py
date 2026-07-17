#!/usr/bin/env python3
"""
Manual evaluation of vibe scores for the Chill playlist.

Views:
  1. Known test songs grouped by mental category
  2. Axis leaderboards — most/least melodic, aggressive, fast within Chill
  3. 2D quadrant map — content × melodic so you can see how the model clusters songs
  4. Score distributions

Usage:
  python review_vibes.py
  python review_vibes.py --playlist <id>   # override playlist
"""
import argparse
import os
import sqlite3

_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH        = os.path.join(_ROOT, "data", "smartshuffle.db")
CHILL_ID       = "3QotZUtXE5w5aLeR3v9PmI"
LEADERBOARD_N  = 15

# ── Your mental categories (subset of Chill) ──────────────────────────────────
TEST_GROUPS = {
    "HYPE-CHILL": [
        "Patiently Waiting", "On Me", "Bump Heads",
        "DONT KILL THE PARTY", "GOMD",
    ],
    "MELODIC-CHILL": [
        "Can't Say", "Calling My Phone", "Greece",
        "Better Now", "Lemonade", "500 lbs",
        "Hold On, We're Going Home",
    ],
    "EMOTIONAL": [
        "Love Yourz", "CHIHIRO", "Lighters",
    ],
    "JUST RAPPING": [
        "Stick Up Kids in Vegas", "Change",
        "Took A While", "Don't Believe The Hype",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("--playlist", default=CHILL_ID)
args = parser.parse_args()

conn = sqlite3.connect(DB_PATH)

pl_name = conn.execute(
    "SELECT playlist_name FROM playlists WHERE playlist_id = ?", (args.playlist,)
).fetchone()
pl_name = (pl_name[0] if pl_name else args.playlist).strip()

rows = conn.execute("""
    SELECT s.song_id, s.song_name, s.artist_name,
           s.vibe_content, s.vibe_melodic, s.vibe_bpm
    FROM playlist_tracks pt
    JOIN songs s ON s.song_id = pt.song_id
    WHERE pt.playlist_id = ?
      AND s.vibe_content IS NOT NULL
    ORDER BY s.song_name
""", (args.playlist,)).fetchall()

total = conn.execute(
    "SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = ?", (args.playlist,)
).fetchone()[0]
conn.close()

print(f"Playlist: {pl_name}  ({len(rows)}/{total} songs scored)\n")

if not rows:
    print("Run score_vibes.py first.")
    exit(0)

by_name = {}
for r in rows:
    key = (r[1] or "").lower()
    by_name.setdefault(key, []).append(r)

def find(name: str):
    exact = by_name.get(name.lower())
    if exact:
        return exact[0]
    matches = [r for r in rows if name.lower() in (r[1] or "").lower()]
    return matches[0] if matches else None

# ── View 1: Test songs by mental category ─────────────────────────────────────
print("=" * 72)
print(f"VIEW 1 — TEST SONGS IN {pl_name.upper()}")
print("  content: -1=philosophical  +1=aggressive")
print("  melodic: -1=singing        +1=pure rap")
print("  bpm:     -1=slow           +1=fast")
print("=" * 72)

for group, names in TEST_GROUPS.items():
    print(f"\n  ── {group} ──")
    print(f"  {'Song':<32} {'Artist':<22} {'content':>8} {'melodic':>8} {'bpm':>6}")
    print("  " + "-" * 78)
    for name in names:
        r = find(name)
        if r:
            print(f"  {(r[1] or '')[:31]:<32} {(r[2] or '')[:21]:<22}"
                  f"  {r[3]:>6.2f}   {r[4]:>6.2f}  {r[5]:>5.2f}")
        else:
            print(f"  {name[:31]:<32} {'(not in playlist)':<22}")

# ── View 2: Axis leaderboards (Chill playlist only) ───────────────────────────
axes = [
    (3, "CONTENT",  "philosophical ← · → aggressive"),
    (4, "MELODIC",  "singing ← · → pure rap"),
    (5, "BPM",      "slow ← · → fast"),
]

print(f"\n\n{'=' * 72}")
print(f"VIEW 2 — AXIS LEADERBOARDS IN {pl_name.upper()}  (top/bottom {LEADERBOARD_N})")
print(f"{'=' * 72}")

for col_idx, label, description in axes:
    sorted_asc  = sorted(rows, key=lambda r: r[col_idx])
    sorted_desc = sorted(rows, key=lambda r: r[col_idx], reverse=True)

    print(f"\n  ── {label}: {description} ──")
    print(f"  {'Rank':<5} {'Song':<32} {'Artist':<22} {label:>7}")
    print("  " + "-" * 68)

    print(f"  [ BOTTOM — low score ]")
    for i, r in enumerate(sorted_asc[:LEADERBOARD_N], 1):
        print(f"  {i:<5} {(r[1] or '')[:31]:<32} {(r[2] or '')[:21]:<22} {r[col_idx]:>6.2f}")

    print(f"  [ TOP — high score ]")
    for i, r in enumerate(sorted_desc[:LEADERBOARD_N], 1):
        print(f"  {i:<5} {(r[1] or '')[:31]:<32} {(r[2] or '')[:21]:<22} {r[col_idx]:>6.2f}")

# ── View 3: 2D quadrant map (content × melodic) ───────────────────────────────
print(f"\n\n{'=' * 72}")
print("VIEW 3 — 2D QUADRANT MAP  (content × melodic)")
print("  Each cell: up to 4 songs. Songs closer to 0,0 are in the middle.")
print("=" * 72)

QUADRANTS = {
    ("lo_content", "lo_melodic"): ("philosophical + singing",  "EMOTIONAL / MELODIC-CHILL"),
    ("lo_content", "hi_melodic"): ("philosophical + pure rap", "JUST RAPPING"),
    ("hi_content", "lo_melodic"): ("aggressive + singing",     "rare"),
    ("hi_content", "hi_melodic"): ("aggressive + pure rap",    "HYPE-CHILL"),
}

bucketed: dict[tuple, list] = {k: [] for k in QUADRANTS}
for r in rows:
    c_bucket = "lo_content" if r[3] < 0 else "hi_content"
    m_bucket = "lo_melodic" if r[4] < 0 else "hi_melodic"
    bucketed[(c_bucket, m_bucket)].append(r)

for (cb, mb), (desc, vibe_label) in QUADRANTS.items():
    songs_in = sorted(bucketed[(cb, mb)], key=lambda r: abs(r[3]) + abs(r[4]), reverse=True)
    c_sign = "+" if cb == "hi_content" else "-"
    m_sign = "+" if mb == "hi_melodic" else "-"
    print(f"\n  [{c_sign}content, {m_sign}melodic]  {desc}  →  {vibe_label}  ({len(songs_in)} songs)")
    for r in songs_in[:6]:
        print(f"    {(r[1] or '')[:30]:<31} {(r[2] or '')[:18]:<19}"
              f"  c={r[3]:+.2f}  m={r[4]:+.2f}  bpm={r[5]:+.2f}")
    if len(songs_in) > 6:
        print(f"    ... and {len(songs_in) - 6} more")

# ── View 4: Distribution ──────────────────────────────────────────────────────
print(f"\n\n{'=' * 72}")
print(f"VIEW 4 — SCORE DISTRIBUTIONS IN {pl_name.upper()}")
print(f"{'=' * 72}")
print(f"  {'Axis':<10} {'min':>6} {'p10':>6} {'p25':>6} {'med':>6} {'p75':>6} {'p90':>6} {'max':>6}")
print("  " + "-" * 52)

for col_idx, label, _ in axes:
    vals = sorted(r[col_idx] for r in rows)
    n = len(vals)
    def pct(p): return vals[int(n * p / 100)]
    print(f"  {label:<10} {vals[0]:>6.2f} {pct(10):>6.2f} {pct(25):>6.2f} "
          f"{vals[n//2]:>6.2f} {pct(75):>6.2f} {pct(90):>6.2f} {vals[-1]:>6.2f}")
