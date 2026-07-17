#!/usr/bin/env python3
"""
learn_vibe_params.py — behavioral learning on the three LLM vibe axes.

Replaces the 1D energy_score as the primary behavioral signal with
(vibe_content, vibe_melodic, vibe_bpm) scored by the LLM.

Computes:
  1. Per-playlist vibe targets   — completion-weighted mean per axis
                                   tells the queue WHERE to aim for each playlist
  2. Per-bucket vibe sigmas      — within-session std dev pooled across sessions
                                   tells the queue HOW TIGHTLY to aim per axis per context

Why within-session variance for sigmas (not global):
  Global variance conflates day-to-day mood shifts with within-session tolerance.
  If you listen to all philosophical rap on Monday and all aggressive rap on Tuesday,
  the global content variance is high — but within each session it's near zero.
  Within-session std dev correctly captures "how much do you let content drift
  while a session is flowing" rather than "how different are your moods across days."

Saves to data/vibe_params.json (loaded by recommend.py).
Run after score_vibes.py has scored all songs.
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(_ROOT, "data", "smartshuffle.db")
OUTPUT_PATH = os.path.join(_ROOT, "data", "vibe_params.json")

MIN_SESSION_SONGS   = 3   # sessions with fewer scored completions excluded from sigma
MIN_BUCKET_SESSIONS = 3   # minimum sessions per bucket for a reliable sigma

AXES = ("vibe_content", "vibe_melodic", "vibe_bpm")
AXIS_LABELS = {"vibe_content": "content", "vibe_melodic": "melodic", "vibe_bpm": "bpm"}

def time_bucket(hour: int) -> str:
    if hour >= 21 or hour < 6: return "late_night"
    if hour < 11: return "morning"
    return "afternoon"


conn = sqlite3.connect(DB_PATH)

# ── Sanity check ──────────────────────────────────────────────────────────────
n_scored = conn.execute("SELECT COUNT(*) FROM songs WHERE vibe_content IS NOT NULL").fetchone()[0]
n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
print(f"Scored songs: {n_scored}  |  Sessions: {n_sessions}\n")

# ── Load completed plays with vibe scores ─────────────────────────────────────
plays_df = pd.read_sql_query("""
    SELECT
        p.song_id,
        p.played_at,
        p.play_duration_ms,
        p.duration_ms,
        s.vibe_content,
        s.vibe_melodic,
        s.vibe_bpm
    FROM plays p
    JOIN songs s ON s.song_id = p.song_id
    WHERE p.inferred_skip IN ('full', 'partial')
      AND p.play_duration_ms IS NOT NULL
      AND p.duration_ms      > 0
      AND s.vibe_content IS NOT NULL
      AND p.play_source NOT IN ('jam_excluded')
""", conn)

plays_df["played_at"]        = pd.to_datetime(plays_df["played_at"], format="ISO8601", utc=True)
plays_df["completion_ratio"] = (plays_df["play_duration_ms"] / plays_df["duration_ms"]).clip(0, 1)

print(f"Completed plays with vibe scores: {len(plays_df)}")

# ── Load sessions and derive time bucket ──────────────────────────────────────
sess_df = pd.read_sql_query("""
    SELECT session_id, start_time, end_time, hour_of_day
    FROM sessions
""", conn)

sess_df["start_time"]  = pd.to_datetime(sess_df["start_time"], format="ISO8601", utc=True)
sess_df["end_time"]    = pd.to_datetime(sess_df["end_time"],   format="ISO8601", utc=True)
sess_df["time_bucket"] = sess_df["hour_of_day"].apply(time_bucket)

# ── Load playlist membership ──────────────────────────────────────────────────
pl_tracks = pd.read_sql_query("""
    SELECT pt.song_id, pt.playlist_id, p.playlist_name
    FROM playlist_tracks pt
    JOIN playlists p ON p.playlist_id = pt.playlist_id
""", conn)

conn.close()

# ── Join plays → sessions ─────────────────────────────────────────────────────
# merge_asof joins each play to the session that started most recently before it.
# We then drop plays that fall outside that session's end_time.
plays_df = plays_df.sort_values("played_at")
sess_df  = sess_df.sort_values("start_time")

merged = pd.merge_asof(
    plays_df,
    sess_df[["session_id", "start_time", "end_time", "time_bucket"]],
    left_on="played_at",
    right_on="start_time",
    direction="backward",
)
merged = merged[merged["played_at"] <= merged["end_time"]].copy()

print(f"Plays matched to sessions: {len(merged)}")
print(f"Sessions contributing: {merged['session_id'].nunique()}")
print(f"Bucket breakdown:\n{merged.groupby('time_bucket')['session_id'].nunique().to_string()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PER-PLAYLIST VIBE TARGETS
#    Completion-weighted mean per axis for each playlist.
#    Tells the queue where to aim — "Chill listeners tend to complete songs with
#    content=-0.2, melodic=-0.1, bpm=+0.2 — aim here."
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("1. PER-PLAYLIST VIBE TARGETS  (completion-weighted mean per axis)")
print("=" * 72)

play_with_pl = merged.merge(pl_tracks, on="song_id", how="inner")

playlist_targets: dict = {}
rows_by_playlist = play_with_pl.groupby(["playlist_id", "playlist_name"])

for (pid, pname), grp in rows_by_playlist:
    if len(grp) < 3:
        continue
    w       = grp["completion_ratio"].clip(0.01)
    total_w = w.sum()

    target = {
        "content": round(float((grp["vibe_content"] * w).sum() / total_w), 4),
        "melodic": round(float((grp["vibe_melodic"] * w).sum() / total_w), 4),
        "bpm":     round(float((grp["vibe_bpm"]     * w).sum() / total_w), 4),
    }
    playlist_targets[pid] = target

    name = pname.strip()[:30]
    print(f"  {name:<30}  c={target['content']:+.3f}  "
          f"m={target['melodic']:+.3f}  bpm={target['bpm']:+.3f}  "
          f"(n={len(grp)} plays)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PER-BUCKET WITHIN-SESSION VIBE SIGMAS
#    For each session, compute the weighted mean per axis then record each
#    song's deviation from that session mean.  Pool all deviations across
#    sessions within a bucket and compute the weighted std dev.
#
#    Low σ_content in late_night → you stay tightly philosophical at night.
#    High σ_bpm in afternoon     → you don't care about BPM variation in the afternoon.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("2. PER-BUCKET WITHIN-SESSION VIBE SIGMAS")
print("   pooled within-session std dev per axis  (not global — removes day-to-day mood shifts)")
print("=" * 72)

# Accumulate per-bucket deviations
bucket_devs: dict = defaultdict(lambda: {ax: {"devs": [], "weights": []} for ax in AXES})
session_records: list = []

for (sid, bucket), grp in merged.groupby(["session_id", "time_bucket"]):
    if len(grp) < MIN_SESSION_SONGS:
        continue

    w       = grp["completion_ratio"].clip(0.01).values
    total_w = w.sum()

    session_means: dict = {}
    for ax in AXES:
        vals             = grp[ax].values.astype(float)
        mean_ax          = float((vals * w).sum() / total_w)
        session_means[ax] = mean_ax
        devs              = vals - mean_ax
        bucket_devs[bucket][ax]["devs"].extend(devs.tolist())
        bucket_devs[bucket][ax]["weights"].extend(w.tolist())

    session_records.append({
        "session_id": sid,
        "bucket":     bucket,
        "n_songs":    len(grp),
        **{AXIS_LABELS[ax]: round(session_means[ax], 3) for ax in AXES},
    })

# Weighted std dev per bucket per axis
vibe_sigmas: dict = {}

for bucket in ("morning", "afternoon", "late_night"):
    if bucket not in bucket_devs:
        print(f"\n  {bucket}: no sessions")
        continue

    n_sessions = sum(1 for r in session_records if r["bucket"] == bucket)
    n_plays    = len(bucket_devs[bucket][AXES[0]]["devs"])

    if n_sessions < MIN_BUCKET_SESSIONS:
        print(f"\n  {bucket}: only {n_sessions} sessions — sigma deferred (need {MIN_BUCKET_SESSIONS})")
        continue

    sigs: dict = {}
    for ax in AXES:
        d  = np.array(bucket_devs[bucket][ax]["devs"])
        wt = np.array(bucket_devs[bucket][ax]["weights"])
        tw = wt.sum()
        # Weighted variance of deviations (mean should be ≈ 0 by construction)
        wvar   = float(((d ** 2) * wt).sum() / tw)
        sigs[AXIS_LABELS[ax]] = round(float(np.sqrt(wvar)), 4)

    vibe_sigmas[bucket] = sigs

    print(f"\n  {bucket}  ({n_sessions} sessions · {n_plays} play observations)")
    for ax in AXES:
        label = AXIS_LABELS[ax]
        sigma = sigs[label]
        bar   = "█" * int(sigma / 0.05)
        print(f"    σ_{label:<8} {sigma:.3f}  {bar}")

    # Per-session breakdown for this bucket
    bucket_sessions = sorted(
        [r for r in session_records if r["bucket"] == bucket],
        key=lambda r: r["n_songs"], reverse=True,
    )
    print(f"\n    Per-session means  ({len(bucket_sessions)} sessions):")
    print(f"    {'session':>8}  {'n':>3}  {'content':>8}  {'melodic':>8}  {'bpm':>6}")
    for r in bucket_sessions[:20]:
        print(f"    {r['session_id']:>8}  {r['n_songs']:>3}  "
              f"{r['content']:>8.3f}  {r['melodic']:>8.3f}  {r['bpm']:>6.3f}")
    if len(bucket_sessions) > 20:
        print(f"    … and {len(bucket_sessions) - 20} more sessions")


# ── How these sigmas compare to current hardcoded values ─────────────────────
print(f"\n{'=' * 72}")
print("COMPARISON: data-derived vs current hardcoded sigmas")
print("=" * 72)
print(f"  Current: σ_max=0.50  σ_min=0.20  (single value, all axes, no context)")
for bucket, sigs in vibe_sigmas.items():
    print(f"  {bucket}:")
    for ax, sigma in sigs.items():
        interpretation = (
            "tight  (you stay consistent on this axis within sessions)"
            if sigma < 0.20 else
            "medium (moderate tolerance — matches current σ_min range)"
            if sigma < 0.35 else
            "loose  (you tolerate variation — could widen beyond current σ_max)"
        )
        print(f"    {ax:<8} σ={sigma:.3f}  →  {interpretation}")


# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "computed_at":      datetime.now(timezone.utc).isoformat(),
    "n_plays":          len(merged),
    "n_sessions":       merged["session_id"].nunique(),
    "playlist_targets": playlist_targets,
    "vibe_sigmas":      vibe_sigmas,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved → {OUTPUT_PATH}")
