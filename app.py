#!/usr/bin/env python3
"""
SmartShuffle — Streamlit frontend (Phase 5)

Run:  streamlit run app.py

Requires:  pip install streamlit pandas
"""
import json
import os
import random
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

DIR                = os.path.dirname(os.path.abspath(__file__))
DB_PATH            = os.path.join(DIR, "data", "smartshuffle.db")
ROLLING_STATE_PATH = os.path.join(DIR, "data", "rolling_queue_state.json")
PY      = sys.executable

st.set_page_config(
    page_title = "SmartShuffle",
    page_icon  = "🎵",
    layout     = "wide",
)

st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
  }
</style>
""", unsafe_allow_html=True)

BUCKETS = ["late_night", "morning", "afternoon"]
DEFAULT_TARGETS = {"late_night": -0.5, "morning": 0.4, "afternoon": 0.2}

# ── Helpers ────────────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def current_bucket() -> str:
    h = datetime.now().hour
    if h >= 21 or h < 6: return "late_night"
    if h < 11:            return "morning"
    return "afternoon"

def get_conn():
    return sqlite3.connect(DB_PATH)

@st.cache_data(ttl=120)
def load_params_json():
    """Read vibe_params.json and learned_params.json for queue-display use."""
    vp, lp = {}, {}
    try:
        with open(os.path.join(DIR, "data", "vibe_params.json")) as f:
            vp = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        with open(os.path.join(DIR, "data", "learned_params.json")) as f:
            lp = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return vp, lp

@st.cache_data
def load_song_stats(_run_id, song_ids: tuple) -> dict:
    """Returns {song_id: {plays_30d, last_played}} for display in song detail."""
    if not song_ids:
        return {}
    conn = get_conn()
    placeholders = ",".join("?" * len(song_ids))
    rows = conn.execute(
        f"SELECT song_id,"
        f"  SUM(CASE WHEN played_at >= datetime('now', '-30 days') THEN 1 ELSE 0 END) AS plays_30d,"
        f"  MAX(played_at) AS last_played"
        f" FROM plays WHERE song_id IN ({placeholders})"
        f" GROUP BY song_id",
        list(song_ids),
    ).fetchall()
    conn.close()
    return {r[0]: {"plays_30d": r[1] or 0, "last_played": r[2]} for r in rows}

def run_cmd(cmd: list) -> tuple[bool, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()

@st.cache_resource(ttl=3300)  # recreate before the 1-hour OAuth token expiry
def get_spotify():
    scope = " ".join([
        "user-read-recently-played", "user-library-read",
        "playlist-read-private", "playlist-modify-private", "user-top-read",
        "user-modify-playback-state", "user-read-playback-state",
    ])
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id     = os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI"),
        scope         = scope,
        cache_path    = os.path.join(DIR, ".spotify_cache"),
    ))

def fetch_devices() -> tuple[list[dict], str | None]:
    """Returns (devices, error_message). error_message is None on success."""
    try:
        sp = get_spotify()
        devices = sp.devices().get("devices", [])
        if not devices:
            # /me/player/devices is unreliable on macOS — fall back to current playback
            pb = sp.current_playback()
            if pb and pb.get("device"):
                devices = [pb["device"]]
        if not devices:
            # Last resort: last device that successfully played via push.py
            conn = get_conn()
            row = conn.execute(
                "SELECT value FROM config WHERE key='last_device_id'"
            ).fetchone()
            name_row = conn.execute(
                "SELECT value FROM config WHERE key='last_device_name'"
            ).fetchone()
            conn.close()
            if row:
                devices = [{"id": row[0],
                            "name": f"{name_row[0] if name_row else row[0]} (last known)",
                            "is_active": False, "type": "Computer"}]
        return devices, None
    except Exception as e:
        return [], str(e)

def _pick_algorithm() -> str:
    return "smartshuffle"

# ── Cached data loaders ───────────────────────────────────────────────────────
# _run_id busts the cache whenever we generate or refresh.

@st.cache_data
def load_playlists(_run_id):
    conn = get_conn()
    rows = conn.execute("""
        SELECT p.playlist_name, p.playlist_id, COUNT(pt.song_id) AS n
        FROM playlists p
        LEFT JOIN playlist_tracks pt ON p.playlist_id = pt.playlist_id
        GROUP BY p.playlist_id ORDER BY n DESC
    """).fetchall()
    conn.close()
    return rows

@st.cache_data(ttl=30)
def load_queue_pair(_run_id, playlist_id, context):
    conn = get_conn()

    # Find the most recent un-guessed session pushed within the last 8 hours.
    # Includes unanalyzed sessions (collect.py has a 2h delay) so you can guess
    # immediately after listening without waiting for the pipeline.
    # Excludes analyzed sessions with 0 plays (confirmed abandoned test pushes).
    # CTE aggregates rolling refills into one session via rolling_session_id.
    # No fallback to old sessions — guessing only makes sense on queues you just heard.
    played = conn.execute("""
        WITH session_agg AS (
            SELECT COALESCE(rolling_session_id, push_id)              AS session_id,
                   algorithm,
                   MAX(push_id)                                        AS last_push_id,
                   MIN(pushed_at)                                      AS first_pushed_at,
                   SUM(COALESCE(songs_played, 0))                      AS analyzed_plays,
                   COUNT(analyzed_at)                                  AS analyzed_count,
                   COUNT(*)                                            AS push_count
            FROM queue_pushes
            GROUP BY algorithm, COALESCE(rolling_session_id, push_id)
        )
        SELECT sa.last_push_id, sa.algorithm, qp.pushed_at,
               q.queue_id, q.songs, q.ab_label, q.generated_at,
               q.playlist_id, q.context,
               COALESCE(TRIM(p.playlist_name), q.playlist_id) AS pl_name
        FROM session_agg sa
        JOIN queue_pushes qp ON qp.push_id = sa.last_push_id
        JOIN queues q         ON qp.queue_id = q.queue_id
        LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
        WHERE (sa.analyzed_plays > 0 OR sa.analyzed_count < sa.push_count)
          AND qp.pushed_at >= datetime('now', '-24 hours')
        ORDER BY sa.last_push_id DESC LIMIT 1
    """).fetchone()

    if not played:
        conn.close()
        return None, None

    last_push_id, played_algorithm, pushed_at, played_qid, played_songs, played_label, \
        generated_at, playlist_id_, context_, pl_name = played

    # For rolling sessions, concatenate songs from every refill in the session
    # (ordered by push_id) so the display shows the full session, not just the last batch.
    session_id_val = conn.execute(
        "SELECT COALESCE(rolling_session_id, push_id) FROM queue_pushes WHERE push_id = ?",
        (last_push_id,)
    ).fetchone()[0]
    all_batches = conn.execute("""
        SELECT q.queue_id, q.songs FROM queue_pushes qp
        JOIN queues q ON qp.queue_id = q.queue_id
        WHERE COALESCE(qp.rolling_session_id, qp.push_id) = ?
          AND qp.algorithm = ?
        ORDER BY qp.pushed_at ASC
    """, (session_id_val, played_algorithm)).fetchall()
    session_queue_ids = [r[0] for r in all_batches]
    combined = []
    for batch_num, (_, batch_json) in enumerate(all_batches, 1):
        for song in json.loads(batch_json):
            song["_batch"] = batch_num
            combined.append(song)
    played_songs = json.dumps(combined)

    # The played row always becomes the "anchor" queue; we find the companion
    # from the same generate run (the other algorithm, within 60 seconds).
    other_algo = "random_baseline" if played_algorithm == "smartshuffle" else "smartshuffle"

    companion = conn.execute("""
        SELECT q.queue_id, q.songs, q.ab_label,
               q.generated_at, q.playlist_id, q.context,
               COALESCE(TRIM(p.playlist_name), q.playlist_id) AS pl_name
        FROM queue_pushes qp
        JOIN queues q ON qp.queue_id = q.queue_id
        LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
        WHERE qp.algorithm = ?
          AND ABS(JULIANDAY(qp.pushed_at) - JULIANDAY(?)) * 86400 < 60
        ORDER BY ABS(JULIANDAY(qp.pushed_at) - JULIANDAY(?)) ASC
        LIMIT 1
    """, (other_algo, pushed_at, pushed_at)).fetchone()

    if not companion:
        # No pushed companion — fall back to nearest generated queue
        companion = conn.execute("""
            SELECT q.queue_id, q.songs, q.ab_label,
                   q.generated_at, q.playlist_id, q.context,
                   COALESCE(TRIM(p.playlist_name), q.playlist_id) AS pl_name
            FROM queues q LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
            WHERE q.algorithm = ?
              AND ABS(JULIANDAY(q.generated_at) -
                      JULIANDAY((SELECT generated_at FROM queues WHERE queue_id = ?))
                  ) * 86400 < 60
            ORDER BY ABS(JULIANDAY(q.generated_at) -
                         JULIANDAY((SELECT generated_at FROM queues WHERE queue_id = ?))
                     ) ASC
            LIMIT 1
        """, (other_algo, played_qid, played_qid)).fetchone()

    conn.close()

    # Return (played_row, companion_row, played_algorithm).
    # played_row always carries the full 7-tuple so the UI has metadata regardless of algorithm.
    # companion_row is the other algorithm's queue from the same generate run (may be None).
    played_row    = (played_qid, played_songs, played_label,
                     generated_at, playlist_id_, context_, pl_name)
    companion_row = companion  # (queue_id, songs, ab_label, ...) or None

    return played_row, played_algorithm

@st.cache_data(ttl=60)
def load_history(_run_id):
    conn = get_conn()
    rows = conn.execute("""
        WITH session_sizes AS (
            -- Sum songs across all batches in a rolling session
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   SUM(json_array_length(q.songs)) AS total_queued
            FROM queue_pushes qp
            JOIN queues q ON qp.queue_id = q.queue_id
            GROUP BY 1
        ),
        session_vibes AS (
            -- Average vibe across every song in every batch of the session
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   ROUND(AVG(CAST(json_extract(je.value, '$.vibe_content') AS REAL)), 3) AS avg_content,
                   ROUND(AVG(CAST(json_extract(je.value, '$.vibe_melodic') AS REAL)), 3) AS avg_melodic,
                   ROUND(AVG(CAST(json_extract(je.value, '$.vibe_bpm')     AS REAL)), 3) AS avg_bpm
            FROM queue_pushes qp
            JOIN queues q ON qp.queue_id = q.queue_id
            JOIN json_each(q.songs) je
            GROUP BY 1
        ),
        play_attrs AS (
            -- Attribute each play to its rolling session via most-recent prior push
            SELECT p.inferred_skip,
                   COALESCE(qp.rolling_session_id, qp.push_id) AS session_id
            FROM plays p
            JOIN queue_pushes qp ON qp.push_id = (
                SELECT qp2.push_id FROM queue_pushes qp2
                WHERE qp2.algorithm || '_queued' = p.play_source
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1
            )
            WHERE p.play_source IN ('smartshuffle_queued', 'random_baseline_queued')
              AND p.inferred_skip IN ('skip', 'partial', 'full')
        ),
        play_counts AS (
            SELECT session_id,
                   COUNT(*) AS plays_n,
                   SUM(CASE WHEN inferred_skip = 'skip' THEN 1 ELSE 0 END) AS hard_skip_n
            FROM play_attrs
            GROUP BY 1
        ),
        session_qs AS (
            -- LIS-inferred queue skips: songs the user rejected before Spotify even logged a play
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS qs_n
            FROM queue_skips qs
            JOIN queue_pushes qp ON qs.push_id = qp.push_id
            WHERE qs.queue_position IS NOT NULL
            GROUP BY 1
        ),
        play_stats AS (
            -- Matches insights combined_rate: (hard_play_skips + queue_skips) / (plays + queue_skips)
            SELECT pc.session_id,
                   pc.plays_n,
                   ROUND((pc.hard_skip_n + COALESCE(sqs.qs_n, 0)) * 1.0
                         / NULLIF(pc.plays_n + COALESCE(sqs.qs_n, 0), 0), 3) AS skip_rate
            FROM play_counts pc
            LEFT JOIN session_qs sqs ON sqs.session_id = pc.session_id
        )
        SELECT qp.push_id, qp.pushed_at, p.playlist_name, q.context,
               qp.algorithm, qp.mode,
               ss.total_queued, ps.plays_n, ps.skip_rate,
               sv.avg_content, sv.avg_melodic, sv.avg_bpm
        FROM queue_pushes qp
        JOIN queues q ON q.queue_id = qp.queue_id
        LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
        LEFT JOIN session_sizes ss ON ss.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
        LEFT JOIN play_stats   ps ON ps.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
        LEFT JOIN session_vibes sv ON sv.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
        WHERE (qp.mode = 'full' OR qp.push_id = qp.rolling_session_id)
        ORDER BY qp.push_id DESC LIMIT 40
    """).fetchall()
    conn.close()
    return rows

@st.cache_data(ttl=120)
def load_insights(_run_id, sel_pl_id):
    params = {}
    try:
        with open(os.path.join(DIR, "data", "learned_params.json")) as f:
            params = json.load(f)
    except FileNotFoundError:
        pass

    vibe_params = {}
    try:
        with open(os.path.join(DIR, "data", "vibe_params.json")) as f:
            vibe_params = json.load(f)
    except FileNotFoundError:
        pass

    conn = get_conn()
    sessions = conn.execute("""
        WITH bucketed AS (
            SELECT p.song_id, p.inferred_skip, p.play_source,
                CASE
                    WHEN CAST(strftime('%H', datetime(p.played_at)) AS INTEGER) >= 21
                      OR CAST(strftime('%H', datetime(p.played_at)) AS INTEGER) < 6
                    THEN 'late_night'
                    WHEN CAST(strftime('%H', datetime(p.played_at)) AS INTEGER) < 11
                    THEN 'morning'
                    ELSE 'afternoon'
                END AS bucket
            FROM plays p
            WHERE p.inferred_skip IN ('skip', 'partial', 'full')
        ),
        skip_stats AS (
            SELECT bucket,
                   COUNT(*) AS n_queued,
                   ROUND(SUM(CASE WHEN inferred_skip = 'skip' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS skip_rate
            FROM bucketed
            WHERE play_source IN ('smartshuffle_queued', 'random_baseline_queued')
            GROUP BY bucket
        ),
        vibe_stats AS (
            SELECT b.bucket,
                   ROUND(AVG(s.vibe_content), 3) AS avg_content,
                   ROUND(AVG(s.vibe_melodic), 3) AS avg_melodic,
                   ROUND(AVG(s.vibe_bpm),     3) AS avg_bpm
            FROM bucketed b
            LEFT JOIN songs s ON b.song_id = s.song_id
            GROUP BY b.bucket
        )
        SELECT ss.bucket, ss.n_queued, ss.skip_rate,
               vs.avg_content, vs.avg_melodic, vs.avg_bpm
        FROM skip_stats ss
        LEFT JOIN vibe_stats vs ON ss.bucket = vs.bucket
        ORDER BY ss.n_queued DESC
    """).fetchall()
    vibe_dists = {}
    for axis in ("vibe_content", "vibe_melodic", "vibe_bpm"):
        vibe_dists[axis] = conn.execute(f"""
            SELECT ROUND({axis}, 1) AS bucket, COUNT(*) AS n
            FROM songs WHERE {axis} IS NOT NULL
            GROUP BY bucket ORDER BY bucket
        """).fetchall()
    ab_stats = conn.execute("""
        WITH play_push AS (
            -- Attribute each play to its push (most recent push of the same algorithm before it)
            SELECT p.id AS play_id, p.play_source, p.inferred_skip,
                   p.duration_ms, p.play_duration_ms,
                   (SELECT qp.push_id FROM queue_pushes qp
                    WHERE qp.algorithm || '_queued' = p.play_source
                      AND qp.pushed_at <= p.played_at
                    ORDER BY qp.pushed_at DESC LIMIT 1) AS push_id
            FROM plays p
            WHERE p.play_source IN ('smartshuffle_queued', 'random_baseline_queued')
              AND p.inferred_skip IN ('skip', 'partial', 'full')
        ),
        session_play_counts AS (
            -- Total plays per session (rolling: all batches combined; full: single push)
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS n
            FROM play_push pp
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            GROUP BY COALESCE(qp.rolling_session_id, qp.push_id)
        ),
        valid_pushes AS (
            -- Rolling sessions: include all batches if the session as a whole has 4+ plays.
            -- Full sessions: the single batch must individually have 4+ plays.
            SELECT qp.push_id,
                   COALESCE(qp.rolling_session_id, qp.push_id) AS session_id
            FROM play_push pp
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            JOIN session_play_counts spc ON spc.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
            WHERE spc.n >= 3
            GROUP BY qp.push_id, COALESCE(qp.rolling_session_id, qp.push_id)
            HAVING qp.mode = 'rolling' OR COUNT(*) >= 3
        ),
        per_push AS (
            SELECT pp.play_source, vp.session_id,
                   COUNT(*)                                                                  AS push_plays_n,
                   SUM(CASE WHEN pp.inferred_skip = 'skip'            THEN 1 ELSE 0 END)    AS skip_n,
                   SUM(CASE WHEN pp.inferred_skip = 'full'            THEN 1 ELSE 0 END)    AS full_n,
                   SUM(CASE WHEN pp.inferred_skip IN ('partial','full') THEN 1 ELSE 0 END)  AS push_engaged_n,
                   AVG(CAST(pp.play_duration_ms AS REAL) / NULLIF(pp.duration_ms,0))        AS avg_comp
            FROM play_push pp
            JOIN valid_pushes vp ON vp.push_id = pp.push_id
            GROUP BY pp.play_source, vp.session_id
        ),
        per_session AS (
            -- Collapse rolling refill pushes into one session row before averaging.
            SELECT play_source, session_id,
                   SUM(push_plays_n)   AS session_plays_n,
                   SUM(skip_n)         AS skip_n,
                   SUM(full_n)         AS full_n,
                   SUM(push_engaged_n) AS session_engaged_n,
                   AVG(avg_comp)       AS avg_comp
            FROM per_push
            GROUP BY play_source, session_id
            HAVING session_plays_n >= 3
        ),
        played AS (
            SELECT play_source,
                   SUM(session_plays_n)                AS n,
                   SUM(skip_n)                         AS skip_n,
                   SUM(full_n)                         AS full_n,
                   AVG(avg_comp)                       AS avg_comp,
                   AVG(CAST(session_engaged_n AS REAL)) AS avg_session_songs,
                   COUNT(*)                            AS session_count
            FROM per_session
            GROUP BY play_source
        ),
        valid_sessions AS (
            SELECT session_id FROM session_play_counts WHERE n >= 3
        ),
        quick_skips AS (
            SELECT qp.algorithm || '_queued'                                    AS play_source,
                   COUNT(*)                                                     AS qs_n,
                   SUM(MAX(0.2, 1.0 - CAST(qs.queue_position AS REAL) / 50))   AS qs_weighted_n
            FROM queue_skips qs
            JOIN queue_pushes qp ON qs.push_id = qp.push_id
            JOIN valid_sessions vs ON vs.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
            WHERE qs.queue_position IS NOT NULL
            GROUP BY qp.algorithm
        )
        SELECT p.play_source,
               p.n                                                                  AS plays_n,
               COALESCE(q.qs_n, 0)                                                  AS qs_n,
               ROUND(p.skip_n * 1.0 / NULLIF(p.n, 0), 3)                           AS skip_rate,
               ROUND((p.skip_n + COALESCE(q.qs_weighted_n, 0)) * 1.0
                     / NULLIF(p.n + COALESCE(q.qs_weighted_n, 0), 0), 3)           AS skip_rate_weighted,
               ROUND(p.full_n * 1.0 / NULLIF(p.n, 0), 3)                           AS full_rate,
               ROUND(p.avg_comp, 3)                                                 AS avg_completion,
               ROUND(p.avg_session_songs, 1)                                        AS avg_session_songs,
               p.session_count
        FROM played p
        LEFT JOIN quick_skips q ON p.play_source = q.play_source
    """).fetchall()
    mode_stats = conn.execute("""
        WITH play_push AS (
            SELECT p.id AS play_id, p.play_source, p.inferred_skip,
                   p.duration_ms, p.play_duration_ms,
                   (SELECT qp.push_id FROM queue_pushes qp
                    WHERE qp.algorithm || '_queued' = p.play_source
                      AND qp.pushed_at <= p.played_at
                    ORDER BY qp.pushed_at DESC LIMIT 1) AS push_id
            FROM plays p
            WHERE p.play_source IN ('smartshuffle_queued', 'random_baseline_queued')
              AND p.inferred_skip IN ('skip', 'partial', 'full')
        ),
        session_play_counts AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   qp.algorithm, qp.mode,
                   COUNT(*) AS n
            FROM play_push pp
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            GROUP BY COALESCE(qp.rolling_session_id, qp.push_id), qp.algorithm, qp.mode
        ),
        valid_pushes AS (
            -- Rolling sessions: include all batches if the session as a whole has 4+ plays.
            -- Full sessions: the single batch must individually have 4+ plays.
            SELECT qp.push_id, qp.algorithm, qp.mode,
                   COALESCE(qp.rolling_session_id, qp.push_id) AS session_id
            FROM play_push pp
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            JOIN session_play_counts spc ON spc.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
                                        AND spc.algorithm = qp.algorithm
                                        AND spc.mode = qp.mode
            WHERE spc.n >= 3
            GROUP BY qp.push_id, qp.algorithm, qp.mode, COALESCE(qp.rolling_session_id, qp.push_id)
            HAVING qp.mode = 'rolling' OR COUNT(*) >= 3
        ),
        per_push AS (
            SELECT vp.session_id, vp.algorithm, vp.mode,
                   COUNT(*)                                                                  AS push_plays_n,
                   SUM(CASE WHEN pp.inferred_skip = 'skip'             THEN 1 ELSE 0 END)   AS skip_n,
                   SUM(CASE WHEN pp.inferred_skip = 'full'             THEN 1 ELSE 0 END)   AS full_n,
                   SUM(CASE WHEN pp.inferred_skip IN ('partial','full') THEN 1 ELSE 0 END)  AS push_engaged_n,
                   AVG(CAST(pp.play_duration_ms AS REAL) / NULLIF(pp.duration_ms,0))        AS avg_comp
            FROM play_push pp
            JOIN valid_pushes vp ON vp.push_id = pp.push_id
            GROUP BY vp.session_id, vp.algorithm, vp.mode
        ),
        per_session AS (
            -- Aggregate all pushes in a rolling session into one row.
            -- Rolling sessions share a rolling_session_id; full-queue sessions have one push each.
            -- Require 3+ total plays so partially-listened sessions don't skew averages.
            SELECT session_id, algorithm, mode,
                   SUM(push_plays_n)   AS session_plays_n,
                   SUM(skip_n)         AS skip_n,
                   SUM(full_n)         AS full_n,
                   SUM(push_engaged_n) AS session_engaged_n,
                   AVG(avg_comp)       AS avg_comp
            FROM per_push
            GROUP BY session_id, algorithm, mode
            HAVING session_plays_n >= 3
        ),
        played AS (
            SELECT algorithm, mode,
                   SUM(session_plays_n)                AS plays_n,
                   SUM(skip_n)                         AS skip_n,
                   SUM(full_n)                         AS full_n,
                   AVG(avg_comp)                       AS avg_comp,
                   AVG(CAST(session_engaged_n AS REAL)) AS avg_session_songs,
                   COUNT(*)                            AS session_count
            FROM per_session
            GROUP BY algorithm, mode
        ),
        valid_sessions AS (
            SELECT session_id, algorithm, mode FROM session_play_counts WHERE n >= 3
        ),
        quick_skips AS (
            -- Rolling queues: each refill is a fresh 10-song generation, so position
            -- weighting (designed for 50-song full queues) doesn't apply — use raw count.
            SELECT qp.algorithm, qp.mode,
                   COUNT(*)                                                   AS qs_n,
                   SUM(CASE WHEN qp.mode = 'full'
                       THEN MAX(0.2, 1.0 - CAST(qs.queue_position AS REAL) / 50)
                       ELSE 1.0 END)                                          AS qs_weighted_n
            FROM queue_skips qs
            JOIN queue_pushes qp ON qs.push_id = qp.push_id
            JOIN valid_sessions vs ON vs.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
                                  AND vs.algorithm = qp.algorithm
                                  AND vs.mode = qp.mode
            WHERE qs.queue_position IS NOT NULL
            GROUP BY qp.algorithm, qp.mode
        )
        SELECT p.algorithm, p.mode, p.plays_n, COALESCE(q.qs_n, 0) AS qs_n,
               ROUND(p.skip_n * 1.0 / NULLIF(p.plays_n, 0), 3)                    AS skip_rate,
               ROUND((p.skip_n + COALESCE(q.qs_weighted_n, 0)) * 1.0
                     / NULLIF(p.plays_n + COALESCE(q.qs_weighted_n, 0), 0), 3)    AS skip_rate_weighted,
               ROUND(p.full_n * 1.0 / NULLIF(p.plays_n, 0), 3)                    AS full_rate,
               ROUND(p.avg_comp, 3)                                                AS avg_completion,
               ROUND(p.avg_session_songs, 1)                                       AS avg_session_songs,
               p.session_count
        FROM played p
        LEFT JOIN quick_skips q ON p.algorithm = q.algorithm AND p.mode = q.mode
        ORDER BY p.algorithm, p.mode
    """).fetchall()

    manual_play_trend = conn.execute("""
        WITH interjections AS (
            SELECT strftime('%Y-W%W', interjected_at) AS week, COUNT(*) AS n
            FROM session_interjections
            WHERE interjected_at >= datetime('now', '-90 days')
            GROUP BY week
        ),
        queued AS (
            SELECT strftime('%Y-W%W', played_at) AS week, COUNT(*) AS n
            FROM plays
            WHERE played_at >= datetime('now', '-90 days')
              AND play_source IN ('smartshuffle_queued', 'random_baseline_queued')
            GROUP BY week
        ),
        all_weeks AS (
            SELECT week FROM interjections
            UNION
            SELECT week FROM queued
        )
        SELECT w.week,
               COALESCE(i.n, 0) AS manual_n,
               COALESCE(q.n, 0) AS queued_n
        FROM all_weeks w
        LEFT JOIN interjections i ON i.week = w.week
        LEFT JOIN queued        q ON q.week = w.week
        ORDER BY w.week
    """).fetchall()

    conn.close()
    return params, vibe_params, sessions, vibe_dists, ab_stats, mode_stats, manual_play_trend

# ── Session state ──────────────────────────────────────────────────────────────

if "run_id"           not in st.session_state: st.session_state.run_id           = 0
if "chosen_algorithm" not in st.session_state: st.session_state.chosen_algorithm = "smartshuffle"

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎵 SmartShuffle")
    st.caption("Personalized Spotify queue orchestrator")
    st.divider()

    playlists = load_playlists(st.session_state.run_id)
    if not playlists:
        st.error("No playlists found. Run collect_data.py first.")
        playlists = [("(none)", None, 0)]

    pl_labels = [f"{name}  ({n})" for name, _, n in playlists]
    pl_idx    = st.selectbox("Playlist", range(len(pl_labels)),
                             format_func=lambda i: pl_labels[i])
    sel_pl_id   = playlists[pl_idx][1]
    sel_pl_name = playlists[pl_idx][0].strip()

    context = st.selectbox("Context", BUCKETS,
                           index=BUCKETS.index(current_bucket()))

    rolling_mode = st.toggle(
        "Rolling queue",
        value=False,
        help="Generates 11 songs at a time; refreshes when 4 remain. "
             "Use when you have Streamlit open. "
             "Off = full 180-song queue pushed once.",
    )

    devices, device_error = fetch_devices()
    if device_error:
        st.warning(f"Spotify error: {device_error}")
        selected_device_id = None
    elif devices:
        dev_labels = [f"{'📱' if d['type'] == 'Smartphone' else '💻' if d['type'] == 'Computer' else '🔊'} {d['name']}" for d in devices]
        active_idx = next((i for i, d in enumerate(devices) if d["is_active"]), 0)
        dev_idx    = st.selectbox("Play on", range(len(dev_labels)),
                                  format_func=lambda i: dev_labels[i],
                                  index=active_idx)
        selected_device_id = devices[dev_idx]["id"]
    else:
        st.caption("No Spotify devices found — Spotify's device API is unreliable on macOS. Play any song in Spotify, then refresh.")
        selected_device_id = None

    if st.button("↻ Refresh devices", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if st.button("▶  Generate & Play", type="primary", use_container_width=True):
        # Sync plays/playlists and recompute scores so the queue reflects
        # the latest listening history and any playlist edits.
        for _label, _cmd in [
            ("Syncing plays & playlists…", [PY, f"{DIR}/src/collect.py"]),
            ("Updating scores…",           [PY, f"{DIR}/src/score.py"]),
            ("Scoring new songs…",         [PY, f"{DIR}/vibes/score_vibes.py",
                                             "--playlist", sel_pl_id]),
        ]:
            with st.spinner(_label):
                run_cmd(_cmd)

        phase34_cmd = [
            PY, f"{DIR}/src/recommend.py",
            "--playlist", sel_pl_id,
            "--context",  context,
        ]
        algorithm = _pick_algorithm()

        if rolling_mode:
            phase34_cmd += ["--count", "11"]
            if algorithm == "random_baseline":
                phase34_cmd += ["--algorithm", "random_baseline"]

        queue_label = f"{sel_pl_name} ({'rolling 11' if rolling_mode else '180 songs'})"
        with st.spinner(f"Generating queue for {queue_label}…"):
            ok, out = run_cmd(phase34_cmd)
        if not ok:
            st.error("Generation failed.")
            st.code(out[:1000])
        else:
            st.session_state.chosen_algorithm = algorithm
            st.session_state.run_id  += 1
            st.session_state.revealed = False
            st.session_state.user_guess = None
            st.cache_data.clear()

            play_cmd = [PY, f"{DIR}/src/push.py", "--algorithm", algorithm]
            if rolling_mode:
                play_cmd += ["--rolling"]
            if selected_device_id:
                play_cmd += ["--device-id", selected_device_id]
            with st.spinner("Pushing to Spotify playlist…"):
                play_ok, play_out = run_cmd(play_cmd)
            if play_ok:
                mode_note = " · rolling mode active" if rolling_mode else ""
                st.success(f"Playing!{mode_note}")
            else:
                st.warning("Queue generated but playback failed — is Spotify open?")
                st.code(play_out[:400])
            st.rerun()

    st.divider()

    with st.expander("Refresh data pipeline"):
        st.caption("collect → phase2 → model")
        if st.button("Run refresh", use_container_width=True):
            for label, cmd in [
                ("Collecting plays…",  [PY, f"{DIR}/src/collect.py"]),
                ("Modeling…",          [PY, f"{DIR}/src/score.py"]),
                ("Training model…",    [PY, f"{DIR}/src/train.py"]),
            ]:
                with st.spinner(label):
                    ok, out = run_cmd(cmd)
                if not ok:
                    st.error(f"Failed at: {label}")
                    st.code(out[:500])
                    break
            else:
                st.session_state.run_id += 1
                st.success("Done!")
                st.cache_data.clear()
                st.rerun()

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _describe_axis(axis: str, v: float) -> str:
    if axis == "Content":
        if v < -0.6: return "introspective"
        if v < -0.2: return "reflective"
        if v <  0.2: return "balanced"
        if v <  0.6: return "assertive"
        return "aggressive"
    if axis == "Melodic":
        if v < -0.6: return "fully sung"
        if v < -0.2: return "mostly melodic"
        if v <  0.2: return "mixed"
        if v <  0.6: return "rap-led"
        return "pure rap"
    if axis == "BPM":
        if v < -0.6: return "slow"
        if v < -0.2: return "mid-tempo"
        if v <  0.2: return "moderate"
        if v <  0.6: return "uptempo"
        return "fast"
    return ""


def _fmt_pushed_at(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        h12 = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        return f"{dt.day} {dt.strftime('%b')} · {h12}:{dt.strftime('%M')} {ampm}"
    except Exception:
        return ts[:16]


_AXIS_ENDS = {
    "Content": ("Introspective", "Aggressive"),
    "Melodic":  ("Melodic",       "Rap"),
    "BPM":      ("Slow",          "Fast"),
}


def _axis_bar_html(axis_name: str, v: float, desc: str,
                   g: float | None = None,
                   color: str = "rgba(99,102,241,0.65)") -> str:
    pct    = max(2.0, min(98.0, (v + 1.0) / 2.0 * 100))
    lo, hi = _AXIS_ENDS.get(axis_name, ("−1", "+1"))
    fit_span = (
        f'<span style="color:#888;white-space:nowrap;min-width:48px;text-align:right">'
        f'{round(g * 100)}% fit</span>'
        if g is not None and math.isfinite(g) else
        '<span style="min-width:48px"></span>'
    )
    return (
        f'<div style="display:flex;align-items:center;gap:12px;margin:10px 0;font-size:0.82em">'
        f'<span style="min-width:52px;color:#888">{axis_name}</span>'
        f'<div style="flex:1;min-width:80px">'
        f'<div style="position:relative;height:16px">'
        f'<span style="position:absolute;left:{pct:.1f}%;transform:translateX(-50%);'
        f'white-space:nowrap;font-size:0.88em;font-weight:500">{desc[0].upper() + desc[1:] if desc and not desc.startswith("<") else desc}</span>'
        f'</div>'
        f'<div style="background:rgba(128,128,128,0.18);border-radius:3px;height:5px">'
        f'<div style="width:{pct:.1f}%;background:{color};border-radius:3px;height:5px"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:3px;color:#aaa;font-size:0.78em">'
        f'<span>{lo}</span><span>{hi}</span></div>'
        f'</div>'
        f'{fit_span}'
        f'</div>'
    )


# ── Main content ───────────────────────────────────────────────────────────────

tab_queue, tab_history, tab_insights = st.tabs(["Queue", "History", "Insights"])

# ══ Queue tab ══════════════════════════════════════════════════════════════════

with tab_queue:
    # Check if a rolling session is active so we can show live state
    _rolling_active = False
    try:
        with open(ROLLING_STATE_PATH) as _rf:
            _rs = json.load(_rf)
        _rolling_active = bool(_rs.get("enabled"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    played_row, played_algorithm = load_queue_pair(st.session_state.run_id, sel_pl_id, context)

    if played_algorithm is None:
        st.info("No recent queue — generate a new queue or wait until you've listened to one in the last 8 hours.")
    else:
        played_id, played_songs_json, _, generated_at, _, _q_ctx, _q_pl_name = played_row
        is_ss         = (played_algorithm or st.session_state.chosen_algorithm) == "smartshuffle"
        display_songs = json.loads(played_songs_json)
        algo_label    = "SmartShuffle ✦" if is_ss else "Random baseline"

        _cap_col, _btn_col = st.columns([5, 1])
        _cap_col.caption(
            f"Generated {generated_at[:16]}  ·  **{_q_pl_name}**  ·  context: **{_q_ctx}**  ·  **{algo_label}**"
            + ("  ·  ● rolling" if _rolling_active else "")
        )
        if _rolling_active and _btn_col.button("↻ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # ── Song list ──────────────────────────────────────────────────────────

        import math

        vibe_params_json, learned_params = load_params_json()
        song_ids = tuple(s["song_id"] for s in display_songs)
        song_stats = load_song_stats(st.session_state.run_id, song_ids)

        # Vibe targets for this playlist
        playlist_targets = vibe_params_json.get("playlist_targets", {})
        target_vibe = playlist_targets.get(sel_pl_id)
        baseline_vibe = dict(target_vibe) if target_vibe else None  # pre-drift copy for delta display

        # Apply session vibe drift — mirrors recommend.py's _apply_session_vibe_drift.
        # Dead zone < 5 completions; linear ramp to 80% over 40 more completions.
        # Above 30 completions, last-10 songs carry 70% of the session component.
        _SV_MIN   = 5     # noise floor
        _SV_RAMP  = 23    # completions past floor to reach max (5+23=28 → 80%)
        _SV_MAX   = 0.80
        _SV_REC_AT = 30   # start recency-blending above this
        _SV_REC_W  = 0.70 # recent-10 weight within session component
        _session_drift_n = 0
        _session_drift_w = 0.0
        if target_vibe is not None:
            try:
                with open(os.path.join(DIR, "data", "session_state.json")) as _sf:
                    _ss = json.load(_sf)
                _n = int(_ss.get("session_vibe_n", 0))
                if _n >= _SV_MIN:
                    _dw = min((_n - _SV_MIN) / _SV_RAMP, 1.0) * _SV_MAX
                    _blended: dict | None = {}
                    for _ax in ("content", "melodic", "bpm"):
                        _sv = _ss.get(f"session_vibe_{_ax}_mean")
                        if _sv is None:
                            _blended = None
                            break
                        _eff = float(_sv)
                        if _n >= _SV_REC_AT:
                            _rv = _ss.get(f"session_vibe_recent_{_ax}")
                            if _rv is not None:
                                _eff = _SV_REC_W * float(_rv) + (1.0 - _SV_REC_W) * float(_sv)
                        _blended[_ax] = round((1.0 - _dw) * target_vibe[_ax] + _dw * _eff, 4)
                    if _blended:
                        target_vibe      = _blended
                        _session_drift_n = _n
                        _session_drift_w = _dw
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
                pass

        # Sigmas — use context bucket; fall back to defaults
        _default_sigmas = {"content": 0.50, "melodic": 0.50, "bpm": 0.50}
        vibe_sigmas = (
            vibe_params_json.get("vibe_sigmas", {}).get(_q_ctx, _default_sigmas)
            or _default_sigmas
        )

        # Weights for score breakdown — prefer learned, fall back to defaults
        _default_weights = {
            "vibe_match": 0.40, "fatigue": 0.25, "artist_fatigue": 0.10,
            "coverage": 0.10, "binge_boost": 0.15, "skip": 0.30, "recency": 0.30,
        }
        weights = {**_default_weights, **(learned_params.get("weights", {}))}

        def _fatigue_note(f: float) -> str:
            if f < 0.15: return "Not played recently — no penalty"
            if f < 0.40: return "Played in the past few days — mild penalty"
            if f < 0.65: return "Played a lot lately — strong penalty"
            return "Overplayed recently — heavy penalty"

        def _debt_note(d: float) -> str:
            if d < 0.5: return "Just surfaced recently — no boost"
            if d < 1.5: return "Hasn't come up in a while — small boost"
            if d < 3.0: return "Rarely surfaced lately — solid boost"
            return "Long overdue — max boost"

        def _last_played_str(ts: str | None) -> str:
            if not ts:
                return "Never"
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                days = (datetime.now(timezone.utc) - dt).days
                if days == 0: return "Today"
                if days == 1: return "Yesterday"
                return f"{days} days ago"
            except Exception:
                return ts[:10]

        # ── Session target ────────────────────────────────────────────────────
        if is_ss and target_vibe is not None:
            if _session_drift_n >= _SV_MIN:
                _drift_pct = round(_session_drift_w * 100)
                st.markdown(
                    f"**Session Target**"
                    f"<span style='color:#888;font-size:0.82em;margin-left:8px'>"
                    f"adapting · {_session_drift_n} plays ({_drift_pct}% session weight)</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("**Session Target**")
            target_bars = []
            for axis_name, tgt_key in [("Content", "content"), ("Melodic", "melodic"), ("BPM", "bpm")]:
                v_tgt = target_vibe.get(tgt_key)
                if v_tgt is not None:
                    desc = _describe_axis(axis_name, v_tgt)
                    if baseline_vibe and _session_drift_n >= _SV_MIN:
                        _delta = v_tgt - baseline_vibe.get(tgt_key, v_tgt)
                        if abs(_delta) >= 0.02:
                            _dcol = "#10b981" if _delta > 0 else "#f59e0b"
                            _dsign = "+" if _delta > 0 else ""
                            desc = (
                                f'{desc}<span style="color:{_dcol};margin-left:6px;'
                                f'font-size:0.82em;font-weight:normal;opacity:0.85">'
                                f'{_dsign}{_delta:.2f}</span>'
                            )
                    target_bars.append(_axis_bar_html(
                        axis_name, v_tgt, desc,
                        color="rgba(16,185,129,0.65)",
                    ))
            if target_bars:
                st.markdown("\n".join(target_bars), unsafe_allow_html=True)
            st.divider()

        current_batch = 0
        for i, song in enumerate(display_songs, 1):
            batch = song.get("_batch", 1)
            if batch != current_batch:
                current_batch = batch
                if batch > 1:
                    st.markdown(
                        f"<div style='text-align:center;color:gray;font-size:0.8em;"
                        f"margin:12px 0 4px'>─── Refill {batch} ───</div>",
                        unsafe_allow_html=True,
                    )

            vc      = song.get("vibe_content")
            vm      = song.get("vibe_melodic")
            vb      = song.get("vibe_bpm")
            cluster = (song.get("cluster_label") or "Unclassified").title()

            stats       = song_stats.get(song["song_id"], {})
            plays_30d   = stats.get("plays_30d", 0)
            last_played = _last_played_str(stats.get("last_played"))

            with st.expander(f"**{i}.** {song['song_name']} — {song['artist_name']}"):
                if is_ss and song.get("score") is not None:
                    st.markdown("**Why This Song?**")

                    # ── Vibe match ────────────────────────────────────────────
                    if target_vibe is not None and vc is not None:
                        axes = [
                            ("Content", vc, target_vibe.get("content"), vibe_sigmas.get("content", 0.5)),
                            ("Melodic", vm, target_vibe.get("melodic"), vibe_sigmas.get("melodic", 0.5)),
                            ("BPM",     vb, target_vibe.get("bpm"),     vibe_sigmas.get("bpm",     0.5)),
                        ]
                        axis_gs  = []
                        bars_html = []
                        for axis_name, v_song, v_tgt, sigma in axes:
                            if v_song is None or v_tgt is None or not math.isfinite(float(v_song)):
                                continue
                            g = math.exp(-0.5 * (v_song - v_tgt) ** 2 / sigma ** 2)
                            axis_gs.append(g)
                            bars_html.append(_axis_bar_html(
                                axis_name, v_song,
                                _describe_axis(axis_name, v_song),
                                g=g,
                            ))
                        if bars_html:
                            st.markdown("\n".join(bars_html), unsafe_allow_html=True)
                        if axis_gs:
                            vibe_match   = sum(axis_gs) / len(axis_gs)
                            vibe_contrib = weights.get("vibe_match", 0.40) * vibe_match
                            st.caption(f"Overall vibe fit: **{vibe_match*100:.0f}%** → **{vibe_contrib:+.3f}** score contribution")
                    elif vc is not None:
                        st.caption("Vibe targets not set for this playlist.")

                    st.divider()

                    # ── Score components ──────────────────────────────────────
                    fatigue        = song.get("fatigue", 0.0)
                    binge          = song.get("binge_score", 0.0)
                    debt           = song.get("coverage_debt", 0.0)
                    coverage_bonus = min(debt / 4.0, 1.0)
                    fat_contrib    = -weights.get("fatigue",     0.25) * fatigue
                    cov_contrib    =  weights.get("coverage",    0.10) * coverage_bonus
                    bng_contrib    =  weights.get("binge_boost", 0.15) * binge

                    rows_md = [
                        f"| Play Fatigue  | `{fat_contrib:+.3f}` | {_fatigue_note(fatigue)} |",
                        f"| Queue Recency | `{cov_contrib:+.3f}` | {_debt_note(debt)} |",
                    ]
                    if binge > 0.05:
                        rows_md.append(f"| Binge Boost   | `{bng_contrib:+.3f}` | Active binge — fatigue penalty reduced |")
                    rows_md.append(f"| **Total Score** | **`{song['score']:+.4f}`** | |")

                    st.markdown(
                        "| Component | Score | Note |\n"
                        "|---|---|---|\n" +
                        "\n".join(rows_md)
                    )

                st.divider()

                # ── Play history ──────────────────────────────────────────────
                col_p, col_l = st.columns(2)
                col_p.metric("Plays (30 days)", plays_30d)
                col_l.metric("Last Played", last_played)

# ══ History tab ════════════════════════════════════════════════════════════════

with tab_history:
    history = load_history(st.session_state.run_id)

    if not history:
        st.info("No queues generated yet.")
    else:
        shown = [r for r in history if (r[7] or 0) >= 4]
        if not shown:
            st.info("No sessions with 4+ attributed plays yet.")
        for push_id, pushed_at, pl, ctx, algo, mode, total_queued, plays_n, skip_rate, avg_c, avg_m, avg_b in shown:
            algo_label  = "SmartShuffle" if algo == "smartshuffle" else "RB"
            pl_label    = (pl or "?").strip()
            is_rolling  = (mode == "rolling")
            n_queued    = total_queued or "?"
            skip_str    = f"{skip_rate:.0%}" if skip_rate is not None else "—"
            mode_tag    = " · rolling" if is_rolling else ""

            header = (
                f"{_fmt_pushed_at(pushed_at)}  ·  {pl_label}  ·  {algo_label}{mode_tag}"
                f"  ·  {n_queued} queued  ·  {plays_n} played  ·  {skip_str} skip"
            )
            with st.expander(header):
                c1, c2, c3 = st.columns(3)
                c1.metric("Played", plays_n)
                c2.metric("Skip rate", skip_str)
                c3.metric("Queued", n_queued)

                if avg_c is not None:
                    st.markdown("**Avg session vibe**")
                    bars = []
                    for axis, val in [("Content", avg_c), ("Melodic", avg_m), ("BPM", avg_b)]:
                        if val is not None:
                            bars.append(_axis_bar_html(axis, val, _describe_axis(axis, val)))
                    if bars:
                        st.markdown("\n".join(bars), unsafe_allow_html=True)

                st.caption(f"Context: {ctx}")

        rb_count = sum(1 for r in shown if r[4] == "random_baseline")
        ss_count = sum(1 for r in shown if r[4] == "smartshuffle")
        st.caption(f"Showing {len(shown)} sessions with 4+ plays: {ss_count} SmartShuffle · {rb_count} Random baseline")

# ══ Insights tab ═══════════════════════════════════════════════════════════════

with tab_insights:
    params, vibe_params, sessions, vibe_dists, ab_stats, mode_stats, manual_play_trend = load_insights(st.session_state.run_id, sel_pl_id)

    # ── A/B skip rate comparison ──────────────────────────────────────────────
    st.subheader("A/B skip rates")
    total_attributed = sum(r[1] for r in ab_stats) if ab_stats else 0
    if total_attributed >= 20:
        ab_map = {r[0]: r for r in ab_stats}
        c_ss, c_base = st.columns(2)
        for col, source, label in [
            (c_ss,   "smartshuffle_queued", "SmartShuffle"),
            (c_base, "random_baseline_queued", "Random baseline"),
        ]:
            row = ab_map.get(source)
            with col:
                st.markdown(f"**{label}**")
                if row:
                    _, plays_n, qs_n, skip, skip_w, full, avg_comp, avg_session_songs, session_count = row
                    total_n = plays_n + qs_n
                    total_skip_n = (round(skip * plays_n) if skip is not None else 0) + qs_n
                    combined_rate = total_skip_n / total_n if total_n else None
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Skip rate", f"{combined_rate:.1%}" if combined_rate is not None else "—")
                    m2.metric("Avg songs / session", f"{avg_session_songs:.1f}" if avg_session_songs else "—")
                    m3.metric("Sessions", session_count)
                    skip_w_str = f"**{skip_w:.1%}**" if skip_w is not None else "—"
                    st.caption(f"Plays: {plays_n}  ·  Queue skips: {qs_n}  ·  Pos-weighted: {skip_w_str}")
                else:
                    st.caption("No attributed plays yet.")
    else:
        needed = 20 - total_attributed
        st.caption(
            f"Need {needed} more attributed plays to show skip-rate comparison. "
            "Keep listening — plays within 2 hours of pressing Play are tracked automatically."
        )

    st.divider()

    # ── Rolling vs full breakdown (both algorithms) ───────────────────────────
    st.subheader("Rolling vs full queue")
    if mode_stats:
        algo_mode = {(r[0], r[1]): r for r in mode_stats}

        for algo, algo_label in [("smartshuffle", "SmartShuffle"), ("random_baseline", "Random baseline")]:
            has_data = any(k[0] == algo for k in algo_mode)
            if not has_data:
                continue
            st.markdown(f"**{algo_label}**")
            col_rolling, col_full = st.columns(2)
            for col, mode in [(col_rolling, "rolling"), (col_full, "full")]:
                with col:
                    st.caption(f"*{mode.capitalize()} queue*")
                    row = algo_mode.get((algo, mode))
                    if row:
                        _, _, plays_n, qs_n, skip_rate, skip_w, full, avg_comp, avg_session_songs, session_count = row
                        total_n       = plays_n + qs_n
                        skip_n_est    = round((skip_rate or 0) * plays_n) if skip_rate else 0
                        combined_rate = (skip_n_est + qs_n) / total_n if total_n else None
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Skip rate", f"{combined_rate:.1%}" if combined_rate is not None else "—")
                        m2.metric("Avg songs / session", f"{avg_session_songs:.1f}" if avg_session_songs else "—")
                        m3.metric("Sessions", session_count)
                        skip_w_str = f"**{skip_w:.1%}**" if skip_w is not None else "—"
                        st.caption(f"Plays: {plays_n}  ·  Queue skips: {qs_n}  ·  Pos-weighted: {skip_w_str}")
                    else:
                        st.caption("No data yet.")
    else:
        st.caption("No data yet.")

    st.divider()

    # ── Manual play % trend ───────────────────────────────────────────────────
    st.subheader("Manual play %")
    st.caption("Spike = suppression may be too aggressive; user is searching for songs SmartShuffle isn't surfacing.")
    if manual_play_trend:
        _trend_rows = []
        for week, manual_n, queued_n in manual_play_trend:
            total = manual_n + queued_n
            _trend_rows.append({
                "Week":       week,
                "Manual %":  round(manual_n / total * 100, 1) if total else 0.0,
                "Manual":    manual_n,
                "Queued":    queued_n,
            })
        _trend_df = pd.DataFrame(_trend_rows).set_index("Week")

        # Summary metric: last 2 weeks vs prior 4 weeks
        _recent = _trend_df.tail(2)
        _prior  = _trend_df.iloc[-6:-2] if len(_trend_df) >= 6 else _trend_df.iloc[:-2]
        _recent_pct = (_recent["Manual"].sum() / (_recent["Manual"].sum() + _recent["Queued"].sum()) * 100
                       if (_recent["Manual"].sum() + _recent["Queued"].sum()) > 0 else None)
        _prior_pct  = (_prior["Manual"].sum()  / (_prior["Manual"].sum()  + _prior["Queued"].sum())  * 100
                       if (_prior["Manual"].sum()  + _prior["Queued"].sum())  > 0 else None)
        _delta = round(_recent_pct - _prior_pct, 1) if (_recent_pct is not None and _prior_pct is not None) else None

        _m1, _m2 = st.columns([1, 3])
        with _m1:
            st.metric(
                "Last 2 weeks",
                f"{_recent_pct:.1f}%" if _recent_pct is not None else "—",
                delta=f"{_delta:+.1f}pp vs prior 4 wks" if _delta is not None else None,
                delta_color="inverse",
            )
        with _m2:
            st.line_chart(_trend_df[["Manual %"]], height=120)
    else:
        st.caption("No play data yet.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Learned vibe targets")
        playlist_targets = vibe_params.get("playlist_targets", {})
        t = playlist_targets.get(sel_pl_id, {})
        if t:
            st.dataframe(pd.DataFrame([{
                "Content": round(t.get("content", 0.0), 3),
                "Melodic": round(t.get("melodic", 0.0), 3),
                "BPM":     round(t.get("bpm",     0.0), 3),
            }]), hide_index=True, use_container_width=True)
        else:
            st.caption("No vibe target learned for this playlist yet.")

        sigmas = vibe_params.get("vibe_sigmas", {})
        if sigmas:
            sigma_rows = []
            for b in BUCKETS:
                sv = sigmas.get(b, {})
                sigma_rows.append({
                    "Bucket":    b,
                    "σ content": round(sv.get("content", 0.0), 3),
                    "σ melodic": round(sv.get("melodic", 0.0), 3),
                    "σ bpm":     round(sv.get("bpm",     0.0), 3),
                })
            st.caption("Per-bucket tolerance (σ)")
            st.dataframe(pd.DataFrame(sigma_rows), hide_index=True, use_container_width=True)

        st.subheader("Scoring weights")
        weights = params.get("weights", {})
        if weights:
            st.bar_chart(pd.DataFrame(weights, index=["weight"]).T)
        else:
            st.caption("No learned weights yet — using defaults.")

        alpha   = params.get("alpha", 1.0)
        n_plays = params.get("n_plays", 0)
        m1, m2 = st.columns(2)
        m1.metric("Training plays", n_plays)
        m2.metric("Formula weight (α)", f"{alpha:.2f}",
                  help="1.0 = pure formula, 0.3 = ML dominant")

    with col2:
        st.subheader("Session breakdown")
        if sessions:
            sess_df = pd.DataFrame(
                sessions,
                columns=["Bucket", "Queued plays", "Skip rate (queued)",
                         "Avg content", "Avg melodic", "Avg bpm"]
            )
            st.dataframe(sess_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No session data yet.")

        st.subheader("Vibe distribution (all scored songs)")
        axis_labels = {
            "vibe_content": "Content (philosophical ← → aggressive)",
            "vibe_melodic": "Melodic (sung ← → rap)",
            "vibe_bpm":     "BPM (chill ← → hype)",
        }
        for axis, label in axis_labels.items():
            rows = vibe_dists.get(axis, [])
            if rows:
                st.caption(label)
                df_v = pd.DataFrame(rows, columns=["Score", "Songs"])
                st.bar_chart(df_v.set_index("Score"))
            else:
                st.caption(f"{label}: no data yet.")
