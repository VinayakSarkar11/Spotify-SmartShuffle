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
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue_guesses (
            guess_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id   INTEGER NOT NULL UNIQUE,
            actual     TEXT NOT NULL,
            guess      TEXT NOT NULL,
            correct    INTEGER NOT NULL,
            guessed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

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
        return devices, None
    except Exception as e:
        return [], str(e)

def _pick_algorithm() -> str:
    """
    Returns 'smartshuffle' or 'random_baseline'.
    Enforces a 60/40 split by push count over the last 10 pushes.
    Forces baseline whenever baseline fraction in that window falls below 35%.
    Push-count window (not play-count) so test/debug pushes register immediately
    and long same-algorithm streaks are impossible.
    """
    WINDOW       = 6
    MIN_BASELINE = 0.40
    MIN_PLAYS    = 4   # sessions with fewer total attributed plays carry no signal
    try:
        conn = get_conn()
        # Count by session, not individual push. Rolling refills in the same session
        # share a rolling_session_id; full-mode pushes use push_id as their own session.
        rows = conn.execute("""
            SELECT algorithm,
                   COALESCE(rolling_session_id, push_id) AS session_id,
                   SUM(COALESCE(songs_played, 0))        AS session_plays,
                   MAX(push_id)                          AS last_push_id
            FROM queue_pushes
            WHERE analyzed_at IS NOT NULL
            GROUP BY algorithm, session_id
            HAVING session_plays >= ?
            ORDER BY last_push_id DESC LIMIT ?
        """, (MIN_PLAYS, WINDOW)).fetchall()
        conn.close()
        if len(rows) >= WINDOW:
            baseline_n = sum(1 for r in rows if r[0] == "random_baseline")
            if baseline_n / WINDOW < MIN_BASELINE:
                return "random_baseline"
    except Exception:
        pass
    return "smartshuffle" if random.random() < 0.60 else "random_baseline"

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

@st.cache_data
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
                   COUNT(analyzed_at)                                  AS analyzed_count
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
        WHERE (sa.analyzed_plays > 0 OR sa.analyzed_count = 0)
          AND q.queue_id NOT IN (SELECT queue_id FROM queue_guesses)
          AND sa.first_pushed_at >= datetime('now', '-8 hours')
        ORDER BY sa.last_push_id DESC LIMIT 1
    """).fetchone()

    if not played:
        conn.close()
        return None, None, None, None

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
        ORDER BY qp.push_id ASC
    """, (session_id_val, played_algorithm)).fetchall()
    session_queue_ids = [r[0] for r in all_batches]
    if len(all_batches) > 1:
        combined = []
        for (_, batch_json) in all_batches:
            combined.extend(json.loads(batch_json))
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

    return played_row, companion_row, played_algorithm, session_queue_ids

@st.cache_data
def load_history(_run_id):
    conn = get_conn()
    # Show all session-starting pushes for both algorithms.
    # A push is a session start if: mode='full', OR it's the initial rolling push
    # (rolling_session_id = push_id, i.e. self-referencing).
    rows = conn.execute("""
        SELECT q.queue_id, qp.pushed_at, p.playlist_name, q.context, qp.algorithm, qp.mode,
               json_array_length(q.songs) AS n_songs
        FROM queue_pushes qp
        JOIN queues q ON q.queue_id = qp.queue_id
        LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
        WHERE (qp.mode = 'full' OR qp.push_id = qp.rolling_session_id)
        ORDER BY qp.push_id DESC LIMIT 40
    """).fetchall()
    conn.close()
    return rows

@st.cache_data
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
        valid_pushes AS (
            -- 3+ attributed plays required so tiny/abandoned pushes don't count
            SELECT pp.push_id,
                   COALESCE(qp.rolling_session_id, pp.push_id) AS session_id
            FROM play_push pp
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            GROUP BY pp.push_id HAVING COUNT(*) >= 4
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
        quick_skips AS (
            SELECT qp.algorithm || '_queued'                                    AS play_source,
                   COUNT(*)                                                     AS qs_n,
                   SUM(MAX(0.2, 1.0 - CAST(qs.queue_position AS REAL) / 50))   AS qs_weighted_n
            FROM queue_skips qs
            JOIN queue_pushes qp ON qs.push_id = qp.push_id
            JOIN valid_pushes vp ON vp.push_id = qp.push_id
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
        valid_pushes AS (
            -- 3+ attributed plays required so tiny/abandoned pushes don't count
            SELECT pp.push_id, qp.algorithm, qp.mode,
                   COALESCE(qp.rolling_session_id, qp.push_id) AS session_id
            FROM play_push pp
            JOIN queue_pushes qp ON qp.push_id = pp.push_id
            GROUP BY pp.push_id HAVING COUNT(*) >= 4
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
            JOIN valid_pushes vp ON vp.push_id = qp.push_id
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

    conn.close()
    return params, vibe_params, sessions, vibe_dists, ab_stats, mode_stats

# ── Session state ──────────────────────────────────────────────────────────────

if "run_id"              not in st.session_state: st.session_state.run_id              = 0
if "revealed"            not in st.session_state: st.session_state.revealed            = False
if "current_queue_id"    not in st.session_state: st.session_state.current_queue_id    = None
if "session_queue_ids"   not in st.session_state: st.session_state.session_queue_ids   = []
if "chosen_algorithm"    not in st.session_state: st.session_state.chosen_algorithm    = "smartshuffle"
if "user_guess"          not in st.session_state: st.session_state.user_guess          = None

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
        st.caption("No Spotify devices found. Open Spotify and play a song first, then refresh.")
        if st.button("↻ Refresh devices", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        selected_device_id = None

    if st.button("▶  Generate & Play", type="primary", use_container_width=True):
        # Sync plays/playlists and recompute scores so the queue reflects
        # the latest listening history and any playlist edits.
        for _label, _cmd in [
            ("Syncing plays & playlists…", [PY, f"{DIR}/src/collect.py"]),
            ("Updating scores…",           [PY, f"{DIR}/src/score.py"]),
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

# ── Main content ───────────────────────────────────────────────────────────────

tab_queue, tab_history, tab_insights = st.tabs(["Queue", "History", "Insights"])

# ══ Queue tab ══════════════════════════════════════════════════════════════════

with tab_queue:
    played_row, companion_row, played_algorithm, session_queue_ids = load_queue_pair(st.session_state.run_id, sel_pl_id, context)

    if played_algorithm is None:
        st.info("No recent queue to guess on — generate a new queue or wait until you've listened to one in the last 8 hours.")
    else:
        played_id, played_songs_json, _, generated_at, _, _q_ctx, _q_pl_name = played_row

        # Always show the played queue — the one that was actually in Spotify.
        # companion_row is the unpushed partner from the same generate run (may be None).
        is_ss = (played_algorithm or st.session_state.chosen_algorithm) == "smartshuffle"
        display_id    = played_id
        display_songs = json.loads(played_songs_json)

        # Reset guesser state when the session has advanced to a new queue
        if display_id != st.session_state.current_queue_id:
            st.session_state.revealed        = False
            st.session_state.user_guess      = None
        st.session_state.current_queue_id  = display_id
        st.session_state.session_queue_ids = session_queue_ids

        st.caption(
            f"Generated {generated_at[:16]}  ·  **{_q_pl_name}**  ·  context: **{_q_ctx}**"
        )

        # ── Guess / Reveal ─────────────────────────────────────────────────────

        algo_label  = "SmartShuffle ✦" if is_ss else "Random baseline"
        actual_algo = played_algorithm or st.session_state.chosen_algorithm
        revealed    = st.session_state.revealed

        if not revealed:
            st.caption("Which algorithm is this?")
            g1, g2, g3 = st.columns([2, 2, 3])
            def _submit_guess(guess):
                correct  = 1 if guess == actual_algo else 0
                ts       = now_iso()
                qids     = st.session_state.session_queue_ids or [st.session_state.current_queue_id]
                conn     = get_conn()
                conn.executemany(
                    "INSERT OR IGNORE INTO queue_guesses"
                    " (queue_id, actual, guess, correct, guessed_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [(qid, actual_algo, guess, correct, ts) for qid in qids],
                )
                conn.commit(); conn.close()
                st.session_state.user_guess = guess
                st.session_state.revealed   = True

            if g1.button("SmartShuffle ✦", use_container_width=True):
                _submit_guess("smartshuffle"); st.rerun()
            if g2.button("Random baseline", use_container_width=True):
                _submit_guess("random_baseline"); st.rerun()
            if g3.button("Skip (just reveal)", use_container_width=True):
                st.session_state.revealed = True
                st.rerun()
        else:
            guess = st.session_state.user_guess
            if guess is None:
                st.success(f"This queue was generated by **{algo_label}**")
            elif guess == actual_algo:
                st.success(f"Correct! It was **{algo_label}**")
            else:
                wrong_label = "SmartShuffle ✦" if guess == "smartshuffle" else "Random baseline"
                st.error(f"Wrong — you guessed **{wrong_label}**, it was **{algo_label}**")

        st.divider()

        # ── Song list ──────────────────────────────────────────────────────────

        for i, song in enumerate(display_songs, 1):
            vc      = song.get("vibe_content")
            vm      = song.get("vibe_melodic")
            vb      = song.get("vibe_bpm")
            cluster = song.get("cluster_label", "")
            with st.container(border=True):
                st.markdown(f"**{i}. {song['song_name']}**  \n{song['artist_name']}")
                detail = cluster if cluster else "unclassified"
                if vc is not None:
                    detail += f"  ·  content {vc:+.2f}  melodic {vm:+.2f}  bpm {vb:+.2f}"
                if is_ss and revealed and song.get("score") is not None:
                    detail += (f"  ·  score {song['score']:+.4f}"
                               f"  ·  fatigue {song.get('fatigue', 0):.2f}"
                               f"  ·  debt {song.get('coverage_debt', 0):.2f}")
                st.caption(detail)

# ══ History tab ════════════════════════════════════════════════════════════════

with tab_history:
    history = load_history(st.session_state.run_id)

    if not history:
        st.info("No queues generated yet.")
    else:
        rows = []
        for q_id, pushed_at, pl, ctx, algo, mode, n_songs in history:
            rows.append({
                "Pushed":    pushed_at[:16],
                "Algorithm": "SmartShuffle" if algo == "smartshuffle" else "Random baseline",
                "Mode":      mode,
                "Songs":     n_songs,
                "Playlist":  (pl or "?").strip(),
                "Context":   ctx,
                "Queue ID":  q_id,
            })
        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, hide_index=True, use_container_width=True)

        rb_count = sum(1 for r in history if r[4] == "random_baseline")
        ss_count = sum(1 for r in history if r[4] == "smartshuffle")
        st.caption(f"Last {len(history)} sessions: {ss_count} SmartShuffle · {rb_count} Random baseline")

# ══ Insights tab ═══════════════════════════════════════════════════════════════

with tab_insights:
    params, vibe_params, sessions, vibe_dists, ab_stats, mode_stats = load_insights(st.session_state.run_id, sel_pl_id)

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

    # ── Perception accuracy ───────────────────────────────────────────────────
    st.subheader("Can you tell them apart?")
    conn = get_conn()
    guess_rows = conn.execute("""
        SELECT actual, guess, correct, guessed_at
        FROM queue_guesses
        ORDER BY guessed_at DESC
    """).fetchall()
    conn.close()

    if not guess_rows:
        st.caption("No guesses yet — use the SmartShuffle / Random baseline buttons on the queue view.")
    else:
        total_g   = len(guess_rows)
        correct_g = sum(r[2] for r in guess_rows)
        pct       = correct_g / total_g
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{pct:.0%}")
        m2.metric("Correct", correct_g)
        m3.metric("Total guesses", total_g)

        # breakdown by actual algorithm
        for actual, label in [("smartshuffle", "SmartShuffle"), ("random_baseline", "Random baseline")]:
            sub = [r for r in guess_rows if r[0] == actual]
            if sub:
                sub_correct = sum(r[2] for r in sub)
                st.caption(f"When it was {label}: {sub_correct}/{len(sub)} correct "
                           f"({sub_correct/len(sub):.0%})")

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
