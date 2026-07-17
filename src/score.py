"""
Phase 2 — Behavioral Modeling
==============================
1. Last.fm tag fetching       — per-song semantic tags (chill, hype, late night, etc.)
2. Session segmentation       — group plays into listening sessions
3. Context label inference    — label each session by time-of-day + tag-based energy
4. Fatigue scoring            — exponential decay + added_at normalization
5. Coverage debt              — plays since last heard / playlist size (not time-based)
6. Engagement baselines       — per-time-bucket skip rate / stakes aggregated from sessions
7. Export                     — writes all results back to DB

Run:  python phase2.py
Requires: smartshuffle.db populated by collect_data.py (phase 1)
          LASTFM_API_KEY in .env

─────────────────────────────────────────────────────────────────────────────
DEFERRED — requires live session loop (Phase 5 FastAPI/Streamlit frontend):

  [ ] Real-time engagement_delta: session_skip_rate_so_far - baseline_skip_rate.
      Needs to observe which song was just skipped, not just historical sessions.

  [ ] Vibe-searching trigger: 2+ consecutive early skips (< 30 s) in the current
      session → narrow remaining queue to familiar + energy-matched songs only.
      Requires knowing the order songs were played in the live session.

  [ ] Mid-session baseline update: if engagement_delta spikes partway through,
      re-classify stakes mid-session and re-score remaining queue positions.

  [ ] Per-session stakes update: stakes_level currently uses HISTORICAL typical
      stakes per time bucket. With a live loop it can update after each song plays.

What IS computed now from historical data:
  - early_skip_rate per session (fraction of skips < 30 s → vibe-searching signal)
  - stakes_level per session (high / normal / low)
  - engagement_baselines table: aggregated per time bucket, used by phase34.py
    to adjust weights at queue generation time
─────────────────────────────────────────────────────────────────────────────
"""

import json
import time
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, date, timedelta
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

# Use the system's local timezone for all hour-of-day bucketing.
# Spotify returns played_at in UTC; without this conversion every bucket
# is shifted by the UTC offset (e.g. PDT = −7 h) and all context labels are wrong.
from datetime import timezone as _tz
LOCAL_TZ = datetime.now(_tz.utc).astimezone().tzinfo

DB_PATH              = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "smartshuffle.db")
LASTFM_API_KEY       = os.getenv("LASTFM_API_KEY")
LASTFM_BASE          = "https://ws.audioscrobbler.com/2.0/"
SESSION_GAP_MINUTES  = 30
MIN_SESSION_PLAYS    = 2
BINGE_END_GAP_DAYS   = 2   # close a binge episode after this many days with no plays
BINGE_MAX_DAYS       = 14  # hard cap: episodes older than this are closed regardless of plays

# Evergreen scoring — framework is computed but stays zeroed until we have sufficient
# play history to reliably distinguish consistent favorites from fading binges.
EVERGREEN_SCRAPE_MIN_DAYS  = 90   # minimum total scrape window before any score is non-zero
EVERGREEN_CLUSTER_GAP_DAYS = 14   # gap that separates two distinct play clusters
EVERGREEN_RECENCY_HALF_LIFE = 30  # recency_confirmation half-life in days
EVERGREEN_MAX_LIFT         = 0.60 # max fraction of fatigue penalty that evergreen can cancel

# Continuous energy weights: -1.0 (pure chill) → +1.0 (pure hype).
# Genre tags (hip-hop, pop, r&b) contribute partial signal so mixed songs
# land between extremes rather than snapping to 0 or ±1.
ENERGY_WEIGHTS = {
    # ── High energy ──────────────────────────────────────────────────────────
    "hype":         1.0,  "energetic":   0.9,  "workout":     0.9,
    "aggressive":   0.9,  "intense":     0.8,  "pump up":     0.8,
    "high energy":  0.8,  "party":       0.8,  "rage":        0.8,
    "dance":        0.7,  "drill":       0.7,
    "hard":         0.7,  "bangers":     0.6,  "club":        0.6,
    "turn up":      0.6,  "lit":         0.6,  "upbeat":      0.6,
    "afrobeat":     0.5,  "afrobeats":   0.5,  "amapiano":    0.5,
    "dancehall":    0.5,  "grime":       0.6,
    "uk drill":     0.7,  "florida drill": 0.7, "chicago drill": 0.7,
    "brooklyn drill": 0.7, "ny drill":   0.7,  "detroit drill": 0.7,
    # ── Rock family ──────────────────────────────────────────────────────────
    "rock":         0.55, "hard rock":   0.70, "punk":        0.65,
    "punk rock":    0.65, "metal":       0.80, "heavy metal": 0.80,
    "classic rock": 0.40, "blues rock":  0.40, "pop rock":    0.30,
    "alternative rock": 0.25, "progressive rock": 0.20,
    "psychedelic rock": 0.20, "indie rock": 0.10, "soft rock": -0.10,
    # ── Electronic sub-genres ─────────────────────────────────────────────────
    "house":        0.60, "tech house":  0.65, "trance":      0.60,
    "drum and bass": 0.70, "dubstep":    0.65,
    # ── Country / blues / folk adjacent ──────────────────────────────────────
    "country":      0.10, "contemporary country": 0.15, "country pop": 0.15,
    "bluegrass":    0.20, "blues":       0.15,
    "folk pop":    -0.10, "sunshine pop": 0.20, "psychedelic pop": -0.20,
    # ── Classical / instrumental ──────────────────────────────────────────────
    "classical":   -0.55, "piano":      -0.35, "baroque":    -0.25,
    "romantic":    -0.45, "instrumental": -0.25, "orchestral": -0.35,
    "impressionist": -0.50, "medieval":  -0.40, "opera":      -0.30,
    # ── Mid positive — generic rap/pop (dampened when specific tags present) ──
    "hip-hop":      0.35, "hip hop":     0.35, "rap":         0.30,
    "trap":         0.15,  # melodic/vibe — intentionally below generic rap
    "pop rap":      0.25, "melodic rap": -0.10, "conscious hip hop": 0.0,
    "r&b":          0.15, "rnb":         0.15, "contemporary rnb": 0.10,
    "alternative rnb": 0.05, "neo soul": 0.05,
    "pop":          0.10, "electropop":  0.15, "synth pop":   0.15,
    "electronic":   0.20, "edm":         0.50,
    "alternative":  0.10, "reggae":      0.15, "reggaeton":   0.40,
    # ── Neutral ──────────────────────────────────────────────────────────────
    "soul":         0.0,  "funk":        0.15, "gospel":      0.05,
    "oldies":       0.10, "jazz":        0.05,
    # ── Low energy ───────────────────────────────────────────────────────────
    "ambient":     -0.9,  "sleep":      -0.9,  "peaceful":   -0.9,
    "lo-fi":       -0.8,  "lofi":       -0.8,  "calm":       -0.8,
    "dreamy":      -0.8,  "chill":      -0.7,  "relaxing":   -0.7,
    "chillout":    -0.6,  "slow":       -0.7,  "soft":       -0.7,
    "mellow":      -0.6,  "laid back":  -0.6,  "background": -0.6,
    "late night":  -0.5,  "focus":      -0.5,  "study":      -0.5,
    "rainy day":   -0.5,  "sad":        -0.4,  "melancholy": -0.4,
    "emotional":   -0.4,  "introspective": -0.4, "acoustic":  -0.3,
    "indie":       -0.1,  "bedroom pop": -0.2, "singer-songwriter": -0.2,
}

# Tags that are noise — too generic to be useful
JUNK_TAGS = {
    "seen live", "favorites", "favourite", "love", "awesome", "good",
    "great", "best", "amazing", "cool", "nice", "favorite", "liked",
    "under 2000 listeners", "spotify", "youtube", "albums i own"
}

# Tags so broad they appear on almost every rap/pop/r&b song.  Their raw vote
# counts swamp subgenre tags (drill, lo-fi, etc.) that actually distinguish
# songs.  We keep them in ENERGY_WEIGHTS so they still influence scores when
# nothing more specific is present, but shrink their effective vote count so a
# single "drill" or "chill" tag can override them.
GENERIC_TAGS = {
    "hip-hop", "hip hop", "rap", "music", "r&b", "rnb", "pop", "soul",
    "electronic", "alternative", "indie", "singer-songwriter",
}
GENERIC_TAG_VOTE_MULTIPLIER = 0.08

# Coverage debt threshold: if a song hasn't played in > 2x its expected
# rotation (playlist_size / 1), it's stale
COVERAGE_DEBT_THRESHOLD     = 2.0
COVERAGE_DEBT_SKIP_REVOKE   = 2  # consecutive queued skips (no completion since) that clear debt

# Artist completion rate: dominant song threshold.
# If one song accounts for >40% of all plays of an artist, it's a binge outlier
# and is excluded from the artist completion rate computation.
ARTIST_DOMINANCE_THRESHOLD  = 0.40

# Exponential half-lives for preference drift:
#   skip_rate per song — single value, no per-context split available
#   engagement baselines — morning gets a longer half-life because plays are sparse
SKIP_RATE_HALF_LIFE_DAYS = 90
BUCKET_HALF_LIFE = {
    "morning":   180,
    "afternoon":  90,
    "late_night": 90,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── DB init ──────────────────────────────────────────────────────────────────

def init_phase2_tables(conn):
    # song_scores migrations
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(song_scores)").fetchall()}
    if existing_cols and "binge_score" not in existing_cols:
        conn.execute("ALTER TABLE song_scores ADD COLUMN binge_score REAL")
    for col in ["artist_fatigue", "artist_binge_score", "binge_velocity", "binge_skip_rate",
                "evergreen_score", "artist_comp_rate", "artist_comp_conf"]:
        if existing_cols and col not in existing_cols:
            conn.execute(f"ALTER TABLE song_scores ADD COLUMN {col} REAL")

    # sessions migrations — add engagement columns to existing tables
    sessions_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    for col, typedef in [
        ("early_skip_rate",     "REAL"),
        ("skip_latency_mean_ms","REAL"),
        ("stakes_level",        "TEXT"),
    ]:
        if sessions_cols and col not in sessions_cols:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typedef}")

    conn.commit()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id          INTEGER PRIMARY KEY,
            context_label       TEXT,
            energy_label        TEXT,
            play_count          INTEGER,
            skip_rate           REAL,
            avg_energy_score    REAL,
            start_time          TEXT,
            end_time            TEXT,
            duration_min        REAL,
            hour_of_day         INTEGER,
            early_skip_rate     REAL,       -- fraction of skips that happened within 30 s
            skip_latency_mean_ms REAL,      -- mean play_duration_ms for skipped songs
            stakes_level        TEXT        -- 'high' | 'normal' | 'low'
        );

        CREATE TABLE IF NOT EXISTS engagement_baselines (
            time_bucket          TEXT PRIMARY KEY,
            baseline_skip_rate   REAL,     -- mean skip_rate across sessions in this bucket
            early_skip_rate      REAL,     -- mean early_skip_rate (< 30 s skips / total skips)
            typical_stakes       TEXT,     -- modal stakes_level across sessions
            session_count        INTEGER,  -- how many sessions this is derived from
            updated_at           TEXT
        );

        CREATE TABLE IF NOT EXISTS song_scores (
            song_id             TEXT PRIMARY KEY,
            song_name           TEXT,
            artist_name         TEXT,
            fatigue             REAL,
            binge_score         REAL,
            artist_fatigue      REAL,
            artist_binge_score  REAL,
            play_count          INTEGER,
            last_played         TEXT,
            skip_rate           REAL,
            pattern             TEXT,
            coverage_debt       REAL,
            stale               INTEGER,
            days_since_added    INTEGER,
            play_rate_per_week  REAL,
            updated_at          TEXT,
            binge_velocity      REAL,
            binge_skip_rate     REAL,
            evergreen_score     REAL,
            artist_comp_rate    REAL,   -- completion rate of non-dominant songs by this artist
            artist_comp_conf    REAL    -- confidence in [0,1]: grows with unique non-dominant songs heard
        );

        CREATE TABLE IF NOT EXISTS binge_score_history (
            song_id     TEXT NOT NULL,
            date        TEXT NOT NULL,
            binge_score REAL NOT NULL,
            PRIMARY KEY (song_id, date)
        );

        CREATE TABLE IF NOT EXISTS song_tags (
            song_id      TEXT PRIMARY KEY,
            tags         TEXT,        -- JSON: [{name, weight}, ...]
            energy_score REAL,        -- -1.0 (chill) to 1.0 (hype)
            top_tags     TEXT,        -- JSON: [str, ...] top 5 clean tags
            fetched_at   TEXT,
            fetch_source TEXT         -- 'lastfm_track' | 'lastfm_artist' | 'none'
        );

        CREATE TABLE IF NOT EXISTS binge_episodes (
            episode_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id        TEXT NOT NULL,
            artist_name    TEXT,
            start_date     TEXT NOT NULL,   -- ISO date when binge first detected
            end_date       TEXT,            -- ISO date when ended; NULL = ongoing
            daily_plays    TEXT NOT NULL DEFAULT '[]',  -- JSON: plays per day from start_date
            peak_plays     INTEGER DEFAULT 0,
            episode_length INTEGER          -- days from start to end; NULL if ongoing
        );

        CREATE INDEX IF NOT EXISTS idx_binge_end ON binge_episodes(end_date);
    """)
    conn.commit()


# ── Last.fm tag fetching ──────────────────────────────────────────────────────

def clean_tags(raw_tags: list) -> list:
    """Remove junk tags, lowercase, deduplicate."""
    seen = set()
    clean = []
    for t in raw_tags:
        name = t["name"].lower().strip()
        if name in JUNK_TAGS:
            continue
        if name in seen:
            continue
        seen.add(name)
        clean.append({"name": name, "weight": int(t.get("count", 0))})
    return clean


def compute_energy_score(tags: list) -> float:
    """
    Returns a score from -1.0 (very chill) to 1.0 (very hype).
    Each tag contributes its ENERGY_WEIGHTS value, scaled by Last.fm vote
    count.  Generic umbrella tags (hip-hop, rap, r&b…) have their vote count
    heavily dampened when specific subgenre tags are also present, so drill /
    lo-fi / afrobeats etc. control the result.  When only generic tags exist,
    they contribute at full weight (better than returning 0.0).
    """
    has_specific = any(
        ENERGY_WEIGHTS.get(t["name"]) is not None and t["name"] not in GENERIC_TAGS
        for t in tags
    )
    weighted_sum  = 0.0
    total_votes   = 0.0
    for t in tags:
        intensity = ENERGY_WEIGHTS.get(t["name"])
        if intensity is None:
            continue
        raw_votes = t["weight"] or 1
        dampen = has_specific and t["name"] in GENERIC_TAGS
        votes = raw_votes * (GENERIC_TAG_VOTE_MULTIPLIER if dampen else 1.0)
        weighted_sum += intensity * votes
        total_votes  += votes
    if total_votes == 0:
        return 0.0
    return round(weighted_sum / total_votes, 4)


def fetch_lastfm_tags(song_name: str, artist_name: str) -> dict:
    """
    Fetches tags from Last.fm. Tries track.getTopTags first (more specific),
    falls back to artist.getTopTags if the track has no tags.
    Returns dict with tags, energy_score, top_tags, fetch_source.
    """
    if not LASTFM_API_KEY:
        return {"tags": [], "energy_score": 0.0, "top_tags": [], "fetch_source": "none"}

    def call(method, params):
        params.update({
            "method":  method,
            "api_key": LASTFM_API_KEY,
            "format":  "json",
            "autocorrect": 1,
        })
        try:
            r = requests.get(LASTFM_BASE, params=params, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}

    # Try track-level tags first
    data = call("track.getTopTags", {"track": song_name, "artist": artist_name})
    raw  = data.get("toptags", {}).get("tag", [])
    source = "lastfm_track"

    # Fall back to artist-level tags if track has none
    if not raw:
        data   = call("artist.getTopTags", {"artist": artist_name})
        raw    = data.get("toptags", {}).get("tag", [])
        source = "lastfm_artist"

    if not raw:
        return {"tags": [], "energy_score": 0.0, "top_tags": [], "fetch_source": "none"}

    tags         = clean_tags(raw)
    energy_score = compute_energy_score(tags)
    top_tags     = [t["name"] for t in sorted(tags, key=lambda x: x["weight"], reverse=True)[:5]]

    return {
        "tags":         tags,
        "energy_score": energy_score,
        "top_tags":     top_tags,
        "fetch_source": source,
    }


def fetch_all_tags(conn, df: pd.DataFrame):
    """
    Fetches Last.fm tags for all songs not already in song_tags.
    Writes to DB incrementally so a crash doesn't lose progress.
    """
    existing = set(
        row[0] for row in conn.execute("SELECT song_id FROM song_tags").fetchall()
    )

    songs = df[["song_id", "song_name", "artist_name"]].drop_duplicates("song_id")
    missing = songs[~songs["song_id"].isin(existing)]

    if missing.empty:
        print("  All songs already have Last.fm tags.")
        return

    print(f"  Fetching Last.fm tags for {len(missing)} songs...")
    for _, row in missing.iterrows():
        result = fetch_lastfm_tags(row["song_name"], row["artist_name"])
        conn.execute("""
            INSERT OR REPLACE INTO song_tags
            (song_id, tags, energy_score, top_tags, fetched_at, fetch_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["song_id"],
            json.dumps(result["tags"]),
            result["energy_score"],
            json.dumps(result["top_tags"]),
            now_iso(),
            result["fetch_source"],
        ))
        conn.commit()
        time.sleep(0.2)   # Last.fm rate limit: 5 req/sec; 0.2s = safe

    print("  Tag fetch complete.")


def recompute_energy_scores(conn):
    """
    Re-applies compute_energy_score to stored raw tags without hitting the API.
    Needed when the scoring formula changes after tags are already cached.
    """
    rows = conn.execute("SELECT song_id, tags FROM song_tags").fetchall()
    updates = []
    for song_id, tags_json in rows:
        if not tags_json:
            continue
        tags         = json.loads(tags_json)
        energy_score = compute_energy_score(tags)
        top_tags     = [t["name"] for t in sorted(tags, key=lambda x: x["weight"], reverse=True)[:5]]
        updates.append((energy_score, json.dumps(top_tags), song_id))
    conn.executemany(
        "UPDATE song_tags SET energy_score = ?, top_tags = ? WHERE song_id = ?",
        updates,
    )
    conn.commit()
    print(f"  Recomputed energy scores for {len(updates)} songs.")


def blend_interjection_energy(conn):
    """
    Nudges a song's energy score toward the context energy observed when the
    user manually inserted it mid-queue (session_interjections table).

    If you're in a 0.65-energy queue and you queue up Song X, that's direct
    evidence that Song X belongs at ~0.65 energy in that context — stronger
    than any tag inference. We blend conservatively: 5 observations yields
    40% weight to the observed mean, capped there.

    Only songs with at least 2 interjection observations are updated so a
    single accidental insertion doesn't corrupt the score.
    """
    try:
        rows = conn.execute("""
            SELECT song_id, COUNT(*) AS n, AVG(context_energy) AS mean_context_energy
            FROM session_interjections
            WHERE context_energy IS NOT NULL
            GROUP BY song_id
            HAVING COUNT(*) >= 2
        """).fetchall()
    except Exception:
        return  # table may not exist yet

    if not rows:
        return

    updated = 0
    for song_id, n, mean_ctx in rows:
        tag_row = conn.execute(
            "SELECT energy_score FROM song_tags WHERE song_id = ?", (song_id,)
        ).fetchone()
        if tag_row is None:
            continue
        base_energy   = tag_row[0] or 0.0
        blend_factor  = min(n / 5.0, 0.40)   # max 40% weight to observations
        blended       = round((1 - blend_factor) * base_energy + blend_factor * mean_ctx, 4)
        conn.execute(
            "UPDATE song_tags SET energy_score = ? WHERE song_id = ?",
            (blended, song_id),
        )
        updated += 1

    conn.commit()
    if updated:
        print(f"  Interjection-blended energy for {updated} songs.")


def impute_missing_energy(conn, df: pd.DataFrame, session_stats: dict):
    """
    For songs where Last.fm returned no energy-relevant tags (score stays 0.0),
    infer energy from the avg_energy_score of sessions the song appeared in.
    Songs that never appear in play history stay at 0.0 (neutral).

    Chicken-and-egg guard: excludes plays sourced from SmartShuffle queues
    (play_source = 'smartshuffle_queued') so we don't circularly infer a song's
    energy from sessions where we placed it ourselves. This field will be set
    by collect_data.py once the Streamlit frontend is live and users follow
    our queues — no-op for now since all current plays are organic.
    """
    zero_ids = {
        row[0] for row in conn.execute(
            "SELECT song_id FROM song_tags WHERE energy_score = 0.0"
        ).fetchall()
    }
    if not zero_ids:
        return

    # Only use organic plays for inference
    organic_df     = df[df["play_source"] != "smartshuffle_queued"]
    session_energy = {sid: s["avg_energy_score"] for sid, s in session_stats.items()}

    organic_zero = organic_df[organic_df["song_id"].isin(zero_ids)].copy()
    organic_zero["_s_energy"] = organic_zero["session_id"].map(session_energy)
    organic_zero = organic_zero.dropna(subset=["_s_energy"])

    if not organic_zero.empty:
        inferred = organic_zero.groupby("song_id")["_s_energy"].mean().round(4)
        conn.executemany(
            "UPDATE song_tags SET energy_score = ?, fetch_source = 'session_inferred'"
            " WHERE song_id = ?",
            [(float(e), sid) for sid, e in inferred.items()],
        )
        conn.commit()
        print(f"  Session-inferred energy for {len(inferred)} songs (no Last.fm signal).")
    else:
        print("  Session-inferred energy for 0 songs (no Last.fm signal).")


# ── Manual-play detection ──────────────────────────────────────────────────────

MAX_INTERJECTION_GAP_MIN = 15


def _session_manual_songs(playlist_plays: pd.DataFrame) -> set:
    """
    Returns song_ids whose null-context plays occurred in a session that also
    contained at least one known-context play (playlist/artist/album URI).

    A session is defined as plays within 30 minutes of each other. If every play
    in a session has null context, we assume it was an offline or BART listen and
    exclude it. Mixed sessions (some known, some null) → the null plays are genuine
    manual seeks the user actively searched for.
    """
    _df = playlist_plays.sort_values("played_at").copy()
    _df["_gap_min"] = _df["played_at"].diff().dt.total_seconds() / 60
    _df["_session_id"] = (_df["_gap_min"].isna() | (_df["_gap_min"] > 30)).cumsum()

    _has_known = _df["context_uri"].notna() & (_df["context_uri"] != "")
    _mixed_ids = set(_df.loc[_has_known, "_session_id"])
    _df["_mixed"] = _df["_session_id"].isin(_mixed_ids)

    _is_null    = _df["context_uri"].isna() | (_df["context_uri"] == "")
    _not_queued = ~_df["play_source"].str.endswith("_queued", na=False)

    return set(_df.loc[_is_null & _not_queued & _df["_mixed"], "song_id"])


def _queue_adjacent_songs(playlist_plays: pd.DataFrame, push_playlist_uri: str | None) -> set:
    """
    Songs sandwiched between two smartshuffle_queued plays within MAX_INTERJECTION_GAP_MIN
    minutes, played from a source other than the push playlist itself.

    Catches genuine interjections that session_interjections misses at push boundaries —
    e.g. user plays one song from a different playlist mid-queue, then SmartShuffle resumes.
    Push-playlist plays (natural queue continuation) are excluded to avoid false positives.
    """
    _df = playlist_plays.sort_values("played_at").reset_index(drop=True)
    _queued = _df["play_source"].str.endswith("_queued", na=False)

    result = set()
    queue_times = _df.loc[_queued, "played_at"]
    if queue_times.empty:
        return result

    for i, row in _df[~_queued].iterrows():
        ctx = row["context_uri"]
        # Skip plays that are just the push playlist playing through naturally
        if push_playlist_uri and ctx == push_playlist_uri:
            continue

        curr_ts = row["played_at"]
        prev_q = queue_times[queue_times < curr_ts]
        next_q = queue_times[queue_times > curr_ts]
        if prev_q.empty or next_q.empty:
            continue

        gap_before = (curr_ts - prev_q.iloc[-1]).total_seconds() / 60
        gap_after  = (next_q.iloc[0]  - curr_ts).total_seconds() / 60
        if gap_before <= MAX_INTERJECTION_GAP_MIN and gap_after <= MAX_INTERJECTION_GAP_MIN:
            result.add(row["song_id"])

    return result


# ── Load data ─────────────────────────────────────────────────────────────────

def load_plays(conn) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT
            p.id, p.played_at, p.song_id, p.song_name, p.artist_name,
            p.duration_ms, p.play_duration_ms, p.inferred_skip, p.play_source,
            p.context_uri,
            st.energy_score, st.top_tags,
            s.audio_features
        FROM plays p
        LEFT JOIN song_tags st ON p.song_id = st.song_id
        LEFT JOIN songs s      ON p.song_id = s.song_id
        WHERE p.play_source != 'jam_excluded'
        ORDER BY p.played_at ASC
    """, conn)

    df["played_at"]   = pd.to_datetime(df["played_at"], format="ISO8601", utc=True)
    df["energy_score"] = pd.to_numeric(df["energy_score"], errors="coerce").fillna(0.0)
    df["top_tags"]    = df["top_tags"].apply(lambda x: json.loads(x) if x else [])

    return df


# ── Session segmentation ──────────────────────────────────────────────────────

def segment_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("played_at").reset_index(drop=True)
    gap_min = df["played_at"].diff().dt.total_seconds().div(60).fillna(0)
    df["session_id"] = (gap_min > SESSION_GAP_MINUTES).cumsum()
    return df


# ── Context label inference ───────────────────────────────────────────────────

def get_time_bucket(hour: int) -> str:
    if hour >= 21 or hour < 6: return "late_night"
    if hour < 11: return "morning"
    return "afternoon"


def get_energy_label(avg_energy: float) -> str:
    if avg_energy >  0.15: return "high_energy"
    if avg_energy < -0.15: return "low_energy"
    return "mid_energy"


def label_sessions(df: pd.DataFrame):
    session_stats = {}

    for sid, group in df.groupby("session_id"):
        if len(group) < MIN_SESSION_PLAYS:
            continue

        hour         = int(group["played_at"].dt.tz_convert(LOCAL_TZ).dt.hour.median())
        time_bucket  = get_time_bucket(hour)
        avg_energy   = group["energy_score"].mean()
        energy_label = get_energy_label(avg_energy)
        context_label = f"{time_bucket}_{energy_label}"

        _known       = group[group["inferred_skip"] != "unknown"]
        skip_rate    = ((_known["inferred_skip"] == "skip").mean()
                        if len(_known) > 0 else 0.0)
        start_time   = group["played_at"].min()
        end_time     = group["played_at"].max()
        duration_min = (end_time - start_time).total_seconds() / 60

        # Skip latency: how quickly skipped songs were abandoned.
        # Early skip (< 30 s) = user knew immediately it wasn't right → vibe-searching signal.
        skipped = group[group["inferred_skip"] == "skip"]
        early_skip_rate    = 0.0
        skip_latency_mean  = None
        if len(skipped) > 0:
            valid_lat = skipped["play_duration_ms"].dropna()
            if len(valid_lat) > 0:
                early_skip_rate   = float((valid_lat < 30_000).sum() / len(valid_lat))
                skip_latency_mean = float(valid_lat.mean())

        # Stakes level: how much it mattered which song played.
        # High  — mostly early skips → user was actively searching for a specific vibe.
        # Low   — almost no skips → passive listening, song choice barely noticed.
        # Normal — everything else.
        if skip_rate > 0.15 and early_skip_rate > 0.5:
            stakes_level = "high"
        elif skip_rate < 0.05:
            stakes_level = "low"
        else:
            stakes_level = "normal"

        session_stats[sid] = {
            "session_id":          sid,
            "context_label":       context_label,
            "energy_label":        energy_label,
            "play_count":          len(group),
            "skip_rate":           round(skip_rate, 3),
            "avg_energy_score":    round(avg_energy, 4),
            "start_time":          start_time.isoformat(),
            "end_time":            end_time.isoformat(),
            "duration_min":        round(duration_min, 1),
            "hour_of_day":         hour,
            "early_skip_rate":     round(early_skip_rate, 4),
            "skip_latency_mean_ms": round(skip_latency_mean, 1) if skip_latency_mean else None,
            "stakes_level":        stakes_level,
        }

    df["context_label"] = df["session_id"].map(
        {sid: s["context_label"] for sid, s in session_stats.items()}
    )
    return df, session_stats


# ── Fatigue scoring ───────────────────────────────────────────────────────────

def compute_fatigue_scores(df: pd.DataFrame, conn) -> dict:
    """
    Fatigue score (0–1) per song using:
    - Exponential decay (half-life 14 days) on recent plays
    - Skip-weighted play contribution
    - Play rate since added_at: fast early adoption + dropoff = binge pattern

    Binge score is only non-zero when BOTH of the following hold:
      play_count >= 3
      at least 2 of 3 binge signals are present:
        newly_added     — added to playlist < 30 days ago
        new_release     — song's album released < 90 days ago
        manually_played — any play via manual_search or artist_browse
      (new_discovery excluded — first-play recency fires for any unplayed library song)
    """
    now = datetime.now(timezone.utc)

    # Pull added_at from playlist_tracks (earliest add date if in multiple playlists)
    added_map = {}
    rows = conn.execute("""
        SELECT song_id, MIN(added_at) as first_added
        FROM playlist_tracks
        WHERE added_at IS NOT NULL
        GROUP BY song_id
    """).fetchall()
    for song_id, added_at in rows:
        try:
            added_map[song_id] = datetime.fromisoformat(
                added_at.replace("Z", "+00:00")
            )
        except Exception:
            pass

    # Pull release dates for the "new release" binge signal
    release_map = {}
    for sid, rd in conn.execute(
        "SELECT song_id, json_extract(audio_features, '$.release_date') "
        "FROM songs WHERE audio_features IS NOT NULL"
    ).fetchall():
        if rd:
            try:
                release_map[sid] = datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)
            except Exception:
                pass

    half_life_days   = 14
    decay_constant   = np.log(2) / half_life_days
    binge_half_life  = 7
    binge_decay      = np.log(2) / binge_half_life
    scores           = {}

    # Include all real play sources — jam_excluded is the only one we skip.
    # Note: energy imputation (above) separately guards against smartshuffle_queued
    # to avoid circular inference; that exclusion is intentional and separate from this.
    playlist_plays = df[df["play_source"] != "jam_excluded"].copy()
    if playlist_plays.empty:
        return {}

    # Vectorized decay — eliminates O(total_plays) Python loop
    now_ts = pd.Timestamp(now)
    playlist_plays["_days_ago"] = (
        (now_ts - playlist_plays["played_at"]).dt.total_seconds() / 86400
    )
    playlist_plays["_skip_factor"] = np.select(
        [playlist_plays["inferred_skip"] == "skip",
         playlist_plays["inferred_skip"] == "partial"],
        [0.3, 0.6], default=1.0,
    )
    playlist_plays["_weighted"] = (
        np.exp(-decay_constant * playlist_plays["_days_ago"]) * playlist_plays["_skip_factor"]
    )
    playlist_plays["_binge_w"] = (
        np.exp(-binge_decay * playlist_plays["_days_ago"]) * playlist_plays["_skip_factor"]
    )

    # Decay weights for skip_rate estimation (independent of skip_factor used for fatigue).
    # Recent skips/listens outweigh old ones so changing taste is reflected quickly.
    # High-stakes sessions (active listening) are weighted more than low-stakes (studying/background).
    stakes_map = {r[0]: r[1] for r in conn.execute(
        "SELECT session_id, stakes_level FROM sessions"
    ).fetchall()}
    stakes_multiplier = playlist_plays["session_id"].map(stakes_map).map(
        {"high": 1.5, "normal": 1.0, "low": 0.5}
    ).fillna(1.0)

    _sr_decay = np.log(2) / SKIP_RATE_HALF_LIFE_DAYS
    _sr_w = np.exp(-_sr_decay * playlist_plays["_days_ago"]) * stakes_multiplier
    playlist_plays["_decay_skip"]  = np.where(playlist_plays["inferred_skip"] == "skip",  _sr_w, 0.0)
    playlist_plays["_decay_known"] = np.where(playlist_plays["inferred_skip"] != "unknown", _sr_w, 0.0)

    agg = playlist_plays.groupby("song_id", sort=False).agg(
        play_count   = ("song_id",       "count"),
        weighted_sum = ("_weighted",     "sum"),
        binge_wsum   = ("_binge_w",      "sum"),
        first_played = ("played_at",     "min"),
        last_played  = ("played_at",     "max"),
        song_name    = ("song_name",     "first"),
        artist_name  = ("artist_name",   "first"),
        skip_count   = ("inferred_skip", lambda x: (x == "skip").sum()),
        unknown_count= ("inferred_skip", lambda x: (x == "unknown").sum()),
        w_skip       = ("_decay_skip",   "sum"),
        w_known      = ("_decay_known",  "sum"),
    )

    # Incorporate queue_skips (songs skipped too fast to appear in recently-played).
    # Weight by position: early skips are strong signal; late skips are noise.
    # position_weight = clip(1.0 - position/50, min=0.2) → 1.0 at pos 0, 0.2 at pos 40+
    qs_rows = conn.execute("""
        SELECT qs.song_id, qs.queue_position, qp.pushed_at
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
        WHERE qs.queue_position IS NOT NULL
    """).fetchall()
    if qs_rows:
        qs_df = pd.DataFrame(qs_rows, columns=["song_id", "queue_position", "pushed_at"])
        qs_df["pushed_at"] = pd.to_datetime(qs_df["pushed_at"], format="ISO8601", utc=True)
        qs_df["_days_ago"]    = (now_ts - qs_df["pushed_at"]).dt.total_seconds() / 86400
        qs_df["_pos_weight"]  = (1.0 - qs_df["queue_position"] / 50).clip(lower=0.2)
        qs_df["_qs_w"]        = np.exp(-_sr_decay * qs_df["_days_ago"]) * qs_df["_pos_weight"]
        qs_agg = qs_df.groupby("song_id")["_qs_w"].sum()
        for song_id, w in qs_agg.items():
            if song_id in agg.index:
                agg.at[song_id, "w_skip"]  += w
                agg.at[song_id, "w_known"] += w

    # Recent skip rate per song (last 14 days)
    cutoff = now_ts - pd.Timedelta(days=14)
    recent = playlist_plays[playlist_plays["played_at"] >= cutoff]
    if not recent.empty:
        recent_agg = recent.groupby("song_id", sort=False).agg(
            recent_count   = ("song_id",       "count"),
            recent_unknown = ("inferred_skip", lambda x: (x == "unknown").sum()),
            recent_skips   = ("inferred_skip", lambda x: (x == "skip").sum()),
        )
        recent_known    = (recent_agg["recent_count"] - recent_agg["recent_unknown"]).clip(lower=0)
        recent_skip_map = (recent_agg["recent_skips"] / recent_known.replace(0, float("nan"))
                           ).fillna(0.0).to_dict()
    else:
        recent_skip_map = {}

    # Reliable manual signal — four sources:
    # 1. artist_browse: explicit context_uri proves user navigated to artist page
    # 2. session_manual: null plays in mixed sessions (other plays have known context)
    # 3. session_interjections: queue interruptions captured at analysis time
    # 4. queue_adjacent: non-push-playlist plays sandwiched between queued plays (≤15 min)
    _push_pl_row = conn.execute("SELECT value FROM config WHERE key='push_playlist_id'").fetchone()
    _push_pl_uri = f"spotify:playlist:{_push_pl_row[0]}" if _push_pl_row else None
    _sm = _session_manual_songs(playlist_plays)
    _qa = _queue_adjacent_songs(playlist_plays, _push_pl_uri)
    try:
        _interjected = set(r[0] for r in conn.execute(
            "SELECT DISTINCT song_id FROM session_interjections"
        ).fetchall())
    except Exception:
        _interjected = set()
    manual_songs = set(
        playlist_plays.loc[
            playlist_plays["play_source"].isin(["artist_browse"]),
            "song_id",
        ]
    ) | _sm | _qa | _interjected

    # Songs whose binge episode has exceeded BINGE_MAX_DAYS (open OR recently closed).
    # Cooling lookback = BINGE_MAX_DAYS: by the time a closed episode's cooling expires,
    # any new episode opened on the same day will have hit BINGE_MAX_DAYS itself — no gap.
    today_date = datetime.now(timezone.utc).date()
    age_capped_songs = set(
        song_id for (song_id,) in conn.execute("""
            SELECT DISTINCT song_id FROM binge_episodes
            WHERE (end_date IS NULL OR julianday('now') - julianday(end_date) <= ?)
              AND julianday('now') - julianday(start_date) >= ?
        """, (BINGE_MAX_DAYS, BINGE_MAX_DAYS)).fetchall()
    )

    # Skips and total plays within the binge window — used for skip penalty and skip_rate.
    _binge_window = playlist_plays[playlist_plays["_days_ago"] < BINGE_MAX_DAYS].copy()
    _binge_skips_df = _binge_window[_binge_window["inferred_skip"] == "skip"].copy()

    # Forgive queued skips where the user already played the same song manually within
    # the last 60 minutes — they wouldn't listen to it again a few queue positions later.
    _queued_sources = {"smartshuffle_queued", "random_baseline_queued"}
    _non_queued = _binge_window[~_binge_window["play_source"].isin(_queued_sources)][["song_id", "played_at"]]
    _dupe_window = pd.Timedelta(minutes=60)

    def _is_forgiven_skip(row):
        if row["play_source"] not in _queued_sources:
            return False
        return not _non_queued[
            (_non_queued["song_id"] == row["song_id"]) &
            (_non_queued["played_at"] >= row["played_at"] - _dupe_window) &
            (_non_queued["played_at"] < row["played_at"])
        ].empty

    if not _binge_skips_df.empty:
        _binge_skips_df["_forgiven"] = _binge_skips_df.apply(_is_forgiven_skip, axis=1)
        _binge_skips_df = _binge_skips_df[~_binge_skips_df["_forgiven"]]

    binge_skip_counts = _binge_skips_df.groupby("song_id").size().to_dict()
    binge_play_counts = _binge_window.groupby("song_id").size().to_dict()

    # 3-day-old binge scores for velocity computation.
    _three_days_ago = (datetime.now(timezone.utc).date() - timedelta(days=3)).isoformat()
    _old_binge = dict(conn.execute(
        "SELECT song_id, binge_score FROM binge_score_history WHERE date = ?",
        (_three_days_ago,)
    ).fetchall())

    # ── Evergreen scoring pre-computation ─────────────────────────────────────
    # Measures long-term consistent play preference: a song you reliably return to
    # across multiple distinct listening phases vs. one that peaks during a binge
    # and then stops getting played.
    #
    # The score stays 0 until the scrape window reaches EVERGREEN_SCRAPE_MIN_DAYS —
    # the cluster/span signals are not meaningful with only weeks of history.
    _scrape_span_days = (
        playlist_plays["played_at"].max() - playlist_plays["played_at"].min()
    ).total_seconds() / 86400
    _evergreen_active = _scrape_span_days >= EVERGREEN_SCRAPE_MIN_DAYS

    # Count distinct play clusters per song: gaps > EVERGREEN_CLUSTER_GAP_DAYS
    # between consecutive plays of the same song start a new cluster.
    # A song that appears in 2+ separate listening phases (gap-and-return) is a
    # stronger evergreen candidate than one with all plays in a single burst.
    if _evergreen_active:
        _ev_recency_decay = np.log(2) / EVERGREEN_RECENCY_HALF_LIFE

        def _count_clusters(dates: pd.Series) -> int:
            if len(dates) < 2:
                return 1
            gaps = dates.sort_values().diff().dt.total_seconds().dropna() / 86400
            return int((gaps > EVERGREEN_CLUSTER_GAP_DAYS).sum()) + 1

        _cluster_counts = (
            playlist_plays.groupby("song_id")["played_at"].apply(_count_clusters).to_dict()
        )
    else:
        _cluster_counts = {}

    for song_id, row in agg.iterrows():
        play_count     = int(row["play_count"])
        weighted_plays = float(row["weighted_sum"])
        binge_weighted = float(row["binge_wsum"])
        fatigue        = min(weighted_plays / 5.0, 1.0)

        added_at         = added_map.get(song_id)
        days_since_added = (now - added_at).days if added_at else None

        binge_signals = sum([
            bool(days_since_added is not None and days_since_added < 30),
            bool(release_map.get(song_id) and (now - release_map[song_id]).days < 90),
            bool(song_id in manual_songs),
        ])
        binge_threshold = max(play_count * 0.8, 5.0)
        raw_binge       = min(binge_weighted / binge_threshold, 1.0)
        binge_score     = round(raw_binge, 4) if (play_count >= 3 and binge_signals >= 2) else 0.0

        # Hard cap: episode >= BINGE_MAX_DAYS old (open or just closed) → song is integrated
        if song_id in age_capped_songs:
            binge_score = 0.0

        # Skip penalty: each skip in the binge window reduces score by 25% (4 skips → 0).
        # Binge songs are almost never skipped; skips are strong evidence it's not a real binge.
        _binge_skips = binge_skip_counts.get(song_id, 0)
        if binge_score > 0 and _binge_skips > 0:
            binge_score = round(binge_score * max(0.0, 1.0 - 0.25 * _binge_skips), 4)

        # Velocity: how fast the binge score changed over the last 3 days.
        # Negative velocity means the binge is cooling — used by the scheduler to cut plays.
        _binge_plays = binge_play_counts.get(song_id, 0)
        binge_velocity  = round(binge_score - _old_binge.get(song_id, binge_score), 4)
        binge_skip_rate = round(_binge_skips / _binge_plays, 4) if _binge_plays > 0 else 0.0

        w_known = float(row["w_known"])
        skip_rate = float(row["w_skip"] / w_known) if w_known > 1e-9 else 0.0

        if added_at:
            weeks_since_added  = max(days_since_added / 7, 0.1)
            play_rate_per_week = round(play_count / weeks_since_added, 3)
        else:
            play_rate_per_week = None

        rs_rate = recent_skip_map.get(song_id, 0.0)

        if play_count >= 4 and fatigue > 0.6:
            pattern = "binge"
        elif play_count >= 3 and rs_rate > 0.5:
            pattern = "growing_tired"
        elif play_count >= 5 and fatigue < 0.2 and days_since_added and days_since_added > 60:
            pattern = "long_term_keeper"
        else:
            pattern = "normal"

        # ── Evergreen score ────────────────────────────────────────────────────
        # historical_strength: how spread out and multi-clustered are the plays?
        #   span_component  — scales 0→1 over 0→365 days between first and last play
        #   cluster_component — 1 cluster=0, 2 clusters=0.5, 3+=1.0
        # recency_confirmation: decays if not played recently (30-day half-life)
        # Both are zero until EVERGREEN_SCRAPE_MIN_DAYS of data exists.
        if _evergreen_active:
            _play_span = (row["last_played"] - row["first_played"]).total_seconds() / 86400
            _clusters  = _cluster_counts.get(song_id, 1)
            _span_comp    = min(_play_span / 365.0, 1.0)
            _cluster_comp = min((_clusters - 1) / 2.0, 1.0)
            _historical   = _span_comp * _cluster_comp

            _days_since_last = (now_ts - row["last_played"]).total_seconds() / 86400
            _recency_conf = float(np.exp(-_ev_recency_decay * _days_since_last))

            evergreen_score = round(_historical * _recency_conf, 4)

            # No double-counting with active binge
            if binge_score > 0.3:
                evergreen_score = 0.0
            # Skip rate revocation: if we have enough queue coverage and skip rate is high,
            # the song is no longer a true favorite regardless of history
            if w_known >= 5.0 and skip_rate > 0.30:
                evergreen_score = 0.0
        else:
            evergreen_score = 0.0

        scores[song_id] = {
            "song_name":          str(row["song_name"]),
            "artist_name":        str(row["artist_name"]),
            "fatigue":            round(fatigue, 4),
            "binge_score":        binge_score,
            "binge_velocity":     binge_velocity,
            "binge_skip_rate":    binge_skip_rate,
            "evergreen_score":    evergreen_score,
            "play_count":         play_count,
            "last_played":        row["last_played"].isoformat(),
            "skip_rate":          round(skip_rate, 3),
            "pattern":            pattern,
            "play_rate_per_week": play_rate_per_week,
            "days_since_added":   days_since_added,
        }

    return scores


def compute_artist_fatigue_scores(df: pd.DataFrame, conn) -> dict:
    """
    Artist-level fatigue and binge — same decay formulas as compute_fatigue_scores
    but summed across ALL songs by the same artist.

    artist_fatigue:      how heavily you've listened to this artist recently (14-day decay)
    artist_binge_score:  whether you're in an active artist-binge (7-day decay)

    Binge score is only non-zero when BOTH of the following hold:
      play_count >= 3
      at least 2 of 3 binge signals are present:
        new_release     — artist released music < 90 days ago
        newly_added     — a song by this artist added to any playlist < 30 days ago
        manually_played — any play via manual_search or artist_browse
      (new_discovery intentionally excluded — fires too broadly and causes false positives)

    Denominator for fatigue is the same per-song threshold (5.0 weighted plays = full
    fatigue), applied to the artist's total play volume — so an artist heard across
    5 different songs once each reads the same as one song heard 5 times. This captures
    artist-level overexposure that per-song fatigue misses.

    Keyed by artist_name; phase2.save_to_db maps each song to its artist's scores.
    """
    now            = datetime.now(timezone.utc)
    decay_constant = np.log(2) / 14
    binge_decay    = np.log(2) / 7

    # Most recent release date per artist
    artist_latest_release: dict = {}
    for aname, rd in conn.execute("""
        SELECT artist_name, MAX(json_extract(audio_features, '$.release_date'))
        FROM songs WHERE audio_features IS NOT NULL GROUP BY artist_name
    """).fetchall():
        if rd:
            try:
                artist_latest_release[aname] = datetime.fromisoformat(rd).replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                pass

    # Most recent playlist add date per artist
    artist_latest_added: dict = {}
    for aname, la in conn.execute("""
        SELECT s.artist_name, MAX(pt.added_at)
        FROM playlist_tracks pt JOIN songs s ON pt.song_id = s.song_id
        WHERE pt.added_at IS NOT NULL GROUP BY s.artist_name
    """).fetchall():
        if la:
            try:
                artist_latest_added[aname] = datetime.fromisoformat(
                    la.replace("Z", "+00:00")
                )
            except Exception:
                pass

    playlist_plays = df[df["play_source"] != "jam_excluded"].copy()
    if playlist_plays.empty:
        return {}

    # Vectorized decay — eliminates O(total_plays) Python loop
    now_ts = pd.Timestamp(now)
    playlist_plays["_days_ago"] = (
        (now_ts - playlist_plays["played_at"]).dt.total_seconds() / 86400
    )
    playlist_plays["_skip_factor"] = np.select(
        [playlist_plays["inferred_skip"] == "skip",
         playlist_plays["inferred_skip"] == "partial"],
        [0.3, 0.6], default=1.0,
    )
    playlist_plays["_weighted"] = (
        np.exp(-decay_constant * playlist_plays["_days_ago"]) * playlist_plays["_skip_factor"]
    )
    playlist_plays["_binge_w"] = (
        np.exp(-binge_decay * playlist_plays["_days_ago"]) * playlist_plays["_skip_factor"]
    )

    agg = playlist_plays.groupby("artist_name", sort=False).agg(
        play_count   = ("artist_name", "count"),
        weighted_sum = ("_weighted",   "sum"),
        binge_wsum   = ("_binge_w",    "sum"),
    )

    # Reliable manual signal for artists: same session-level logic as song-level
    _sm_artist = _session_manual_songs(playlist_plays)
    manual_artists = set(
        playlist_plays.loc[
            playlist_plays["play_source"].isin(["artist_browse"])
            | playlist_plays["song_id"].isin(_sm_artist),
            "artist_name",
        ]
    )

    # Three artist-level binge signals — new_discovery excluded because it fires
    # for any artist found in the last quarter and causes too many false positives.
    # A classics or full-album listen hits 0 signals; a new-drop binge hits 2-3.
    scores: dict = {}
    for artist_name, row in agg.iterrows():
        play_count     = int(row["play_count"])
        weighted_plays = float(row["weighted_sum"])
        binge_weighted = float(row["binge_wsum"])

        latest_rel = artist_latest_release.get(artist_name)
        latest_add = artist_latest_added.get(artist_name)
        binge_signals = sum([
            bool(latest_rel and (now - latest_rel).days < 90),
            bool(latest_add and (now - latest_add).days < 30),
            bool(artist_name in manual_artists),
        ])
        binge_threshold    = max(play_count * 0.8, 5.0)
        raw_binge          = min(binge_weighted / binge_threshold, 1.0)
        artist_binge_score = round(raw_binge, 4) if (play_count >= 3 and binge_signals >= 2) else 0.0

        scores[artist_name] = {
            "artist_fatigue":     round(min(weighted_plays / 5.0, 1.0), 4),
            "artist_binge_score": artist_binge_score,
        }
    return scores


# ── Artist completion rate ────────────────────────────────────────────────────

def compute_artist_completion_rates(df: pd.DataFrame) -> dict:
    """
    Per-artist completion rate with dominant-song exclusion.

    A song is "dominant" if it accounts for > ARTIST_DOMINANCE_THRESHOLD of the
    artist's total known-outcome plays. Dominant songs are outlier binges that
    don't generalize to the rest of the artist's catalog — we exclude them so
    one beloved song doesn't artificially inflate the completion rate for every
    other song by that artist.

    Confidence scales with unique non-dominant songs heard, not raw play count.
    Hearing 10 songs by Drake twice each (n_unique=10) is much stronger evidence
    than hearing 1 Drake song 20 times (n_unique=1). Full confidence at 5+
    unique songs; below that, the signal is shrunk toward neutral (0.5).

    Returns: {artist_name: {comp_rate, n_unique, confidence}}
    """
    known = df[df["inferred_skip"].isin({"full", "partial", "skip"})].copy()
    if known.empty:
        return {}

    song_plays = (
        known.groupby(["artist_name", "song_id"])
        .agg(
            n_plays=("song_id",       "count"),
            n_comp =("inferred_skip", lambda x: x.isin({"full", "partial"}).sum()),
        )
        .reset_index()
    )

    result = {}
    for artist, group in song_plays.groupby("artist_name"):
        if not artist:
            continue
        total = group["n_plays"].sum()
        dominant_ids = set(
            group.loc[group["n_plays"] / total > ARTIST_DOMINANCE_THRESHOLD, "song_id"]
        )
        non_dom = group[~group["song_id"].isin(dominant_ids)]

        n_unique = len(non_dom)
        if n_unique == 0:
            result[artist] = {"comp_rate": 0.5, "n_unique": 0, "confidence": 0.0}
            continue

        nd_plays = non_dom["n_plays"].sum()
        nd_comp  = non_dom["n_comp"].sum()
        comp_rate  = round(nd_comp / nd_plays, 4) if nd_plays > 0 else 0.5
        confidence = round(min(n_unique / 5.0, 1.0), 4)

        result[artist] = {
            "comp_rate":  comp_rate,
            "n_unique":   n_unique,
            "confidence": confidence,
        }

    return result


# ── Coverage debt ─────────────────────────────────────────────────────────────

def compute_coverage_debt(df: pd.DataFrame, conn) -> dict:
    """
    Coverage debt measures how buried a song is relative to its playlist size,
    in units of plays (not time).

    debt = plays_since_last_heard / playlist_size

    Debt is cleared to zero when the song has been queue-skipped
    >= COVERAGE_DEBT_SKIP_REVOKE consecutive times from smartshuffle or random
    baseline sessions with no completion since. We surfaced it, the user declined
    repeatedly — stop pushing it.

    Note: songs completed recently naturally have near-zero debt via the plays_since
    formula (plays_since ≈ 0 right after completion), so no explicit completion-
    clearing is needed. Explicitly zeroing on completion would incorrectly suppress
    debt for songs completed months ago that should be surfaced again.
    """
    total_plays = len(df)

    df_sorted = df.sort_values("played_at").reset_index(drop=True)
    df_sorted["_pos"] = np.arange(len(df_sorted))
    last_play_index = df_sorted.groupby("song_id")["_pos"].max().to_dict()

    # ── Debt-clearing: repeated queued skips without a completion since ──────────
    # A song's debt is cleared when it has been skipped >= COVERAGE_DEBT_SKIP_REVOKE
    # times from queued sessions (smartshuffle or random baseline) with no queued
    # completion since the last skip.  "Skipped from a queue" means either:
    #   • duration-skip: play_source = *_queued AND inferred_skip = 'skip' (in plays)
    #   • queue-skip:    song bypassed entirely (queue_skips table, LIS-inferred)
    # We count PUSH EVENTS (not individual play rows) to avoid inflation from
    # the same push appearing multiple times.
    #
    # Algorithm:
    #   1. Build a timeline of (timestamp, event_type) per song where event_type
    #      is 'skip' (queue-skip or duration-skip from queued session) or
    #      'completion' (full/partial from any source).
    #   2. Walk backwards. Count consecutive skips from the end until hitting a
    #      completion or running out of events. If count >= threshold → clear.
    queued_sources = {"smartshuffle_queued", "random_baseline_queued"}

    # Duration skips from queued plays → use pushed_at ≈ played_at for ordering
    dur_skips_q = df_sorted[
        df_sorted["play_source"].isin(queued_sources)
        & (df_sorted["inferred_skip"] == "skip")
    ][["song_id", "played_at"]].copy()
    dur_skips_q["_ev"] = "skip"

    # Completions from any source (organic or queued) — marks "was enjoyed"
    completions_q = df_sorted[
        df_sorted["inferred_skip"].isin({"full", "partial"})
    ][["song_id", "played_at"]].copy()
    completions_q["_ev"] = "completion"

    # Queue-skips: songs bypassed in the queue entirely.
    # Use qp.pushed_at as the timestamp (proxy for when the skip "happened").
    qs_raw = conn.execute("""
        SELECT qs.song_id, qp.pushed_at
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
    """).fetchall()
    if qs_raw:
        qs_df = pd.DataFrame(qs_raw, columns=["song_id", "played_at"])
        qs_df["played_at"] = pd.to_datetime(qs_df["played_at"], format="ISO8601", utc=True)
        # Deduplicate: one skip event per (song_id, push) — already distinct by design
        qs_df["_ev"] = "skip"
    else:
        qs_df = pd.DataFrame(columns=["song_id", "played_at", "_ev"])

    # Combine all events
    timeline = pd.concat([dur_skips_q, completions_q, qs_df], ignore_index=True)
    timeline  = timeline.sort_values("played_at")

    def _consec_skips_from_end(events: list[str]) -> int:
        """Count consecutive 'skip' events from the tail before any 'completion'."""
        n = 0
        for ev in reversed(events):
            if ev == "skip":
                n += 1
            else:  # completion
                break
        return n

    if not timeline.empty:
        consec_queued_skips = (
            timeline.groupby("song_id")["_ev"]
            .apply(list)
            .apply(_consec_skips_from_end)
            .to_dict()
        )
    else:
        consec_queued_skips = {}

    # ── Playlist metadata ─────────────────────────────────────────────────────
    playlist_sizes = dict(conn.execute("""
        SELECT playlist_id, COUNT(song_id) FROM playlist_tracks GROUP BY playlist_id
    """).fetchall())

    song_playlists = defaultdict(list)
    for song_id, playlist_id in conn.execute(
        "SELECT song_id, playlist_id FROM playlist_tracks"
    ).fetchall():
        song_playlists[song_id].append(playlist_id)

    all_songs = conn.execute("""
        SELECT DISTINCT pt.song_id, s.song_name, s.artist_name
        FROM playlist_tracks pt
        LEFT JOIN songs s ON pt.song_id = s.song_id
    """).fetchall()

    debt_map = {}
    for song_id, song_name, artist_name in all_songs:
        playlists = song_playlists.get(song_id, [])
        if not playlists:
            continue

        max_playlist_size = max(playlist_sizes.get(pl, 1) for pl in playlists)

        # Debt is cleared only when the song has been queue-skipped repeatedly
        # without a completion since. This stops us from resurfacing songs the
        # user has been given chances at and consistently declined.
        _consec = consec_queued_skips.get(song_id, 0)
        debt_cleared = _consec >= COVERAGE_DEBT_SKIP_REVOKE

        if debt_cleared:
            coverage_debt = 0.0
        elif song_id not in last_play_index:
            coverage_debt = float(total_plays) / max(max_playlist_size, 1)
        else:
            plays_since   = total_plays - 1 - last_play_index[song_id]
            coverage_debt = plays_since / max(max_playlist_size, 1)

        stale = coverage_debt >= COVERAGE_DEBT_THRESHOLD

        debt_map[song_id] = {
            "song_name":      song_name,
            "artist_name":    artist_name,
            "coverage_debt":  round(coverage_debt, 3),
            "stale":          stale,
            "skip_revoked":   debt_cleared,
        }

    return debt_map


# ── Binge episode tracking ────────────────────────────────────────────────────

def _build_daily_plays(song_plays: pd.DataFrame, start: date, end: date) -> list:
    """Count plays per day from start to end inclusive, as a flat int array."""
    n_days = (end - start).days + 1
    if song_plays.empty or "play_date" not in song_plays.columns:
        return [0] * n_days
    mask   = (song_plays["play_date"] >= start) & (song_plays["play_date"] <= end)
    counts = song_plays.loc[mask, "play_date"].value_counts()
    return [int(counts.get(start + timedelta(days=i), 0)) for i in range(n_days)]


def record_binge_episodes(fatigue_scores: dict, df: pd.DataFrame, conn) -> None:
    """
    Maintains the binge_episodes table. Called each phase2.py run.

    - Opens a new episode the first time binge_score > 0 for a song.
    - Updates daily_plays for open episodes each run.
    - Closes an episode when last play was >= BINGE_END_GAP_DAYS ago
      (binge naturally wound down — no explicit plays for N days).
    """
    today = datetime.now(timezone.utc).date()

    playlist_plays = df[df["play_source"] != "jam_excluded"].copy()
    playlist_plays["play_date"] = playlist_plays["played_at"].dt.date

    # Load all open episodes indexed by song_id
    open_episodes: dict = {}
    for ep_id, song_id, start_str, dp_json in conn.execute(
        "SELECT episode_id, song_id, start_date, daily_plays "
        "FROM binge_episodes WHERE end_date IS NULL"
    ).fetchall():
        open_episodes[song_id] = {
            "episode_id": ep_id,
            "start_date": date.fromisoformat(start_str),
            "daily_plays": json.loads(dp_json),
        }

    opened = closed = updated = 0

    for song_id, scores in fatigue_scores.items():
        binge_score     = scores.get("binge_score", 0.0)
        artist_name     = scores.get("artist_name")
        last_played_str = scores.get("last_played")

        last_played: date | None = None
        if last_played_str:
            try:
                last_played = datetime.fromisoformat(last_played_str).date()
            except Exception:
                pass

        days_since_played = (today - last_played).days if last_played else 999
        song_plays = playlist_plays[playlist_plays["song_id"] == song_id]

        if binge_score > 0.0:
            if song_id not in open_episodes:
                daily_plays = _build_daily_plays(song_plays, today, today)
                peak = max(daily_plays) if daily_plays else 0
                conn.execute(
                    "INSERT INTO binge_episodes "
                    "(song_id, artist_name, start_date, daily_plays, peak_plays) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (song_id, artist_name, today.isoformat(), json.dumps(daily_plays), peak),
                )
                opened += 1
            else:
                ep = open_episodes[song_id]
                daily_plays = _build_daily_plays(song_plays, ep["start_date"], today)
                peak = max(daily_plays) if daily_plays else 0
                conn.execute(
                    "UPDATE binge_episodes SET daily_plays=?, peak_plays=? WHERE episode_id=?",
                    (json.dumps(daily_plays), peak, ep["episode_id"]),
                )
                updated += 1

        elif song_id in open_episodes and (
            days_since_played >= BINGE_END_GAP_DAYS
            or (today - open_episodes[song_id]["start_date"]).days >= BINGE_MAX_DAYS
        ):
            ep = open_episodes[song_id]
            daily_plays    = _build_daily_plays(song_plays, ep["start_date"], today)
            peak           = max(daily_plays) if daily_plays else 0
            episode_length = (today - ep["start_date"]).days
            conn.execute(
                "UPDATE binge_episodes "
                "SET end_date=?, daily_plays=?, peak_plays=?, episode_length=? "
                "WHERE episode_id=?",
                (today.isoformat(), json.dumps(daily_plays), peak, episode_length, ep["episode_id"]),
            )
            closed += 1

    conn.commit()
    total_open   = conn.execute("SELECT COUNT(*) FROM binge_episodes WHERE end_date IS NULL").fetchone()[0]
    total_closed = conn.execute("SELECT COUNT(*) FROM binge_episodes WHERE end_date IS NOT NULL").fetchone()[0]
    print(f"  Episodes: {opened} opened, {updated} updated, {closed} closed "
          f"({total_open} active, {total_closed} completed)")


# ── Persist results ───────────────────────────────────────────────────────────

def save_to_db(conn, session_stats, fatigue_scores, coverage_debt,
               artist_fatigue_scores: dict = None,
               artist_comp_rates: dict = None):
    n = now_iso()
    artist_fatigue_scores = artist_fatigue_scores or {}
    artist_comp_rates     = artist_comp_rates     or {}

    conn.executemany("""
        INSERT OR REPLACE INTO sessions
        (session_id, context_label, energy_label, play_count, skip_rate,
         avg_energy_score, start_time, end_time, duration_min, hour_of_day,
         early_skip_rate, skip_latency_mean_ms, stakes_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (s["session_id"], s["context_label"], s["energy_label"],
         s["play_count"], s["skip_rate"], s["avg_energy_score"],
         s["start_time"], s["end_time"], s["duration_min"], s["hour_of_day"],
         s.get("early_skip_rate"), s.get("skip_latency_mean_ms"), s.get("stakes_level"))
        for s in session_stats.values()
    ])

    # Merge fatigue + coverage debt + artist fatigue for song_scores
    all_song_ids = set(fatigue_scores.keys()) | set(coverage_debt.keys())
    song_rows = []
    for song_id in all_song_ids:
        fat    = fatigue_scores.get(song_id, {})
        debt   = coverage_debt.get(song_id, {})
        artist = fat.get("artist_name") or debt.get("artist_name")
        af     = artist_fatigue_scores.get(artist, {})
        ac     = artist_comp_rates.get(artist, {})
        song_rows.append((
            song_id,
            fat.get("song_name") or debt.get("song_name"),
            artist,
            fat.get("fatigue", 0.0),
            fat.get("binge_score"),
            af.get("artist_fatigue"),
            af.get("artist_binge_score"),
            fat.get("play_count", 0),
            fat.get("last_played"),
            fat.get("skip_rate"),
            fat.get("pattern"),
            debt.get("coverage_debt"),
            int(debt.get("stale", True)),
            fat.get("days_since_added"),
            fat.get("play_rate_per_week"),
            n,
            fat.get("binge_velocity"),
            fat.get("binge_skip_rate"),
            fat.get("evergreen_score", 0.0),
            ac.get("comp_rate",  0.5),
            ac.get("confidence", 0.0),
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO song_scores
        (song_id, song_name, artist_name, fatigue, binge_score,
         artist_fatigue, artist_binge_score,
         play_count, last_played, skip_rate, pattern, coverage_debt, stale,
         days_since_added, play_rate_per_week, updated_at,
         binge_velocity, binge_skip_rate, evergreen_score,
         artist_comp_rate, artist_comp_conf)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, song_rows)

    # Write daily binge score snapshot for velocity tracking
    today = datetime.now(timezone.utc).date().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO binge_score_history (song_id, date, binge_score) VALUES (?, ?, ?)",
        [(sid, today, fat.get("binge_score", 0.0))
         for sid, fat in fatigue_scores.items()
         if fat.get("binge_score", 0.0) > 0]
    )

    conn.commit()


# ── Engagement baselines ─────────────────────────────────────────────────────

def compute_baseline_engagement(session_stats: dict, conn):
    """
    Aggregates session-level skip stats by time bucket.

    Produces the engagement_baselines table, which phase34.py reads to decide
    how aggressively to weight energy_match vs coverage at queue generation time.

    Interpretation:
      high stakes bucket   → user typically searches hard in this context
                             → raise energy_match weight, lower coverage
      low stakes bucket    → user passively listens here
                             → raise coverage (good time to surface stale songs)
      normal               → use learned/default weights as-is
    """
    from collections import Counter
    bucket_sessions = defaultdict(list)
    for s in session_stats.values():
        bucket = get_time_bucket(s["hour_of_day"])
        bucket_sessions[bucket].append(s)

    now_dt  = datetime.now(timezone.utc)
    now_str = now_iso()
    for bucket, sessions in bucket_sessions.items():
        half_life = BUCKET_HALF_LIFE.get(bucket, 90)
        decay_k   = np.log(2) / half_life

        weights = []
        for s in sessions:
            start = datetime.fromisoformat(s["start_time"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            days_ago = (now_dt - start).total_seconds() / 86400
            weights.append(np.exp(-decay_k * days_ago))
        weights_arr = np.array(weights)

        skip_rates       = np.array([s["skip_rate"] for s in sessions])
        early_skip_rates = np.array([s.get("early_skip_rate", 0.0) for s in sessions])

        baseline_skip = float(np.average(skip_rates,       weights=weights_arr))
        early_skip    = float(np.average(early_skip_rates, weights=weights_arr))

        # Weighted modal: stakes level with the highest summed weight
        stakes_weight: dict = defaultdict(float)
        for s, w in zip(sessions, weights):
            stakes_weight[s.get("stakes_level", "normal")] += w
        typical_stakes = max(stakes_weight, key=stakes_weight.get)

        conn.execute("""
            INSERT OR REPLACE INTO engagement_baselines
            (time_bucket, baseline_skip_rate, early_skip_rate, typical_stakes,
             session_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            bucket,
            round(baseline_skip, 4),
            round(early_skip,    4),
            typical_stakes,
            len(sessions),
            now_str,
        ))

    conn.commit()
    print(f"  Engagement baselines updated for {len(bucket_sessions)} time buckets:")
    for bucket, sessions in sorted(bucket_sessions.items()):
        half_life = BUCKET_HALF_LIFE.get(bucket, 90)
        decay_k   = np.log(2) / half_life
        weights   = []
        for s in sessions:
            start = datetime.fromisoformat(s["start_time"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            days_ago = (now_dt - start).total_seconds() / 86400
            weights.append(np.exp(-decay_k * days_ago))
        w_skip = float(np.average([s["skip_rate"] for s in sessions], weights=weights))
        stakes_w: dict = defaultdict(float)
        for s, w in zip(sessions, weights):
            stakes_w[s.get("stakes_level", "normal")] += w
        modal = max(stakes_w, key=stakes_w.get)
        print(f"    {bucket:12s}  skip_rate={w_skip:.1%}  typical_stakes={modal}"
              f"  (n={len(sessions)} sessions, half_life={half_life}d)")


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(session_stats, fatigue_scores, coverage_debt, df):
    print(f"\n── Session summary ({len(session_stats)} sessions) ──")
    label_counts = defaultdict(int)
    for s in session_stats.values():
        label_counts[s["context_label"]] += 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print(f"\n── Top 10 highest fatigue songs ──")
    top = sorted(fatigue_scores.items(), key=lambda x: x[1]["fatigue"], reverse=True)[:10]
    for _, s in top:
        print(f"  [{s['fatigue']:.2f}] {s['song_name']} — {s['artist_name']} "
              f"({s['play_count']} plays, {s['pattern']})")

    print(f"\n── Songs showing growing skip trend ──")
    tired = [s for s in fatigue_scores.values() if s["pattern"] == "growing_tired"]
    if tired:
        for s in tired:
            print(f"  {s['song_name']} — {s['artist_name']} (skip rate: {s['skip_rate']:.0%})")
    else:
        print("  None yet — need more data to detect trends")

    stale_count   = sum(1 for d in coverage_debt.values() if d["stale"])
    cleared_count = sum(1 for d in coverage_debt.values() if d["coverage_debt"] == 0.0
                        and d.get("song_name"))  # 0.0 can also mean recently played
    print(f"\n── Coverage debt ──")
    print(f"  Stale songs (debt >= {COVERAGE_DEBT_THRESHOLD}x): {stale_count} / {len(coverage_debt)}")
    print(f"  Skip-revoked (debt cleared after {COVERAGE_DEBT_SKIP_REVOKE}+ queued skips): "
          f"{sum(1 for d in coverage_debt.values() if d.get('skip_revoked'))}")
    most_buried = sorted(
        [(sid, d) for sid, d in coverage_debt.items() if d["stale"]],
        key=lambda x: x[1]["coverage_debt"], reverse=True
    )[:5]
    if most_buried:
        print(f"  Most buried:")
        for _, d in most_buried:
            print(f"    [{d['coverage_debt']:.1f}x] {d['song_name']} — {d['artist_name']}")

    # Tag coverage report
    tag_rows = conn_global.execute(
        "SELECT fetch_source, COUNT(*) FROM song_tags GROUP BY fetch_source"
    ).fetchall()
    print(f"\n── Last.fm tag coverage ──")
    for source, count in tag_rows:
        print(f"  {source}: {count} songs")

    songs_with_energy = conn_global.execute(
        "SELECT COUNT(*) FROM song_tags WHERE energy_score != 0.0"
    ).fetchone()[0]
    print(f"  Songs with energy signal: {songs_with_energy}")

    # Sample tags from a few songs
    print(f"\n── Sample tags (5 songs) ──")
    samples = conn_global.execute(
        "SELECT song_id, top_tags, energy_score FROM song_tags WHERE top_tags != '[]' LIMIT 5"
    ).fetchall()
    for song_id, top_tags_raw, energy in samples:
        song_name = df[df["song_id"] == song_id]["song_name"].iloc[0] \
                    if song_id in df["song_id"].values else song_id
        tags = json.loads(top_tags_raw)
        print(f"  {song_name}: {tags} [energy: {energy:+.2f}]")


# ── Main ──────────────────────────────────────────────────────────────────────

conn_global = None  # module-level so print_summary can access

def main():
    global conn_global
    conn = sqlite3.connect(DB_PATH)
    conn_global = conn

    if not LASTFM_API_KEY:
        print("WARNING: LASTFM_API_KEY not set in .env — tag fetch will be skipped.")
        print("         Energy scores will default to 0.0 (neutral).")
        print()

    print("=== SmartShuffle Phase 2 — Behavioral Modeling ===")
    print(f"Started: {now_iso()}\n")

    init_phase2_tables(conn)

    print("Loading plays...")
    # Load without tags first to get song list for tag fetch
    df_bare = pd.read_sql_query("""
        SELECT DISTINCT song_id, song_name, artist_name FROM plays
        WHERE play_source != 'jam_excluded'
    """, conn)

    print("Fetching Last.fm tags for played songs...")
    fetch_all_tags(conn, df_bare)

    # Also fetch tags for playlist songs that haven't been played yet.
    # This fixes the energy=0.0 problem for unplayed songs and gives phase34.py
    # real energy scores for songs even before they appear in play history.
    print("Fetching Last.fm tags for unplayed playlist songs...")
    df_playlist = pd.read_sql_query("""
        SELECT DISTINCT pt.song_id, s.song_name, s.artist_name
        FROM playlist_tracks pt
        LEFT JOIN songs s ON pt.song_id = s.song_id
        WHERE s.song_name IS NOT NULL
    """, conn)
    fetch_all_tags(conn, df_playlist)

    print("Recomputing energy scores from stored tags...")
    recompute_energy_scores(conn)

    print("Blending interjection-observed energy signals...")
    blend_interjection_energy(conn)

    print("Loading full play dataset with tags...")
    df = load_plays(conn)
    print(f"  {len(df)} plays loaded")

    print("Segmenting sessions...")
    df = segment_sessions(df)

    print("Inferring context labels...")
    df, session_stats = label_sessions(df)

    print("Imputing missing energy from session context...")
    impute_missing_energy(conn, df, session_stats)

    print("Computing fatigue scores...")
    fatigue_scores = compute_fatigue_scores(df, conn)

    print("Computing artist fatigue scores...")
    artist_fatigue_scores = compute_artist_fatigue_scores(df, conn)

    print("Computing artist completion rates...")
    artist_comp_rates = compute_artist_completion_rates(df)
    n_with_signal = sum(1 for v in artist_comp_rates.values() if v["confidence"] > 0)
    print(f"  {len(artist_comp_rates)} artists  ({n_with_signal} with useful signal)")

    print("Computing coverage debt...")
    coverage_debt = compute_coverage_debt(df, conn)

    print("Saving to database...")
    save_to_db(conn, session_stats, fatigue_scores, coverage_debt,
               artist_fatigue_scores, artist_comp_rates)

    print("Recording binge episodes...")
    record_binge_episodes(fatigue_scores, df, conn)

    print("Computing engagement baselines...")
    compute_baseline_engagement(session_stats, conn)

    print_summary(session_stats, fatigue_scores, coverage_debt, df)

    conn.close()
    print(f"\nDone: {now_iso()}")


if __name__ == "__main__":
    main()