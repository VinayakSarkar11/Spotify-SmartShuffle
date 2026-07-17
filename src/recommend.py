"""
Phase 3+4 — Vibe Clustering + Queue Orchestrator
=================================================
3. Vibe clustering    — K-means on (vibe_content, vibe_melodic, vibe_bpm) per playlist,
                        auto-selects k via silhouette score
4. Queue orchestrator — greedy constrained ranking combining:
                          3-axis vibe match  (content/melodic/bpm Gaussians)
                          fatigue × binge interaction (binge suppresses penalty)
                          artist completion rate (non-dominant-song signal)
                          coverage debt bonus (stale songs get surfaced)
                          skip penalty (growing_tired gets extra suppression)
                          stakes-aware weight adjustment (per time bucket baseline)
   A/B harness        — blind comparison: SmartShuffle vs random baseline

Run:
    python phase34.py                          # all songs, infer context from clock
    python phase34.py --playlist Chill         # specific playlist
    python phase34.py --playlist Hype --k 3   # force 3 clusters
    python phase34.py --reveal                 # show which queue was which after printing
    python phase34.py --list                   # list available playlists

Requires: smartshuffle.db with phase 2 tables (song_scores, song_tags) populated.

─────────────────────────────────────────────────────────────────────────────
DEFERRED — requires live session loop (Phase 5 FastAPI/Streamlit frontend):

  [ ] Real-time engagement_delta: observe which song was skipped and when during
      the CURRENT session, compare to the historical baseline_skip_rate.
      Currently SessionContext.engagement_delta always returns 0.0.

  [ ] Mid-session queue re-scoring: after each skip, re-rank remaining queue
      positions using updated session context weights. Needs the frontend to
      call back into the orchestrator after each track change.

  [ ] Vibe-searching mode: 2+ consecutive early skips (< 30 s) in the live
      session → pin remaining queue to familiar + tightly energy-matched songs,
      zero coverage debt weight.

  [ ] Skip latency detection: poll /v1/me/player every ~5 s to compute real-time
      play_duration_ms and detect whether a skip was early or late.
    
Ideas:
    1. Playlist generation - once listening trends identified, we can generate new 
    sub-playlists - sequences they like, group playlists by vibe
    2. Ensure queues don't transfer across devices - queue from macbook should not
    be identical to queue on phone when playing the same playlist later

What IS implemented now:
  - SessionContext class: correct interface; engagement_delta is stubbed at 0.0
  - Historical stakes_level per time bucket loaded from engagement_baselines
  - Stakes-aware weight adjustment at queue generation time (static, not live)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import random
import sqlite3
import argparse
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timezone, date
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

_ROOT               = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH             = os.path.join(_ROOT, "data", "smartshuffle.db")
PARAMS_PATH         = os.path.join(_ROOT, "data", "learned_params.json")
SESSION_STATE_PATH  = os.path.join(_ROOT, "data", "session_state.json")
VIBE_PARAMS_PATH    = os.path.join(_ROOT, "data", "vibe_params.json")

# ── Load learned params (written by model.py) ─────────────────────────────────
# Falls back to hardcoded defaults if model.py hasn't been run yet.
def _load_params() -> dict:
    try:
        with open(PARAMS_PATH) as f:
            p = json.load(f)
        trained = p.get("trained_at", "unknown")
        n       = p.get("n_plays", 0)
        alpha   = p.get("alpha", 1.0)
        print(f"[model] Loaded learned params "
              f"(trained {trained[:10]}, n={n} plays, α={alpha:.2f})")
        return p
    except FileNotFoundError:
        print("[model] No learned_params.json found — using hardcoded defaults. "
              "Run model.py to train.")
        return {}

_PARAMS = _load_params()

# ── Vibe params: per-axis sigmas from behavioral data ─────────────────────────
def _load_vibe_params() -> dict:
    try:
        with open(VIBE_PARAMS_PATH) as f:
            vp = json.load(f)
        print(f"[vibe] Loaded vibe params ({vp.get('n_plays', 0)} plays, "
              f"computed {vp.get('computed_at', 'unknown')[:10]})")
        return vp
    except FileNotFoundError:
        print("[vibe] No vibe_params.json — run learn_vibe_params.py for data-derived sigmas.")
        return {}

_VIBE_PARAMS = _load_vibe_params()
# Per-axis sigma defaults used when vibe_params.json is absent.
_DEFAULT_VIBE_SIGMAS: dict = {"content": 0.30, "melodic": 0.30, "bpm": 0.30}

# ── Context → target energy ───────────────────────────────────────────────────
# Values learned by model.py from completion-weighted play history.
# Hardcoded defaults are used until model.py has been run.
_DEFAULT_CONTEXT_TARGETS = {
    "late_night":  -0.5,
    "morning":      0.4,
    "afternoon":    0.2,
}
TIME_BUCKET_ENERGY = _PARAMS.get("context_targets", _DEFAULT_CONTEXT_TARGETS)

# Behavioral context targets are computed from per-context energy distributions
# (behavioral scale: late_night ≈ -0.35, afternoon ≈ +0.15, morning ≈ +0.35).
# Used instead of TIME_BUCKET_ENERGY when per-context behavioral energy is active.
_BEHAVIORAL_CONTEXT_TARGETS = _PARAMS.get("behavioral_context_targets", _DEFAULT_CONTEXT_TARGETS)

def current_time_bucket() -> str:
    h = datetime.now().hour
    if h >= 21 or h < 6: return "late_night"
    if h < 11: return "morning"
    return "afternoon"

def get_target_energy(time_bucket: str, playlist_id: str = None) -> float:
    """
    Returns the energy target for the current context.
    Prefers the per-(playlist, time_bucket) learned target when available —
    so a Hype queue at night gets a different target than a Chill queue at night.
    Falls back to the global per-bucket target, then 0.0.
    """
    if playlist_id and playlist_id in _PLAYLIST_CONTEXT_TARGETS:
        pl = _PLAYLIST_CONTEXT_TARGETS[playlist_id]
        if time_bucket in pl:
            return pl[time_bucket]
    return TIME_BUCKET_ENERGY.get(time_bucket, 0.0)

def get_target_vibe(playlist_id: str | None) -> dict | None:
    """
    Returns the learned vibe target for a playlist, or None if not yet available.
    Keys: 'content', 'melodic', 'bpm' — each in [-1, 1].
    """
    if playlist_id and playlist_id in _PLAYLIST_VIBE_TARGETS:
        return _PLAYLIST_VIBE_TARGETS[playlist_id]
    return None

# ── Ranking weights ───────────────────────────────────────────────────────────
# Alpha-blended: formula weights * alpha + learned weights * (1-alpha).
# model.py handles the blend; phase34.py just loads the result.
_DEFAULT_WEIGHTS = {
    "energy_match":   0.00,  # disabled (Option B) — redundant with 3-axis vibe_match
    "vibe_match":     0.40,  # was 0.20; absorbs the reallocated energy_match weight
    "fatigue":        0.25,
    "artist_fatigue": 0.10,
    "coverage":       0.20,
    "skip":           0.05,
    "recency":        0.30,  # short-term: 1.0 at 0 h, 0.0 at 24 h
    "binge_boost":    0.15,
    "artist_comp":    0.0,   # disabled: signal runs backwards on current data
}
WEIGHTS = _PARAMS.get("weights", _DEFAULT_WEIGHTS)
WEIGHTS.setdefault("vibe_match",   _DEFAULT_WEIGHTS["vibe_match"])
WEIGHTS.setdefault("binge_boost",  _DEFAULT_WEIGHTS["binge_boost"])
WEIGHTS["energy_match"] = 0.0  # Option B: always disabled regardless of trained value
WEIGHTS["artist_comp"]  = 0.0  # disabled until more data validates signal direction

# Learned binge arc: list of BINGE_SHAPE_LENGTH floats in [0,1].
# None until model.py has >= MIN_BINGE_EPISODES completed episodes to learn from.
# When present: adjusted_binge_score = binge_score × shape[days_since_binge_start]
_BINGE_SHAPE: list | None = _PARAMS.get("binge_shape")

# Per-playlist context targets (playlist_id → {time_bucket → energy}).
# Preferred over TIME_BUCKET_ENERGY when available.
_PLAYLIST_CONTEXT_TARGETS: dict = _PARAMS.get("playlist_context_targets", {})

# Per-playlist vibe targets (playlist_id → {content, melodic, bpm}).
# Learned by train.py from completion-weighted play history.
# None for any playlist not yet in learned_params.json.
_PLAYLIST_VIBE_TARGETS: dict = _PARAMS.get("playlist_vibe_targets", {})

ARTIST_FATIGUE_WINDOW    = 4    # penalise same artist if seen in last N queue positions
ARTIST_FATIGUE_FACTOR    = 0.10 # score multiplier applied to recently-queued artists
BINGE_ALLOW_THRESHOLD    = 0.60 # artist_binge_score above this lifts the back-to-back block
EVERGREEN_MAX_LIFT       = 0.60 # max fraction of fatigue penalty that evergreen score can cancel
ARTIST_CAP_MULTIPLIER    = 2.2  # artists may take up to N× their pool-share of queue slots
RECENTLY_QUEUED_PENALTY  = 0.50 # max score penalty for songs from the last generated queue;
                                 # decays linearly to ~0 for songs at the tail of that queue
SCORE_TEMPERATURE        = 0.20 # softmax sampling temperature; controls how strictly the
                                 # score ordering is enforced vs. randomness among candidates.
                                 # Lower → more deterministic, higher → more shuffled.
                                 # At 0.20: songs with score diff <0.05 are roughly equally
                                 # likely; songs 0.30 apart still favour the better one ~4.5×.

# ── Adaptive bandwidth (σ) ────────────────────────────────────────────────────
# Data-derived baseline sigmas (~0.30 per axis) replace the old hardcoded 0.50.
# High session skip rate tightens σ to 50% of baseline (e.g. 0.30 → 0.15).
# The same multiplier applies to both the 1D energy Gaussian and all vibe axes.
_SIGMA_TIGHTEN_ABOVE  = 0.30   # start tightening above this skip rate
_SIGMA_TIGHT_AT       = 0.80   # fully tight at or above this skip rate
_SIGMA_MIN_MULTIPLIER = 0.50   # at max skip, sigma = 50% of data baseline

def _sigma_multiplier(recent_skip_rate: float) -> float:
    """1.0 at low skip rate → _SIGMA_MIN_MULTIPLIER at high skip rate."""
    t = max(0.0, min(1.0,
        (recent_skip_rate - _SIGMA_TIGHTEN_ABOVE) / (_SIGMA_TIGHT_AT - _SIGMA_TIGHTEN_ABOVE)
    ))
    return round(1.0 - (1.0 - _SIGMA_MIN_MULTIPLIER) * t, 3)

def _get_vibe_sigmas(time_bucket: str, recent_skip_rate: float = 0.0) -> dict:
    """Per-axis sigmas from vibe_params.json, tightened by session skip rate."""
    baseline = _VIBE_PARAMS.get("vibe_sigmas", {}).get(time_bucket, _DEFAULT_VIBE_SIGMAS)
    mult = _sigma_multiplier(recent_skip_rate)
    return {ax: round(v * mult, 4) for ax, v in baseline.items()}

def _read_session_skip_rate(time_bucket: str) -> float:
    """Read recent_skip_rate from session_state.json for the matching bucket."""
    try:
        with open(SESSION_STATE_PATH) as f:
            state = json.load(f)
        if state.get("time_bucket") == time_bucket and int(state.get("plays_seen", 0)) >= 3:
            return float(state.get("recent_skip_rate", 0.0))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return 0.0

# ── Cross-refill session vibe drift ──────────────────────────────────────────
# When the watcher generates a rolling refill, it reads session_vibe_*_mean from
# session_state.json (mean vibe of completed songs so far) and blends it with the
# learned playlist target.  More completions = more weight on session evidence.
# This lets long sessions drift toward the actual vibe the user is settling into.
_SESSION_VIBE_MAX_DRIFT = 0.40  # max fraction session can shift the target
_SESSION_VIBE_CREDIBLE  = 10    # completions at which drift reaches the maximum

def _apply_session_vibe_drift(target_vibe: dict | None, time_bucket: str) -> dict | None:
    """
    Blends the learned playlist vibe target with the mean vibe of completed plays
    so far this session. Only activates after 3 completions and only when a vibe
    target exists. Returns target_vibe unchanged if no drift data is available.
    """
    if target_vibe is None:
        return target_vibe
    try:
        with open(SESSION_STATE_PATH) as f:
            state = json.load(f)
        if state.get("time_bucket") != time_bucket:
            return target_vibe
        n = int(state.get("session_vibe_n", 0))
        if n < 3:
            return target_vibe
        drift_w = min(n / _SESSION_VIBE_CREDIBLE, _SESSION_VIBE_MAX_DRIFT)
        blended = {}
        for ax in ("content", "melodic", "bpm"):
            sess_v = state.get(f"session_vibe_{ax}_mean")
            if sess_v is None:
                return target_vibe  # incomplete state — skip drift
            blended[ax] = round(
                (1.0 - drift_w) * target_vibe[ax] + drift_w * float(sess_v), 4
            )
        print(
            f"  Session vibe drift: {n} completions  w={drift_w:.2f}"
            f"  c {target_vibe['content']:+.3f}→{blended['content']:+.3f}"
            f"  m {target_vibe['melodic']:+.3f}→{blended['melodic']:+.3f}"
            f"  bpm {target_vibe['bpm']:+.3f}→{blended['bpm']:+.3f}"
        )
        return blended
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return target_vibe

# ── Within-queue session continuity ──────────────────────────────────────────
# After each pick the effective energy target drifts slightly toward recent picks,
# creating a road rather than oscillating around a fixed context target.
_SESSION_DRIFT_WEIGHT    = 0.30  # max pull of recent picks on the context target
_SESSION_DRIFT_MIN_PICKS = 2     # picks before drift activates

# ── Engagement baselines ─────────────────────────────────────────────────────

def load_engagement_baselines(conn) -> dict:
    """
    Loads per-time-bucket engagement baselines written by phase2.py.
    Returns {} if phase2 hasn't been run yet or table doesn't exist.

    Schema: {time_bucket: {baseline_skip_rate, early_skip_rate, typical_stakes, session_count}}
    """
    try:
        rows = conn.execute("""
            SELECT time_bucket, baseline_skip_rate, early_skip_rate,
                   typical_stakes, session_count
            FROM engagement_baselines
        """).fetchall()
        return {
            row[0]: {
                "baseline_skip_rate": row[1] or 0.0,
                "early_skip_rate":    row[2] or 0.0,
                "typical_stakes":     row[3] or "normal",
                "session_count":      row[4] or 0,
            }
            for row in rows
        }
    except sqlite3.OperationalError:
        return {}


class SessionContext:
    """
    Tracks session-level engagement state and adjusts scoring weights accordingly.

    Currently only the historical path is implemented (stakes from engagement_baselines).
    The real-time path (engagement_delta, vibe-searching) is stubbed — it requires
    the Phase 5 frontend to report skip events back into this object mid-session.

    Usage:
        ctx = SessionContext(time_bucket, baselines.get(time_bucket, {}))
        weights = ctx.adjusted_weights(base_weights)
        queue = generate_queue(df, context, weights=weights, ...)

    When frontend is live, call ctx.record_play(song_id, play_duration_ms, skipped)
    after each track change to update engagement_delta and re-score the queue.
    """

    def __init__(self, time_bucket: str, baseline: dict):
        self.time_bucket = time_bucket
        self.baseline    = baseline          # from engagement_baselines table
        self.songs_played = 0
        self.skips        = 0
        self.early_skips  = 0               # < 30 s plays — needs frontend

    # ── DEFERRED: call this after each track change once frontend exists ───────
    def record_play(self, _song_id: str, play_duration_ms: float | None, skipped: bool):
        """Update live session state. No-op until frontend wires this up."""
        self.songs_played += 1
        if skipped:
            self.skips += 1
            if play_duration_ms and play_duration_ms < 30_000:
                self.early_skips += 1

    @property
    def session_skip_rate(self) -> float:
        if self.songs_played == 0:
            return 0.0
        return self.skips / self.songs_played

    @property
    def engagement_delta(self) -> float:
        """
        Positive = skipping more than usual (vibe-searching or bad queue).
        Negative = skipping less than usual (flowing well or passive listening).

        Reads from session_state.json written by session_watcher.py when the
        watcher is running alongside a play session. Falls back to the in-object
        live counter (which is 0.0 until a frontend wires up record_play).
        """
        try:
            with open(SESSION_STATE_PATH) as f:
                state = json.load(f)
            # Only trust the file if it's for the same time bucket and has enough data
            if state.get("time_bucket") == self.time_bucket and state.get("plays_seen", 0) >= 3:
                return float(state["engagement_delta"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        # Fall back to in-object counter only if record_play() has been called
        if self.songs_played >= 3:
            return self.session_skip_rate - self.baseline.get("baseline_skip_rate", 0.0)
        return 0.0

    def adjusted_weights(self, base_weights: dict) -> dict:
        """
        Adjusts scoring weights based on live session signal + historical stakes.

        Live signal (from session_watcher via session_state.json) takes precedence
        once enough plays have been seen. Asymmetric rules:

          plays_seen < 7        → not enough data; fall back to historical stakes
          recent_delta > +0.15  → high: raise energy_match, lower coverage
          ever_high_skip = True → normal: recovered sessions don't earn coverage reward
          plays_seen >= 20 AND overall low AND never high → low: increase coverage
          otherwise             → normal

        The ever_high_skip flag is a one-way ratchet written by session_watcher.
        It ensures that finding a good run after a rough start doesn't incorrectly
        signal "the queue is working — explore more."
        """
        # Read live session state
        plays_seen     = 0
        overall_delta  = 0.0
        recent_rate    = 0.0
        ever_high_skip = False
        try:
            with open(SESSION_STATE_PATH) as f:
                state = json.load(f)
            if state.get("time_bucket") == self.time_bucket:
                plays_seen     = int(state.get("plays_seen", 0))
                overall_delta  = float(state.get("engagement_delta", 0.0))
                recent_rate    = float(state.get("recent_skip_rate", 0.0))
                ever_high_skip = bool(state.get("ever_high_skip", False))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        baseline_rate = self.baseline.get("baseline_skip_rate", 0.0)
        recent_delta  = recent_rate - baseline_rate

        if plays_seen >= 7:
            # Live signal available — apply asymmetric rules
            if recent_delta > 0.15 or overall_delta > 0.15:
                stakes = "high"
            elif ever_high_skip:
                # Skip rate recovered but session had a rough start — no coverage reward
                stakes = "normal"
            elif plays_seen >= 20 and overall_delta < -0.05:
                # Both batches had low skip rate with no prior high-skip episode
                stakes = "low"
            else:
                stakes = "normal"
        else:
            # Not enough live data — fall back to historical stakes
            stakes = self.baseline.get("typical_stakes", "normal")
            n = self.baseline.get("session_count", 0)
            if n < 3:
                return dict(base_weights)

        if stakes == "normal":
            return dict(base_weights)

        w = dict(base_weights)
        if stakes == "high":
            w["vibe_match"] = min(w["vibe_match"] + 0.10, 0.70)
            w["coverage"]   = max(w["coverage"]   - 0.10, 0.05)
        elif stakes == "low":
            w["coverage"]   = min(w["coverage"]   + 0.10, 0.40)
            w["vibe_match"] = max(w["vibe_match"] - 0.05, 0.20)

        total = sum(w.values())
        return {k: round(v / total, 4) for k, v in w.items()}


# ── Clustering feature: vibe axes ─────────────────────────────────────────────
# Songs are clustered on their 3-axis behavioral vibe scores (vibe_content,
# vibe_melodic, vibe_bpm) computed by score_vibes.py.  Songs without vibe scores
# yet are filled with 0 (neutral center) so they don't distort cluster geometry.
#
# Legacy tag-based genre_score and RAP_TAGS/MELODIC_TAGS are kept for reference
# but no longer drive clustering.

def cluster_label(vibe_content: float, vibe_bpm: float) -> str:
    """
    Human-readable label from the two most discriminating vibe axes.
    content: +1 = rap-heavy, -1 = melodic/vocal.
    bpm:     +1 = fast/hype, -1 = slow/chill.
    """
    tempo = "hype" if vibe_bpm > 0.2 else ("chill" if vibe_bpm < -0.2 else "mid")
    style = "rap"  if vibe_content > 0.2 else ("melodic" if vibe_content < -0.2 else "mixed")
    return f"{tempo}_{style}"

# ── DB init ───────────────────────────────────────────────────────────────────

def init_tables(conn):
    # Migrate song_clusters from energy/genre schema to vibe schema if needed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(song_clusters)").fetchall()}
    if existing_cols and "energy_score" in existing_cols and "vibe_content" not in existing_cols:
        conn.execute("DROP TABLE song_clusters")
        conn.commit()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS song_clusters (
            song_id       TEXT,
            playlist_id   TEXT,
            cluster_id    INTEGER,
            cluster_label TEXT,
            vibe_content  REAL,
            vibe_melodic  REAL,
            vibe_bpm      REAL,
            PRIMARY KEY (song_id, playlist_id)
        );

        CREATE TABLE IF NOT EXISTS queues (
            queue_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT,
            context      TEXT,
            algorithm    TEXT,   -- 'smartshuffle' | 'baseline'
            playlist_id  TEXT,
            songs        TEXT,   -- JSON: [{song_id, song_name, artist_name, score, cluster_label}]
            ab_label     TEXT    -- 'A' | 'B', randomly assigned for blind eval
        );

        CREATE TABLE IF NOT EXISTS queue_ratings (
            rating_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id    INTEGER,
            song_id     TEXT,
            position    INTEGER,   -- 0 = queue-level A/B preference row
            song_rating INTEGER,   -- A/B: 1 = SmartShuffle preferred, 0 = baseline preferred
            rated_at    TEXT,
            FOREIGN KEY (queue_id) REFERENCES queues(queue_id)
        );

        CREATE INDEX IF NOT EXISTS idx_queues_algo ON queues(algorithm, queue_id);
    """)
    conn.commit()

# ── Data loading ──────────────────────────────────────────────────────────────

def load_songs(conn, playlist_id=None) -> pd.DataFrame:
    """
    Joins playlist_tracks → songs → song_scores → song_tags.
    Songs with no play history get fatigue=0, binge=0, coverage_debt=0 (unheard).
    """
    where  = "WHERE pt.playlist_id = ?" if playlist_id else ""
    params = (playlist_id,) if playlist_id else ()

    df = pd.read_sql_query(f"""
        SELECT
            pt.song_id,
            pt.playlist_id,
            COALESCE(s.song_name,  ss.song_name)   AS song_name,
            COALESCE(s.artist_name, ss.artist_name) AS artist_name,
            COALESCE(ss.fatigue,             0.0)    AS fatigue,
            COALESCE(ss.binge_score,         0.0)    AS binge_score,
            COALESCE(ss.binge_velocity,      0.0)    AS binge_velocity,
            COALESCE(ss.binge_skip_rate,     0.0)    AS binge_skip_rate,
            COALESCE(ss.artist_fatigue,      0.0)    AS artist_fatigue,
            COALESCE(ss.artist_binge_score,  0.0)    AS artist_binge_score,
            COALESCE(ss.play_count,          0)      AS play_count,
            ss.last_played,
            COALESCE(ss.skip_rate,    0.0)          AS skip_rate,
            COALESCE(ss.pattern,   'normal')        AS pattern,
            COALESCE(ss.coverage_debt, 0.0)         AS coverage_debt,
            COALESCE(ss.stale,      0)              AS stale,
            COALESCE(ss.artist_comp_rate, 0.5)      AS artist_comp_rate,
            COALESCE(ss.artist_comp_conf, 0.0)      AS artist_comp_conf,
            COALESCE(st.behavioral_energy_score, st.energy_score, 0.0)                           AS energy_score,
            COALESCE(st.behavioral_energy_morning,    st.behavioral_energy_score, st.energy_score, 0.0) AS energy_morning,
            COALESCE(st.behavioral_energy_afternoon,  st.behavioral_energy_score, st.energy_score, 0.0) AS energy_afternoon,
            COALESCE(st.behavioral_energy_late_night, st.behavioral_energy_score, st.energy_score, 0.0) AS energy_late_night,
            st.top_tags,
            s.vibe_content,
            s.vibe_melodic,
            s.vibe_bpm
        FROM playlist_tracks pt
        LEFT JOIN songs       s  ON pt.song_id = s.song_id
        LEFT JOIN song_scores ss ON pt.song_id = ss.song_id
        LEFT JOIN song_tags   st ON pt.song_id = st.song_id
        {where}
    """, conn, params=params)

    df["top_tags"]     = df["top_tags"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    df["energy_score"] = pd.to_numeric(df["energy_score"], errors="coerce").fillna(0.0)
    df["stale"]        = df["stale"].astype(bool)
    return df.drop_duplicates(subset=["song_id", "playlist_id"])

# ── Phase 3: vibe clustering ──────────────────────────────────────────────────

def _select_k(scaled: np.ndarray, max_k: int = 5) -> int:
    n = len(scaled)
    if n < 4:
        return 2
    best_k, best_score = 2, -1.0
    for k in range(2, min(max_k + 1, n)):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        score  = silhouette_score(scaled, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def cluster_playlist(conn, df: pd.DataFrame, k: int = None) -> tuple[pd.DataFrame, dict]:
    """
    Clusters songs in df using (vibe_content, vibe_melodic, vibe_bpm) features.
    Songs without vibe scores yet are filled with 0 (neutral center) so they
    land near the mean rather than distorting cluster geometry.
    Returns df with cluster_id + cluster_label added, and a labels dict.
    """
    df = df.copy()
    vc = pd.to_numeric(df.get("vibe_content"), errors="coerce").fillna(0.0)
    vm = pd.to_numeric(df.get("vibe_melodic"), errors="coerce").fillna(0.0)
    vb = pd.to_numeric(df.get("vibe_bpm"),     errors="coerce").fillna(0.0)

    features = np.column_stack([vc.values, vm.values, vb.values])
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(features)

    if k is None:
        k = _select_k(scaled, max_k=3)
    print(f"  k={k} clusters selected")

    km               = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(scaled)

    centroids = scaler.inverse_transform(km.cluster_centers_)
    labels    = {i: cluster_label(centroids[i, 0], centroids[i, 2]) for i in range(k)}
    df["cluster_label"] = df["cluster_id"].map(labels)

    for _, row in df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO song_clusters
            (song_id, playlist_id, cluster_id, cluster_label, vibe_content, vibe_melodic, vibe_bpm)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["song_id"], row["playlist_id"],
            int(row["cluster_id"]), row["cluster_label"],
            row.get("vibe_content"), row.get("vibe_melodic"), row.get("vibe_bpm"),
        ))
    conn.commit()
    return df, labels

# ── Phase 4: queue orchestrator ───────────────────────────────────────────────

def _score(row: pd.Series, target_energy: float, weights: dict,
           sigma: float = 0.30, target_vibe: dict | None = None,
           vibe_sigmas: dict | None = None) -> float:
    """
    Ranking score for a single candidate song. Higher = more likely to be queued.

    Energy uses a Gaussian rather than linear distance. σ is adaptive:
    permissive (0.5) when session is flowing, tight (0.2) when user is picky.
    The same σ also controls the vibe Gaussian — high skip rate tightens both.

    fatigue × (1 − binge_score):
      Active binge  → binge_score ≈ 1 → penalty ≈ 0 (let the binge run).
      Cooling binge → binge_score decays (7-day half-life) → penalty re-engages.
      Cold song     → binge_score = 0 → full fatigue penalty applies.

    Cluster variety is handled architecturally (80/20 primary/secondary split)
    rather than as a score term, so it doesn't fight within-cluster coherence.
    """
    # Binge state — velocity gate uses -0.5 instead of -0.1 so natural half-life
    # decay doesn't falsely deactivate an active binge; only rapid collapse does.
    # Guard on binge_skip_rate < 0.30: skipping the binge song = not a real binge.
    _binge_velocity  = row.get("binge_velocity",  0.0)
    _binge_skip_rate = row.get("binge_skip_rate", 0.0)
    _effective_binge    = row["binge_score"] if (_binge_velocity >= -0.5 and _binge_skip_rate < 0.30) else 0.0
    # Evergreen: reduces fatigue for songs with long-term consistent play history.
    # Capped at EVERGREEN_MAX_LIFT; binge takes precedence via max().
    _effective_evergreen = min(float(row.get("evergreen_score", 0.0)), EVERGREEN_MAX_LIFT)

    # Binge sigma widening: binge songs are accepted across a wider energy/vibe
    # range — you want to hear the song regardless of current session energy context.
    # At full binge (1.0): sigma × 1.5. At zero: no change.
    _binge_sigma_mult = 1.0 + _effective_binge * 0.50

    # Energy match — weight is 0.0 (Option B: disabled, replaced by 3-axis vibe_match).
    # Computed but not used; formula term is kept so it can be re-enabled via weights.
    energy_match = float(np.exp(-0.5 * ((row["energy_score"] - target_energy) / (sigma * _binge_sigma_mult)) ** 2))

    # Song-level fatigue suppression, modulated by binge momentum + skip tendency.
    # Skip-rate modulation: fatigue only suppresses songs you're playing AND skipping.
    # 0% skip → 25% of fatigue penalty applied; 100% skip → full penalty.
    _song_skip_rate   = max(0.0, min(1.0, float(row.get("skip_rate", 0.0) or 0.0)))
    _fatigue_skip_mod = 0.25 + 0.75 * _song_skip_rate
    effective_fatigue = row["fatigue"] * (1.0 - max(_effective_binge, _effective_evergreen)) * _fatigue_skip_mod

    # Artist-level fatigue suppression, modulated by artist binge momentum.
    # Separate from song fatigue so Ridge can learn independent coefficients.
    effective_artist_fatigue = (
        row.get("artist_fatigue", 0.0) * (1.0 - row.get("artist_binge_score", 0.0))
    )

    # Coverage: stale songs get a boost, capped so one very buried song
    # doesn't always win regardless of energy fit
    coverage_bonus = min(row["coverage_debt"] / 4.0, 1.0)

    # Skip: growing_tired gets an extra hit on top of base skip_rate
    skip_penalty = row["skip_rate"]
    if row["pattern"] == "growing_tired":
        skip_penalty = min(skip_penalty + 0.5, 1.0)

    # Short-term recency: linear decay over 24 h — 1.0 at 0 h, 0.0 at 24+ h.
    # Never-played songs (hours_since_played = inf) get no penalty.
    hours_ago       = row.get("hours_since_played", float("inf"))
    recency_penalty = max(0.0, 1.0 - hours_ago / 24.0)

    # Artist completion rate: confidence-weighted signal from non-dominant songs.
    # Dominant songs (>40% of artist plays) excluded — one binge track shouldn't
    # inflate the rate for every other song by that artist.
    # Signal is centered at 0: 1.0 comp_rate → +1.0, 0.5 (neutral) → 0.0, 0.0 → -1.0.
    _artist_comp_rate = float(row.get("artist_comp_rate") or 0.5)
    _artist_comp_conf = float(row.get("artist_comp_conf") or 0.0)
    _eff_comp         = 0.5 + (_artist_comp_rate - 0.5) * _artist_comp_conf
    artist_comp_signal = (_eff_comp - 0.5) * 2.0

    # Vibe match — binge widens the per-axis sigmas by the same multiplier
    # as energy, so vibe mismatch also doesn't block binge songs.
    vibe_match = 1.0
    if (target_vibe is not None
            and row.get("vibe_content") is not None
            and row.get("vibe_melodic") is not None
            and row.get("vibe_bpm")     is not None):
        vc = float(row["vibe_content"])
        vm = float(row["vibe_melodic"])
        vb = float(row["vibe_bpm"])
        vs = vibe_sigmas or {"content": sigma, "melodic": sigma, "bpm": sigma}
        gc = float(np.exp(-0.5 * (vc - target_vibe["content"])**2 / (vs["content"] * _binge_sigma_mult)**2))
        gm = float(np.exp(-0.5 * (vm - target_vibe["melodic"])**2 / (vs["melodic"] * _binge_sigma_mult)**2))
        gb = float(np.exp(-0.5 * (vb - target_vibe["bpm"])**2     / (vs["bpm"]     * _binge_sigma_mult)**2))
        vibe_match = (gc + gm + gb) / 3.0

    return round(
          weights["energy_match"]               * energy_match
        + weights.get("vibe_match", 0.0)        * vibe_match
        - weights["fatigue"]                    * effective_fatigue
        - weights.get("artist_fatigue", 0.10)   * effective_artist_fatigue
        + weights["coverage"]                   * coverage_bonus
        - weights["skip"]                       * skip_penalty
        - weights.get("recency", 0.30)          * recency_penalty
        + weights.get("binge_boost", 0.15)      * _effective_binge
        + weights.get("artist_comp", 0.10)      * artist_comp_signal,
        6,
    )

def _make_entry(pick: pd.Series, score: float) -> dict:
    def _safe_vibe(key):
        v = pick.get(key)
        return round(float(v), 3) if v is not None else None

    return {
        "song_id":           pick["song_id"],
        "song_name":         pick["song_name"],
        "artist_name":       pick["artist_name"],
        "score":             round(score, 4),
        "cluster_label":     pick.get("cluster_label", "unknown"),
        "energy_score":      round(float(pick["energy_score"]), 3),
        "vibe_content":      _safe_vibe("vibe_content"),
        "vibe_melodic":      _safe_vibe("vibe_melodic"),
        "vibe_bpm":          _safe_vibe("vibe_bpm"),
        "fatigue":           round(float(pick["fatigue"]), 3),
        "binge_score":       round(float(pick["binge_score"]), 3),
        "artist_fatigue":    round(float(pick.get("artist_fatigue",    0.0)), 3),
        "artist_binge_score":round(float(pick.get("artist_binge_score",0.0)), 3),
        "coverage_debt":     round(float(pick["coverage_debt"]), 3),
        "artist_comp_rate":  round(float(pick.get("artist_comp_rate", 0.5)), 3),
        "artist_comp_conf":  round(float(pick.get("artist_comp_conf", 0.0)), 3),
    }

def _compute_scores(pool: pd.DataFrame, target_energy: float, weights: dict,
                    sigma: float = 0.30,
                    target_vibe: dict | None = None,
                    vibe_sigmas: dict | None = None) -> pd.Series:
    """
    Vectorized base score for the full pool.

    Identical math to _score() but operates on entire columns at once.
    Called per pick in the queue loop (cheap — pool is ~200-300 rows).
    """
    energy_match = np.exp(-0.5 * ((pool["energy_score"] - target_energy) / sigma) ** 2)

    binge    = pool["binge_score"].fillna(0.0)
    fatigue  = pool["fatigue"].fillna(0.0)
    evergreen = (pool["evergreen_score"].fillna(0.0).clip(upper=EVERGREEN_MAX_LIFT)
                 if "evergreen_score" in pool.columns else pd.Series(0.0, index=pool.index))
    eff_fat  = fatigue * (1.0 - np.maximum(binge, evergreen))

    art_fat   = (pool["artist_fatigue"].fillna(0.0)    if "artist_fatigue"   in pool.columns
                 else pd.Series(0.0, index=pool.index))
    art_binge = (pool["artist_binge_score"].fillna(0.0) if "artist_binge_score" in pool.columns
                 else pd.Series(0.0, index=pool.index))
    eff_arti  = art_fat * (1.0 - art_binge)

    coverage = (pool["coverage_debt"].fillna(0.0) / 4.0).clip(upper=1.0)

    skip = pool["skip_rate"].fillna(0.0)
    if "pattern" in pool.columns:
        tired = pool["pattern"] == "growing_tired"
        skip  = pd.Series(
            np.where(tired, np.minimum(skip + 0.5, 1.0), skip.to_numpy()),
            index=pool.index,
        )

    if "hours_since_played" in pool.columns:
        hrs = pool["hours_since_played"].fillna(float("inf"))
    else:
        hrs = pd.Series(float("inf"), index=pool.index)
    recency = (1.0 - hrs / 24.0).clip(lower=0.0)

    # Vibe match: mean of three independent per-axis Gaussians.
    # Each axis uses its own data-derived sigma so content/melodic/bpm tighten
    # independently.  Songs missing vibe scores get 1.0 (neutral, no penalty).
    vibe_w = weights.get("vibe_match", 0.0)
    if target_vibe is not None and vibe_w > 0 and "vibe_content" in pool.columns:
        vc = pool["vibe_content"].fillna(float("nan"))
        vm = pool["vibe_melodic"].fillna(float("nan"))
        vb = pool["vibe_bpm"].fillna(float("nan"))
        vs = vibe_sigmas or {"content": sigma, "melodic": sigma, "bpm": sigma}
        gc = np.exp(-0.5 * (vc - target_vibe["content"])**2 / vs["content"]**2)
        gm = np.exp(-0.5 * (vm - target_vibe["melodic"])**2 / vs["melodic"]**2)
        gb = np.exp(-0.5 * (vb - target_vibe["bpm"])**2     / vs["bpm"]**2)
        vibe_match = ((gc + gm + gb) / 3.0).fillna(1.0)
    else:
        vibe_match = pd.Series(1.0, index=pool.index)

    binge_w = weights.get("binge_boost", 0.15)
    binge_vel  = pool["binge_velocity"].fillna(0.0)  if "binge_velocity"  in pool.columns else pd.Series(0.0, index=pool.index)
    binge_skr  = pool["binge_skip_rate"].fillna(0.0) if "binge_skip_rate" in pool.columns else pd.Series(0.0, index=pool.index)
    eff_binge  = binge.where((binge_vel >= -0.5) & (binge_skr < 0.30), 0.0)

    comp_rate = (pool["artist_comp_rate"].fillna(0.5) if "artist_comp_rate" in pool.columns
                 else pd.Series(0.5, index=pool.index))
    comp_conf = (pool["artist_comp_conf"].fillna(0.0) if "artist_comp_conf" in pool.columns
                 else pd.Series(0.0, index=pool.index))
    eff_comp   = 0.5 + (comp_rate - 0.5) * comp_conf
    comp_signal = (eff_comp - 0.5) * 2.0

    return (
          weights["energy_match"]               * energy_match
        + vibe_w                                * vibe_match
        - weights["fatigue"]                    * eff_fat
        - weights.get("artist_fatigue", 0.10)   * eff_arti
        + weights["coverage"]                   * coverage
        - weights["skip"]                       * skip
        - weights.get("recency", 0.30)          * recency
        + binge_w                               * eff_binge
        + weights.get("artist_comp", 0.10)      * comp_signal
    )


def _softmax_pick(scores: pd.Series, temperature: float = SCORE_TEMPERATURE) -> int:
    """Sample one index from scores proportionally to exp(score / T).
    Songs with clearly higher scores are still more likely; songs with
    similar scores are chosen roughly uniformly — no detectable order.
    temperature=0 falls back to argmax (used by tests)."""
    if temperature <= 0:
        return scores.idxmax()
    vals  = scores.values.astype(float)
    vals  = vals - vals.max()                        # numerical stability
    probs = np.exp(vals / temperature)
    probs = probs / probs.sum()
    return scores.index[np.random.choice(len(scores), p=probs)]


def _pick_best(pool: pd.DataFrame, used: set, target_energy: float,
               weights: dict, recent_artists: list = None,
               base_scores: pd.Series = None,
               temperature: float = SCORE_TEMPERATURE,
               sigma: float = 0.30,
               target_vibe: dict | None = None,
               vibe_sigmas: dict | None = None):
    avail = pool[~pool["song_id"].isin(used)]
    if avail.empty:
        return None, None

    if recent_artists:
        last_artist = recent_artists[-1]

        # Hard filter: always remove last artist unless no other choice.
        no_last       = avail[avail["artist_name"] != last_artist]
        filtered_last = not no_last.empty
        if filtered_last:
            avail = no_last

        # Graduated soft penalty for artists earlier in the recency window.
        # When last_artist couldn't be filtered (only artist left), give them
        # ARTIST_FATIGUE_FACTOR as a fallback penalty instead of skipping entirely.
        artist_soft: dict = {}
        for i, artist in enumerate(recent_artists):
            if artist == last_artist:
                if not filtered_last:
                    if artist not in artist_soft or ARTIST_FATIGUE_FACTOR < artist_soft[artist]:
                        artist_soft[artist] = ARTIST_FATIGUE_FACTOR
                continue
            dist   = len(recent_artists) - 1 - i
            factor = ARTIST_FATIGUE_FACTOR + (1.0 - ARTIST_FATIGUE_FACTOR) * (
                dist / ARTIST_FATIGUE_WINDOW
            )
            if artist not in artist_soft or factor < artist_soft[artist]:
                artist_soft[artist] = factor

        scores = (base_scores.loc[avail.index] if base_scores is not None
                  else avail.apply(
                      lambda r: _score(r, target_energy, weights, sigma, target_vibe, vibe_sigmas), axis=1))
        for artist, factor in artist_soft.items():
            mask = avail["artist_name"] == artist
            scores = scores.where(~mask, scores * factor)
    else:
        scores = (base_scores.loc[avail.index] if base_scores is not None
                  else avail.apply(
                      lambda r: _score(r, target_energy, weights, sigma, target_vibe, vibe_sigmas), axis=1))

    best_idx = _softmax_pick(scores, temperature)
    return avail.loc[best_idx], scores[best_idx]

def generate_queue(df: pd.DataFrame, context: str,
                   n: int = None, weights: dict = None, conn=None,
                   playlist_id: str = None) -> list:
    """
    Score-ranked queue builder.

    All n queue slots are filled by _pick_best against the full song pool,
    ranked by the scoring formula (energy Gaussian + fatigue × binge + coverage).

    Stakes-aware weight adjustment: loads historical engagement_baselines from
    phase2.py to detect whether this time bucket is typically high-stakes (user
    actively searching for the right vibe) or low-stakes (passive listening).
    High stakes → raise energy_match, lower coverage.
    Low stakes  → raise coverage, relax energy constraint.

    Artist caps and the hard back-to-back block keep the queue varied.
    Binge shape modifier scales binge_score by the learned arc position.
    """
    if weights is None:
        weights = WEIGHTS

    # context may be a bare bucket ("late_night") or a full session label
    # ("late_night_high_energy"). Match longest-prefix bucket name so "late_night"
    # isn't truncated to "late" by a naive split.
    _BUCKETS = ("late_night", "morning", "afternoon")
    time_bucket   = next((b for b in _BUCKETS if context == b or context.startswith(b + "_")), context)
    target_energy = get_target_energy(time_bucket, playlist_id)
    # Behavioral context target (on behavioral energy scale) — used after we assign
    # per-context behavioral energy to pool["energy_score"] below.
    behavioral_target = _BEHAVIORAL_CONTEXT_TARGETS.get(time_bucket, target_energy)

    # Apply stakes-aware weight adjustment from historical engagement baselines
    if conn is not None:
        baselines   = load_engagement_baselines(conn)
        session_ctx = SessionContext(time_bucket, baselines.get(time_bucket, {}))
        weights     = session_ctx.adjusted_weights(weights)
        baseline    = baselines.get(time_bucket, {})
        if baseline:
            delta = session_ctx.engagement_delta
            rt    = f"  live_delta={delta:+.1%}" if abs(delta) >= 0.01 else ""
            print(f"  Stakes context: {baseline.get('typical_stakes', 'normal')}"
                  f"  (skip_rate={baseline.get('baseline_skip_rate', 0):.1%}"
                  f"  early_skips={baseline.get('early_skip_rate', 0):.1%}"
                  f"  n={baseline.get('session_count', 0)} sessions){rt}")
            print(f"  Adjusted weights: {weights}")

    pool = df.drop_duplicates("song_id").copy()

    # n defaults to the full pool so the queue covers the entire playlist
    if n is None:
        n = len(pool)

    # Adaptive σ: data-derived per-axis baselines (~0.30), tightened by skip rate.
    # Mean of per-axis sigmas used for the 1D energy Gaussian.
    recent_skip_rate = _read_session_skip_rate(time_bucket)
    vibe_sigmas      = _get_vibe_sigmas(time_bucket, recent_skip_rate)
    sigma            = round(sum(vibe_sigmas.values()) / 3, 4)
    mult             = _sigma_multiplier(recent_skip_rate)
    print(f"  σ  mult={mult:.2f}"
          f"  c={vibe_sigmas['content']:.3f}"
          f"  m={vibe_sigmas['melodic']:.3f}"
          f"  bpm={vibe_sigmas['bpm']:.3f}"
          + ("  (tightened)" if mult < 1.0 else "  (baseline)"))

    # Vibe target for this playlist (from learned_params.json).
    target_vibe = get_target_vibe(playlist_id)
    if target_vibe:
        print(f"  Vibe targets  c={target_vibe['content']:+.3f}  "
              f"m={target_vibe['melodic']:+.3f}  b={target_vibe['bpm']:+.3f}")
        # Cross-refill session vibe drift: blend learned target with the mean vibe
        # of completed songs so far this session (written by watcher.py).
        target_vibe = _apply_session_vibe_drift(target_vibe, time_bucket)
    else:
        print("  Vibe targets: none yet (run train.py after scoring songs with score_vibes.py)")

    # Vectorised hours-since-last-play for the short-term recency penalty.
    if "last_played" in pool.columns:
        _now_ts = pd.Timestamp.now(tz="UTC")
        lp_dt   = pd.to_datetime(pool["last_played"], utc=True, errors="coerce")
        delta_h = (_now_ts - lp_dt).dt.total_seconds() / 3600
        pool["hours_since_played"] = delta_h.where(delta_h >= 0, float("inf")).fillna(float("inf"))
    else:
        pool["hours_since_played"] = float("inf")

    # Look up active binge episode start dates for binge shape modifier.
    # Falls back gracefully if table doesn't exist yet (pre-phase2 run).
    _binge_starts: dict = {}
    if conn is not None:
        try:
            for _sid, _start_str in conn.execute(
                "SELECT song_id, start_date FROM binge_episodes WHERE end_date IS NULL"
            ).fetchall():
                _binge_starts[_sid] = (date.today() - date.fromisoformat(_start_str)).days
        except sqlite3.OperationalError:
            pass

    pool["days_since_binge_start"] = pool["song_id"].map(
        lambda sid: _binge_starts.get(sid, -1)
    )

    # Apply learned binge shape modifier (Option B).
    # shape[day_N] ≈ 1.0 at peak (early binge), decays toward 0 as the binge winds down.
    # When _BINGE_SHAPE is None (not enough episodes yet), binge_score is unchanged —
    # flat suppression remains in effect, same as pre-feature behaviour.
    if _BINGE_SHAPE is not None:
        _shape_arr  = np.array(_BINGE_SHAPE)
        _days       = pool["days_since_binge_start"].clip(lower=0, upper=len(_shape_arr) - 1).astype(int)
        _active     = (pool["binge_score"] > 0.0) & (pool["days_since_binge_start"] >= 0)
        _multiplier = np.where(_active, _shape_arr[_days], 1.0)
        pool["binge_score"]        = (pool["binge_score"]        * _multiplier).round(4)
        pool["artist_binge_score"] = (pool["artist_binge_score"] * _multiplier).round(4)

    # Per-artist queue cap: how many slots an artist may occupy in this queue.
    # Proportional to their share of the pool — Drake with 10/200 songs gets
    # ceil(10 × 10/200) = 1 slot; Drake with 50/200 gets ceil(10 × 50/200) = 3.
    # Minimum 1 so every artist in the pool can appear at least once.
    # Binge exception is applied at pick time (artist is removed from the cap set).
    _n_pool     = len(pool)
    artist_caps: dict = {
        a: max(1, int(np.ceil(n * cnt / _n_pool * ARTIST_CAP_MULTIPLIER)))
        for a, cnt in pool.groupby("artist_name")["song_id"].count().items()
    }

    # Pre-compute the recently-queued penalty once — it doesn't change across picks.
    # Songs near the front of the last queue get RECENTLY_QUEUED_PENALTY; tail songs
    # get ~0. This rotates ordering across consecutive runs without hard-excluding anything.
    queued_penalty = pd.Series(0.0, index=pool.index)
    if conn is not None:
        _row = conn.execute(
            "SELECT songs FROM queues WHERE algorithm='smartshuffle'"
            " ORDER BY queue_id DESC LIMIT 1"
        ).fetchone()
        if _row:
            _last_queue = json.loads(_row[0])
            _n_last     = len(_last_queue)
            if _n_last > 0:
                _pos_map       = {s["song_id"]: i for i, s in enumerate(_last_queue)}
                _pos           = pool["song_id"].map(_pos_map)   # NaN = not in last queue
                queued_penalty = (RECENTLY_QUEUED_PENALTY * (1.0 - _pos / _n_last)).fillna(0.0)

    used            = set()
    queue           = []
    recent_artists  = deque(maxlen=ARTIST_FATIGUE_WINDOW)
    artist_picks:   dict  = {}
    recent_vibes: deque = deque(maxlen=4)  # last 4 picks for within-queue vibe drift

    for pos in range(n):
        # Session continuity: drift the vibe target toward what's been playing.
        # Creates a road (each pick flows from the last) rather than oscillating
        # around a fixed playlist target.  Only activates when a vibe target exists
        # and enough picks have been made; otherwise uses the playlist target as-is.
        if (target_vibe is not None
                and len(recent_vibes) >= _SESSION_DRIFT_MIN_PICKS):
            effective_vibe = {
                ax: round(
                    (1 - _SESSION_DRIFT_WEIGHT) * target_vibe[ax]
                    + _SESSION_DRIFT_WEIGHT * float(np.mean([v[ax] for v in recent_vibes])),
                    4,
                )
                for ax in ("content", "melodic", "bpm")
            }
        else:
            effective_vibe = target_vibe

        # Recompute scores with the current effective vibe target, adaptive σ.
        # Pool is ~200-300 rows so vectorised recompute per pick is negligible.
        base_scores = _compute_scores(pool, target_energy, weights, sigma, effective_vibe, vibe_sigmas) - queued_penalty

        # Artists that have used up their per-queue cap, minus actively-binged ones
        capped: set = set()
        for artist, count in artist_picks.items():
            if count >= artist_caps.get(artist, n):
                artist_rows = pool[pool["artist_name"] == artist]
                binge = (
                    float(artist_rows["artist_binge_score"].max())
                    if not artist_rows.empty and "artist_binge_score" in pool.columns
                    else 0.0
                )
                if binge < BINGE_ALLOW_THRESHOLD:
                    capped.add(artist)

        # Temporarily exclude capped artists' songs, but only when uncapped
        # alternatives exist — releasing caps beats forcing a back-to-back.
        if capped:
            capped_ids     = set(pool[pool["artist_name"].isin(capped)]["song_id"])
            avail_uncapped = pool[~pool["song_id"].isin(used | capped_ids)]
            _last          = list(recent_artists)[-1] if recent_artists else None
            has_alt_uncapped = not avail_uncapped.empty and (
                _last is None or (avail_uncapped["artist_name"] != _last).any()
            )
            effective_used = used | capped_ids if has_alt_uncapped else used
        else:
            effective_used = used

        pick, score = _pick_best(
            pool, effective_used, target_energy, weights, list(recent_artists),
            base_scores=base_scores, sigma=sigma, target_vibe=effective_vibe,
            vibe_sigmas=vibe_sigmas,
        )
        if pick is None:
            break
        queue.append(_make_entry(pick, score))
        used.add(pick["song_id"])
        artist_picks[pick["artist_name"]] = artist_picks.get(pick["artist_name"], 0) + 1
        recent_artists.append(pick["artist_name"])
        # Track vibe of each pick for within-queue drift (only when vibe scores exist)
        if (target_vibe is not None
                and pick.get("vibe_content") is not None
                and not np.isnan(float(pick.get("vibe_content", float("nan"))))):
            recent_vibes.append({
                "content": float(pick["vibe_content"]),
                "melodic": float(pick["vibe_melodic"]),
                "bpm":     float(pick["vibe_bpm"]),
            })

    if target_vibe is not None and len(recent_vibes) >= _SESSION_DRIFT_MIN_PICKS:
        final_vibe = {
            ax: round(
                (1 - _SESSION_DRIFT_WEIGHT) * target_vibe[ax]
                + _SESSION_DRIFT_WEIGHT * float(np.mean([v[ax] for v in recent_vibes])),
                4,
            )
            for ax in ("content", "melodic", "bpm")
        }
        print(
            f"  Session vibe drift over {len(queue)} picks:"
            f"  c {target_vibe['content']:+.3f}→{final_vibe['content']:+.3f}"
            f"  m {target_vibe['melodic']:+.3f}→{final_vibe['melodic']:+.3f}"
            f"  bpm {target_vibe['bpm']:+.3f}→{final_vibe['bpm']:+.3f}"
        )

    return queue[:n]

def save_queue(conn, ss_queue: list, context: str, playlist_id: str) -> int:
    """Saves the SmartShuffle queue to DB. Returns the new queue_id."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("""
        INSERT INTO queues (generated_at, context, algorithm, playlist_id, songs)
        VALUES (?, ?, 'smartshuffle', ?, ?)
    """, (now, context, playlist_id, json.dumps(ss_queue)))
    conn.commit()
    return cur.lastrowid


def generate_random_baseline(df: pd.DataFrame, n: int | None = None) -> list[dict]:
    """Generates a random shuffle of the same pool for A/B comparison."""
    shuffled = df.drop_duplicates("song_id").sample(frac=1).reset_index(drop=True)
    if n:
        shuffled = shuffled.head(n)
    return [
        {
            "song_id":    row["song_id"],
            "song_name":  row["song_name"],
            "artist_name": row["artist_name"],
            "score":      None,
            "energy_score": float(row.get("energy_score", 0.0)),
        }
        for _, row in shuffled.iterrows()
    ]


def save_random_baseline(conn, baseline: list, context: str, playlist_id: str) -> int:
    """Saves the random baseline queue to DB. Returns the new queue_id."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute("""
        INSERT INTO queues (generated_at, context, algorithm, playlist_id, songs)
        VALUES (?, ?, 'random_baseline', ?, ?)
    """, (now, context, playlist_id, json.dumps(baseline)))
    conn.commit()
    return cur.lastrowid

# ── Print helpers ─────────────────────────────────────────────────────────────

def print_clusters(df: pd.DataFrame, labels: dict):
    print(f"\n── Vibe clusters ──")
    for cid, lbl in sorted(labels.items()):
        members = df[df["cluster_id"] == cid].sort_values(
            "vibe_bpm", ascending=False, na_position="last"
        )
        print(f"  [{lbl}]  {len(members)} songs")
        for _, r in members.head(5).iterrows():
            vc = r.get("vibe_content")
            vm = r.get("vibe_melodic")
            vb = r.get("vibe_bpm")
            vibe_str = (f"  c={vc:+.2f} m={vm:+.2f} bpm={vb:+.2f}"
                        if vc is not None and not (isinstance(vc, float) and np.isnan(vc))
                        else "  (no vibe score)")
            print(f"    {r['song_name']:<35} {r['artist_name']:<20}{vibe_str}")
        if len(members) > 5:
            print(f"    … and {len(members)-5} more")

def print_queue(ss_queue: list, context: str, scores: bool = False):
    print(f"\n── SmartShuffle Queue  (context: {context}) ──\n")
    for i, s in enumerate(ss_queue, 1):
        line = f"  {i:2}. {s['song_name']:<38} {s['artist_name']}"
        if scores and s.get("score") is not None:
            vc = s.get("vibe_content")
            vm = s.get("vibe_melodic")
            vb = s.get("vibe_bpm")
            vibe_str = (f"  c={vc:+.2f} m={vm:+.2f} bpm={vb:+.2f}"
                        if vc is not None else "")
            line += (f"  score={s['score']:+.4f}  e={s['energy_score']:+.2f}{vibe_str}"
                     f"  fat={s['fatigue']:.2f}  binge={s['binge_score']:.2f}"
                     f"  art_fat={s['artist_fatigue']:.2f}  debt={s['coverage_debt']:.2f}"
                     f"  [{s['cluster_label']}]")
        print(line)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmartShuffle phase 3+4")
    parser.add_argument("--playlist", default=None,
                        help="Playlist name or ID (default: all songs across all playlists)")
    parser.add_argument("--context",  default=None,
                        help="Time-of-day bucket: late_night | morning | afternoon")
    parser.add_argument("--k",        default=None, type=int,
                        help="Number of vibe clusters (auto-selected if omitted)")
    parser.add_argument("--scores",    action="store_true",
                        help="Print per-song scoring details alongside the queue")
    parser.add_argument("--list",     action="store_true",
                        help="List available playlists and exit")
    parser.add_argument("--count",    default=None, type=int,
                        help="Number of songs in the queue (default: full playlist)")
    parser.add_argument("--exclude",  default=None,
                        help="Comma-separated song IDs to exclude (used by rolling queue)")
    parser.add_argument("--algorithm", default="smartshuffle",
                        choices=["smartshuffle", "random_baseline"],
                        help="Queue algorithm to generate (default: smartshuffle)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    init_tables(conn)

    if args.list:
        rows = conn.execute("""
            SELECT p.playlist_name, COUNT(pt.song_id) AS n_tracks
            FROM playlists p
            LEFT JOIN playlist_tracks pt ON p.playlist_id = pt.playlist_id
            GROUP BY p.playlist_id
            ORDER BY n_tracks DESC
        """).fetchall()
        print("Available playlists:")
        for name, n in rows:
            print(f"  {n:4d} tracks  {name}")
        conn.close()
        return

    # Guard: require phase 2 output
    tags_count = conn.execute("SELECT COUNT(*) FROM song_tags").fetchone()[0]
    if tags_count == 0:
        print("ERROR: song_tags is empty. Run phase2.py first.")
        conn.close()
        return

    pt_count = conn.execute("SELECT COUNT(*) FROM playlist_tracks").fetchone()[0]
    if pt_count == 0:
        print("ERROR: playlist_tracks is empty. Run collect_data.py first.")
        conn.close()
        return

    # Resolve playlist: explicit arg → energy-matched default → error
    playlist_id   = None
    playlist_name = None
    if args.playlist:
        row = conn.execute(
            "SELECT playlist_id, playlist_name FROM playlists "
            "WHERE playlist_name LIKE ? OR playlist_id = ? LIMIT 1",
            (f"%{args.playlist}%", args.playlist)
        ).fetchone()
        if not row:
            print(f"ERROR: playlist '{args.playlist}' not found. Use --list to see options.")
            conn.close()
            return
        playlist_id, playlist_name = row
    else:
        # Default to the playlist the user has played most recently/frequently.
        # Playlist selection is the user's job — SmartShuffle only orders within it.
        row = conn.execute("""
            SELECT REPLACE(context_uri, 'spotify:playlist:', '') AS pid, COUNT(*) AS n
            FROM plays
            WHERE context_uri LIKE 'spotify:playlist:%'
            GROUP BY context_uri
            ORDER BY n DESC
            LIMIT 1
        """).fetchone()
        if row:
            playlist_id = row[0]
            name_row    = conn.execute(
                "SELECT playlist_name FROM playlists WHERE playlist_id = ?", (playlist_id,)
            ).fetchone()
            playlist_name = name_row[0] if name_row else playlist_id
            print(f"No playlist specified — defaulting to most-played: {playlist_name}")

    context = args.context or current_time_bucket()
    print(f"=== SmartShuffle Phase 3+4 ===")
    print(f"Context: {context}  |  Playlist: {playlist_name or 'all'}\n")

    print("Loading songs...")
    df = load_songs(conn, playlist_id)
    if df.empty:
        print("No songs found. Check that playlist_tracks is populated.")
        conn.close()
        return

    exclude_ids = set(args.exclude.split(",")) if args.exclude else set()
    if exclude_ids:
        df = df[~df["song_id"].isin(exclude_ids)]
        print(f"  {len(df)} songs ({len(exclude_ids)} excluded)")
    else:
        print(f"  {len(df)} songs")

    if args.algorithm == "random_baseline":
        baseline = generate_random_baseline(df, n=args.count)
        save_random_baseline(conn, baseline, context, playlist_id)
        print(f"Random baseline: {len(baseline)} songs")
        conn.close()
        print("\nDone.")
        return

    print("Clustering vibes (Phase 3)...")
    df, labels = cluster_playlist(conn, df, k=args.k)
    print_clusters(df, labels)

    print("\nGenerating queue (Phase 4)...")
    ss_queue = generate_queue(df, context, n=args.count, conn=conn, playlist_id=playlist_id)
    save_queue(conn, ss_queue, context, playlist_id)

    baseline = generate_random_baseline(df) if not args.count else []
    if baseline:
        save_random_baseline(conn, baseline, context, playlist_id)

    print_queue(ss_queue, context, scores=args.scores)

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
