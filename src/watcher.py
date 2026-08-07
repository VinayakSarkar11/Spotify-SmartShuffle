"""
session_watcher.py — polls Spotify every POLL_INTERVAL seconds during a session,
infers skip status from timestamp gaps (same logic as collect_data.py), and writes
engagement_delta to SESSION_STATE_PATH so the next generate_queue call can apply
real-time stakes adjustment.

In rolling mode (rolling_state.enabled = True), also monitors remaining songs in
the SmartShuffle Queue playlist and appends a new batch when < REFILL_REMAINING songs
are left ahead of the current position.

Run via run.py (--play flag spawns this as a background thread). Can also be run
standalone for testing:
    python session_watcher.py --bucket afternoon
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

DIR                  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH              = os.path.join(DIR, "data", "smartshuffle.db")
SESSION_STATE_PATH   = os.path.join(DIR, "data", "session_state.json")
ROLLING_STATE_PATH   = os.path.join(DIR, "data", "rolling_queue_state.json")

ALLOW_RB_ROLLING      = False  # set True to re-enable RB rolling refills

POLL_INTERVAL         = 180  # seconds between polls (non-rolling)
ROLLING_POLL_INTERVAL = 15   # fast poll when approaching refill threshold
ROLLING_SLOW_INTERVAL = 90   # slow poll when plenty of songs remain
QUICK_SKIP_MS        = 30_000
SKIP_RATIO           = 0.30
PARTIAL_RATIO        = 0.80
MIN_PLAYS            = 3
RECENT_WINDOW        = 11    # songs for recent skip rate (one batch)
REFILL_REMAINING     = 4     # base: append new songs when this many remain in playlist
REFILL_SCALE_EVERY   = 10   # add 1 to threshold for every N plays in the session
REFILL_MAX           = 10   # cap so we never refill absurdly early
INACTIVITY_STANDBY   = 600   # 10 min of no plays → enter standby
STANDBY_POLL         = 300   # 5 min between standby checks (1 API call each)
STANDBY_EXIT_AFTER   = 7200  # 2 hours in standby → fully exit (session truly over)
MIN_REFILL_INTERVAL  = 90    # don't refill twice within 90 seconds

# ── Rate-limit backoff ────────────────────────────────────────────────────────
# When any Spotify call returns 429, all polling pauses until this timestamp.
_rate_limit_until: float = 0.0


def _is_rate_limited() -> bool:
    return time.time() < _rate_limit_until


def _apply_rate_limit(e: spotipy.exceptions.SpotifyException):
    global _rate_limit_until
    retry_after = 60  # conservative default
    if hasattr(e, "headers") and e.headers:
        ra = e.headers.get("Retry-After") or e.headers.get("retry-after")
        if ra:
            try:
                retry_after = int(ra)
            except (ValueError, TypeError):
                pass
    # Spotify sometimes embeds the value in the message body
    m = re.search(r"Retry will occur after:\s*(\d+)", str(e))
    if m:
        retry_after = int(m.group(1))
    _rate_limit_until = time.time() + retry_after
    resume = datetime.fromtimestamp(_rate_limit_until).strftime("%H:%M:%S")
    print(f"  [watcher] RATE LIMITED — pausing {retry_after}s (resume ~{resume})")


SCOPE = " ".join([
    "user-read-recently-played",
    "user-modify-playback-state",
    "user-read-playback-state",
    "user-library-read",
    "playlist-read-private",
    "playlist-modify-private",
    "user-top-read",
])


def _make_sp():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=SCOPE,
        cache_path=os.path.join(DIR, ".spotify_cache"),
    ))


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _infer_skip(play_duration_ms: float | None, song_duration_ms: int | None) -> str:
    if play_duration_ms is None or not song_duration_ms:
        return "unknown"
    ratio = play_duration_ms / song_duration_ms
    if ratio < SKIP_RATIO:
        return "skip"
    if ratio < PARTIAL_RATIO:
        return "partial"
    return "full"


def _fetch_recent(sp, session_start: datetime, limit: int = 50) -> list[dict]:
    """Return plays since session_start, oldest-first, with inferred skip status."""
    try:
        result = sp.current_user_recently_played(limit=limit)
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            _apply_rate_limit(e)
        return []
    except Exception:
        return []

    items = result.get("items", [])
    plays = []
    for item in reversed(items):
        ts = _parse_ts(item["played_at"])
        if ts < session_start:
            continue
        plays.append({
            "played_at":      item["played_at"],
            "song_id":        item["track"]["id"],
            "duration_ms":    item["track"]["duration_ms"],
            "play_duration_ms": None,
        })

    for i in range(len(plays) - 1):
        curr = _parse_ts(plays[i]["played_at"])
        nxt  = _parse_ts(plays[i + 1]["played_at"])
        gap_ms = (nxt - curr).total_seconds() * 1000
        plays[i]["play_duration_ms"] = min(gap_ms, plays[i]["duration_ms"] or gap_ms)

    for p in plays:
        p["skip_status"] = _infer_skip(p["play_duration_ms"], p["duration_ms"])

    return plays


def _compute_engagement(plays: list[dict], baseline_skip_rate: float,
                        prev_ever_high: bool = False) -> dict:
    """
    Returns overall_delta, recent_skip_rate, and the sticky ever_high_skip flag.

    ever_high_skip is a one-way ratchet: once True it never resets within a session,
    so the weight system can't reward "recovery" with a coverage increase.
    """
    if len(plays) < MIN_PLAYS:
        return {"overall_delta": 0.0, "recent_skip_rate": 0.0, "ever_high_skip": prev_ever_high}

    skipped_all  = sum(1 for p in plays if p["skip_status"] in ("skip", "partial"))
    overall_rate  = skipped_all / len(plays)
    overall_delta = round(overall_rate - baseline_skip_rate, 4)

    recent        = plays[-RECENT_WINDOW:]
    recent_skip   = sum(1 for p in recent if p["skip_status"] in ("skip", "partial"))
    recent_rate   = round(recent_skip / len(recent), 4)

    is_high_now   = (recent_rate - baseline_skip_rate) > 0.15 or overall_delta > 0.15
    return {
        "overall_delta":    overall_delta,
        "recent_skip_rate": recent_rate,
        "ever_high_skip":   prev_ever_high or is_high_now,
    }


RECENT_VIBE_N           = 10   # window size for recency-weighted session vibe
_MANUAL_VIBE_BOOST      = 3.0  # global session: manual play multiplier
_MANUAL_VIBE_MAX        = 0.60 # global session: manual plays cap
_RECENT_MANUAL_BOOST    = 6.0  # recent window: much heavier — one manual in last 10 ≈ 40% weight
_RECENT_MANUAL_MAX      = 0.85 # recent window: can nearly take over if manual plays dominate
_SKIP_REPULSION_W       = 0.40 # how hard recent skips push the recent vibe away from skipped vibes
_SKIP_SIGNAL_WINDOW     = 4    # play window for vibe-level skip detection
_SKIP_SIGNAL_THRESHOLD  = 2    # min skips in window to trigger signal
_SKIP_SIGNAL_MIN_COMPS  = 5    # min total completions required — prevents early-session false positives
_SKIP_VIBE_DELTA_MIN    = 0.30 # min euclidean dist between skip cluster and last-5-comp vibes
_RECENT_COMPLETIONS_N   = 5    # completions used for signal baseline and Case 2 parking target
_FLUSH_MAX              = 2    # max vibe-shift queue flushes per session before velocity is frozen
_FLUSH_COOLDOWN         = 30   # seconds between flush-triggered refills (shorter than MIN_REFILL_INTERVAL)
_VELOCITY_FREEZE_MINUTES = 20  # minutes velocity stays zeroed after hitting _FLUSH_MAX


def _get_session_queued_ids(rolling: dict) -> set[str]:
    """Return the set of song_ids across all pushes in the current rolling session."""
    session_push_id = rolling.get("session_push_id")
    if not session_push_id:
        return set()
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(DB_PATH)
        pushes = conn.execute("""
            SELECT q.songs FROM queue_pushes qp
            JOIN queues q ON q.queue_id = qp.queue_id
            WHERE COALESCE(qp.rolling_session_id, qp.push_id) = ?
        """, (session_push_id,)).fetchall()
        conn.close()
        ids: set[str] = set()
        for (songs_json,) in pushes:
            for s in json.loads(songs_json):
                ids.add(s["song_id"])
        return ids
    except Exception:
        return set()


def _load_vibe_map(song_ids: list[str]) -> dict:
    """Load (vibe_content, vibe_melodic, vibe_bpm) tuples for song_ids from the DB."""
    if not song_ids:
        return {}
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(DB_PATH)
        ph   = ",".join("?" * len(song_ids))
        rows = conn.execute(
            f"SELECT song_id, vibe_content, vibe_melodic, vibe_bpm "
            f"FROM songs WHERE song_id IN ({ph})",
            song_ids,
        ).fetchall()
        conn.close()
        return {r[0]: (r[1], r[2], r[3]) for r in rows
                if r[1] is not None and r[2] is not None and r[3] is not None}
    except Exception:
        return {}


def _compute_skip_signal(
    plays: list[dict],
    vibe_map: dict,
) -> tuple[dict | None, dict | None, bool]:
    """
    Inspect the last _SKIP_SIGNAL_WINDOW plays for a vibe-level rejection signal.

    Triggers when ≥ _SKIP_SIGNAL_THRESHOLD of the window are skips AND the
    euclidean distance between skip-vibe and completion-vibe exceeds _SKIP_VIBE_DELTA_MIN
    (small distances mean wrong song, not wrong vibe — ignore those).

    Returns (skip_cluster_vibe, recent_completions_vibe, delta_clear).
    skip_cluster_vibe:      mean vibe of skipped songs in window (for epsilon shaping)
    recent_completions_vibe: mean of last _RECENT_COMPLETIONS_N completions (for Case 2 parking)
    delta_clear:            True when the signal is a real vibe mismatch
    """
    _AXES = ("content", "melodic", "bpm")

    def _mean_v(song_list):
        vs = [vibe_map[p["song_id"]] for p in song_list if p["song_id"] in vibe_map]
        if not vs:
            return None
        return {ax: sum(v[i] for v in vs) / len(vs)
                for i, ax in enumerate(_AXES)}

    window = plays[-_SKIP_SIGNAL_WINDOW:]
    w_skip = [p for p in window if p["skip_status"] == "skip"]

    # Baseline: last _RECENT_COMPLETIONS_N completions across the whole session.
    # Using the full-session window (not just the 4-play window) prevents any one
    # outlier song from dominating the distance check.
    all_comp = [p for p in plays if p["skip_status"] in ("full", "partial")]
    recent_comp_vibe = _mean_v(all_comp[-_RECENT_COMPLETIONS_N:]) if all_comp else None

    # Require minimum completions so we have a credible baseline before signalling.
    if len(w_skip) < _SKIP_SIGNAL_THRESHOLD or len(all_comp) < _SKIP_SIGNAL_MIN_COMPS:
        return None, recent_comp_vibe, False

    skip_vibe = _mean_v(w_skip)
    if skip_vibe is None:
        return None, recent_comp_vibe, False

    delta_clear = False
    if recent_comp_vibe is not None:
        dist = sum((skip_vibe[ax] - recent_comp_vibe[ax]) ** 2 for ax in _AXES) ** 0.5
        delta_clear = dist >= _SKIP_VIBE_DELTA_MIN

    return skip_vibe, recent_comp_vibe, delta_clear


def _compute_session_vibe(
    plays: list[dict],
    queued_ids: set[str] | None = None,
    vibe_map: dict | None = None,
) -> tuple[dict | None, int, dict | None, int, dict | None]:
    """
    Mean vibe (content, melodic, bpm) of completed songs so far in this session,
    plus the mean of the most recent RECENT_VIBE_N completions for recency weighting.

    When queued_ids is provided, manual plays (song_id not in queued_ids) are
    blended in with boosted weight so the user's steering signal is amplified.

    Returns (blended_mean, n_with_vibes, recent_mean, n_manual, manual_vibe_raw).
    manual_vibe_raw is the unblended mean of manual plays (for velocity reasoning).
    """
    completed = [p for p in plays if p["skip_status"] in ("full", "partial")]
    if not completed:
        return None, 0, None, 0, None

    if vibe_map is None:
        unique_ids = list(dict.fromkeys(p["song_id"] for p in plays))
        vibe_map   = _load_vibe_map(unique_ids)
        if not vibe_map:
            return None, 0, None, 0, None

    def _mean(song_list: list[dict]) -> dict | None:
        vibes = [vibe_map[p["song_id"]] for p in song_list if p["song_id"] in vibe_map]
        if not vibes:
            return None
        return {
            "content": round(sum(v[0] for v in vibes) / len(vibes), 4),
            "melodic": round(sum(v[1] for v in vibes) / len(vibes), 4),
            "bpm":     round(sum(v[2] for v in vibes) / len(vibes), 4),
        }

    n_valid = sum(1 for p in completed if p["song_id"] in vibe_map)

    if queued_ids:
        queued   = [p for p in completed if p["song_id"] in queued_ids]
        manual   = [p for p in completed if p["song_id"] not in queued_ids]
        n_manual = len(manual)

        queued_vibe = _mean(queued)
        manual_vibe = _mean(manual)

        manual_vibe_raw = manual_vibe
        if manual_vibe and queued_vibe and n_manual > 0:
            n_q = sum(1 for p in queued if p["song_id"] in vibe_map)
            n_m = sum(1 for p in manual if p["song_id"] in vibe_map)
            boosted_m = n_m * _MANUAL_VIBE_BOOST
            manual_frac = min(boosted_m / (boosted_m + n_q), _MANUAL_VIBE_MAX)
            all_mean = {
                ax: round((1 - manual_frac) * queued_vibe[ax]
                          + manual_frac * manual_vibe[ax], 4)
                for ax in ("content", "melodic", "bpm")
            }
            print(f"  [watcher] manual plays={n_manual}  frac={manual_frac:.2f}"
                  f"  manual_vibe c={manual_vibe['content']:+.3f}"
                  f" m={manual_vibe['melodic']:+.3f}"
                  f" bpm={manual_vibe['bpm']:+.3f}", flush=True)
        elif manual_vibe and not queued_vibe:
            all_mean = manual_vibe
        else:
            all_mean = queued_vibe or _mean(completed)
    else:
        n_manual = 0
        manual_vibe_raw = None
        all_mean = _mean(completed)

    # Recent window: last RECENT_VIBE_N plays regardless of skip status, so skips
    # and manual plays within the window all factor in immediately.
    recent_window    = plays[-RECENT_VIBE_N:]
    recent_completed = [p for p in recent_window if p["skip_status"] in ("full", "partial")]
    recent_skipped   = [p for p in recent_window if p["skip_status"] == "skip"]

    if queued_ids:
        recent_q  = [p for p in recent_completed if p["song_id"] in queued_ids]
        recent_m  = [p for p in recent_completed if p["song_id"] not in queued_ids]
        recent_qv = _mean(recent_q)
        recent_mv = _mean(recent_m)
        if recent_mv and recent_qv:
            n_rq = sum(1 for p in recent_q if p["song_id"] in vibe_map)
            n_rm = sum(1 for p in recent_m if p["song_id"] in vibe_map)
            frac = min(n_rm * _RECENT_MANUAL_BOOST / (n_rm * _RECENT_MANUAL_BOOST + n_rq),
                       _RECENT_MANUAL_MAX)
            recent_mean = {ax: round((1 - frac) * recent_qv[ax] + frac * recent_mv[ax], 4)
                           for ax in ("content", "melodic", "bpm")}
        else:
            recent_mean = recent_mv or recent_qv or _mean(recent_completed)
    else:
        recent_mean = _mean(recent_completed)

    # Skip repulsion: push recent_mean away from the mean vibe of recently-skipped songs.
    # Only uses the current window so old skips don't accumulate.
    if recent_mean:
        skip_vibe = _mean(recent_skipped)
        if skip_vibe:
            recent_mean = {
                ax: round(max(-1.0, min(1.0,
                    recent_mean[ax] + _SKIP_REPULSION_W * (recent_mean[ax] - skip_vibe[ax])
                )), 4)
                for ax in ("content", "melodic", "bpm")
            }

    return all_mean, n_valid, recent_mean, n_manual, manual_vibe_raw


def _write_state(
    time_bucket: str,
    eng: dict,
    plays_seen: int,
    session_vibe: tuple = (None, 0, None, 0, None),
    skip_signal: tuple = (None, None, False),
    velocity_zeroed_until: str | None = None,
):
    vibe_data, vibe_n, recent_vibe, n_manual, manual_vibe_raw = session_vibe
    skip_cluster_vibe, recent_comp_vibe, delta_clear = skip_signal
    state = {
        "time_bucket":               time_bucket,
        "engagement_delta":          eng["overall_delta"],
        "recent_skip_rate":          eng["recent_skip_rate"],
        "ever_high_skip":            eng["ever_high_skip"],
        "plays_seen":                plays_seen,
        "session_vibe_n":            vibe_n,
        "session_vibe_content_mean": vibe_data["content"] if vibe_data else None,
        "session_vibe_melodic_mean": vibe_data["melodic"] if vibe_data else None,
        "session_vibe_bpm_mean":     vibe_data["bpm"]     if vibe_data else None,
        "session_vibe_recent_content": recent_vibe["content"] if recent_vibe else None,
        "session_vibe_recent_melodic": recent_vibe["melodic"] if recent_vibe else None,
        "session_vibe_recent_bpm":     recent_vibe["bpm"]     if recent_vibe else None,
        "manual_plays_n":            n_manual,
        "manual_vibe_content":       manual_vibe_raw["content"] if manual_vibe_raw else None,
        "manual_vibe_melodic":       manual_vibe_raw["melodic"] if manual_vibe_raw else None,
        "manual_vibe_bpm":           manual_vibe_raw["bpm"]     if manual_vibe_raw else None,
        "skip_cluster_vibe_content": skip_cluster_vibe["content"] if skip_cluster_vibe else None,
        "skip_cluster_vibe_melodic": skip_cluster_vibe["melodic"] if skip_cluster_vibe else None,
        "skip_cluster_vibe_bpm":     skip_cluster_vibe["bpm"]     if skip_cluster_vibe else None,
        "recent_comp_vibe_content":  recent_comp_vibe["content"]  if recent_comp_vibe else None,
        "recent_comp_vibe_melodic":  recent_comp_vibe["melodic"]  if recent_comp_vibe else None,
        "recent_comp_vibe_bpm":      recent_comp_vibe["bpm"]      if recent_comp_vibe else None,
        "skip_delta_clear":          delta_clear,
        "velocity_zeroed_until":     velocity_zeroed_until,
        "updated_at":                datetime.now(timezone.utc).isoformat(),
    }
    tmp = SESSION_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, SESSION_STATE_PATH)


def read_session_state() -> dict:
    """Read the current session state written by the watcher. Returns {} if missing."""
    try:
        with open(SESSION_STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_rolling_state() -> dict:
    try:
        with open(ROLLING_STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_rolling_state(state: dict):
    tmp = ROLLING_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, ROLLING_STATE_PATH)


def _current_bucket() -> str:
    h = datetime.now().hour
    if h >= 21 or h < 6: return "late_night"
    if h < 11: return "morning"
    return "afternoon"


def _item_uri(playlist_entry: dict) -> str | None:
    """Extract track URI from a playlist item (Spotify API uses 'item' key, older 'track' for compat)."""
    t = playlist_entry.get("track") or playlist_entry.get("item")
    return t.get("uri") if t else None


def _playlist_position_info(sp, playlist_id: str) -> tuple[int | None, list[str]]:
    """
    Single call pair: current_playback + playlist_items.
    Returns (remaining, upcoming_song_ids), or (None, []) if not playing from this playlist.
    Replaces the old _songs_remaining_in_playlist + _get_upcoming_song_ids pair,
    which each made the same two API calls independently.
    """
    try:
        pb = sp.current_playback()
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            _apply_rate_limit(e)
        return None, []
    except Exception:
        return None, []

    if not pb or not pb.get("item") or not pb.get("context"):
        return None, []
    if pb["context"]["uri"] != f"spotify:playlist:{playlist_id}":
        return None, []

    current_uri = pb["item"]["uri"]
    try:
        result = sp.playlist_items(playlist_id, limit=100)
        uris = [_item_uri(t) for t in result["items"]]
        uris = [u for u in uris if u]
        pos  = uris.index(current_uri)
        remaining    = len(uris) - pos - 1
        upcoming_ids = [u.split(":")[-1] for u in uris[pos + 1:]]
        return remaining, upcoming_ids
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            _apply_rate_limit(e)
        return None, []
    except (Exception, ValueError):
        return None, []


def _should_flush(
    session_state: dict,
    upcoming_ids: list[str],
    vibe_map: dict,
    rolling: dict,
) -> bool:
    """
    Returns True when a vibe-shift queue flush is warranted.

    All gates must pass:
    1. skip_delta_clear is True in session state (real vibe mismatch, ≥5 completions)
    2. Signal is new — session state was updated after the last flush action
    3. flush_count < _FLUSH_MAX (stop flushing when we're thrashing)
    4. velocity not currently frozen
    5. upcoming songs (positions 1+, keeping pos 0 as buffer) have mean vibe
       distance > _SKIP_VIBE_DELTA_MIN from the last-5-comp baseline AND
       that distance is in the same direction as the skip cluster
    """
    if not session_state.get("skip_delta_clear", False):
        return False

    # Gate 2: new signal since last flush
    last_flush_signal = rolling.get("last_flush_signal_at")
    updated_at = session_state.get("updated_at", "")
    if last_flush_signal and updated_at <= last_flush_signal:
        return False

    # Gate 3: flush count
    if rolling.get("flush_count", 0) >= _FLUSH_MAX:
        return False

    # Gate 4: velocity not frozen
    vel_frozen = rolling.get("velocity_zeroed_until")
    if vel_frozen:
        try:
            if datetime.fromisoformat(vel_frozen) > datetime.now(timezone.utc):
                return False
        except (ValueError, TypeError):
            pass

    # Need at least one song beyond the buffer to evaluate (upcoming[1:])
    flush_candidates = upcoming_ids[1:]
    if not flush_candidates:
        return False

    # Load vibes for upcoming songs (may not be in the session vibe_map)
    upcoming_vibe_map = _load_vibe_map(flush_candidates)
    if not upcoming_vibe_map:
        return False

    _AXES = ("content", "melodic", "bpm")

    def _mean_v(ids):
        vs = [upcoming_vibe_map[sid] for sid in ids if sid in upcoming_vibe_map]
        if not vs:
            return None
        return {ax: sum(v[i] for v in vs) / len(vs) for i, ax in enumerate(_AXES)}

    upcoming_vibe = _mean_v(flush_candidates)
    if upcoming_vibe is None:
        return False

    # Baseline: last-5-comp mean from session state (same as skip signal baseline)
    comp_vibe = {ax: session_state.get(f"recent_comp_vibe_{ax}") for ax in _AXES}
    skip_vibe = {ax: session_state.get(f"skip_cluster_vibe_{ax}") for ax in _AXES}
    if any(comp_vibe[ax] is None for ax in _AXES) or any(skip_vibe[ax] is None for ax in _AXES):
        return False

    # Gate 5: upcoming mean is distance > 0.3 from comp baseline
    upcoming_dir = {ax: upcoming_vibe[ax] - float(comp_vibe[ax]) for ax in _AXES}
    dist = sum(v ** 2 for v in upcoming_dir.values()) ** 0.5
    if dist < _SKIP_VIBE_DELTA_MIN:
        return False

    # Gate 5b: upcoming is in the same direction as the skip cluster from comp baseline
    skip_dir = {ax: float(skip_vibe[ax]) - float(comp_vibe[ax]) for ax in _AXES}
    dot = sum(upcoming_dir[ax] * skip_dir[ax] for ax in _AXES)
    return dot > 0


def _refill_threshold(plays_count: int) -> int:
    """Refill trigger grows with session depth — more plays = refill earlier."""
    return min(REFILL_MAX, REFILL_REMAINING + plays_count // REFILL_SCALE_EVERY)


def _check_refill(
    sp, plays: list[dict], rolling: dict,
    session_state: dict | None = None,
    vibe_map: dict | None = None,
) -> int | None:
    """
    Append a new 10-song batch to the SmartShuffle Queue playlist when fewer
    than _refill_threshold(plays_count) songs remain after the current track.

    Also handles vibe-shift flushes: when _should_flush returns True, removes
    upcoming songs (keeping pos 0 as a buffer) and triggers an immediate refill
    regardless of remaining count, bypassing the normal MIN_REFILL_INTERVAL guard.

    Returns the current remaining count, or None if not playing / rate limited.
    """
    playlist_id = rolling.get("playlist_id")
    if not playlist_id:
        return None

    threshold = _refill_threshold(len(plays))
    remaining, upcoming_ids = _playlist_position_info(sp, playlist_id)
    if remaining is None:
        return None

    # ── Vibe-shift flush ────────────────────────────────────────────────────
    flush_triggered = False
    if (session_state and vibe_map is not None
            and _should_flush(session_state, upcoming_ids, vibe_map, rolling)):
        flush_candidates = upcoming_ids[1:]  # keep pos 0 as Spotify buffer
        if flush_candidates:
            flush_uris = [f"spotify:track:{sid}" for sid in flush_candidates]
            try:
                sp.playlist_remove_all_occurrences_of_items(playlist_id, flush_uris)
                flush_count = rolling.get("flush_count", 0) + 1
                rolling["flush_count"]           = flush_count
                rolling["last_flush_signal_at"]  = session_state.get("updated_at", "")
                print(
                    f"  [watcher] vibe-shift flush #{flush_count}: removed {len(flush_candidates)}"
                    f" upcoming songs, triggering immediate refill", flush=True,
                )
                flush_triggered = True
                remaining = 0  # force refill branch below

                if flush_count >= _FLUSH_MAX:
                    frozen_until = (
                        datetime.now(timezone.utc)
                        + __import__("datetime").timedelta(minutes=_VELOCITY_FREEZE_MINUTES)
                    ).isoformat()
                    rolling["velocity_zeroed_until"] = frozen_until
                    print(
                        f"  [watcher] flush_count={flush_count} ≥ {_FLUSH_MAX}:"
                        f" velocity frozen until {frozen_until}", flush=True,
                    )
                _write_rolling_state(rolling)
            except Exception as _fe:
                print(f"  [watcher] flush failed: {_fe}", flush=True)

    # ── Normal cooldown guard (skipped on flush) ────────────────────────────
    if not flush_triggered:
        last_refill = rolling.get("last_refill_at")
        if last_refill:
            since = (datetime.now(timezone.utc) - _parse_ts(last_refill)).total_seconds()
            if since < MIN_REFILL_INTERVAL:
                return remaining

        if remaining >= threshold:
            return remaining

    print(f"  [watcher] rolling refill triggered — {remaining} remaining"
          f" (threshold={threshold}, session_plays={len(plays)})", flush=True)

    source_playlist_id = rolling.get("source_playlist_id")
    algorithm          = rolling.get("algorithm", "smartshuffle")

    if algorithm == "random_baseline" and not ALLOW_RB_ROLLING:
        print("  [watcher] RB rolling refill blocked (ALLOW_RB_ROLLING=False).", flush=True)
        return remaining

    context            = _current_bucket()

    # All songs ever generated for this rolling session (initial push + all prior refills).
    # This is the ground truth of what's been presented — Spotify's recently_played
    # misses very quick skips, and plays[-30:] drops songs from early in long sessions.
    session_push_id = rolling.get("session_push_id")
    session_song_ids: list[str] = []
    if session_push_id:
        import sqlite3 as _sqlite3
        conn_exc = _sqlite3.connect(DB_PATH)
        _rows = conn_exc.execute("""
            SELECT q.songs
            FROM queue_pushes qp
            JOIN queues q ON qp.queue_id = q.queue_id
            WHERE COALESCE(qp.rolling_session_id, qp.push_id) = ?
        """, (session_push_id,)).fetchall()
        conn_exc.close()
        for (_songs_json,) in _rows:
            for s in json.loads(_songs_json):
                session_song_ids.append(s["song_id"])

    played_ids  = [p["song_id"] for p in plays[-30:]]
    exclude_ids = list(dict.fromkeys(session_song_ids + played_ids + upcoming_ids))

    phase34_cmd = [sys.executable, os.path.join(DIR, "src", "recommend.py"),
                   "--context", context,
                   "--count",   "10"]
    if algorithm == "random_baseline":
        phase34_cmd += ["--algorithm", "random_baseline"]
    if source_playlist_id:
        phase34_cmd += ["--playlist", source_playlist_id]
    if exclude_ids:
        phase34_cmd += ["--exclude", ",".join(exclude_ids)]

    try:
        subprocess.run(phase34_cmd, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, "stderr", "")[:200]
        print(f"  [watcher] refill generation failed: {stderr}")
        return remaining

    # Fetch newly generated songs and append to playlist
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT queue_id, songs FROM queues WHERE algorithm=? ORDER BY queue_id DESC LIMIT 1",
        (algorithm,)
    ).fetchone()
    conn.close()

    if not row:
        return remaining

    refill_queue_id = row[0]
    new_songs       = json.loads(row[1])
    new_uris        = [f"spotify:track:{s['song_id']}" for s in new_songs]

    try:
        sp.playlist_add_items(playlist_id, new_uris)
        print(f"  [watcher] appended {len(new_uris)} songs  context={context}", flush=True)
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            _apply_rate_limit(e)
        else:
            print(f"  [watcher] playlist append failed: {e}")
        return remaining

    # Record the refill as its own push so collect_data.py can attribute plays
    # from this batch. Without this, only the initial 11 songs ever get attributed.
    # rolling_session_id links this refill to the initial push so the display
    # can stitch all batches into one full queue.
    now = datetime.now(timezone.utc).isoformat()
    session_push_id = rolling.get("session_push_id")
    conn2 = sqlite3.connect(DB_PATH)
    conn2.execute("""
        INSERT INTO queue_pushes (queue_id, algorithm, pushed_at, mode, rolling_session_id)
        VALUES (?, ?, ?, 'rolling', ?)
    """, (refill_queue_id, algorithm, now, session_push_id))
    conn2.commit()
    conn2.close()

    rolling["last_refill_at"] = now
    _write_rolling_state(rolling)
    return remaining


def _clear_state():
    try:
        os.remove(SESSION_STATE_PATH)
    except FileNotFoundError:
        pass


def watch(time_bucket: str, baseline_skip_rate: float, stop_event: threading.Event):
    """
    Main watcher loop. Runs until stop_event is set.

    Args:
        time_bucket:        current time bucket (e.g. "afternoon")
        baseline_skip_rate: historical skip rate for this bucket from engagement_baselines
        stop_event:         set this to stop the loop cleanly
    """
    sp            = _make_sp()
    session_start = datetime.now(timezone.utc)
    _clear_state()

    print(f"  [watcher] started — bucket={time_bucket}"
          f"  baseline_skip_rate={baseline_skip_rate:.1%}", flush=True)

    last_remaining:    int | None = None  # for adaptive interval decisions
    standby:           bool       = False
    standby_since:     float      = 0.0

    while True:
        rolling = _read_rolling_state()

        # Rate-limit sleep — 60s chunks so stop_event stays responsive
        if _is_rate_limited():
            wait_s = max(0, _rate_limit_until - time.time())
            print(f"  [watcher] rate limited — skipping poll, {wait_s:.0f}s remaining")
            if stop_event.wait(timeout=min(wait_s + 1, 60)):
                break
            continue

        # ── Standby mode ──────────────────────────────────────────────────────
        # One cheap current_playback call every 5 min; resume when the
        # SmartShuffle playlist is playing again. Exit after 4 hours of standby.
        if standby:
            if time.time() - standby_since > STANDBY_EXIT_AFTER:
                print("  [watcher] standby timeout (2 h) — stopping")
                break
            if stop_event.wait(timeout=STANDBY_POLL):
                break
            playlist_id = rolling.get("playlist_id")
            try:
                pb = sp.current_playback()
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    _apply_rate_limit(e)
                continue
            except Exception:
                continue
            if (pb and pb.get("is_playing") and pb.get("context") and playlist_id
                    and pb["context"]["uri"] == f"spotify:playlist:{playlist_id}"):
                print("  [watcher] playback resumed — exiting standby", flush=True)
                standby        = False
                last_remaining = None  # force fresh position check
            continue

        # ── Normal mode ───────────────────────────────────────────────────────
        if not rolling.get("enabled"):
            interval = POLL_INTERVAL
        elif last_remaining is not None and last_remaining > _refill_threshold(len(plays)) + 3:
            interval = ROLLING_SLOW_INTERVAL
        else:
            interval = ROLLING_POLL_INTERVAL

        if stop_event.wait(timeout=interval):
            break

        plays      = _fetch_recent(sp, session_start)
        prev_state = read_session_state()
        eng        = _compute_engagement(plays, baseline_skip_rate,
                                         prev_state.get("ever_high_skip", False))
        queued_ids = _get_session_queued_ids(rolling)
        all_song_ids = list(dict.fromkeys(p["song_id"] for p in plays))
        vibe_map   = _load_vibe_map(all_song_ids)
        session_v  = _compute_session_vibe(plays, queued_ids, vibe_map)
        skip_sig   = _compute_skip_signal(plays, vibe_map)
        vel_frozen = rolling.get("velocity_zeroed_until")
        _write_state(time_bucket, eng, len(plays), session_v, skip_sig, vel_frozen)

        if len(plays) >= MIN_PLAYS:
            vibe_data, vibe_n, recent_vibe, n_manual, _manual_vibe_raw = session_v
            v_str = ""
            if vibe_data:
                v_str = (
                    f"  vibe c={vibe_data['content']:+.2f}"
                    f" m={vibe_data['melodic']:+.2f}"
                    f" bpm={vibe_data['bpm']:+.2f}(n={vibe_n})"
                )
                if recent_vibe:
                    v_str += (
                        f"  r10 c={recent_vibe['content']:+.2f}"
                        f" m={recent_vibe['melodic']:+.2f}"
                    )
            manual_str = f"  manual={n_manual}" if n_manual else ""
            print(f"  [watcher] {len(plays)} plays"
                  f"  overall={eng['overall_delta']:+.1%}"
                  f"  recent={eng['recent_skip_rate']:.1%}"
                  f"  ever_high={eng['ever_high_skip']}"
                  f"{manual_str}{v_str}", flush=True)

        # Refill check before standby — ensures we don't miss the threshold
        # if the user paused and the idle timer fires while songs remain.
        if rolling.get("enabled"):
            last_remaining = _check_refill(
                sp, plays, rolling,
                session_state=read_session_state(),
                vibe_map=vibe_map,
            )

        # Enter standby after 10 min of no plays
        if plays:
            idle_s = (datetime.now(timezone.utc) - _parse_ts(plays[-1]["played_at"])).total_seconds()
            if idle_s > INACTIVITY_STANDBY:
                print(f"  [watcher] idle {idle_s / 60:.1f} min — entering standby", flush=True)
                standby       = True
                standby_since = time.time()
                last_remaining = None
                continue

    _clear_state()
    print("  [watcher] stopped")


def start_watcher(time_bucket: str, baseline_skip_rate: float) -> threading.Event:
    """
    Spawn the watcher as a daemon thread. Returns the stop_event so the caller
    can halt the watcher when the session ends.
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=watch,
        args=(time_bucket, baseline_skip_rate, stop_event),
        daemon=True,
        name="session-watcher",
    )
    t.start()
    return stop_event


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=None,
                        help="Override time bucket (default: auto-detect)")
    args = parser.parse_args()

    from recommend import current_time_bucket, load_engagement_baselines, DB_PATH

    bucket = args.bucket or current_time_bucket()
    conn   = sqlite3.connect(DB_PATH)
    baselines = load_engagement_baselines(conn)
    conn.close()

    baseline_rate = baselines.get(bucket, {}).get("baseline_skip_rate", 0.0)

    stop = threading.Event()
    try:
        watch(bucket, baseline_rate, stop)
    except KeyboardInterrupt:
        stop.set()
