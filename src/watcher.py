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


def _compute_session_vibe(plays: list[dict]) -> tuple[dict | None, int]:
    """
    Mean vibe (content, melodic, bpm) of completed songs so far in this session.
    Used by generate_queue() to drift the vibe target toward what's actually been
    completing, so rolling refills follow the session instead of resetting.

    Returns ({content, melodic, bpm}, n_completed). (None, 0) when no completed
    plays with vibe scores yet. Skipped songs excluded — strong rejection signal.
    """
    completed_ids = [p["song_id"] for p in plays if p["skip_status"] in ("full", "partial")]
    if not completed_ids:
        return None, 0

    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(DB_PATH)
        ph   = ",".join("?" * len(completed_ids))
        rows = conn.execute(
            f"SELECT vibe_content, vibe_melodic, vibe_bpm "
            f"FROM songs WHERE song_id IN ({ph})",
            completed_ids,
        ).fetchall()
        conn.close()
    except Exception:
        return None, 0

    valid = [(r[0], r[1], r[2]) for r in rows
             if r[0] is not None and r[1] is not None and r[2] is not None]
    if not valid:
        return None, 0

    mean_vibe = {
        "content": round(sum(v[0] for v in valid) / len(valid), 4),
        "melodic": round(sum(v[1] for v in valid) / len(valid), 4),
        "bpm":     round(sum(v[2] for v in valid) / len(valid), 4),
    }
    return mean_vibe, len(valid)


def _write_state(time_bucket: str, eng: dict, plays_seen: int,
                 session_vibe: tuple[dict | None, int] = (None, 0)):
    vibe_data, vibe_n = session_vibe
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


def _refill_threshold(plays_count: int) -> int:
    """Refill trigger grows with session depth — more plays = refill earlier."""
    return min(REFILL_MAX, REFILL_REMAINING + plays_count // REFILL_SCALE_EVERY)


def _check_refill(sp, plays: list[dict], rolling: dict) -> int | None:
    """
    Append a new 10-song batch to the SmartShuffle Queue playlist when fewer
    than _refill_threshold(plays_count) songs remain after the current track.
    Threshold grows with session depth: deeper sessions skip more, so we refill
    earlier to avoid running out.
    Returns the current remaining count for the caller to use in adaptive polling,
    or None if not playing from this playlist / rate limited.
    """
    playlist_id = rolling.get("playlist_id")
    if not playlist_id:
        return None

    # Don't double-refill within MIN_REFILL_INTERVAL seconds
    last_refill = rolling.get("last_refill_at")
    if last_refill:
        since = (datetime.now(timezone.utc) - _parse_ts(last_refill)).total_seconds()
        if since < MIN_REFILL_INTERVAL:
            return None  # caller keeps last known remaining for interval decisions

    threshold = _refill_threshold(len(plays))
    remaining, upcoming_ids = _playlist_position_info(sp, playlist_id)
    if remaining is None or remaining >= threshold:
        return remaining

    print(f"  [watcher] rolling refill triggered — {remaining} remaining"
          f" (threshold={threshold}, session_plays={len(plays)})", flush=True)

    source_playlist_id = rolling.get("source_playlist_id")
    algorithm          = rolling.get("algorithm", "smartshuffle")
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
        session_v  = _compute_session_vibe(plays)
        _write_state(time_bucket, eng, len(plays), session_v)

        if len(plays) >= MIN_PLAYS:
            vibe_data, vibe_n = session_v
            v_str = (
                f"  vibe c={vibe_data['content']:+.2f}"
                f" m={vibe_data['melodic']:+.2f}"
                f" bpm={vibe_data['bpm']:+.2f}(n={vibe_n})"
                if vibe_data else ""
            )
            print(f"  [watcher] {len(plays)} plays"
                  f"  overall={eng['overall_delta']:+.1%}"
                  f"  recent={eng['recent_skip_rate']:.1%}"
                  f"  ever_high={eng['ever_high_skip']}"
                  f"{v_str}", flush=True)

        # Refill check before standby — ensures we don't miss the threshold
        # if the user paused and the idle timer fires while songs remain.
        if rolling.get("enabled"):
            last_remaining = _check_refill(sp, plays, rolling)

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
