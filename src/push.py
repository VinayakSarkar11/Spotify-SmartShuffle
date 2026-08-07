#!/usr/bin/env python3
"""
Pushes the most recent SmartShuffle queue to the active Spotify device.

Usage:
    python play.py               # play most recent SmartShuffle queue
    python play.py --id 12       # play a specific queue_id
    python play.py --list        # list recent SmartShuffle queues
    python play.py --rolling     # mark this session as rolling (session_watcher will refill)

Requires Spotify Premium — playback control is a Premium-only API.
Run once from the CLI before using the Streamlit app's Play button so
the auth cache gets the playback scope.
"""
import argparse
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DIR, "data", "smartshuffle.db")
ROLLING_STATE_PATH = os.path.join(DIR, "data", "rolling_queue_state.json")
WATCHER_PID_PATH   = os.path.join(DIR, "data", "watcher.pid")

RANDOM_BASELINE_FRACTION = 0.0
ALLOW_RB_ROLLING         = False   # set True to re-enable RB rolling sessions

SCOPE = " ".join([
    "user-read-recently-played",
    "user-library-read",
    "playlist-read-private",
    "playlist-modify-private",
    "user-top-read",
    "user-modify-playback-state",
    "user-read-playback-state",
])

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id     = os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI"),
    scope         = SCOPE,
    cache_path    = os.path.join(DIR, ".spotify_cache"),
))


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    """Active device first, otherwise first available, otherwise current playback device."""
    devices = sp.devices().get("devices", [])
    if devices:
        active = [d for d in devices if d["is_active"]]
        return active[0] if active else devices[0]
    # /me/player/devices is unreliable on macOS — fall back to current playback device
    try:
        pb = sp.current_playback()
        if pb and pb.get("device"):
            return pb["device"]
    except Exception:
        pass
    return None


def _init_push_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue_pushes (
            push_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id           INTEGER NOT NULL,
            algorithm          TEXT NOT NULL,
            pushed_at          TEXT NOT NULL,
            mode               TEXT NOT NULL DEFAULT 'full',
            rolling_session_id INTEGER
        )
    """)
    for col, defn in [
        ("mode",               "TEXT NOT NULL DEFAULT 'full'"),
        ("rolling_session_id", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE queue_pushes ADD COLUMN {col} {defn}")
        except Exception:
            pass
    conn.commit()


def _ensure_config_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()


def _get_or_create_push_playlist() -> str:
    """
    Find or create the 'SmartShuffle Queue' private playlist.
    Caches the playlist_id in the DB so we only search once.
    """
    conn = sqlite3.connect(DB_PATH)
    _ensure_config_table(conn)
    row = conn.execute("SELECT value FROM config WHERE key='push_playlist_id'").fetchone()
    if row:
        conn.close()
        return row[0]

    user_id = sp.me()["id"]
    offset = 0
    while True:
        result = sp.current_user_playlists(limit=50, offset=offset)
        for pl in (result.get("items") or []):
            if pl and pl["name"] == "SmartShuffle Queue" and pl["owner"]["id"] == user_id:
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES ('push_playlist_id', ?)",
                    (pl["id"],)
                )
                conn.commit()
                conn.close()
                return pl["id"]
        if not result.get("next"):
            break
        offset += 50

    # spotipy's user_playlist_create uses /users/{id}/playlists which now returns 403;
    # use the /me/playlists endpoint directly instead.
    pl = sp._post("me/playlists", payload={
        "name": "SmartShuffle Queue",
        "public": False,
        "description": "Auto-managed by SmartShuffle — do not edit manually",
    })
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES ('push_playlist_id', ?)",
        (pl["id"],)
    )
    conn.commit()
    conn.close()
    print(f"Created 'SmartShuffle Queue' playlist ({pl['id']})")
    return pl["id"]


def _push_to_playlist(playlist_id: str, uris: list[str]):
    """Replace playlist contents with uris. Handles Spotify's 100-URI-per-call limit."""
    sp.playlist_replace_items(playlist_id, uris[:100])
    for i in range(100, len(uris), 100):
        sp.playlist_add_items(playlist_id, uris[i:i + 100])


def _write_rolling_state(state: dict):
    tmp = ROLLING_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, ROLLING_STATE_PATH)


def _pick_algorithm() -> str:
    return "random_baseline" if random.random() < RANDOM_BASELINE_FRACTION else "smartshuffle"


def load_queue(queue_id: int | None, algorithm: str = "smartshuffle") -> tuple | None:
    conn = sqlite3.connect(DB_PATH)
    if queue_id:
        row = conn.execute("""
            SELECT q.queue_id, q.songs, p.playlist_name, q.context, q.ab_label,
                   q.algorithm, q.playlist_id
            FROM queues q
            LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
            WHERE q.queue_id = ?
        """, (queue_id,)).fetchone()
    else:
        row = conn.execute("""
            SELECT q.queue_id, q.songs, p.playlist_name, q.context, q.ab_label,
                   q.algorithm, q.playlist_id
            FROM queues q
            LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
            WHERE q.algorithm = ?
            ORDER BY q.queue_id DESC LIMIT 1
        """, (algorithm,)).fetchone()
    conn.close()
    return row


# ── Commands ──────────────────────────────────────────────────────────────────

def play(queue_id: int | None = None, algorithm: str | None = None,
         device_id: str | None = None, rolling: bool = False):
    if algorithm is None:
        algorithm = _pick_algorithm()

    if rolling and algorithm == "random_baseline" and not ALLOW_RB_ROLLING:
        print("RB rolling is disabled (ALLOW_RB_ROLLING=False in push.py). Set to True to re-enable.")
        return

    row = load_queue(queue_id, algorithm)
    if not row:
        print("No queue found. Run phase34.py first.")
        return

    q_id, songs_json, pl_name, context, _, algorithm, source_playlist_id = row
    songs = json.loads(songs_json)
    uris  = [f"spotify:track:{s['song_id']}" for s in songs]

    label = "SmartShuffle" if algorithm == "smartshuffle" else "Random baseline"
    print(f"\n[{label}]  Queue {q_id}  [{pl_name or '?'}  ·  {context}]")
    for i, s in enumerate(songs, 1):
        print(f"  {i:2}. {s['song_name']:<42} {s['artist_name']}")

    all_devices = sp.devices().get("devices", [])
    if not all_devices:
        # /me/player/devices unreliable on macOS — pull device from current playback
        try:
            pb = sp.current_playback()
            if pb and pb.get("device"):
                all_devices = [pb["device"]]
        except Exception:
            pass

    if not all_devices:
        # Last resort: use the most recently seen device from config table
        _cfg_conn = sqlite3.connect(DB_PATH)
        _ensure_config_table(_cfg_conn)
        _last = _cfg_conn.execute(
            "SELECT value FROM config WHERE key='last_device_id'"
        ).fetchone()
        _last_name = _cfg_conn.execute(
            "SELECT value FROM config WHERE key='last_device_name'"
        ).fetchone()
        _cfg_conn.close()
        if _last:
            print(f"  Spotify API returned no devices; falling back to last known: "
                  f"{_last_name[0] if _last_name else _last[0]}")
            all_devices = [{"id": _last[0], "name": _last_name[0] if _last_name else "last known device",
                            "is_active": False, "type": "Computer"}]

    use_applescript = False
    if not all_devices:
        # API is completely dark — check if the desktop app is at least running
        try:
            import subprocess as _subprocess
            _as = _subprocess.run(
                ["osascript", "-e", 'tell application "Spotify" to player state as string'],
                capture_output=True, text=True, timeout=3,
            )
            if _as.returncode == 0 and _as.stdout.strip() in ("playing", "paused", "stopped"):
                print("  Spotify API unreachable but desktop app is running; will use AppleScript for playback.")
                use_applescript = True
                all_devices = [{"id": "applescript", "name": "Spotify (AppleScript)",
                                "is_active": True, "type": "Computer"}]
        except Exception:
            pass

    if device_id:
        device = next((d for d in all_devices if d["id"] == device_id), None)
        if not device:
            device = all_devices[0] if all_devices else None
            if device:
                print(f"  Selected device not active; falling back to {device['name']}")
            else:
                print("\nNo Spotify devices found. Open Spotify and play a song first.")
                raise SystemExit(1)
    else:
        active = [d for d in all_devices if d["is_active"]]
        device = active[0] if active else (all_devices[0] if all_devices else None)
        if not device:
            print("\nNo Spotify devices found. Open Spotify and play a song first.")
            raise SystemExit(1)

    tag = " (active)" if device["is_active"] else " (will transfer playback)"
    print(f"\nDevice: {device['name']}{tag}")

    # Persist device so we can fall back to it when the Spotify API goes dark
    if not use_applescript:
        _save_conn = sqlite3.connect(DB_PATH)
        _ensure_config_table(_save_conn)
        _save_conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('last_device_id', ?)", (device["id"],)
        )
        _save_conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('last_device_name', ?)", (device["name"],)
        )
        _save_conn.commit()
        _save_conn.close()

    # Get/create the persistent push playlist and populate it
    push_playlist_id = _get_or_create_push_playlist()
    push_playlist_uri = f"spotify:playlist:{push_playlist_id}"

    import time

    # Pause first — forces Spotify to fully restart the context rather than
    # continuing from wherever it was in the same playlist.
    if not use_applescript:
        try:
            sp.pause_playback(device_id=device["id"])
            time.sleep(0.5)
        except spotipy.exceptions.SpotifyException:
            pass  # already paused or not playing — fine

    print(f"Updating SmartShuffle Queue playlist ({len(songs)} songs)...")
    _push_to_playlist(push_playlist_id, uris)
    time.sleep(1.5)  # wait for Spotify's backend to propagate new tracks

    if not use_applescript:
        try:
            sp.shuffle(False, device_id=device["id"])
        except spotipy.exceptions.SpotifyException:
            pass  # non-fatal

    if not use_applescript:
        try:
            # Use offset uri (not position) so Spotify unambiguously starts from
            # song 1 even when the same playlist was already the active context.
            sp.start_playback(
                device_id=device["id"],
                context_uri=push_playlist_uri,
                offset={"uri": uris[0]},
            )
        except spotipy.exceptions.SpotifyException as e:
            err = str(e)
            if "403" in err or "PREMIUM" in err.upper():
                print("\nSpotify Premium required for playback control via API.")
                raise SystemExit(1)
            elif "404" in err:
                # Stale device ID — fall back to AppleScript if desktop app is running
                print(f"  Device {device['id'][:8]}… not found (stale ID); trying AppleScript…")
                use_applescript = True
            else:
                print(f"\nPlayback error: {e}")
                raise SystemExit(1)

        if not use_applescript:
            # Second call (resume without context) ensures playback actually starts —
            # start_playback with context_uri can leave the device in a paused state on macOS.
            time.sleep(0.5)
            try:
                sp.start_playback(device_id=device["id"])
            except spotipy.exceptions.SpotifyException:
                pass  # non-fatal: context already set, user can press play manually

    if use_applescript:
        import subprocess as _subprocess
        _as_cmd = f'tell application "Spotify" to play track "{push_playlist_uri}"'
        _as_result = _subprocess.run(["osascript", "-e", _as_cmd], capture_output=True, text=True)
        if _as_result.returncode != 0:
            print(f"\nAppleScript playback failed: {_as_result.stderr.strip()}")
            raise SystemExit(1)
        print("Playback started via AppleScript.")

    # Record the push for play attribution
    conn = sqlite3.connect(DB_PATH)
    _init_push_table(conn)
    now_iso   = datetime.now(timezone.utc).isoformat()
    push_mode = "rolling" if rolling else "full"

    # Every user-initiated push always starts a fresh rolling session.
    # Watcher refills (watcher.py _check_refill) do their own INSERT with the
    # correct rolling_session_id — they never go through this code path.
    rolling_session_id = None

    conn.execute(
        "INSERT INTO queue_pushes (queue_id, algorithm, pushed_at, mode, rolling_session_id)"
        " VALUES (?,?,?,?,?)",
        (q_id, algorithm, now_iso, push_mode, rolling_session_id),
    )
    # For the first push in a rolling session, set rolling_session_id = its own push_id
    session_push_id = rolling_session_id  # may already be set (rejoining existing session)
    if rolling and rolling_session_id is None:
        new_push_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "UPDATE queue_pushes SET rolling_session_id = ? WHERE push_id = ?",
            (new_push_id, new_push_id),
        )
        session_push_id = new_push_id
    conn.commit()
    conn.close()

    # Write rolling state so session_watcher knows what to monitor
    _write_rolling_state({
        "enabled":            rolling,
        "playlist_id":        push_playlist_id,
        "source_playlist_id": source_playlist_id,
        "algorithm":          algorithm,
        "device_id":          device["id"],
        "last_refill_at":     None,
        "session_push_id":    session_push_id,
    })

    print(f"\nPlaying: {songs[0]['song_name']} — {songs[0]['artist_name']}")
    print(f"+ {len(songs) - 1} songs in queue")

    if rolling:
        _start_watcher()
        print("Rolling mode active — watcher monitoring skip rate and playlist.")


def _start_watcher():
    """Kill any running watcher and start a fresh one for the new session."""
    if os.path.exists(WATCHER_PID_PATH):
        try:
            with open(WATCHER_PID_PATH) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, signal.SIGTERM)
        except (ProcessLookupError, ValueError, OSError):
            pass

    log_path = os.path.join(DIR, "logs", "watcher.log")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(DIR, "src", "watcher.py")],
        start_new_session=True,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
    )
    with open(WATCHER_PID_PATH, "w") as f:
        f.write(str(proc.pid))
    print(f"Session watcher started (PID {proc.pid}, log: watcher.log)")


def list_queues():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT qp.push_id, qp.pushed_at, qp.algorithm, qp.mode,
               json_array_length(q.songs) AS n_songs,
               p.playlist_name, q.context
        FROM queue_pushes qp
        JOIN queues q ON q.queue_id = qp.queue_id
        LEFT JOIN playlists p ON q.playlist_id = p.playlist_id
        WHERE qp.mode = 'full' OR qp.push_id = qp.rolling_session_id
        ORDER BY qp.push_id DESC LIMIT 20
    """).fetchall()
    conn.close()

    if not rows:
        print("No pushes yet.")
        return

    print(f"\n  {'Push':>4}  {'Pushed':^16}  {'Algorithm':<18}  {'Mode':<8}  {'Songs':>5}  {'Playlist':<22}  Context")
    print(f"  {'─'*4}  {'─'*16}  {'─'*18}  {'─'*8}  {'─'*5}  {'─'*22}  {'─'*10}")
    for push_id, pushed_at, algo, mode, n_songs, pl, ctx in rows:
        algo_label = "SmartShuffle" if algo == "smartshuffle" else "Random baseline"
        print(f"  {push_id:>4}  {pushed_at[:16]}  {algo_label:<18}  {mode:<8}  {n_songs:>5}  {(pl or '?'):<22}  {ctx}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Play a SmartShuffle queue on Spotify"
    )
    parser.add_argument("--id",        type=int, help="Queue ID to play (default: most recent)")
    parser.add_argument("--algorithm", choices=["smartshuffle", "random_baseline"],
                        help="Force a specific algorithm instead of random 60/40 draw")
    parser.add_argument("--device-id", help="Spotify device ID to play on")
    parser.add_argument("--rolling",   action="store_true",
                        help="Enable rolling mode: session_watcher will auto-refill playlist")
    parser.add_argument("--list",      action="store_true", help="List recent queues")
    args = parser.parse_args()

    if args.list:
        list_queues()
    else:
        play(args.id, args.algorithm, args.device_id, rolling=args.rolling)


if __name__ == "__main__":
    main()
