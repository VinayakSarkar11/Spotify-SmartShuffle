import os
import time
import json
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

# ── Auth ────────────────────────────────────────────────────────────────────

SCOPE = " ".join([
    "user-read-recently-played",
    "user-library-read",
    "playlist-read-private",
    "user-top-read",
    "user-modify-playback-state",   # needed by play.py
    "user-read-playback-state",     # needed by play.py
])

def _make_sp():
    token = os.getenv("SS_ACCESS_TOKEN")
    if token:
        return spotipy.Spotify(auth=token)
    cache_path = os.getenv("SS_TOKEN_CACHE_PATH") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".spotify_cache"
    )
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=SCOPE,
        cache_path=cache_path,
    ))

sp = _make_sp()

# ── Database setup ───────────────────────────────────────────────────────────

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS plays (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at           TEXT UNIQUE,        -- ISO timestamp; unique prevents duplicates
            song_id             TEXT NOT NULL,
            song_name           TEXT,
            artist_name         TEXT,
            duration_ms         INTEGER,
            play_duration_ms    INTEGER,            -- inferred from gap to next play
            inferred_skip       TEXT,               -- 'skip' | 'partial' | 'full' | 'unknown'
            play_source         TEXT,               -- 'playlist' | 'manual_search' | 'artist_browse' | 'album_browse' | 'jam_excluded'
            context_type        TEXT,               -- raw from Spotify
            context_uri         TEXT,               -- raw from Spotify
            audio_features      TEXT,               -- JSON blob
            collected_at        TEXT                -- when we polled this
        );

        CREATE TABLE IF NOT EXISTS songs (
            song_id             TEXT PRIMARY KEY,
            song_name           TEXT,
            artist_name         TEXT,
            artist_id           TEXT,
            album_name          TEXT,
            duration_ms         INTEGER,
            audio_features      TEXT,               -- JSON blob: popularity, genres, explicit, etc.
            fetched_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS playlists (
            playlist_id         TEXT PRIMARY KEY,
            playlist_name       TEXT,
            track_count         INTEGER,
            snapshot_id         TEXT,
            fetched_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS playlist_tracks (
            playlist_id         TEXT,
            song_id             TEXT,
            added_at            TEXT,
            PRIMARY KEY (playlist_id, song_id)
        );

        CREATE INDEX IF NOT EXISTS idx_plays_source    ON plays(play_source);
        CREATE INDEX IF NOT EXISTS idx_plays_played_at ON plays(played_at);
        CREATE INDEX IF NOT EXISTS idx_plays_song_id   ON plays(song_id);
    """)
    try:
        conn.execute("ALTER TABLE playlists ADD COLUMN snapshot_id TEXT")
    except Exception:
        pass
    conn.commit()


# ── Helpers ──────────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── Play source classification ───────────────────────────────────────────────

def classify_play_source(context):
    """
    Derives play_source from Spotify context object.
    null context = user searched and played directly.
    """
    if context is None:
        return "manual_search"
    t = context.get("type", "")
    if t == "playlist":
        return "playlist"
    if t == "artist":
        return "artist_browse"
    if t == "album":
        return "album_browse"
    return "manual_search"


def flag_jam_sessions(plays):
    """
    Heuristic: a session is a suspected Jam if it has:
      - >= 4 plays
      - median play duration < 60s (someone else skipping quickly)
      - >= 3 distinct artists in quick succession
    Mutates play_source in place.
    """
    if len(plays) < 4:
        return plays

    # Group into sessions (gap > 30 min = new session)
    sessions = []
    current = [plays[0]]
    for i in range(1, len(plays)):
        prev_ts = datetime.fromisoformat(plays[i - 1]["played_at"].replace("Z", "+00:00"))
        curr_ts = datetime.fromisoformat(plays[i]["played_at"].replace("Z", "+00:00"))
        gap = abs((curr_ts - prev_ts).total_seconds())
        if gap > 1800:
            sessions.append(current)
            current = [plays[i]]
        else:
            current.append(plays[i])
    sessions.append(current)

    result = []
    for session in sessions:
        if len(session) >= 4:
            durations = [p["play_duration_ms"] or 0 for p in session]
            sorted_d = sorted(durations)
            median = sorted_d[len(sorted_d) // 2]
            artists = set(p["artist_name"] for p in session)
            if median < 60_000 and len(artists) >= 3:
                for p in session:
                    p["play_source"] = "jam_excluded"
        result.extend(session)
    return result


# ── Skip inference ───────────────────────────────────────────────────────────

def infer_skip(play_duration_ms, song_duration_ms):
    """
    Infers skip status from how much of the song was played.
    play_duration_ms is estimated from the gap to the next track.
    """
    if play_duration_ms is None or song_duration_ms is None or song_duration_ms == 0:
        return "unknown"
    if play_duration_ms < 1_000:  # <1s is a Spotify UI glitch, not a real play
        return "unknown"
    ratio = play_duration_ms / song_duration_ms
    if ratio < 0.30:
        return "skip"
    if ratio < 0.80:
        return "partial"
    return "full"


# ── Track metadata (replaces audio features — deprecated Nov 2024) ───────────

def fetch_track_metadata(conn, song_ids):
    """
    Fetches per-song metadata using single-track and single-artist endpoints.
    Batch endpoints (/v1/tracks, /v1/audio-features) are 403 for new apps
    following Spotify's Nov 2024 + Feb 2026 API restrictions.
    Falls back gracefully on any per-song error.
    """
    existing = set(
        row[0] for row in conn.execute(
            "SELECT song_id FROM songs WHERE audio_features IS NOT NULL"
        ).fetchall()
    )
    missing = [sid for sid in set(song_ids) if sid not in existing]
    if not missing:
        return

    print(f"  Fetching metadata for {len(missing)} songs...")
    for song_id in missing:
        try:
            track = sp.track(song_id)
            artist_id = track["artists"][0]["id"] if track.get("artists") else None
            genres = []
            if artist_id:
                try:
                    artist = sp.artist(artist_id)
                    genres = artist.get("genres", [])
                except Exception:
                    pass

            features = {
                "popularity":   track.get("popularity"),
                "explicit":     track.get("explicit"),
                "genres":       genres,
                "artist_id":    artist_id,
                "album_name":   track["album"]["name"] if track.get("album") else None,
                "release_date": track["album"].get("release_date") if track.get("album") else None,
            }
            conn.execute("""
                INSERT INTO songs (song_id, song_name, artist_name, duration_ms, audio_features, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    audio_features = excluded.audio_features,
                    fetched_at     = excluded.fetched_at
            """, (
                song_id,
                track.get("name"),
                track["artists"][0]["name"] if track.get("artists") else None,
                track.get("duration_ms"),
                json.dumps(features),
                now_iso()
            ))
            conn.commit()
            time.sleep(0.05)   # stay well under rate limits
        except Exception as e:
            print(f"  Skipping {song_id}: {e}")
            continue


# ── Recently played ──────────────────────────────────────────────────────────

def collect_recently_played(conn):
    """
    Pulls up to 50 recently played tracks, infers play duration from
    inter-play gaps, classifies play source, flags Jam sessions, and
    writes to the plays table. Returns number of new rows inserted.
    """
    results = sp.current_user_recently_played(limit=50)
    items = results.get("items", [])

    if not items:
        print("No recent plays found.")
        return 0

    plays = []
    for item in items:
        track = item["track"]
        context = item.get("context")
        played_at = item["played_at"]

        plays.append({
            "played_at":        played_at,
            "song_id":          track["id"],
            "song_name":        track["name"],
            "artist_name":      track["artists"][0]["name"] if track["artists"] else None,
            "duration_ms":      track["duration_ms"],
            "play_duration_ms": None,
            "inferred_skip":    "unknown",
            "play_source":      classify_play_source(context),
            "context_type":     context.get("type") if context else None,
            "context_uri":      context.get("uri") if context else None,
            "audio_features":   None,
            "collected_at":     now_iso(),
        })

    # Infer play duration from gap between consecutive plays.
    # Spotify returns newest-first; reverse for chronological order.
    plays.reverse()
    for i in range(len(plays) - 1):
        curr_ts = datetime.fromisoformat(plays[i]["played_at"].replace("Z", "+00:00"))
        next_ts = datetime.fromisoformat(plays[i + 1]["played_at"].replace("Z", "+00:00"))
        gap_ms = int((next_ts - curr_ts).total_seconds() * 1000)
        plays[i]["play_duration_ms"] = min(gap_ms, plays[i]["duration_ms"] or gap_ms)
        plays[i]["inferred_skip"] = infer_skip(
            plays[i]["play_duration_ms"], plays[i]["duration_ms"]
        )
    plays[-1]["play_duration_ms"] = None
    plays[-1]["inferred_skip"] = "unknown"

    # Flag suspected Jam sessions
    plays = flag_jam_sessions(plays)

    # Fetch metadata for any new songs
    fetch_track_metadata(conn, [p["song_id"] for p in plays])

    # Attach metadata to each play row for denormalised storage
    features_map = {
        row[0]: row[1]
        for row in conn.execute("SELECT song_id, audio_features FROM songs").fetchall()
    }
    for p in plays:
        p["audio_features"] = features_map.get(p["song_id"])

    # Insert — IGNORE on UNIQUE played_at to avoid duplicates across polls
    inserted = 0
    for p in plays:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO plays
                (played_at, song_id, song_name, artist_name, duration_ms,
                 play_duration_ms, inferred_skip, play_source,
                 context_type, context_uri, audio_features, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p["played_at"], p["song_id"], p["song_name"], p["artist_name"],
                p["duration_ms"], p["play_duration_ms"], p["inferred_skip"],
                p["play_source"], p["context_type"], p["context_uri"],
                p["audio_features"], p["collected_at"]
            ))
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                inserted += 1
        except Exception as e:
            print(f"  Error inserting {p['song_name']}: {e}")

    conn.commit()
    return inserted


# ── Playlists ────────────────────────────────────────────────────────────────

def collect_playlists(conn):
    """
    Pulls all user playlists and their tracks. Stores basic song info from
    the playlist response (no extra API calls needed here — metadata is
    fetched lazily via fetch_track_metadata when plays come in).
    """
    results = sp.current_user_playlists(limit=50)
    playlists = results.get("items", [])

    print(f"  Found {len(playlists)} playlists")
    all_song_ids = []

    for pl in playlists:
        if not pl:
            continue
        pl_id        = pl["id"]
        pl_name      = pl["name"]
        new_snapshot = pl.get("snapshot_id")
        track_count  = (pl.get("tracks") or {}).get("total", 0)

        # Skip the app-managed playlist — its songs aren't user library content
        if pl_name == "SmartShuffle Queue":
            continue

        stored = conn.execute(
            "SELECT snapshot_id FROM playlists WHERE playlist_id = ?", (pl_id,)
        ).fetchone()

        conn.execute("""
            INSERT INTO playlists (playlist_id, playlist_name, track_count, snapshot_id, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(playlist_id) DO UPDATE SET
                playlist_name = excluded.playlist_name,
                track_count   = excluded.track_count,
                snapshot_id   = excluded.snapshot_id,
                fetched_at    = excluded.fetched_at
        """, (pl_id, pl_name, track_count, new_snapshot, now_iso()))

        if stored and stored[0] == new_snapshot:
            print(f"    Unchanged: {pl_name}")
            continue

        print(f"    Fetching: {pl_name}")

        # Paginate through tracks, collecting the full current set from Spotify
        offset = 0
        current_ids = []
        while True:
            tracks = sp.playlist_tracks(pl_id, limit=100, offset=offset)
            for item in tracks.get("items", []):
                # Spotify renamed "track" → "item" in playlist responses; fall back for older clients
                track = item.get("item") or item.get("track")
                if not track or not track.get("id"):
                    continue

                conn.execute("""
                    INSERT OR IGNORE INTO playlist_tracks (playlist_id, song_id, added_at)
                    VALUES (?, ?, ?)
                """, (pl_id, track["id"], item.get("added_at")))

                # Store basic song info from the playlist response —
                # no extra API call needed at this stage
                conn.execute("""
                    INSERT OR IGNORE INTO songs
                        (song_id, song_name, artist_name, duration_ms, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    track["id"],
                    track.get("name"),
                    track["artists"][0]["name"] if track.get("artists") else None,
                    track.get("duration_ms"),
                    now_iso()
                ))
                current_ids.append(track["id"])
                all_song_ids.append(track["id"])

            if not tracks.get("next"):
                break
            offset += 100
            time.sleep(0.1)

        # Remove any tracks that were deleted from the Spotify playlist since last sync
        if current_ids:
            ph = ",".join("?" * len(current_ids))
            removed = conn.execute(
                f"DELETE FROM playlist_tracks WHERE playlist_id = ? AND song_id NOT IN ({ph})",
                [pl_id] + current_ids,
            ).rowcount
            if removed:
                print(f"      Removed {removed} deleted track(s) from {pl_name}")

    conn.commit()
    print(f"  Stored {len(set(all_song_ids))} unique songs across all playlists.")
    print("  Note: full metadata (genres, popularity) fetched lazily as songs appear in plays.")
    print("  Playlists collected.")


# ── Cold start bootstrap ─────────────────────────────────────────────────────

def bootstrap_cold_start(conn):
    """
    Seeds the songs table from Spotify top tracks for users with no play history.

    These songs don't appear in playlist_tracks (so they won't be queued), but
    they ARE inserted into songs so phase2.py can fetch Last.fm tags for them.
    A richer tag pool gives model.py a better energy distribution to use as a
    cold-start prior for context targets (see seed_context_targets_from_tags).

    LIMITATION: top tracks carry no play timestamps, so we cannot assign them
    to time buckets. The energy distribution is used as a global offset only —
    see model.py for how it shifts default context targets.

    Only called when plays table is empty (true first run).
    """
    inserted = 0
    for time_range in ("short_term", "medium_term"):
        try:
            results = sp.current_user_top_tracks(time_range=time_range, limit=50)
        except Exception as e:
            print(f"  Could not fetch top tracks ({time_range}): {e}")
            continue
        for track in results.get("items", []):
            if not track or not track.get("id"):
                continue
            conn.execute("""
                INSERT OR IGNORE INTO songs
                    (song_id, song_name, artist_name, duration_ms, fetched_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                track["id"],
                track.get("name"),
                track["artists"][0]["name"] if track.get("artists") else None,
                track.get("duration_ms"),
                now_iso(),
            ))
            inserted += 1
    conn.commit()
    print(f"  Seeded {inserted} top tracks into songs table.")
    print("  Run phase2.py — it will fetch energy tags for these and calibrate")
    print("  initial context targets without needing play history.")


# ── Queue push attribution + session analysis ─────────────────────────────────

def _init_session_tables(conn):
    """
    Adds completion-tracking columns to queue_pushes and creates the
    session_interjections table. Safe to call multiple times (idempotent).
    """
    for col in [
        "completion_rate REAL",
        "songs_played INTEGER",
        "abandoned_at INTEGER",
        "analyzed_at TEXT",
        "skips_inferred_at TEXT",
    ]:
        try:
            conn.execute(f"ALTER TABLE queue_pushes ADD COLUMN {col}")
        except Exception:
            pass  # column already exists

    # Migrate: old schema had UNIQUE(push_id, song_id, interjected_at) which allowed the same
    # play to be recorded as an interjection in multiple overlapping push windows.
    # New schema uses UNIQUE(song_id, interjected_at) so each play is attributed once.
    old = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='session_interjections'"
    ).fetchone()
    if old and "push_id, song_id" in (old[0] or ""):
        conn.execute("DROP TABLE session_interjections")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_interjections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            push_id        INTEGER NOT NULL,
            song_id        TEXT    NOT NULL,
            interjected_at TEXT    NOT NULL,
            queue_before   INTEGER,
            queue_after    INTEGER,
            context_energy REAL,
            context_label  TEXT,
            UNIQUE (song_id, interjected_at)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue_skips (
            push_id        INTEGER NOT NULL,
            song_id        TEXT    NOT NULL,
            algorithm      TEXT    NOT NULL,
            inferred_at    TEXT    NOT NULL,
            queue_position INTEGER,
            PRIMARY KEY (push_id, song_id)
        )
    """)
    # Migration: add queue_position to existing tables and backfill from queues JSON.
    try:
        conn.execute("ALTER TABLE queue_skips ADD COLUMN queue_position INTEGER")
    except sqlite3.OperationalError:
        pass
    for push_id, song_id in conn.execute(
        "SELECT push_id, song_id FROM queue_skips WHERE queue_position IS NULL"
    ).fetchall():
        row = conn.execute("""
            SELECT q.songs FROM queues q
            JOIN queue_pushes qp ON qp.queue_id = q.queue_id
            WHERE qp.push_id = ?
        """, (push_id,)).fetchone()
        if row:
            ids = [s["song_id"] for s in json.loads(row[0])]
            if song_id in ids:
                conn.execute(
                    "UPDATE queue_skips SET queue_position = ? WHERE push_id = ? AND song_id = ?",
                    (ids.index(song_id), push_id, song_id),
                )
    conn.commit()


def _hour_to_bucket(hour: int) -> str:
    if hour >= 21 or hour < 6: return "late_night"
    if hour < 11:              return "morning"
    return "afternoon"


def _sqlite_ts(ts: str) -> str:
    """
    Normalize any ISO-8601 timestamp to the bare 'YYYY-MM-DD HH:MM:SS' form
    that SQLite's datetime() functions understand.

    Spotify stores played_at as '2024-01-01T20:00:00.000Z'.
    Python isoformat() stores pushed_at as '2026-01-01T08:00:00+00:00'.
    Slicing to 19 chars then swapping T→space handles both without needing
    any regex or tz-aware parsing.
    """
    return ts[:19].replace("T", " ")


MAX_INTERJECTION_GAP_MIN = 15  # null-context play must be within 15 min of neighboring queue songs


def analyze_queue_session(conn, push_id: int, algorithm: str, pushed_at: str, songs_json: str):
    """
    Analyzes the play sequence against a pushed queue to compute:
      - completion_rate  : fraction of queue songs actually played
      - songs_played     : count of unique queue songs heard
      - abandoned_at     : queue index where the user truly left the queue
                           (NULL if they completed or only had interjections)

    Also records interjections — non-queue songs inserted mid-session where
    the queue was later resumed. Three classes of plays are excluded from
    interjection candidacy:
      1. Plays attributed to a different queue (cross-push overlap)
      2. Plays from a different known playlist (context switch, not a seek)
      3. Null-context plays > MAX_INTERJECTION_GAP_MIN from neighboring queue songs
         (catches offline/BART sessions sandwiched between queue plays)

    Interjection vs. abandonment:
      [A, B, X, C, D]   → X is an interjection (queue resumed after)
      [A, B, X, Y, Z…]  → abandoned at B (never returned to queue)
    """
    queue_songs = json.loads(songs_json)
    if not queue_songs:
        return

    queue_ids  = [s["song_id"] for s in queue_songs]
    queue_set  = set(queue_ids)
    queue_pos  = {sid: i for i, sid in enumerate(queue_ids)}

    # Energy scores for context calculation (song_tags may not exist on all deployments)
    energy_map = {}
    if queue_ids:
        try:
            ph = ",".join("?" * len(queue_ids))
            energy_map = dict(conn.execute(
                f"SELECT song_id, energy_score FROM song_tags WHERE song_id IN ({ph})",
                queue_ids,
            ).fetchall())
        except sqlite3.OperationalError:
            pass

    # URI of our managed push playlist — plays from other playlists are context switches.
    push_playlist_row = conn.execute("SELECT value FROM config WHERE key='push_playlist_id'").fetchone()
    push_playlist_uri = f"spotify:playlist:{push_playlist_row[0]}" if push_playlist_row else None

    # All plays during the 2-hour attribution window, chronological.
    # Normalize both sides to 'YYYY-MM-DD HH:MM:SS' so SQLite datetime()
    # comparisons work regardless of whether timestamps use T/Z/+00:00 suffixes.
    pushed_dt = _sqlite_ts(pushed_at)
    plays = conn.execute("""
        SELECT song_id, played_at, play_source, context_uri
        FROM plays
        WHERE REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') >= ?
          AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') <= datetime(?, '+2 hours')
        ORDER BY played_at ASC
    """, (pushed_dt, pushed_dt)).fetchall()

    if not plays:
        conn.execute(
            "UPDATE queue_pushes SET completion_rate=0.0, songs_played=0, analyzed_at=? WHERE push_id=?",
            (now_iso(), push_id),
        )
        conn.commit()
        return

    # Build annotated sequence
    # Tuple: (song_id, is_queue, queue_index | None, played_at, context_uri)
    sequence = []
    queue_played = set()
    for song_id, played_at, play_source, context_uri in plays:
        if song_id in queue_set:
            idx = queue_pos[song_id]
            queue_played.add(idx)
            sequence.append((song_id, True, idx, played_at, context_uri))
        elif play_source and play_source.endswith("_queued"):
            # Attributed to a different queue push — overlapping window artifact, skip
            continue
        else:
            sequence.append((song_id, False, None, played_at, context_uri))

    # Classify each non-queue song: interjection or post-abandonment
    interjections = []
    for i, (song_id, is_queue, _, played_at, context_uri) in enumerate(sequence):
        if is_queue:
            continue

        # Nearest queue song before and after this position
        prev_q = next((sequence[j] for j in range(i - 1, -1, -1) if sequence[j][1]), None)
        next_q = next((sequence[j] for j in range(i + 1, len(sequence)) if sequence[j][1]), None)

        before_idx = prev_q[2] if prev_q else None
        after_idx  = next_q[2] if next_q else None

        if before_idx is None or after_idx is None:
            continue

        # Non-queue plays (null context or a different playlist/source) must be close
        # in time to both neighboring queue songs. Null = offline/BART risk; other-playlist
        # = possible accidental context switch. The gap filter distinguishes deliberate
        # single-song interjections from longer detours.
        is_other_source = (not context_uri) or (
            context_uri != push_playlist_uri
            and context_uri.startswith("spotify:")
        )
        if is_other_source:
            curr_ts = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
            prev_ts = datetime.fromisoformat(prev_q[3].replace("Z", "+00:00"))
            next_ts = datetime.fromisoformat(next_q[3].replace("Z", "+00:00"))
            if ((curr_ts - prev_ts).total_seconds() / 60 > MAX_INTERJECTION_GAP_MIN
                    or (next_ts - curr_ts).total_seconds() / 60 > MAX_INTERJECTION_GAP_MIN):
                continue

        # Queue resumed after this song → true interjection
        interjections.append({
            "song_id":      song_id,
            "played_at":    played_at,
            "queue_before": before_idx,
            "queue_after":  after_idx,
        })

    # Abandonment: does the session end with non-queue songs after the last queue song?
    last_queue_item = next((s for s in reversed(sequence) if s[1]), None)
    if last_queue_item:
        after_last = [s for s in sequence if s[3] > last_queue_item[3] and not s[1]]
        abandoned_at = last_queue_item[2] if after_last else None
    else:
        abandoned_at = None  # no queue songs played at all

    completion_rate = round(len(queue_played) / len(queue_ids), 3)

    # Persist completion stats
    conn.execute("""
        UPDATE queue_pushes
        SET completion_rate = ?, songs_played = ?, abandoned_at = ?, analyzed_at = ?
        WHERE push_id = ?
    """, (completion_rate, len(queue_played), abandoned_at, now_iso(), push_id))

    # Persist interjections with context energy
    try:
        dt     = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        bucket = _hour_to_bucket(dt.astimezone().hour)
    except Exception:
        bucket = None

    # Exponential decay weight: the queue song immediately before the interjection
    # gets weight 1.0; each earlier song is discounted by INTERJECTION_DECAY.
    # Rationale: the whole preceding queue shaped the vibe that led the user to
    # insert this song, but recency still matters — you're more likely to queue
    # something that fits the last song than the first one.
    INTERJECTION_DECAY = 0.65

    for inj in interjections:
        before_idx = inj["queue_before"]

        # Weighted average over all queue songs up to and including before_idx.
        # song at before_idx → weight 1.0, song at before_idx-1 → 0.65, etc.
        weighted_sum = 0.0
        total_weight = 0.0
        for i in range(before_idx + 1):
            e = energy_map.get(queue_ids[i])
            if e is None:
                continue
            w = INTERJECTION_DECAY ** (before_idx - i)
            weighted_sum += w * e
            total_weight += w

        ctx_e = round(weighted_sum / total_weight, 4) if total_weight > 0 else None

        conn.execute("""
            INSERT OR IGNORE INTO session_interjections
                (push_id, song_id, interjected_at, queue_before, queue_after,
                 context_energy, context_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (push_id, inj["song_id"], inj["played_at"],
              inj["queue_before"], inj["queue_after"], ctx_e, bucket))

    conn.commit()


def attribute_plays_to_queues(conn):
    """
    Step 1 — Attribution: tags plays within 2 hours of each push with the
    algorithm that generated the queue ('smartshuffle_queued' / 'baseline_queued').

    Step 2 — Analysis: for pushes whose 2-hour window has closed (pushed 2–24h
    ago), runs analyze_queue_session to compute completion rate and record
    interjections. Skips pushes already analyzed.

    Both steps are idempotent.
    """
    try:
        _init_session_tables(conn)
        pushes = conn.execute("""
            SELECT qp.push_id, qp.queue_id, qp.algorithm, qp.pushed_at, q.songs
            FROM queue_pushes qp
            JOIN queues q ON qp.queue_id = q.queue_id
            ORDER BY qp.pushed_at DESC
        """).fetchall()
    except Exception:
        return 0   # queue_pushes table may not exist yet

    attributed = 0
    for push_id, _, algorithm, pushed_at, songs_json in pushes:
        song_ids = [s["song_id"] for s in json.loads(songs_json)]
        if not song_ids:
            continue
        ph = ",".join("?" * len(song_ids))
        pushed_dt = _sqlite_ts(pushed_at)
        rows = conn.execute(f"""
            UPDATE plays SET play_source = ?
            WHERE song_id IN ({ph})
              AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') >= ?
              AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') <= datetime(?, '+2 hours')
              AND (play_source IS NULL OR play_source NOT LIKE '%_queued')
        """, [f"{algorithm}_queued"] + song_ids + [pushed_dt, pushed_dt]).rowcount
        attributed += rows

    if attributed:
        conn.commit()

    # Analyze all unanalyzed sessions (no upper bound — handles historical backfill).
    completed = conn.execute("""
        SELECT qp.push_id, qp.algorithm, qp.pushed_at, q.songs
        FROM queue_pushes qp
        JOIN queues q ON qp.queue_id = q.queue_id
        WHERE REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' ') < datetime('now', '-2 hours')
          AND qp.analyzed_at IS NULL
    """).fetchall()

    for push_id, algorithm, pushed_at, songs_json in completed:
        analyze_queue_session(conn, push_id, algorithm, pushed_at, songs_json)

    # Infer skips for newly closed sessions only (skip already-inferred ones).
    # Rolling sessions are combined into one session queue before inference;
    # full-mode pushes are processed individually as before.
    closed = conn.execute("""
        SELECT qp.push_id, qp.algorithm, qp.pushed_at, q.songs,
               qp.mode, COALESCE(qp.rolling_session_id, qp.push_id) AS session_id
        FROM queue_pushes qp
        JOIN queues q ON qp.queue_id = q.queue_id
        WHERE REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' ') < datetime('now', '-2 hours')
          AND qp.skips_inferred_at IS NULL
    """).fetchall()

    # Process non-rolling pushes individually.
    for push_id, algorithm, pushed_at, songs_json, mode, session_id in closed:
        if mode != 'rolling':
            _infer_queue_skips(conn, push_id, algorithm, pushed_at, songs_json)

    # Process rolling sessions as combined queues.
    # Only run once all pushes in the session are closed (> 2 hours old).
    pending_sessions: dict = {}
    for push_id, algorithm, pushed_at, songs_json, mode, session_id in closed:
        if mode == 'rolling':
            pending_sessions.setdefault(session_id, []).append(push_id)

    for session_id, unprocessed_push_ids in pending_sessions.items():
        total_in_session = conn.execute(
            "SELECT COUNT(*) FROM queue_pushes WHERE COALESCE(rolling_session_id, push_id) = ?",
            (session_id,),
        ).fetchone()[0]
        if len(unprocessed_push_ids) == total_in_session:
            # All pushes closed and unprocessed — safe to run combined inference.
            _infer_rolling_session_skips(conn, session_id)

    return attributed


def _lis_detect_skips(
    combined_ids: list[str],
    played_rows: list[tuple[str, str]],
    min_plays: int = 4,
    max_consecutive_skips: int = 10,
) -> list[tuple[int, str]]:
    """
    Core LIS skip inference over an ordered song list and a set of (song_id, ts) plays.

    Returns list of (position, song_id) confirmed skips, or [] when the
    sequence is too short to be reliable.

    Works for both per-push (10 songs) and combined rolling session (40 songs).
    """
    from bisect import bisect_left

    if not combined_ids:
        return []

    pos_map = {sid: i for i, sid in enumerate(combined_ids)}

    # Deduplicate same-timestamp plays: keep lowest queue position.
    seen_ts: dict = {}
    for sid, ts in played_rows:
        if sid not in pos_map:
            continue
        p = pos_map[sid]
        if ts not in seen_ts or p < seen_ts[ts][0]:
            seen_ts[ts] = (p, sid)

    timed_ordered = [seen_ts[ts] for ts in sorted(seen_ts)]
    if not timed_ordered:
        return []

    # LIS on queue positions (O(n log n)).
    positions = [p for p, _ in timed_ordered]
    n         = len(positions)
    dp        = []
    dp_idx    = []
    parent    = [-1] * n

    for i, p in enumerate(positions):
        idx = bisect_left(dp, p)
        if idx == len(dp):
            dp.append(p)
            dp_idx.append(i)
        else:
            dp[idx] = p
            dp_idx[idx] = i
        parent[i] = dp_idx[idx - 1] if idx > 0 else -1

    lis_indices: set = set()
    cur = dp_idx[-1]
    while cur != -1:
        lis_indices.add(cur)
        cur = parent[cur]

    # Apply consecutive-skip cap: a gap > max_consecutive_skips almost certainly
    # means the next play came from a different Spotify context.
    lis_seq   = sorted(timed_ordered[i] for i in lis_indices)
    truncated = [lis_seq[0]] if lis_seq else []
    for pos, sid in lis_seq[1:]:
        if pos - truncated[-1][0] - 1 > max_consecutive_skips:
            break
        truncated.append((pos, sid))

    if len(truncated) < min_plays:
        return []

    played_ids = {sid for _, sid in truncated}
    last_pos   = truncated[-1][0]

    return [
        (i, sid) for i, sid in enumerate(combined_ids[:last_pos])
        if sid not in played_ids
    ]


def _infer_queue_skips(conn, push_id: int, algorithm: str, pushed_at: str, songs_json: str):
    """
    Per-push skip inference (used for full-mode queues and single rolling pushes).
    Looks up plays within 2 hours of the push and runs LIS over the push's songs.
    """
    queued_ids = [s["song_id"] for s in json.loads(songs_json)]
    if not queued_ids:
        return

    play_source = f"{algorithm}_queued"
    pushed_dt   = _sqlite_ts(pushed_at)
    ph          = ",".join("?" * len(queued_ids))

    played_rows = conn.execute(f"""
        SELECT song_id, REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') AS ts
        FROM plays
        WHERE play_source = ?
          AND song_id IN ({ph})
          AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') >= ?
          AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') <= datetime(?, '+2 hours')
        ORDER BY played_at
    """, [play_source] + queued_ids + [pushed_dt, pushed_dt]).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    skipped = _lis_detect_skips(queued_ids, played_rows)

    if skipped:
        conn.executemany("""
            INSERT OR IGNORE INTO queue_skips (push_id, song_id, algorithm, inferred_at, queue_position)
            VALUES (?, ?, ?, ?, ?)
        """, [(push_id, sid, algorithm, now, pos) for pos, sid in skipped])

    conn.execute(
        "UPDATE queue_pushes SET skips_inferred_at = ? WHERE push_id = ?",
        (now, push_id),
    )
    conn.commit()


def _infer_rolling_session_skips(conn, rolling_session_id: int):
    """
    Combined skip inference across all pushes in a rolling session.

    Treats the full sequence of batches as one large queue: if the user jumped
    from batch-1 song 8 to batch-2 song 3, the songs skipped over the batch
    boundary are correctly detected.  Per-push inference misses these entirely
    because each batch only knows about its own 10 songs.

    All resulting queue_skips are stored under push_id = rolling_session_id
    (the first push).  Old per-push queue_skips for the session are replaced.
    """
    pushes = conn.execute("""
        SELECT push_id, queue_id, pushed_at, algorithm
        FROM queue_pushes
        WHERE COALESCE(rolling_session_id, push_id) = ?
        ORDER BY pushed_at
    """, (rolling_session_id,)).fetchall()

    if not pushes:
        return

    algorithm  = pushes[0][3]
    first_push = _sqlite_ts(pushes[0][2])
    last_push  = _sqlite_ts(pushes[-1][2])

    # Combined song list in push order; first occurrence wins on duplicates.
    seen: set = set()
    combined_ids: list = []
    for _, queue_id, _, _ in pushes:
        row = conn.execute("SELECT songs FROM queues WHERE queue_id = ?", (queue_id,)).fetchone()
        if not row:
            continue
        for song in json.loads(row[0]):
            sid = song["song_id"]
            if sid not in seen:
                combined_ids.append(sid)
                seen.add(sid)

    if not combined_ids:
        return

    ph = ",".join("?" * len(combined_ids))
    play_source = f"{algorithm}_queued"

    played_rows = conn.execute(f"""
        SELECT song_id, REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') AS ts
        FROM plays
        WHERE play_source = ?
          AND song_id IN ({ph})
          AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') >= ?
          AND REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ') <= datetime(?, '+2 hours')
        ORDER BY played_at
    """, [play_source] + combined_ids + [first_push, last_push]).fetchall()

    skipped = _lis_detect_skips(combined_ids, played_rows)
    now     = datetime.now(timezone.utc).isoformat()

    # Replace per-push results with combined session result.
    push_ids = [p[0] for p in pushes]
    ph_ids   = ",".join("?" * len(push_ids))
    conn.execute(f"DELETE FROM queue_skips WHERE push_id IN ({ph_ids})", push_ids)

    if skipped:
        conn.executemany("""
            INSERT OR IGNORE INTO queue_skips (push_id, song_id, algorithm, inferred_at, queue_position)
            VALUES (?, ?, ?, ?, ?)
        """, [(rolling_session_id, sid, algorithm, now, pos) for pos, sid in skipped])

    conn.executemany(
        "UPDATE queue_pushes SET skips_inferred_at = ? WHERE push_id = ?",
        [(now, pid) for pid in push_ids],
    )
    conn.commit()


# ── Data retention ───────────────────────────────────────────────────────────

def purge_old_plays(conn, days: int = 90) -> int:
    """Delete raw play records older than `days` days.

    Binge episodes, sessions, and song scores are intentionally preserved —
    they are aggregated/derived and are needed for longitudinal analysis.
    Only the raw Spotify play rows are subject to the retention window.
    """
    conn.execute("""
        DELETE FROM plays
        WHERE REPLACE(SUBSTR(played_at, 1, 19), 'T', ' ')
            < datetime('now', '-' || ? || ' days')
    """, (days,))
    deleted = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return deleted


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--plays-only", action="store_true",
                         help="Skip playlist sync — only collect recent plays and attribute them")
    _args, _ = _parser.parse_known_args()

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _db_path = os.getenv("SS_DB_PATH") or os.path.join(_root, "data", "smartshuffle.db")
    conn = sqlite3.connect(_db_path)
    init_db(conn)

    print("=== SmartShuffle Data Collection ===")
    print(f"Started: {now_iso()}\n")

    if not _args.plays_only:
        # Cold start: no play history yet — seed songs from top tracks so phase2.py
        # has enough tag data to produce a non-trivial energy prior.
        play_count = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
        if play_count == 0:
            print("No play history yet — bootstrapping from your top tracks...")
            bootstrap_cold_start(conn)
            print()

        print("Syncing playlists...")
        collect_playlists(conn)
        print()

    print("Collecting recently played tracks...")
    inserted = collect_recently_played(conn)
    print(f"  {inserted} new plays added to database.")

    attributed = attribute_plays_to_queues(conn)
    if attributed:
        print(f"  {attributed} plays attributed to pushed queues.")

    # Summary
    total_plays = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
    total_songs = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    songs_with_metadata = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE audio_features IS NOT NULL"
    ).fetchone()[0]

    source_breakdown = conn.execute(
        "SELECT play_source, COUNT(*) FROM plays GROUP BY play_source"
    ).fetchall()
    skip_breakdown = conn.execute(
        "SELECT inferred_skip, COUNT(*) FROM plays GROUP BY inferred_skip"
    ).fetchall()

    print(f"\n── Database summary ──")
    print(f"  Total plays:          {total_plays}")
    print(f"  Unique songs:         {total_songs}")
    print(f"  Songs with metadata:  {songs_with_metadata}")
    print(f"  Play sources:")
    for source, count in source_breakdown:
        print(f"    {source or 'unknown'}: {count}")
    print(f"  Skip inference:")
    for skip, count in skip_breakdown:
        print(f"    {skip or 'unknown'}: {count}")

    purged = purge_old_plays(conn)
    if purged:
        print(f"\n  Purged {purged} play records older than 90 days.")

    conn.close()
    print(f"\nDone: {now_iso()}")


if __name__ == "__main__":
    main()