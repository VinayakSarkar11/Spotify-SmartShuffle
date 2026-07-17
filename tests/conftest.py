"""
Shared fixtures for SmartShuffle test suite.

All DB tests use an isolated in-memory SQLite connection so they never
touch smartshuffle.db.  Spotify / Last.fm calls are always mocked.
"""
import os
import sys

import pytest

# Ensure project root and src/ are importable from test files.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# In-memory DB with the full schema (collect_data + phase2 + phase34 tables)
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_db():
    import sqlite3
    from collect  import init_db, _init_session_tables
    from score    import init_phase2_tables
    from recommend import init_tables as init_phase34_tables
    from push     import _init_push_table

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    init_phase2_tables(conn)
    init_phase34_tables(conn)
    _init_push_table(conn)      # creates queue_pushes
    _init_session_tables(conn)  # creates queue_skips, adds columns to queue_pushes
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Minimal song rows (inserted once, reused across tests)
# ---------------------------------------------------------------------------

SONG_ROWS = [
    # (song_id, song_name, artist_name, duration_ms)
    ("AAA", "Track A", "Artist 1", 210_000),
    ("BBB", "Track B", "Artist 2", 195_000),
    ("CCC", "Track C", "Artist 3", 180_000),
    ("DDD", "Track D", "Artist 1", 220_000),
    ("EEE", "Track E", "Artist 4", 200_000),
]


@pytest.fixture
def seeded_db(mem_db):
    """mem_db with songs + playlist rows pre-inserted."""
    conn = mem_db
    for sid, name, artist, dur in SONG_ROWS:
        conn.execute(
            "INSERT OR IGNORE INTO songs (song_id, song_name, artist_name, duration_ms)"
            " VALUES (?,?,?,?)",
            (sid, name, artist, dur),
        )
    conn.execute(
        "INSERT OR IGNORE INTO playlists (playlist_id, playlist_name)"
        " VALUES ('PL1','Test Playlist')"
    )
    for sid, *_ in SONG_ROWS:
        conn.execute(
            "INSERT OR IGNORE INTO playlist_tracks (playlist_id, song_id)"
            " VALUES ('PL1', ?)", (sid,)
        )
    conn.commit()
    return conn
