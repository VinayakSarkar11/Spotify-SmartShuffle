"""
Tests for play.py — DB logic tested directly; Spotify calls mocked.
"""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Mock spotipy before play.py is imported ──────────────────────────────────
# play.py instantiates spotipy.Spotify at module level; patch it here so the
# import doesn't require real credentials.

_mock_sp = MagicMock()
_mock_sp_module     = MagicMock()
_mock_oauth2_module = MagicMock()
_mock_sp_module.Spotify.return_value     = _mock_sp
_mock_oauth2_module.SpotifyOAuth         = MagicMock()
_mock_sp_exceptions = MagicMock()
_mock_sp_exceptions.SpotifyException     = Exception

with patch.dict(sys.modules, {
    "spotipy":        _mock_sp_module,
    "spotipy.oauth2": _mock_oauth2_module,
    "spotipy.exceptions": _mock_sp_exceptions,
}):
    import importlib
    import push as play_module
    importlib.reload(play_module)   # reload so the patched modules take effect


# ── load_queue ────────────────────────────────────────────────────────────────

def _insert_queue(conn, algorithm="smartshuffle", songs=None):
    songs = songs or [{"song_id": "AAA", "song_name": "Track A", "artist_name": "Artist 1"}]
    conn.execute("""
        INSERT INTO queues (generated_at, context, algorithm, playlist_id, songs, ab_label)
        VALUES ('2024-01-01T20:00:00+00:00', 'morning', ?, 'PL1', ?, 'A')
    """, (algorithm, json.dumps(songs)))
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


class TestLoadQueue:
    def test_returns_most_recent_smartshuffle(self, seeded_db):
        _insert_queue(seeded_db)
        with patch.object(play_module, "DB_PATH", ":memory:"), \
             patch("sqlite3.connect", return_value=seeded_db):
            row = play_module.load_queue(None)
        assert row is not None
        assert row[2] == "Test Playlist"

    def test_load_by_specific_id(self, seeded_db):
        qid = _insert_queue(seeded_db)
        with patch("sqlite3.connect", return_value=seeded_db):
            row = play_module.load_queue(qid)
        assert row is not None
        assert row[0] == qid

    def test_no_queue_returns_none(self, mem_db):
        with patch("sqlite3.connect", return_value=mem_db):
            row = play_module.load_queue(None)
        assert row is None


# ── get_device ────────────────────────────────────────────────────────────────

class TestGetDevice:
    def test_prefers_active_device(self):
        inactive = {"id": "d1", "name": "Desktop",  "is_active": False}
        active   = {"id": "d2", "name": "Phone",    "is_active": True}
        _mock_sp.devices.return_value = {"devices": [inactive, active]}
        device = play_module.get_device()
        assert device["id"] == "d2"

    def test_falls_back_to_first_if_none_active(self):
        d1 = {"id": "d1", "name": "Desktop", "is_active": False}
        d2 = {"id": "d2", "name": "TV",      "is_active": False}
        _mock_sp.devices.return_value = {"devices": [d1, d2]}
        device = play_module.get_device()
        assert device["id"] == "d1"

    def test_returns_none_when_no_devices(self):
        _mock_sp.devices.return_value = {"devices": []}
        assert play_module.get_device() is None
