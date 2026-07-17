"""
Tests for collect_data.py — pure logic only (no Spotify API calls).
"""
import json
import sqlite3

import pytest

from collect import (
    _hour_to_bucket,
    _init_session_tables,
    analyze_queue_session,
    classify_play_source,
    flag_jam_sessions,
    infer_skip,
    init_db,
)


# ── infer_skip ──────────────────────────────────────────────────────────────

class TestInferSkip:
    def test_full_play(self):
        assert infer_skip(190_000, 200_000) == "full"

    def test_skip_under_30_pct(self):
        assert infer_skip(50_000, 200_000) == "skip"

    def test_partial_between_30_and_80_pct(self):
        assert infer_skip(100_000, 200_000) == "partial"

    def test_exact_boundary_30_pct_is_partial(self):
        # ratio == 0.30 → not < 0.30, so partial
        assert infer_skip(60_000, 200_000) == "partial"

    def test_exact_boundary_80_pct_is_full(self):
        # ratio == 0.80 → not < 0.80, so full
        assert infer_skip(160_000, 200_000) == "full"

    def test_none_play_duration(self):
        assert infer_skip(None, 200_000) == "unknown"

    def test_none_song_duration(self):
        assert infer_skip(100_000, None) == "unknown"

    def test_zero_song_duration(self):
        assert infer_skip(0, 0) == "unknown"


# ── classify_play_source ─────────────────────────────────────────────────────

class TestClassifyPlaySource:
    def test_playlist_context(self):
        assert classify_play_source({"type": "playlist"}) == "playlist"

    def test_album_context(self):
        assert classify_play_source({"type": "album"}) == "album_browse"

    def test_artist_context(self):
        assert classify_play_source({"type": "artist"}) == "artist_browse"

    def test_none_context_returns_manual(self):
        assert classify_play_source(None) == "manual_search"

    def test_empty_dict_returns_manual(self):
        assert classify_play_source({}) == "manual_search"

    def test_unknown_type_returns_manual(self):
        assert classify_play_source({"type": "show"}) == "manual_search"


# ── flag_jam_sessions ────────────────────────────────────────────────────────

def _make_plays(n, duration_ms, artists, base_ts="2024-01-01T20:00:00+00:00"):
    """Build a list of fake play dicts spaced 2 min apart."""
    from datetime import datetime, timedelta, timezone
    t = datetime.fromisoformat(base_ts)
    plays = []
    for i in range(n):
        plays.append({
            "played_at":      t.isoformat(),
            "play_duration_ms": duration_ms,
            "artist_name":    artists[i % len(artists)],
            "play_source":    "playlist",
        })
        t += timedelta(minutes=2)
    return plays


class TestFlagJamSessions:
    def test_marks_jam_when_all_criteria_met(self):
        # 4 plays, <60s median, ≥3 distinct artists
        plays = _make_plays(4, 30_000, ["A1", "A2", "A3", "A4"])
        result = flag_jam_sessions(plays)
        assert all(p["play_source"] == "jam_excluded" for p in result)

    def test_no_flag_if_fewer_than_4_plays(self):
        plays = _make_plays(3, 30_000, ["A1", "A2", "A3"])
        result = flag_jam_sessions(plays)
        assert all(p["play_source"] == "playlist" for p in result)

    def test_no_flag_if_median_duration_long(self):
        # Long plays (>60s median) → not a jam
        plays = _make_plays(4, 180_000, ["A1", "A2", "A3", "A4"])
        result = flag_jam_sessions(plays)
        assert all(p["play_source"] == "playlist" for p in result)

    def test_no_flag_if_fewer_than_3_artists(self):
        # Only 2 distinct artists
        plays = _make_plays(4, 30_000, ["A1", "A2"])
        result = flag_jam_sessions(plays)
        assert all(p["play_source"] == "playlist" for p in result)

    def test_gap_creates_separate_sessions(self):
        # First 4 plays close together (jam), then a 2-hour gap, then 4 more (clean)
        from datetime import datetime, timedelta, timezone
        jam_plays   = _make_plays(4, 30_000, ["A1", "A2", "A3", "A4"],
                                  "2024-01-01T20:00:00+00:00")
        # offset second batch by 3 hours
        clean_plays = _make_plays(4, 200_000, ["A1", "A2", "A3", "A4"],
                                  "2024-01-01T23:30:00+00:00")
        result = flag_jam_sessions(jam_plays + clean_plays)
        jam_part   = result[:4]
        clean_part = result[4:]
        assert all(p["play_source"] == "jam_excluded" for p in jam_part)
        assert all(p["play_source"] == "playlist"     for p in clean_part)

    def test_empty_list(self):
        assert flag_jam_sessions([]) == []


# ── _hour_to_bucket ───────────────────────────────────────────────────────────

class TestHourToBucket:
    @pytest.mark.parametrize("hour,expected", [
        (0,  "late_night"), (5,  "late_night"), (22, "late_night"), (23, "late_night"),
        (6,  "morning"),    (11, "morning"),
        (12, "afternoon"),  (17, "afternoon"),
        (18, "evening"),    (21, "evening"),
    ])
    def test_boundaries(self, hour, expected):
        assert _hour_to_bucket(hour) == expected


# ── init_db schema ───────────────────────────────────────────────────────────

def test_init_db_creates_tables():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    for expected in ("plays", "songs", "playlists", "playlist_tracks"):
        assert expected in tables, f"Missing table: {expected}"
    conn.close()


# ── _init_session_tables ──────────────────────────────────────────────────────

class TestInitSessionTables:
    def test_creates_session_interjections(self, mem_db):
        # queue_pushes doesn't exist yet in mem_db; create a minimal version first
        mem_db.execute("""
            CREATE TABLE IF NOT EXISTS queue_pushes (
                push_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id  INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                pushed_at TEXT NOT NULL
            )
        """)
        _init_session_tables(mem_db)
        tables = {r[0] for r in mem_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "session_interjections" in tables

    def test_adds_columns_to_queue_pushes(self, mem_db):
        mem_db.execute("""
            CREATE TABLE IF NOT EXISTS queue_pushes (
                push_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id  INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                pushed_at TEXT NOT NULL
            )
        """)
        _init_session_tables(mem_db)
        cols = {r[1] for r in mem_db.execute("PRAGMA table_info(queue_pushes)")}
        for expected in ("completion_rate", "songs_played", "abandoned_at", "analyzed_at"):
            assert expected in cols

    def test_idempotent(self, mem_db):
        mem_db.execute("""
            CREATE TABLE IF NOT EXISTS queue_pushes (
                push_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id  INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                pushed_at TEXT NOT NULL
            )
        """)
        _init_session_tables(mem_db)
        _init_session_tables(mem_db)   # second call must not raise
        tables = {r[0] for r in mem_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "session_interjections" in tables


# ── analyze_queue_session ─────────────────────────────────────────────────────

def _setup_push_tables(conn):
    """Create queue_pushes + session_interjections in an in-memory DB."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue_pushes (
            push_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id       INTEGER NOT NULL,
            algorithm      TEXT NOT NULL,
            pushed_at      TEXT NOT NULL,
            completion_rate REAL,
            songs_played   INTEGER,
            abandoned_at   INTEGER,
            analyzed_at    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_interjections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            push_id        INTEGER NOT NULL,
            song_id        TEXT NOT NULL,
            interjected_at TEXT NOT NULL,
            queue_before   INTEGER,
            queue_after    INTEGER,
            context_energy REAL,
            context_label  TEXT,
            UNIQUE (push_id, song_id, interjected_at)
        )
    """)
    conn.commit()


def _insert_push(conn, queue_songs, pushed_at="2026-01-01T08:00:00+00:00"):
    conn.execute(
        "INSERT INTO queue_pushes (queue_id, algorithm, pushed_at) VALUES (1, 'smartshuffle', ?)",
        (pushed_at,),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_play(conn, song_id, played_at, duration_ms=210000, play_duration_ms=200000):
    conn.execute("""
        INSERT OR IGNORE INTO plays
            (song_id, played_at, duration_ms, play_duration_ms, inferred_skip)
        VALUES (?, ?, ?, ?, 'full')
    """, (song_id, played_at, duration_ms, play_duration_ms))
    conn.commit()


def _songs_json(song_ids):
    return json.dumps([{"song_id": sid} for sid in song_ids])


class TestAnalyzeQueueSession:
    def test_full_completion(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC"]
        push_id = _insert_push(mem_db, songs)
        for i, sid in enumerate(songs):
            _insert_play(mem_db, sid, f"2026-01-01T08:0{i}:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        row = mem_db.execute(
            "SELECT completion_rate, songs_played, abandoned_at FROM queue_pushes WHERE push_id=?",
            (push_id,)
        ).fetchone()
        assert row[0] == pytest.approx(1.0)
        assert row[1] == 3
        assert row[2] is None

    def test_partial_completion_abandonment(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        push_id = _insert_push(mem_db, songs)
        # Play only first 2 queue songs, then non-queue songs → abandoned at index 1
        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "BBB", "2026-01-01T08:02:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:03:00+00:00")  # non-queue

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        row = mem_db.execute(
            "SELECT completion_rate, songs_played, abandoned_at FROM queue_pushes WHERE push_id=?",
            (push_id,)
        ).fetchone()
        assert row[0] == pytest.approx(2 / 5)
        assert row[1] == 2
        assert row[2] == 1  # last queue index played before abandonment

    def test_interjection_not_counted_as_abandonment(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC", "DDD"]
        push_id = _insert_push(mem_db, songs)
        # A, B, [X interjection], C, D — queue resumed after X
        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "BBB", "2026-01-01T08:02:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:03:00+00:00")  # interjection
        _insert_play(mem_db, "CCC", "2026-01-01T08:04:00+00:00")
        _insert_play(mem_db, "DDD", "2026-01-01T08:05:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        row = mem_db.execute(
            "SELECT completion_rate, songs_played, abandoned_at FROM queue_pushes WHERE push_id=?",
            (push_id,)
        ).fetchone()
        assert row[0] == pytest.approx(1.0)
        assert row[2] is None  # no abandonment

    def test_interjection_recorded(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC"]
        push_id = _insert_push(mem_db, songs)
        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:02:00+00:00")  # interjection
        _insert_play(mem_db, "BBB", "2026-01-01T08:03:00+00:00")
        _insert_play(mem_db, "CCC", "2026-01-01T08:04:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        inj = mem_db.execute(
            "SELECT song_id, queue_before, queue_after FROM session_interjections WHERE push_id=?",
            (push_id,)
        ).fetchall()
        assert len(inj) == 1
        assert inj[0][0] == "ZZZ"
        assert inj[0][1] == 0   # after AAA (index 0)
        assert inj[0][2] == 1   # before BBB (index 1)

    def test_post_abandonment_song_not_recorded_as_interjection(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC"]
        push_id = _insert_push(mem_db, songs)
        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:02:00+00:00")  # post-abandonment
        _insert_play(mem_db, "WWW", "2026-01-01T08:03:00+00:00")  # post-abandonment

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        inj = mem_db.execute(
            "SELECT COUNT(*) FROM session_interjections WHERE push_id=?", (push_id,)
        ).fetchone()[0]
        assert inj == 0  # ZZZ and WWW are post-abandonment, not interjections

    def test_empty_queue(self, mem_db):
        _setup_push_tables(mem_db)
        push_id = _insert_push(mem_db, [])
        # Should not raise
        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", "[]")

    def test_no_plays_in_window(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB"]
        push_id = _insert_push(mem_db, songs)
        # Insert plays outside the window
        _insert_play(mem_db, "AAA", "2026-01-01T11:00:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        row = mem_db.execute(
            "SELECT completion_rate, songs_played FROM queue_pushes WHERE push_id=?",
            (push_id,)
        ).fetchone()
        assert row[0] == 0.0
        assert row[1] == 0

    def test_context_energy_weighted_toward_preceding_songs(self, mem_db):
        """
        Context energy should be a decay-weighted average of queue songs BEFORE
        the interjection (not a simple average of immediate neighbors).
        With queue [A=0.2, B=0.6, C=0.8] and interjection after C (index 2):
          weights: C=1.0, B=0.65, A=0.65^2=0.4225
          weighted energy = (1.0*0.8 + 0.65*0.6 + 0.4225*0.2) / (1.0+0.65+0.4225)
                         = (0.8 + 0.39 + 0.0845) / 2.0725
                         ≈ 0.611
        Crucially, this is NOT simply 0.8 (just C) nor 0.75 (avg of B and C).
        """
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC", "DDD"]
        push_id = _insert_push(mem_db, songs)

        # Seed energy scores for queue songs
        for sid, energy in [("AAA", 0.2), ("BBB", 0.6), ("CCC", 0.8)]:
            mem_db.execute("""
                INSERT OR REPLACE INTO song_tags (song_id, tags, energy_score, top_tags, fetched_at, fetch_source)
                VALUES (?, '[]', ?, '[]', '2026-01-01', 'test')
            """, (sid, energy))
        mem_db.commit()

        # Play A, B, C, then interjection ZZZ, then D (queue resumes)
        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "BBB", "2026-01-01T08:02:00+00:00")
        _insert_play(mem_db, "CCC", "2026-01-01T08:03:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:04:00+00:00")  # interjection after C
        _insert_play(mem_db, "DDD", "2026-01-01T08:05:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        ctx_e = mem_db.execute(
            "SELECT context_energy FROM session_interjections WHERE push_id=? AND song_id='ZZZ'",
            (push_id,)
        ).fetchone()[0]

        # Weighted: C=1.0*0.8, B=0.65*0.6, A=0.65^2*0.2
        decay   = 0.65
        weights = [decay**2, decay**1, decay**0]   # A, B, C
        energies = [0.2, 0.6, 0.8]
        expected = sum(w * e for w, e in zip(weights, energies)) / sum(weights)

        assert ctx_e == pytest.approx(expected, abs=0.001)
        # Should be meaningfully influenced by B, not just equal to C (0.8)
        assert ctx_e < 0.80
        assert ctx_e > 0.60  # B's energy still pulls it down from C's 0.8

    def test_context_energy_single_preceding_song(self, mem_db):
        """When interjection happens after the very first queue song, context_energy = that song's energy."""
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC"]
        push_id = _insert_push(mem_db, songs)

        mem_db.execute("""
            INSERT OR REPLACE INTO song_tags (song_id, tags, energy_score, top_tags, fetched_at, fetch_source)
            VALUES ('AAA', '[]', 0.5, '[]', '2026-01-01', 'test')
        """)
        mem_db.commit()

        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:02:00+00:00")  # interjection after AAA
        _insert_play(mem_db, "BBB", "2026-01-01T08:03:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        ctx_e = mem_db.execute(
            "SELECT context_energy FROM session_interjections WHERE push_id=? AND song_id='ZZZ'",
            (push_id,)
        ).fetchone()[0]
        assert ctx_e == pytest.approx(0.5, abs=0.001)

    def test_idempotent_no_duplicate_interjections(self, mem_db):
        _setup_push_tables(mem_db)
        songs = ["AAA", "BBB", "CCC"]
        push_id = _insert_push(mem_db, songs)
        _insert_play(mem_db, "AAA", "2026-01-01T08:01:00+00:00")
        _insert_play(mem_db, "ZZZ", "2026-01-01T08:02:00+00:00")
        _insert_play(mem_db, "BBB", "2026-01-01T08:03:00+00:00")
        _insert_play(mem_db, "CCC", "2026-01-01T08:04:00+00:00")

        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))
        analyze_queue_session(mem_db, push_id, "2026-01-01T08:00:00+00:00", _songs_json(songs))

        count = mem_db.execute(
            "SELECT COUNT(*) FROM session_interjections WHERE push_id=?", (push_id,)
        ).fetchone()[0]
        assert count == 1  # not doubled
