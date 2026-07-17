"""
Tests for phase2.py — pure logic (no Last.fm calls).
"""
import sqlite3
import pandas as pd
import pytest

import json
from datetime import date, timedelta, datetime, timezone

def _utc_today() -> date:
    return datetime.now(timezone.utc).date()

from score import (
    clean_tags,
    compute_artist_fatigue_scores,
    compute_energy_score,
    get_energy_label,
    get_time_bucket,
    segment_sessions,
    record_binge_episodes,
    _build_daily_plays,
    JUNK_TAGS,
    ENERGY_WEIGHTS,
    GENERIC_TAGS,
    GENERIC_TAG_VOTE_MULTIPLIER,
    BINGE_END_GAP_DAYS,
)


# ── get_time_bucket ──────────────────────────────────────────────────────────

class TestGetTimeBucket:
    @pytest.mark.parametrize("hour,expected", [
        (0,  "late_night"),
        (3,  "late_night"),
        (5,  "late_night"),
        (6,  "morning"),
        (9,  "morning"),
        (10, "morning"),
        (11, "afternoon"),
        (12, "afternoon"),
        (15, "afternoon"),
        (17, "afternoon"),
        (18, "evening"),
        (21, "evening"),
        (22, "late_night"),
        (23, "late_night"),
    ])
    def test_bucket_boundaries(self, hour, expected):
        assert get_time_bucket(hour) == expected


# ── get_energy_label ─────────────────────────────────────────────────────────

class TestGetEnergyLabel:
    def test_high_energy(self):
        assert get_energy_label(0.5) == "high_energy"

    def test_low_energy(self):
        assert get_energy_label(-0.5) == "low_energy"

    def test_mid_energy_positive_boundary(self):
        assert get_energy_label(0.15) == "mid_energy"

    def test_mid_energy_negative_boundary(self):
        assert get_energy_label(-0.15) == "mid_energy"

    def test_just_above_high_threshold(self):
        assert get_energy_label(0.16) == "high_energy"

    def test_just_below_low_threshold(self):
        assert get_energy_label(-0.16) == "low_energy"


# ── clean_tags ───────────────────────────────────────────────────────────────

class TestCleanTags:
    def test_removes_junk_tags(self):
        junk = next(iter(JUNK_TAGS))   # grab any known junk tag
        tags = [{"name": junk, "count": 5}]
        assert clean_tags(tags) == []

    def test_deduplicates(self):
        tags = [
            {"name": "rap", "count": 10},
            {"name": "rap", "count": 5},
        ]
        result = clean_tags(tags)
        assert len(result) == 1

    def test_lowercases(self):
        tags = [{"name": "HIP-HOP", "count": 10}]
        result = clean_tags(tags)
        assert result[0]["name"] == "hip-hop"

    def test_strips_whitespace(self):
        tags = [{"name": "  trap  ", "count": 3}]
        result = clean_tags(tags)
        assert result[0]["name"] == "trap"

    def test_empty_list(self):
        assert clean_tags([]) == []

    def test_preserves_order(self):
        tags = [
            {"name": "trap", "count": 10},
            {"name": "rap",  "count": 5},
        ]
        result = clean_tags(tags)
        assert [t["name"] for t in result] == ["trap", "rap"]


# ── compute_energy_score ─────────────────────────────────────────────────────

class TestComputeEnergyScore:
    def test_no_tags_returns_zero(self):
        assert compute_energy_score([]) == 0.0

    def test_unknown_tags_return_zero(self):
        tags = [{"name": "unknown_genre_xyz", "weight": 100}]
        assert compute_energy_score(tags) == 0.0

    def test_positive_energy_tag(self):
        # Find a tag that has a positive weight in ENERGY_WEIGHTS
        pos_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v > 0)
        tags = [{"name": pos_tag, "weight": 10}]
        assert compute_energy_score(tags) > 0.0

    def test_negative_energy_tag(self):
        neg_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v < 0)
        tags = [{"name": neg_tag, "weight": 10}]
        assert compute_energy_score(tags) < 0.0

    def test_score_bounded(self):
        # Even with extreme weights, score should stay within [-1, 1]
        pos_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v > 0)
        tags = [{"name": pos_tag, "weight": 1_000_000}]
        score = compute_energy_score(tags)
        assert -1.0 <= score <= 1.0

    def test_mixed_tags_intermediate(self):
        pos_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v > 0)
        neg_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v < 0)
        # Equal weights → score closer to 0 than either extreme
        tags = [
            {"name": pos_tag, "weight": 5},
            {"name": neg_tag, "weight": 5},
        ]
        score = compute_energy_score(tags)
        pos_score = compute_energy_score([{"name": pos_tag, "weight": 5}])
        neg_score = compute_energy_score([{"name": neg_tag, "weight": 5}])
        assert neg_score < score < pos_score

    def test_higher_vote_weight_dominates(self):
        pos_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v > 0)
        neg_tag = next(k for k, v in ENERGY_WEIGHTS.items() if v < 0)
        heavy_pos = [{"name": pos_tag, "weight": 100}, {"name": neg_tag, "weight": 1}]
        heavy_neg = [{"name": pos_tag, "weight": 1},   {"name": neg_tag, "weight": 100}]
        assert compute_energy_score(heavy_pos) > compute_energy_score(heavy_neg)

    def test_specific_tag_beats_generic_flood(self):
        # 1000 votes of "rap" (generic, dampened) vs 50 votes of "drill" (specific).
        # Without dampening, rap (0.30 × 1000) would dominate; with dampening,
        # drill (0.7 × 50) should push the score well above 0.30.
        tags = [
            {"name": "rap",   "weight": 1000},
            {"name": "drill", "weight": 50},
        ]
        score = compute_energy_score(tags)
        # Pure rap would score ~0.30; drill pushing it to >0.40 means specific tag won
        assert score > 0.40, f"drill should dominate generic rap flood, got {score}"

    def test_generic_only_still_scores(self):
        # When only generic tags exist (no specific subgenre), they still
        # contribute — just at reduced vote weight.
        tags = [{"name": "hip-hop", "weight": 500}, {"name": "rap", "weight": 300}]
        score = compute_energy_score(tags)
        assert 0.25 < score < 0.45

    def test_chill_beats_generic_flood(self):
        # "chill" (-0.7) at 50 votes should pull below "rap" (0.30) at 1000 votes
        tags = [
            {"name": "rap",   "weight": 1000},
            {"name": "chill", "weight": 50},
        ]
        score = compute_energy_score(tags)
        assert score < 0.0, f"chill should pull score negative despite rap flood, got {score}"

    def test_generic_tags_constant_defined(self):
        assert "rap" in GENERIC_TAGS
        assert "hip-hop" in GENERIC_TAGS
        assert GENERIC_TAG_VOTE_MULTIPLIER < 0.5


# ── segment_sessions ─────────────────────────────────────────────────────────

class TestSegmentSessions:
    def _make_df(self, timestamps):
        return pd.DataFrame({"played_at": pd.to_datetime(timestamps, utc=True)})

    def test_single_session_no_gap(self):
        df = self._make_df([
            "2024-01-01 20:00", "2024-01-01 20:05", "2024-01-01 20:10",
        ])
        out = segment_sessions(df)
        assert out["session_id"].nunique() == 1

    def test_gap_splits_sessions(self):
        df = self._make_df([
            "2024-01-01 20:00",
            "2024-01-01 20:05",
            "2024-01-01 21:10",  # >30 min gap
            "2024-01-01 21:15",
        ])
        out = segment_sessions(df)
        assert out["session_id"].nunique() == 2

    def test_sorted_output(self):
        # Input is reverse-chronological; output should be sorted
        df = self._make_df([
            "2024-01-01 21:00",
            "2024-01-01 20:00",
            "2024-01-01 20:05",
        ])
        out = segment_sessions(df)
        assert list(out["played_at"]) == sorted(out["played_at"])

    def test_exactly_30_min_gap_is_same_session(self):
        df = self._make_df(["2024-01-01 20:00", "2024-01-01 20:30"])
        out = segment_sessions(df)
        assert out["session_id"].nunique() == 1

    def test_single_row(self):
        df = self._make_df(["2024-01-01 20:00"])
        out = segment_sessions(df)
        assert len(out) == 1
        assert out["session_id"].iloc[0] == 0


# ── compute_artist_fatigue_scores ────────────────────────────────────────────

def _empty_conn():
    """In-memory SQLite with empty songs/playlist_tracks — satisfies binge signal queries."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE songs (song_id TEXT, artist_name TEXT, audio_features TEXT);
        CREATE TABLE playlist_tracks (playlist_id TEXT, song_id TEXT, added_at TEXT);
    """)
    return conn

def _plays_df(rows: list) -> pd.DataFrame:
    """Build a minimal plays DataFrame for compute_artist_fatigue_scores."""
    data = []
    for song_id, artist, ts, skip in rows:
        data.append({
            "song_id":      song_id,
            "artist_name":  artist,
            "played_at":    pd.Timestamp(ts, tz="UTC"),
            "inferred_skip": skip,
            "play_source":  "playlist",
        })
    return pd.DataFrame(data)


class TestComputeArtistFatigueScores:
    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["song_id","artist_name","played_at",
                                    "inferred_skip","play_source"])
        assert compute_artist_fatigue_scores(df, _empty_conn()) == {}

    def test_artist_present_in_result(self):
        df = _plays_df([("s1", "Drake", "2026-06-09 10:00:00", "full")])
        result = compute_artist_fatigue_scores(df, _empty_conn())
        assert "Drake" in result

    def test_result_keys(self):
        df = _plays_df([("s1", "Drake", "2026-06-09 10:00:00", "full")])
        result = compute_artist_fatigue_scores(df, _empty_conn())
        assert "artist_fatigue" in result["Drake"]
        assert "artist_binge_score" in result["Drake"]

    def test_fatigue_bounded(self):
        plays = [("s1", "Drake", f"2026-06-0{i+1} 10:00:00", "full") for i in range(8)]
        df = _plays_df(plays)
        result = compute_artist_fatigue_scores(df, _empty_conn())
        assert 0.0 <= result["Drake"]["artist_fatigue"] <= 1.0
        assert 0.0 <= result["Drake"]["artist_binge_score"] <= 1.0

    def test_more_plays_more_fatigue(self):
        few  = _plays_df([("s1","Drake","2026-06-09 10:00:00","full")])
        many = _plays_df([("s1","Drake",f"2026-06-0{i+1} 10:00:00","full") for i in range(5)])
        assert (compute_artist_fatigue_scores(many, _empty_conn())["Drake"]["artist_fatigue"]
                > compute_artist_fatigue_scores(few, _empty_conn())["Drake"]["artist_fatigue"])

    def test_skips_reduce_fatigue_vs_full_plays(self):
        full = _plays_df([("s1","Drake","2026-06-09 10:00:00","full")])
        skip = _plays_df([("s1","Drake","2026-06-09 10:00:00","skip")])
        assert (compute_artist_fatigue_scores(full, _empty_conn())["Drake"]["artist_fatigue"]
                > compute_artist_fatigue_scores(skip, _empty_conn())["Drake"]["artist_fatigue"])

    def test_multiple_songs_same_artist_adds_up(self):
        one_song = _plays_df([("s1","Drake","2026-06-09 10:00:00","full")])
        two_songs = _plays_df([
            ("s1","Drake","2026-06-09 10:00:00","full"),
            ("s2","Drake","2026-06-09 11:00:00","full"),
        ])
        assert (compute_artist_fatigue_scores(two_songs, _empty_conn())["Drake"]["artist_fatigue"]
                > compute_artist_fatigue_scores(one_song, _empty_conn())["Drake"]["artist_fatigue"])

    def test_different_artists_independent(self):
        df = _plays_df([
            ("s1","Drake",   "2026-06-09 10:00:00","full"),
            ("s2","Kendrick","2026-06-09 10:00:00","full"),
        ])
        result = compute_artist_fatigue_scores(df, _empty_conn())
        assert "Drake" in result
        assert "Kendrick" in result
        # Each artist has only 1 play, so scores should be identical
        assert result["Drake"]["artist_fatigue"] == result["Kendrick"]["artist_fatigue"]

    def test_jam_excluded_plays_ignored(self):
        data = [{
            "song_id": "s1", "artist_name": "Drake",
            "played_at": pd.Timestamp("2026-06-09 10:00:00", tz="UTC"),
            "inferred_skip": "full", "play_source": "jam_excluded",
        }]
        df = pd.DataFrame(data)
        result = compute_artist_fatigue_scores(df, _empty_conn())
        assert "Drake" not in result


# ── _build_daily_plays ────────────────────────────────────────────────────────

def _plays_with_dates(dates: list, song_id="s1") -> pd.DataFrame:
    """Build a plays DataFrame with play_date column."""
    rows = []
    for d in dates:
        rows.append({
            "song_id":      song_id,
            "play_date":    d,
            "play_source":  "playlist",
        })
    return pd.DataFrame(rows)


class TestBuildDailyPlays:
    def test_single_day_one_play(self):
        today = date(2026, 6, 10)
        df = _plays_with_dates([today])
        result = _build_daily_plays(df, today, today)
        assert result == [1]

    def test_three_days_with_gap(self):
        d0 = date(2026, 6, 8)
        d1 = date(2026, 6, 9)
        d2 = date(2026, 6, 10)
        df = _plays_with_dates([d0, d0, d2])  # 2 plays day 0, 0 day 1, 1 day 2
        result = _build_daily_plays(df, d0, d2)
        assert result == [2, 0, 1]

    def test_plays_outside_window_excluded(self):
        d0 = date(2026, 6, 8)
        d1 = date(2026, 6, 9)
        before = date(2026, 6, 7)
        df = _plays_with_dates([before, d0, d1])
        result = _build_daily_plays(df, d0, d1)
        assert result == [1, 1]  # before-window play excluded

    def test_no_plays_returns_zeros(self):
        d0 = date(2026, 6, 8)
        d1 = date(2026, 6, 10)
        df = _plays_with_dates([])
        result = _build_daily_plays(df, d0, d1)
        assert result == [0, 0, 0]


# ── record_binge_episodes ─────────────────────────────────────────────────────

def _episode_conn():
    """In-memory SQLite with binge_episodes + required tables."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE songs (song_id TEXT, artist_name TEXT, audio_features TEXT);
        CREATE TABLE playlist_tracks (playlist_id TEXT, song_id TEXT, added_at TEXT);
        CREATE TABLE binge_episodes (
            episode_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id       TEXT NOT NULL,
            artist_name   TEXT,
            start_date    TEXT NOT NULL,
            end_date      TEXT,
            daily_plays   TEXT NOT NULL DEFAULT '[]',
            peak_plays    INTEGER DEFAULT 0,
            episode_length INTEGER
        );
    """)
    return conn


def _binge_fatigue_scores(song_id="s1", binge_score=0.8,
                           artist_name="Drake", days_ago=0) -> dict:
    """Minimal fatigue_scores dict for record_binge_episodes."""
    last_played = (_utc_today() - timedelta(days=days_ago)).isoformat() + "T00:00:00+00:00"
    return {song_id: {
        "binge_score": binge_score,
        "artist_name": artist_name,
        "last_played": last_played,
    }}


def _df_with_plays(song_id="s1", play_dates: list = None):
    """DataFrame with play_source and played_at for record_binge_episodes.
    play_dates may be date objects or ISO date strings."""
    rows = []
    for d in (play_dates or []):
        rows.append({
            "song_id":       song_id,
            "played_at":     pd.Timestamp(str(d), tz="UTC"),
            "play_source":   "playlist",
            "inferred_skip": "full",
        })
    return pd.DataFrame(rows, columns=["song_id", "played_at", "play_source", "inferred_skip"])


class TestRecordBingeEpisodes:
    def test_opens_new_episode_when_binge_detected(self):
        conn   = _episode_conn()
        scores = _binge_fatigue_scores(binge_score=0.8)
        df     = _df_with_plays(play_dates=["2026-06-10"])
        record_binge_episodes(scores, df, conn)
        rows = conn.execute("SELECT * FROM binge_episodes").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "s1"       # song_id
        assert rows[0][4] is None       # end_date NULL = open

    def test_no_episode_when_binge_score_zero(self):
        conn   = _episode_conn()
        scores = _binge_fatigue_scores(binge_score=0.0)
        df     = _df_with_plays(play_dates=["2026-06-10"])
        record_binge_episodes(scores, df, conn)
        rows = conn.execute("SELECT * FROM binge_episodes").fetchall()
        assert rows == []

    def test_updates_existing_open_episode(self):
        conn  = _episode_conn()
        today = _utc_today()
        d0    = today - timedelta(days=2)
        d1    = today - timedelta(days=1)
        # Seed an open episode that started 2 days ago
        conn.execute(
            "INSERT INTO binge_episodes "
            "(song_id, artist_name, start_date, daily_plays, peak_plays) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "Drake", d0.isoformat(), json.dumps([2, 1]), 2),
        )
        conn.commit()
        scores = _binge_fatigue_scores(binge_score=0.6, days_ago=0)
        df = _df_with_plays(play_dates=[d0, d0, d1, today])
        record_binge_episodes(scores, df, conn)
        row = conn.execute("SELECT daily_plays, peak_plays FROM binge_episodes").fetchone()
        daily = json.loads(row[0])
        assert len(daily) == 3          # d0, d1, today = 3 days
        assert daily[0] == 2            # 2 plays on day 0
        assert row[1] == 2              # peak still 2

    def test_closes_episode_after_gap(self):
        conn  = _episode_conn()
        today = _utc_today()
        start = today - timedelta(days=9)
        d1    = today - timedelta(days=8)
        conn.execute(
            "INSERT INTO binge_episodes "
            "(song_id, artist_name, start_date, daily_plays, peak_plays) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "Drake", start.isoformat(), json.dumps([3, 2, 1]), 3),
        )
        conn.commit()
        # binge_score = 0 and last played BINGE_END_GAP_DAYS + 1 days ago
        scores = _binge_fatigue_scores(binge_score=0.0, days_ago=BINGE_END_GAP_DAYS + 1)
        df = _df_with_plays(play_dates=[start, d1])
        record_binge_episodes(scores, df, conn)
        row = conn.execute("SELECT end_date FROM binge_episodes").fetchone()
        assert row[0] is not None       # episode closed

    def test_does_not_close_episode_within_gap(self):
        conn  = _episode_conn()
        today = _utc_today()
        start = today - timedelta(days=2)
        d1    = today - timedelta(days=1)
        conn.execute(
            "INSERT INTO binge_episodes "
            "(song_id, artist_name, start_date, daily_plays, peak_plays) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "Drake", start.isoformat(), json.dumps([2]), 2),
        )
        conn.commit()
        # binge_score = 0 but last played only 1 day ago (within gap threshold)
        scores = _binge_fatigue_scores(binge_score=0.0, days_ago=1)
        df = _df_with_plays(play_dates=[d1])
        record_binge_episodes(scores, df, conn)
        row = conn.execute("SELECT end_date FROM binge_episodes").fetchone()
        assert row[0] is None           # still open
