"""
Tests for model.py — pure logic + serialization (no Spotify/Last.fm calls).
"""
import json
import os
import tempfile

import pytest

import sqlite3
import numpy as np
import pandas as pd

from train import (
    _time_bucket,
    blend_weights,
    compare_smartshuffle_vs_natural,
    compute_alpha,
    compute_weekly_loss,
    learn_context_targets,
    load_params,
    save_params,
    DEFAULT_CONTEXT_TARGETS,
    MIN_PLAYS,
    DEFAULT_WEIGHTS,
    QUICK_SKIP_WEIGHT,
    MID_SKIP_WEIGHT,
    QUICK_SKIP_THRESHOLD,
    MID_SKIP_THRESHOLD,
    MIN_LOSS_PLAYS_PER_WEEK,
)


# ── _time_bucket ─────────────────────────────────────────────────────────────

class TestTimeBucket:
    @pytest.mark.parametrize("hour,expected", [
        (0, "late_night"), (5, "late_night"), (22, "late_night"), (23, "late_night"),
        (6, "morning"),    (11, "morning"),
        (12, "afternoon"), (17, "afternoon"),
        (18, "evening"),   (21, "evening"),
    ])
    def test_boundaries(self, hour, expected):
        assert _time_bucket(hour) == expected


# ── compute_alpha ─────────────────────────────────────────────────────────────

class TestComputeAlpha:
    def test_below_min_plays_is_one(self):
        assert compute_alpha(0) == 1.0
        assert compute_alpha(MIN_PLAYS - 1) == 1.0

    def test_at_min_plays_is_one(self):
        assert compute_alpha(MIN_PLAYS) == 1.0

    def test_decreases_with_more_plays(self):
        assert compute_alpha(100) < compute_alpha(50)
        assert compute_alpha(300) < compute_alpha(100)

    def test_floor_at_0_3(self):
        # 530+ plays should hit the 0.30 floor
        assert compute_alpha(530) == 0.3
        assert compute_alpha(10_000) == 0.3

    def test_monotone_decreasing(self):
        values = [compute_alpha(n) for n in range(0, 600, 20)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]


# ── blend_weights ─────────────────────────────────────────────────────────────

class TestBlendWeights:
    DEFAULTS = {"energy_match": 0.4, "fatigue": 0.35, "coverage": 0.20, "skip": 0.05}
    LEARNED  = {"energy_match": 0.6, "fatigue": 0.20, "coverage": 0.15, "skip": 0.05}

    def test_alpha_1_returns_defaults(self):
        blended = blend_weights(self.DEFAULTS, self.LEARNED, alpha=1.0)
        for k in self.DEFAULTS:
            assert blended[k] == pytest.approx(self.DEFAULTS[k], abs=1e-6)

    def test_alpha_0_returns_learned(self):
        blended = blend_weights(self.DEFAULTS, self.LEARNED, alpha=0.0)
        for k in self.LEARNED:
            assert blended[k] == pytest.approx(self.LEARNED[k], abs=1e-6)

    def test_alpha_half_is_midpoint(self):
        blended = blend_weights(self.DEFAULTS, self.LEARNED, alpha=0.5)
        for k in self.DEFAULTS:
            mid = 0.5 * self.DEFAULTS[k] + 0.5 * self.LEARNED[k]
            assert blended[k] == pytest.approx(mid, abs=1e-4)

    def test_all_keys_present(self):
        blended = blend_weights(self.DEFAULTS, self.LEARNED, alpha=0.5)
        assert set(blended.keys()) == set(self.DEFAULTS.keys())


# ── save_params / load_params ─────────────────────────────────────────────────

class TestParamsSerialization:
    def test_round_trip(self):
        params = {
            "weights": DEFAULT_WEIGHTS,
            "context_targets": {"morning": 0.3, "late_night": -0.2},
            "alpha": 0.75,
            "n_plays": 120,
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_params(params, path)
            loaded = load_params(path)
            assert loaded["weights"]  == params["weights"]
            assert loaded["alpha"]    == params["alpha"]
            assert loaded["n_plays"]  == params["n_plays"]
            assert "trained_at" in loaded   # save_params stamps this
        finally:
            os.unlink(path)

    def test_load_missing_file_returns_none(self):
        result = load_params("/tmp/__nonexistent_params_file__.json")
        assert result is None

    def test_saved_file_is_valid_json(self):
        params = {"weights": DEFAULT_WEIGHTS, "alpha": 1.0, "n_plays": 0}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_params(params, path)
            with open(path) as f:
                parsed = json.load(f)
            assert "weights" in parsed
        finally:
            os.unlink(path)


# ── compare_smartshuffle_vs_natural ──────────────────────────────────────────

_play_counter = 0

def _to_iso_time(n: int, date: str = "2026-01-01") -> str:
    """Convert a counter to a valid ISO datetime: counter as seconds from midnight."""
    h, r = divmod(n, 3600)
    m, s = divmod(r, 60)
    return f"{date}T{h:02d}:{m:02d}:{s:02d}"

def _insert_plays(conn, source, skip_count, full_count):
    """Insert plays with unique timestamps into the plays table."""
    global _play_counter
    # Create a matching queue_push with skips_inferred_at set so loss queries include these plays.
    if source.endswith("_queued"):
        algo = source[:-len("_queued")]
        push_ts = _to_iso_time(_play_counter)
        conn.execute(
            "INSERT INTO queue_pushes (queue_id, algorithm, pushed_at, skips_inferred_at)"
            " VALUES (999, ?, ?, ?)",
            (algo, push_ts, push_ts),
        )
    rows = (
        [("skip", 60000)]  * skip_count +
        [("full", 210000)] * full_count
    )
    for skip_label, dur_played in rows:
        _play_counter += 1
        conn.execute("""
            INSERT INTO plays (song_id, played_at, play_duration_ms, duration_ms,
                               play_source, inferred_skip)
            VALUES ('AAA', ?, ?, 210000, ?, ?)
        """, (_to_iso_time(_play_counter), dur_played, source, skip_label))
    conn.commit()


class TestCompareSmartshuffleVsNatural:
    def test_returns_empty_when_no_data(self, mem_db):
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result == {}

    def test_returns_empty_below_20_smartshuffle_plays(self, mem_db):
        # 19 smartshuffle plays — threshold requires 20+
        _insert_plays(mem_db, "smartshuffle_queued", 9, 10)
        _insert_plays(mem_db, "playlist",            5, 10)
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result == {}

    def test_returns_data_at_20_smartshuffle_plays(self, mem_db):
        _insert_plays(mem_db, "smartshuffle_queued",     10, 10)
        _insert_plays(mem_db, "random_baseline_queued",   5,  5)
        result = compare_smartshuffle_vs_natural(mem_db)
        assert "smartshuffle" in result
        assert "baseline"     in result

    def test_skip_rate_calculation(self, mem_db):
        _insert_plays(mem_db, "smartshuffle_queued",    4, 16)   # 20 plays, skip=0.2
        _insert_plays(mem_db, "random_baseline_queued", 8,  2)   # 10 plays, skip=0.8
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result["smartshuffle"]["skip_rate"] == pytest.approx(0.2, abs=0.01)
        assert result["baseline"]["skip_rate"]     == pytest.approx(0.8, abs=0.01)

    def test_full_rate_calculation(self, mem_db):
        _insert_plays(mem_db, "smartshuffle_queued",    4, 16)   # full_rate = 0.8
        _insert_plays(mem_db, "random_baseline_queued", 5,  5)   # full_rate = 0.5
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result["smartshuffle"]["full_rate"] == pytest.approx(0.8, abs=0.01)
        assert result["baseline"]["full_rate"]     == pytest.approx(0.5, abs=0.01)

    def test_n_counts_match_inserted(self, mem_db):
        _insert_plays(mem_db, "smartshuffle_queued",    7, 13)   # 20
        _insert_plays(mem_db, "random_baseline_queued", 3,  7)   # 10
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result["smartshuffle"]["n"] == 20
        assert result["baseline"]["n"]     == 10

    def test_null_inferred_skip_excluded(self, mem_db):
        global _play_counter
        _insert_plays(mem_db, "smartshuffle_queued", 10, 10)
        for _ in range(50):
            _play_counter += 1
            mem_db.execute("""
                INSERT INTO plays (song_id, played_at, play_duration_ms, duration_ms,
                                   play_source, inferred_skip)
                VALUES ('AAA', ?, 100000, 210000, 'smartshuffle_queued', NULL)
            """, (f"2026-01-01T{_play_counter:06d}",))
        mem_db.commit()
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result["smartshuffle"]["n"] == 20

    def test_other_sources_not_counted(self, mem_db):
        global _play_counter
        _insert_plays(mem_db, "smartshuffle_queued",    10, 10)
        _insert_plays(mem_db, "random_baseline_queued",  5,  5)
        # manual_search and playlist plays — not in either comparison group
        for _ in range(100):
            _play_counter += 1
            mem_db.execute("""
                INSERT INTO plays (song_id, played_at, play_duration_ms, duration_ms,
                                   play_source, inferred_skip)
                VALUES ('AAA', ?, 60000, 210000, 'playlist', 'skip')
            """, (f"2026-01-01T{_play_counter:06d}",))
        mem_db.commit()
        result = compare_smartshuffle_vs_natural(mem_db)
        assert result["smartshuffle"]["n"] == 20
        assert result["baseline"]["n"]     == 10


# ── DEFAULT_WEIGHTS includes artist_fatigue ───────────────────────────────────

class TestDefaultWeights:
    def test_artist_fatigue_present(self):
        assert "artist_fatigue" in DEFAULT_WEIGHTS

    def test_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-6


# ── learn_context_targets — playlist-specific ─────────────────────────────────

def _ct_df(rows):
    """Build a minimal DataFrame for learn_context_targets."""
    return pd.DataFrame([
        {"song_id": r[0], "energy_score": r[1], "completion_ratio": r[2], "time_bucket": r[3]}
        for r in rows
    ])

def _ct_conn(song_playlist_map):
    """In-memory DB with playlist_tracks for context-target tests."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE playlist_tracks (song_id TEXT, playlist_id TEXT)")
    for song_id, playlist_id in song_playlist_map.items():
        conn.execute("INSERT INTO playlist_tracks VALUES (?, ?)", (song_id, playlist_id))
    conn.commit()
    return conn


class TestLearnContextTargets:
    def test_returns_two_values(self):
        df   = _ct_df([("s1", 0.5, 1.0, "morning")] * 3)
        conn = _ct_conn({})
        result = learn_context_targets(df, conn)
        assert isinstance(result, tuple) and len(result) == 2

    def test_bucket_target_learned_from_data(self):
        df   = _ct_df([("s1", 0.8, 1.0, "morning"),
                       ("s2", 0.6, 1.0, "morning"),
                       ("s3", 0.7, 1.0, "morning")])
        conn = _ct_conn({})
        targets, _ = learn_context_targets(df, conn)
        assert abs(targets["morning"] - np.mean([0.8, 0.6, 0.7])) < 0.05

    def test_below_3_plays_uses_default(self):
        df   = _ct_df([("s1", 0.9, 1.0, "afternoon"), ("s2", 0.1, 1.0, "afternoon")])
        conn = _ct_conn({})
        targets, _ = learn_context_targets(df, conn)
        assert targets["afternoon"] == DEFAULT_CONTEXT_TARGETS["afternoon"]

    def test_playlist_target_learned_when_enough_data(self):
        df   = _ct_df([("s1", 0.9, 1.0, "morning"),
                       ("s2", 0.8, 1.0, "morning"),
                       ("s3", 0.7, 1.0, "morning")])
        conn = _ct_conn({"s1": "PL1", "s2": "PL1", "s3": "PL1"})
        _, playlist_targets = learn_context_targets(df, conn)
        assert "PL1" in playlist_targets
        assert "morning" in playlist_targets["PL1"]

    def test_playlist_targets_empty_when_no_playlist_data(self):
        df   = _ct_df([("s1", 0.5, 1.0, "evening")] * 5)
        conn = _ct_conn({})
        _, playlist_targets = learn_context_targets(df, conn)
        assert playlist_targets == {}

    def test_ambiguous_song_excluded_from_playlist_targets(self):
        # s1 in two playlists → ambiguous → should not contribute to either
        df   = _ct_df([("s1", 0.9, 1.0, "morning")] * 5)
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE playlist_tracks (song_id TEXT, playlist_id TEXT)")
        conn.execute("INSERT INTO playlist_tracks VALUES ('s1', 'PL1')")
        conn.execute("INSERT INTO playlist_tracks VALUES ('s1', 'PL2')")
        conn.commit()
        _, playlist_targets = learn_context_targets(df, conn)
        assert "PL1" not in playlist_targets
        assert "PL2" not in playlist_targets

    def test_different_playlists_get_different_targets(self):
        df = _ct_df([
            ("s1", 0.9, 1.0, "morning"),
            ("s2", 0.8, 1.0, "morning"),
            ("s3", 0.7, 1.0, "morning"),
            ("s4", -0.5, 1.0, "morning"),
            ("s5", -0.6, 1.0, "morning"),
            ("s6", -0.4, 1.0, "morning"),
        ])
        conn = _ct_conn({"s1": "Hype", "s2": "Hype", "s3": "Hype",
                         "s4": "Chill", "s5": "Chill", "s6": "Chill"})
        _, playlist_targets = learn_context_targets(df, conn)
        assert playlist_targets["Hype"]["morning"] > 0
        assert playlist_targets["Chill"]["morning"] < 0


# ── compute_weekly_loss ───────────────────────────────────────────────────────

_loss_counter = 0

def _loss_play(conn, source: str, ratio: float, date_iso: str):
    """Insert one play with a specific completion ratio on a given date."""
    global _loss_counter
    _loss_counter += 1
    total_ms  = 200_000
    played_ms = int(total_ms * ratio)
    conn.execute("""
        INSERT INTO plays (song_id, played_at, play_duration_ms, duration_ms,
                           play_source, inferred_skip)
        VALUES ('AAA', ?, ?, ?, ?, ?)
    """, (_to_iso_time(_loss_counter, date_iso),
          played_ms, total_ms, source,
          "skip" if ratio < 0.8 else "full"))
    conn.commit()

def _fill_week(conn, source, ratio, date_iso, n=MIN_LOSS_PLAYS_PER_WEEK):
    global _loss_counter
    if source.endswith("_queued"):
        algo = source[:-len("_queued")]
        push_ts = _to_iso_time(_loss_counter, date_iso)
        conn.execute(
            "INSERT INTO queue_pushes (queue_id, algorithm, pushed_at, skips_inferred_at)"
            " VALUES (999, ?, ?, ?)",
            (algo, push_ts, push_ts),
        )
        conn.commit()
    for _ in range(n):
        _loss_play(conn, source, ratio, date_iso)


class TestComputeWeeklyLoss:
    def test_returns_empty_when_no_data(self, mem_db):
        assert compute_weekly_loss(mem_db) == {}

    def test_full_plays_give_zero_loss(self, mem_db):
        _fill_week(mem_db, "smartshuffle_queued", 0.95, "2026-01-05")
        result = compute_weekly_loss(mem_db)
        assert result["smartshuffle"]["total"] == 0.0
        assert result["smartshuffle"]["weekly"][0]["loss"] == 0.0

    def test_all_quick_skips_give_quick_weight(self, mem_db):
        ratio = 0.10   # below QUICK_SKIP_THRESHOLD
        _fill_week(mem_db, "smartshuffle_queued", ratio, "2026-01-05")
        result = compute_weekly_loss(mem_db)
        assert result["smartshuffle"]["total"] == pytest.approx(QUICK_SKIP_WEIGHT, abs=1e-4)

    def test_all_mid_skips_give_mid_weight(self, mem_db):
        ratio = 0.50   # in [QUICK_SKIP_THRESHOLD, MID_SKIP_THRESHOLD)
        _fill_week(mem_db, "smartshuffle_queued", ratio, "2026-01-05")
        result = compute_weekly_loss(mem_db)
        assert result["smartshuffle"]["total"] == pytest.approx(MID_SKIP_WEIGHT, abs=1e-4)

    def test_mixed_weighted_average(self, mem_db):
        # 6 quick skips + 4 mid skips, 0 full plays → n=10
        _fill_week(mem_db, "smartshuffle_queued", 0.10, "2026-01-05", n=6)
        _fill_week(mem_db, "smartshuffle_queued", 0.50, "2026-01-05", n=4)
        result    = compute_weekly_loss(mem_db)
        expected  = QUICK_SKIP_WEIGHT * 0.6 + MID_SKIP_WEIGHT * 0.4
        assert result["smartshuffle"]["total"] == pytest.approx(expected, abs=1e-4)

    def test_weekly_grouping_produces_separate_entries(self, mem_db):
        # Two distinct ISO weeks
        _fill_week(mem_db, "smartshuffle_queued", 0.10, "2026-01-05")   # week 1 (Mon)
        _fill_week(mem_db, "smartshuffle_queued", 0.95, "2026-01-12")   # week 2 (Mon)
        result  = compute_weekly_loss(mem_db)
        weekly  = result["smartshuffle"]["weekly"]
        assert len(weekly) == 2
        assert weekly[0]["week"] < weekly[1]["week"]
        assert weekly[0]["loss"] > weekly[1]["loss"]   # week 1 all skips, week 2 all full

    def test_plays_on_same_week_different_days_grouped(self, mem_db):
        # Mon–Sun of the same ISO week should all land in one bucket.
        # Each day gets its own push (simulates one session per day).
        for day in ("2026-01-05", "2026-01-06", "2026-01-07",
                    "2026-01-08", "2026-01-09"):
            _fill_week(mem_db, "smartshuffle_queued", 0.10, day, n=1)
        result = compute_weekly_loss(mem_db)
        assert len(result["smartshuffle"]["weekly"]) == 1
        assert result["smartshuffle"]["weekly"][0]["n"] == 5

    def test_week_below_min_plays_excluded(self, mem_db):
        # One full week (n=5, meets threshold) + one thin week (n=2, excluded)
        _fill_week(mem_db, "smartshuffle_queued", 0.95, "2026-01-05", n=MIN_LOSS_PLAYS_PER_WEEK)
        for _ in range(MIN_LOSS_PLAYS_PER_WEEK - 1):
            _loss_play(mem_db, "smartshuffle_queued", 0.10, "2026-01-12")
        result = compute_weekly_loss(mem_db)
        weekly = result["smartshuffle"]["weekly"]
        assert len(weekly) == 1
        assert weekly[0]["week"] == "2026-01-05"

    def test_other_sources_excluded(self, mem_db):
        _fill_week(mem_db, "manual_search", 0.10, "2026-01-05", n=20)
        _fill_week(mem_db, "album_browse",  0.10, "2026-01-05", n=20)
        result = compute_weekly_loss(mem_db)
        assert result == {}

    def test_both_sources_computed_independently(self, mem_db):
        _fill_week(mem_db, "smartshuffle_queued",    0.10, "2026-01-05")   # all quick skip
        _fill_week(mem_db, "random_baseline_queued", 0.95, "2026-01-05")   # all full
        result = compute_weekly_loss(mem_db)
        assert "smartshuffle" in result
        assert "baseline"     in result
        assert result["smartshuffle"]["total"] == pytest.approx(QUICK_SKIP_WEIGHT, abs=1e-4)
        assert result["baseline"]["total"]     == pytest.approx(0.0, abs=1e-4)

    def test_total_aggregates_across_weeks(self, mem_db):
        # Week 1: all quick skips (n=5); week 2: all full plays (n=5)
        # Total quick_rate = 5/10 = 0.5 → loss = QUICK_SKIP_WEIGHT * 0.5
        _fill_week(mem_db, "smartshuffle_queued", 0.10, "2026-01-05")
        _fill_week(mem_db, "smartshuffle_queued", 0.95, "2026-01-12")
        result   = compute_weekly_loss(mem_db)
        expected = QUICK_SKIP_WEIGHT * 0.5
        assert result["smartshuffle"]["total"] == pytest.approx(expected, abs=1e-4)

    def test_weekly_list_sorted_chronologically(self, mem_db):
        _fill_week(mem_db, "smartshuffle_queued", 0.10, "2026-02-02")
        _fill_week(mem_db, "smartshuffle_queued", 0.10, "2026-01-05")
        _fill_week(mem_db, "smartshuffle_queued", 0.10, "2026-01-19")
        weeks = [w["week"] for w in compute_weekly_loss(mem_db)["smartshuffle"]["weekly"]]
        assert weeks == sorted(weeks)
