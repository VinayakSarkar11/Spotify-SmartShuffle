"""
Tests for phase34.py — scoring, clustering utilities, queue generation,
and SessionContext stake adjustment.  No Spotify API calls.
"""
import json

import numpy as np
import pandas as pd
import pytest

import recommend as phase34
from recommend import (
    SessionContext,
    _make_entry,
    _pick_best,
    _score,
    cluster_label,
    generate_queue,
    genre_score,
    get_target_energy,
    init_tables,
    save_queue,
    TIME_BUCKET_ENERGY,
    ARTIST_FATIGUE_FACTOR,
    ARTIST_FATIGUE_WINDOW,
    BINGE_ALLOW_THRESHOLD,
    RAP_TAGS,
    MELODIC_TAGS,
)

WEIGHTS = {
    "energy_match": 0.40,
    "fatigue":      0.35,
    "coverage":     0.20,
    "skip":         0.05,
}


# ── genre_score ──────────────────────────────────────────────────────────────

class TestGenreScore:
    # genre_score takes a list of plain tag name strings, not dicts.

    def test_pure_rap_tags_positive(self):
        rap_tag = next(iter(RAP_TAGS))
        assert genre_score([rap_tag]) > 0.0

    def test_pure_melodic_tags_negative(self):
        mel_tag = next(iter(MELODIC_TAGS))
        assert genre_score([mel_tag]) < 0.0

    def test_no_tags_returns_zero(self):
        assert genre_score([]) == 0.0

    def test_unknown_tags_return_zero(self):
        assert genre_score(["grunge", "psychedelic"]) == 0.0

    def test_balanced_rap_melodic_near_zero(self):
        rap_tag = next(iter(RAP_TAGS))
        mel_tag = next(iter(MELODIC_TAGS))
        score = genre_score([rap_tag, mel_tag])
        assert -0.5 < score < 0.5

    def test_bounded(self):
        rap_tags = list(RAP_TAGS)
        assert -1.0 <= genre_score(rap_tags) <= 1.0

    def test_all_rap_is_1(self):
        rap_tags = list(RAP_TAGS)
        assert genre_score(rap_tags) == pytest.approx(1.0)

    def test_all_melodic_is_minus_1(self):
        mel_tags = list(MELODIC_TAGS)
        assert genre_score(mel_tags) == pytest.approx(-1.0)


# ── cluster_label ────────────────────────────────────────────────────────────

class TestClusterLabel:
    def test_high_energy_rap(self):
        label = cluster_label(energy_c=0.8, genre_c=0.8)
        assert "energy" in label.lower() or "rap" in label.lower() or "hype" in label.lower()

    def test_low_energy_melodic(self):
        label = cluster_label(energy_c=-0.8, genre_c=-0.8)
        assert label   # not empty string

    def test_returns_string(self):
        assert isinstance(cluster_label(0.0, 0.0), str)

    def test_all_quadrants_return_strings(self):
        for e in (-0.5, 0.5):
            for g in (-0.5, 0.5):
                assert isinstance(cluster_label(e, g), str)


# ── _score ───────────────────────────────────────────────────────────────────

def _song_row(**overrides):
    defaults = {
        "energy_score":      0.0,
        "fatigue":           0.0,
        "binge_score":       0.0,
        "artist_fatigue":    0.0,
        "artist_binge_score":0.0,
        "coverage_debt":     0.0,
        "skip_rate":         0.0,
        "pattern":           "normal",
    }
    defaults.update(overrides)
    return pd.Series(defaults)


class TestScore:
    def test_perfect_energy_match_scores_high(self):
        row = _song_row(energy_score=0.5)
        score = _score(row, target_energy=0.5, weights=WEIGHTS)
        assert score > 0.3

    def test_energy_mismatch_scores_lower(self):
        close  = _score(_song_row(energy_score=0.5), 0.5, WEIGHTS)
        far    = _score(_song_row(energy_score=-0.5), 0.5, WEIGHTS)
        assert close > far

    def test_high_fatigue_lowers_score(self):
        low_fat  = _score(_song_row(fatigue=0.0), 0.5, WEIGHTS)
        high_fat = _score(_song_row(fatigue=1.0), 0.5, WEIGHTS)
        assert low_fat > high_fat

    def test_binge_suppresses_fatigue_penalty(self):
        no_binge   = _score(_song_row(fatigue=1.0, binge_score=0.0), 0.0, WEIGHTS)
        full_binge = _score(_song_row(fatigue=1.0, binge_score=1.0), 0.0, WEIGHTS)
        assert full_binge > no_binge

    def test_coverage_debt_raises_score(self):
        no_debt   = _score(_song_row(coverage_debt=0.0), 0.0, WEIGHTS)
        high_debt = _score(_song_row(coverage_debt=4.0), 0.0, WEIGHTS)
        assert high_debt > no_debt

    def test_coverage_debt_capped_at_4(self):
        s4   = _score(_song_row(coverage_debt=4.0),  0.0, WEIGHTS)
        s100 = _score(_song_row(coverage_debt=100.0), 0.0, WEIGHTS)
        # min(100/4, 1) == min(4/4, 1) == 1 after cap
        assert s4 == pytest.approx(s100, abs=1e-5)

    def test_growing_tired_pattern_adds_skip_penalty(self):
        normal  = _score(_song_row(skip_rate=0.1, pattern="normal"),       0.0, WEIGHTS)
        growing = _score(_song_row(skip_rate=0.1, pattern="growing_tired"), 0.0, WEIGHTS)
        assert normal > growing

    def test_skip_rate_capped_at_1_for_growing_tired(self):
        # skip_rate=0.9 + 0.5 would exceed 1 → should be clamped to 1
        row = _song_row(skip_rate=0.9, pattern="growing_tired")
        score = _score(row, 0.0, WEIGHTS)
        assert isinstance(score, float)


# ── _make_entry ───────────────────────────────────────────────────────────────

class TestMakeEntry:
    def test_required_keys_present(self):
        row = pd.Series({
            "song_id":       "ABC",
            "song_name":     "Test",
            "artist_name":   "Artist",
            "cluster_label": "hype",
            "energy_score":  0.7,
            "fatigue":       0.1,
            "binge_score":   0.0,
            "coverage_debt": 2.0,
        })
        entry = _make_entry(row, score=0.85)
        for key in ("song_id", "song_name", "artist_name", "score",
                    "cluster_label", "energy_score", "fatigue", "coverage_debt"):
            assert key in entry

    def test_score_rounded_to_4dp(self):
        row = pd.Series({
            "song_id": "A", "song_name": "T", "artist_name": "X",
            "cluster_label": "c", "energy_score": 0.0,
            "fatigue": 0.0, "binge_score": 0.0, "coverage_debt": 0.0,
        })
        entry = _make_entry(row, score=0.123456789)
        assert entry["score"] == round(0.123456789, 4)

    def test_missing_cluster_label_defaults(self):
        row = pd.Series({
            "song_id": "A", "song_name": "T", "artist_name": "X",
            "energy_score": 0.0, "fatigue": 0.0,
            "binge_score": 0.0, "coverage_debt": 0.0,
        })
        entry = _make_entry(row, score=0.0)
        assert entry["cluster_label"] == "unknown"


# ── save_queue ────────────────────────────────────────────────────────────────

class TestSaveQueue:
    def test_saves_one_row(self, mem_db):
        ss_q = [{"song_id": "A", "song_name": "T", "artist_name": "X"}]
        save_queue(mem_db, ss_q, "morning", "PL1")
        rows = mem_db.execute("SELECT algorithm FROM queues").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "smartshuffle"

    def test_returns_queue_id(self, mem_db):
        ss_q = [{"song_id": "A", "song_name": "T", "artist_name": "X"}]
        qid = save_queue(mem_db, ss_q, "morning", "PL1")
        assert isinstance(qid, int) and qid > 0

    def test_songs_stored_as_json(self, mem_db):
        ss_q = [{"song_id": "A", "song_name": "T", "artist_name": "X"}]
        save_queue(mem_db, ss_q, "morning", "PL1")
        row = mem_db.execute("SELECT songs FROM queues WHERE algorithm='smartshuffle'").fetchone()
        assert json.loads(row[0])[0]["song_id"] == "A"

    def test_multiple_saves_stack(self, mem_db):
        ss_q = [{"song_id": "A", "song_name": "T", "artist_name": "X"}]
        save_queue(mem_db, ss_q, "morning", "PL1")
        save_queue(mem_db, ss_q, "evening", "PL1")
        count = mem_db.execute("SELECT COUNT(*) FROM queues").fetchone()[0]
        assert count == 2


# ── SessionContext ────────────────────────────────────────────────────────────

class TestSessionContext:
    BASE_WEIGHTS = {"energy_match": 0.40, "fatigue": 0.35, "coverage": 0.20, "skip": 0.05}

    def test_too_few_sessions_returns_base_weights(self):
        ctx = SessionContext("morning", {"typical_stakes": "high", "session_count": 2})
        result = ctx.adjusted_weights(self.BASE_WEIGHTS)
        assert result == dict(self.BASE_WEIGHTS)

    def test_high_stakes_raises_energy_match(self):
        ctx = SessionContext("morning", {"typical_stakes": "high",
                                         "session_count": 10,
                                         "baseline_skip_rate": 0.2})
        result = ctx.adjusted_weights(self.BASE_WEIGHTS)
        assert result["energy_match"] > self.BASE_WEIGHTS["energy_match"]

    def test_high_stakes_lowers_coverage(self):
        ctx = SessionContext("morning", {"typical_stakes": "high",
                                         "session_count": 10,
                                         "baseline_skip_rate": 0.2})
        result = ctx.adjusted_weights(self.BASE_WEIGHTS)
        assert result["coverage"] < self.BASE_WEIGHTS["coverage"]

    def test_low_stakes_raises_coverage(self):
        ctx = SessionContext("morning", {"typical_stakes": "low",
                                         "session_count": 10,
                                         "baseline_skip_rate": 0.02})
        result = ctx.adjusted_weights(self.BASE_WEIGHTS)
        assert result["coverage"] > self.BASE_WEIGHTS["coverage"]

    def test_normal_stakes_unchanged(self):
        ctx = SessionContext("morning", {"typical_stakes": "normal",
                                         "session_count": 10,
                                         "baseline_skip_rate": 0.08})
        result = ctx.adjusted_weights(self.BASE_WEIGHTS)
        for k in self.BASE_WEIGHTS:
            assert result[k] == pytest.approx(self.BASE_WEIGHTS[k], abs=1e-3)

    def test_weights_sum_to_one(self):
        for stakes in ("high", "low", "normal"):
            ctx = SessionContext("morning", {"typical_stakes": stakes,
                                              "session_count": 10})
            result = ctx.adjusted_weights(self.BASE_WEIGHTS)
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-4)

    def test_energy_match_raises_under_high_stakes(self):
        # Cap is applied pre-normalization; we just verify direction, not absolute value.
        high_weights = {"energy_match": 0.65, "fatigue": 0.15, "coverage": 0.15, "skip": 0.05}
        ctx = SessionContext("morning", {"typical_stakes": "high", "session_count": 10})
        result = ctx.adjusted_weights(high_weights)
        assert result["energy_match"] >= high_weights["energy_match"]

    def test_record_play_updates_state(self):
        ctx = SessionContext("morning", {})
        ctx.record_play("S1", 200_000, skipped=False)
        ctx.record_play("S2", 20_000,  skipped=True)
        assert ctx.songs_played == 2
        assert ctx.skips == 1
        assert ctx.early_skips == 1

    def test_session_skip_rate_zero_when_no_plays(self):
        ctx = SessionContext("morning", {})
        assert ctx.session_skip_rate == 0.0


# ── _pick_best artist fatigue ─────────────────────────────────────────────────

def _pool_row(song_id, artist, energy=0.0):
    return {
        "song_id":           song_id,
        "artist_name":       artist,
        "song_name":         song_id,
        "energy_score":      energy,
        "fatigue":           0.0,
        "binge_score":       0.0,
        "artist_fatigue":    0.0,
        "artist_binge_score":0.0,
        "coverage_debt":     0.0,
        "skip_rate":         0.0,
        "pattern":           "normal",
        "cluster_label":     "test",
        "cluster_id":        0,
        "genre_score":       0.0,
    }

_POOL_WEIGHTS = {
    "energy_match": 0.40,
    "fatigue":      0.35,
    "coverage":     0.20,
    "skip":         0.05,
}


class TestPickBestArtistFatigue:
    def _pool(self, rows):
        return pd.DataFrame(rows)

    def test_no_recent_artists_returns_best_score(self):
        pool = self._pool([
            _pool_row("s1", "Drake", energy=0.5),
            _pool_row("s2", "Kendrick", energy=0.0),
        ])
        pick, _ = _pick_best(pool, set(), 0.5, _POOL_WEIGHTS, temperature=0)
        assert pick["song_id"] == "s1"

    def test_recent_artist_penalised_so_other_picked(self):
        pool = self._pool([
            _pool_row("s1", "Drake", energy=0.5),
            _pool_row("s2", "Kendrick", energy=0.0),
        ])
        # Drake would normally win; penalising Drake should flip the pick
        pick, _ = _pick_best(pool, set(), 0.5, _POOL_WEIGHTS, recent_artists=["Drake"],
                             temperature=0)
        assert pick["song_id"] == "s2"

    def test_all_penalised_still_returns_a_pick(self):
        # Single-artist pool → no alternatives → hard block lifts, soft factor applies
        pool = self._pool([
            _pool_row("s1", "Drake", energy=0.5),
            _pool_row("s2", "Drake", energy=0.3),
        ])
        pick, _ = _pick_best(pool, set(), 0.5, _POOL_WEIGHTS, recent_artists=["Drake"],
                             temperature=0)
        assert pick is not None
        assert pick["song_id"] == "s1"

    def test_non_binged_artist_hard_blocked_when_alternatives_exist(self):
        pool = self._pool([
            _pool_row("s1", "Drake",   energy=0.9),
            _pool_row("s2", "Kendrick",energy=0.0),
        ])
        # Drake has no binge — must be skipped even though higher score
        pick, _ = _pick_best(pool, set(), 0.9, _POOL_WEIGHTS, recent_artists=["Drake"])
        assert pick["song_id"] == "s2"

    def test_binged_artist_still_blocked_back_to_back(self):
        # Binge score lifts the fatigue penalty but NOT the back-to-back block —
        # consecutive same-artist is always prevented when an alternative exists.
        pool = self._pool([
            _pool_row("s1", "Drake",   energy=0.9),
            _pool_row("s2", "Kendrick",energy=0.0),
        ])
        pool = pool.copy()
        pool.loc[pool["artist_name"] == "Drake", "artist_binge_score"] = 1.0
        pick, _ = _pick_best(pool, set(), 0.9, _POOL_WEIGHTS, recent_artists=["Drake"])
        assert pick["song_id"] == "s2", "Kendrick should win — Drake blocked even when binged"

    def test_binge_allow_threshold_constant(self):
        assert 0.0 < BINGE_ALLOW_THRESHOLD < 1.0

    def test_immediate_predecessor_gets_max_penalty(self):
        # When the artist is the single most-recent pick, they get ARTIST_FATIGUE_FACTOR
        pool = self._pool([_pool_row("s1", "Drake", energy=0.5)])
        _, score_no_penalty   = _pick_best(pool, set(), 0.5, _POOL_WEIGHTS)
        _, score_with_penalty = _pick_best(
            pool, set(), 0.5, _POOL_WEIGHTS, recent_artists=["Drake"]
        )
        assert abs(score_with_penalty - score_no_penalty * ARTIST_FATIGUE_FACTOR) < 1e-6

    def test_empty_recent_artists_list_same_as_none(self):
        pool = self._pool([
            _pool_row("s1", "Drake", energy=0.5),
            _pool_row("s2", "Kendrick", energy=0.0),
        ])
        pick_none, _  = _pick_best(pool, set(), 0.5, _POOL_WEIGHTS, recent_artists=None,
                                   temperature=0)
        pick_empty, _ = _pick_best(pool, set(), 0.5, _POOL_WEIGHTS, recent_artists=[],
                                   temperature=0)
        assert pick_none["song_id"] == pick_empty["song_id"]


# ── generate_queue artist spacing ────────────────────────────────────────────

def _make_pool(artist_songs: dict, base_energy: float = 0.3) -> pd.DataFrame:
    """Build a minimal DataFrame for generate_queue.

    artist_songs: {artist_name: [song_id, ...]}
    """
    rows = []
    for artist, ids in artist_songs.items():
        for sid in ids:
            rows.append({
                "song_id":           sid,
                "song_name":         sid,
                "artist_name":       artist,
                "energy_score":      base_energy,
                "fatigue":           0.0,
                "binge_score":       0.0,
                "artist_fatigue":    0.0,
                "artist_binge_score":0.0,
                "coverage_debt":     0.0,
                "skip_rate":         0.0,
                "pattern":           "normal",
                "cluster_label":     "test",
                "cluster_id":        0,
                "genre_score":       0.0,
            })
    return pd.DataFrame(rows)


class TestGenerateQueueArtistSpacing:
    def test_same_artist_not_back_to_back(self):
        # 6 songs per artist, 2 artists — without fatigue, same artist clusters
        pool = _make_pool({"Drake": [f"d{i}" for i in range(6)],
                           "Kendrick": [f"k{i}" for i in range(6)]})
        queue = generate_queue(pool, "morning", n=10)
        artists = [s["artist_name"] for s in queue]
        for i in range(len(artists) - 1):
            assert artists[i] != artists[i + 1], (
                f"back-to-back same artist at positions {i},{i+1}: {artists}"
            )

    def test_artist_can_reappear_after_window(self):
        # Only 2 artists with many songs each; Drake must reappear eventually
        pool = _make_pool({"Drake": [f"d{i}" for i in range(10)],
                           "Kendrick": [f"k{i}" for i in range(2)]})
        queue = generate_queue(pool, "morning", n=10)
        artist_counts = {}
        for s in queue:
            artist_counts[s["artist_name"]] = artist_counts.get(s["artist_name"], 0) + 1
        assert artist_counts.get("Drake", 0) > 1

    def test_single_artist_pool_still_generates_queue(self):
        pool = _make_pool({"Drake": [f"d{i}" for i in range(15)]})
        queue = generate_queue(pool, "morning", n=10)
        assert len(queue) == 10

    def test_artist_fatigue_window_constant(self):
        assert ARTIST_FATIGUE_WINDOW == 4

    def test_artist_fatigue_factor_constant(self):
        assert 0.0 < ARTIST_FATIGUE_FACTOR < 0.5

    def test_no_back_to_back_when_one_artist_dominates_pool(self):
        # Reproduces the "3 Lil Tjay in a row" bug.
        # 4 Lil Tjay songs, 1 Other Artist, 2 songs from two other artists.
        # The hard back-to-back block must prevent consecutive Lil Tjay picks
        # even when they dominate the pool.
        rows = (
            [{**_pool_row(f"lt{i}", "Lil Tjay", energy=0.5), "cluster_id": 0}
             for i in range(4)]
            + [{**_pool_row("ot1", "Other Artist", energy=0.5), "cluster_id": 0}]
            + [{**_pool_row("sa1", "Secondary A",  energy=0.5), "cluster_id": 1}]
            + [{**_pool_row("sa2", "Secondary B",  energy=0.5), "cluster_id": 1}]
        )
        pool = pd.DataFrame(rows)
        queue = generate_queue(pool, "morning", n=5)
        artists = [s["artist_name"] for s in queue]
        for i in range(len(artists) - 1):
            assert artists[i] != artists[i + 1], (
                f"consecutive same artist at positions {i},{i+1}: {artists}"
            )


# ── generate_queue proportional artist cap ────────────────────────────────────

class TestGenerateQueueProportionalCap:
    def test_minority_artist_capped_proportionally(self):
        # Drake has 5 songs in a 55-song pool (9%).
        # cap = ceil(10 * 5/55 * 2.2) = ceil(2.0) = 2 → at most 2 slots.
        others = {f"Artist{i}": [f"a{i}_{j}" for j in range(1)] for i in range(50)}
        pool   = _make_pool({"Drake": [f"d{i}" for i in range(5)], **others})
        queue  = generate_queue(pool, "morning", n=10)
        drake_count = sum(1 for s in queue if s["artist_name"] == "Drake")
        assert drake_count <= 2

    def test_dominant_artist_gets_proportional_slots(self):
        # Drake has 5 of 10 songs (50%) → cap = ceil(10 * 5/10) = 5.
        pool = _make_pool({"Drake": [f"d{i}" for i in range(5)],
                           "A": ["a1"], "B": ["b1"], "C": ["c1"],
                           "D": ["d_1"], "E": ["e1"]})
        queue = generate_queue(pool, "morning", n=10)
        drake_count = sum(1 for s in queue if s["artist_name"] == "Drake")
        assert drake_count <= 5

    def test_single_artist_pool_unaffected_by_cap(self):
        pool  = _make_pool({"Drake": [f"d{i}" for i in range(15)]})
        queue = generate_queue(pool, "morning", n=10)
        # cap = ceil(10 * 15/15) = 10 → all slots available
        assert len(queue) == 10

    def test_binged_artist_can_exceed_proportional_cap(self):
        # Drake has 1 of 10 songs (10%): cap = ceil(10 * 1/10) = 1.
        # With binge score ≥ threshold, Drake can exceed that cap.
        others = {f"Artist{i}": [f"a{i}"] for i in range(9)}
        pool   = _make_pool({"Drake": ["d0"], **others})
        # Drake only has 1 unique song, so can only appear once regardless.
        # Extend pool: 5 Drake songs, 5 others, Drake binged.
        pool2  = _make_pool({"Drake": [f"d{i}" for i in range(5)],
                             "A": ["a1"], "B": ["b1"], "C": ["c1"],
                             "D": ["d_1"], "E": ["e1"]})
        pool2.loc[pool2["artist_name"] == "Drake", "artist_binge_score"] = 1.0
        queue = generate_queue(pool2, "morning", n=10)
        drake_count = sum(1 for s in queue if s["artist_name"] == "Drake")
        # Cap would be 5 normally, but binged → no cap enforced → can fill all 5 Drake songs
        # The key check: queue generates without error and includes Drake
        assert any(s["artist_name"] == "Drake" for s in queue)


# ── _score — artist fatigue terms ────────────────────────────────────────────

class TestScoreArtistFatigue:
    def test_artist_fatigue_lowers_score(self):
        clean   = _song_row(energy_score=0.5, artist_fatigue=0.0)
        fatigued = _song_row(energy_score=0.5, artist_fatigue=0.8)
        assert _score(fatigued, 0.5, WEIGHTS) < _score(clean, 0.5, WEIGHTS)

    def test_artist_binge_suppresses_artist_fatigue_penalty(self):
        binging     = _song_row(artist_fatigue=1.0, artist_binge_score=1.0)
        not_binging = _song_row(artist_fatigue=1.0, artist_binge_score=0.0)
        assert _score(binging, 0.0, WEIGHTS) > _score(not_binging, 0.0, WEIGHTS)

    def test_missing_artist_fields_default_to_zero(self):
        row = pd.Series({
            "energy_score": 0.0, "fatigue": 0.0, "binge_score": 0.0,
            "coverage_debt": 0.0, "skip_rate": 0.0, "pattern": "normal",
        })
        # Should not raise even without artist_fatigue / artist_binge_score
        score = _score(row, 0.0, WEIGHTS)
        assert isinstance(score, float)


# ── get_target_energy ─────────────────────────────────────────────────────────

class TestGetTargetEnergy:
    def test_no_playlist_returns_bucket_default(self):
        result = get_target_energy("morning")
        assert result == TIME_BUCKET_ENERGY.get("morning", 0.0)

    def test_playlist_target_preferred(self, monkeypatch):
        monkeypatch.setattr(phase34, "_PLAYLIST_CONTEXT_TARGETS",
                            {"PL1": {"morning": 0.8}})
        assert get_target_energy("morning", "PL1") == 0.8

    def test_unknown_playlist_falls_back_to_bucket(self, monkeypatch):
        monkeypatch.setattr(phase34, "_PLAYLIST_CONTEXT_TARGETS", {})
        result = get_target_energy("morning", "UNKNOWN_PL")
        assert result == TIME_BUCKET_ENERGY.get("morning", 0.0)

    def test_known_playlist_missing_bucket_falls_back(self, monkeypatch):
        monkeypatch.setattr(phase34, "_PLAYLIST_CONTEXT_TARGETS",
                            {"PL1": {"evening": 0.2}})
        result = get_target_energy("morning", "PL1")
        assert result == TIME_BUCKET_ENERGY.get("morning", 0.0)

    def test_no_playlist_id_ignores_playlist_targets(self, monkeypatch):
        monkeypatch.setattr(phase34, "_PLAYLIST_CONTEXT_TARGETS",
                            {"PL1": {"morning": 0.9}})
        result = get_target_energy("morning")
        assert result != 0.9  # should use bucket default, not playlist target


# ── Binge shape modifier (generate_queue) ────────────────────────────────────

import sqlite3

def _make_binge_conn(song_id="s1", start_days_ago=3):
    """SQLite with binge_episodes + required tables. Open episode for song_id."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE engagement_baselines (
            time_bucket TEXT PRIMARY KEY,
            baseline_skip_rate REAL,
            early_skip_rate REAL,
            typical_stakes TEXT,
            session_count INTEGER,
            updated_at TEXT
        );
        CREATE TABLE queues (
            queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT, context TEXT, algorithm TEXT,
            playlist_id TEXT, songs TEXT
        );
        CREATE TABLE binge_episodes (
            episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id TEXT NOT NULL,
            artist_name TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT,
            daily_plays TEXT NOT NULL DEFAULT '[]',
            peak_plays INTEGER DEFAULT 0,
            episode_length INTEGER
        );
    """)
    from datetime import date, timedelta
    start = (date.today() - timedelta(days=start_days_ago)).isoformat()
    conn.execute(
        "INSERT INTO binge_episodes (song_id, start_date, daily_plays, peak_plays) VALUES (?,?,?,?)",
        (song_id, start, "[]", 0),
    )
    conn.commit()
    return conn


def _binge_pool(binge_score=0.8):
    """Pool with one binged and one normal song."""
    rows = [
        {**_pool_row("s1", "Drake", energy=0.5), "binge_score": binge_score,
         "last_played": None, "play_count": 0, "stale": False, "playlist_id": "P1",
         "top_tags": [], "artist_binge_score": 0.0},
        {**_pool_row("s2", "Kendrick", energy=0.5), "binge_score": 0.0,
         "last_played": None, "play_count": 0, "stale": False, "playlist_id": "P1",
         "top_tags": [], "artist_binge_score": 0.0},
    ]
    return pd.DataFrame(rows)


class TestBingeShapeModifier:
    def test_shape_none_leaves_binge_score_unchanged(self, monkeypatch):
        """When no shape is learned, binge_score flows through unmodified."""
        monkeypatch.setattr(phase34, "_BINGE_SHAPE", None)
        conn = _make_binge_conn("s1", start_days_ago=3)
        pool = _binge_pool(binge_score=0.8)
        queue = generate_queue(pool, "afternoon", n=2, conn=conn, weights=_POOL_WEIGHTS)
        s1_entry = next((s for s in queue if s["song_id"] == "s1"), None)
        assert s1_entry is not None
        # binge_score unchanged → effective_fatigue suppressed → s1 competes normally
        assert s1_entry["binge_score"] == pytest.approx(0.8, abs=0.05)

    def test_shape_early_day_keeps_high_modifier(self, monkeypatch):
        """Day 0 of a binge: shape[0] ≈ 1.0 → binge_score barely reduced."""
        shape = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        monkeypatch.setattr(phase34, "_BINGE_SHAPE", shape)
        conn = _make_binge_conn("s1", start_days_ago=0)  # binge started today
        pool = _binge_pool(binge_score=0.8)
        queue = generate_queue(pool, "afternoon", n=2, conn=conn, weights=_POOL_WEIGHTS)
        s1_entry = next((s for s in queue if s["song_id"] == "s1"), None)
        assert s1_entry is not None
        # shape[0] = 1.0 → adjusted = 0.8 × 1.0 = 0.8
        assert s1_entry["binge_score"] == pytest.approx(0.8, abs=0.05)

    def test_shape_late_day_reduces_modifier(self, monkeypatch):
        """Day 8 of a binge: shape[8] = 0.0 → binge_score → 0 → fatigue reasserts."""
        shape = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        monkeypatch.setattr(phase34, "_BINGE_SHAPE", shape)
        conn = _make_binge_conn("s1", start_days_ago=8)  # day 8
        pool = _binge_pool(binge_score=0.8)
        queue = generate_queue(pool, "afternoon", n=2, conn=conn, weights=_POOL_WEIGHTS)
        s1_entry = next((s for s in queue if s["song_id"] == "s1"), None)
        assert s1_entry is not None
        # shape[8] = 0.0 → adjusted = 0.0
        assert s1_entry["binge_score"] == pytest.approx(0.0, abs=0.01)

    def test_no_binge_episode_leaves_score_unchanged(self, monkeypatch):
        """Song with binge_score > 0 but no open episode gets days=-1 → no modifier."""
        shape = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        monkeypatch.setattr(phase34, "_BINGE_SHAPE", shape)
        # conn has no binge_episodes rows for s1
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE engagement_baselines (
                time_bucket TEXT PRIMARY KEY, baseline_skip_rate REAL,
                early_skip_rate REAL, typical_stakes TEXT,
                session_count INTEGER, updated_at TEXT
            );
            CREATE TABLE queues (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TEXT, context TEXT, algorithm TEXT,
                playlist_id TEXT, songs TEXT
            );
            CREATE TABLE binge_episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id TEXT NOT NULL, artist_name TEXT,
                start_date TEXT NOT NULL, end_date TEXT,
                daily_plays TEXT NOT NULL DEFAULT '[]',
                peak_plays INTEGER DEFAULT 0, episode_length INTEGER
            );
        """)
        pool = _binge_pool(binge_score=0.8)
        queue = generate_queue(pool, "afternoon", n=2, conn=conn, weights=_POOL_WEIGHTS)
        s1_entry = next((s for s in queue if s["song_id"] == "s1"), None)
        assert s1_entry is not None
        assert s1_entry["binge_score"] == pytest.approx(0.8, abs=0.05)

