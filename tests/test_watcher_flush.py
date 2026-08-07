"""
Tests for watcher.py vibe-shift flush and no-gap guarantees.

Core invariant: after a flush, upcoming_ids[0] is ALWAYS kept as a Spotify
buffer so playback never has an empty next-song slot during the refill window.
"""
import json
import sys
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# ── Patch spotipy before watcher is imported ─────────────────────────────────
_mock_sp_module     = MagicMock()
_mock_oauth2_module = MagicMock()
_mock_sp_exceptions = MagicMock()
_mock_sp_exceptions.SpotifyException = Exception
_mock_sp_module.exceptions = _mock_sp_exceptions

with patch.dict(sys.modules, {
    "spotipy":            _mock_sp_module,
    "spotipy.oauth2":     _mock_oauth2_module,
    "spotipy.exceptions": _mock_sp_exceptions,
}):
    import watcher

# Keep watcher registered so patch("watcher._load_vibe_map") targets the same object.
# Without this, the patch.dict restore removes watcher from sys.modules, causing
# patch() to import a fresh copy and modify that instead of the one we imported.
sys.modules["watcher"] = watcher

from watcher import (
    _should_flush,
    _check_refill,
    _SKIP_VIBE_DELTA_MIN,
    _FLUSH_MAX,
    _VELOCITY_FREEZE_MINUTES,
    MIN_REFILL_INTERVAL,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session_state(
    skip_cluster=None,
    comp_vibe=None,
    delta_clear=True,
    updated_at=None,
):
    now = updated_at or datetime.now(timezone.utc).isoformat()
    state = {
        "time_bucket": "afternoon",
        "skip_delta_clear": delta_clear,
        "updated_at": now,
    }
    if skip_cluster:
        state["skip_cluster_vibe_content"] = skip_cluster[0]
        state["skip_cluster_vibe_melodic"] = skip_cluster[1]
        state["skip_cluster_vibe_bpm"]     = skip_cluster[2]
    if comp_vibe:
        state["recent_comp_vibe_content"]  = comp_vibe[0]
        state["recent_comp_vibe_melodic"]  = comp_vibe[1]
        state["recent_comp_vibe_bpm"]      = comp_vibe[2]
    return state


def _make_rolling(flush_count=0, last_flush_signal_at=None, velocity_zeroed_until=None):
    return {
        "playlist_id":          "PLAYLIST123",
        "source_playlist_id":   "SOURCE456",
        "algorithm":            "smartshuffle",
        "session_push_id":      None,
        "enabled":              True,
        "flush_count":          flush_count,
        "last_flush_signal_at": last_flush_signal_at or "",
        "velocity_zeroed_until": velocity_zeroed_until,
        "last_refill_at":       None,
    }


# Vibes: comp at melodic=-0.5, skip at melodic=+0.5, upcoming (same dir) at +0.4
COMP        = (0.0, -0.5, 0.0)
SKIP        = (0.0, +0.5, 0.0)
UP_SAME_DIR = (0.0, +0.4, 0.0)   # dist from COMP ≈ 0.9 > 0.3; dot with SKIP dir > 0
UP_OPP_DIR  = (0.0, -0.6, 0.0)   # dist from COMP ≈ 0.1 and/or opposite direction


# ── _should_flush unit tests ──────────────────────────────────────────────────

class TestShouldFlush:
    """
    _load_vibe_map is patched in every test so the DB is never touched.
    The patch makes upcoming IDs resolve to UP_SAME_DIR by default.
    """

    def _run(self, state, upcoming, rolling, upcoming_vibe=UP_SAME_DIR):
        vibe_map_return = {sid: upcoming_vibe for sid in upcoming[1:]}
        with patch("watcher._load_vibe_map", return_value=vibe_map_return):
            return _should_flush(state, upcoming, {}, rolling)

    def test_passes_all_gates(self):
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling()
        result  = self._run(state, ["BUF", "S1", "S2", "S3"], rolling)
        assert result is True

    def test_no_signal_when_delta_clear_false(self):
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP, delta_clear=False)
        rolling = _make_rolling()
        assert self._run(state, ["BUF", "S1", "S2"], rolling) is False

    def test_no_signal_when_already_acted_on(self):
        old_ts  = "2026-08-07T10:00:00+00:00"
        new_ts  = "2026-08-07T10:05:00+00:00"
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP, updated_at=old_ts)
        rolling = _make_rolling(last_flush_signal_at=new_ts)
        assert self._run(state, ["BUF", "S1"], rolling) is False

    def test_no_signal_when_flush_count_maxed(self):
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling(flush_count=_FLUSH_MAX)
        assert self._run(state, ["BUF", "S1", "S2"], rolling) is False

    def test_no_signal_when_velocity_frozen(self):
        future  = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling(velocity_zeroed_until=future)
        assert self._run(state, ["BUF", "S1", "S2"], rolling) is False

    def test_no_signal_when_only_buffer_remaining(self):
        """Only 1 upcoming song — keeping it as buffer leaves nothing to evaluate."""
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling()
        assert self._run(state, ["BUF"], rolling) is False

    def test_no_signal_when_upcoming_distance_too_small(self):
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling()
        near_comp = (0.0, -0.45, 0.0)   # distance ~0.05 from COMP, below threshold
        assert self._run(state, ["BUF", "S1", "S2"], rolling, upcoming_vibe=near_comp) is False

    def test_no_signal_when_upcoming_in_opposite_direction(self):
        """Upcoming is far from comp but in the OPPOSITE direction from the skip."""
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling()
        assert self._run(state, ["BUF", "S1", "S2"], rolling, upcoming_vibe=UP_OPP_DIR) is False

    def test_frozen_velocity_expired_allows_flush(self):
        past    = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        state   = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP)
        rolling = _make_rolling(velocity_zeroed_until=past)
        assert self._run(state, ["BUF", "S1", "S2"], rolling) is True


# ── _check_refill flush path ──────────────────────────────────────────────────

class TestCheckRefillFlush:
    """
    Patches:
    - watcher._playlist_position_info   → controls what Spotify says is upcoming
    - watcher._load_vibe_map            → injects vibe data without hitting DB
    - watcher.subprocess.run            → prevents real recommend.py execution
    - watcher._write_rolling_state      → capture rolling state writes
    - sqlite3.connect                   → mock DB for push recording
    """

    def _make_state(self, updated_at=None):
        return _make_session_state(
            skip_cluster=SKIP, comp_vibe=COMP, delta_clear=True,
            updated_at=updated_at,
        )

    def _db_mock(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__  = MagicMock(return_value=False)
        return conn

    def _run_check_refill(self, upcoming, rolling=None, state=None, plays=None,
                          upcoming_vibe=UP_SAME_DIR):
        if rolling is None:
            rolling = _make_rolling()
        if state is None:
            state = self._make_state()

        sp = MagicMock()
        sp.playlist_remove_all_occurrences_of_items.return_value = {}
        sp.playlist_add_items.return_value = {}

        vibe_map_return = {sid: upcoming_vibe for sid in upcoming[1:]}

        with patch("watcher._playlist_position_info", return_value=(len(upcoming), upcoming)), \
             patch("watcher._load_vibe_map", return_value=vibe_map_return), \
             patch("watcher.subprocess.run", return_value=MagicMock(returncode=0, stderr="")), \
             patch("watcher._write_rolling_state"), \
             patch("sqlite3.connect", return_value=self._db_mock()):
            _check_refill(sp, plays or [], rolling, session_state=state, vibe_map={})

        return sp, rolling

    # ── Core no-gap invariant ─────────────────────────────────────────────────

    def test_buffer_song_never_removed(self):
        """pos 0 of upcoming must NEVER appear in the playlist remove call."""
        upcoming = ["BUFFER", "S1", "S2", "S3", "S4", "S5"]
        sp, _    = self._run_check_refill(upcoming)

        remove_calls = sp.playlist_remove_all_occurrences_of_items.call_args_list
        assert len(remove_calls) == 1, "Expected exactly one remove call"
        removed_uris = remove_calls[0][0][1]
        removed_ids  = [u.split(":")[-1] for u in removed_uris]

        assert "BUFFER" not in removed_ids, "Buffer song (pos 0) must never be removed"
        assert set(removed_ids) == {"S1", "S2", "S3", "S4", "S5"}

    def test_refill_runs_immediately_after_flush(self):
        """recommend.py subprocess must be called right after the flush."""
        upcoming = ["BUFFER", "S1", "S2", "S3"]
        with patch("watcher._playlist_position_info", return_value=(len(upcoming), upcoming)), \
             patch("watcher._load_vibe_map", return_value={s: UP_SAME_DIR for s in upcoming[1:]}), \
             patch("watcher._write_rolling_state"), \
             patch("sqlite3.connect", return_value=self._db_mock()) as _, \
             patch("watcher.subprocess.run",
                   return_value=MagicMock(returncode=0, stderr="")) as mock_run:
            sp = MagicMock()
            sp.playlist_remove_all_occurrences_of_items.return_value = {}
            sp.playlist_add_items.return_value = {}
            _check_refill(sp, [], _make_rolling(), session_state=self._make_state(), vibe_map={})
            assert mock_run.called, "recommend.py subprocess must run after flush"

    def test_flush_bypasses_cooldown(self):
        """Flush triggers even if MIN_REFILL_INTERVAL has not elapsed."""
        recent_refill = (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat()
        rolling = _make_rolling()
        rolling["last_refill_at"] = recent_refill   # only 15s ago, under 90s guard

        upcoming = ["BUFFER", "S1", "S2"]
        sp, _    = self._run_check_refill(upcoming, rolling=rolling)

        assert sp.playlist_remove_all_occurrences_of_items.called, \
            "Flush must bypass MIN_REFILL_INTERVAL"

    def test_no_flush_when_only_buffer_remains(self):
        """Only 1 upcoming song → no candidates to remove, no remove call."""
        upcoming = ["BUFFER"]
        sp, _    = self._run_check_refill(upcoming)
        assert not sp.playlist_remove_all_occurrences_of_items.called

    def test_no_flush_when_signal_not_clear(self):
        """No flush when delta_clear=False, even if songs are far in vibe space."""
        upcoming = ["BUFFER", "S1", "S2"]
        state    = _make_session_state(skip_cluster=SKIP, comp_vibe=COMP, delta_clear=False)
        sp, _    = self._run_check_refill(upcoming, state=state)
        assert not sp.playlist_remove_all_occurrences_of_items.called

    def test_flush_count_incremented_in_rolling(self):
        upcoming        = ["BUFFER", "S1", "S2"]
        rolling         = _make_rolling(flush_count=0)
        _, rolling_out  = self._run_check_refill(upcoming, rolling=rolling)
        assert rolling_out["flush_count"] == 1

    def test_velocity_frozen_after_max_flushes(self):
        """After _FLUSH_MAX-th flush, velocity_zeroed_until must be a future timestamp."""
        upcoming = ["BUFFER", "S1", "S2"]
        rolling  = _make_rolling(flush_count=_FLUSH_MAX - 1)
        _, r     = self._run_check_refill(upcoming, rolling=rolling)

        assert r["flush_count"] == _FLUSH_MAX
        assert "velocity_zeroed_until" in r
        frozen_dt = datetime.fromisoformat(r["velocity_zeroed_until"])
        assert frozen_dt > datetime.now(timezone.utc), \
            "velocity_zeroed_until must be in the future"
        minutes_ahead = (frozen_dt - datetime.now(timezone.utc)).total_seconds() / 60
        assert minutes_ahead <= _VELOCITY_FREEZE_MINUTES + 1

    def test_no_second_flush_on_same_signal(self):
        """Repeated calls with the same session_state.updated_at must not flush twice."""
        upcoming  = ["BUFFER", "S1", "S2"]
        fixed_ts  = datetime.now(timezone.utc).isoformat()
        state     = self._make_state(updated_at=fixed_ts)
        rolling   = _make_rolling()

        sp1, _ = self._run_check_refill(upcoming, rolling=rolling, state=state)
        assert sp1.playlist_remove_all_occurrences_of_items.called
        # rolling now has last_flush_signal_at = fixed_ts

        sp2, _ = self._run_check_refill(upcoming, rolling=rolling, state=state)
        assert not sp2.playlist_remove_all_occurrences_of_items.called, \
            "Same signal timestamp must not trigger a second flush"
