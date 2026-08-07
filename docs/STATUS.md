# SmartShuffle — Project Status
**Date:** August 7, 2026  
**Data:** 2,504 plays · 825 unique songs with vibe scores · 116 sessions · 14 playlists · 226 queue pushes (146 SmartShuffle / 80 random baseline) · 1,162 attributed queued plays · 568 inferred queue skips

---

## 1. File Breakdown

### `collect_data.py` — Data Collection (Phase 1)
Pulls listening history from Spotify and stores it in SQLite. Runs on a cron job every 12 hours.

- Fetches up to 50 recently played tracks via `user-read-recently-played`
- Infers play duration from gap between consecutive timestamps (since Spotify doesn't expose it directly)
- Classifies play source: `playlist` / `manual_search` / `artist_browse` / `album_browse` / `smartshuffle_queued` / `random_baseline_queued`
- Flags suspected Jam sessions (≥4 plays, median duration <60s, ≥3 distinct artists) as `jam_excluded` so group listening doesn't corrupt your personal model
- Fetches per-song metadata (popularity, explicit, genres, release date) via individual track + artist endpoints — batch endpoints were deprecated by Spotify in Nov 2024
- Full playlist sync on every run: fetches current tracks and DELETEs songs removed from the Spotify playlist so `playlist_tracks` always reflects current state
- **Cold start bootstrap**: if plays table is empty, seeds top tracks (short + medium term) into songs table so phase2 has a richer tag pool before any listening data exists
- `INSERT OR IGNORE` on `played_at` (unique constraint) — safe to re-run at any frequency; duplicate calls where no new songs have been played add 0 rows
- **`_infer_queue_skips()`**: after each push window closes, infers skipped songs by comparing queue order against actual plays. Three-layer noise filtering: (1) same-timestamp dedup (keep lowest queue position per timestamp), (2) LIS on queue positions ordered by play time to filter mis-attributed plays from other Spotify contexts, (3) consecutive skip cap of 10 to prevent a single mis-attributed play from generating dozens of false skips. Writes results to `queue_skips` table.

**Cron:** `0 */12 * * *` (every 12 hours)

---

### `phase2.py` — Behavioral Modeling (Phase 2)
Transforms raw play history into per-song scores and per-session context labels that the queue orchestrator uses.

- **Last.fm tag fetching**: fetches semantic tags (energy, genre, mood) for every played song and every playlist song. Falls back from track-level to artist-level tags. Fetching all playlist songs (not just played ones) means unplayed songs get real energy scores instead of defaulting to 0.0
- **Energy scoring**: continuous -1.0 to +1.0 weighted average across ENERGY_WEIGHTS dict (70+ tags). Genre tags contribute partial signal so a rap song with a melodic hook lands between extremes rather than snapping to ±1
- **Session segmentation**: groups plays into sessions using a 30-minute inactivity gap
- **Session labeling**: each session tagged with time bucket (late_night / morning / afternoon / evening) + energy label (high / mid / low)
- **Fatigue scoring**: exponential decay with 14-day half-life, skip-weighted (skips count 0.3×, partials 0.6×, full plays 1.0×). Uses decay weights so recent plays suppress more than old ones.
- **Skip rate with concept drift**: per-song skip rates computed as decay-weighted averages (90-day half-life) rather than flat means. `w_skip / w_known` where weights are `exp(-λ × days_ago)`. Adapts to changing tastes without requiring a year of new data to dilute old signals.
- **Binge scoring**: 7-day half-life decay, threshold scales with play count so a 10-play song needs proportionally more recent plays to register as a binge. Effective fatigue = `fatigue × (1 − binge_score)` — active binges suppress the fatigue penalty
- **Coverage debt**: `plays_since_last_heard / playlist_size` — playlist-size-normalized so a stale song in a 100-song playlist and a stale song in a 10-song playlist are measured on the same scale
- **Session imputation**: for songs with no Last.fm signal, infers energy from the average energy of sessions they appeared in. Guards against chicken-and-egg by excluding `smartshuffle_queued` plays
- **Engagement baselines**: per session, computes `early_skip_rate` (fraction of skips < 30s), `skip_latency_mean_ms`, and `stakes_level` (high / normal / low). Aggregates by time bucket into `engagement_baselines` table using decay-weighted averages (180-day half-life for sparse morning bucket, 90-day for others) so recent sessions carry more weight.

---

### `phase34.py` / `src/recommend.py` — Vibe Clustering + Queue Orchestration (Phases 3 & 4)
Takes the per-song scores from phase2/score.py and generates an ordered queue.

- **Vibe clustering (Phase 3)**: K-means on **(vibe_content, vibe_melodic, vibe_bpm)** 3D space (replaced energy_score/genre_score). Auto-selects k via silhouette score (capped at k=3). Labels clusters by vibe_bpm (hype/mid/chill) and vibe_content (rap/mixed/melodic).
- **Full-pool scoring (Phase 4)**: all songs scored against the full pool using vibe targets. Removed forced 80/20 primary/secondary cluster split — Gaussian vibe penalty naturally suppresses off-target songs
- **Full-playlist queue**: queue length defaults to the entire playlist (`n=None` → `len(pool)`). Stops cleanly if pool exhausted.
- **Pre-computed score cache**: `_compute_scores()` runs one vectorized pass; `_pick_best` reuses via index lookup
- **Scoring formula** (current):
  ```
  score = 0.40 × vibe_match
        - 0.15 × eff_fatigue          # fatigue × (1 − max(binge, evergreen)) × (0.25 + 0.75×skip_rate)
        - 0.10 × artist_fatigue × (1 − binge)
        + 0.20 × min(coverage_debt/4, 1)
        - 0.05 × skip_rate
        + 0.15 × binge_boost
        # energy_match: weight=0.0 (disabled — vibe axis supersedes it)
        # artist_comp:  weight=0.0 (disabled — signal runs backwards on current data)
  ```
- **Vibe match**: 3-axis Gaussian `mean(exp(-0.5×((song_vibe_axis − target_axis)/σ_axis)²))` across content, melodic, bpm axes. Sigma per axis and bucket from `vibe_params.json`.
- **Within-queue vibe drift**: tracks last 4 picks' vibe axes, blends `effective_vibe` each pick toward recent mean (30% blend weight)
- **Cross-refill vibe drift**: watcher computes mean vibe of completed songs → `session_state.json` → `_apply_session_vibe_drift()` blends into playlist target at next refill
- **Stakes-aware weight adjustment**: loads `engagement_baselines`, adjusts `vibe_match` (not energy_match) based on historical stakes. High stakes → tighter vibe match; low stakes → raise coverage
- **A/B harness**: every run generates both SmartShuffle and random baseline queue. Written to `queues` table.

---

### `src/train.py` — ML Learning Layer
Trains on play completion data (implicit feedback) and writes `learned_params.json`, which `recommend.py` loads at startup.

- **Context target learning**: completion-weighted mean energy per time bucket. Replaces hardcoded defaults when ≥3 plays exist in that bucket
- **Vibe target learning**: per-(playlist, bucket) vibe targets from completion-weighted play history. Written to `learned_params.json` as `playlist_vibe_targets`.
- **Weight learning**: Ridge regression (L2 regularised) on behavioral features → completion_ratio. Falls back to defaults if R² < 0.05 or n < 30 plays. Default weights: `energy_match=0.0, vibe_match=0.40, fatigue=0.15, artist_fatigue=0.10, coverage=0.20, skip=0.05, binge_boost=0.15, artist_comp=0.0`
- **Alpha blending**: `max(0.3, 1.0 - (n_plays - 30) / 500)`. Formula dominates early, ML takes over as data accumulates.
- **Custom skip-loss function**: `loss = 0.70 × quick_skip_rate + 0.30 × mid_skip_rate`. Weekly trend + total per algorithm.
- **Binge threshold calibration**: stubbed — needs songs with 4+ plays

---

### `app.py` — Streamlit Frontend
Single-page app with sidebar playlist/context selector and two tabs: Generate and Insights.

- **Generate & Play button**: single button that runs the full pipeline (collect_data → phase2 → phase34) then pushes to the active Spotify device. Spinner labels each step. Chosen algorithm displayed after push.
- **`_pick_algorithm()`**: rolling-window A/B enforcer. Checks last 10 pushes; if baseline fraction < 35%, forces `random_baseline` regardless of the 60/40 random draw. Prevents runs of all-SmartShuffle.
- **Queue display**: shows all songs in the generated queue with song name, artist, and score metadata.
- **Insights tab**: skip rate per algorithm (folds `queue_skips` into denominator via CTE), A/B push distribution, per-session engagement history. Filters use `IN ('skip', 'partial', 'full')` — not `IS NOT NULL` — to exclude `'unknown'` values from denominators.
- **`load_queue_pair()`**: loads the SmartShuffle queue and finds its companion random baseline queue using a 60-second proximity window on `generated_at` (exact timestamp match was too brittle).

---

### `play.py` — Spotify Playback Control
Pushes a generated queue to the active Spotify device.

- Finds active device; falls back to first available if none active
- Disables Spotify shuffle (`sp.shuffle(False)`) before starting playback so the queue order is preserved
- Records push to `queue_pushes` (queue_id, algorithm, pushed_at) so `collect_data` can attribute plays back to the correct algorithm
- `--algorithm` flag: forces a specific algorithm instead of the default 60/40 draw
- `--list` flag: shows recent queues

---

### `session_watcher.py` — Real-time Session Monitor
Background process that polls Spotify every 3 minutes during an active listening session.

- Infers skip status from timestamp gaps (same logic as `collect_data.py`)
- Computes `engagement_delta = session_skip_rate_so_far - baseline_skip_rate` once ≥3 plays are seen
- Writes delta to `session_state.json` so the next `phase34.py` run can apply real-time stakes adjustment
- Designed to run as a background thread from `run.py --play`; can also be run standalone for testing

---

### `run.py` — Pipeline Runner
Single entry point: `python run.py --playlist Hype --context late_night --play`

---

## 2. Features Added

| Feature | File | Notes |
|---|---|---|
| Continuous energy scoring (-1 to +1) | phase2 | Replaces binary HIGH/LOW tag sets |
| Binge score (7-day decay) with effective fatigue | phase2 | `fatigue × (1 - binge_score)` |
| Session imputation for missing energy | phase2 | Chicken-and-egg guard for queued plays |
| Last.fm tags for ALL playlist songs | phase2 | Fixes energy=0.0 for unplayed songs |
| Early skip rate + skip latency per session | phase2 | Vibe-searching signal |
| Stakes level per session (high/normal/low) | phase2 | From skip rate + early skip clustering |
| `engagement_baselines` table | phase2 | Per-bucket aggregation of session stakes |
| Gaussian energy scoring (σ=0.5) | phase34 | Replaces linear distance |
| Full-pool scoring (removed 80/20 split) | phase34 | Gaussian penalty replaces hard cluster barriers |
| Full-playlist queue default | phase34 | n=None → len(pool); stops cleanly if pool exhausted |
| Pre-computed score cache | phase34 | One vectorized pass; _pick_best does index lookup |
| Custom skip-loss function | model | 0.70×quick + 0.30×mid; weekly trend + total |
| Stakes-aware weight adjustment | phase34 | Loads baselines, shifts weights per context |
| `SessionContext` class | phase34 | Interface ready for real-time adaptation |
| `load_engagement_baselines()` | phase34 | Reads phase2 engagement table |
| Jam session detection | collect | Excludes group listening from model |
| Cold start bootstrap (top tracks) | collect | Seeds songs table before any plays |
| Alpha-blended ML weights | model | Formula → ML transition as data grows |
| Context target learning per bucket | model | Completion-weighted mean energy |
| Cold start context prior from tags | model | Global energy offset when n < 30 |
| Duplicate-safe re-runs | collect | `INSERT OR IGNORE` on unique `played_at` |
| DB migrations for new columns | phase2 | `ALTER TABLE` guards for existing databases |
| Late_night bucket fix (10pm, not midnight) | all | `hour >= 22 or hour < 6` |
| Time bucket split bug fix | phase34 | `late_night` was truncating to `late` |
| Streamlit frontend | app | Generate & Play + Insights in one UI |
| A/B rolling window enforcer | app | Forces baseline if recent fraction < 35% |
| Algorithm string consistency | app/play/collect | `'baseline'` → `'random_baseline'` everywhere |
| Companion queue proximity match | app | 60s window instead of exact timestamp match |
| Queue display anchored to last push | app | Queries queue_pushes JOIN queues, not queues filtered by context |
| Same-day push timestamp fix | collect | REPLACE 'T' ' ' before SQLite datetime() string compare |
| Session-level null detection for binge | score | Null plays in mixed sessions → genuine manual seek |
| Queue-adjacent interjection detection | score | Non-queue plays sandwiched between queued plays within 15 min |
| Other-playlist interjection detection | collect | 15-min gap check instead of context-switch exclusion |
| Binge skip penalty | score | 25% per skip in 14-day window; 4 skips → score = 0 |
| Binge velocity term | score | Δbinge_score over 3 days; < -0.1 suppresses effective_binge |
| Binge skip rate override | recommend | binge_skip_rate ≥ 0.30 suppresses effective_binge |
| Binge skip forgiveness | score | Queued skips within 60 min of non-queued play of same song excluded from binge_skip_counts |
| Evening bucket → afternoon | all | 3 buckets: late_night/morning/afternoon; evening collapsed after showing identical behavior |
| Guessing 8-hour recency gate | app | `load_queue_pair` requires first_pushed_at ≥ now − 8h; stale-session fallback removed |
| Rolling session stitching | push, watcher | session_push_id in rolling_queue_state.json lets refills set correct rolling_session_id |
| Unanalyzed session support in guesser | app | Sessions with analyzed_count=0 (pre-collect delay) immediately available for guessing |
| binge_score_history table | score | Daily snapshots for velocity lookback |
| Full playlist sync on every run | collect | Deletes removed songs from playlist_tracks |
| LIS-based skip inference | collect | Filters mis-attributed plays, 10-skip gap cap |
| Same-timestamp dedup in skip inference | collect | Two songs can't play simultaneously |
| `queue_skips` folded into insights | app | CTE unions queue_skips into skip rate display |
| Exponential time decay for skip rates | phase2 | 90-day half-life; adapts to changing tastes |
| Decay-weighted engagement baselines | phase2 | 180-day morning, 90-day other buckets |
| session_watcher.py | session_watcher | Real-time engagement_delta via polling |
| `IS NOT NULL` → `IN (...)` fix | app | Excludes 'unknown' from skip rate denominators |
| AppleScript playback fallback | push | Bypasses Spotify Web API device lookup when no devices found or stale device ID (404) returns; uses `osascript` to trigger Spotify desktop app directly |
| Multi-session exposure penalty | recommend | Position-weighted exposure score across last 6 sessions (rolling_session_id-grouped); soft (1.4) and hard (2.2) suppression tiers replace flat 50% recently-queued penalty |
| Manual play % metric | app | Weekly % of plays that were session interjections (queued-session plays not in the queue); spike signals SS is over-suppressing songs |
| Session vibe drift acceleration | recommend, watcher | `MIN_CREDIBLE` 5→2, `RAMP_OVER` 23→10, `RECENCY_AT` 30→10, `RECENCY_W` 0.70→0.85 — drift now activates faster and recent plays dominate the signal sooner |
| Skip repulsion in session vibe | watcher | Recent-window mean is pushed away from mean vibe of skipped songs (`_SKIP_REPULSION_W=0.40`) |
| Manual play boost in recent window | watcher | Manual plays in last-10 window get 6× weight (capped at 85%) vs 3× global; ensures the user's steering signal dominates |
| Velocity-based in-session drift | recommend, watcher | Inter-refill velocity vector (`effective_target − prior_baseline`) shapes asymmetric epsilon balls: 2.5σ in drift direction, 1.5σ against; Case 2 (overshoot) parks target at recent completions and zeros velocity |
| Skip signal detection | watcher | `_compute_skip_signal`: ≥2 of last 4 plays are skips AND skip-vibe euclidean distance >0.3 from completion-vibe; writes `skip_cluster_vibe`, `recent_comp_vibe`, `skip_delta_clear` to session state |
| Refill baseline snapshotting | recommend | Post-drift effective vibe written to `refill_baseline.json` at each refill; read at next refill to compute inter-refill velocity without accumulating baseline drift |

---

## 3. Features to Add (Not Yet Built)

### Needs more data (will resolve with time)
- **Evergreen scoring**: framework is fully implemented (`evergreen_score` column, scoring formula, `recommend.py` integration). Activates automatically once the play history reaches 90 days (target: ~September 2026). Signal requires two things: play span across many months AND at least one gap-and-return cycle (14+ day gap between play clusters). With only weeks of data, a current binge and a genuine long-term favorite are indistinguishable by shape. Until then, all `evergreen_score` values are stored as 0.0 and have no effect on queues. Formula: `historical_strength (span × cluster_count) × recency_confirmation (30-day decay)`, capped at 0.60 fatigue lift, zeroed during active binges and revoked if skip rate > 0.30 with sufficient queue coverage.
- **Binge phase detection**: needs songs with 4+ plays in a short window
- **Skip probability model**: needs more skip events across diverse contexts — model currently learns from `queue_skips` (216 rows) plus play-duration skips (14 rows); morning bucket signal is already strong (coef +1.25)
- **Fatigue decay learner**: MLR with exponential features at different λ values — needs decay curve data
- **Per-song binge pattern**: learn whether *this user* binges by quick successive plays vs volume over time
- **Song co-play embeddings** (word2vec-style): 500+ attributed plays before the co-occurrence matrix is dense enough to be useful
- **Interjection score → phase34 boost**: `interjection_score` tracked in `song_scores` but not yet used. Once ≥5 observations exist per song, validate correlation with lower skip rate; if confirmed add `+0.03 × min(interjection_score, 5)` boost (capped at +0.15)
- **Manually played songs as binge candidates**: once ≥70% of plays are queue-attributed, `play_source = 'manual_search'` becomes a reliable intent signal. Not useful now — most plays are still organic.
- **Behavioral cluster re-assignment**: Last.fm tags capture what a song is; skip behavior reveals where it actually belongs in your listening. A drill song consistently completed in chill contexts should have its effective cluster membership shifted toward chill, regardless of energy score. Implementation: for each (song_id, time_bucket) pair with ≥5 plays, compute completion-weighted mean energy and blend with tag-based energy: `effective_energy = (1 - β) × tag_energy + β × behavioral_energy` where β scales up with play count. Gate: requires 5+ plays per song per context — at current pace, ~3-4 months of data. Note: energy is useful for cluster selection (drill vs lo-fi vs pop) but not fine-grained ranking within a cluster — a 0.02 energy difference is tag noise, not a real perceptual difference.

### Buildable now (no data required)
- **Binge scheduling**: use `binge_score` to determine daily play targets per binged song; velocity gates whether the schedule continues. High-strength binge (≥0.8) → ~2 pushes/day; mid (0.5–0.8) → 1 push/day; low (< 0.5) → opportunistic only
- **Binge-specific stats**: per-episode tracking of skip_rate_while_binged, shuffler_plays, manual_plays, days_to_peak, peak_score — retrospective query from existing tables when binge ends
- **Mid-session queue re-scoring**: after each skip, re-rank remaining queue positions using updated session weights (requires session_watcher to trigger phase34 re-run mid-session)
- **Vibe-searching mode**: 2+ consecutive early skips (< 30s) → narrow remaining queue to familiar + tightly energy-matched songs, zero coverage weight
- **Onboarding survey** (`--setup` flag): 3 questions to immediately calibrate cold-start context targets without waiting for data

### Deferred
- **Deployment on Render**: depends on deciding whether the frontend needs a persistent server or stays local

---

## 4. Sample Queue — June 8, 2026, 11:09 PM

**Pipeline run:**
- `collect_data.py`: 10 new plays added (64 total). 0 duplicates — `INSERT OR IGNORE` on unique `played_at` makes re-runs safe at any frequency.
- `phase2.py`: Last.fm tags fetched for 680 previously untagged playlist songs. Energy signal now on 664/733 songs (91%).
- `model.py`: Context targets learned from 61 plays (3 excluded: jam/unknown). Weights defaulted (R² = -0.60, completion too uniform).
- `phase34.py`: late_night context, Hype playlist, k=2 clusters.

**Mood / Context**
- Time bucket: `late_night` (after 10 PM)
- Learned target energy: **+0.374** (vs hardcoded default of -0.50 — you consistently listen to high-energy music at night)
- Engagement baseline: **low stakes** — 0% skip rate across 7 late-night sessions, meaning you passively enjoy whatever plays without searching hard for a specific track
- Stakes adjustment applied: `energy_match` lowered slightly, `coverage` raised (safe context to surface less-heard songs)

**Adjusted weights (low-stakes late_night):**

| Dimension | Default | Adjusted |
|---|---|---|
| energy_match | 0.400 | 0.333 |
| fatigue | 0.350 | 0.333 |
| coverage | 0.200 | 0.286 |
| skip | 0.050 | 0.048 |

**Vibe clusters (Hype playlist, 88 songs):**

| Cluster | Songs | Energy range | Genre |
|---|---|---|---|
| `hype_rap` | 75 | +0.28 to +0.70 | Rap / trap dominant |
| `hype_mixed` | 13 | +0.00 to +0.70 | No strong genre anchor |

Primary cluster: **hype_rap** (centroid closest to +0.374 target)

**SmartShuffle Queue (Queue B):**

| # | Song | Artist | Score | Energy | Fatigue | Binge | Debt | Cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | Body (Remix) | Tion Wayne | +0.3852 | +0.38 | 0.00 | 0.00 | 0.73 | hype_rap |
| 2 | Wid It | Tion Wayne | +0.3852 | +0.38 | 0.00 | 0.00 | 0.73 | hype_rap |
| 3 | Brucky | SR | +0.3852 | +0.38 | 0.00 | 0.00 | 0.73 | hype_rap |
| 4 | Snap It - Remix | SR | +0.3852 | +0.38 | 0.00 | 0.00 | 0.73 | hype_rap |
| 5 | Not Alike (feat. Royce Da 5'9) | Eminem | +0.3833 | +0.32 | 0.00 | 0.00 | 0.73 | hype_mixed |
| 6 | Wants and Needs (feat. Lil Baby) | Drake | +0.3850 | +0.36 | 0.00 | 0.00 | 0.73 | hype_rap |
| 7 | Reggae & Calypso (Remix) | Russ Millions | +0.3849 | +0.40 | 0.00 | 0.00 | 0.73 | hype_rap |
| 8 | Keisha & Becky - Remix | Russ Millions | +0.3849 | +0.40 | 0.00 | 0.00 | 0.73 | hype_rap |
| 9 | Bla Bla (feat. Fivio Foreign) | Lil Tjay | +0.3849 | +0.40 | 0.00 | 0.00 | 0.73 | hype_rap |
| 10 | Zeus (feat. White Gold) | Eminem | +0.3833 | +0.32 | 0.00 | 0.00 | 0.73 | hype_mixed |

**Score interpretation:**
`score = 0.333 × energy_match - 0.333 × (fatigue × (1 - binge)) + 0.286 × min(debt/4, 1) - 0.048 × skip_penalty`

Songs cluster tightly (0.3833–0.3852) because fatigue and binge are both 0 for all unplayed songs — coverage debt (0.73) is the only secondary differentiator alongside the energy Gaussian. Scores will spread further as play history accumulates and fatigue differentiates frequently vs rarely heard songs.

**Energy match explanation:**
Target energy = +0.374. Gaussian σ=0.5: a song exactly at +0.374 scores 1.0 on energy_match; a song at ±0.874 (one full unit away) scores ~0.14. Songs at +0.38 are nearly on-target; songs at +0.32 are slightly further but still in the high-score region.

**Baseline Queue (Queue A — random):** Two Six (J. Cole), Under The Sun (Dreamville), Touchable (Remble), Bad Man (Polo G), Who Want Smoke?? (Nardo Wick), Stick to the Models (Future), The Way I Am (Eminem), girl$ (Dom Dolla), 1F2F (€URO TRA$H), 0 To 100 (Drake)

---

## Changes

**2026-08-07 — Velocity-based in-session drift** (`src/watcher.py`, `src/recommend.py`)
Replaced the fixed session vibe target with a velocity vector system that interprets skip signals relative to the direction of drift and shapes epsilon balls asymmetrically. Key changes:
- `_load_vibe_map()` extracted from `_compute_session_vibe()` and called once per poll cycle; passed to both `_compute_session_vibe` and `_compute_skip_signal`.
- `_compute_skip_signal()`: fires when ≥2 of last 4 plays are skips AND skip-vibe euclidean distance > 0.30 from completion-vibe. Returns `(skip_cluster_vibe, recent_comp_vibe, delta_clear)`. Written to `session_state.json` as `skip_cluster_vibe_{ax}`, `recent_comp_vibe_{ax}`, `skip_delta_clear`.
- `_compute_session_vibe()` returns 5-tuple (added `manual_vibe_raw`) and accepts `vibe_map` to avoid re-querying.
- `_apply_velocity_reasoning()`: reads `refill_baseline.json` (previous refill's post-drift target) and `session_state.json`, computes `velocity = effective_target − prior_baseline`. Interprets skip signal: Case 1 (dot ≤ 0.05, skip behind drift) → keep target, return `velocity_unit` for epsilon shaping; Case 2 (dot > 0.05, overshoot) → park at `recent_comp_vibe`, return `None` (symmetric epsilon).
- `_write_refill_baseline()`: snapshots post-drift target to `refill_baseline.json` after each refill so next refill's velocity reflects inter-refill delta.
- Asymmetric epsilon in queue loop: when `velocity_unit` is set, songs in the drift direction get 2.5σ (`_EPSILON_LOOSE`), songs against it get 1.5σ (`_EPSILON_TIGHT`); falls back to symmetric 2.0σ when no velocity or Case 2 park.

**2026-08-07 — Multi-session exposure penalty** (`src/recommend.py`)
Replaced flat 50% `RECENTLY_QUEUED_PENALTY` with a position-weighted multi-session exposure system. Songs accumulate `exposure_score = sum(max(0, 1 − position/70))` across the last 6 distinct rolling sessions. Scores above 1.4 (soft) receive a 0.50 score penalty; above 2.2 (hard) receive a 2.00 penalty (near-suppression). A 2-session cooldown prevents freshly-penalized songs from immediately re-entering.

**2026-08-07 — AppleScript playback fallback** (`src/push.py`)
Multi-layer fallback for macOS Spotify device detection. When Web API returns no devices, checks `osascript` for a running Spotify desktop instance and sets `use_applescript = True`. Also catches 404 from `start_playback` (stale device ID) and falls through to AppleScript. AppleScript plays the playlist URI directly via `tell application "Spotify" to play track`.

**2026-08-07 — Session vibe drift acceleration + manual play boost** (`src/recommend.py`, `src/watcher.py`)
`MIN_CREDIBLE` 5→2, `RAMP_OVER` 23→10, `RECENCY_AT` 30→10, `RECENCY_W` 0.70→0.85. Manual plays in last-10 window weighted 6× (capped 85%) vs 3× global. Skip repulsion (`_SKIP_REPULSION_W=0.40`) pushes recent window mean away from skipped vibes.

**2026-08-07 — Manual play % metric** (`app.py`)
Added weekly manual play % to Insights tab. Numerator: plays from `session_interjections` (songs played during an active SS/RB session that weren't in the queue). Denominator: total SS + RB queued plays that week. A spike indicates SS is over-suppressing songs the user is seeking out manually. Shown as a line chart + "Last 2 weeks" metric with delta vs prior 4 weeks.

**2026-07-16 — Vibe axis replaces Last.fm energy in scoring and clustering** (`src/recommend.py`, `src/watcher.py`, `src/train.py`)
Removed Last.fm energy from all active pipeline components while preserving infrastructure. Energy scoring, `energy_score` columns, and `behavioral_energy` remain in the DB and `score.py` but are no longer used in queue generation. The vibe axis (`vibe_content`, `vibe_melodic`, `vibe_bpm`) now drives all downstream logic:
- **Clustering**: K-means switched from `(energy_score, genre_score)` to `(vibe_content, vibe_melodic, vibe_bpm)`. Schema migration drops and recreates `song_clusters` table when old columns detected.
- **Queue scoring**: `energy_match` weight hard-overridden to 0.0; `vibe_match` (3-axis Gaussian) added at 0.40.
- **Within-queue drift**: replaced `recent_energies` deque with `recent_vibes` dict tracking all 3 vibe axes. Each pick blends `effective_vibe` 30% toward recent queue mean.
- **Cross-refill drift**: `_compute_session_vibe()` replaces `_compute_session_energy()` in watcher; writes `session_vibe_{axis}_mean` and `session_vibe_n` to `session_state.json`. `_apply_session_vibe_drift()` blends into playlist target at next refill when n ≥ 3 songs.
- **Stakes adjustment**: `adjusted_weights()` now moves `vibe_match` weight (not `energy_match`) for high/low stakes.

**2026-07-16 — artist_comp_rate feature added and immediately disabled** (`src/score.py`, `src/recommend.py`)
Added `compute_artist_completion_rates()` to `score.py`: per-artist completion rate computed from non-dominant songs only (dominant = >40% of artist plays is one song). Confidence scales with unique songs heard (max at 5+ unique songs). Stored in `song_scores.artist_comp_rate` and `.artist_comp_conf`. Integrated into `recommend.py` scoring formula as `artist_comp_signal = (eff_comp - 0.5) × 2.0`. **Disabled at weight=0.0** after retrospective analysis (skip_AUC=0.559 — signal runs backwards, predicting skips not completions on current data). Infrastructure stays in DB for future validation.

**2026-07-16 — energy_match disabled in formula** (`src/recommend.py`, `src/train.py`)
`energy_match` weight hard-overridden to 0.0 in both `recommend.py` and `train.py` defaults. Reason: retrospective AUC analysis showed `energy_match` is counter-predictive (skip_AUC=0.593–0.634), driven by a dead zone where 76.8% of songs cluster above 0.90 match — no discriminating power. Option B (zero weight, reallocate to vibe_match). Infrastructure kept for potential Option C (cross-calibrate behavioral energy via regression for all songs).

**2026-07-13 — Binge skip forgiveness** (`src/score.py`)
Queued skips where the same song was played non-queued within 60 minutes prior are now excluded from `binge_skip_counts`. Rationale: if you manually played a song and SmartShuffle surfaced it again a few positions later, skipping it is not evidence of binge fatigue — you literally just heard it. The forgiveness check (`_is_forgiven_skip`) compares each queued skip against the non-queued subset of the binge window: if a non-queued play of the same song falls within the 60-minute window before the skip's `played_at`, the skip is dropped before `binge_skip_counts` is accumulated.

**2026-07-13 — Guessing system overhaul** (`app.py`, `src/push.py`, `src/watcher.py`)
Three fixes to the "Guess what I heard" queue display: (1) **8-hour recency gate** — `load_queue_pair` now requires `first_pushed_at >= datetime('now', '-8 hours')`; sessions older than 8 hours no longer appear. Stale-session fallback removed entirely. (2) **Rolling session stitching** — `push.py` now writes `session_push_id` into `rolling_queue_state.json` when a rolling session starts; `watcher.py` `_check_refill` reads it and sets `rolling_session_id` on all refill pushes, so the full stitched session displays as one queue instead of isolated 10-song batches. (3) **Unanalyzed session support** — `load_queue_pair` previously required `analyzed_at IS NOT NULL`, blocking sessions that `collect.py` hadn't processed yet (2-hour analysis delay). Fixed by tracking `analyzed_count`; sessions where `analyzed_count = 0` (fully unanalyzed) are included alongside sessions with `analyzed_plays > 0`.

**2026-07-13 — Evening bucket elimination** (`src/score.py`, `src/collect.py`, `src/recommend.py`, `app.py`)
Collapsed the `evening` time bucket (18:00–20:59) into `afternoon` (11:00–20:59). Analysis showed evening and afternoon had identical skip rates and listening behavior with too few evening sessions to meaningfully differentiate. Updated all six `get_time_bucket` implementations across the codebase. DB migrated: deleted duplicate afternoon `engagement_baselines` rows, replaced evening rows with afternoon. Final three buckets: `late_night` (≥21:00 or <06:00), `morning` (06:00–10:59), `afternoon` (11:00–20:59).

**2026-07-13 — Evergreen score framework** (`src/score.py`, `src/recommend.py`)
Added `evergreen_score` to `song_scores` (schema migration included). Measures long-term consistent play preference — a song you reliably return to across multiple listening phases — to prevent suppressing genuine favorites that have accumulated fatigue purely from being loved over a long period. Two-component formula: `historical_strength (span_component × cluster_component) × recency_confirmation (30-day decay)`. Play span scales 0→1 over 365 days; cluster count is distinct play phases separated by 14+ day gaps (1 cluster=0, 2=0.5, 3+=1.0). Integrated into both the row-level `_score()` and the vectorized `_compute_scores()` in recommend.py as `fatigue × (1 − max(binge_effect, min(evergreen, 0.60)))`. Zeroed during active binges (binge_score > 0.3, no double-counting) and revoked by skip rate (w_known ≥ 5 and skip_rate > 0.30). Gated behind `EVERGREEN_SCRAPE_MIN_DAYS = 90`: all scores are 0.0 until the total scrape window hits 90 days, at which point the feature activates automatically.

**2026-07-08 — Binge velocity term + skip rate override** (`src/score.py`, `src/recommend.py`)
Added `binge_velocity` and `binge_skip_rate` to the binge scoring pipeline. Each day, binge scores are snapshotted to a new `binge_score_history` table (keyed by `song_id, date`). At score time, velocity is computed as `binge_score_today − binge_score_3days_ago`. In the scheduler, if velocity < −0.1 (binge clearly declining) or `binge_skip_rate >= 0.30` (song being skipped during binge window), `effective_binge` is zeroed so fatigue re-engages immediately — stops overplaying a song the moment obsession fades. Both fields written to `song_scores` via schema migration + `save_to_db`.

**2026-07-08 — Binge skip penalty** (`src/score.py`)
Each skip in the 14-day binge window reduces binge_score by 25%: `binge_score × max(0, 1 − 0.25 × binge_window_skips)`. Four or more skips zero the score entirely. A separately tracked `binge_play_counts` dict computes `binge_skip_rate` for the velocity override.

**2026-07-08 — Expanded manual play detection for binge** (`src/score.py`)
Binge `manual_songs` detection extended from `artist_browse` alone to four sources: (1) `artist_browse` / `manual_search` play source, (2) session-level null detection (`_session_manual_songs` — null-context plays in sessions with ≥1 known-context play, excluding all-null offline sessions), (3) queue-adjacent detection (`_queue_adjacent_songs` — non-queue plays sandwiched within 15 min between `smartshuffle_queued` plays from a different source, catching push-boundary interjections), (4) `session_interjections` table. This unblocked several songs from qualifying: DKTP (3 session-manual plays across June) and Take You Dancing (push-boundary interjection on Jul 8).

**2026-07-08 — Other-playlist interjection detection in queue analysis** (`src/collect.py`)
`analyze_queue_session` previously skipped any play from a non-push-playlist context as a "context switch." Replaced the exclusion with a 15-min gap check: a play from any non-queue source is logged as an interjection if it falls within 15 min of both the preceding and following queued play. Catches songs played manually mid-queue that were being filtered out as context switches.

**2026-07-08 — Same-day push timestamp fix** (`src/collect.py`)
`attribute_plays_to_queues` was filtering `WHERE pushed_at < datetime('now', '-2 hours')`. `pushed_at` is stored as ISO format (`2026-07-08T06:11...`) while SQLite's `datetime()` returns space-separated format (`2026-07-08 17:23`). Because `T` (0x54) > ` ` (0x20), same-day pushes always compared as "in the future" and were never analyzed. Fix: `REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' ') < datetime('now', '-2 hours')`.

**2026-07-08 — Queue display anchored to last pushed queue** (`app.py`)
`load_queue_pair` was querying `queues` filtered by current context sidebar selection — if a queue was pushed at "afternoon" but viewed at "evening," the UI showed a different historical queue than what was actually on Spotify. Fixed by querying via `queue_pushes JOIN queues ORDER BY push_id DESC LIMIT 1`, anchoring display to the most recently pushed queue regardless of context.

**2026-06-15 — Streamlit frontend** (`app.py`)
Built Streamlit UI with sidebar playlist/context selector and two tabs. Generate & Play button runs the full pipeline (collect → phase2 → phase34) then pushes to the active Spotify device in a single click. Insights tab shows skip rate per algorithm, A/B push distribution, and session history. Replaced the previous two-button flow (Generate Queue + separate Play on Spotify).

**2026-06-15 — Exponential time decay for preference drift** (`phase2.py`)
Old skip rates used flat means, giving equal weight to a high-skip session from 6 months ago and one from yesterday. Added exponential decay (90-day half-life) to `compute_fatigue_scores()` skip rate computation: `w_skip / w_known` where weights are `exp(-λ × days_ago)`. Engagement baselines in `compute_baseline_engagement()` now use decay-weighted averages (180-day half-life for sparse morning bucket, 90-day for others) for skip rates, early skip rates, and modal stakes.

**2026-06-15 — Playlist sync fix** (`collect_data.py`)
`collect_playlists()` previously used `INSERT OR IGNORE` and was only called when `playlist_count == 0`, so songs removed from a Spotify playlist stayed in `playlist_tracks` indefinitely and continued to appear in generated queues. Fixed: (1) always call `collect_playlists()` on every run, (2) after each playlist fetch, `DELETE FROM playlist_tracks WHERE playlist_id = ? AND song_id NOT IN (...)` to remove songs the user has removed from Spotify.

**2026-06-15 — LIS-based queue skip inference** (`collect_data.py`)
Rewrote `_infer_queue_skips()` with three-layer noise filtering. Root cause: Spotify's recently-played endpoint attributes any play of a song to whatever context (radio, liked songs, etc.) is active, so songs playing from other contexts that share a track ID with a SmartShuffle queue were inflating `last_pos` — one song attributed to queue position 132 (18 minutes into a session) caused 129 false skips. Fix: (1) same-timestamp dedup (two songs can't play simultaneously; keep lowest queue position per timestamp), (2) LIS on queue positions ordered by play time filters out-of-sequence mis-attributed plays, (3) consecutive skip cap of 10 truncates the sequence if gap between consecutive plays exceeds 10 positions. Reduced inferred skip rate from 83% → 44%.

**2026-06-15 — A/B split rolling window enforcer** (`app.py`)
Added `_pick_algorithm()` that checks the last 10 `queue_pushes` rows; if baseline fraction < 35%, forces `random_baseline` regardless of the 60/40 random draw. Prevents runs of all-SmartShuffle caused by random variance.

**2026-06-15 — Algorithm string consistency fix** (`app.py`, `play.py`, `collect_data.py`)
`load_queue_pair()` was querying `algorithm = 'baseline'` but phase34 saves `'random_baseline'`, causing `base_row` to always be None and all pushes to fall back to the SmartShuffle queue ID. Fixed all mismatches. Companion queue lookup changed from exact `generated_at` match to 60-second proximity window (JULIANDAY difference < 60s) to handle sub-second timing differences between SmartShuffle and baseline queue generation.

**2026-06-15 — `queue_skips` folded into insights** (`app.py`)
`ab_stats` CTE now UNIONs `queue_skips` into the skip rate display so inferred quick skips (songs in the queue that never appeared in recently-played) are counted alongside play-duration-based skip classifications. Denominator filter changed from `IS NOT NULL` to `IN ('skip', 'partial', 'full')` to exclude the string `'unknown'` which was diluting skip rate denominators.

**2026-06-08 — `late_night` time bucket split fix** (`phase34.py`)
`context.split("_")[0]` on `"late_night"` extracted `"late"`, which matched no key in `TIME_BUCKET_ENERGY` and silently fell back to `target_energy = 0.0`, ignoring the learned +0.374 target. Fixed by matching the full bucket name. Songs with energy~+0.38 now rank first instead of songs near 0.0.

**2026-06-08 — Artist fatigue + binge as independent Ridge features** (`phase2.py`, `phase34.py`, `model.py`)
Song-level fatigue/binge captures how tired you are of a specific track, but not how saturated you are with an artist overall. Added `artist_fatigue` and `artist_binge_score` columns to `song_scores`, computed by summing decay-weighted plays across all songs by an artist. Exposed as separate features in Ridge regression so they get independent learned coefficients from song-level fatigue.

**2026-06-08 — Graduated artist penalty + hard back-to-back block** (`phase34.py`)
Queues were clustering the same artist back-to-back. Added a graduated penalty across a 4-song window (most-recent predecessor gets strongest penalty, older positions fade toward 1.0), plus a hard block (score → −∞) on the immediate predecessor unless the artist's binge score exceeds 0.60. Binge exception removes all penalties so active binges aren't interrupted.

**2026-06-08 — Proportional per-queue artist cap** (`phase34.py`)
Without a cap, artists with many songs in a playlist could occupy 40–50% of a queue. Cap is `ceil(queue_length × artist_pool_share × 2.2)` — proportional to the artist's actual representation in the pool, with a 2.2× headroom so smaller queues and playlists scale naturally. Binge exception bypasses the cap.

**2026-06-08 — Per-playlist energy targets** (`model.py`, `phase34.py`)
A single per-bucket energy target (e.g. one "late_night" target) averaged across all playlists obscured the fact that hype playlists at night should target higher energy than chill playlists at night. `learn_context_targets` now returns per-(playlist, bucket) targets using only songs belonging to exactly one playlist to avoid ambiguity. Queue generation prefers the playlist-specific target and falls back to the global bucket target.

**2026-06-09 — Generic tag dampening for energy scoring** (`phase2.py`)
Ultra-broad tags like `hip-hop`, `rap`, `r&b` appear on virtually every song and, because they accumulate thousands of Last.fm votes, were swamping specific subgenre tags. A drill song tagged `drill` (50 votes) + `rap` (1000 votes) was scoring ~0.31 — same as a generic rap song. Added `GENERIC_TAGS` set with a `0.08` vote multiplier applied only when specific subgenre tags are also present, so `drill` / `lo-fi` / `afrobeats` etc. control the score. When only generic tags exist the multiplier is not applied so those songs still get a meaningful score rather than falling to 0.0.

**2026-06-10 — Expanded energy vocabulary** (`phase2.py`)
~50 songs were scoring exactly 0.0 because their genre tags (rock, classical, country, folk, etc.) had no entry in `ENERGY_WEIGHTS`. Added anchors for the rock family (`rock` 0.55, `hard rock` 0.70, `classic rock` 0.40, `soft rock` −0.10, etc.), classical (`classical` −0.55, `piano` −0.35, `baroque` −0.25, `romantic` −0.45), country (`country` 0.10), blues, electronic sub-genres (`house` 0.60, `tech house` 0.65), and regional drill variants (`florida drill`, `chicago drill`, `brooklyn drill`, `ny drill`, `detroit drill` — all 0.70). Also added `melodic rap` −0.10 and moved `trap` to 0.15 (intentionally below generic rap at ~0.31, because trap is atmospheric and melodic rather than high-energy).

**2026-06-10 — Fatigue now includes smartshuffle_queued plays** (`phase2.py`)
`compute_fatigue_scores` and `compute_artist_fatigue_scores` were filtering to `play_source IN ('playlist', 'artist_browse')`, silently excluding all plays sourced from generated queues. Songs played from SmartShuffle queues were accumulating zero fatigue and winning every subsequent queue. Fixed by filtering only `jam_excluded`; all real plays now contribute.

**2026-06-10 — Back-to-back artist fix** (`phase34.py`)
Two root causes identified and fixed: (1) `artist_binge_score >= 0.60` was bypassing the back-to-back filter entirely — binge now only reduces fatigue penalty, never allows consecutive picks. (2) Artist caps were stranding picks: the guard that released caps when the pool was exhausted only fired when `avail_uncapped` was completely empty, but since the last-queued artist's songs are always uncapped, it was never empty — caps stayed applied, all other artists got excluded, and the filter fell back to the last artist. Fixed by releasing caps whenever they'd leave only the last-queued artist as an option. Simplified `_pick_best` to use direct set-subtraction filtering instead of score-to-−∞ with an `others_available` flag.

**2026-06-10 — Queue variety across consecutive generations** (`phase34.py`)
Generating a new queue when the previous hadn't been played always produced identical songs because deterministic scoring on zero-fatigue songs yields the same ranking every time. Fixed by seeding `used` with song IDs from the most recently generated queue at the start of each generation, provided the pool has enough alternatives (`unqueued_count >= n`). Songs from the prior queue are excluded, not penalised — they don't count as skips.

**2026-06-10 — Short-term recency penalty** (`phase34.py`)
Added a 24-hour linear decay penalty (`recency` weight = 0.30) in `_score` to suppress songs played very recently even before the longer-term 14-day fatigue decay takes hold. A song played 1 hour ago takes a −0.29 hit; at 12 hours −0.15; at 24+ hours the penalty is zero. Precomputed as `hours_since_played` column on the pool dataframe.

**2026-06-10 — Binge detection overhauled** (`phase2.py`)
Previous formula (`max(play_count * 0.6, 2.0)` denominator) triggered binge from just 2 plays in a week — too easy for established artists with many songs. Replaced with a multi-signal gate: binge_score is non-zero only when `play_count >= 3` AND at least 2 of 3 contextual signals are present: new release (album < 90 days old), newly added to playlist (< 30 days), manually sought out (`manual_search` or `artist_browse` play). `new_discovery` signal removed from both per-song and per-artist because it fires for any old library song you happen to play for the first time — that's normal rotation, not discovery.

**2026-06-10 — Re-scored all 735 songs with updated energy formula** (`song_tags`, `song_clusters`)
Re-ran `compute_energy_score` against the stored Last.fm tags for all songs and updated both `song_tags.energy_score` and `song_clusters.energy_score` in-place. No re-fetching from Last.fm required — tags were already stored. Key outcomes: drill songs moved from ~0.31 to 0.47–0.70; pure trap songs settled at 0.15–0.20; classical settled at −0.42 to −0.49; rock landed at 0.19–0.55 depending on subgenre.

**2026-06-10 — Removed 80/20 cluster split** (`phase34.py`)
All queue slots now use the full-pool Gaussian scoring. The secondary-cluster 20% interleave was designed to add variety, but since SmartShuffle can't be used 100% of the time — Spotify's own shuffle fills the gaps — the forced 20% random picks were just noise. The Gaussian energy penalty already handles variety implicitly: songs slightly off-target score lower but aren't excluded, so the occasional different-vibe pick still surfaces naturally via coverage debt.

**2026-06-10 — Full-playlist queue default** (`phase34.py`)
Queue length changed from a fixed 10 to the full playlist. `n=None` (default) resolves to `len(pool)` after the pool is built. If the pool is exhausted before `n` picks (e.g. an explicit `--n` larger than the playlist), generation stops cleanly. The `used` seeding guard (`unqueued_count >= n`) continues to prevent replay of the immediately prior queue.

**2026-06-10 — Custom skip-loss function** (`model.py`)
Added `compute_weekly_loss(conn)` which computes a skip-weighted loss score per source per week: `loss = 0.70 × quick_skip_rate + 0.30 × mid_skip_rate`. Quick skips (completion < 30%) are penalized more heavily — they signal immediate rejection rather than partial interest. SmartShuffle loss is stored as a weekly array so the trend is visible as the model improves. Natural Spotify plays are the control series. `main()` now prints the last 4 weeks of loss and the week-over-week delta.

**2026-06-10 — Runtime optimizations across all files**
Eliminated all hot-path Python loops over rows, replacing them with vectorized numpy/pandas operations:
- `segment_sessions`: `iterrows()` → `diff().cumsum()` (one call)
- `compute_fatigue_scores` / `compute_artist_fatigue_scores`: inner `group.iterrows()` → `np.select` + `np.exp` + `groupby.agg` — O(total\_plays) Python calls → O(total\_plays) vectorized + O(unique\_songs) loop
- `compute_coverage_debt`: `iterrows()` → `groupby("song_id")["_pos"].max().to_dict()`
- `impute_missing_energy`: per-song DataFrame scan → `.isin()` + `.map()` + `groupby.mean()` + `executemany`
- `_build_daily_plays`: Python dict loop → `value_counts()` Series lookup
- `save_to_db` / `recompute_energy_scores`: N+1 `execute` calls → `executemany`
- `learn_energy_weights`: `iterrows()` → `explode()` + `groupby.agg`
- DB indexes added: `plays(play_source)`, `plays(played_at)`, `plays(song_id)`, `binge_episodes(end_date)`, `queues(algorithm, queue_id)`
- 226/226 tests pass.
