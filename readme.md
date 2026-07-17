# SmartShuffle

A personalized music queue generator built on top of Spotify and Last.fm. Learns from your actual listening behavior — what you skip, when you listen, which songs you've worn out — and generates ranked queues that fit the moment instead of randomizing it.

---

## The Problem

Spotify shuffle is random. It ignores:
- **Context** — you listen differently at 2am than during a workout
- **Fatigue** — the same songs surface constantly while others never play
- **Skip patterns** — skipping a song four times changes nothing
- **Internal playlist vibes** — a "Chill" playlist might have many different subgenres, shuffle treats them identically

The current shuffle forces the user to consistently optimize playlists to get the best result. If Chill morning, Chill afternoon, and Chill evening have different vibes, the user must manually create separate playlists for each one. We aim to create a new shuffle that uses context to determine the queue.

---

## What's Built

### Data Pipeline (`src/collect.py`)
- Polls Spotify's recently played endpoint every 12 hours via cron
- Infers play duration from gaps between consecutive timestamps (Spotify doesn't expose this directly)
- Classifies play source: `playlist`, `manual_search`, `artist_browse`, `album_browse`, `smartshuffle_queued`, `random_baseline_queued`
- Detects and excludes suspected Jam sessions (≥4 plays, near-zero gaps, ≥3 distinct artists) so group listening doesn't corrupt the personal model
- `INSERT OR IGNORE` on `played_at` — safe to re-run at any frequency
- Full playlist sync on every run: fetches current tracks and deletes removed songs from `playlist_tracks` so queue generation always reflects the current playlist state

### Behavioral Modeling (`src/score.py`)

**Energy Scoring**
Songs are placed on a single axis from −1.0 (ambient/chill) to +1.0 (hype/aggressive) using Last.fm tag data. The axis isn't genre — it's *feel*. Its real job is to split any playlist into subgenre clusters that actually sound different from each other: soft pop vs. hyperpop, lo-fi hip-hop vs. drill, chamber jazz vs. fusion, indie folk vs. folk metal. K-means clustering then groups songs by (energy, genre) so the queue generator can stay within a tonal zone instead of jumping between vibes.

Tags are weighted by vote count against a hand-tuned dictionary of 70+ genre/mood mappings. Two design decisions make this work across genres:
- *Generic tag dampening*: broad umbrella tags (`rap`, `hip-hop`, `pop`, `rock`) accumulate tens of thousands of votes and would drown out every specific subgenre tag beneath them. When more specific tags are also present, generic tags are dampened to 8% of their vote weight — so a soft pop song tagged `dream pop` (60 votes) + `pop` (2000 votes) scores as dream pop, not as generic pop. Same logic applies across every genre.
- *Full vocabulary*: the tag dictionary covers pop subgenres (`synth-pop`, `bedroom pop`, `electropop`), rock variants (`hard rock`, `shoegaze`, `post-punk`), jazz and classical, country, blues, and electronic sub-genres — preventing any song from defaulting to 0.0 just because its genre isn't rap.

**Fatigue Scoring**:
Higher fatigue score decreases likelihood of a song/artist being played to avoid songs/artists being overplayed. Exponential decay with 14-day half-life. Skips count 0.3×, partial listens 0.6×, full plays 1.0×. Computed at both song level and artist level independently so Ridge regression can learn separate coefficients for each.

**Binge Detection**:
Binge score suppresses the fatigue penalty (`effective_fatigue = fatigue × (1 − binge_score)`) so a song you're currently obsessed with doesn't get penalized for high play count. Detection is intentionally strict — false positives are worse UX than false negatives, since overplaying a song someone grew tired of is harder to recover from than underplaying one they're still into (they'll manually correct).

Signal detection:
- Requires `play_count >= 3`
- Requires at least 2 of 3 contextual signals: new release (album < 90 days old), newly added to playlist (< 30 days), or manually sought out
- A classics session or full-album listen hits 0 signals and correctly returns binge_score = 0

Manual play detection uses four sources to catch genuine seek behavior:
1. `artist_browse` or `manual_search` play_source (Spotify context)
2. Session-level null detection: null-context plays in sessions that also contain known-context plays (BART/offline-only sessions excluded)
3. Queue-adjacent detection: non-queue plays sandwiched within 15 minutes between `smartshuffle_queued` plays from a different source — catches songs played at push boundaries
4. `session_interjections` table (explicit interjections logged during queue analysis)

Skip penalty: each skip in the 14-day binge window reduces binge_score by 25% (`binge_score × max(0, 1 − 0.25 × skips)`). Four skips zero the score entirely — a song you're consistently skipping isn't a binge.

Velocity term: `binge_velocity = binge_score_today − binge_score_3days_ago`. If velocity drops below −0.1 (binge clearly declining) OR binge_skip_rate ≥ 0.30, effective_binge is zeroed in the scheduler — stops overplaying a song the moment fatigue sets in. Daily snapshots stored in `binge_score_history` for the 3-day lookback.

**Evergreen Scoring** *(framework implemented; activates automatically at 90 days of play history)*:
Binge detection lifts fatigue during an active listening fixation. Evergreen scoring is the complement: it reduces fatigue suppression for songs with demonstrated long-term preference — songs you reliably return to across multiple distinct listening phases, not just songs with high play counts.

The key distinction is *play distribution over time*, not recency or volume. A song played 9 times spread across 6 months, with gaps followed by returns, is a different signal from one played 9 times in a two-week burst. The binge peak-and-die pattern and the evergreen pattern can produce identical play counts; only the temporal shape separates them.

Two components multiply to form `evergreen_score`:
- **Historical strength** — product of span (days between first and last play, scaled 0→1 over 365 days) and cluster count (distinct play phases separated by 14+ day gaps — 1 cluster = 0, 2 clusters = 0.5, 3+ clusters = 1.0). A song needs both breadth of time *and* evidence of gap-and-return to score highly.
- **Recency confirmation** — exponential decay (30-day half-life) on days since last play. Allows the score to go dormant during periods when other music dominates, then recover quickly when you return to the song.

Applied as: `effective_fatigue = fatigue × (1 − max(effective_binge, min(evergreen_score, 0.60)))`. Binge and evergreen are mutually exclusive — binge_score > 0.3 zeroes evergreen to prevent double-counting. The 0.60 cap means evergreen reduces but never eliminates fatigue suppression. Skip rate revokes the score if queue coverage is sufficient (≥5 weighted plays) and skip_rate > 0.30.

The 90-day gate prevents misfires while the scrape history is short: with only weeks of data, a current binge looks identical to a genuine long-term favorite. Once the window crosses 90 days, scores will turn on automatically without any code change.

**Coverage Debt**
`plays_since_last_heard / playlist_size` — playlist-size-normalized so buried songs in large playlists surface at the same rate as buried songs in small ones.

**Session Segmentation + Engagement Baselines**:
Groups plays into sessions (30-minute inactivity gap), labels each by time bucket and energy level, and aggregates per-bucket skip rate and early-skip rate into `engagement_baselines` for the queue orchestrator. Baselines use exponential time decay (180-day half-life for sparse morning data, 90-day for other buckets) so recent listening patterns carry more weight than sessions from months ago.

Three time buckets: `late_night` (21:00–05:59), `morning` (06:00–10:59), `afternoon` (11:00–20:59). An earlier "evening" bucket (18:00–20:59) was collapsed into afternoon after analysis showed it had identical listening behavior with too few sessions to meaningfully differentiate.

**Skip Rate with Concept Drift**:
Per-song skip rates use exponential decay (90-day half-life) rather than flat means, so the model adapts as your tastes change without needing to wait for a year of new data to dilute old signals.

### Vibe Clustering + Queue Generation (`src/recommend.py`)

**Vibe Axis**:
Songs are placed on three axes — `vibe_content` (melodic vs. rap), `vibe_melodic` (instrumental vs. vocal), `vibe_bpm` (chill vs. hype) — scored by `score_vibes.py` using a combination of audio features and LLM-based inference. The vibe axis replaced Last.fm energy as the primary tonal signal because energy scores were counter-predictive in retrospective analysis (AUC=0.59 for skip prediction — high energy match actually predicted skips, driven by a dead zone where 77% of songs clustered at >0.90 match with no discrimination).

**Clustering (Phase 3)**:
K-means on `(vibe_content, vibe_melodic, vibe_bpm)` per playlist. K auto-selected via silhouette score (capped at 3). Cluster labels derived from vibe_bpm (hype/mid/chill) and vibe_content (rap/mixed/melodic).

**Scoring Formula**:
Every candidate song gets a single score answering: *given where you are in the queue, how good a pick is this song right now?*

```
score = 0.40 × vibe_match                              # 3-axis Gaussian vs playlist vibe target
      - 0.15 × eff_fatigue                             # played too much lately?
      - 0.10 × artist_fatigue × (1 − binge_score)      # artist overexposure?
      + 0.20 × min(coverage_debt / 4, 1.0)             # how overdue is this song?
      - 0.05 × skip_rate                               # how often do you skip this?
      + 0.15 × binge_score                             # active binge boost

# eff_fatigue = fatigue × (1 − max(binge_effect, evergreen_effect)) × (0.25 + 0.75 × skip_rate)
# skip_rate modifier: only suppress songs you play AND skip (skip_mod floors at 0.25)
```

Disabled signals (infrastructure kept, weight=0.0):
- `energy_match`: Last.fm energy axis — replaced by `vibe_match`. Retroactively shown to predict skips (AUC=0.59), not completions.
- `artist_comp`: artist completion rate signal — runs backwards on current data (skip_AUC=0.56); disabled pending more data.

Key design choices:
- *3-axis vibe match*: three independent Gaussians (one per axis, σ calibrated per bucket from play history), averaged. Allows songs to match on some axes but not others, giving finer tonal resolution than a single energy axis.
- *Skip-rate fatigue modifier*: `eff_fatigue = fatigue × (0.25 + 0.75 × skip_rate)`. A song you complete repeatedly (skip_rate ≈ 0) takes only 25% of its raw fatigue hit — so binge songs don't get suppressed just for being played a lot.
- *Binge override*: binge_score both boosts the score (+0.15 × binge) and attenuates fatigue (via eff_fatigue formula). Velocity and skip rate act as circuit breakers.
- *Recency vs. fatigue*: fatigue is long-term (14-day half-life). Recency is a separate 24-hour linear penalty, now folded into fatigue via skip_mod rather than as an explicit term.

**Queue Building (Phase 4)**
Greedy constrained selection over the full playlist:
- Defaults to full-playlist length — generates one pass through every song in the pool
- All slots scored against the full pool; no forced cluster split. The Gaussian energy penalty naturally suppresses off-target songs; coverage debt surfaces the rest
- Hard back-to-back artist block — always enforced
- Proportional per-artist queue caps (`ceil(n × artist_pool_share × 2.2)`) that release when they'd leave only the last-queued artist as an option
- Stakes-aware weight adjustment based on historical engagement for the current time bucket
- Recently-generated-queue exclusion so consecutive runs produce different songs

**A/B Harness**
Every run generates both a SmartShuffle queue and a random baseline queue. The algorithm is chosen per-session (60% SmartShuffle / 40% random baseline) with a rolling-window enforcement: if the last 10 pushes contain < 35% baseline, the next push is forced to baseline regardless of the random draw. Attribution is tracked via `queue_pushes` and `play_source` so skip rates are computed separately per algorithm.

### Skip Inference (`src/collect.py`)

After each listening session, infers which queued songs were skipped by comparing the generated queue order against what actually played. Uses three-layer noise filtering to handle Spotify attribution ambiguity:

1. **Same-timestamp dedup**: two songs can't play simultaneously — keeps the lowest queue position per timestamp
2. **Longest Increasing Subsequence (LIS)**: applies LIS to queue positions ordered by play time to filter out mis-attributed plays from other Spotify contexts (radio, liked songs) that appear out of sequence
3. **Consecutive skip cap (10)**: if the gap between consecutive plays exceeds 10 queue positions, truncates the inferred sequence there — prevents a single mis-attributed play from generating dozens of false skips

Quick skips inferred this way are stored in `queue_skips` and folded into the insights skip rate alongside play-duration-based skip classifications.

### Weight Learning (`src/train.py`)
Ridge regression (L2 regularized) on `[energy_score, fatigue, coverage_debt, skip_rate, binge_score]` → completion ratio. Alpha-blended with hardcoded defaults: formula dominates early, ML takes over as data accumulates (`alpha = max(0.3, 1 − (n_plays − 30) / 500)`). Also learns per-(playlist, time-bucket) energy targets from completion-weighted play history.

**Skip-loss tracking**: every run computes `loss = 0.70 × quick_skip_rate + 0.30 × mid_skip_rate` separately for SmartShuffle and natural Spotify plays. SmartShuffle loss is stored as a weekly array (plus a total) so the trend is visible as the model improves. Weeks with fewer than 5 plays are excluded to avoid noise. Natural Spotify is the control series.

### Streamlit Frontend (`app.py`)
Single "Generate & Play" button that syncs listening data, updates scores, generates a queue, and pushes it to the active Spotify device in one click. Playlist and time-context are selectable. Insights page shows:
- Skip rate per algorithm (SmartShuffle vs. random baseline) with queue_skips folded in
- A/B push history and algorithm distribution
- Per-session engagement history (avg songs/session, session count)

### Session Watcher (`src/watcher.py`)
Background process that polls Spotify every 3 minutes during a listening session. Infers skip status from timestamp gaps (same logic as `collect_data.py`) and writes `engagement_delta` to `session_state.json`. The next queue generation reads this delta to apply real-time stakes adjustment — if the current session is running hotter than baseline, weights shift toward tighter energy matching.

---

## Current State

**Data collected:** 1,611 plays · 497 unique songs with metadata · 14 playlists · 111 queue pushes (59 SmartShuffle / 52 random baseline) · 655 attributed queued plays · 372 inferred queue skips

**A/B results (combined, valid sessions ≥4 plays):**

| Algorithm | Sessions | Plays | Skip rate | Pos-weighted | Avg songs/session |
|-----------|----------|-------|-----------|--------------|-------------------|
| SmartShuffle | 26 | 324 | **37.6%** | **30.6%** | **12.2** |
| Random baseline | 22 | 194 | **42.6%** | **37.2%** | **8.8** |

SS has a 5.0pp lower skip rate, 6.6pp lower position-weighted skip rate, and 38% longer sessions (12.2 vs 8.8 songs). Rolling queues show the largest gap: SS 31.1% vs RB 41.5%. χ²=1.34, p=0.25 — not yet significant; ~3–4× more data needed. See `docs/statistics.md` for full breakdown including rolling vs full mode split, per-feature AUC, and song coverage analysis.

**Retrospective feature AUC** (skip prediction, SS plays excluding morning, n=376 songs):

| Feature | skip_AUC | Notes |
|---------|----------|-------|
| energy_match | 0.634 | Disabled (weight=0) — correctly so; predicted skips, not completions |
| hist_fatigue / eff_fatigue | 0.622 | Best active skip signal; formula penalty working correctly |
| artist_comp_signal | 0.559 | Disabled (weight=0) — runs backwards on current data |
| vibe_match | 0.471 | Best completion signal; validates vibe axis shift from Last.fm energy |
| coverage_debt | 0.361 | Strongest completion signal; correctly given highest positive weight |
| Combined (no artist_comp) | **0.537** | Q2→Q5 monotonic; formula discriminates within moderate-to-high risk range |

AUC > 0.5 = feature predicts skips; AUC < 0.5 = predicts completions. Morning excluded (47.7% skip rate — uncalibrated).

**Song coverage:** SmartShuffle has played 197/764 playlist songs at least once (25.8%) vs random baseline 168/764 (22.0%). Over any 1-week window SS coverage ratio is 94% (unique/total plays) vs RB 90%.

---

## Architecture

```
Spotify API ──► src/collect.py ──► SQLite (plays, songs, playlist_tracks,
Last.fm API ──►       │                    queue_pushes, queue_skips)
                      ▼
               src/score.py ──► song_scores, song_tags, engagement_baselines
                      │
                      ▼
               src/train.py ──► data/learned_params.json
                      │
                      ▼
               src/recommend.py ──► queues table
                      │
              ┌───────┴──────────────────┐
           src/push.py           src/watcher.py
      (Spotify playback)    (real-time skip tracking)
              │
           app.py (Streamlit)
      (Generate & Play + Insights)
```

**Project layout:**
```
app.py              Streamlit frontend
run.py              CLI pipeline runner
src/
  collect.py        Data collection, skip inference, queue attribution
  score.py          Behavioral scoring (fatigue, binge, skip rate, coverage debt)
  recommend.py      Vibe clustering + queue generation
  train.py          ML weight learning (Ridge + alpha blending)
  push.py           Spotify playback control
  watcher.py        Real-time session monitor + engagement delta
vibes/
  score_vibes.py    LLM-based vibe scoring (vibe_content, vibe_melodic, vibe_bpm)
  learn_vibe_params.py  Behavioral target + sigma learning from play history
  evaluate_vibes.py Vibe distance validation against skip/complete pairs
  review_vibes.py   Manual inspection of vibe scores
analysis/
  analyze_rb_predictions.py  Retrospective feature AUC + skip-risk quintiles
  analysis_significance.py   Chi-squared and Mann-Whitney SS vs RB significance tests
  analysis_trend.py          Week-by-week skip rate trend plots
  analysis_energy_weights.py Static vs learned energy weight divergence
  analyze_cooccurrence.py    Session co-occurrence SVD
  fetch_audio_features.py    One-time Spotify audio feature fetch
  recompute_rolling_skips.py Maintenance: re-infer rolling session skips
data/
  smartshuffle.db            SQLite database
  learned_params.json        Ridge weights + vibe targets (written by train.py)
  vibe_params.json           Per-bucket vibe targets + sigmas (written by learn_vibe_params.py)
  rolling_queue_state.json   Rolling session state for the watcher
docs/
  STATUS.md                  Detailed file-by-file breakdown + changelog
  statistics.md              Full statistical report (SS vs RB A/B results, AUC, coverage)
tests/
  test_collect.py · test_score.py · test_recommend.py · test_train.py · test_push.py
logs/
  collect.log · watcher.log · streamlit.log
```

Single command: `python run.py --playlist Hype --play`
Or via Streamlit: `streamlit run app.py`

---

## Key Design Decisions

**Why exponential decay for fatigue?** A song played yesterday should suppress future plays more than a song played last month. Exponential decay with a 14-day half-life captures this without needing an arbitrary cutoff. The binge overlay (7-day half-life) detects current obsession and lifts the penalty while it's active.

**Why exponential decay for skip rates and baselines?** Interests change. A flat mean gives equal weight to a listening session from 8 months ago and one from yesterday. The 90-day half-life for per-song skip rates means a song you used to skip but now enjoy will have its old history diluted within a few months of new data — rather than requiring a full year of replays to override the old signal.

**Why LIS for skip inference?** Spotify's recently-played endpoint attributes any play of a song to whatever context is active at collection time. A song you played from radio that happens to share a track ID with your queue gets mis-attributed to SmartShuffle. LIS on queue positions filters these out: legitimate queue listening produces an increasing position sequence (song 3 plays before song 7); out-of-sequence plays are by definition not from the queue.

**Why Gaussian energy matching instead of linear distance?** Linear distance penalizes songs proportionally regardless of how far they are from target. Gaussian suppresses distant songs much harder — giving the algorithm more useful signal in the tail and preventing songs at the wrong end of the energy spectrum from quietly slipping into queues.

**Why generic tag dampening instead of excluding generic tags?** Excluding `rap` and `hip-hop` entirely breaks scoring for songs that only have generic tags. Dampening to 8% preserves the fallback signal while preventing genre floods from overriding specific subgenre tags like `drill` or `lo-fi`.

**Why strict binge detection?** A false positive (incorrectly suppressing fatigue) causes the same songs to dominate every queue — exactly the problem being solved. False negatives are cheap: the user manually finds the song, and that play feeds back as signal.

---

## Stack

- **Language:** Python 3.12
- **Data:** SQLite, pandas, numpy
- **ML:** scikit-learn (Ridge regression, K-means, silhouette score)
- **APIs:** Spotify (Spotipy), Last.fm
- **Frontend:** Streamlit
- **Testing:** pytest (227 tests)

---

## Setup

```bash
git clone https://github.com/yourusername/smartshuffle
cd smartshuffle
pip install -r requirements.txt

# Add credentials to .env
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
LASTFM_API_KEY=...

# Run full pipeline
python run.py --playlist Hype --play

# Or use the Streamlit app
streamlit run app.py
```

---

## What's Next

- Let data accumulate (targeting 500+ attributed plays for Ridge to learn meaningful weights)
- Watch the weekly skip-loss trend in `model.py` output — when SmartShuffle loss drops below natural Spotify, the model has real signal
- Re-run `model.py` once R² improves and validate fatigue is differentiating songs
- Binge scheduling: use binge_score strength to predict daily play targets, so a high-strength binge (score ~0.9) schedules more plays per day than a weak one (score ~0.3). Velocity gates whether the schedule runs — if score is declining, drop off rather than pushing plays at a song the user is growing tired of
- Binge-specific stats: per-song episode tracking (skip rate while binged, shuffler plays vs manual plays, days to peak) to evaluate whether the model is overplaying or underplaying binged songs
- Song co-play embeddings (word2vec-style): 500+ attributed plays before the co-occurrence matrix is dense enough to be useful
- Mid-session queue re-scoring: after each skip, re-rank remaining queue positions using updated session weights (requires session_watcher to feed back into phase34 in real-time)
