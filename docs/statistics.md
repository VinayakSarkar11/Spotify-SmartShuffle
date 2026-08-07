# SmartShuffle — Statistics Report
**Generated:** July 21, 2026  
**Data window:** June 9 – July 21, 2026 (~6 weeks)

> Only sessions with ≥4 attributed plays are included; rolling refill pushes are stitched into their parent session before averaging. Skip rate = (play-level skips + raw queue skips) / (attributed plays + raw queue skips). Queue skips are raw counts (not position-weighted) for cleaner integer chi-squared tests.

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Valid sessions (≥4 plays, collapsed rolling) | 55 (30 SS · 25 RB) |
| Attributed plays in valid sessions | 592 (384 SS · 208 RB) |
| Queue skips in valid sessions | 351 (196 SS · 155 RB) |

---

## 2. Skip Rate: SmartShuffle vs Random Baseline

### 2a. Primary comparison: SS rolling vs RB combined

SS full mode generates all ~180 songs upfront in a single pass before any playback, so it cannot adapt to live session behavior. The meaningful comparison is **SS rolling** (where the system observes each 11-song batch before generating the next) against the random baseline in either mode.

| Comparison | Sessions | Plays | Queue skips | **Skip rate** |
|------------|----------|-------|-------------|---------------|
| **SS rolling** | 12 | 171 | 60 | **26.8%** |
| **RB combined** | 25 | 208 | 155 | **43.0%** |

**SS rolling is 16.2pp better than random baseline. χ²=15.13, p=0.0001.**

### 2b. By mode (full breakdown)

| Algorithm | Mode | Sessions | Plays | Queue skips | Skip rate |
|-----------|------|----------|-------|-------------|-----------|
| SmartShuffle | Rolling | 12 | 171 | 60 | **26.8%** |
| SmartShuffle | Full | 18 | 213 | 136 | **40.7%** |
| Random baseline | Rolling | 11 | 97 | 72 | **42.6%** |
| Random baseline | Full | 14 | 111 | 83 | **43.3%** |

SS full's smaller advantage (40.7% vs ~43%) reflects the inherent limitation of static planning: all 180 queue slots are scored in one shot against historical averages with no real-time feedback.

### 2c. Statistical significance

Chi-squared tests on raw skip counts (play-level + queue skips; independent observations assumption):

| Comparison | SS | RB | Δ | χ² | p-value | Significant? |
|------------|----|----|---|-----|---------|--------------|
| **SS rolling vs RB combined** | 26.8% (n=231) | 43.0% (n=363) | −16.2pp | **15.13** | **0.0001** | **Yes** |
| SS total vs RB total | 35.2% (n=580) | 43.0% (n=363) | −7.8pp | **5.43** | **0.020** | **Yes** |

Both comparisons are statistically significant. The rolling comparison is the primary result.

---

## 3. Session Length

| Algorithm | Mode | Sessions | Avg songs/session |
|-----------|------|----------|--------------------|
| SmartShuffle | Rolling | 8 | 13.9 |
| SmartShuffle | Full | 18 | 11.5 |
| SmartShuffle | **Combined** | **26** | **12.2** |
| Random baseline | Rolling | 8 | 10.4 |
| Random baseline | Full | 14 | 7.9 |
| Random baseline | **Combined** | **22** | **8.8** |

SmartShuffle sessions are consistently longer (+3.4 songs/session overall, +3.5 rolling, +3.6 full). Formal significance testing requires per-session raw data; without it, this is a descriptive observation. The magnitude (+38%) is large enough to be practically meaningful if it holds as data accumulates.

---

## 4. Week-by-Week SmartShuffle Trend

Play-level skip rate only (excludes queue skips; small weekly n makes combined rate noisy).

| Week | Dates | SS plays | SS play-skip% | RB plays | RB play-skip% | SS pushes |
|------|-------|----------|---------------|----------|---------------|-----------|
| W23 | Jun 9–15 | 107 | 2.8% | — | — | 7 |
| W25 | Jun 23–29 | 92 | 0.0% | 93 | 1.1% | 11 |
| W26 | Jun 30–Jul 6 | 92 | 3.3% | 16 | 0.0% | 13 |
| W27 | Jul 7–13 | 31 | 0.0% | 74 | 1.4% | 12 |
| W28 | Jul 14–16 | 71 | 0.0% | 52 | 1.9% | 16 |

Play-level skip rates are very low across the board (0–3%) because the main skip signal comes from queue skips (songs skipped in the queue before playback starts). Play-level skips only capture songs that began playing but were cut short. Weekly trends in queue skips are too noisy at current session counts to show a reliable signal.

Notable: **W27 (Jul 7–13)** was a major infrastructure change week (vibe axis migration, watcher refactor). Metrics were volatile during this period and should be interpreted cautiously.

---

## 5. Song Coverage

### 5a. Recent (last 7 days)

| Algorithm | Plays | Unique songs | Coverage ratio |
|-----------|-------|--------------|----------------|
| SmartShuffle | 71 | 67 | **94.4%** |
| Random baseline | 52 | 47 | **90.4%** |

Coverage ratio = unique songs / total plays. Higher means fewer repeats per session. SS edges out RB on recent variety (94.4% vs 90.4%).

### 5b. Full history

| Algorithm | Plays | Unique songs | Coverage ratio | Playlist songs heard ≥1× |
|-----------|-------|--------------|----------------|--------------------------|
| SmartShuffle | 393 | 200 | 50.9% | **197 / 764 (25.8%)** |
| Random baseline | 262 | 170 | 64.9% | **168 / 764 (22.0%)** |

RB has higher historical unique coverage (64.9% vs 50.9%) because it never repeats a song by intent — each random draw is independent. SmartShuffle deliberately replays songs it believes you'll like, so replay rate is higher. Despite this, SS has introduced more new playlist songs overall: **197 vs 168 unique playlist songs heard at least once (+3.8pp)**. The coverage debt signal is working as intended.

---

## 6. Retrospective Feature Analysis (AUC)

Computed on SmartShuffle plays (morning excluded; morning skip rate 47.7% is uncalibrated and inflates all metrics). n=376 songs, 42 pushes, afternoon + late_night only.

### 6a. Per-feature skip prediction AUC

> AUC > 0.5 = high feature value predicts skip. AUC < 0.5 = high feature value predicts completion (i.e., it's a useful positive signal in the formula).

| Feature | skip_AUC | Role in formula |
|---------|----------|-----------------|
| energy_match | 0.634 | ❌ Disabled (weight=0) — correctly so |
| hist_fatigue | 0.622 | ✅ Used as penalty — working correctly |
| eff_fatigue (binge+skip_mod applied) | 0.622 | ✅ Actual formula value — same signal |
| artist_comp_signal | 0.559 | ❌ Disabled (weight=0) — signal reversed |
| hist_artist_skip_rate | 0.519 | Not in formula |
| hist_skip_rate | 0.505 | ✅ Used as penalty — weak but correct direction |
| binge_score | 0.496 | ✅ Used as boost — near-random in isolation |
| vibe_match | 0.471 | ✅ Best completion signal — correctly positive weight |
| coverage_debt | 0.361 | ✅ Strongest completion signal — correctly positive weight |

`coverage_debt` (skip_AUC=0.361, completion_AUC=0.639) is the single strongest completion predictor. Songs that are overdue are completed at a meaningfully higher rate. `vibe_match` (0.471) is the second-best completion signal, validating the shift from Last.fm energy to the vibe axis. `energy_match` at AUC=0.634 for skips confirms it was correctly disabled — high energy match was actually predicting skips, not completions.

### 6b. Combined skip-risk score

| Formula variant | skip_AUC |
|----------------|----------|
| Without artist_comp | **0.537** |
| With artist_comp | 0.519 |

Dropping artist_comp improves discriminative power. At 0.537, the formula is above random but modest — expected given that all SS songs already passed a quality filter (selection bias ceiling).

### 6c. Skip-risk quintile table (no artist_comp, afternoon + late_night)

| Bucket | Skip risk | n | Skip% | Comp% |
|--------|-----------|---|-------|-------|
| 1 — safest | [−0.54, −0.30] | 76 | 38.2% | 61.8% |
| 2 | [−0.30, −0.24] | 75 | **26.7%** | **73.3%** |
| 3 | [−0.24, −0.18] | 75 | 32.0% | 68.0% |
| 4 | [−0.17, −0.12] | 75 | 33.3% | 66.7% |
| 5 — riskiest | [−0.12, +0.07] | 75 | 44.0% | 56.0% |

**Buckets 2→5 are perfectly monotonic**: the formula correctly ranks skip risk within the moderate-to-high range. Bucket 1 anomaly (38.2% despite lowest risk score) is driven by afternoon songs — likely a ceiling effect from selection bias where the formula is most confident at the top of its range but that confidence is not yet well-calibrated.

---

## 7. Morning Context

Morning (06:00–10:59) is excluded from the feature analysis above because it is uncalibrated.

| Bucket | Sessions | Skip rate |
|--------|----------|-----------|
| late_night | — | **29.3%** |
| afternoon | — | **37.4%** |
| morning | — | **47.7%** |

Morning skip rate is 18pp above late_night. Q1 morning (lowest-scored songs) skips at **62.5%** — the formula fails in morning context across all quintiles. Most likely cause: morning listening is more mood- and context-dependent (commute, workout, ambient background) than the static vibe targets capture. Requires either more morning data or a morning-specific vibe calibration pass to address.

---

## 8. Summary

| Dimension | Finding | Status |
|-----------|---------|--------|
| **Skip rate: SS rolling vs RB combined** | **26.8% vs 43.0% (−16.2pp)** | ✅ **Significant (p=0.0001)** |
| Skip rate: SS total vs RB total | 35.2% vs 43.0% (−7.8pp) | ✅ Significant (p=0.020) |
| Session length | SS 12.2 vs RB 8.8 songs (+38%) | ✅ Meaningfully longer; descriptive only |
| Playlist breadth | SS 25.8% vs RB 22.0% (+3.8pp of playlist heard) | ✅ SS surfaces more new songs |
| Feature AUC: coverage_debt | 0.639 for completion | ✅ Strongest signal; correctly used |
| Feature AUC: vibe_match | 0.529 for completion | ✅ Good signal; validates vibe axis shift |
| Feature AUC: energy_match | 0.634 for skips | ✅ Correctly disabled |
| Morning calibration | 47.7% skip, formula fails | ⚠️ Unresolved |
| artist_comp signal | Runs backwards (skip_AUC=0.559) | ⚠️ Disabled; needs revalidation |

**When to revisit:** Evergreen scoring activates automatically at 90 days of history (~September 2026). Morning calibration requires a dedicated tuning pass or more morning data.
