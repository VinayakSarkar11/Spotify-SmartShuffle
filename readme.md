# SmartShuffle

A personalized music queue generator built on Spotify. Learns from your actual listening behavior — what you skip, when you listen, which songs you've worn out — and builds queues that fit the moment instead of randomizing it.

---

## The Problem

Spotify shuffle is random. It ignores:
- **Context** — you listen differently at 2am than during a workout
- **Fatigue** — the same songs surface constantly while others never play
- **Skip patterns** — skipping a song four times changes nothing
- **Internal playlist vibes** — a "Chill" playlist might have many different subgenres; shuffle treats them identically

If Chill morning, Chill afternoon, and Chill evening have different vibes, you'd have to manually split them into separate playlists to get good results. SmartShuffle learns the context for you.

---

## Results

A/B test — SS rolling vs. random baseline, sessions with at least 4 plays:

| Algorithm | Sessions | Plays | Skip rate | Avg plays/session |
|-----------|----------|-------|-----------|-------------------|
| SmartShuffle (rolling) | 34 | 682 | **26.4%** | **20.1** |
| Random baseline | 14 | 174 | **45.7%** | **12.4** |

**19.3pp lower skip rate. χ²=41.55, p<0.0001.** SS sessions average 62% more plays. Full breakdown in `docs/statistics.md`.

The meaningful comparison is rolling mode because the system observes each batch of songs before generating the next — it can adapt to live session behavior in a way that generating 180 songs upfront cannot.

**Song coverage:** SmartShuffle surfaces more of your library — 25.8% of playlist songs played at least once vs. 22.0% for random, with a higher unique-to-total play ratio week over week.

---

## How We Know What You Skip

Spotify doesn't expose skip events directly, and it doesn't tell you whether you manually sought out a song or just let it play. SmartShuffle infers both from listening history.

**Skip inference**: After a session, we compare the queue order against what actually played. If the queue had songs at positions 1, 2, 3, 4, 5 and your history shows 1, 2, 4, 5 — position 3 was skipped. This uses **Longest Increasing Subsequence (LIS)** to filter out mis-attributed plays from other Spotify contexts (radio, liked songs) that appear out of order relative to the queue.

**Manual play detection**: If your history shows positions 1, 2, 8, 3, 4, 5 — song 8 was manually selected mid-session (the queue continued after it). These are recorded as interjections: a signal that you wanted something the queue wasn't giving you at that moment.

---

## How It Scores Songs

Each time the queue is built, every candidate song in your playlist gets a score. The highest-scoring songs that fit the current vibe go in.

**Vibe match**: Songs are placed on three axes — how rap vs. melodic they sound, how vocal vs. instrumental, and how chill vs. hype. The target vibe for the moment is learned from which songs you complete vs. skip in each listening context (late night, morning, afternoon). Songs closer to your current target score higher.

The allowed range around the target is asymmetric: the window is wider in the direction your taste has been drifting during the session, and tighter against it. If you've been moving toward more melodic songs, the algorithm gives itself room to follow but resists snapping back.

**Fatigue**: Every play counts against a song, with more recent plays penalized harder. This decays exponentially with a 14-day half-life, so a song you played six months ago has near-zero fatigue. Songs you consistently finish get a softer penalty than ones you sometimes skip — the idea being that high play count on something you love isn't the same problem as high play count on something you're growing tired of.

**Binge**: Sometimes you're obsessed with a song and play it constantly. SmartShuffle detects this and lowers the fatigue penalty while the obsession is active, so the song doesn't get suppressed just for having a high play count. Once the obsession fades — play rate drops off, or you start skipping it — normal fatigue kicks back in.

Binge detection requires a meaningful play count plus at least two of three signals that suggest it's a current fixation rather than a longtime favorite:
- **New release**: if the album just came out, high play count almost certainly means you're into it now — not that it's been a staple for years
- **Newly added to your playlist**: same logic — you just added it, so suddenly playing it a lot means active interest
- **Manually sought out**: you went and found this song rather than letting it play through — active intent, not passive exposure

This matters because a song you've loved for years will often have a high play count too. Binge detection only fires when the novelty signals are present. Songs without those signals — longtime favorites you return to consistently over months — are handled separately by evergreen scoring, which also reduces fatigue suppression but based on the long-term shape of play history rather than a recent spike.

**Coverage**: Songs you haven't heard in a long time get a boost, normalized by playlist size so a buried track in a 400-song playlist surfaces at the same rate as one buried in a 40-song playlist.

**Skip rate**: Songs you consistently skip get a small persistent penalty.

**Learned weights**: Every listener is different. Some people skip songs primarily because of vibe mismatch; others skip based on fatigue. SmartShuffle uses Ridge regression to learn how much each factor — fatigue, vibe match, coverage, skip rate, binge — actually predicts whether you'll complete a song vs. skip it. The weights start from sensible defaults and shift toward your personal patterns as data accumulates.

---

## Rolling Queue + Live Adaptation

The core feature of SmartShuffle is the rolling queue. Rather than generating one big playlist upfront, the system pushes a batch of songs to Spotify and watches what happens. A background process polls Spotify every 15 seconds and automatically appends the next batch before you run out — so playback never stops.

Each new batch is generated using everything observed so far in the session: which songs you completed, which you skipped, and how your completions are shifting the vibe target. If you've been completing more melodic songs than the initial target predicted, the next batch shifts toward that. This is session drift — the queue follows where your listening is actually going, not just where the historical average predicted it would go.

The UI exposes this directly. Three sliders let you drag the vibe target along each axis and hit "Flush & regenerate" to immediately push a new batch tuned to the new position. The vibe percentages shown on each song update to reflect the new target, so you can see at a glance what the queue considers a good match.

**Sigma tightening**: When a session is running hotter (higher skip rate than your baseline), the algorithm narrows the allowed vibe range. Rather than sampling broadly from the playlist, it tightens the window around the target and only selects songs that closely match. This is automatic — no manual tuning needed.

**Velocity**: Between refills, the system computes a velocity vector: how much the effective vibe target has moved since the previous batch. If skips are happening in the direction of drift — meaning the algorithm overshot — the target parks at your recent completions rather than continuing to drift. If skips are perpendicular or behind the direction of movement, drift continues.

---

## Data Pipeline

Spotify's API only returns your last 50 plays. We poll repeatedly during active sessions and run a background sync every 12 hours to capture plays before they fall off that window.

Each play is tagged with how it happened — queue play, manual search, artist browse — so skip rates and engagement stats can be computed separately for SmartShuffle vs. natural Spotify listening, which powers the A/B comparison.

---

## Architecture

```
Spotify API ──► src/collect.py ──► SQLite (plays, songs, playlist_tracks,
                      │                    queue_pushes, queue_skips)
                      ▼
               src/score.py ──► fatigue, binge, skip rate, coverage scores
                      │
                      ▼
               src/recommend.py ──► ranked queue
                      │
              ┌───────┴────────────────┐
           src/push.py           src/watcher.py
      (Spotify playback)    (rolling session refills)
              │
           web/app.py (Flask web app)
```

```
src/
  collect.py        Data collection, skip inference, queue attribution
  score.py          Behavioral scoring (fatigue, binge, skip rate, coverage)
  recommend.py      Vibe clustering + queue generation
  push.py           Spotify playback control + session management
  watcher.py        Rolling queue refills (appends songs as queue drains)
vibes/
  score_vibes.py    LLM-based vibe scoring
  learn_vibe_params.py  Vibe target + calibration from play history
web/
  app.py            Flask web app
data/
  smartshuffle.db   SQLite database
  learned_params.json   Scoring weights + vibe targets
  vibe_params.json  Per-context vibe calibration
docs/
  statistics.md     Full A/B results, coverage analysis
tests/
  test_collect.py · test_score.py · test_recommend.py · test_train.py · test_push.py
```

---

## Stack

- **Language:** Python 3.12
- **Data:** SQLite, pandas, numpy
- **ML:** scikit-learn (Ridge regression, K-means)
- **APIs:** Spotify (Spotipy)
- **Frontend:** Flask + vanilla JS
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

# Run
python web/app.py
```
