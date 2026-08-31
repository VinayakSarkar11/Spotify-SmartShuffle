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

A/B test — SmartShuffle vs. random shuffle, sessions with at least 4 plays:

| Algorithm | Sessions | Plays | Skip rate | Avg songs/session |
|-----------|----------|-------|-----------|-------------------|
| SmartShuffle | 26 | 324 | **37.6%** | **12.2** |
| Random baseline | 22 | 194 | **42.6%** | **8.8** |

5pp lower skip rate, 38% longer sessions (12.2 vs 8.8 songs). Rolling queues show the largest gap: SS 31.1% vs random 41.5%. Not yet statistically significant — about 3–4× more data needed. Full breakdown in `docs/statistics.md`.

**Song coverage:** SmartShuffle surfaces more of your library — 25.8% of playlist songs played at least once vs. 22.0% for random, with a higher unique-to-total play ratio week over week.

---

## How We Know What You Skip

Spotify doesn't expose skip events directly, and it doesn't tell you whether you manually sought out a song or just let it play. SmartShuffle infers both from listening history.

**Skip inference**: After a session, we compare the order songs were generated in the queue against the order they actually played. If the queue had songs at positions 1, 2, 3, 4, 5 and you only played 1, 3, 5 — positions 2 and 4 were skipped.

In practice this is messier: Spotify attributes any play to whatever context is active at collection time. If you open the radio mid-session, those songs appear in your history as if they came from your SmartShuffle queue. To filter them out, we use **Longest Increasing Subsequence (LIS)**: legitimate queue listening produces plays in increasing position order (song 3 before song 7). A song pulled in from another source appears out of sequence — say position 12 pops up between positions 2 and 3 — and gets excluded. This lets us cleanly separate songs you actually heard from the queue vs. ones that ended up in your history for other reasons.

**Manual play detection**: If you exit the queue to go find a specific song, that's a signal the queue wasn't giving you what you wanted. We detect this through Spotify's context metadata (whether a play came from a search, artist browse, or a queue) and through position analysis within sessions.

---

## How It Scores Songs

Each time the queue is built, every candidate song in your playlist gets a score. The highest-scoring songs that fit the current vibe go in.

**Vibe match**: Songs are scored on three axes — how rap vs. melodic they sound, how vocal vs. instrumental, and how chill vs. hype. The current vibe target is learned from which songs you actually complete vs. skip in different listening contexts (late night, morning, afternoon). Songs closer to your target vibe for the moment score higher.

**Fatigue**: Every play counts against a song. The more recently you've played it, the harder it's penalized — this fades out over a couple of weeks, so a song you played six months ago has near-zero fatigue. Songs you consistently finish get a softer penalty than ones you sometimes skip, so the algorithm doesn't over-suppress things you genuinely like.

**Binge**: Sometimes you're obsessed with a song and play it constantly. SmartShuffle detects this and lowers the fatigue penalty while the obsession is active — so the song doesn't get suppressed just for having a high play count. Once the obsession fades (play rate drops off, or you start skipping it), normal fatigue kicks back in.

Binge detection requires a meaningful play count, plus at least two of three signals that suggest it's a current fixation rather than a longtime favorite:
- **New release**: if the album came out recently, high play count is almost certainly because it's new — not because it's been a staple for years
- **Newly added to your playlist**: same logic — you just added it, so suddenly playing it a lot means you're into it right now
- **Manually sought out**: you went and found this song rather than just letting it play, which means active intent rather than passive exposure

A song you've loved for years that happens to have a lot of plays won't trigger binge detection — it lacks the novelty signals, and that's correct behavior.

**Coverage**: Songs you haven't heard in a long time get a boost, normalized by playlist size so a buried track in a 400-song playlist surfaces at the same rate as one buried in a 40-song playlist.

**Skip rate**: Songs you consistently skip get a small persistent penalty.

---

## Data Pipeline

Spotify's API only returns your last 50 plays. We poll repeatedly during active sessions to capture plays before they fall off that window, and run a background sync periodically otherwise. All data is stored locally in SQLite.

Each play is tagged with how it happened — queue play, manual search, artist browse — so skip rates and engagement stats can be computed separately for SmartShuffle vs. natural Spotify listening, which is what powers the A/B comparison.

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
