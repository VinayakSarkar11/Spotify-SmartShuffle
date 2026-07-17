#!/usr/bin/env python3
"""
Fetch Spotify audio features for every song in the songs table.

Adds a `spotify_audio_features` column (JSON blob) to songs if not already present.
Skips songs that already have features. Safe to re-run.

Fields stored per song:
  energy, valence, danceability, acousticness, instrumentalness,
  speechiness, liveness, tempo, loudness, key, mode, time_signature
"""
import json
import os
import sqlite3
import time

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(_ROOT, "data", "smartshuffle.db")
BATCH_SIZE = 100
SLEEP_S    = 0.2   # between batches — Spotify rate limit is generous

KEEP_FIELDS = {
    "energy", "valence", "danceability", "acousticness",
    "instrumentalness", "speechiness", "liveness",
    "tempo", "loudness", "key", "mode", "time_signature",
}

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
    scope="user-read-recently-played",
    cache_path=os.path.join(src_dir, ".spotify_cache"),
))

conn = sqlite3.connect(DB_PATH)

# Add column if missing
try:
    conn.execute("ALTER TABLE songs ADD COLUMN spotify_audio_features TEXT")
    conn.commit()
    print("Added spotify_audio_features column to songs.")
except Exception:
    pass  # already exists

# Songs still needing features
rows = conn.execute(
    "SELECT song_id FROM songs WHERE spotify_audio_features IS NULL"
).fetchall()
song_ids = [r[0] for r in rows]
total = len(song_ids)
print(f"Songs needing features: {total}  (already fetched: "
      f"{conn.execute('SELECT COUNT(*) FROM songs WHERE spotify_audio_features IS NOT NULL').fetchone()[0]})")

if not song_ids:
    print("Nothing to do.")
    conn.close()
    exit(0)

fetched, failed = 0, 0

for i in range(0, total, BATCH_SIZE):
    batch = song_ids[i : i + BATCH_SIZE]
    try:
        results = sp.audio_features(batch)   # list of dicts or None per track
    except Exception as e:
        print(f"  API error on batch {i//BATCH_SIZE + 1}: {e}")
        failed += len(batch)
        time.sleep(2)
        continue

    updates = []
    for song_id, feat in zip(batch, results or []):
        if feat is None:
            # Track not found in Spotify's audio analysis (rare)
            updates.append(("{}", song_id))
            failed += 1
        else:
            slim = {k: feat[k] for k in KEEP_FIELDS if k in feat}
            updates.append((json.dumps(slim), song_id))
            fetched += 1

    conn.executemany(
        "UPDATE songs SET spotify_audio_features = ? WHERE song_id = ?",
        updates,
    )
    conn.commit()

    done = min(i + BATCH_SIZE, total)
    print(f"  [{done}/{total}]  batch ok  (fetched={fetched}, failed={failed})")
    if i + BATCH_SIZE < total:
        time.sleep(SLEEP_S)

conn.close()
print(f"\nDone. Fetched={fetched}  failed/missing={failed}")

# ── Quick sanity check ────────────────────────────────────────────────────────
conn2 = sqlite3.connect(DB_PATH)
sample = conn2.execute(
    "SELECT s.song_name, s.artist_name, s.spotify_audio_features "
    "FROM songs s WHERE s.spotify_audio_features IS NOT NULL AND s.spotify_audio_features != '{}' "
    "ORDER BY RANDOM() LIMIT 6"
).fetchall()
conn2.close()

print("\nSample (random 6):")
print(f"  {'Song':<35} {'Artist':<22} {'energy':>7} {'valence':>8} {'dance':>7} {'acoust':>7}")
print("  " + "-"*85)
for name, artist, feat_json in sample:
    f = json.loads(feat_json)
    print(f"  {(name or '?')[:34]:<35} {(artist or '?')[:21]:<22}"
          f"  {f.get('energy', '?'):>6.3f}  {f.get('valence', '?'):>7.3f}"
          f"  {f.get('danceability', '?'):>6.3f}  {f.get('acousticness', '?'):>6.3f}")
