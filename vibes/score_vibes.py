#!/usr/bin/env python3
"""
Score every song on three vibe axes using an LLM.

Adds vibe_content, vibe_melodic, vibe_bpm columns to songs table.
Safe to re-run — skips already-scored songs.

Axes:
  vibe_content : -1.0 philosophical/introspective  →  +1.0 aggressive/street
  vibe_melodic : -1.0 sung/melodic throughout      →  +1.0 pure rap delivery
  vibe_bpm     : -1.0 slow (≤80 BPM)              →  +1.0 fast (≥130 BPM)
"""
import argparse
import json
import os
import sqlite3
import time

from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--song", default=None,
                    help="Rescore a specific song by name (case-insensitive substring). "
                         "Clears existing score and re-runs just that song.")
parser.add_argument("--playlist", default=None,
                    help="Only score unscored songs in this playlist_id. "
                         "When omitted, scores all unscored songs in the library.")
args = parser.parse_args()

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "smartshuffle.db")
BATCH_SIZE = 80   # songs per LLM call

# ── LLM config — swap provider/model here ─────────────────────────────────────
LLM_PROVIDER = "anthropic"   # "anthropic" | "openai"
LLM_MODEL    = "claude-haiku-4-5-20251001"
# For OpenAI swap to e.g. "gpt-4o-mini" and set OPENAI_API_KEY env var
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are scoring songs on three axes for a music recommendation system.
The listener primarily plays hip-hop, rap, and R&B.

AXES — all values from -1.0 to +1.0, step 0.1:

  vibe_content
    -1.0  philosophical / introspective / uplifting
          (J. Cole "Love Yourz", Kendrick "Sing About Me", NF, logic)
     0.0  neutral — everyday life, relationships, flexing without aggression
    +1.0  aggressive / street / violent
          (50 Cent "Many Men", Pop Smoke, drill, UK drill)

  vibe_melodic
    -1.0  highly melodic — sustained singing, sung hooks throughout the song
          (Billie Eilish, The Weeknd full songs, Drake sung-heavy tracks)
     0.0  mixed — rapping with a melodic hook or a sung chorus
    +1.0  pure rap delivery — minimal to no singing, lyrical or street bars
          (classic boom bap, gritty storytelling rap, no hook singing)

  vibe_bpm
    -1.0  very slow  (55–80 BPM: slow trap, R&B ballads, lo-fi hip-hop)
     0.0  mid-tempo  (85–115 BPM: most mainstream rap)
    +1.0  fast       (120–160 BPM: trap bangers, drill, club rap)

Rules:
- Prioritise your knowledge of the actual song over the tags.
- For vibe_bpm, use the song's real tempo if you know it; otherwise estimate
  from artist/production style.
- Reserve ±0.8 and beyond for clear extremes.
- If you genuinely don't recognise the song, use the tags to estimate.
- Return ONLY a raw JSON object — no markdown, no explanation.

Format exactly:
{"SONG_ID": {"vibe_content": 0.0, "vibe_melodic": 0.0, "vibe_bpm": 0.0}, ...}
"""


def call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif LLM_PROVIDER == "openai":
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


def build_prompt(batch: list[dict]) -> str:
    # Format each song as a compact object; include top tags as context
    song_list = []
    for s in batch:
        song_list.append({
            "id":     s["song_id"],
            "name":   s["song_name"],
            "artist": s["artist_name"],
            "tags":   s["tags"][:12],   # top 12 tags by weight
        })
    return _SYSTEM_PROMPT + "\n\nSongs:\n" + json.dumps(song_list, ensure_ascii=False)


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown fences if the LLM adds them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(raw)


def clamp(v) -> float:
    return max(-1.0, min(1.0, round(float(v), 3)))


# ── Main ──────────────────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)

for col in ("vibe_content", "vibe_melodic", "vibe_bpm"):
    try:
        conn.execute(f"ALTER TABLE songs ADD COLUMN {col} REAL")
        conn.commit()
        print(f"Added column: {col}")
    except Exception:
        pass

if args.song:
    # Clear scores for the matched song(s) so they get rescored
    conn.execute(
        "UPDATE songs SET vibe_content=NULL, vibe_melodic=NULL, vibe_bpm=NULL "
        "WHERE LOWER(song_name) LIKE LOWER(?)",
        (f"%{args.song}%",),
    )
    conn.commit()
    cleared = conn.execute(
        "SELECT song_name, artist_name FROM songs "
        "WHERE LOWER(song_name) LIKE LOWER(?) AND vibe_content IS NULL",
        (f"%{args.song}%",),
    ).fetchall()
    print(f"Cleared scores for: {[r[0] for r in cleared]}")

already_scored = conn.execute(
    "SELECT COUNT(*) FROM songs WHERE vibe_content IS NOT NULL"
).fetchone()[0]

if args.playlist:
    rows = conn.execute("""
        SELECT s.song_id, s.song_name, s.artist_name, st.tags
        FROM songs s
        JOIN playlist_tracks pt ON pt.song_id = s.song_id
        LEFT JOIN song_tags st ON st.song_id = s.song_id
        WHERE s.vibe_content IS NULL
          AND pt.playlist_id = ?
        ORDER BY s.song_name
    """, (args.playlist,)).fetchall()
else:
    rows = conn.execute("""
        SELECT s.song_id, s.song_name, s.artist_name, st.tags
        FROM songs s
        LEFT JOIN song_tags st ON st.song_id = s.song_id
        WHERE s.vibe_content IS NULL
        ORDER BY s.song_name
    """).fetchall()

songs = []
for r in rows:
    raw_tags = r[3]
    if raw_tags:
        try:
            tag_objs = json.loads(raw_tags)
            tags = [t["name"] for t in sorted(tag_objs, key=lambda x: -x.get("weight", 0))]
        except Exception:
            tags = []
    else:
        tags = []
    songs.append({
        "song_id":     r[0],
        "song_name":   r[1] or "Unknown",
        "artist_name": r[2] or "Unknown",
        "tags":        tags,
    })

total = len(songs)
print(f"To score: {total}  |  Already scored: {already_scored}")

if not songs:
    print("Nothing to do.")
    conn.close()
    exit(0)

scored, failed = 0, 0

for i in range(0, total, BATCH_SIZE):
    batch = songs[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    prompt = build_prompt(batch)

    try:
        raw = call_llm(prompt)
        result = parse_response(raw)
    except Exception as e:
        print(f"  Batch {batch_num} error: {e}")
        failed += len(batch)
        time.sleep(3)
        continue

    updates = []
    for s in batch:
        sid = s["song_id"]
        sc = result.get(sid)
        if sc is None:
            print(f"  No score returned for: {s['song_name']} ({sid})")
            failed += 1
            continue
        try:
            updates.append((
                clamp(sc["vibe_content"]),
                clamp(sc["vibe_melodic"]),
                clamp(sc["vibe_bpm"]),
                sid,
            ))
            scored += 1
        except (KeyError, TypeError, ValueError) as e:
            print(f"  Bad score for {s['song_name']}: {sc} — {e}")
            failed += 1

    if updates:
        conn.executemany(
            "UPDATE songs SET vibe_content=?, vibe_melodic=?, vibe_bpm=? WHERE song_id=?",
            updates,
        )
        conn.commit()

    done = min(i + BATCH_SIZE, total)
    print(f"  [{done}/{total}]  scored={scored}  failed={failed}")
    if i + BATCH_SIZE < total:
        time.sleep(0.5)

conn.close()
print(f"\nDone.  Scored={scored}  Failed/missing={failed}")

# ── Sanity check ──────────────────────────────────────────────────────────────
conn2 = sqlite3.connect(DB_PATH)
sample = conn2.execute("""
    SELECT song_name, artist_name, vibe_content, vibe_melodic, vibe_bpm
    FROM songs
    WHERE vibe_content IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 12
""").fetchall()
conn2.close()

print(f"\n{'Song':<32} {'Artist':<22} {'content':>8} {'melodic':>8} {'bpm':>6}")
print("-" * 80)
for row in sample:
    print(f"{(row[0] or '?')[:31]:<32} {(row[1] or '?')[:21]:<22}"
          f"  {row[2]:>6.2f}   {row[3]:>6.2f}  {row[4]:>5.2f}")
