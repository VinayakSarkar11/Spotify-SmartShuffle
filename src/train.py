"""
ML Learning Layer
=================
Trains on play completion data (implicit feedback) to replace handcrafted
parameters with values learned from actual listening behaviour.

What gets learned:
  context_targets  — what energy level you actually listen to per time-of-day,
                     instead of the assumed TIME_BUCKET_ENERGY values
  weights          — how much each scoring dimension (energy match, fatigue,
                     coverage, skip) actually predicts completion, instead of
                     the handcrafted 0.40/0.35/0.20/0.05 split

Blend factor (alpha):
  At low data volume, formula dominates (alpha ≈ 1.0).
  At high data volume, learned weights dominate (alpha → 0.3).
  phase34.py interpolates: final_weights = alpha * defaults + (1-alpha) * learned

When to retrain:
  Run after every ~50 new plays, or after collecting A/B ratings.
  Will warn when data is below the minimum useful threshold.

Run:  python model.py
Out:  learned_params.json (loaded by phase34.py automatically)
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(_ROOT, "data", "smartshuffle.db")

from datetime import timezone as _tz
LOCAL_TZ = datetime.now(_tz.utc).astimezone().tzinfo
PARAMS_PATH = os.path.join(_ROOT, "data", "learned_params.json")

MIN_PLAYS           = 30   # below this, fall back to formula defaults entirely
MIN_BINGE_EPISODES  = 5    # completed episodes needed before shape learning kicks in
BINGE_SHAPE_LENGTH  = 14   # days — standard arc length for normalization

# ── Loss function ─────────────────────────────────────────────────────────────
QUICK_SKIP_THRESHOLD    = 0.30   # completion ratio below this is a quick skip
MID_SKIP_THRESHOLD      = 0.80   # below this (but ≥ quick) is a mid skip; ≥ this is full
QUICK_SKIP_WEIGHT       = 0.70   # quick skips penalised more — strong rejection signal
MID_SKIP_WEIGHT         = 0.30   # mid skips — mild rejection
MIN_LOSS_PLAYS_PER_WEEK = 5      # weeks below this are excluded from the trend

# ── Defaults (mirror phase34.py) ──────────────────────────────────────────────

DEFAULT_CONTEXT_TARGETS = {
    "late_night": -0.5,
    "morning":     0.4,
    "afternoon":   0.2,
}

DEFAULT_WEIGHTS = {
    "energy_match":   0.00,  # disabled — replaced by vibe_match
    "vibe_match":     0.40,
    "fatigue":        0.25,
    "artist_fatigue": 0.10,
    "coverage":       0.20,
    "skip":           0.05,
    "binge_boost":    0.15,
    "artist_comp":    0.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_bucket(hour: int) -> str:
    if hour >= 21 or hour < 6: return "late_night"
    if hour < 11: return "morning"
    return "afternoon"

# ── Training data ─────────────────────────────────────────────────────────────

def load_training_data(conn) -> pd.DataFrame:
    """
    Builds a per-play feature matrix joined across plays + song_scores + song_tags.

    Features:
      energy_score        — from Last.fm tags (stable)
      fatigue/skip_rate   — current values; approximation that improves with more data
      prior_play_count    — how many times this song had been played before this play
      days_since_last_play— days since the previous play of this song (null → 365)
      time_bucket         — morning / afternoon / late_night

    Targets:
      completion_ratio    — for context target learning
      is_skip             — binary (1 = inferred_skip == 'skip') for logistic regression

    Includes smartshuffle_queued and random_baseline_queued plays — these are the
    primary training signal for the skip model.
    """
    df = pd.read_sql_query("""
        SELECT
            p.played_at,
            p.song_id,
            p.duration_ms,
            p.play_duration_ms,
            p.inferred_skip,
            p.play_source,
            COALESCE(st.energy_score,        0.0) AS energy_score,
            COALESCE(ss.fatigue,             0.0) AS fatigue,
            COALESCE(ss.binge_score,         0.0) AS binge_score,
            COALESCE(ss.artist_fatigue,      0.0) AS artist_fatigue,
            COALESCE(ss.artist_binge_score,  0.0) AS artist_binge_score,
            COALESCE(ss.skip_rate,           0.0) AS skip_rate,
            COALESCE(ss.coverage_debt,       0.0) AS coverage_debt,
            ROW_NUMBER() OVER (PARTITION BY p.song_id ORDER BY p.played_at) - 1
                AS prior_play_count,
            (JULIANDAY(p.played_at) - JULIANDAY(
                LAG(p.played_at) OVER (PARTITION BY p.song_id ORDER BY p.played_at)
            )) AS days_since_last_play
        FROM plays p
        LEFT JOIN song_tags   st ON p.song_id = st.song_id
        LEFT JOIN song_scores ss ON p.song_id = ss.song_id
        WHERE p.play_source NOT IN ('jam_excluded')
          AND p.play_duration_ms IS NOT NULL
          AND p.duration_ms > 0
    """, conn)

    df["played_at"]            = pd.to_datetime(df["played_at"], format="ISO8601", utc=True)
    df["hour_of_day"]          = df["played_at"].dt.tz_convert(LOCAL_TZ).dt.hour
    df["time_bucket"]          = df["hour_of_day"].apply(_time_bucket)
    df["completion_ratio"]     = (df["play_duration_ms"] / df["duration_ms"]).clip(0.0, 1.0)
    df["days_since_last_play"] = df["days_since_last_play"].fillna(365.0)
    df["from_queue_skips"]     = False

    # Union in queue_skips as synthetic skip rows.
    # These songs were in the queue but never appeared in recently-played — hard skips.
    # This is where the majority of skip signal lives (216 rows vs 14 in plays table).
    qs = pd.read_sql_query("""
        SELECT
            qp.pushed_at                                                        AS played_at,
            qs.song_id,
            qs.queue_position,
            COALESCE(st.energy_score,       0.0)                               AS energy_score,
            COALESCE(ss.fatigue,            0.0)                               AS fatigue,
            COALESCE(ss.binge_score,        0.0)                               AS binge_score,
            COALESCE(ss.artist_fatigue,     0.0)                               AS artist_fatigue,
            COALESCE(ss.artist_binge_score, 0.0)                               AS artist_binge_score,
            COALESCE(ss.skip_rate,          0.0)                               AS skip_rate,
            COALESCE(ss.coverage_debt,      0.0)                               AS coverage_debt,
            (SELECT COUNT(*) FROM plays p2
             WHERE p2.song_id = qs.song_id
               AND p2.played_at < qp.pushed_at)                                AS prior_play_count,
            (SELECT JULIANDAY(qp.pushed_at) - JULIANDAY(MAX(p3.played_at))
             FROM plays p3
             WHERE p3.song_id = qs.song_id
               AND p3.played_at < qp.pushed_at)                                AS days_since_last_play
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
        LEFT JOIN song_tags   st ON qs.song_id = st.song_id
        LEFT JOIN song_scores ss ON qs.song_id = ss.song_id
    """, conn)

    if not qs.empty:
        qs["played_at"]            = pd.to_datetime(qs["played_at"], format="ISO8601", utc=True)
        qs["hour_of_day"]          = qs["played_at"].dt.tz_convert(LOCAL_TZ).dt.hour
        qs["time_bucket"]          = qs["hour_of_day"].apply(_time_bucket)
        qs["inferred_skip"]        = "skip"
        qs["completion_ratio"]     = 0.0
        qs["play_duration_ms"]     = 0
        qs["duration_ms"]          = 1
        qs["play_source"]          = "queue_skip"
        qs["days_since_last_play"] = qs["days_since_last_play"].fillna(365.0)
        qs["from_queue_skips"]     = True
        df = pd.concat([df, qs], ignore_index=True)

    return df

# ── Context target learning ───────────────────────────────────────────────────

def learn_context_targets(df: pd.DataFrame, conn) -> tuple:
    """
    Learns what energy level you gravitate toward per time bucket, weighted by
    play completion.  Also learns per-(playlist, time_bucket) targets so that
    a Hype queue at night gets a different energy target than a Chill queue at night.

    Per-playlist learning uses only songs that belong to exactly one playlist to
    avoid ambiguity when a song appears in multiple playlists.

    Returns (bucket_targets: dict, playlist_targets: dict)
      playlist_targets: {playlist_id: {time_bucket: energy}}
    Falls back to the hardcrafted default for any cell with < 3 plays.
    """
    targets = dict(DEFAULT_CONTEXT_TARGETS)

    for bucket, group in df.groupby("time_bucket"):
        if len(group) < 3:
            continue
        w       = group["completion_ratio"].clip(0.01)
        learned = float(np.average(group["energy_score"], weights=w))
        default = DEFAULT_CONTEXT_TARGETS.get(bucket, 0.0)
        targets[bucket] = round(learned, 4)
        direction = "↑" if learned > default else "↓"
        print(f"    {bucket:12s}  default {default:+.2f}  →  learned {learned:+.3f} {direction}"
              f"  (n={len(group)})")

    # Per-playlist: map each song to its playlist (only unambiguous single-playlist songs)
    rows = conn.execute("""
        SELECT song_id, playlist_id
        FROM playlist_tracks
        GROUP BY song_id
        HAVING COUNT(DISTINCT playlist_id) = 1
    """).fetchall()
    song_playlist = {song_id: playlist_id for song_id, playlist_id in rows}

    df2               = df.copy()
    df2["playlist_id"] = df2["song_id"].map(song_playlist)

    playlist_targets: dict = {}
    pl_df = df2.dropna(subset=["playlist_id"])
    if not pl_df.empty:
        print("  Per-playlist targets:")
        for (playlist_id, bucket), group in pl_df.groupby(["playlist_id", "time_bucket"]):
            if len(group) < 3:
                continue
            w       = group["completion_ratio"].clip(0.01)
            learned = float(np.average(group["energy_score"], weights=w))
            playlist_targets.setdefault(playlist_id, {})[bucket] = round(learned, 4)
            print(f"    {playlist_id[:24]:24s}  {bucket:12s}  {learned:+.3f}  (n={len(group)})")

    return targets, playlist_targets


def learn_playlist_vibe_targets(conn) -> dict:
    """
    Learns completion-weighted mean vibe scores per playlist.

    Returns {playlist_id: {content, melodic, bpm}}.
    Songs with no play history get weight 0.5 (neutral prior).
    Playlists with fewer than 3 scored songs are skipped.
    """
    rows = conn.execute("""
        WITH song_completion AS (
            SELECT
                song_id,
                AVG(CAST(play_duration_ms AS REAL) / duration_ms) AS completion
            FROM plays
            WHERE play_duration_ms IS NOT NULL AND duration_ms > 0
            GROUP BY song_id
        )
        SELECT
            pt.playlist_id,
            s.vibe_content,
            s.vibe_melodic,
            s.vibe_bpm,
            COALESCE(sc.completion, 0.5) AS w
        FROM playlist_tracks pt
        JOIN songs s ON s.song_id = pt.song_id
        LEFT JOIN song_completion sc ON sc.song_id = s.song_id
        WHERE s.vibe_content IS NOT NULL
    """).fetchall()

    from collections import defaultdict as _defaultdict
    data: dict = _defaultdict(list)
    for pid, vc, vm, vb, w in rows:
        data[pid].append((vc, vm, vb, max(float(w), 0.01)))

    pl_names = dict(conn.execute(
        "SELECT playlist_id, playlist_name FROM playlists"
    ).fetchall())

    targets: dict = {}
    for pid, entries in data.items():
        if len(entries) < 3:
            continue
        total_w = sum(e[3] for e in entries)
        targets[pid] = {
            "content": round(sum(e[0] * e[3] for e in entries) / total_w, 4),
            "melodic": round(sum(e[1] * e[3] for e in entries) / total_w, 4),
            "bpm":     round(sum(e[2] * e[3] for e in entries) / total_w, 4),
        }
        name = (pl_names.get(pid) or pid)[:28]
        t = targets[pid]
        print(f"    {name:<28}  c={t['content']:+.3f}  m={t['melodic']:+.3f}  "
              f"b={t['bpm']:+.3f}  (n={len(entries)})")

    return targets

# ── Ranking weight learning ───────────────────────────────────────────────────

def learn_weights(df: pd.DataFrame, context_targets: dict) -> dict:
    """
    Fits logistic regression: song + context features → is_skip (binary).

    Logistic regression suits this better than Ridge on completion_ratio because:
      — completion_ratio clusters near 1.0 (low variance target); logistic uses
        a binary skip signal which has much better class separation
      — The log-odds interpretation maps cleanly to the scoring formula:
        features with large positive coefficients increase skip probability;
        those with negative coefficients decrease it

    Features:
      energy_match        — Gaussian similarity to context energy target (key new feature)
      fatigue/skip_rate   — per-song penalty signals
      prior_play_count    — familiarity; first listens are more likely to be skipped
      days_since_last_play— recency; recently heard songs may be skipped sooner
      time_bucket (1-hot) — captures context-specific skip patterns

    Coefficient → weight mapping:
      Positive coef on X → X increases P(skip) → X is a penalty signal
      Negative coef on X → X decreases P(skip) → X is a boost signal
      energy_match expected negative; fatigue/skip_rate expected positive.

    ROC-AUC is the right metric here (not R²) — measures ranking quality
    for a binary outcome regardless of class imbalance.
    """
    # Only train on plays with a known skip label
    skip_df = df[df["inferred_skip"].isin(["skip", "partial", "full"])].copy()
    skip_df["is_skip"] = (skip_df["inferred_skip"] == "skip").astype(int)

    n_skips = skip_df["is_skip"].sum()
    print(f"    {len(skip_df)} labeled plays  ({n_skips} skips / {len(skip_df) - n_skips} non-skips)")

    if n_skips < 20 or (len(skip_df) - n_skips) < 20:
        print("    Too few skips or non-skips for logistic regression. Using defaults.")
        return DEFAULT_WEIGHTS

    # Compute energy_match: Gaussian similarity to the context energy target.
    # This is the key signal — was the song's energy close to what we targeted?
    skip_df["target_energy"] = skip_df["time_bucket"].map(context_targets).fillna(0.0)
    skip_df["energy_match"]  = np.exp(
        -0.5 * ((skip_df["energy_score"] - skip_df["target_energy"]) / 0.5) ** 2
    )

    # Time bucket one-hots (drop late_night as reference category)
    for b in ["morning", "afternoon"]:
        skip_df[f"bucket_{b}"] = (skip_df["time_bucket"] == b).astype(float)

    # Features: only include features not confounded by queue selection bias.
    # fatigue and coverage_debt are ALREADY used to filter songs into queues, so
    # the training data shows low-fatigue, high-debt songs being skipped at higher
    # rates than their true baseline — selection bias inverts the learned relationship.
    # energy_match is excluded for low within-playlist variance (songs in the same
    # playlist cluster in a narrow energy range, giving the model nothing to learn from).
    # These three stay at handcrafted defaults; only unconfounded features are learned.
    FEATURES = [
        "artist_fatigue",
        "skip_rate",
        "days_since_last_play",
        "bucket_morning", "bucket_afternoon",
    ]

    # Expected signs: positive coef = increases P(skip).
    EXPECTED_SIGN = {
        "artist_fatigue": +1,
        "skip_rate":      +1,
    }

    X = skip_df[FEATURES].fillna(0).values
    y = skip_df["is_skip"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_folds  = min(5, len(X) // 10)

    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 20), cv=n_folds,
        max_iter=1000, scoring="roc_auc",
    )
    model.fit(X_scaled, y)

    cv_auc = cross_val_score(model, X_scaled, y, cv=n_folds, scoring="roc_auc").mean()
    coefs  = dict(zip(FEATURES, model.coef_[0]))

    print(f"    Logistic C={model.C_[0]:.4f}   CV ROC-AUC={cv_auc:.3f}")
    for feat, c in coefs.items():
        print(f"      {feat:<25} coef={c:+.4f}")

    if cv_auc < 0.55:
        print("    AUC too low — model is near-random. Using defaults.")
        return DEFAULT_WEIGHTS

    # Sanity check: reject the model if any feature with a known expected sign
    # has a coefficient pointing the wrong direction with meaningful magnitude.
    wrong = [
        feat for feat, sign in EXPECTED_SIGN.items()
        if sign is not None
        and feat in coefs
        and np.sign(coefs[feat]) != sign
        and abs(coefs[feat]) > 0.10
    ]
    if wrong:
        print(f"    Sanity check failed — wrong sign on: {wrong}. Using defaults.")
        return DEFAULT_WEIGHTS

    # Map learned coefficients to scoring weights.
    # fatigue, coverage, energy_match stay at handcrafted defaults — their
    # relationships cannot be learned cleanly from queue-selected training data.
    raw = {
        "energy_match":   DEFAULT_WEIGHTS["energy_match"],
        "fatigue":        DEFAULT_WEIGHTS["fatigue"],
        "coverage":       DEFAULT_WEIGHTS["coverage"],
        "artist_fatigue": max(coefs["artist_fatigue"], 0.01),
        "skip":           max(coefs["skip_rate"],      0.01),
    }
    total   = sum(raw.values())
    weights = {k: round(v / total, 4) for k, v in raw.items()}
    print(f"    Learned weights: {weights}")
    return weights

# ── Cold-start context target seeding ────────────────────────────────────────

def seed_context_targets_from_tags(conn) -> dict:
    """
    New-user prior: when play data is too sparse to learn context targets,
    use the energy distribution across all tagged songs (played + playlist)
    to estimate the user's overall energy preference, then shift the hardcoded
    defaults proportionally.

    Example: user's songs average +0.4 energy → all context targets shift up
    by +0.2 (50% dampening to avoid overcorrecting on a small sample).

    Why 50% dampening: the playlist tag distribution is a weak signal. Not all
    songs in a playlist reflect what a user actively chooses to listen to —
    playlists include things added speculatively, gifts, collaborative additions.
    Cutting the offset in half preserves the shape of the default curve while
    nudging it toward the user's apparent taste.

    Falls back to DEFAULT_CONTEXT_TARGETS if fewer than 10 tagged songs exist.
    """
    rows = conn.execute(
        "SELECT energy_score FROM song_tags WHERE energy_score != 0.0"
    ).fetchall()

    if len(rows) < 10:
        print(f"    Only {len(rows)} tagged songs — cold-start prior unavailable, "
              "using hardcoded defaults.")
        return dict(DEFAULT_CONTEXT_TARGETS)

    energies      = [r[0] for r in rows]
    global_energy = float(np.mean(energies))
    offset        = global_energy * 0.5   # 50% dampening

    targets = {
        bucket: round(min(max(target + offset, -1.0), 1.0), 4)
        for bucket, target in DEFAULT_CONTEXT_TARGETS.items()
    }
    print(f"    Global energy preference: {global_energy:+.3f} "
          f"across {len(rows)} tagged songs")
    print(f"    Cold-start offset: {offset:+.3f} applied to all context targets")
    for bucket in DEFAULT_CONTEXT_TARGETS:
        print(f"      {bucket:12s}  "
              f"default {DEFAULT_CONTEXT_TARGETS[bucket]:+.2f}  →  "
              f"seeded  {targets[bucket]:+.3f}")
    return targets


# ── Per-song behavioral energy learning ──────────────────────────────────────

# Energy demand scale: structurally defined, independent of tag-derived energies.
# Using the broad theoretical defaults (not the narrow learned averages) so the
# resulting per-song energy estimates span the full ±1 range and the tag
# regression has real variance to fit.
_CONTEXT_DEMAND = {"late_night": -0.5, "morning": +0.4, "afternoon": +0.2}
_STAKES_MODIFIER = {"high": +0.15, "normal": 0.0, "low": -0.15}


def _play_energy_demand(time_bucket: str, stakes: str) -> float:
    """Energy demand implied by the context a play occurred in."""
    return float(np.clip(
        _CONTEXT_DEMAND.get(time_bucket, 0.0) + _STAKES_MODIFIER.get(stakes or "normal", 0.0),
        -1.0, 1.0,
    ))


def _bayesian_blend(behavioral: float, tag: float, n: int, k: int = 10) -> float:
    """Credibility-weighted blend: as n grows relative to k, behavioral dominates.

    k=10  → 50/50 at 10 direct plays, 75/25 at 30 plays, ~25/75 at 3 plays.
    k=5   → for per-context buckets (fewer plays needed to be credible).
    k=15  → for artist priors (less credible than direct observation).
    """
    w = n / (n + k)
    return round(w * behavioral + (1 - w) * tag, 4)


def learn_song_energies(
    conn, min_plays: int = 3, min_bucket_plays: int = 2
) -> tuple[dict, dict]:
    """
    Learns per-song energy scores from behavioral signals (completion × context demand).

    Energy demand is structural — time-of-day bucket + stakes level — so it is
    independent of tag-based energies and breaks the circular dependency.

    Each raw score is Bayesian-blended with the tag-based energy_score so that
    songs with thin play data shrink toward the tag prior rather than taking
    extreme values from 2–3 plays in the same context.

    Returns:
      global_energies:  {song_id: blended_energy}
        Computed for songs with >= min_plays total plays.
      context_energies: {song_id: {bucket: blended_energy}}
        Populated only for (song, bucket) pairs with >= min_bucket_plays plays.
        Uses k=5 in the blend (per-context credibility accumulates faster).
    """
    plays_df = pd.read_sql_query("""
        SELECT p.song_id, p.played_at, p.play_duration_ms, p.duration_ms
        FROM plays p
        WHERE p.play_source != 'jam_excluded'
          AND p.play_duration_ms IS NOT NULL AND p.duration_ms > 0
          AND p.inferred_skip IN ('skip','partial','full')
    """, conn)

    sess_df = pd.read_sql_query("""
        SELECT start_time, end_time, hour_of_day, stakes_level
        FROM sessions
    """, conn)

    if plays_df.empty or sess_df.empty:
        return {}, {}

    plays_df["played_at"]  = pd.to_datetime(plays_df["played_at"], format="ISO8601", utc=True)
    sess_df["start_time"]  = pd.to_datetime(sess_df["start_time"], format="ISO8601", utc=True)
    sess_df["end_time"]    = pd.to_datetime(sess_df["end_time"],   format="ISO8601", utc=True)
    plays_df["completion"] = (plays_df["play_duration_ms"] / plays_df["duration_ms"]).clip(0.0, 1.0)

    sess_df["time_bucket"]   = sess_df["hour_of_day"].apply(_time_bucket)
    sess_df["energy_demand"] = sess_df.apply(
        lambda r: _play_energy_demand(r["time_bucket"], r["stakes_level"]), axis=1
    )

    plays_df = plays_df.sort_values("played_at")
    sess_df  = sess_df.sort_values("start_time")

    merged = pd.merge_asof(
        plays_df,
        sess_df[["start_time", "end_time", "energy_demand", "time_bucket"]],
        left_on="played_at", right_on="start_time",
        direction="backward",
    )
    merged = merged[merged["played_at"] <= merged["end_time"]]

    if merged.empty:
        return {}, {}

    # Tag energies and total play counts for Bayesian blending
    tag_energies = dict(conn.execute(
        "SELECT song_id, COALESCE(energy_score, 0.0) FROM song_tags"
    ).fetchall())
    play_counts = dict(conn.execute(
        "SELECT song_id, COUNT(*) FROM plays "
        "WHERE play_source != 'jam_excluded' GROUP BY song_id"
    ).fetchall())

    global_energies:  dict = {}
    context_energies: dict = {}

    for song_id, grp in merged.groupby("song_id"):
        if len(grp) < min_plays:
            continue

        tag_e  = tag_energies.get(song_id, 0.0)
        n_play = play_counts.get(song_id, len(grp))

        # Global: completion-weighted mean across all contexts, blended with tag
        w          = grp["completion"].clip(0.01)
        raw_global = float(np.average(grp["energy_demand"], weights=w))
        global_energies[song_id] = _bayesian_blend(raw_global, tag_e, n_play, k=10)

        # Per-context: same logic but only plays in each time bucket
        ctx: dict = {}
        for bucket in ("morning", "afternoon", "late_night"):
            sub = grp[grp["time_bucket"] == bucket]
            if len(sub) < min_bucket_plays:
                continue
            w_b     = sub["completion"].clip(0.01)
            raw_ctx = float(np.average(sub["energy_demand"], weights=w_b))
            ctx[bucket] = _bayesian_blend(raw_ctx, tag_e, len(sub), k=5)
        if ctx:
            context_energies[song_id] = ctx

    n_ctx_total = sum(len(v) for v in context_energies.values())
    print(f"  Behavioral energy learned: {len(global_energies)} songs globally, "
          f"{n_ctx_total} (song, bucket) context scores")
    return global_energies, context_energies


def compute_artist_priors(global_energies: dict, conn) -> dict:
    """
    Cold-start prior for songs with no direct behavioral data.

    Uses the average behavioral energy of other songs by the same artist
    as a prior, blended with the song's tag energy using k=15 (weaker
    credibility than direct observation — the prior can only capture the
    artist's general tendency, not this song's specific context fit).

    Returns {song_id: blended_energy} only for songs without global behavioral
    data that share an artist with at least one song that has data.
    """
    from collections import defaultdict

    artist_songs = conn.execute("SELECT song_id, artist_name FROM songs").fetchall()
    tag_energies = dict(conn.execute(
        "SELECT song_id, COALESCE(energy_score, 0.0) FROM song_tags"
    ).fetchall())

    artist_behavioral: dict = defaultdict(list)
    for song_id, artist_name in artist_songs:
        if song_id in global_energies:
            artist_behavioral[artist_name].append(global_energies[song_id])

    priors: dict = {}
    for song_id, artist_name in artist_songs:
        if song_id in global_energies:
            continue  # already has direct data
        samples = artist_behavioral.get(artist_name, [])
        if not samples:
            continue
        artist_mean     = float(np.mean(samples))
        tag_e           = tag_energies.get(song_id, 0.0)
        priors[song_id] = _bayesian_blend(artist_mean, tag_e, len(samples), k=15)

    print(f"  Artist priors computed for {len(priors)} cold-start songs")
    return priors


def regress_tags_on_song_energies(
    song_energies: dict, conn, min_tag_songs: int = 5
) -> dict | None:
    """
    Given per-song behavioral energies, fits ridge regression: energy ~ tags.
    Returns updated tag weights on the same [-1, 1] scale as ENERGY_WEIGHTS.

    This is the "bottom-up" approach:
      1. Individual songs get their own energy level from behavior.
      2. Tag weights are derived from those song energies — many songs
         per tag, not many plays per tag. Tag X gets the mean energy of
         all songs whose behavioral energy we've learned and which have tag X.

    Why this fixes the "hype = -0.421" problem: the old approach used raw
    completion deviation per tag, which is dominated by queue skips on a single
    song. Here, each tag gets a regression coefficient from all songs carrying
    that tag, and each song's energy is an individualized estimate — songs with
    "hype" that genuinely complete in high-energy sessions will push the "hype"
    coefficient positive even if one song was frequently queue-skipped.

    Returns None if fewer than 20 songs have learned energies.
    """
    from sklearn.linear_model import RidgeCV

    if len(song_energies) < 20:
        print(f"  Tag regression deferred: only {len(song_energies)} songs with behavioral energy"
              " (need 20)")
        return None

    tags_df = pd.read_sql_query(
        "SELECT song_id, top_tags FROM song_tags WHERE top_tags IS NOT NULL", conn
    )
    tags_df["top_tags"] = tags_df["top_tags"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    tags_df = tags_df[tags_df["song_id"].isin(song_energies)].copy()
    tags_df["energy"] = tags_df["song_id"].map(song_energies)

    from collections import Counter
    tag_counts = Counter(t for row in tags_df["top_tags"] for t in row)
    vocab = sorted(tag for tag, cnt in tag_counts.items() if cnt >= min_tag_songs)

    if len(vocab) < 5:
        print(f"  Tag regression deferred: only {len(vocab)} tags with ≥{min_tag_songs} songs")
        return None

    tag_idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(tags_df), len(vocab)), dtype=np.float32)
    for i, tags in enumerate(tags_df["top_tags"]):
        for t in tags:
            if t in tag_idx:
                X[i, tag_idx[t]] = 1.0
    y = tags_df["energy"].values.astype(np.float32)

    n_folds = min(5, max(2, len(y) // 10))
    model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=n_folds)
    model.fit(X, y)

    result = {
        tag: round(float(np.clip(model.coef_[i], -1.0, 1.0)), 3)
        for tag, i in tag_idx.items()
    }

    top = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print(f"  Tag regression: {len(vocab)} tags  (α={model.alpha_:.3f})")
    print(f"    Top 10 by |weight|:")
    for tag, w in top:
        n = tag_counts[tag]
        print(f"      {w:+.3f}  {tag:<30}  (n={n} songs)")

    return result


# ── Energy weight learning from completion data (legacy, kept for reference) ──

def learn_energy_weights(conn, min_tag_samples: int = 5) -> dict | None:
    """
    DEPRECATED — superseded by learn_song_energies + regress_tags_on_song_energies.

    The old approach measures unconditional completion deviation per tag:
      learned_weight = (mean_completion_for_tag - global_mean) / 0.40

    Problem: this is context-blind and dominated by queue skips. A tag appearing
    on one song with two queue skips will produce a strongly negative weight
    even if that song is genuinely liked in the right context. With only 5 plays
    minimum per tag, one bad session poisons the whole tag.

    The new bottom-up approach (learn_song_energies → regress_tags_on_song_energies)
    fixes this by: (a) assigning per-song energies from session context, not raw
    completion; (b) fitting tag weights from many songs per tag, not many plays
    per tag, so single-song artifacts can't dominate.

    Kept here to keep learned_params.json backward-compatible.
    """
    df = load_training_data(conn)
    if len(df) < MIN_PLAYS:
        return None

    # Load tags for all played songs
    tags_df = pd.read_sql_query(
        "SELECT song_id, top_tags FROM song_tags WHERE top_tags IS NOT NULL", conn
    )
    tags_df["top_tags"] = tags_df["top_tags"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    df = df.merge(tags_df, on="song_id", how="left")
    global_mean = df["completion_ratio"].mean()

    # Vectorized: explode per-song tag lists and groupby-agg
    df_tags = df[["completion_ratio", "top_tags"]].copy()
    df_tags["top_tags"] = df_tags["top_tags"].apply(lambda x: (x or [])[:8])
    df_exp = df_tags.explode("top_tags").dropna(subset=["top_tags"])
    df_exp = df_exp[df_exp["top_tags"].astype(str).str.len() > 0]

    tag_stats = df_exp.groupby("top_tags")["completion_ratio"].agg(["mean", "count"])
    tag_stats = tag_stats[tag_stats["count"] >= min_tag_samples]

    # Scale: ±0.40 deviation in completion ratio maps to ±1.0 weight.
    # 0.40 chosen because that's roughly the gap between full-listen and skip.
    deviation = tag_stats["mean"] - global_mean
    learned: dict[str, float] = {
        tag: round(float(np.clip(dev / 0.40, -1.0, 1.0)), 3)
        for tag, dev in deviation.items()
    }

    if len(learned) < 10:
        return None

    print(f"  Learned energy weights from {len(df)} plays  "
          f"({len(learned)} tags with ≥{min_tag_samples} samples)")
    top = sorted(learned.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for tag, w in top:
        n = int(tag_stats.loc[tag, "count"])
        print(f"    {w:+.3f}  {tag:<30} (n={n})")

    return learned


# ── Binge shape learning ─────────────────────────────────────────────────────

def compute_binge_shape(conn, min_episodes: int = MIN_BINGE_EPISODES) -> list | None:
    """
    Learns the typical normalized binge arc from completed episodes.

    Each episode's daily_plays array is normalized by its peak so a 4x/day
    binge and a 2x/day binge have the same shape. Episodes are padded with 0s
    or truncated to BINGE_SHAPE_LENGTH days, then averaged.

    Returns a list of BINGE_SHAPE_LENGTH floats in [0, 1].
    Gated behind min_episodes — returns None until enough data accumulates.

    phase34.py uses this as:
        adjusted_binge_score = binge_score × shape[days_since_binge_start]
    On day 0 (peak, shape ≈ 1.0): full suppression.
    On day 10 (tail, shape ≈ 0.1): fatigue largely reasserts.
    Before min_episodes is reached: shape is None → binge_score unchanged (flat suppression).
    """
    try:
        rows = conn.execute("""
            SELECT daily_plays, peak_plays
            FROM binge_episodes
            WHERE end_date IS NOT NULL AND peak_plays > 0 AND episode_length >= 2
        """).fetchall()
    except Exception:
        print("    binge_episodes table not found — run phase2.py first.")
        return None

    completed = [(json.loads(dp), peak) for dp, peak in rows if peak > 0]

    if len(completed) < min_episodes:
        print(f"    Only {len(completed)} completed binge episodes — "
              f"shape learning deferred (need {min_episodes}).")
        return None

    shapes = []
    for daily_plays, peak_plays in completed:
        if not daily_plays:
            continue
        normalized = [p / peak_plays for p in daily_plays]
        if len(normalized) < BINGE_SHAPE_LENGTH:
            normalized += [0.0] * (BINGE_SHAPE_LENGTH - len(normalized))
        shapes.append(normalized[:BINGE_SHAPE_LENGTH])

    if len(shapes) < min_episodes:
        return None

    shape = [round(float(np.mean([s[i] for s in shapes])), 4)
             for i in range(BINGE_SHAPE_LENGTH)]
    print(f"    Binge shape from {len(shapes)} episodes "
          f"(first 7 days): {[f'{v:.2f}' for v in shape[:7]]}")
    return shape


# ── Binge threshold calibration (stubbed — needs more data) ──────────────────

def calibrate_binge_threshold(df: pd.DataFrame) -> dict:
    """
    Future: learn the play-count and fatigue thresholds that define a 'binge'
    for this user's specific listening patterns, rather than using
    play_count >= 4 AND fatigue > 0.6.

    Requires: songs with 4+ plays in a short window — not enough data yet.
    Currently returns None (phase34.py uses hardcoded thresholds).
    """
    songs_with_multiple_plays = df.groupby("song_id").size()
    eligible = (songs_with_multiple_plays >= 4).sum()
    if eligible < 5:
        print(f"    Only {eligible} songs with 4+ plays — binge threshold learning deferred.")
        return {}
    # TODO: fit a classifier on (play_velocity, time_cluster) → binge_label
    return {}

# ── A/B skip rate comparison ─────────────────────────────────────────────────

def compare_smartshuffle_vs_natural(conn) -> dict:
    """
    Compares skip rates for SmartShuffle-attributed plays vs natural Spotify listening.

    SmartShuffle plays: play_source = 'smartshuffle_queued'
      (attributed by collect_data.attribute_plays_to_queues — plays within 2 hours
      of a queue push whose song IDs match the pushed queue)

    Natural baseline: play_source = 'playlist'
      (songs played via Spotify's own shuffle, without SmartShuffle involvement)

    This is the real A/B comparison: did SmartShuffle produce better listening
    behaviour than just letting Spotify shuffle?

    Returns a dict with per-condition n, skip_rate, full_rate, avg_completion —
    or an empty dict if there aren't enough attributed plays yet (< 20 total).
    """
    if not _table_exists(conn, "queue_skips"):
        return {}

    # Only count plays from sessions where queue-comparison skip inference is complete.
    # Each play is matched to its most recent push to avoid double-counting plays that
    # fall within two overlapping 2-hour push windows.
    rows = conn.execute("""
        SELECT p.play_source,
               COUNT(*)                                                    AS n,
               ROUND(AVG(p.inferred_skip = 'skip'),  3)                   AS skip_rate,
               ROUND(AVG(p.inferred_skip = 'full'),  3)                   AS full_rate,
               ROUND(AVG(CAST(p.play_duration_ms AS REAL)
                         / NULLIF(p.duration_ms, 0)), 3)                  AS avg_completion
        FROM plays p
        WHERE p.inferred_skip IN ('skip', 'partial', 'full')
          AND EXISTS (
              SELECT 1 FROM queue_pushes qp
              WHERE qp.algorithm || '_queued' = p.play_source
                AND qp.skips_inferred_at IS NOT NULL
                AND REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' ')
                    <= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                AND datetime(REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' '), '+2 hours')
                    >= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                AND qp.push_id = (
                    SELECT push_id FROM queue_pushes qp2
                    WHERE qp2.algorithm || '_queued' = p.play_source
                      AND REPLACE(SUBSTR(qp2.pushed_at, 1, 19), 'T', ' ')
                          <= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                      AND datetime(REPLACE(SUBSTR(qp2.pushed_at, 1, 19), 'T', ' '), '+2 hours')
                          >= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                      AND qp2.skips_inferred_at IS NOT NULL
                    ORDER BY qp2.pushed_at DESC LIMIT 1
                )
          )
        GROUP BY p.play_source
    """).fetchall()

    # Inferred skips from sessions with complete inference
    inferred = conn.execute("""
        SELECT qs.algorithm, COUNT(*) AS n
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
        WHERE qp.skips_inferred_at IS NOT NULL
        GROUP BY qs.algorithm
    """).fetchall()
    inferred_counts = {algo: n for algo, n in inferred}

    ss_rows = [r for r in rows if r[0] == 'smartshuffle_queued']
    if not ss_rows or ss_rows[0][1] < 20:
        return {}

    result = {}
    for source, n, skip_rate, full_rate, avg_comp in rows:
        algo    = source.replace("_queued", "")
        key     = "smartshuffle" if algo == "smartshuffle" else "baseline"
        n_inf   = inferred_counts.get(algo, 0)
        n_total = n + n_inf
        n_quick = round(skip_rate * n) + n_inf
        result[key] = {
            "n":              n_total,
            "skip_rate":      round(n_quick / n_total, 3) if n_total > 0 else skip_rate,
            "inferred_skips": n_inf,
            "full_rate":      round(full_rate * n / n_total, 3) if n_total > 0 else full_rate,
            "avg_completion": avg_comp,
        }
    return result


def _table_exists(conn, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


# ── Weekly skip-loss tracking ─────────────────────────────────────────────────

def compute_weekly_loss(conn) -> dict:
    """
    Computes a skip-weighted loss score per week for SmartShuffle vs natural Spotify.

    Loss formula (range [0, 1]):
        loss = QUICK_SKIP_WEIGHT × quick_skip_rate + MID_SKIP_WEIGHT × mid_skip_rate

    quick skip  completion < 30%  — listener rejected the song immediately
    mid skip    30% ≤ completion < 80%  — listener gave it a chance but bailed
    full play   completion ≥ 80%  — no loss contribution

    SmartShuffle loss is stored as a weekly trend so we can see whether the model
    is improving over time. Natural Spotify is the control series.

    Returns:
        {
            "smartshuffle": {
                "total": float,
                "weekly": [{"week": "YYYY-MM-DD", "loss": float, "n": int}, ...]
            },
            "natural": { ... }
        }
    Weeks below MIN_LOSS_PLAYS_PER_WEEK are excluded. Returns {} if no data.
    """
    from collections import defaultdict
    from datetime import date, timedelta

    if not _table_exists(conn, "queue_skips"):
        return {}

    rows = conn.execute("""
        SELECT p.play_source, p.played_at, p.play_duration_ms, p.duration_ms
        FROM plays p
        WHERE p.play_duration_ms IS NOT NULL
          AND p.duration_ms > 0
          AND EXISTS (
              SELECT 1 FROM queue_pushes qp
              WHERE qp.algorithm || '_queued' = p.play_source
                AND qp.skips_inferred_at IS NOT NULL
                AND REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' ')
                    <= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                AND datetime(REPLACE(SUBSTR(qp.pushed_at, 1, 19), 'T', ' '), '+2 hours')
                    >= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                AND qp.push_id = (
                    SELECT push_id FROM queue_pushes qp2
                    WHERE qp2.algorithm || '_queued' = p.play_source
                      AND REPLACE(SUBSTR(qp2.pushed_at, 1, 19), 'T', ' ')
                          <= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                      AND datetime(REPLACE(SUBSTR(qp2.pushed_at, 1, 19), 'T', ' '), '+2 hours')
                          >= REPLACE(SUBSTR(p.played_at, 1, 19), 'T', ' ')
                      AND qp2.skips_inferred_at IS NOT NULL
                    ORDER BY qp2.pushed_at DESC LIMIT 1
                )
          )
    """).fetchall()

    if not rows:
        return {}

    weekly: dict = defaultdict(lambda: {"quick": 0, "mid": 0, "n": 0})
    for source, played_at, played_ms, total_ms in rows:
        ratio      = min(played_ms / total_ms, 1.0)
        play_date  = date.fromisoformat(str(played_at)[:10])
        week_start = (play_date - timedelta(days=play_date.weekday())).isoformat()
        d = weekly[(source, week_start)]
        d["n"] += 1
        if ratio < QUICK_SKIP_THRESHOLD:
            d["quick"] += 1
        elif ratio < MID_SKIP_THRESHOLD:
            d["mid"] += 1

    # Fold in inferred skips from sessions with complete inference
    inf_rows = conn.execute("""
        SELECT qs.algorithm, qp.pushed_at
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
        WHERE qp.skips_inferred_at IS NOT NULL
    """).fetchall()
    for algo, pushed_at in inf_rows:
        source     = f"{algo}_queued"
        play_date  = date.fromisoformat(str(pushed_at)[:10])
        week_start = (play_date - timedelta(days=play_date.weekday())).isoformat()
        d = weekly[(source, week_start)]
        d["n"]     += 1
        d["quick"] += 1

    result = {}
    for source_key, result_key in (("smartshuffle_queued", "smartshuffle"),
                                    ("random_baseline_queued", "baseline")):
        weeks_data = sorted(
            [(wk, d) for (src, wk), d in weekly.items() if src == source_key],
            key=lambda x: x[0],
        )
        weekly_losses = []
        total_quick = total_mid = total_n = 0

        for week, d in weeks_data:
            n = d["n"]
            if n < MIN_LOSS_PLAYS_PER_WEEK:
                continue
            loss = round(
                QUICK_SKIP_WEIGHT * d["quick"] / n
                + MID_SKIP_WEIGHT * d["mid"] / n,
                4,
            )
            weekly_losses.append({"week": week, "loss": loss, "n": n})
            total_quick += d["quick"]
            total_mid   += d["mid"]
            total_n     += n

        if total_n == 0:
            continue

        total_loss = round(
            QUICK_SKIP_WEIGHT * total_quick / total_n
            + MID_SKIP_WEIGHT * total_mid / total_n,
            4,
        )
        result[result_key] = {"total": total_loss, "weekly": weekly_losses}

    return result


# ── Alpha blending ────────────────────────────────────────────────────────────

def compute_alpha(n_plays: int) -> float:
    """
    Controls how much the formula contributes vs the learned weights.
    1.0 = pure formula, 0.0 = pure ML.

    Transition schedule:
      < 30 plays:  1.00  (formula only — no data to learn from)
      100 plays:   0.86  (formula strongly dominant)
      250 plays:   0.56  (balanced)
      500 plays:   0.30  (ML dominant, formula as regulariser)
      > 500 plays: 0.30  (floor — never fully discard the structural prior)
    """
    if n_plays < MIN_PLAYS:
        return 1.0
    return round(max(0.3, 1.0 - (n_plays - MIN_PLAYS) / 500), 3)

def blend_weights(defaults: dict, learned: dict, alpha: float) -> dict:
    """Linear interpolation between default and learned weights."""
    return {
        k: round(alpha * defaults[k] + (1 - alpha) * learned[k], 4)
        for k in defaults
    }

# ── Persistence ───────────────────────────────────────────────────────────────

def save_params(params: dict, path: str = PARAMS_PATH):
    params["trained_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved → {path}")

def load_params(path: str = PARAMS_PATH) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)

    for table in ["plays", "song_tags", "song_scores"]:
        if conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
            print(f"ERROR: {table} is empty. Run phase2.py first.")
            conn.close()
            return

    print("=== SmartShuffle — ML Model Training ===\n")

    print("Loading training data...")
    df = load_training_data(conn)
    n  = len(df)
    skip_counts = df["inferred_skip"].value_counts().to_dict()
    print(f"  {n} plays  (skip={skip_counts.get('skip',0)}  partial={skip_counts.get('partial',0)}"
          f"  full={skip_counts.get('full',0)}  unknown={skip_counts.get('unknown',0)})")
    print(f"  completion: mean={df['completion_ratio'].mean():.2f}")
    print(f"  time buckets: {dict(df['time_bucket'].value_counts())}")

    if n < MIN_PLAYS:
        print(f"\nLearning context energy targets (cold start — {n}/{MIN_PLAYS} plays minimum)...")
        context_targets  = seed_context_targets_from_tags(conn)
        playlist_targets = {}
        learned = DEFAULT_WEIGHTS
        alpha   = compute_alpha(n)   # will be 1.0, pure formula
    else:
        print("\nLearning context energy targets...")
        context_targets, playlist_targets = learn_context_targets(df, conn)

        print("\nLearning ranking weights...")
        learned = learn_weights(df, context_targets)
        alpha   = compute_alpha(n)

    blended = blend_weights(DEFAULT_WEIGHTS, learned, alpha)
    print(f"\n  Alpha = {alpha:.2f}  "
          f"(formula {alpha*100:.0f}% / ML {(1-alpha)*100:.0f}%)")
    print(f"  Blended weights: {blended}")

    print("\nLearning per-song behavioral energies...")
    global_energies, context_energies = learn_song_energies(conn)

    print("\nComputing artist priors for cold-start songs...")
    artist_priors = compute_artist_priors(global_energies, conn)

    print("\nFitting tag weights from per-song energies (bottom-up)...")
    regression_tag_weights = regress_tags_on_song_energies(global_energies, conn)

    # Ensure all behavioral energy columns exist
    for col in ("behavioral_energy_score",
                "behavioral_energy_morning",
                "behavioral_energy_afternoon",
                "behavioral_energy_late_night"):
        try:
            conn.execute(f"ALTER TABLE song_tags ADD COLUMN {col} REAL")
        except Exception:
            pass  # column already exists

    # Write global scores: artist prior first, then direct behavioral overrides it
    all_global = {**artist_priors, **global_energies}
    if all_global:
        conn.executemany(
            "UPDATE song_tags SET behavioral_energy_score = ? WHERE song_id = ?",
            [(e, sid) for sid, e in all_global.items()],
        )
        print(f"  Wrote behavioral_energy_score: {len(global_energies)} direct "
              f"+ {len(artist_priors)} artist prior")

    # Write per-context scores
    for bucket in ("morning", "afternoon", "late_night"):
        col     = f"behavioral_energy_{bucket}"
        updates = [
            (ctx[bucket], sid)
            for sid, ctx in context_energies.items()
            if bucket in ctx
        ]
        if updates:
            conn.executemany(
                f"UPDATE song_tags SET {col} = ? WHERE song_id = ?",
                updates,
            )
            print(f"  Wrote {col} for {len(updates)} songs")

    conn.commit()

    # Compute behavioral context targets from the per-context distributions we just wrote.
    # These are on the behavioral energy scale (-0.5 late_night → +0.4 morning) and must
    # be used instead of the tag-based context_targets when scoring behavioral energies.
    print("\nComputing behavioral context targets...")
    behavioral_context_targets: dict = {}
    for bucket in ("morning", "afternoon", "late_night"):
        col  = f"behavioral_energy_{bucket}"
        rows = conn.execute(
            f"SELECT {col} FROM song_tags WHERE {col} IS NOT NULL"
        ).fetchall()
        if len(rows) >= 5:
            vals = [r[0] for r in rows]
            behavioral_context_targets[bucket] = round(float(np.mean(vals)), 4)
            print(f"    {bucket:<12}  mean={behavioral_context_targets[bucket]:+.3f}  (n={len(rows)})")
        else:
            behavioral_context_targets[bucket] = DEFAULT_CONTEXT_TARGETS.get(bucket, 0.0)
            print(f"    {bucket:<12}  too few ({len(rows)}) — using structural default")

    # Legacy path kept for backward-compatibility in learned_params.json
    print("\nLearning energy weights from completion data (legacy)...")
    energy_weights = learn_energy_weights(conn)
    if energy_weights:
        print(f"  → {len(energy_weights)} tag weights learned")
    else:
        print("  → Insufficient data, using static ENERGY_WEIGHTS")

    print("\nSmartShuffle vs random baseline comparison...")
    comparison = compare_smartshuffle_vs_natural(conn)
    if comparison:
        ss  = comparison.get("smartshuffle", {})
        bl  = comparison.get("baseline", {})
        def _fmt(d):
            inf = d.get("inferred_skips", 0)
            inf_str = f"  +{inf} inferred" if inf else ""
            return (f"n={d.get('n'):4d}  skip={d.get('skip_rate'):.1%}"
                    f"  full={d.get('full_rate'):.1%}"
                    f"  completion={d.get('avg_completion') or 0:.2f}{inf_str}")
        print(f"  SmartShuffle   {_fmt(ss)}")
        if bl:
            print(f"  Random baseline {_fmt(bl)}")
        else:
            print("  Random baseline: no plays yet (need 20+ random_baseline_queued plays).")
    else:
        print("  Insufficient smartshuffle_queued plays yet (need 20+).")

    print("\nComputing weekly skip loss...")
    loss = compute_weekly_loss(conn)
    if loss:
        for src, data in loss.items():
            weekly = data["weekly"]
            trend  = "  →  ".join(f"{w['loss']:.3f}(w{i+1})" for i, w in enumerate(weekly[-4:]))
            delta  = ""
            if len(weekly) >= 2:
                delta = f"  Δ={weekly[-1]['loss'] - weekly[-2]['loss']:+.3f}"
            print(f"  {src:<13}  total={data['total']:.3f}  "
                  f"weeks={len(weekly)}  trend: {trend}{delta}")
    else:
        print("  Insufficient play data yet.")

    print("\nCalibrating binge threshold...")
    binge_params = calibrate_binge_threshold(df)

    print("\nLearning binge shape...")
    binge_shape = compute_binge_shape(conn)
    if binge_shape:
        print(f"  → shape learned ({BINGE_SHAPE_LENGTH}-day arc)")
    else:
        print("  → deferred; flat suppression in effect until enough episodes accumulate")

    print("\nLearning playlist vibe targets...")
    playlist_vibe_targets = learn_playlist_vibe_targets(conn)
    if playlist_vibe_targets:
        print(f"  → vibe targets for {len(playlist_vibe_targets)} playlists")
    else:
        print("  → no vibe-scored songs yet — run score_vibes.py first")

    params = {
        "context_targets":              context_targets,
        "behavioral_context_targets":   behavioral_context_targets,
        "playlist_context_targets":     playlist_targets,
        "playlist_vibe_targets":        playlist_vibe_targets,
        "weights":                      blended,
        "weights_learned":              learned,
        "weights_default":              DEFAULT_WEIGHTS,
        "alpha":                        alpha,
        "n_plays":                      n,
        "binge_params":                 binge_params,
        "binge_shape":                  binge_shape,
        "learned_energy_weights":       energy_weights,
        "regression_tag_weights":       regression_tag_weights,
        "loss":                         loss or None,
    }
    save_params(params)
    conn.close()

    print("\nRe-run after ~50 new plays to update. "
          "Weights will shift toward ML as data accumulates.")

if __name__ == "__main__":
    main()
