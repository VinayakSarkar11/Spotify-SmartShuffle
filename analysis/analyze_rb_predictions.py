#!/usr/bin/env python3
"""
analyze_rb_predictions.py — Retrospective model validation.

Evaluates SmartShuffle features against observed skip/complete outcomes.

Two modes:
  --mode rb   Random baseline queues (unbiased song sample, smaller n)
  --mode ss   SmartShuffle queues (larger n, but songs are pre-filtered by formula)

Usage:
  python analyze_rb_predictions.py          # defaults to rb
  python analyze_rb_predictions.py --mode ss
  python analyze_rb_predictions.py --mode rb
"""

import argparse
import json
import math
import os
import sqlite3
from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "smartshuffle.db")
VIBE_PARAMS_PATH    = os.path.join(_ROOT, "data", "vibe_params.json")
LEARNED_PARAMS_PATH = os.path.join(_ROOT, "data", "learned_params.json")

# ── Constants (mirrors recommend.py) ─────────────────────────────────────────
FATIGUE_HALF_LIFE_DAYS = 14.0
FATIGUE_DECAY_RATE     = math.log(2) / FATIGUE_HALF_LIFE_DAYS
FATIGUE_MAX_PLAYS      = 5.0  # weighted plays at which fatigue = 1.0
ENERGY_SIGMA           = 0.30

WEIGHTS = {
    "energy_match":   0.00,   # disabled (Option B) — replaced by vibe_match
    "vibe_match":     0.40,
    "fatigue":        0.25,
    "artist_fatigue": 0.10,
    "coverage":       0.20,
    "skip":           0.05,
    "recency":        0.30,
    "binge_boost":    0.15,
    "artist_comp":    0.10,
}

ARTIST_DOMINANCE_THRESHOLD = 0.40  # mirrors score.py

# ── Load external params ───────────────────────────────────────────────────────
with open(VIBE_PARAMS_PATH) as f:
    vibe_params = json.load(f)

with open(LEARNED_PARAMS_PATH) as f:
    learned = json.load(f)

PLAYLIST_VIBE_TARGETS  = learned.get("playlist_vibe_targets", {})
PLAYLIST_CTX_TARGETS   = learned.get("playlist_context_targets", {})
CONTEXT_TARGETS        = learned.get("context_targets", {
    "morning": 0.2392, "afternoon": 0.2476, "late_night": 0.2032,
})
VIBE_SIGMAS = vibe_params.get("vibe_sigmas", {
    "morning":    {"content": 0.31, "melodic": 0.31, "bpm": 0.28},
    "afternoon":  {"content": 0.31, "melodic": 0.30, "bpm": 0.28},
    "late_night": {"content": 0.31, "melodic": 0.29, "bpm": 0.30},
})


# ── Feature helpers ────────────────────────────────────────────────────────────

def time_bucket(dt: pd.Timestamp) -> str:
    h = dt.hour
    if h >= 21 or h < 6:  return "late_night"
    if h < 11:             return "morning"
    return "afternoon"


def hist_fatigue_at(plays_before: pd.DataFrame, push_at: pd.Timestamp) -> float:
    """Exponential-decay weighted plays as of push_at. Same formula as score.py."""
    if plays_before.empty:
        return 0.0
    days_ago = (push_at - plays_before["played_at"]).dt.total_seconds() / 86400
    weights  = np.exp(-FATIGUE_DECAY_RATE * days_ago.clip(lower=0))
    return float(min(weights.sum() / FATIGUE_MAX_PLAYS, 1.0))


def hist_skip_rate_at(plays_before: pd.DataFrame) -> Optional[float]:
    """Skip rate from plays before push (None if no play history)."""
    n = len(plays_before)
    if n == 0:
        return None  # no history — song is cold, can't compute
    skips = (plays_before["inferred_skip"] == "skip").sum()
    return float(skips / n)


def energy_match(energy_score: float, target_energy: float, sigma: float = ENERGY_SIGMA) -> float:
    return float(np.exp(-0.5 * ((energy_score - target_energy) / sigma) ** 2))


def vibe_match_score(
    vc: Optional[float], vm: Optional[float], vb: Optional[float],
    target: Optional[dict], sigmas: dict,
) -> Optional[float]:
    if target is None or any(v is None for v in [vc, vm, vb]):
        return None
    gc = math.exp(-0.5 * (vc - target["content"]) ** 2 / sigmas["content"] ** 2)
    gm = math.exp(-0.5 * (vm - target["melodic"]) ** 2 / sigmas["melodic"] ** 2)
    gb = math.exp(-0.5 * (vb - target["bpm"])     ** 2 / sigmas["bpm"]     ** 2)
    return (gc + gm + gb) / 3.0


def get_context_target(playlist_id: str, bucket: str) -> float:
    if playlist_id and playlist_id in PLAYLIST_CTX_TARGETS:
        return PLAYLIST_CTX_TARGETS[playlist_id].get(bucket, CONTEXT_TARGETS.get(bucket, 0.25))
    return CONTEXT_TARGETS.get(bucket, 0.25)


# ── Main analysis ──────────────────────────────────────────────────────────────

def main(mode: str = "rb"):
    algorithm  = "random_baseline" if mode == "rb" else "smartshuffle"
    play_source = "random_baseline_queued" if mode == "rb" else "smartshuffle_queued"

    conn = sqlite3.connect(DB_PATH)

    # ── 1. Load all pushes ordered by time ────────────────────────────────────
    all_pushes = pd.read_sql_query("""
        SELECT qp.push_id, qp.pushed_at, qp.algorithm, qp.mode,
               q.songs, q.context, q.playlist_id
        FROM queue_pushes qp
        JOIN queues q ON q.queue_id = qp.queue_id
        ORDER BY qp.pushed_at
    """, conn)
    all_pushes["pushed_at"] = pd.to_datetime(all_pushes["pushed_at"], format="ISO8601", utc=True)

    # Build per-push window end: next push start (or 12 h after for last push)
    all_pushes["window_end"] = (
        all_pushes["pushed_at"].shift(-1)
        .fillna(all_pushes["pushed_at"].iloc[-1] + pd.Timedelta(hours=12))
    )

    target_pushes = all_pushes[all_pushes["algorithm"] == algorithm].copy()
    print(f"Total {algorithm} pushes: {len(target_pushes)}")

    if mode == "ss":
        print("  Note: SmartShuffle songs are pre-filtered by formula — AUCs reflect")
        print("  within-queue rank discrimination, not absolute score quality.")

    # ── 2. Load queued plays ───────────────────────────────────────────────────
    all_rb_plays = pd.read_sql_query(f"""
        SELECT song_id, played_at, inferred_skip, play_source
        FROM plays
        WHERE play_source = '{play_source}'
          AND inferred_skip IS NOT NULL
        ORDER BY played_at
    """, conn)
    all_rb_plays["played_at"] = pd.to_datetime(all_rb_plays["played_at"], format="ISO8601", utc=True)

    # Load ALL plays (for historical feature computation)
    all_plays = pd.read_sql_query("""
        SELECT song_id, artist_name, played_at, inferred_skip
        FROM plays
        WHERE inferred_skip IS NOT NULL
        ORDER BY played_at
    """, conn)
    all_plays["played_at"] = pd.to_datetime(all_plays["played_at"], format="ISO8601", utc=True)

    # Load queue_skips
    all_queue_skips = pd.read_sql_query(
        "SELECT push_id, song_id FROM queue_skips", conn
    )

    # Load song features (static: vibe scores, energy, current skip_rate)
    song_features = pd.read_sql_query("""
        SELECT
            pt.song_id, pt.playlist_id,
            COALESCE(st.behavioral_energy_morning,    st.behavioral_energy_score, st.energy_score, 0.0) AS energy_morning,
            COALESCE(st.behavioral_energy_afternoon,  st.behavioral_energy_score, st.energy_score, 0.0) AS energy_afternoon,
            COALESCE(st.behavioral_energy_late_night, st.behavioral_energy_score, st.energy_score, 0.0) AS energy_late_night,
            s.vibe_content, s.vibe_melodic, s.vibe_bpm,
            ss.coverage_debt, ss.binge_score,
            ss.artist_fatigue, ss.artist_binge_score,
            ss.evergreen_score,
            ss.artist_comp_rate, ss.artist_comp_conf,
            ss.artist_name
        FROM playlist_tracks pt
        LEFT JOIN songs       s  ON pt.song_id = s.song_id
        LEFT JOIN song_tags   st ON pt.song_id = st.song_id
        LEFT JOIN song_scores ss ON pt.song_id = ss.song_id
    """, conn)
    song_features = song_features.drop_duplicates(subset=["song_id"])

    # Precompute historical artist completion rates at the global level
    # (approximation: uses all plays, not as-of push date — acceptable since
    #  artist taste is stable over the analysis window).
    artist_plays_all = pd.read_sql_query("""
        SELECT artist_name, song_id, inferred_skip
        FROM plays
        WHERE inferred_skip IN ('full','partial','skip')
    """, conn)

    def _artist_comp_rates(plays_df):
        """Mirrors score.py compute_artist_completion_rates() with dominance exclusion."""
        song_counts = (
            plays_df.groupby(["artist_name", "song_id"])
            .agg(n_plays=("song_id","count"),
                 n_comp=("inferred_skip", lambda x: x.isin({"full","partial"}).sum()))
            .reset_index()
        )
        out = {}
        for artist, grp in song_counts.groupby("artist_name"):
            if not artist:
                continue
            total = grp["n_plays"].sum()
            dominant = set(grp.loc[grp["n_plays"]/total > ARTIST_DOMINANCE_THRESHOLD, "song_id"])
            nd = grp[~grp["song_id"].isin(dominant)]
            n_unique = len(nd)
            if n_unique == 0:
                out[artist] = (0.5, 0.0)
                continue
            nd_plays = nd["n_plays"].sum()
            nd_comp  = nd["n_comp"].sum()
            comp_rate = nd_comp / nd_plays if nd_plays > 0 else 0.5
            confidence = min(n_unique / 5.0, 1.0)
            out[artist] = (round(comp_rate, 4), round(confidence, 4))
        return out

    artist_comp_lookup = _artist_comp_rates(artist_plays_all)
    conn.close()

    # ── 3. Build labeled dataset ───────────────────────────────────────────────
    records = []

    for _, push in target_pushes.iterrows():
        push_id      = int(push["push_id"])
        push_at      = push["pushed_at"]
        window_end   = push["window_end"]
        bucket       = push["context"]
        playlist_id  = push["playlist_id"]

        songs        = json.loads(push["songs"])
        queue_ids    = {s["song_id"] for s in songs}

        # Queue skips for this push
        q_skips      = set(all_queue_skips[all_queue_skips["push_id"] == push_id]["song_id"])

        # Plays in this push's window for songs in this queue
        mask = (
            all_rb_plays["song_id"].isin(queue_ids)
            & (all_rb_plays["played_at"] >= push_at)
            & (all_rb_plays["played_at"] <  window_end)
        )
        window_plays = all_rb_plays[mask].drop_duplicates(subset=["song_id"])
        played_ids   = set(window_plays["song_id"])

        # Context targets and vibe target for this push
        context_target = get_context_target(playlist_id, bucket)
        vibe_target    = PLAYLIST_VIBE_TARGETS.get(playlist_id)
        vs             = VIBE_SIGMAS.get(bucket, {"content": 0.30, "melodic": 0.30, "bpm": 0.30})

        for song_id in queue_ids:
            # Determine outcome
            if song_id in q_skips:
                outcome = 1  # queue-skipped
            elif song_id in played_ids:
                row_play = window_plays[window_plays["song_id"] == song_id].iloc[0]
                outcome  = 1 if row_play["inferred_skip"] == "skip" else 0
            else:
                continue  # unknown (never reached in queue)

            # Historical features: all plays for this song BEFORE push_at
            plays_before = all_plays[
                (all_plays["song_id"] == song_id) & (all_plays["played_at"] < push_at)
            ]

            fatigue   = hist_fatigue_at(plays_before, push_at)
            skip_rate = hist_skip_rate_at(plays_before)  # None if cold

            # Energy match (using time-bucket specific behavioral energy)
            sf = song_features[song_features["song_id"] == song_id]
            if sf.empty:
                continue

            sf_row = sf.iloc[0]
            energy_col = f"energy_{bucket}"
            e_score   = float(sf_row[energy_col]) if energy_col in sf_row else 0.0
            e_match   = energy_match(e_score, context_target)

            # Vibe match
            v_match = vibe_match_score(
                sf_row.get("vibe_content"),
                sf_row.get("vibe_melodic"),
                sf_row.get("vibe_bpm"),
                vibe_target, vs,
            )

            def _f(v) -> float:
                return 0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

            # Artist completion rate: global precomputed with dominance exclusion
            artist_name = sf_row.get("artist_name") or ""
            ac_rate, ac_conf = artist_comp_lookup.get(artist_name, (0.5, 0.0))
            # Confidence-shrunk signal centered at 0: 1.0 comp→+1.0, 0.5→0.0, 0.0→-1.0
            eff_comp     = 0.5 + (ac_rate - 0.5) * ac_conf
            ac_signal    = (eff_comp - 0.5) * 2.0

            # Temporal artist skip rate: only plays before this push (avoids leakage)
            if artist_name:
                ap_before = all_plays[
                    (all_plays["artist_name"] == artist_name)
                    & (all_plays["played_at"] < push_at)
                ]
                hist_artist_skip = (
                    round(float((ap_before["inferred_skip"] == "skip").mean()), 4)
                    if len(ap_before) >= 2 else None
                )
            else:
                hist_artist_skip = None

            binge_s = _f(sf_row.get("binge_score"))
            ev_s    = min(_f(sf_row.get("evergreen_score")), 0.60)
            skip_r_val = skip_rate if skip_rate is not None else 0.0
            skip_mod   = 0.25 + 0.75 * skip_r_val
            eff_fatigue = fatigue * (1.0 - max(binge_s, ev_s)) * skip_mod

            records.append({
                "push_id":             push_id,
                "song_id":             song_id,
                "bucket":              bucket,
                "playlist_id":         playlist_id,
                "outcome":             outcome,
                "hist_fatigue":        fatigue,
                "eff_fatigue":         eff_fatigue,
                "hist_skip_rate":      skip_rate,
                "hist_artist_skip_rate": hist_artist_skip,
                "energy_match":        e_match,
                "vibe_match":          v_match,
                "coverage_debt":       _f(sf_row.get("coverage_debt")),
                "binge_score":         binge_s,
                "artist_fatigue":      _f(sf_row.get("artist_fatigue")),
                "evergreen_score":     _f(sf_row.get("evergreen_score")),
                "artist_comp_rate":    ac_rate,
                "artist_comp_conf":    ac_conf,
                "artist_comp_signal":  ac_signal,
            })

    df = pd.DataFrame(records)

    if mode == "ss":
        before = len(df)
        df = df[df["bucket"] != "morning"].copy()
        print(f"  [Dropped morning bucket: {before - len(df)} songs removed]")

    print(f"\n{'=' * 68}")
    print("  Random Baseline Retrospective Prediction Analysis")
    print(f"{'=' * 68}")
    print(f"\n  Songs with known outcome: {len(df)}")
    print(f"  Skipped:                  {df['outcome'].sum()} ({df['outcome'].mean():.1%})")
    print(f"  Completed:                {(df['outcome']==0).sum()} ({(df['outcome']==0).mean():.1%})")
    print(f"  Pushes covered:           {df['push_id'].nunique()}")
    print(f"  Unique songs:             {df['song_id'].nunique()}")
    print(f"  Unique playlists:         {df['playlist_id'].nunique()}")

    print(f"\n  Outcome breakdown by time bucket:")
    for bucket_name, grp in df.groupby("bucket"):
        skip_r = grp["outcome"].mean()
        print(f"    {bucket_name:<12}  n={len(grp):>4}  skip={skip_r:.1%}")

    # ── 4. Per-feature AUC ────────────────────────────────────────────────────
    # AUC of feature predicting COMPLETION (outcome=0), so we flip outcome.
    # AUC > 0.5: high feature value → more completions.
    # AUC < 0.5: high feature value → more skips (inverted predictor).

    completion = 1 - df["outcome"]  # 1=completed, 0=skipped

    print(f"\n  {'─' * 64}")
    print(f"  Per-feature AUC  (predicting completion;  > 0.5 = useful signal)")
    print(f"  {'─' * 64}")

    def feature_auc(col, invert=False):
        mask = df[col].notna()
        if mask.sum() < 10:
            return None, mask.sum()
        vals = df.loc[mask, col].astype(float)
        y    = completion[mask].astype(int)
        if invert:
            vals = -vals
        try:
            return roc_auc_score(y, vals), mask.sum()
        except Exception:
            return None, mask.sum()

    feature_rows = [
        ("vibe_match",             False, "3-axis vibe Gaussian match to playlist target"),
        ("energy_match",           False, "Energy Gaussian match (disabled in formula, weight=0)"),
        ("hist_skip_rate",         True,  "Per-song historical skip rate (inverted)"),
        ("hist_artist_skip_rate",  True,  "Per-artist historical skip rate (inverted)"),
        ("hist_fatigue",           True,  "Raw exponential-decay fatigue (inverted)"),
        ("eff_fatigue",            True,  "Effective fatigue after binge+skip_mod (inverted) — what formula uses"),
        ("coverage_debt",          False, "Coverage debt (stale songs played more)"),
        ("binge_score",            False, "Binge momentum"),
        ("artist_fatigue",         True,  "Artist fatigue (inverted)"),
        ("evergreen_score",        False, "Evergreen score"),
        ("artist_comp_rate",       False, "Artist completion rate (raw)"),
        ("artist_comp_signal",     False, "Artist comp signal (confidence-weighted, centered at 0)"),
    ]

    # Per-feature AUC framed as skip prediction (high value → skip)
    # AUC > 0.5 means high feature value predicts skip
    # For "invert=True" features (already useful as skip predictors), we flip the sign
    auc_results = {}
    for col, inv, desc in feature_rows:
        # skip AUC = 1 - completion AUC when inv=False
        # skip AUC = completion AUC when inv=True (already inverted for completion,
        #   so raw value predicts skip, meaning high raw value -> more skips)
        comp_auc, n = feature_auc(col, invert=inv)
        if comp_auc is not None:
            skip_auc = 1.0 - comp_auc  # flip: high feature → skip
            auc_results[col] = skip_auc
            stars = "***" if skip_auc >= 0.65 else ("**" if skip_auc >= 0.60 else ("*" if skip_auc >= 0.55 else ""))
            print(f"  {col:<26}  skip_AUC={skip_auc:.3f}  {stars}  n={n}")
        else:
            auc_results[col] = None
            print(f"  {col:<26}  skip_AUC=N/A  (n={n})")

    # ── 5. Skip-risk scores ────────────────────────────────────────────────────
    def _base_components(row):
        binge  = float(row["binge_score"])
        ev     = min(float(row["evergreen_score"]), 0.60)
        vm     = row["vibe_match"] if pd.notna(row["vibe_match"]) else 1.0
        fat    = row["hist_fatigue"]
        skip_r = row["hist_skip_rate"] if pd.notna(row["hist_skip_rate"]) else 0.0
        skip_mod = 0.25 + 0.75 * skip_r
        eff_fat  = fat * (1.0 - max(binge, ev)) * skip_mod
        af       = float(row["artist_fatigue"]) * (1.0 - binge)
        cov      = min(row["coverage_debt"] / 4.0, 1.0)
        completion_score = (
              WEIGHTS["vibe_match"]     * vm
            - WEIGHTS["fatigue"]        * eff_fat
            - WEIGHTS["artist_fatigue"] * af
            + WEIGHTS["coverage"]       * cov
            - WEIGHTS["skip"]           * skip_r
            + WEIGHTS["binge_boost"]    * binge
        )
        return completion_score

    # skip_risk = −completion_score  (higher → more likely to skip)
    df["skip_risk"]       = df.apply(lambda r: -_base_components(r), axis=1)
    df["skip_risk_no_ac"] = df["skip_risk"]  # same: artist_comp not in base
    df["skip_risk_with_ac"] = df.apply(
        lambda r: -(_base_components(r) + WEIGHTS["artist_comp"] * float(r["artist_comp_signal"])),
        axis=1,
    )

    skip_outcome = df["outcome"].astype(int)  # 1=skipped, 0=completed

    def score_auc(col):
        try:
            return roc_auc_score(skip_outcome, df[col])
        except Exception:
            return None

    auc_no_ac   = score_auc("skip_risk_no_ac")
    auc_with_ac = score_auc("skip_risk_with_ac")

    def quintile_table(score_col, label):
        print(f"\n  {'─' * 64}")
        print(f"  {label}")
        auc = score_auc(score_col)
        stars = "***" if (auc or 0) >= 0.65 else ("**" if (auc or 0) >= 0.60 else ("*" if (auc or 0) >= 0.55 else ""))
        print(f"  skip_AUC={auc:.3f}  {stars}  (higher bucket = higher skip risk)")
        print(f"  {'─' * 64}")
        df["_q"] = pd.qcut(df[score_col], q=5, labels=False, duplicates="drop")
        stats = (
            df.groupby("_q")
            .agg(
                n         = ("outcome", "count"),
                skipped   = ("outcome", "sum"),
                score_min = (score_col, "min"),
                score_max = (score_col, "max"),
            )
            .assign(
                skip_rate  = lambda d: d["skipped"] / d["n"],
                comp_rate  = lambda d: 1 - d["skipped"] / d["n"],
            )
        )
        print(f"  {'Bucket':>8}  {'Skip-risk range':>22}  {'n':>5}  {'skip%':>7}  {'comp%':>7}  bar")
        for q, row in stats.iterrows():
            bar = "█" * int(row["comp_rate"] * 20)
            print(
                f"  {int(q)+1:>8}  [{row['score_min']:+.3f}, {row['score_max']:+.3f}]"
                f"  {int(row['n']):>5}  {row['skip_rate']:>6.1%}  {row['comp_rate']:>6.1%}  {bar}"
            )
        df.drop(columns=["_q"], inplace=True)

    quintile_table("skip_risk_no_ac",   "Skip-risk buckets  — WITHOUT artist_comp")
    quintile_table("skip_risk_with_ac", "Skip-risk buckets  — WITH artist_comp")

    # ── 6. Q1 breakdown by time bucket ────────────────────────────────────────
    print(f"\n  {'─' * 64}")
    print(f"  Q1 (lowest skip-risk) breakdown — diagnosing the anomaly")
    print(f"  {'─' * 64}")
    df["_q"] = pd.qcut(df["skip_risk_no_ac"], q=5, labels=False, duplicates="drop")
    q1 = df[df["_q"] == 0]
    rest = df[df["_q"] != 0]
    print(f"  {'Bucket':<14}  {'n':>5}  {'skip%':>7}  {'comp%':>7}  bar")
    for bucket_name, grp in df.groupby("bucket"):
        q1_grp   = q1[q1["bucket"] == bucket_name]
        rest_grp = rest[rest["bucket"] == bucket_name]
        q1_skip  = q1_grp["outcome"].mean() if len(q1_grp) else float("nan")
        all_skip = grp["outcome"].mean()
        q1_n     = len(q1_grp)
        bar = "█" * int((1 - q1_skip) * 20) if q1_n else ""
        print(f"  {bucket_name:<14}  Q1 n={q1_n:>3}  skip={q1_skip:>5.1%}  (all songs skip={all_skip:.1%})")
    df.drop(columns=["_q"], inplace=True)

    # Skip rate by time bucket across all quintiles
    print(f"\n  Skip rate by bucket × quintile (no-AC):")
    df["_q"] = pd.qcut(df["skip_risk_no_ac"], q=5, labels=False, duplicates="drop")
    pivot = df.groupby(["_q", "bucket"])["outcome"].mean().unstack(fill_value=float("nan"))
    buckets_sorted = sorted(pivot.columns)
    print(f"  {'Quintile':>8}  " + "  ".join(f"{b:>12}" for b in buckets_sorted))
    for q, row in pivot.iterrows():
        vals = "  ".join(
            f"{row[b]:>11.1%}" if not pd.isna(row.get(b, float("nan"))) else f"{'—':>12}"
            for b in buckets_sorted
        )
        print(f"  {int(q)+1:>8}  {vals}")
    df.drop(columns=["_q"], inplace=True)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print(f"\n  {'─' * 64}")
    print(f"  Summary  (skip_AUC > 0.5 = useful skip signal)")
    print(f"  {'─' * 64}")
    useful  = [(c, a) for c, a in auc_results.items() if a is not None and a > 0.55]
    neutral = [(c, a) for c, a in auc_results.items() if a is not None and 0.45 <= a <= 0.55]
    anti    = [(c, a) for c, a in auc_results.items() if a is not None and a < 0.45]

    if useful:
        print(f"  Good skip signals (skip_AUC > 0.55):")
        for c, a in sorted(useful, key=lambda x: -x[1]):
            print(f"    {c:<26}  skip_AUC={a:.3f}")
    if neutral:
        print(f"  Near-random (skip_AUC 0.45–0.55):")
        for c, a in neutral:
            print(f"    {c:<26}  skip_AUC={a:.3f}")
    if anti:
        print(f"  Anti-signals (skip_AUC < 0.45 = predicts completion):")
        for c, a in anti:
            print(f"    {c:<26}  skip_AUC={a:.3f}")

    print(f"\n  Combined skip_AUC: without artist_comp={auc_no_ac:.3f}  |  with artist_comp={auc_with_ac:.3f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rb", "ss"], default="rb",
                        help="rb = random baseline (default), ss = smartshuffle")
    args = parser.parse_args()
    main(mode=args.mode)
