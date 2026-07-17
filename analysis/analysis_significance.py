#!/usr/bin/env python3
"""
Statistical significance tests: SmartShuffle vs Random Baseline
Mirrors Streamlit's exact skip-rate formula, then adds per-session tests.

Metrics
-------
1. Unweighted skip rate  = (dur_skips + raw_qs)          / (plays + raw_qs)
2. Weighted skip rate    = (dur_skips + pos_weighted_qs)  / (plays + pos_weighted_qs)
   pos_weight = MAX(0.2, 1.0 - queue_position / 50)  (earlier positions count more)
3. Session length        = engaged plays (partial + full) per session

Both skip rates combine duration-based skips (play < 30% of song length) and
LIS-inferred queue skips (song in queue but never appeared in recently_played).

Play attribution: most-recent push of the same algorithm before each play_at.
Valid pushes: >= 3 attributed plays (same as Streamlit after the threshold change).
Rolling sessions: all refill pushes aggregated into one session row.
"""
import os
import sqlite3
from collections import defaultdict

import numpy as np
from scipy import stats

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "smartshuffle.db")
MIN_PLAYS = 4   # must match Streamlit's valid_pushes threshold


def fmt_p(p: float) -> str:
    if p < 0.001: return f"p<0.001 ***"
    if p < 0.01:  return f"p={p:.4f} **"
    if p < 0.05:  return f"p={p:.4f} *"
    if p < 0.10:  return f"p={p:.4f} (marginal)"
    return        f"p={p:.4f}"


conn = sqlite3.connect(DB_PATH)

# ── Per-push play counts (valid pushes only) ──────────────────────────────────
push_play_rows = conn.execute(f"""
    WITH play_push AS (
        SELECT p.play_source, p.inferred_skip,
               (SELECT qp2.push_id
                FROM queue_pushes qp2
                WHERE qp2.algorithm || '_queued' = p.play_source
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1) AS push_id
        FROM plays p
        WHERE p.play_source IN ('smartshuffle_queued','random_baseline_queued')
          AND p.inferred_skip IN ('skip','partial','full')
    ),
    valid_pushes AS (
        SELECT push_id FROM play_push
        GROUP BY push_id HAVING COUNT(*) >= {MIN_PLAYS}
    )
    SELECT pp.push_id,
           qp.algorithm,
           qp.mode,
           COALESCE(qp.rolling_session_id, qp.push_id)             AS session_id,
           COUNT(*)                                                  AS plays_n,
           SUM(CASE WHEN pp.inferred_skip='skip'              THEN 1 ELSE 0 END) AS dur_skips,
           SUM(CASE WHEN pp.inferred_skip='full'              THEN 1 ELSE 0 END) AS full_n,
           SUM(CASE WHEN pp.inferred_skip IN ('partial','full') THEN 1 ELSE 0 END) AS engaged_n
    FROM play_push pp
    JOIN valid_pushes  vp  ON vp.push_id  = pp.push_id
    JOIN queue_pushes  qp  ON qp.push_id  = pp.push_id
    GROUP BY pp.push_id
""").fetchall()

# ── Per-push queue_skip counts (only for valid pushes) ───────────────────────
qs_rows = conn.execute(f"""
    WITH play_push AS (
        SELECT p.play_source,
               (SELECT qp2.push_id
                FROM queue_pushes qp2
                WHERE qp2.algorithm || '_queued' = p.play_source
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1) AS push_id
        FROM plays p
        WHERE p.play_source IN ('smartshuffle_queued','random_baseline_queued')
          AND p.inferred_skip IN ('skip','partial','full')
    ),
    valid_pushes AS (
        SELECT push_id FROM play_push
        GROUP BY push_id HAVING COUNT(*) >= {MIN_PLAYS}
    )
    SELECT qs.push_id,
           COUNT(*)                                                            AS raw_qs,
           SUM(MAX(0.2, 1.0 - CAST(qs.queue_position AS REAL) / 50.0))       AS weighted_qs
    FROM queue_skips qs
    JOIN valid_pushes vp ON vp.push_id = qs.push_id
    GROUP BY qs.push_id
""").fetchall()

conn.close()

qs_by_push = {r[0]: {'raw': r[1], 'weighted': r[2]} for r in qs_rows}

# ── Aggregate pushes → sessions ───────────────────────────────────────────────
sess: dict = defaultdict(lambda: dict(
    algorithm=None, mode=None,
    plays_n=0, dur_skips=0, full_n=0, engaged_n=0,
    raw_qs=0, weighted_qs=0.0,
))

for push_id, alg, mode, session_id, plays_n, dur_skips, full_n, engaged_n in push_play_rows:
    key = (alg, mode, session_id)
    s = sess[key]
    s['algorithm'] = alg
    s['mode']      = mode
    s['plays_n']   += plays_n
    s['dur_skips'] += dur_skips
    s['full_n']    += full_n
    s['engaged_n'] += engaged_n
    q = qs_by_push.get(push_id, {'raw': 0, 'weighted': 0.0})
    s['raw_qs']    += q['raw']
    s['weighted_qs'] += q['weighted']

sessions = []
for (alg, mode, session_id), s in sess.items():
    if s['plays_n'] < MIN_PLAYS:
        continue
    tot_uw  = s['plays_n'] + s['raw_qs']
    skip_uw = s['dur_skips'] + s['raw_qs']
    tot_w   = s['plays_n'] + s['weighted_qs']
    skip_w  = s['dur_skips'] + s['weighted_qs']
    sessions.append({
        **s,
        'session_id':        session_id,
        'unweighted_rate':   skip_uw / tot_uw  if tot_uw  > 0 else 0.0,
        'weighted_rate':     skip_w  / tot_w   if tot_w   > 0 else 0.0,
        'total_uw':          tot_uw,
        'skip_uw':           skip_uw,
        'total_w':           tot_w,
        'skip_w_n':          skip_w,
    })


# ── Statistical tests ─────────────────────────────────────────────────────────
def run_comparison(label: str, ss: list, rb: list):
    n_ss, n_rb = len(ss), len(rb)
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")
    print(f"  Sessions:  SmartShuffle n={n_ss}   Baseline n={n_rb}")
    if n_ss < 2 or n_rb < 2:
        print("  ⚠  Too few sessions for testing")
        return

    # ── 1. Unweighted skip rate ───────────────────────────────────────────────
    print(f"\n  1. Unweighted skip rate  = (dur_skips + raw_qs) / (plays + raw_qs)")

    ss_uw  = [s['unweighted_rate'] for s in ss]
    rb_uw  = [s['unweighted_rate'] for s in rb]
    ss_skip_tot = sum(s['skip_uw'] for s in ss)
    ss_tot      = sum(s['total_uw'] for s in ss)
    rb_skip_tot = sum(s['skip_uw'] for s in rb)
    rb_tot      = sum(s['total_uw'] for s in rb)

    print(f"     SS  per-session mean {np.mean(ss_uw):.1%} ± {np.std(ss_uw,ddof=1):.1%}"
          f"   aggregate {ss_skip_tot}/{ss_tot} = {ss_skip_tot/ss_tot:.1%}")
    print(f"     RB  per-session mean {np.mean(rb_uw):.1%} ± {np.std(rb_uw,ddof=1):.1%}"
          f"   aggregate {rb_skip_tot}/{rb_tot} = {rb_skip_tot/rb_tot:.1%}")

    _, fp = stats.fisher_exact([[ss_skip_tot, ss_tot - ss_skip_tot],
                                 [rb_skip_tot, rb_tot - rb_skip_tot]])
    _, mp = stats.mannwhitneyu(ss_uw, rb_uw, alternative='two-sided')
    print(f"     Fisher's exact (aggregate):       {fmt_p(fp)}")
    print(f"     Mann-Whitney U (per-session):     {fmt_p(mp)}")

    # ── 2. Weighted skip rate ─────────────────────────────────────────────────
    print(f"\n  2. Weighted skip rate   = (dur_skips + pos_weighted_qs) / (plays + pos_weighted_qs)")

    ss_w  = [s['weighted_rate'] for s in ss]
    rb_w  = [s['weighted_rate'] for s in rb]
    ss_sw_tot = sum(s['skip_w_n'] for s in ss)
    ss_tw_tot = sum(s['total_w']  for s in ss)
    rb_sw_tot = sum(s['skip_w_n'] for s in rb)
    rb_tw_tot = sum(s['total_w']  for s in rb)

    print(f"     SS  per-session mean {np.mean(ss_w):.1%} ± {np.std(ss_w,ddof=1):.1%}"
          f"   aggregate {ss_sw_tot:.0f}/{ss_tw_tot:.0f} = {ss_sw_tot/ss_tw_tot:.1%}")
    print(f"     RB  per-session mean {np.mean(rb_w):.1%} ± {np.std(rb_w,ddof=1):.1%}"
          f"   aggregate {rb_sw_tot:.0f}/{rb_tw_tot:.0f} = {rb_sw_tot/rb_tw_tot:.1%}")

    # Fisher's on rounded weighted counts
    ss_sk_r, ss_tot_r = round(ss_sw_tot), round(ss_tw_tot)
    rb_sk_r, rb_tot_r = round(rb_sw_tot), round(rb_tw_tot)
    _, fp = stats.fisher_exact([[ss_sk_r, ss_tot_r - ss_sk_r],
                                 [rb_sk_r, rb_tot_r - rb_sk_r]])
    _, mp = stats.mannwhitneyu(ss_w, rb_w, alternative='two-sided')
    print(f"     Fisher's exact (aggregate):       {fmt_p(fp)}")
    print(f"     Mann-Whitney U (per-session):     {fmt_p(mp)}")

    # ── 3. Session length ─────────────────────────────────────────────────────
    print(f"\n  3. Session length (engaged plays = partial + full per session)")

    ss_len = [s['engaged_n'] for s in ss]
    rb_len = [s['engaged_n'] for s in rb]

    print(f"     SS  mean {np.mean(ss_len):.1f} ± {np.std(ss_len,ddof=1):.1f}"
          f"   median {np.median(ss_len):.0f}")
    print(f"     RB  mean {np.mean(rb_len):.1f} ± {np.std(rb_len,ddof=1):.1f}"
          f"   median {np.median(rb_len):.0f}")

    _, mp  = stats.mannwhitneyu(ss_len, rb_len, alternative='two-sided')
    _, tp  = stats.ttest_ind(ss_len, rb_len, equal_var=False)
    print(f"     Mann-Whitney U:                   {fmt_p(mp)}")
    print(f"     Welch's t-test:                   {fmt_p(tp)}")


# ── Slice into groups ─────────────────────────────────────────────────────────
def ss(mode=None):
    return [s for s in sessions if s['algorithm'] == 'smartshuffle'
            and (mode is None or s['mode'] == mode)]
def rb(mode=None):
    return [s for s in sessions if s['algorithm'] == 'random_baseline'
            and (mode is None or s['mode'] == mode)]

print("\n" + "="*64)
print("  SmartShuffle — Statistical Significance Analysis")
print("  α=0.05 two-tailed  |  * p<.05  ** p<.01  *** p<.001")
print("="*64)
print(f"\n  Valid sessions (≥{MIN_PLAYS} attributed plays after session rollup)")
print(f"  Rolling:   SS={len(ss('rolling'))}   RB={len(rb('rolling'))}")
print(f"  Full:      SS={len(ss('full'))}   RB={len(rb('full'))}")
print(f"  Combined:  SS={len(ss())}   RB={len(rb())}")

run_comparison("1. ROLLING — SmartShuffle vs Random Baseline",   ss('rolling'), rb('rolling'))
run_comparison("2. FULL QUEUE — SmartShuffle vs Random Baseline", ss('full'),    rb('full'))
run_comparison("3. COMBINED — SmartShuffle vs Random Baseline",   ss(),          rb())

print()
print("  Notes")
print("  ─────")
print("  Duration skip: play lasted < 30% of song length (inferred from timestamp gaps).")
print("  LIS queue skip: song was in queue but never appeared in recently_played.")
print("  Unweighted: raw queue_skip count. Weighted: earlier-position skips count more.")
print("  Fisher's exact tests aggregate counts; Mann-Whitney tests per-session rates.")
print("  Rolling sessions: all refill batches collapsed into one session before testing.")
print()
