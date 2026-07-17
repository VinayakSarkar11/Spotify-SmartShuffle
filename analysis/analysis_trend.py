#!/usr/bin/env python3
"""
Weekly trend analysis: is SmartShuffle improving over time?
Plots skip rate, weighted skip rate (full queue only), and session length
for SS and RB side-by-side. Uses RB as a control to distinguish algorithm
improvement from shifting user preferences.

Statistical test: linear regression on (SS - RB) difference per week.
Note: with ~5 weeks of overlapping data, significance is very hard to achieve
(Mann-Kendall needs 8+ points for p<0.05 on a perfect monotonic trend).
"""
import os
import sqlite3
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy import stats

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(_ROOT, "data", "smartshuffle.db")
MIN_PLAYS = 4
OUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_trend.png")


conn = sqlite3.connect(DB_PATH)

# ── Per-push play data with session date ──────────────────────────────────────
push_rows = conn.execute(f"""
    WITH play_push AS (
        SELECT p.play_source, p.inferred_skip,
               (SELECT qp2.push_id FROM queue_pushes qp2
                WHERE qp2.algorithm || '_queued' = p.play_source
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1) AS push_id
        FROM plays p
        WHERE p.play_source IN ('smartshuffle_queued','random_baseline_queued')
          AND p.inferred_skip IN ('skip','partial','full')
    ),
    valid_pushes AS (
        SELECT pp.push_id,
               COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
               qp.algorithm, qp.mode
        FROM play_push pp
        JOIN queue_pushes qp ON qp.push_id = pp.push_id
        GROUP BY pp.push_id HAVING COUNT(*) >= {MIN_PLAYS}
    )
    SELECT vp.session_id, vp.algorithm, vp.mode,
           -- ISO week of the session's first push
           strftime('%Y-W%W', DATE(MIN(REPLACE(SUBSTR(qp.pushed_at,1,10),'T',' ')))) AS week,
           COUNT(pp.play_source)                                                       AS plays_n,
           SUM(CASE WHEN pp.inferred_skip='skip'               THEN 1 ELSE 0 END)     AS dur_skips,
           SUM(CASE WHEN pp.inferred_skip IN ('partial','full') THEN 1 ELSE 0 END)    AS engaged_n
    FROM play_push pp
    JOIN valid_pushes vp ON vp.push_id = pp.push_id
    JOIN queue_pushes qp ON qp.push_id = vp.session_id   -- date from initial push
    GROUP BY vp.session_id
""").fetchall()

# ── Per-push queue skips ──────────────────────────────────────────────────────
qs_rows = conn.execute(f"""
    WITH play_push AS (
        SELECT p.play_source,
               (SELECT qp2.push_id FROM queue_pushes qp2
                WHERE qp2.algorithm || '_queued' = p.play_source
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1) AS push_id
        FROM plays p
        WHERE p.play_source IN ('smartshuffle_queued','random_baseline_queued')
          AND p.inferred_skip IN ('skip','partial','full')
    ),
    valid_pushes AS (
        SELECT pp.push_id,
               COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
               qp.mode
        FROM play_push pp
        JOIN queue_pushes qp ON qp.push_id = pp.push_id
        GROUP BY pp.push_id HAVING COUNT(*) >= {MIN_PLAYS}
    )
    SELECT vp.session_id, vp.mode,
           COUNT(*)  AS raw_qs,
           SUM(CASE WHEN vp.mode = 'full'
               THEN MAX(0.2, 1.0 - CAST(qs.queue_position AS REAL) / 50.0)
               ELSE 1.0 END) AS weighted_qs
    FROM queue_skips qs
    JOIN valid_pushes vp ON vp.push_id = qs.push_id
    GROUP BY vp.session_id
""").fetchall()

conn.close()

qs_by_session = {r[0]: {'mode': r[1], 'raw': r[2], 'weighted': r[3]} for r in qs_rows}

# ── Aggregate pushes → sessions ───────────────────────────────────────────────
sess = defaultdict(lambda: dict(
    algorithm=None, mode=None, week=None,
    plays_n=0, dur_skips=0, engaged_n=0, raw_qs=0, weighted_qs=0.0,
))

for session_id, alg, mode, week, plays_n, dur_skips, engaged_n in push_rows:
    key = (alg, mode, session_id)
    s = sess[key]
    s.update(algorithm=alg, mode=mode, week=week)
    s['plays_n']   += plays_n
    s['dur_skips'] += dur_skips
    s['engaged_n'] += engaged_n
    q = qs_by_session.get(session_id, {'raw': 0, 'weighted': 0.0})
    s['raw_qs']    += q['raw']
    s['weighted_qs'] += q['weighted']

sessions = []
for (alg, mode, sid), s in sess.items():
    if s['plays_n'] < MIN_PLAYS or s['week'] is None:
        continue
    tot_uw  = s['plays_n'] + s['raw_qs']
    skip_uw = s['dur_skips'] + s['raw_qs']
    tot_w   = s['plays_n'] + s['weighted_qs']
    skip_w  = s['dur_skips'] + s['weighted_qs']
    sessions.append({
        **s, 'session_id': sid,
        'unweighted_rate': skip_uw / tot_uw if tot_uw > 0 else None,
        'weighted_rate':   skip_w  / tot_w  if tot_w  > 0 else None,
        'tot_uw': tot_uw, 'skip_uw': skip_uw,
        'tot_w':  tot_w,  'skip_w':  skip_w,
    })

# ── Weekly aggregates ─────────────────────────────────────────────────────────
# week_data[(week, alg, mode)] = list of session dicts
from collections import defaultdict as dd

def weekly_agg(mode_filter=None):
    """
    Returns dict: week -> {alg: {skip_uw, tot_uw, skip_w, tot_w, engaged, n_sessions}}
    """
    buckets = defaultdict(lambda: defaultdict(lambda: dict(
        skip_uw=0, tot_uw=0, skip_w=0.0, tot_w=0.0, engaged=0, n=0
    )))
    for s in sessions:
        if mode_filter and s['mode'] != mode_filter:
            continue
        w   = s['week']
        alg = s['algorithm']
        b   = buckets[w][alg]
        b['skip_uw'] += s['skip_uw']
        b['tot_uw']  += s['tot_uw']
        b['skip_w']  += s['skip_w']
        b['tot_w']   += s['tot_w']
        b['engaged'] += s['engaged_n']
        b['n']       += 1
    return buckets

all_weeks  = weekly_agg()           # combined
full_weeks = weekly_agg('full')     # full-queue only (for weighted skip)

weeks_sorted = sorted(set(s['week'] for s in sessions))
week_idx     = {w: i for i, w in enumerate(weeks_sorted)}


def extract_series(buckets, metric):
    """Returns (week_labels, ss_vals, rb_vals) — None where data missing."""
    ss_vals, rb_vals = [], []
    for w in weeks_sorted:
        ss = buckets[w].get('smartshuffle')
        rb = buckets[w].get('random_baseline')
        def val(b):
            if b is None or b['n'] == 0:
                return None
            if metric == 'unweighted':
                return b['skip_uw'] / b['tot_uw'] if b['tot_uw'] > 0 else None
            if metric == 'weighted':
                return b['skip_w']  / b['tot_w']  if b['tot_w']  > 0 else None
            if metric == 'length':
                return b['engaged'] / b['n']
        ss_vals.append(val(ss))
        rb_vals.append(val(rb))
    return weeks_sorted, ss_vals, rb_vals


def trend_test(weeks, ss_vals, rb_vals, metric_name):
    """
    Linear regression on (SS - RB) per week where both exist.
    Also tests SS alone for context.
    """
    pairs = [(i, s, r) for i, (s, r) in enumerate(zip(ss_vals, rb_vals))
             if s is not None and r is not None]
    print(f"\n  {metric_name}")
    if len(pairs) < 3:
        print(f"    ⚠  Only {len(pairs)} weeks with both SS and RB — can't test trend")
        return

    xs     = np.array([p[0] for p in pairs])
    diffs  = np.array([p[1] - p[2] for p in pairs])
    slope, intercept, r, p, se = stats.linregress(xs, diffs)
    direction = "↓ improving" if slope < 0 else "↑ worsening"
    print(f"    SS−RB difference trend:  slope={slope:+.3f}/week  r={r:.2f}  p={p:.3f}"
          f"  ({direction})")
    print(f"    Weeks with both: {len(pairs)}  (need ~8 for p<0.05 on perfect monotone trend)")

    # SS alone
    ss_only = [(i, s) for i, s, r in pairs]
    xs2   = np.array([p[0] for p in ss_only])
    ss_y  = np.array([p[1] for p in ss_only])
    slope2, _, r2, p2, _ = stats.linregress(xs2, ss_y)
    dir2 = "↓ improving" if slope2 < 0 else "↑ worsening"
    print(f"    SS alone trend:          slope={slope2:+.3f}/week  r={r2:.2f}  p={p2:.3f}"
          f"  ({dir2})")


# ── Plot ──────────────────────────────────────────────────────────────────────
SS_COLOR = "#1DB954"   # Spotify green
RB_COLOR = "#A0A0A0"

fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
fig.suptitle("SmartShuffle vs Random Baseline — Weekly Trends", fontsize=13, y=0.98)

week_labels = [w.replace("2026-", "") for w in weeks_sorted]   # "W25", "W26" etc.
x = np.arange(len(weeks_sorted))

def plot_metric(ax, weeks, ss_vals, rb_vals, title, ylabel, pct=True, note=None):
    ss_x = [x[i] for i, v in enumerate(ss_vals) if v is not None]
    ss_y = [v    for v in ss_vals if v is not None]
    rb_x = [x[i] for i, v in enumerate(rb_vals) if v is not None]
    rb_y = [v    for v in rb_vals if v is not None]

    ax.plot(ss_x, ss_y, "o-", color=SS_COLOR, lw=2, ms=7, label="SmartShuffle")
    ax.plot(rb_x, rb_y, "s--", color=RB_COLOR, lw=2, ms=7, label="Random baseline")

    # Annotate sample sizes
    buckets_ref = all_weeks if note != 'full_only' else full_weeks
    for i, w in enumerate(weeks_sorted):
        for alg, vals, col in [("smartshuffle", ss_vals, SS_COLOR),
                                ("random_baseline", rb_vals, RB_COLOR)]:
            b = buckets_ref[w].get(alg)
            if b and b['n'] > 0 and vals[i] is not None:
                ax.annotate(f"n={b['n']}", (x[i], vals[i]),
                            textcoords="offset points", xytext=(0, 8),
                            ha='center', fontsize=7.5, color=col, alpha=0.8)

    ax.set_title(title, fontsize=10, pad=4)
    ax.set_ylabel(ylabel, fontsize=9)
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(week_labels, fontsize=9)
    if note:
        ax.set_xlabel(note, fontsize=7.5, style='italic', color='gray')

# Panel 1: unweighted skip rate (combined)
_, ss_uw, rb_uw = extract_series(all_weeks, 'unweighted')
plot_metric(axes[0], weeks_sorted, ss_uw, rb_uw,
            "Unweighted skip rate  (dur skips + raw queue skips) / (plays + queue skips)",
            "Skip rate")

# Panel 2: weighted skip rate — full queue only
_, ss_w, rb_w = extract_series(full_weeks, 'weighted')
plot_metric(axes[1], weeks_sorted, ss_w, rb_w,
            "Weighted skip rate — full queue only  (earlier queue positions penalised more)",
            "Skip rate", note="full_only")

# Panel 3: session length (engaged plays)
_, ss_len, rb_len = extract_series(all_weeks, 'length')
plot_metric(axes[2], weeks_sorted, ss_len, rb_len,
            "Session length (partial + full plays per session)",
            "Songs", pct=False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Plot saved → {OUT_PATH}\n")

# ── Trend tests ───────────────────────────────────────────────────────────────
print("Trend significance tests (linear regression on SS−RB difference per week)")
print("="*64)
trend_test(weeks_sorted, ss_uw,  rb_uw,  "Unweighted skip rate (combined)")
trend_test(weeks_sorted, ss_w,   rb_w,   "Weighted skip rate   (full queue)")
trend_test(weeks_sorted, ss_len, rb_len, "Session length       (combined)")
print()
