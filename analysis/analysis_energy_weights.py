#!/usr/bin/env python3
"""
Energy weight drift: static Last.fm assumptions vs. user-learned weights.

Three panels:
  1. Scatter: static weight vs. learned weight per tag (divergence = colour)
  2. Bar chart: top 25 biggest divergers (where learned most contradicts static)
  3. Song-level impact: how much would each song's energy score shift if we
     replaced static weights with learned ones?

Also prints:
  - Tags in learned but absent from static (novel user signal)
  - Tags where learned flips the sign (static says hype, user actually skips)
  - Per-song energy score before/after for the top movers
"""
import json
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(_ROOT, "data", "smartshuffle.db")
PARAMS_PATH = os.path.join(_ROOT, "data", "learned_params.json")
OUT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_energy_weights.png")

# ── Static ENERGY_WEIGHTS (mirror of score.py) ────────────────────────────────
STATIC_WEIGHTS = {
    "hype":         1.0,  "energetic":   0.9,  "workout":     0.9,
    "aggressive":   0.9,  "intense":     0.8,  "pump up":     0.8,
    "high energy":  0.8,  "party":       0.8,  "rage":        0.8,
    "dance":        0.7,  "drill":       0.7,
    "hard":         0.7,  "bangers":     0.6,  "club":        0.6,
    "turn up":      0.6,  "lit":         0.6,  "upbeat":      0.6,
    "afrobeat":     0.5,  "afrobeats":   0.5,  "amapiano":    0.5,
    "dancehall":    0.5,  "grime":       0.6,
    "uk drill":     0.7,  "florida drill": 0.7, "chicago drill": 0.7,
    "brooklyn drill": 0.7, "ny drill":   0.7,  "detroit drill": 0.7,
    "rock":         0.55, "hard rock":   0.70, "punk":        0.65,
    "punk rock":    0.65, "metal":       0.80, "heavy metal": 0.80,
    "classic rock": 0.40, "blues rock":  0.40, "pop rock":    0.30,
    "alternative rock": 0.25, "progressive rock": 0.20,
    "psychedelic rock": 0.20, "indie rock": 0.10, "soft rock": -0.10,
    "house":        0.60, "tech house":  0.65, "trance":      0.60,
    "drum and bass": 0.70, "dubstep":    0.65,
    "country":      0.10, "contemporary country": 0.15, "country pop": 0.15,
    "bluegrass":    0.20, "blues":       0.15,
    "folk pop":    -0.10, "sunshine pop": 0.20, "psychedelic pop": -0.20,
    "classical":   -0.55, "piano":      -0.35, "baroque":    -0.25,
    "romantic":    -0.45, "instrumental": -0.25, "orchestral": -0.35,
    "impressionist": -0.50, "medieval":  -0.40, "opera":      -0.30,
    "hip-hop":      0.35, "hip hop":     0.35, "rap":         0.30,
    "trap":         0.15, "pop rap":     0.25, "melodic rap": -0.10,
    "conscious hip hop": 0.0,
    "r&b":          0.15, "rnb":         0.15, "contemporary rnb": 0.10,
    "alternative rnb": 0.05, "neo soul": 0.05,
    "pop":          0.10, "electropop":  0.15, "synth pop":   0.15,
    "electronic":   0.20, "edm":         0.50,
    "alternative":  0.10, "reggae":      0.15, "reggaeton":   0.40,
    "soul":         0.0,  "funk":        0.15, "gospel":      0.05,
    "oldies":       0.10, "jazz":        0.05,
    "ambient":     -0.9,  "sleep":      -0.9,  "peaceful":   -0.9,
    "lo-fi":       -0.8,  "lofi":       -0.8,  "calm":       -0.8,
    "dreamy":      -0.8,  "chill":      -0.7,  "relaxing":   -0.7,
    "chillout":    -0.6,  "slow":       -0.7,  "soft":       -0.7,
    "mellow":      -0.6,  "laid back":  -0.6,  "background": -0.6,
    "late night":  -0.5,  "focus":      -0.5,  "study":      -0.5,
    "rainy day":   -0.5,  "sad":        -0.4,  "melancholy": -0.4,
    "emotional":   -0.4,  "introspective": -0.4, "acoustic":  -0.3,
    "indie":       -0.1,  "bedroom pop": -0.2, "singer-songwriter": -0.2,
}

GENERIC_TAGS = {
    "hip-hop", "hip hop", "rap", "music", "r&b", "rnb", "pop", "soul",
    "electronic", "alternative", "indie", "singer-songwriter",
}
GENERIC_MULT = 0.08


# ── Load learned weights ──────────────────────────────────────────────────────
with open(PARAMS_PATH) as f:
    params = json.load(f)

learned = params.get("learned_energy_weights") or {}
alpha   = params.get("alpha", 0.3)

print(f"Loaded learned_params.json  (trained {params['trained_at'][:10]},"
      f"  α={alpha:.2f},  {len(learned)} learned tags)\n")


# ── Overlap analysis ──────────────────────────────────────────────────────────
both_tags   = sorted(set(STATIC_WEIGHTS) & set(learned))
learned_only = sorted(set(learned) - set(STATIC_WEIGHTS))

static_vals  = np.array([STATIC_WEIGHTS[t] for t in both_tags])
learned_vals = np.array([learned[t]         for t in both_tags])
divergence   = learned_vals - static_vals          # positive = user likes it more than assumed

sign_flipped = [t for t in both_tags
                if np.sign(STATIC_WEIGHTS[t]) != np.sign(learned[t])
                and abs(STATIC_WEIGHTS[t]) >= 0.1]

print(f"Tags in both dicts:          {len(both_tags)}")
print(f"Tags in learned only:        {len(learned_only)}")
print(f"Tags where sign is flipped:  {len(sign_flipped)}")

print("\n── Sign-flipped tags (static says X, user actually skips/listens opposite) ──")
flipped_sorted = sorted(sign_flipped, key=lambda t: abs(divergence[both_tags.index(t)]), reverse=True)
for t in flipped_sorted:
    s = STATIC_WEIGHTS[t]
    l = learned[t]
    print(f"  {t:<30}  static={s:+.2f}  learned={l:+.3f}  Δ={l-s:+.3f}")

print("\n── Learned-only tags (no static prior, pure user signal) — top 20 by |weight| ──")
top_novel = sorted(learned_only, key=lambda t: abs(learned[t]), reverse=True)[:20]
for t in top_novel:
    print(f"  {learned[t]:+.3f}  {t}")


# ── Per-song energy score recomputation ───────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

tag_rows = conn.execute(
    "SELECT st.song_id, s.song_name, s.artist_name, st.tags, st.energy_score "
    "FROM song_tags st "
    "LEFT JOIN songs s ON st.song_id = s.song_id "
    "WHERE st.tags IS NOT NULL"
).fetchall()

# also get skip_rate for context
skip_map = dict(conn.execute(
    "SELECT song_id, skip_rate FROM song_scores WHERE skip_rate IS NOT NULL"
).fetchall())

conn.close()


def score_with_weights(tags: list, weight_dict: dict) -> float:
    has_specific = any(
        weight_dict.get(t["name"]) is not None and t["name"] not in GENERIC_TAGS
        for t in tags
    )
    wsum = 0.0
    tvot = 0.0
    for t in tags:
        intensity = weight_dict.get(t["name"])
        if intensity is None:
            continue
        raw_votes = t["weight"] or 1
        dampen    = has_specific and t["name"] in GENERIC_TAGS
        votes     = raw_votes * (GENERIC_MULT if dampen else 1.0)
        wsum     += intensity * votes
        tvot     += votes
    return round(wsum / tvot, 4) if tvot > 0 else 0.0


def blended_weights(alpha: float) -> dict:
    """α*static + (1-α)*learned for tags in learned; static otherwise."""
    merged = dict(STATIC_WEIGHTS)
    for tag, lw in learned.items():
        if tag in STATIC_WEIGHTS:
            merged[tag] = alpha * STATIC_WEIGHTS[tag] + (1 - alpha) * lw
        else:
            merged[tag] = (1 - alpha) * lw   # no static prior — scale down
    return merged

BLENDED = blended_weights(alpha)

songs = []
for song_id, song_name, artist_name, tags_json, current_score in tag_rows:
    try:
        tags = json.loads(tags_json)
    except Exception:
        continue
    if not tags:
        continue

    static_score  = score_with_weights(tags, STATIC_WEIGHTS)
    learned_score = score_with_weights(tags, learned)         # learned where available
    blend_score   = score_with_weights(tags, BLENDED)

    songs.append({
        "song_id":      song_id,
        "name":         f"{song_name or '?'} — {artist_name or '?'}",
        "current":      float(current_score or 0),
        "static":       static_score,
        "learned":      learned_score,
        "blended":      blend_score,
        "delta_blend":  blend_score - static_score,
        "skip_rate":    skip_map.get(song_id),
    })

deltas     = [s["delta_blend"] for s in songs]
movers_up  = sorted(songs, key=lambda s: s["delta_blend"], reverse=True)[:10]
movers_dn  = sorted(songs, key=lambda s: s["delta_blend"])[:10]

print(f"\n── Song energy score: static vs blended (α={alpha:.2f}) ──")
print(f"  Songs analysed:  {len(songs)}")
print(f"  Mean Δ:          {np.mean(deltas):+.3f}")
print(f"  Std Δ:           {np.std(deltas):.3f}")
print(f"  Songs shifted >0.10:  {sum(1 for d in deltas if abs(d) > 0.10)}")

print("\n  Top 10 songs scoring HIGHER with learned weights (user completes these more)")
for s in movers_up:
    sr = f"  skip={s['skip_rate']:.0%}" if s['skip_rate'] is not None else ""
    print(f"    {s['delta_blend']:+.3f}  {s['static']:+.2f}→{s['blended']:+.2f}  "
          f"{s['name'][:55]}{sr}")

print("\n  Top 10 songs scoring LOWER with learned weights (user skips these more)")
for s in movers_dn:
    sr = f"  skip={s['skip_rate']:.0%}" if s['skip_rate'] is not None else ""
    print(f"    {s['delta_blend']:+.3f}  {s['static']:+.2f}→{s['blended']:+.2f}  "
          f"{s['name'][:55]}{sr}")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 7))
fig.suptitle(
    f"Energy weight drift  —  static Last.fm tags vs. user-learned  (α={alpha:.2f})",
    fontsize=12, y=1.01,
)

# ── Panel 1: scatter static vs learned ───────────────────────────────────────
ax = axes[0]
abs_div = np.abs(divergence)
sc = ax.scatter(static_vals, learned_vals,
                c=divergence, cmap="RdYlGn", vmin=-1.5, vmax=1.5,
                s=60 + 80*abs_div, alpha=0.75, edgecolors="none")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
ax.plot([-1, 1], [-1, 1], color="grey", lw=0.8, ls=":")   # diagonal = perfect agreement

# label the 15 most divergent tags
top_idx = np.argsort(abs_div)[-15:]
for i in top_idx:
    t = both_tags[i]
    ax.annotate(t, (static_vals[i], learned_vals[i]),
                fontsize=6.5, alpha=0.9,
                xytext=(4, 2), textcoords="offset points")

plt.colorbar(sc, ax=ax, label="Δ (learned − static)", shrink=0.8)
ax.set_xlabel("Static weight (Last.fm genre theory)")
ax.set_ylabel("Learned weight (your completion data)")
ax.set_title(f"Static vs. learned per tag  (n={len(both_tags)} shared tags)")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# ── Panel 2: top 25 divergers bar chart ──────────────────────────────────────
ax = axes[1]
n_show = 25
order = np.argsort(np.abs(divergence))[-n_show:][::-1]
d_show = divergence[order]
t_show = [both_tags[i] for i in order]
colors = ["#e06c75" if d < 0 else "#98c379" for d in d_show]
bars   = ax.barh(range(n_show), d_show, color=colors, height=0.7)

# annotate bars with static → learned
for i, (tag, d) in enumerate(zip(t_show, d_show)):
    s = STATIC_WEIGHTS[tag]
    l = learned[tag]
    ax.text(d + (0.02 if d >= 0 else -0.02), i,
            f"{s:+.2f}→{l:+.2f}",
            va="center", ha="left" if d >= 0 else "right", fontsize=7)

ax.set_yticks(range(n_show))
ax.set_yticklabels(t_show, fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ weight  (learned − static)")
ax.set_title("Biggest divergences from static assumption")
green = mpatches.Patch(color="#98c379", label="User listens more than expected")
red   = mpatches.Patch(color="#e06c75", label="User skips more than expected")
ax.legend(handles=[green, red], fontsize=7.5, loc="lower right")

# ── Panel 3: energy score distribution shift ──────────────────────────────────
ax = axes[2]
s_scores = [s["static"]  for s in songs]
b_scores = [s["blended"] for s in songs]

bins = np.linspace(-1, 1, 35)
ax.hist(s_scores, bins=bins, alpha=0.55, color="#61afef", label="Static (current)")
ax.hist(b_scores, bins=bins, alpha=0.55, color="#e5c07b", label=f"Blended (α={alpha:.2f})")

ax.axvline(np.mean(s_scores), color="#61afef", lw=1.5, ls="--",
           label=f"Static mean {np.mean(s_scores):+.2f}")
ax.axvline(np.mean(b_scores), color="#e5c07b", lw=1.5, ls="--",
           label=f"Blended mean {np.mean(b_scores):+.2f}")

ax.set_xlabel("Energy score  (−1 chill → +1 hype)")
ax.set_ylabel("Songs")
ax.set_title("Per-song energy score distribution shift")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {OUT_PATH}")
