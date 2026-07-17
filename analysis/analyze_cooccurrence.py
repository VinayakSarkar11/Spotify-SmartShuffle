#!/usr/bin/env python3
"""
Session co-occurrence analysis — READ ONLY, no DB writes.
Builds a session x song matrix and runs SVD to find latent vibe dimensions.
Run this to explore whether session data clusters songs by sub-genre.
"""
import os
import sqlite3
import numpy as np
import pandas as pd

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "smartshuffle.db")

TEST_SONGS = {
    "4OYGHze5MiMkgozardIRxU": ("Bump Heads",          "HYPE-CHILL"),
    "3ORfa5ilEthp2U0TRcv7kv": ("Patiently Waiting",   "HYPE-CHILL"),
    "4Iedi94TIaB2GGb1nMB68v": ("On Me",               "HYPE-CHILL"),
    "3J8EOeKLTLXORtWPpOU5bE": ("Calling My Phone",    "MELODIC-CHILL"),
    "1p0rEzrK7YtdRZVtiyV7RN": ("Lemonade",            "MELODIC-CHILL"),
    "7dt6x5M1jzdTEt8oCbisTK": ("Better Now",          "MELODIC-CHILL"),
    "2e3Ea0o24lReQFR4FA7yXH": ("Love Yourz",          "EMOTIONAL"),
    "7BRD7x5pt8Lqa1eGYC4dzj": ("CHIHIRO",             "EMOTIONAL"),
    "5zpDHEU12zATwLGvozxPw2": ("Lighters",            "EMOTIONAL"),
    "3QPUYDIiihlsAxqhmDG008": ("Stick Up Kids",       "RAPPING"),
    "3pjUyVbFmM96tYhSaKJwTt": ("Change",              "RAPPING"),
}

conn = sqlite3.connect(DB_PATH)

# --- Load session-play data ---
df = pd.read_sql_query("""
    SELECT s.session_id, p.song_id
    FROM sessions s
    JOIN plays p ON p.played_at >= s.start_time AND p.played_at <= s.end_time
    WHERE p.inferred_skip IN ('full', 'partial')
""", conn)
conn.close()

n_sessions = df["session_id"].nunique()
n_songs    = df["song_id"].nunique()
print(f"Sessions: {n_sessions}  |  Unique songs: {n_songs}  |  Total plays: {len(df)}")

# --- Co-occurrence density ---
pairs = (
    df.merge(df, on="session_id", suffixes=("_a", "_b"))
    .query("song_id_a < song_id_b")
    .groupby(["song_id_a", "song_id_b"])
    .size()
    .reset_index(name="co_count")
)
print(f"\nPair co-occurrence distribution:")
print(pairs["co_count"].value_counts().sort_index().to_string())
print(f"\nPairs with ≥3 co-occurrences: {(pairs['co_count'] >= 3).sum()}")
print(f"Pairs with ≥2 co-occurrences: {(pairs['co_count'] >= 2).sum()}")

# --- Test song co-occurrences ---
test_pairs = pairs[
    pairs["song_id_a"].isin(TEST_SONGS) & pairs["song_id_b"].isin(TEST_SONGS)
].copy()
test_pairs["song_a"] = test_pairs["song_id_a"].map(lambda x: TEST_SONGS[x][0])
test_pairs["song_b"] = test_pairs["song_id_b"].map(lambda x: TEST_SONGS[x][0])
test_pairs["group_a"] = test_pairs["song_id_a"].map(lambda x: TEST_SONGS[x][1])
test_pairs["group_b"] = test_pairs["song_id_b"].map(lambda x: TEST_SONGS[x][1])
print(f"\nTest song co-occurrences (max signal we have):")
if len(test_pairs) == 0:
    print("  (no test songs ever played in the same session)")
else:
    print(test_pairs[["song_a", "song_b", "co_count", "group_a", "group_b"]].sort_values("co_count", ascending=False).to_string(index=False))

# --- SVD ---
try:
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    session_idx = {s: i for i, s in enumerate(sorted(df["session_id"].unique()))}
    song_idx    = {s: i for i, s in enumerate(sorted(df["song_id"].unique()))}
    idx_to_song = {v: k for k, v in song_idx.items()}

    rows = df["session_id"].map(session_idx).values
    cols = df["song_id"].map(song_idx).values
    data = np.ones(len(df))
    M = csr_matrix((data, (rows, cols)), shape=(n_sessions, n_songs))

    n_components = min(6, n_sessions - 1, n_songs - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit_transform(M)
    song_factors = svd.components_.T  # (n_songs, n_components)

    print(f"\nSVD variance explained per component:")
    var = svd.explained_variance_ratio_
    for i, v in enumerate(var):
        print(f"  Component {i+1}: {v:.3f} ({v*100:.1f}%)")

    # Show test songs in factor space (components 2 and 3 = non-popularity axes)
    print(f"\nTest songs in SVD factor space (F2, F3 — beyond pure popularity):")
    print(f"  {'Song':<25} {'Group':<14} {'F2':>6} {'F3':>6} {'F4':>6}")
    print(f"  " + "-"*65)
    test_results = []
    for sid, (name, group) in TEST_SONGS.items():
        if sid in song_idx:
            idx = song_idx[sid]
            f = song_factors[idx]
            test_results.append((name, group, f[1], f[2], f[3] if n_components > 3 else 0))
    test_results.sort(key=lambda x: x[2])
    for name, group, f2, f3, f4 in test_results:
        print(f"  {name:<25} {group:<14} {f2:>6.3f} {f3:>6.3f} {f4:>6.3f}")

except ImportError:
    print("\nsklearn/scipy not available — skipping SVD. Install with: pip install scikit-learn scipy")
