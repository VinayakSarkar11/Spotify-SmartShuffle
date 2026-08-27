#!/usr/bin/env python3
"""FastAPI web frontend for SmartShuffle."""
import glob
import json
import math
import os
import secrets
import sqlite3
import subprocess
import sys
from datetime import datetime
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from auth import (
    current_user, delete_user, get_access_token, get_user_paths,
    spotify_for_user, upsert_user, validate_device_id, validate_playlist_id,
    write_spotipy_cache, SPOTIFY_SCOPES,
)

DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)

_IS_PROD = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PRODUCTION"))

app = FastAPI(title="SmartShuffle", docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", secrets.token_hex(32)),
    https_only=_IS_PROD,   # secure flag on cookie in production
    same_site="lax",
)
templates = Jinja2Templates(directory=os.path.join(DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(DIR, "static")), name="static")

_QUALIFY_PLAYS = 4


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db(user_id: str) -> sqlite3.Connection:
    paths = get_user_paths(user_id)
    conn  = sqlite3.connect(paths["db"])
    conn.row_factory = sqlite3.Row
    return conn


def _sp(user_id: str):
    return spotify_for_user(user_id)


def _subprocess_env(user_id: str) -> dict:
    """Build env for subprocess calls: inject user's access token and paths."""
    paths      = get_user_paths(user_id)
    cache_path = os.path.join(os.path.dirname(paths["db"]), "sp_watcher_cache.json")
    write_spotipy_cache(user_id, cache_path)
    return {
        **os.environ,
        "SS_ACCESS_TOKEN":    get_access_token(user_id),
        "SS_DB_PATH":         paths["db"],
        "SS_ROLLING_STATE":   paths["rolling_state"],
        "SS_SESSION_STATE":   paths["session_state"],
        "SS_TOKEN_CACHE_PATH": cache_path,
    }


# ── Shared SQL ────────────────────────────────────────────────────────────────

def _load_vibe_targets() -> dict:
    targets = {}
    for fname, key in [("vibe_params.json", "playlist_targets"),
                       ("learned_params.json", "playlist_vibe_targets")]:
        try:
            with open(os.path.join(ROOT, "data", fname)) as f:
                targets.update(json.load(f).get(key, {}))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return targets


_ROLLING_PLAYS_CTE = """
    rolling_plays AS (
        SELECT p.inferred_skip,
               COALESCE(qp.rolling_session_id, qp.push_id) AS session_id
        FROM plays p
        JOIN queue_pushes qp ON qp.push_id = (
            SELECT qp2.push_id FROM queue_pushes qp2
            WHERE qp2.algorithm || '_queued' = p.play_source
              AND qp2.pushed_at <= p.played_at
            ORDER BY qp2.pushed_at DESC LIMIT 1
        )
        WHERE p.play_source = 'smartshuffle_queued'
          AND p.inferred_skip IN ('skip', 'partial', 'full')
          AND qp.mode = 'rolling'
    )
"""


# ── OAuth routes ──────────────────────────────────────────────────────────────

@app.get("/login")
async def login(request: Request):
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    params = urlencode({
        "client_id":     os.getenv("SPOTIFY_CLIENT_ID", ""),
        "response_type": "code",
        "redirect_uri":  os.getenv("SPOTIFY_REDIRECT_URI", ""),
        "scope":         SPOTIFY_SCOPES,
        "state":         state,
        "show_dialog":   "false",
    })
    return RedirectResponse(f"https://accounts.spotify.com/authorize?{params}")


@app.get("/switch-account")
async def switch_account(request: Request):
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    params = urlencode({
        "client_id":     os.getenv("SPOTIFY_CLIENT_ID", ""),
        "response_type": "code",
        "redirect_uri":  os.getenv("SPOTIFY_REDIRECT_URI", ""),
        "scope":         SPOTIFY_SCOPES,
        "state":         state,
        "show_dialog":   "true",
    })
    return RedirectResponse(f"https://accounts.spotify.com/authorize?{params}")


@app.get("/callback")
async def callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return RedirectResponse(f"/?auth_error={error}")

    if not state or state != request.session.pop("oauth_state", None):
        return RedirectResponse("/?auth_error=state_mismatch")

    # Exchange code for tokens
    import base64
    import requests as http
    client_id     = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI", "")
    creds_b64     = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    resp = http.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": redirect_uri,
        },
        headers={"Authorization": f"Basic {creds_b64}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    if resp.status_code != 200:
        return RedirectResponse("/?auth_error=token_exchange_failed")

    tokens = resp.json()
    access_token  = tokens["access_token"]
    refresh_token = tokens["refresh_token"]
    expires_in    = tokens.get("expires_in", 3600)
    scope         = tokens.get("scope", "")

    # Fetch Spotify user profile to get user_id
    import spotipy
    sp      = spotipy.Spotify(auth=access_token)
    profile = sp.current_user()
    user_id = profile["id"]
    name    = profile.get("display_name") or user_id

    upsert_user(user_id, name, access_token, refresh_token, expires_in, scope)

    request.session["user_id"]      = user_id
    request.session["display_name"] = name

    return RedirectResponse("/dashboard")


@app.post("/disconnect")
async def disconnect(request: Request, user: dict = Depends(current_user)):
    """Remove user tokens and data, then log out."""
    user_id = user["user_id"]
    paths   = get_user_paths(user_id)

    # Delete user's play data (Spotify API data — required by Spotify policy)
    owner_id = os.getenv("SPOTIFY_OWNER_ID", "")
    if user_id != owner_id and os.path.exists(paths["db"]):
        try:
            os.remove(paths["db"])
        except OSError:
            pass

    delete_user(user_id)
    request.session.clear()
    return RedirectResponse("/", status_code=303)


# ── Pages ─────────────────────────────────────────────────────────────────────

_global_stats_cache: dict = {}
_global_stats_at: float  = 0.0
_STATS_TTL = 60 * 60  # 1 hour


def _global_stats() -> dict:
    import time
    global _global_stats_cache, _global_stats_at
    if time.time() - _global_stats_at < _STATS_TTL and _global_stats_cache.get("total_plays"):
        return _global_stats_cache

    db_paths: list[str] = []
    owner_db = os.path.join(ROOT, "data", "smartshuffle.db")
    if os.path.exists(owner_db):
        db_paths.append(owner_db)
    for p in glob.glob(os.path.join(ROOT, "data", "users", "*", "smartshuffle.db")):
        db_paths.append(p)

    if not db_paths:
        return {"total_plays": None, "scored_songs": None}

    try:
        total_plays = 0
        scored_ids: set[str] = set()
        for db_path in db_paths:
            conn = sqlite3.connect(db_path)
            try:
                total_plays += conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
                for (sid,) in conn.execute(
                    "SELECT song_id FROM songs WHERE vibe_content IS NOT NULL"
                ).fetchall():
                    scored_ids.add(sid)
            except sqlite3.OperationalError:
                pass  # DB exists but schema not yet initialised
            finally:
                conn.close()
        if total_plays > 0 or scored_ids:
            _global_stats_cache = {"total_plays": total_plays, "scored_songs": len(scored_ids)}
            _global_stats_at    = time.time()
        else:
            return {"total_plays": None, "scored_songs": None}
    except Exception:
        return {"total_plays": None, "scored_songs": None}

    return _global_stats_cache


@app.get("/health")
async def health():
    keys = ["SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_OWNER_ID",
            "SPOTIFY_REDIRECT_URI", "FERNET_KEY", "SESSION_SECRET_KEY",
            "LASTFM_API_KEY", "RAILWAY_ENVIRONMENT"]
    return JSONResponse({k: bool(os.getenv(k)) for k in keys})


_ADMIN_ALLOWED_FILES = {
    "smartshuffle.db", "vibe_params.json", "learned_params.json",
    "refill_baseline.json",
}


@app.post("/admin/upload-file")
async def admin_upload_file(request: Request, filename: str):
    import gzip as _gzip
    secret = os.getenv("ADMIN_UPLOAD_SECRET", "")
    if not secret or request.headers.get("X-Admin-Secret") != secret:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if filename not in _ADMIN_ALLOWED_FILES:
        return JSONResponse({"error": "Filename not allowed"}, status_code=400)
    body = await request.body()
    if not body:
        return JSONResponse({"error": "Empty body"}, status_code=400)
    if body[:2] == b'\x1f\x8b':
        body = _gzip.decompress(body)
    dest = os.path.join(ROOT, "data", filename)
    tmp  = dest + ".upload.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, dest)
        if filename == "smartshuffle.db":
            global _global_stats_at
            _global_stats_at = 0.0
        return JSONResponse({"ok": True, "file": filename, "bytes": len(body)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



@app.get("/")
async def index(request: Request):
    user_id = request.session.get("user_id")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats":   _global_stats(),
        "user":    user_id,
    })


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@app.get("/dashboard")
async def dashboard(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse("/login")
    user = {"user_id": request.session["user_id"],
            "display_name": request.session.get("display_name", "")}
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})


@app.get("/queue")
async def queue_page(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse("/login")
    user = {"user_id": request.session["user_id"],
            "display_name": request.session.get("display_name", "")}
    return templates.TemplateResponse("queue.html", {"request": request, "user": user})


# ── API: stats ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats(user: dict = Depends(current_user)):
    user_id      = user["user_id"]
    conn         = _db(user_id)
    vibe_targets = _load_vibe_targets()

    try:
        conn.execute("SELECT 1 FROM plays LIMIT 1")
    except sqlite3.OperationalError:
        conn.close()
        return JSONResponse({"summary": {"total_plays": 0, "scored_songs": 0,
                                         "rolling_sessions": 0, "avg_session_songs": None},
                             "rolling_skip_rate": None, "qs_n": 0, "plays_n": 0,
                             "trend": [], "vibe_dists": {}, "history": []})

    total_plays  = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
    scored_songs = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE vibe_content IS NOT NULL"
    ).fetchone()[0]

    qual_row = conn.execute(f"""
        WITH {_ROLLING_PLAYS_CTE},
        per_session AS (
            SELECT session_id, COUNT(*) AS plays_n
            FROM rolling_plays
            GROUP BY session_id
            HAVING plays_n >= {_QUALIFY_PLAYS}
        )
        SELECT COUNT(*) AS session_count,
               ROUND(AVG(plays_n), 1) AS avg_session_songs
        FROM per_session
    """).fetchone()

    rolling_sessions  = qual_row["session_count"]    if qual_row else 0
    avg_session_songs = qual_row["avg_session_songs"] if qual_row else None

    play_row = conn.execute(f"""
        WITH {_ROLLING_PLAYS_CTE}
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN inferred_skip='skip' THEN 1 ELSE 0 END) AS skip_n
        FROM rolling_plays
    """).fetchone()
    qs_row = conn.execute("""
        SELECT COUNT(*) AS qs_n
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
        WHERE qp.algorithm = 'smartshuffle'
          AND qp.mode = 'rolling'
          AND qs.queue_position IS NOT NULL
    """).fetchone()

    n      = (play_row["n"]      if play_row else 0) or 0
    skip_n = (play_row["skip_n"] if play_row else 0) or 0
    qs_n   = (qs_row["qs_n"]     if qs_row   else 0) or 0
    total  = n + qs_n
    rolling_skip_rate = round((skip_n + qs_n) / total, 3) if total else None

    trend_plays = conn.execute("""
        SELECT strftime('%Y-W%W', p.played_at) AS week,
               COUNT(*) AS plays_n
        FROM plays p
        JOIN queue_pushes qp ON qp.push_id = (
            SELECT qp2.push_id FROM queue_pushes qp2
            WHERE qp2.algorithm || '_queued' = p.play_source
              AND qp2.pushed_at <= p.played_at
            ORDER BY qp2.pushed_at DESC LIMIT 1
        )
        WHERE p.play_source = 'smartshuffle_queued'
          AND p.inferred_skip IN ('skip', 'partial', 'full')
          AND qp.mode = 'rolling'
          AND p.played_at >= datetime('now', '-84 days')
        GROUP BY week ORDER BY week
    """).fetchall()

    trend_qs = conn.execute("""
        SELECT strftime('%Y-W%W', qs.inferred_at) AS week,
               COUNT(*) AS qs_n
        FROM queue_skips qs
        JOIN queue_pushes qp ON qs.push_id = qp.push_id
        WHERE qp.algorithm = 'smartshuffle'
          AND qp.mode = 'rolling'
          AND qs.queue_position IS NOT NULL
          AND qs.inferred_at >= datetime('now', '-84 days')
        GROUP BY week ORDER BY week
    """).fetchall()

    qs_by_week = {r["week"]: r["qs_n"] for r in trend_qs}
    trend = [
        {"week": r["week"], "plays_n": r["plays_n"],
         "qs_n": qs_by_week.get(r["week"], 0)}
        for r in trend_plays
    ]

    vibe_dists = {}
    for axis in ("vibe_content", "vibe_melodic", "vibe_bpm"):
        rows = conn.execute(f"""
            SELECT ROUND({axis}, 1) AS bucket, COUNT(*) AS n
            FROM songs WHERE {axis} IS NOT NULL
            GROUP BY bucket ORDER BY bucket
        """).fetchall()
        vibe_dists[axis] = [{"bucket": r["bucket"], "n": r["n"]} for r in rows]

    history = conn.execute(f"""
        WITH session_vibes AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   ROUND(AVG(CAST(json_extract(je.value, '$.vibe_content') AS REAL)), 3) AS avg_content,
                   ROUND(AVG(CAST(json_extract(je.value, '$.vibe_melodic') AS REAL)), 3) AS avg_melodic,
                   ROUND(AVG(CAST(json_extract(je.value, '$.vibe_bpm')     AS REAL)), 3) AS avg_bpm
            FROM queue_pushes qp
            JOIN queues q ON qp.queue_id = q.queue_id
            JOIN json_each(q.songs) je
            WHERE qp.algorithm = 'smartshuffle'
            GROUP BY 1
        ),
        session_sizes AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   SUM(json_array_length(q.songs)) AS total_queued
            FROM queue_pushes qp
            JOIN queues q ON qp.queue_id = q.queue_id
            WHERE qp.algorithm = 'smartshuffle'
            GROUP BY 1
        ),
        play_counts AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS plays_n,
                   SUM(CASE WHEN p.inferred_skip='skip' THEN 1 ELSE 0 END) AS play_skip_n
            FROM plays p
            JOIN queue_pushes qp ON qp.push_id = (
                SELECT qp2.push_id FROM queue_pushes qp2
                WHERE qp2.algorithm || '_queued' = p.play_source
                  AND qp2.pushed_at <= p.played_at
                ORDER BY qp2.pushed_at DESC LIMIT 1
            )
            WHERE p.play_source = 'smartshuffle_queued'
              AND p.inferred_skip IN ('skip', 'partial', 'full')
              AND qp.mode = 'rolling'
            GROUP BY 1
            HAVING COUNT(*) >= {_QUALIFY_PLAYS}
        ),
        session_qs AS (
            SELECT COALESCE(qp.rolling_session_id, qp.push_id) AS session_id,
                   COUNT(*) AS qs_n
            FROM queue_skips qs
            JOIN queue_pushes qp ON qs.push_id = qp.push_id
            WHERE qp.algorithm = 'smartshuffle'
              AND qp.mode = 'rolling'
              AND qs.queue_position IS NOT NULL
            GROUP BY 1
        )
        SELECT qp.push_id, qp.pushed_at,
               COALESCE(TRIM(pl.playlist_name), 'Unknown') AS playlist_name,
               q.playlist_id,
               ss.total_queued,
               pc.plays_n,
               COALESCE(sqs.qs_n, 0) AS qs_n,
               ROUND(
                   (pc.play_skip_n + COALESCE(sqs.qs_n, 0)) * 1.0
                   / NULLIF(pc.plays_n + COALESCE(sqs.qs_n, 0), 0),
               3) AS skip_rate,
               sv.avg_content, sv.avg_melodic, sv.avg_bpm
        FROM queue_pushes qp
        JOIN queues q ON q.queue_id = qp.queue_id
        LEFT JOIN playlists pl     ON q.playlist_id  = pl.playlist_id
        LEFT JOIN session_sizes ss ON ss.session_id  = COALESCE(qp.rolling_session_id, qp.push_id)
        LEFT JOIN play_counts pc   ON pc.session_id  = COALESCE(qp.rolling_session_id, qp.push_id)
        LEFT JOIN session_qs sqs   ON sqs.session_id = COALESCE(qp.rolling_session_id, qp.push_id)
        LEFT JOIN session_vibes sv ON sv.session_id  = COALESCE(qp.rolling_session_id, qp.push_id)
        WHERE qp.push_id = COALESCE(qp.rolling_session_id, qp.push_id)
          AND qp.algorithm = 'smartshuffle'
          AND pc.plays_n IS NOT NULL
        ORDER BY qp.push_id DESC
        LIMIT 30
    """).fetchall()

    conn.close()

    history_out = []
    for r in history:
        row   = dict(r)
        pl_id = row.get("playlist_id")
        row["vibe_target"] = vibe_targets.get(pl_id) if pl_id else None
        history_out.append(row)

    return JSONResponse({
        "summary": {
            "total_plays":       total_plays,
            "scored_songs":      scored_songs,
            "rolling_sessions":  rolling_sessions,
            "avg_session_songs": avg_session_songs,
        },
        "rolling_skip_rate": rolling_skip_rate,
        "qs_n":              qs_n,
        "plays_n":           n,
        "trend":             [dict(r) for r in trend],
        "vibe_dists":        vibe_dists,
        "history":           history_out,
    })


# ── Queue dashboard helpers ───────────────────────────────────────────────────

_SESSION_VIBE_MIN_CREDIBLE = 2
_SESSION_VIBE_RAMP_OVER    = 10
_SESSION_VIBE_MAX_DRIFT    = 0.80
_SESSION_VIBE_RECENCY_AT   = 10
_SESSION_VIBE_RECENCY_W    = 0.85
_MELODIC_ASYM              = (1.4, 0.6)


def _time_bucket() -> str:
    h = datetime.now().hour
    if h >= 21 or h < 6: return "late_night"
    if h < 11: return "morning"
    return "afternoon"


def _compute_drift(base_target: dict, state: dict, time_bucket: str):
    if state.get("time_bucket") != time_bucket:
        return base_target, None
    n = int(state.get("session_vibe_n", 0))
    if n < _SESSION_VIBE_MIN_CREDIBLE:
        return base_target, None

    drift_w = min((n - _SESSION_VIBE_MIN_CREDIBLE) / _SESSION_VIBE_RAMP_OVER, 1.0) * _SESSION_VIBE_MAX_DRIFT
    means   = {ax: state.get(f"session_vibe_{ax}_mean") for ax in ("content", "melodic", "bpm")}
    recency = {ax: state.get(f"session_vibe_recent_{ax}") for ax in ("content", "melodic", "bpm")} \
              if n >= _SESSION_VIBE_RECENCY_AT else None

    eff_sess = {}
    for ax in ("content", "melodic", "bpm"):
        m = means[ax]
        if m is None:
            return base_target, None
        r = (recency or {}).get(ax)
        eff_sess[ax] = (_SESSION_VIBE_RECENCY_W * float(r) + (1 - _SESSION_VIBE_RECENCY_W) * float(m)) \
                       if r is not None else float(m)

    effective    = {ax: round((1 - drift_w) * base_target[ax] + drift_w * eff_sess[ax], 4)
                    for ax in ("content", "melodic", "bpm")}
    session_info = {
        "n":              n,
        "drift_w":        round(drift_w, 3),
        "means":          {ax: round(float(v), 4) for ax, v in means.items() if v is not None},
        "recency":        {ax: round(float(v), 4) for ax, v in recency.items() if v is not None}
                          if recency else None,
        "effective_sess": {ax: round(v, 4) for ax, v in eff_sess.items()},
    }
    return effective, session_info


def _song_explanations(songs: list, effective_target: dict | None,
                       vibe_sigmas: dict, weights: dict) -> list:
    out = []
    for i, s in enumerate(songs):
        vc, vm, vb = s.get("vibe_content"), s.get("vibe_melodic"), s.get("vibe_bpm")

        vibe_fit = vibe_match = None
        fatigue        = float(s.get("fatigue", 0))
        binge_score    = float(s.get("binge_score", 0))
        if effective_target and vc is not None and vm is not None and vb is not None:
            # Mirror recommend.py: binge songs are scored with widened sigmas.
            _binge_mult = 1.0 + max(float(binge_score), 0.0) * 0.50
            sc = vibe_sigmas["content"]  * _binge_mult
            sm = vibe_sigmas["melodic"]  * _binge_mult
            sb = vibe_sigmas["bpm"]      * _binge_mult
            gc = math.exp(-0.5 * (vc - effective_target["content"])**2 / sc**2)
            dm = vm - effective_target["melodic"]
            gm = math.exp(-0.5 * dm**2 / (sm * (_MELODIC_ASYM[0] if dm < 0 else _MELODIC_ASYM[1]))**2)
            gb = math.exp(-0.5 * (vb - effective_target["bpm"])**2 / sb**2)
            vibe_fit   = {"content": round(gc, 3), "melodic": round(gm, 3), "bpm": round(gb, 3)}
            vibe_match = round((gc + gm + gb) / 3.0, 3)
        art_fatigue    = float(s.get("artist_fatigue", 0))
        coverage_debt  = float(s.get("coverage_debt", 0))
        coverage_bonus = min(coverage_debt / 4.0, 1.0)

        wv  = float(weights.get("vibe_match",     0.4))
        wf  = float(weights.get("fatigue",        0.2067))
        wbb = float(weights.get("binge_boost",    0.15))
        wc  = float(weights.get("coverage",       0.1654))
        waf = float(weights.get("artist_fatigue", 0.0591))

        contributions = {
            "vibe":            round(wv  * (vibe_match or 0), 4),
            "fatigue_penalty": round(-wf * fatigue * (1 - max(binge_score, 0)), 4),
            "artist_fatigue":  round(-waf * art_fatigue * (1 - binge_score), 4),
            "binge_boost":     round(wbb * binge_score, 4),
            "coverage":        round(wc  * coverage_bonus, 4),
        }

        raw_score = s.get("score")
        try:
            score = float(raw_score) if raw_score is not None and math.isfinite(float(raw_score)) else None
        except (TypeError, ValueError):
            score = None

        out.append({
            "position":       i,
            "song_name":      s["song_name"],
            "artist_name":    s["artist_name"],
            "score":          score,
            "cluster_label":  s.get("cluster_label"),
            "vibe":           {"content": vc, "melodic": vm, "bpm": vb},
            "vibe_fit":       vibe_fit,
            "vibe_match":     vibe_match,
            "fatigue":        round(fatigue, 3),
            "binge_score":    round(binge_score, 3),
            "artist_fatigue": round(art_fatigue, 3),
            "coverage_debt":  round(coverage_debt, 3),
            "coverage_bonus": round(coverage_bonus, 3),
            "contributions":  contributions,
        })
    return out


# ── API: queue ────────────────────────────────────────────────────────────────

@app.get("/api/queue")
async def api_queue(user: dict = Depends(current_user)):
    user_id = user["user_id"]
    paths   = get_user_paths(user_id)

    rqs_path = paths["rolling_state"]
    try:
        with open(rqs_path) as f:
            rqs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"error": "No rolling queue state found"}, status_code=404)

    enabled         = rqs.get("enabled", False)
    playlist_id     = rqs.get("playlist_id")
    source_pl_id    = rqs.get("source_playlist_id")
    session_push_id = rqs.get("session_push_id")
    last_refill_at  = rqs.get("last_refill_at")

    tb = _time_bucket()

    vp_path = os.path.join(ROOT, "data", "vibe_params.json")
    try:
        with open(vp_path) as f:
            vp = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        vp = {}

    _vibe_sigmas_baseline = vp.get("vibe_sigmas", {}).get(tb, {"content": 0.30, "melodic": 0.30, "bpm": 0.30})
    base_target = _load_vibe_targets().get(source_pl_id)

    lp_path = os.path.join(ROOT, "data", "learned_params.json")
    try:
        with open(lp_path) as f:
            weights = json.load(f).get("weights", {})
    except (FileNotFoundError, json.JSONDecodeError):
        weights = {}

    effective_target = base_target
    session_info     = None
    recent_skip_rate = 0.0
    if base_target:
        try:
            with open(paths["session_state"]) as f:
                state = json.load(f)
            effective_target, session_info = _compute_drift(base_target, state, tb)
            if state.get("time_bucket") == tb and int(state.get("plays_seen", 0)) >= 3:
                recent_skip_rate = float(state.get("recent_skip_rate", 0.0))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # If the user dragged sliders to a custom target, score songs against that
    # target — not the learned/drifted one — so displayed vibe % reflects what
    # the recommender actually optimised for.
    target_override = rqs.get("target_override")
    if target_override:
        effective_target = target_override

    # Match the sigma tightening recommend.py applies so displayed vibe scores agree with the floor.
    _sigma_mult = max(0.0, min(1.0, (recent_skip_rate - 0.30) / (0.80 - 0.30)))
    _sigma_mult = round(1.0 - (1.0 - 0.50) * _sigma_mult, 3)
    vibe_sigmas = {ax: round(v * _sigma_mult, 4) for ax, v in _vibe_sigmas_baseline.items()}

    conn          = _db(user_id)
    songs_out     = []
    playlist_name = None
    push_at       = None
    try:
        if session_push_id:
            row = conn.execute("""
                SELECT qp.pushed_at, q.songs
                FROM queue_pushes qp
                JOIN queues q ON q.queue_id = qp.queue_id
                WHERE COALESCE(qp.rolling_session_id, qp.push_id) = ?
                ORDER BY qp.push_id DESC
                LIMIT 1
            """, (session_push_id,)).fetchone()
            if row:
                push_at   = row["pushed_at"]
                raw_songs = json.loads(row["songs"])
                songs_out = _song_explanations(raw_songs, effective_target, vibe_sigmas, weights)

        if source_pl_id:
            r = conn.execute(
                "SELECT playlist_name FROM playlists WHERE playlist_id = ?",
                (source_pl_id,)
            ).fetchone()
            if r:
                playlist_name = r["playlist_name"].strip()

        pool_vibes = []
        if source_pl_id:
            pv_rows = conn.execute("""
                SELECT s.vibe_content AS c, s.vibe_melodic AS m, s.vibe_bpm AS b
                FROM playlist_tracks pt
                JOIN songs s ON s.song_id = pt.song_id
                WHERE pt.playlist_id = ?
                  AND s.vibe_content IS NOT NULL
                  AND s.vibe_melodic IS NOT NULL
                  AND s.vibe_bpm IS NOT NULL
            """, (source_pl_id,)).fetchall()
            pool_vibes = [[round(r["c"], 3), round(r["m"], 3), round(r["b"], 3)] for r in pv_rows]
    finally:
        conn.close()

    return JSONResponse({
        "enabled":            enabled,
        "last_refill_at":     last_refill_at or push_at,
        "playlist_name":      playlist_name,
        "playlist_id":        playlist_id,
        "source_playlist_id": source_pl_id,
        "session_push_id":    session_push_id,
        "time_bucket":        tb,
        "base_target":        base_target,
        "session":            session_info,
        "effective_target":   effective_target,
        "vibe_sigmas":        vibe_sigmas,
        "songs":              songs_out,
        "pool_vibes":         pool_vibes,
        "target_override":    rqs.get("target_override"),
    })


@app.get("/api/queue/devices")
async def api_queue_devices(user: dict = Depends(current_user)):
    try:
        data    = _sp(user["user_id"]).devices()
        devices = data.get("devices", [])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse([
        {"device_id": d["id"], "name": d["name"], "type": d["type"], "is_active": d["is_active"]}
        for d in devices
    ])


@app.get("/api/queue/playlists")
async def api_queue_playlists(user: dict = Depends(current_user)):
    vp_path = os.path.join(ROOT, "data", "vibe_params.json")
    try:
        with open(vp_path) as f:
            scored_ids = set(json.load(f).get("playlist_targets", {}).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        scored_ids = set()

    vibe_targets = _load_vibe_targets()

    conn = _db(user["user_id"])
    rows = conn.execute("""
        SELECT p.playlist_id, TRIM(p.playlist_name) AS playlist_name,
               COUNT(pt.song_id) AS n_tracks
        FROM playlists p
        LEFT JOIN playlist_tracks pt ON p.playlist_id = pt.playlist_id
        GROUP BY p.playlist_id
        HAVING n_tracks > 10
        ORDER BY n_tracks DESC
    """).fetchall()
    conn.close()

    excluded_names = {"smartshuffle queue"}
    return JSONResponse([
        {
            "playlist_id":   r["playlist_id"],
            "playlist_name": r["playlist_name"],
            "n_tracks":      r["n_tracks"],
            "vibe_target":   vibe_targets.get(r["playlist_id"]),
        }
        for r in rows
        if r["playlist_id"] in scored_ids
        and r["playlist_name"].lower() not in excluded_names
    ])


@app.post("/api/queue/retarget")
async def api_queue_retarget(request: Request, user: dict = Depends(current_user)):
    user_id = user["user_id"]
    body    = await request.json()
    try:
        new_target = {
            "content": float(body["content"]),
            "melodic": float(body["melodic"]),
            "bpm":     float(body["bpm"]),
        }
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"error": "content, melodic, bpm required as floats"}, status_code=400)

    paths    = get_user_paths(user_id)
    rqs_path = paths["rolling_state"]
    try:
        with open(rqs_path) as f:
            rqs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"error": "No active rolling session"}, status_code=404)

    source_pl_id = rqs.get("source_playlist_id")
    device_id    = rqs.get("device_id") or ""

    vp_path = os.path.join(ROOT, "data", "vibe_params.json")
    try:
        with open(vp_path) as f:
            vibe_sigmas = json.load(f).get("vibe_sigmas", {}).get(_time_bucket(),
                          {"content": 0.30, "melodic": 0.30, "bpm": 0.30})
    except (FileNotFoundError, json.JSONDecodeError):
        vibe_sigmas = {"content": 0.30, "melodic": 0.30, "bpm": 0.30}

    conn = _db(user_id)
    try:
        rows = conn.execute("""
            SELECT s.vibe_content, s.vibe_melodic, s.vibe_bpm
            FROM playlist_tracks pt
            JOIN songs s ON s.song_id = pt.song_id
            WHERE pt.playlist_id = ?
              AND s.vibe_content IS NOT NULL
              AND s.vibe_melodic IS NOT NULL
              AND s.vibe_bpm IS NOT NULL
        """, (source_pl_id,)).fetchall()
    finally:
        conn.close()

    def _avg_fit(vc, vm, vb):
        sc, sm, sb = vibe_sigmas["content"], vibe_sigmas["melodic"], vibe_sigmas["bpm"]
        gc = math.exp(-0.5 * (vc - new_target["content"])**2 / sc**2)
        dm = vm - new_target["melodic"]
        gm = math.exp(-0.5 * dm**2 / (sm * (_MELODIC_ASYM[0] if dm < 0 else _MELODIC_ASYM[1]))**2)
        gb = math.exp(-0.5 * (vb - new_target["bpm"])**2 / sb**2)
        return (gc + gm + gb) / 3.0

    fitting = sum(1 for r in rows if _avg_fit(r[0], r[1], r[2]) >= 0.25)
    if fitting < 10:
        return JSONResponse({
            "error": f"Only {fitting} songs fit this target (need 10). Move the target closer to the playlist centre."
        }, status_code=422)

    rqs["target_override"] = new_target
    tmp = rqs_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rqs, f)
    os.replace(tmp, rqs_path)

    src_dir    = os.path.join(ROOT, "src")
    target_str = f"{new_target['content']},{new_target['melodic']},{new_target['bpm']}"
    env        = _subprocess_env(user_id)

    if not validate_playlist_id(source_pl_id or ""):
        return JSONResponse({"error": "Invalid playlist ID"}, status_code=400)

    try:
        r1 = subprocess.run(
            [sys.executable, os.path.join(src_dir, "recommend.py"),
             "--playlist", source_pl_id, "--count", "10", "--target", target_str],
            capture_output=True, text=True, timeout=120, cwd=ROOT, env=env,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "recommend.py timed out"}, status_code=500)
    if r1.returncode != 0:
        return JSONResponse({"error": "Queue generation failed",
                             "detail": (r1.stdout + r1.stderr)[-2000:]}, status_code=500)

    push_cmd = [sys.executable, os.path.join(src_dir, "push.py"), "--rolling"]
    if device_id and validate_device_id(device_id):
        push_cmd += ["--device-id", device_id]
    try:
        r2 = subprocess.run(push_cmd, capture_output=True, text=True, timeout=60, cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "push.py timed out"}, status_code=500)
    if r2.returncode != 0:
        return JSONResponse({"error": "Spotify push failed",
                             "detail": (r2.stdout + r2.stderr)[-2000:]}, status_code=500)

    return JSONResponse({"ok": True, "fitting_songs": fitting, "target": new_target})


@app.post("/api/queue/start")
async def api_queue_start(request: Request, user: dict = Depends(current_user)):
    user_id     = user["user_id"]
    body        = await request.json()
    playlist_id = (body.get("playlist_id") or "").strip()
    device_id   = (body.get("device_id")   or "").strip()
    vibe_target = body.get("vibe_target")   # optional {content, melodic, bpm}

    if not playlist_id:
        return JSONResponse({"error": "playlist_id required"}, status_code=400)
    if not validate_playlist_id(playlist_id):
        return JSONResponse({"error": "Invalid playlist ID"}, status_code=400)

    src_dir = os.path.join(ROOT, "src")
    env     = _subprocess_env(user_id)

    # Clear stale session state so recommend.py starts fresh with no carry-over
    # drift from the previous session — otherwise the floor target is shifted and
    # songs that barely pass the old drifted floor appear below 75% on display.
    paths = get_user_paths(user_id)
    try:
        tmp = paths["session_state"] + ".tmp"
        with open(tmp, "w") as f:
            json.dump({}, f)
        os.replace(tmp, paths["session_state"])
    except OSError:
        pass

    rec_cmd = [sys.executable, os.path.join(src_dir, "recommend.py"),
               "--playlist", playlist_id, "--count", "10"]
    if vibe_target:
        try:
            c = float(vibe_target["content"])
            m = float(vibe_target["melodic"])
            b = float(vibe_target["bpm"])
            rec_cmd += ["--target", f"{c},{m},{b}"]
        except (KeyError, TypeError, ValueError):
            pass

    try:
        r1 = subprocess.run(rec_cmd, capture_output=True, text=True, timeout=120, cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "recommend.py timed out"}, status_code=500)
    if r1.returncode != 0:
        return JSONResponse({"error": "Queue generation failed"}, status_code=500)

    push_cmd = [sys.executable, os.path.join(src_dir, "push.py"), "--rolling"]
    if device_id and validate_device_id(device_id):
        push_cmd += ["--device-id", device_id]
    try:
        r2 = subprocess.run(push_cmd, capture_output=True, text=True, timeout=60, cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "push.py timed out"}, status_code=500)
    if r2.returncode != 0:
        return JSONResponse({"error": "Spotify push failed"}, status_code=500)

    # Persist or clear target_override in rolling state.
    paths    = get_user_paths(user_id)
    rqs_path = paths["rolling_state"]
    try:
        with open(rqs_path) as f:
            rqs = json.load(f)
        if vibe_target:
            rqs["target_override"] = {"content": float(vibe_target["content"]),
                                      "melodic": float(vibe_target["melodic"]),
                                      "bpm":     float(vibe_target["bpm"])}
        else:
            rqs.pop("target_override", None)
        tmp = rqs_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rqs, f)
        os.replace(tmp, rqs_path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass

    return JSONResponse({"ok": True, "stdout": r2.stdout[-1000:]})
