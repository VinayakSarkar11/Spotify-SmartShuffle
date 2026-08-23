"""Authentication helpers: OAuth flow, encrypted token storage, session dependency."""
import base64
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import requests as http
from cryptography.fernet import Fernet
from fastapi import HTTPException, Request

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
AUTH_DB  = os.path.join(DATA_DIR, "auth.db")

SPOTIFY_SCOPES = " ".join([
    "user-read-recently-played",
    "user-library-read",
    "playlist-read-private",
    "playlist-modify-private",
    "user-top-read",
    "user-modify-playback-state",
    "user-read-playback-state",
])

_PLAYLIST_RE = re.compile(r'^[A-Za-z0-9]{22}$')
_DEVICE_RE   = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')
_USER_ID_RE  = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')


def validate_playlist_id(s: str) -> bool:
    return bool(s and _PLAYLIST_RE.match(s))


def validate_device_id(s: str) -> bool:
    return bool(s and _DEVICE_RE.match(s))


# ── Encryption ────────────────────────────────────────────────────────────────

def _fernet() -> Fernet:
    key = os.getenv("FERNET_KEY", "")
    if not key:
        raise RuntimeError(
            "FERNET_KEY is not set. Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            "and add it to .env"
        )
    return Fernet(key.encode())


def encrypt_token(token: str) -> str:
    return _fernet().encrypt(token.encode()).decode()


def decrypt_token(enc: str) -> str:
    return _fernet().decrypt(enc.encode()).decode()


# ── Auth database ─────────────────────────────────────────────────────────────

def _auth_conn() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id                  TEXT PRIMARY KEY,
            display_name             TEXT,
            encrypted_access_token   TEXT NOT NULL,
            encrypted_refresh_token  TEXT NOT NULL,
            token_expires_at         TEXT NOT NULL,
            token_scope              TEXT,
            created_at               TEXT NOT NULL,
            last_login_at            TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def upsert_user(user_id: str, display_name: str, access_token: str,
                refresh_token: str, expires_in: int, scope: str) -> None:
    now        = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
    conn = _auth_conn()
    conn.execute("""
        INSERT INTO users
            (user_id, display_name, encrypted_access_token, encrypted_refresh_token,
             token_expires_at, token_scope, created_at, last_login_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            display_name            = excluded.display_name,
            encrypted_access_token  = excluded.encrypted_access_token,
            encrypted_refresh_token = excluded.encrypted_refresh_token,
            token_expires_at        = excluded.token_expires_at,
            token_scope             = excluded.token_scope,
            last_login_at           = excluded.last_login_at
    """, (user_id, display_name,
          encrypt_token(access_token), encrypt_token(refresh_token),
          expires_at, scope, now, now))
    conn.commit()
    conn.close()


def delete_user(user_id: str) -> None:
    conn = _auth_conn()
    conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def _owner_token_from_cache() -> str | None:
    """Fall back to the SpotifyOAuth .spotify_cache when the owner hasn't used the web login yet."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        auth_manager = SpotifyOAuth(
            client_id     = os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret = os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI"),
            scope         = SPOTIFY_SCOPES,
            cache_path    = os.path.join(ROOT, ".spotify_cache"),
            open_browser  = False,
        )
        token_info = auth_manager.get_cached_token()
        if not token_info:
            return None
        if auth_manager.is_token_expired(token_info):
            token_info = auth_manager.refresh_access_token(token_info["refresh_token"])
        return token_info["access_token"] if token_info else None
    except Exception:
        return None


def get_access_token(user_id: str) -> str:
    """Return a valid access token, refreshing automatically when < 5 min from expiry."""
    conn = _auth_conn()
    row  = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        owner_id = os.getenv("SPOTIFY_OWNER_ID", "")
        if user_id == owner_id:
            token = _owner_token_from_cache()
            if token:
                return token
        raise ValueError(f"User {user_id!r} not found in auth DB")

    expires_at = datetime.fromisoformat(row["token_expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        token = decrypt_token(row["encrypted_access_token"])
        conn.close()
        return token

    # Token expiring soon — refresh it
    refresh_token = decrypt_token(row["encrypted_refresh_token"])
    client_id     = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    creds_b64     = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    resp = http.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        headers={"Authorization": f"Basic {creds_b64}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    new_access     = data["access_token"]
    new_refresh    = data.get("refresh_token", refresh_token)  # Spotify may rotate it
    new_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))
    ).isoformat()

    conn.execute("""
        UPDATE users SET
            encrypted_access_token  = ?,
            encrypted_refresh_token = ?,
            token_expires_at        = ?
        WHERE user_id = ?
    """, (encrypt_token(new_access), encrypt_token(new_refresh), new_expires_at, user_id))
    conn.commit()
    conn.close()
    return new_access


# ── Per-user filesystem paths ─────────────────────────────────────────────────

def get_user_paths(user_id: str) -> dict:
    """Return {db, rolling_state, session_state} paths scoped to this user.

    The owner (SPOTIFY_OWNER_ID) maps to the existing data files so existing
    data is preserved without migration.
    """
    owner_id = os.getenv("SPOTIFY_OWNER_ID", "")
    if user_id == owner_id:
        return {
            "db":            os.path.join(DATA_DIR, "smartshuffle.db"),
            "rolling_state": os.path.join(DATA_DIR, "rolling_queue_state.json"),
            "session_state": os.path.join(DATA_DIR, "session_state.json"),
        }
    if not _USER_ID_RE.match(user_id):
        raise ValueError(f"Invalid user_id: {user_id!r}")
    user_dir = os.path.join(DATA_DIR, "users", user_id)
    os.makedirs(user_dir, exist_ok=True)
    return {
        "db":            os.path.join(user_dir, "smartshuffle.db"),
        "rolling_state": os.path.join(user_dir, "rolling_queue_state.json"),
        "session_state": os.path.join(user_dir, "session_state.json"),
    }


def spotify_for_user(user_id: str):
    """Return a spotipy.Spotify client authenticated as this user."""
    import spotipy
    return spotipy.Spotify(auth=get_access_token(user_id))


def write_spotipy_cache(user_id: str, cache_path: str) -> None:
    """Write a SpotifyOAuth-format token cache so the watcher can auto-refresh.

    The watcher subprocess strips SS_ACCESS_TOKEN (short-lived) and falls back
    to SpotifyOAuth; it needs a cache file with the refresh_token so it can
    obtain new access tokens without an interactive OAuth flow.
    """
    import json, time as _time
    conn = _auth_conn()
    row  = conn.execute(
        "SELECT encrypted_access_token, encrypted_refresh_token, token_expires_at, token_scope "
        "FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return
    expires_at = datetime.fromisoformat(row["token_expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    cache = {
        "access_token":  decrypt_token(row["encrypted_access_token"]),
        "token_type":    "Bearer",
        "expires_in":    3600,
        "refresh_token": decrypt_token(row["encrypted_refresh_token"]),
        "scope":         row["token_scope"] or "",
        "expires_at":    expires_at.timestamp(),
    }
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)


# ── FastAPI session dependency ────────────────────────────────────────────────

def current_user(request: Request) -> dict:
    """Dependency — returns {user_id, display_name} or raises 401."""
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    return {
        "user_id":      user_id,
        "display_name": request.session.get("display_name", ""),
    }
