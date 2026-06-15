"""
SQLite database layer for the Smart Access Control System.

Schema
------
users       — identity, hashed credentials, role, averaged face encoding
access_log  — timestamped record of every recognition event
"""

import hashlib
import os
import pickle
import secrets
import sqlite3
from datetime import datetime

import numpy as np

DB_DIR  = "database"
DB_PATH = os.path.join(DB_DIR, "access_control.db")

ROLES = ("admin", "authorized", "blacklisted")


# ── Connection ────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables on first run. Safe to call on every startup."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'authorized'
                            CHECK(role IN ('admin', 'authorized', 'blacklisted')),
            face_encoding BLOB,
            created_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS access_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    DEFAULT (datetime('now')),
            detected_name TEXT,
            username      TEXT,
            role          TEXT,
            confidence    REAL,
            action        TEXT,
            reason        TEXT,
            tailgating    INTEGER DEFAULT 0,
            source        TEXT    DEFAULT 'image'
        );
    """)
    conn.commit()
    conn.close()


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


# ── User management ───────────────────────────────────────────────────────────

def create_user(name: str, username: str, password: str,
                role: str, face_encoding=None) -> tuple[bool, str]:
    """Insert a new user. Returns (success, message)."""
    if role not in ROLES:
        return False, f"Invalid role '{role}'. Choose from: {', '.join(ROLES)}"

    salt     = secrets.token_hex(16)
    pw_hash  = _hash_password(password, salt)
    enc_blob = pickle.dumps(np.array(face_encoding)) if face_encoding is not None else None

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (name, username, password_hash, salt, role, face_encoding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name.strip(), username.strip(), pw_hash, salt, role, enc_blob),
        )
        conn.commit()
        return True, f"User '{username}' registered with role '{role}'."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already taken."
    finally:
        conn.close()


def update_face_encoding(username: str, face_encoding) -> None:
    enc_blob = pickle.dumps(np.array(face_encoding))
    conn = _connect()
    conn.execute("UPDATE users SET face_encoding=? WHERE username=?", (enc_blob, username))
    conn.commit()
    conn.close()


def verify_credentials(username: str, password: str):
    """Return user dict if credentials are valid, else None."""
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if row is None:
        return None
    if _hash_password(password, row["salt"]) == row["password_hash"]:
        return dict(row)
    return None


def get_all_users() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, name, username, role, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_by_username(username: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_role(username: str, new_role: str) -> None:
    conn = _connect()
    conn.execute("UPDATE users SET role=? WHERE username=?", (new_role, username))
    conn.commit()
    conn.close()


def delete_user(username: str) -> None:
    conn = _connect()
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()


def get_all_face_encodings() -> list[dict]:
    """Return [{username, name, role, encoding}] for every user with a stored encoding."""
    conn = _connect()
    rows = conn.execute(
        "SELECT username, name, role, face_encoding FROM users "
        "WHERE face_encoding IS NOT NULL"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        enc = pickle.loads(r["face_encoding"])
        result.append({
            "username": r["username"],
            "name":     r["name"],
            "role":     r["role"],
            "encoding": enc,
        })
    return result


def user_count() -> int:
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n


# ── Access log ────────────────────────────────────────────────────────────────

def log_access_event(detected_name: str, username: str, role: str,
                     confidence: float, action: str, reason: str = "",
                     tailgating: bool = False, source: str = "image") -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO access_log "
        "(detected_name, username, role, confidence, action, reason, tailgating, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (detected_name, username, role, round(confidence, 4),
         action, reason, int(tailgating), source),
    )
    conn.commit()
    conn.close()


def get_access_log(limit: int = 500):
    import pandas as pd
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def clear_access_log() -> None:
    conn = _connect()
    conn.execute("DELETE FROM access_log")
    conn.commit()
    conn.close()


def get_log_stats() -> dict:
    conn = _connect()
    rows = conn.execute(
        "SELECT action, COUNT(*) as cnt FROM access_log GROUP BY action"
    ).fetchall()
    conn.close()
    stats = {r["action"]: r["cnt"] for r in rows}
    return {
        "ALLOW": stats.get("ALLOW", 0),
        "DENY":  stats.get("DENY",  0),
        "ALERT": stats.get("ALERT", 0),
    }
