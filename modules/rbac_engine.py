"""
Role-Based Access Control (RBAC) Engine
========================================

Maps a face-recognition result to an access decision.

Authorized user roles
---------------------
admin       → ALLOW   (full access; gold box)
authorized  → ALLOW   (standard access; green box)
unknown     → DENY    (no match or confidence below threshold; grey box)

Blacklisted individuals are handled separately via make_blacklist_decision()
before this engine is ever called.  They are threat entities stored in the
blacklist table, not valid system users.

Security modes — adjustable confidence threshold
------------------------------------------------
normal   ≥ 0.25  (standard operation)
relaxed  ≥ 0.12  (permissive; demo / low-risk environments)
"""

THRESHOLDS = {
    "normal":  0.55,
    "relaxed": 0.50,
}

# BGR colours for bounding boxes
ROLE_COLORS = {
    "admin":       (0, 215, 255),    # gold
    "authorized":  (0, 200, 0),      # green
    "blacklisted": (0, 0, 220),      # red  (used by make_blacklist_decision)
    "unknown":     (160, 160, 160),  # grey
    "spoof":       (0, 80, 255),     # deep orange
}

ROLE_ACTIONS = {
    "admin":      "ALLOW",
    "authorized": "ALLOW",
}


def make_decision(match: dict | None, confidence: float,
                  spoof_passed: bool = True,
                  security_mode: str = "normal") -> dict:
    """
    Produce an RBAC access decision for one detected face.

    Args:
        match:          User dict from the database (or None if unrecognised).
        confidence:     Float in [0, 1] — higher is more certain.
        spoof_passed:   False if anti-spoofing flagged this face as fake.
        security_mode:  'normal' or 'relaxed'.

    Returns dict:
        action   — 'ALLOW', 'DENY', or 'ALERT'
        reason   — human-readable explanation
        role     — effective role string
        color    — BGR tuple for bounding-box colour
        label    — single-line overlay string
    """
    # ── Anti-spoofing gate ────────────────────────────────────────────────────
    if not spoof_passed:
        return {
            "action": "DENY",
            "reason": "Anti-spoofing check failed — possible printed/replay attack",
            "role":   "spoof",
            "color":  ROLE_COLORS["spoof"],
            "label":  "SPOOF DETECTED",
        }

    threshold = THRESHOLDS.get(security_mode, 0.40)

    # ── Low-confidence or unrecognised ────────────────────────────────────────
    if match is None or confidence < threshold:
        reason = (
            "No match found in database"
            if match is None
            else f"Confidence {confidence:.0%} < threshold {threshold:.0%} ({security_mode})"
        )
        label = f"Unknown  ({confidence:.0%})" if confidence > 0 else "Unknown"
        return {
            "action": "DENY",
            "reason": reason,
            "role":   "unknown",
            "color":  ROLE_COLORS["unknown"],
            "label":  label,
        }

    # ── Known authorized/admin user ───────────────────────────────────────────
    role   = match["role"]
    action = ROLE_ACTIONS.get(role, "DENY")
    color  = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])

    badge  = " [ADMIN]" if role == "admin" else ""
    label  = f"{match['name']}  {confidence:.0%}{badge}"

    reason = (
        f"Role: {role} | Confidence: {confidence:.0%} | "
        f"Mode: {security_mode} | Threshold: {threshold:.0%}"
    )

    return {
        "action":   action,
        "reason":   reason,
        "role":     role,
        "color":    color,
        "label":    label,
        "name":     match["name"],
        "username": match["username"],
    }


def make_blacklist_decision(entry: dict) -> dict:
    """
    Produce an ALERT decision for a face that matched the blacklist.
    Called before any authorized-user RBAC logic — processing stops here.
    """
    name   = entry.get("name", "Unknown")
    threat = entry.get("threat_reason", "Security threat")
    return {
        "action":   "ALERT",
        "reason":   f"BLACKLISTED INDIVIDUAL — {threat}",
        "role":     "blacklisted",
        "color":    ROLE_COLORS["blacklisted"],
        "label":    f"{name}  [BLACKLISTED]",
        "name":     name,
        "username": "",
    }
