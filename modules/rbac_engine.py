"""
Role-Based Access Control (RBAC) Engine
========================================

Maps a face-recognition result to an access decision.

Roles
-----
admin       → ALLOW   (full access; highlighted gold box)
authorized  → ALLOW   (standard access; green box)
blacklisted → ALERT   (denied + security alert; red box + banner)
unknown     → DENY    (no match or confidence below threshold; grey box)

Security modes — adjustable confidence threshold
------------------------------------------------
strict   ≥ 0.60  (high-security; guards against impersonation)
normal   ≥ 0.40  (standard operation)
relaxed  ≥ 0.25  (permissive; demo / low-risk environments)
"""

THRESHOLDS = {
    "strict":  0.40,
    "normal":  0.25,
    "relaxed": 0.12,
}

# BGR colours for bounding boxes
ROLE_COLORS = {
    "admin":       (0, 215, 255),    # gold
    "authorized":  (0, 200, 0),      # green
    "blacklisted": (0, 0, 220),      # red
    "unknown":     (160, 160, 160),  # grey
    "spoof":       (0, 80, 255),     # deep orange
}

ROLE_ACTIONS = {
    "admin":       "ALLOW",
    "authorized":  "ALLOW",
    "blacklisted": "ALERT",
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
        security_mode:  'strict', 'normal', or 'relaxed'.

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

    # ── Known user ────────────────────────────────────────────────────────────
    role   = match["role"]
    action = ROLE_ACTIONS.get(role, "DENY")
    color  = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])

    badge = {"admin": " [ADMIN]", "blacklisted": " [BLACKLISTED]"}.get(role, "")
    label = f"{match['name']}  {confidence:.0%}{badge}"

    reason = (
        f"Role: {role} | Confidence: {confidence:.0%} | "
        f"Mode: {security_mode} | Threshold: {threshold:.0%}"
    )

    return {
        "action": action,
        "reason": reason,
        "role":   role,
        "color":  color,
        "label":  label,
        "name":   match["name"],
        "username": match["username"],
    }
