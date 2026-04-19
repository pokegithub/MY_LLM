import re

BLOCKLIST = [
    r"\bpassword\b",
    r"\bapi[_-]?key\b",
    r"\bcredit card\b",
]


def sanitize_output(text: str) -> str:
    out = text
    for pat in BLOCKLIST:
        out = re.sub(pat, "[REDACTED]", out, flags=re.IGNORECASE)
    return out


def is_policy_safe(text: str) -> bool:
    low = text.lower()
    return not any(re.search(p, low) for p in BLOCKLIST)
