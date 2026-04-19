ATTACK_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "bypass safety",
    "jailbreak",
]


def detect_prompt_attack(prompt: str) -> bool:
    low = prompt.lower()
    return any(p in low for p in ATTACK_PATTERNS)
