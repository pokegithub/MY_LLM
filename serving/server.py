"""Minimal serving stub for production extension."""

from safety.prompt_attack_checks import detect_prompt_attack
from safety.policy_filters import sanitize_output


def handle_request(prompt: str) -> dict:
    if detect_prompt_attack(prompt):
        return {"ok": False, "error": "prompt blocked"}

    # Placeholder response path for integration testing.
    response = f"Echo: {prompt[:200]}"
    response = sanitize_output(response)
    return {"ok": True, "response": response}


if __name__ == "__main__":
    print(handle_request("hello"))
