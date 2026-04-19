"""Minimal serving boundary with fail-closed model loading semantics."""

from typing import Protocol

from safety.policy_filters import sanitize_output
from safety.prompt_attack_checks import detect_prompt_attack


class ServingError(RuntimeError):
    """Raised when serving cannot produce a valid model response."""


class ModelBackend(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate a response for a validated prompt."""


class LLMService:
    def __init__(self, backend: ModelBackend | None = None):
        self.backend = backend

    def handle_request(self, prompt: str) -> dict:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if detect_prompt_attack(prompt):
            return {"ok": False, "error": "prompt blocked"}
        if self.backend is None:
            return {"ok": False, "error": "model backend not loaded"}

        response = self.backend.generate(prompt)
        if not isinstance(response, str):
            raise ServingError("model backend returned a non-string response")
        return {"ok": True, "response": sanitize_output(response)}


_service = LLMService()


def configure_backend(backend: ModelBackend | None) -> None:
    global _service
    _service = LLMService(backend)


def handle_request(prompt: str) -> dict:
    return _service.handle_request(prompt)


if __name__ == "__main__":
    print(handle_request("hello"))
