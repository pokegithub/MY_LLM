"""Targeted context builder for the Phase 1 coding agent."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from agent.types import ContextBundle, ContextFile, TaskRequest
from security.validator import validate_local_path


def _excerpt_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + "\n...\n" + tail


def _resolve_hint(workspace_root: str, hint: str) -> Path:
    path = Path(hint)
    if not path.is_absolute():
        path = Path(workspace_root) / path
    return Path(
        validate_local_path(path, allowed_roots=(workspace_root,), must_exist=False)
    )


def _infer_related_tests(workspace_root: str, file_hints: Sequence[str]) -> List[Path]:
    tests_dir = Path(workspace_root) / "tests"
    if not tests_dir.is_dir():
        return []
    stems = {Path(item).stem.lower() for item in file_hints if Path(item).stem}
    related: List[Path] = []
    for candidate in tests_dir.rglob("test_*.py"):
        lowered = candidate.name.lower()
        if any(stem and stem in lowered for stem in stems):
            related.append(candidate.resolve())
    return related[:3]


def build_context(
    request: TaskRequest,
    *,
    max_file_chars: int = 4000,
    include_system_map: bool = False,
) -> ContextBundle:
    workspace_root = str(Path(request.workspace_root).resolve())
    considered: List[str] = []
    files: List[ContextFile] = []
    notes: List[str] = []

    for hint in request.file_hints:
        resolved = _resolve_hint(workspace_root, hint)
        considered.append(str(resolved))
        if resolved.exists() and resolved.is_file():
            raw = resolved.read_text(encoding="utf-8")
            files.append(
                ContextFile(
                    path=str(resolved),
                    exists=True,
                    included=True,
                    size_bytes=resolved.stat().st_size,
                    content_excerpt=_excerpt_text(raw, max_file_chars),
                    note="explicit file hint",
                )
            )
        else:
            files.append(
                ContextFile(
                    path=str(resolved),
                    exists=False,
                    included=False,
                    size_bytes=0,
                    note="explicit file hint not found",
                )
            )

    related_tests = _infer_related_tests(workspace_root, request.file_hints)
    for test_path in related_tests:
        considered.append(str(test_path))
        raw = test_path.read_text(encoding="utf-8")
        files.append(
            ContextFile(
                path=str(test_path),
                exists=True,
                included=True,
                size_bytes=test_path.stat().st_size,
                content_excerpt=_excerpt_text(raw, max_file_chars),
                note="related test inferred from file hint",
            )
        )

    if include_system_map:
        system_map = Path(workspace_root) / "SYSTEM_MAP.md"
        if system_map.is_file():
            considered.append(str(system_map.resolve()))
            raw = system_map.read_text(encoding="utf-8")
            files.append(
                ContextFile(
                    path=str(system_map.resolve()),
                    exists=True,
                    included=True,
                    size_bytes=system_map.stat().st_size,
                    content_excerpt=_excerpt_text(raw, max_file_chars),
                    note="system map requested for architecture-sensitive task",
                )
            )

    if not files:
        notes.append("No targeted files were included; plan must remain conservative.")

    return ContextBundle(
        workspace_root=workspace_root,
        files_considered=tuple(considered),
        files=tuple(files),
        notes=tuple(notes),
    )
