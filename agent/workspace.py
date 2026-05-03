"""Reversible workspace edit session for Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agent.types import Candidate
from security.validator import validate_local_path


class WorkspaceEditSession:
    def __init__(self, workspace_root: str):
        self.workspace_root = str(Path(workspace_root).resolve())
        self._snapshots: Dict[str, Optional[bytes]] = {}
        self._touched: List[str] = []
        self._committed = False

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = Path(self.workspace_root) / candidate
        resolved = validate_local_path(
            candidate,
            allowed_roots=(self.workspace_root,),
            must_exist=False,
        )
        return Path(resolved)

    def _snapshot(self, path: Path) -> None:
        key = str(path)
        if key in self._snapshots:
            return
        self._snapshots[key] = path.read_bytes() if path.exists() else None

    def apply_candidate(self, candidate: Candidate) -> List[str]:
        touched: List[str] = []
        for edit in candidate.edits:
            path = self._resolve_path(edit.path)
            self._snapshot(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit.new_content, encoding="utf-8")
            touched.append(str(path))
        self._touched = touched
        return touched

    def rollback(self) -> None:
        for path_str, original in self._snapshots.items():
            path = Path(path_str)
            if original is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
        self._committed = False

    def commit(self) -> None:
        self._committed = True

    @property
    def touched_files(self) -> List[str]:
        return list(self._touched)
