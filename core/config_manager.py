"""Hierarchical YAML/JSON config manager with env overrides."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

from security.validator import get_allowed_data_roots, safe_load_json, validate_local_path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _deep_merge(base: MutableMapping[str, Any], incoming: Dict[str, Any]) -> None:
    for key, value in incoming.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _parse_env_value(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


@dataclass(frozen=True)
class ConfigSnapshot:
    data: Dict[str, Any]
    sources: List[str]
    config_hash: str
    created_at: float


class ConfigManager:
    """Load layered config files and optional environment overrides."""

    def __init__(self, env_prefix: str = "MYLLM__"):
        self.env_prefix = env_prefix

    def load(
        self,
        config_paths: Optional[Sequence[str]] = None,
        include_env: bool = True,
    ) -> ConfigSnapshot:
        merged: Dict[str, Any] = {}
        sources: List[str] = []

        for path in config_paths or []:
            loaded = self._load_file(path)
            _deep_merge(merged, loaded)
            sources.append(str(Path(path).resolve()))

        if include_env:
            env_cfg = self._collect_env_overrides()
            if env_cfg:
                _deep_merge(merged, env_cfg)
                sources.append("env")

        config_hash = hashlib.sha256(
            json.dumps(merged, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return ConfigSnapshot(
            data=merged,
            sources=sources,
            config_hash=config_hash,
            created_at=time.time(),
        )

    def _load_file(self, path: str) -> Dict[str, Any]:
        cfg_path = validate_local_path(
            path,
            must_exist=True,
            expected_kind="file",
            allowed_roots=get_allowed_data_roots(),
        )
        max_config_bytes = int(os.environ.get("MYLLM_MAX_CONFIG_BYTES", "5000000"))
        if cfg_path.stat().st_size > max_config_bytes:
            raise ValueError(
                f"Config file exceeds size limit ({max_config_bytes} bytes): {cfg_path}"
            )

        suffix = cfg_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required for YAML configs. Install pyyaml."
                )
            with open(cfg_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        elif suffix == ".json":
            payload = safe_load_json(
                cfg_path,
                max_bytes=max_config_bytes,
                allowed_roots=get_allowed_data_roots(),
            )
        else:
            raise ValueError(
                f"Unsupported config extension: {cfg_path.suffix}"
            )

        if not isinstance(payload, dict):
            raise ValueError(
                f"Config root must be a dict, got {type(payload)!r}"
            )
        return payload

    def _collect_env_overrides(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue

            path = key[len(self.env_prefix):]
            if not path:
                continue

            sections = [segment.lower() for segment in path.split("__") if segment]
            if not sections:
                continue

            node = out
            for section in sections[:-1]:
                existing = node.get(section)
                if not isinstance(existing, dict):
                    node[section] = {}
                node = node[section]

            node[sections[-1]] = _parse_env_value(value)
        return out

    @staticmethod
    def get(snapshot: ConfigSnapshot, dotted_path: str, default: Any = None) -> Any:
        node: Any = snapshot.data
        for segment in dotted_path.split("."):
            if not isinstance(node, dict) or segment not in node:
                return default
            node = node[segment]
        return node

    @staticmethod
    def require(snapshot: ConfigSnapshot, dotted_paths: Sequence[str]) -> None:
        missing = [
            key for key in dotted_paths
            if ConfigManager.get(snapshot, key, default=None) is None
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required config keys: {missing_str}")

    @staticmethod
    def save_json(snapshot: ConfigSnapshot, path: str) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": snapshot.created_at,
            "config_hash": snapshot.config_hash,
            "sources": snapshot.sources,
            "data": snapshot.data,
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
