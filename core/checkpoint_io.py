"""Secure checkpoint I/O helpers with atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch

from observability.security_events import log_security_event
from security.validator import get_allowed_data_roots, is_production_runtime, safe_load_json


class CheckpointError(RuntimeError):
    """Raised when checkpoint content is invalid or missing."""


class CheckpointSecurityError(CheckpointError):
    """Raised when checkpoint path violates security constraints."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_MANIFEST_NAME = "checkpoint_manifest.json"
_DEFAULT_MANIFEST_MAX_BYTES = 5_000_000


def _resolve(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _env_is_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_sha256_sidecar(path: Path) -> Optional[str]:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.exists():
        return None

    raw = sidecar.read_text(encoding="utf-8").strip()
    if not raw:
        raise CheckpointSecurityError(f"Empty SHA256 sidecar: {sidecar}")

    expected = raw.split()[0].strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise CheckpointSecurityError(
            f"Invalid SHA256 value in sidecar {sidecar}: {expected!r}"
        )
    return expected


def _normalize_sha256(raw: str) -> str:
    value = str(raw).strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise CheckpointSecurityError(f"Invalid SHA256 value: {raw!r}")
    return value


def _manifest_required() -> bool:
    if _env_is_truthy("MYLLM_CHECKPOINT_MANIFEST_REQUIRED", default="0"):
        return True
    return is_production_runtime()


def resolve_checkpoint_manifest_path(
    checkpoint_path: Path,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve manifest path from explicit arg/env/default sibling file."""
    if manifest_path is not None:
        return _resolve(manifest_path)

    from_env = os.environ.get("MYLLM_CHECKPOINT_MANIFEST_PATH", "").strip()
    if from_env:
        return _resolve(from_env)

    default_path = checkpoint_path.parent / _DEFAULT_MANIFEST_NAME
    if default_path.exists():
        return _resolve(default_path)
    return None


def _extract_manifest_entry_hash(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        return _normalize_sha256(entry)
    if isinstance(entry, Mapping):
        for key in ("sha256", "hash", "digest"):
            if key in entry:
                return _normalize_sha256(entry[key])
    return None


def _manifest_expected_hash(
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    manifest_path: Path,
) -> Optional[str]:
    candidates = [
        checkpoint_path.name,
        str(checkpoint_path),
        checkpoint_path.as_posix(),
    ]
    try:
        candidates.append(os.path.relpath(checkpoint_path, manifest_path.parent))
    except Exception:
        pass

    table = manifest.get("checkpoints")
    if isinstance(table, Mapping):
        for key in candidates:
            if key in table:
                return _extract_manifest_entry_hash(table[key])

    for key in candidates:
        if key in manifest:
            return _extract_manifest_entry_hash(manifest[key])
    return None


def verify_checkpoint_integrity(
    checkpoint_path: Path,
    *,
    enforce_hash: bool,
    require_manifest: bool,
    manifest_path: Optional[Union[str, Path]] = None,
) -> None:
    """Validate checkpoint hash against manifest and/or sidecar policies."""
    resolved_manifest = resolve_checkpoint_manifest_path(
        checkpoint_path, manifest_path=manifest_path
    )

    expected_manifest_hash: Optional[str] = None
    if resolved_manifest is not None:
        if not resolved_manifest.exists():
            raise CheckpointSecurityError(
                f"Checkpoint manifest path does not exist: {resolved_manifest}"
            )
        manifest = safe_load_json(
            resolved_manifest,
            max_bytes=_DEFAULT_MANIFEST_MAX_BYTES,
            allowed_roots=get_allowed_data_roots(),
        )
        if not isinstance(manifest, Mapping):
            raise CheckpointSecurityError(
                f"Checkpoint manifest must be a mapping: {resolved_manifest}"
            )
        expected_manifest_hash = _manifest_expected_hash(
            manifest,
            checkpoint_path=checkpoint_path,
            manifest_path=resolved_manifest,
        )

    if require_manifest and resolved_manifest is None:
        raise CheckpointSecurityError(
            "Checkpoint manifest required by policy but not found. "
            f"Expected sibling manifest: {checkpoint_path.parent / _DEFAULT_MANIFEST_NAME} "
            "or set MYLLM_CHECKPOINT_MANIFEST_PATH."
        )
    if require_manifest and expected_manifest_hash is None:
        manifest_ref = resolved_manifest if resolved_manifest is not None else "<missing>"
        raise CheckpointSecurityError(
            "Checkpoint manifest required by policy but no entry found for "
            f"{checkpoint_path.name} in {manifest_ref}"
        )

    expected_sidecar_hash = _read_sha256_sidecar(checkpoint_path)

    if expected_manifest_hash and expected_sidecar_hash and expected_manifest_hash != expected_sidecar_hash:
        raise CheckpointSecurityError(
            "Checkpoint sidecar hash does not match manifest hash for "
            f"{checkpoint_path}"
        )

    expected_hash = expected_manifest_hash or expected_sidecar_hash
    if expected_hash is None:
        if enforce_hash or require_manifest:
            raise CheckpointSecurityError(
                "Checkpoint integrity policy requires hash but no manifest/sidecar hash was found "
                f"for {checkpoint_path}"
            )
        return

    actual_hash = _sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise CheckpointSecurityError(
            f"Checkpoint hash mismatch for {checkpoint_path}: expected {expected_hash}, got {actual_hash}"
        )


def verify_checkpoint_hash(path: Path, required: bool = False) -> None:
    """Verify checkpoint hash using optional <checkpoint>.sha256 sidecar."""
    expected = _read_sha256_sidecar(path)
    if expected is None:
        if required:
            raise CheckpointSecurityError(
                f"Checkpoint hash sidecar required but missing: {path}.sha256"
            )
        return

    actual = _sha256_file(path)
    if actual != expected:
        raise CheckpointSecurityError(
            f"Checkpoint hash mismatch for {path}: expected {expected}, got {actual}"
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def validate_checkpoint_path(
    path: Union[str, Path],
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
) -> Path:
    """Resolve and validate checkpoint path against optional root allowlist."""
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")

    if allowed_roots:
        roots = [_resolve(root) for root in allowed_roots]
        if not any(_is_within(resolved, root) for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise CheckpointSecurityError(
                f"Checkpoint path {resolved} is outside allowed roots: {allowed}"
            )

    return resolved


def pick_checkpoint_file(
    path_or_dir: Union[str, Path],
    prefer_best: bool = True,
) -> Path:
    """Select a checkpoint file from a path or directory."""
    resolved = _resolve(path_or_dir)
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {resolved}")

    checkpoints = sorted(p for p in resolved.iterdir() if p.suffix == ".pt")
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in: {resolved}")

    if prefer_best:
        best = [p for p in checkpoints if "best" in p.name.lower()]
        if best:
            return best[-1]

    return checkpoints[-1]


def safe_torch_load(
    path_or_dir: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
    prefer_best: bool = True,
    enforce_hash: Optional[bool] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    enforce_manifest: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Load checkpoint with safer torch deserialization mode."""
    checkpoint_path = pick_checkpoint_file(path_or_dir, prefer_best=prefer_best)
    checkpoint_path = validate_checkpoint_path(
        checkpoint_path, allowed_roots=allowed_roots
    )

    if enforce_manifest is None:
        enforce_manifest = _manifest_required()
    if enforce_hash is None:
        enforce_hash = _env_is_truthy("MYLLM_ENFORCE_CHECKPOINT_SHA256", default="0")

    try:
        verify_checkpoint_integrity(
            checkpoint_path,
            enforce_hash=bool(enforce_hash) or bool(enforce_manifest),
            require_manifest=bool(enforce_manifest),
            manifest_path=manifest_path,
        )
        log_security_event(
            "checkpoint_validation_passed",
            {
                "checkpoint_path": str(checkpoint_path),
                "enforce_hash": bool(enforce_hash),
                "enforce_manifest": bool(enforce_manifest),
            },
        )
    except Exception as exc:
        log_security_event(
            "checkpoint_validation_failed",
            {
                "checkpoint_path": str(checkpoint_path),
                "enforce_hash": bool(enforce_hash),
                "enforce_manifest": bool(enforce_manifest),
                "error": str(exc)[:200],
            },
        )
        raise

    try:
        payload = torch.load(
            str(checkpoint_path),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        log_security_event(
            "checkpoint_load_failed",
            {
                "checkpoint_path": str(checkpoint_path),
                "error": str(exc)[:200],
            },
        )
        raise

    if not isinstance(payload, dict):
        log_security_event(
            "checkpoint_load_failed",
            {
                "checkpoint_path": str(checkpoint_path),
                "error": f"invalid_payload_type:{type(payload)!r}",
            },
        )
        raise CheckpointError(
            f"Checkpoint payload must be a dict, got {type(payload)!r}"
        )
    return payload, checkpoint_path


def write_checkpoint_hash_sidecar(path: Union[str, Path]) -> Path:
    """Create or update <checkpoint>.sha256 sidecar."""
    target = _resolve(path)
    digest = _sha256_file(target)
    sidecar = Path(f"{target}.sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return sidecar


def update_checkpoint_manifest(
    checkpoint_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Upsert checkpoint hash metadata into JSON manifest."""
    target = _resolve(checkpoint_path)
    resolved_manifest = resolve_checkpoint_manifest_path(
        target,
        manifest_path=manifest_path,
    )
    if resolved_manifest is None:
        resolved_manifest = target.parent / _DEFAULT_MANIFEST_NAME

    manifest: Dict[str, Any] = {"version": 1, "checkpoints": {}}
    if resolved_manifest.exists():
        existing = safe_load_json(
            resolved_manifest,
            max_bytes=_DEFAULT_MANIFEST_MAX_BYTES,
            allowed_roots=get_allowed_data_roots(),
        )
        if isinstance(existing, Mapping):
            manifest.update({k: v for k, v in existing.items() if k != "checkpoints"})
            existing_table = existing.get("checkpoints", {})
            if isinstance(existing_table, Mapping):
                manifest["checkpoints"] = dict(existing_table)

    digest = _sha256_file(target)
    table = manifest.setdefault("checkpoints", {})
    if not isinstance(table, dict):
        table = {}
        manifest["checkpoints"] = table

    table[target.name] = {
        "sha256": digest,
        "size_bytes": int(target.stat().st_size),
        "updated_at": int(time.time()),
    }
    manifest["updated_at"] = int(time.time())

    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return resolved_manifest


def extract_state_dict(
    payload: Mapping[str, Any],
    key: str = "model_state_dict",
) -> Dict[str, Any]:
    """Extract state dict from payload and validate schema."""
    if key not in payload:
        raise CheckpointError(f"Checkpoint missing required key: {key}")
    state_dict = payload[key]
    if not isinstance(state_dict, dict):
        raise CheckpointError(
            f"Checkpoint key '{key}' must map to a dict, got {type(state_dict)!r}"
        )
    if not state_dict:
        raise CheckpointError(f"Checkpoint key '{key}' contains empty state_dict")

    invalid_keys = [k for k in state_dict.keys() if not isinstance(k, str)]
    if invalid_keys:
        raise CheckpointError(
            f"Checkpoint state_dict keys must be strings, got invalid key {invalid_keys[0]!r}"
        )
    return state_dict


def atomic_torch_save(obj: Any, path: Union[str, Path]) -> None:
    """Write checkpoint atomically to avoid partial-file corruption."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)

    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, target)

        sidecar = write_checkpoint_hash_sidecar(target)
        if _env_is_truthy("MYLLM_WRITE_CHECKPOINT_MANIFEST", default="1"):
            manifest_path = update_checkpoint_manifest(target)
        else:
            manifest_path = None

        log_security_event(
            "checkpoint_saved",
            {
                "checkpoint_path": str(target),
                "hash_sidecar": str(sidecar),
                "manifest_path": str(manifest_path) if manifest_path else "",
            },
        )
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
