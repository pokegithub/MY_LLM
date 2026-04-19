"""Security validation helpers for paths, payloads, and runtime policy."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

from observability.security_events import log_security_event


class ValidationError(ValueError):
    """Raised when untrusted inputs fail validation."""


DEFAULT_BLOCKED_SYMBOLS: FrozenSet[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "requests",
        "urllib",
        "builtins",
        "open",
        "input",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
    }
)

DEFAULT_ALLOWED_IMPORT_ROOTS: FrozenSet[str] = frozenset(
    {
        "math",
        "re",
        "string",
        "typing",
        "itertools",
        "functools",
        "collections",
        "heapq",
        "bisect",
        "statistics",
        "operator",
    }
)

_PROD_MARKERS: FrozenSet[str] = frozenset(
    {
        "prod",
        "production",
        "staging",
        "stage",
    }
)

_ALLOWED_EXEC_SCOPES: FrozenSet[str] = frozenset(
    {
        "offline_eval",
        "benchmark",
        "local_dev",
    }
)

_DEFAULT_JSON_MAX_BYTES = 10_000_000
_DEFAULT_JSONL_LINE_MAX_BYTES = 2_000_000


def _resolve(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_is_truthy(name: str, default: str = "0") -> bool:
    return _is_truthy(os.environ.get(name, default))


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def validate_local_path(
    path: Union[str, Path],
    *,
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
    must_exist: bool = False,
    expected_kind: str = "any",
) -> Path:
    """Validate path exists/type and optional root allowlist constraints."""
    resolved = _resolve(path)

    if must_exist and not resolved.exists():
        raise ValidationError(f"Path does not exist: {resolved}")

    if expected_kind == "file" and resolved.exists() and not resolved.is_file():
        raise ValidationError(f"Expected file path, got directory: {resolved}")
    if expected_kind == "dir" and resolved.exists() and not resolved.is_dir():
        raise ValidationError(f"Expected directory path, got file: {resolved}")

    if allowed_roots:
        roots = [_resolve(root) for root in allowed_roots]
        if not any(_is_within(resolved, root) for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise ValidationError(
                f"Path {resolved} is outside allowed roots: {allowed}"
            )

    return resolved


def get_allowed_data_roots() -> Tuple[Path, ...]:
    """Resolve default allowlisted roots for local JSON/data ingestion."""
    roots = []
    workspace_root = os.environ.get("MYLLM_WORKSPACE_ROOT", os.getcwd())
    roots.append(_resolve(workspace_root))

    extra = os.environ.get("MYLLM_ALLOWED_DATA_ROOTS", "").strip()
    if extra:
        for token in extra.split(os.pathsep):
            token = token.strip()
            if token:
                roots.append(_resolve(token))

    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        deduped.append(root)
        seen.add(key)
    return tuple(deduped)


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    required_keys: Sequence[str] = ("model_state_dict",),
) -> None:
    """Validate basic checkpoint payload structure."""
    if not isinstance(payload, Mapping):
        raise ValidationError(
            f"Checkpoint payload must be mapping, got {type(payload)!r}"
        )

    for key in required_keys:
        if key not in payload:
            raise ValidationError(f"Checkpoint payload missing key: {key}")


def safe_load_json(
    path: Union[str, Path],
    required_keys: Optional[Sequence[str]] = None,
    max_bytes: int = _DEFAULT_JSON_MAX_BYTES,
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
) -> Any:
    """Load JSON content with optional required-key validation."""
    resolved = validate_local_path(
        path,
        must_exist=True,
        expected_kind="file",
        allowed_roots=allowed_roots,
    )
    if resolved.stat().st_size > max_bytes:
        raise ValidationError(
            f"JSON file exceeds size limit ({max_bytes} bytes): {resolved}"
        )
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Invalid JSON in {resolved}: line={exc.lineno} col={exc.colno}"
        ) from exc

    if required_keys:
        if not isinstance(obj, dict):
            raise ValidationError("JSON root must be object when required_keys is set")
        missing = [key for key in required_keys if key not in obj]
        if missing:
            raise ValidationError(f"JSON missing required keys: {', '.join(missing)}")

    return obj


def iter_jsonl(
    path: Union[str, Path],
    *,
    max_bytes: int = _DEFAULT_JSON_MAX_BYTES,
    max_line_bytes: int = _DEFAULT_JSONL_LINE_MAX_BYTES,
    allowed_roots: Optional[Iterable[Union[str, Path]]] = None,
) -> Iterator[Any]:
    """Safely stream JSONL records with file and per-line size guards."""
    resolved = validate_local_path(
        path,
        must_exist=True,
        expected_kind="file",
        allowed_roots=allowed_roots,
    )
    if resolved.stat().st_size > max_bytes:
        raise ValidationError(
            f"JSONL file exceeds size limit ({max_bytes} bytes): {resolved}"
        )

    with open(resolved, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw_bytes = len(raw.encode("utf-8", errors="replace"))
            if raw_bytes > max_line_bytes:
                raise ValidationError(
                    f"JSONL line {line_no} exceeds limit ({max_line_bytes} bytes) in {resolved}"
                )

            line = raw.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"Invalid JSONL in {resolved} at line {line_no}: "
                    f"line={exc.lineno} col={exc.colno}"
                ) from exc


def is_production_runtime() -> bool:
    """Infer production-like runtime from conventional environment markers."""
    markers = [
        os.environ.get("MYLLM_RUNTIME_ENV", ""),
        os.environ.get("MYLLM_ENV", ""),
        os.environ.get("ENV", ""),
    ]
    return any(str(marker).strip().lower() in _PROD_MARKERS for marker in markers)


def code_execution_policy_message() -> str:
    return (
        "Code execution disabled by policy. Offline-only enablement requires "
        "MYLLM_ENABLE_CODE_EXEC=1 and "
        "MYLLM_CODE_EXEC_SCOPE in {offline_eval, benchmark, local_dev}, "
        "and runtime must not be production/staging."
    )


def is_code_execution_enabled(env_var: str = "MYLLM_ENABLE_CODE_EXEC") -> bool:
    """Gate code execution behind strict non-production, scoped opt-in policy."""
    if is_production_runtime():
        return False

    if not _env_is_truthy(env_var, default="0"):
        return False

    scope = os.environ.get("MYLLM_CODE_EXEC_SCOPE", "").strip().lower()
    if not scope:
        return False
    return scope in _ALLOWED_EXEC_SCOPES


def validate_python_snippet_safety(
    code: str,
    *,
    blocked_names: FrozenSet[str] = DEFAULT_BLOCKED_SYMBOLS,
    allowed_import_roots: FrozenSet[str] = DEFAULT_ALLOWED_IMPORT_ROOTS,
) -> Tuple[bool, str]:
    """Apply static AST checks before any sandboxed snippet execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_import_roots:
                    return False, f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed_import_roots:
                return False, f"Blocked import-from: {node.module}"
        elif isinstance(node, ast.Name) and node.id in blocked_names:
            return False, f"Blocked symbol: {node.id}"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in blocked_names:
                return False, f"Blocked attribute access: {node.value.id}.{node.attr}"

    return True, "ok"


def execute_python_snippet(
    code: str,
    *,
    timeout: int = 10,
) -> Tuple[bool, str]:
    """Execute Python snippet in a constrained subprocess when policy allows."""
    if not is_code_execution_enabled():
        log_security_event(
            "code_execution_blocked",
            {
                "reason": "policy_disabled",
                "scope": os.environ.get("MYLLM_CODE_EXEC_SCOPE", ""),
                "runtime_env": os.environ.get("MYLLM_RUNTIME_ENV", ""),
            },
        )
        return False, code_execution_policy_message()

    safe_code = sanitize_text(code, max_chars=200_000)
    ok, reason = validate_python_snippet_safety(safe_code)
    if not ok:
        log_security_event(
            "code_execution_blocked",
            {
                "reason": reason,
                "code_size": len(safe_code),
            },
        )
        return False, reason

    try:
        with tempfile.TemporaryDirectory() as td:
            candidate_path = os.path.join(td, "candidate.py")
            with open(candidate_path, "w", encoding="utf-8") as handle:
                handle.write(safe_code)

            result = subprocess.run(
                [sys.executable, "-I", "-S", candidate_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env={
                    "PYTHONHASHSEED": "0",
                    "PYTHONNOUSERSITE": "1",
                },
            )
        if result.returncode == 0:
            log_security_event(
                "code_execution_completed",
                {
                    "success": True,
                    "returncode": 0,
                    "code_size": len(safe_code),
                },
            )
            return True, sanitize_text(result.stdout, max_chars=500)
        log_security_event(
            "code_execution_completed",
            {
                "success": False,
                "returncode": int(result.returncode),
                "code_size": len(safe_code),
            },
        )
        return False, sanitize_text(result.stderr, max_chars=500)
    except subprocess.TimeoutExpired:
        log_security_event(
            "code_execution_timeout",
            {
                "timeout_sec": int(timeout),
                "code_size": len(safe_code),
            },
        )
        return False, "Timeout"
    except Exception as exc:
        log_security_event(
            "code_execution_error",
            {
                "error": sanitize_text(str(exc), max_chars=200),
                "code_size": len(safe_code),
            },
        )
        return False, sanitize_text(str(exc), max_chars=200)


def sanitize_text(value: Any, max_chars: int = 20000) -> str:
    """Convert to text and apply minimal sanitization for logs/IO."""
    text = str(value)
    text = text.replace("\x00", "")
    return text[:max_chars]
