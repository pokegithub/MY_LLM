import json
import os
import time

from security.validator import ValidationError, get_allowed_data_roots, safe_load_json


class _LockFile:
    """Simple lockfile for cross-process mutual exclusion."""

    def __init__(
        self,
        path: str,
        timeout_sec: float = 10.0,
        poll_sec: float = 0.05,
        stale_sec: float = 300.0,
    ):
        self.path = path
        self.timeout_sec = timeout_sec
        self.poll_sec = poll_sec
        self.stale_sec = stale_sec
        self._fd = None

    def __enter__(self):
        deadline = time.time() + self.timeout_sec
        while True:
            try:
                self._fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                return self
            except FileExistsError:
                # Best-effort stale lock cleanup.
                try:
                    age = time.time() - os.path.getmtime(self.path)
                    if age > self.stale_sec:
                        os.remove(self.path)
                        continue
                except FileNotFoundError:
                    continue

                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring lock: {self.path}"
                    )
                time.sleep(self.poll_sec)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


class RunRegistry:
    def __init__(self, path: str = "./eval_results/run_registry.json"):
        self.path = path
        self.lock_path = f"{path}.lock"
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            self._atomic_write({"_version": 1, "runs": []})

    def _read(self):
        try:
            obj = safe_load_json(
                self.path,
                max_bytes=8_000_000,
                allowed_roots=get_allowed_data_roots(),
            )
        except (ValidationError, OSError, ValueError):
            obj = {"_version": 1, "runs": []}
        if not isinstance(obj, dict):
            obj = {"_version": 1, "runs": []}
        obj.setdefault("_version", 1)
        obj.setdefault("runs", [])
        return obj

    def _atomic_write(self, obj: dict):
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, self.path)

    def add(self, kind: str, meta: dict):
        with _LockFile(self.lock_path):
            obj = self._read()
            obj.setdefault("runs", []).append(
                {
                    "ts": int(time.time()),
                    "kind": kind,
                    "meta": meta,
                }
            )
            self._atomic_write(obj)
