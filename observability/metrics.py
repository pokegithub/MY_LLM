import json
import os
import threading
import time


class MetricsLogger:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, payload: dict):
        item = {"ts": time.time(), **payload}
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")
