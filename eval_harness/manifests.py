import json
import os
import time


def start_manifest(output_dir: str, config: dict) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    return {
        "run_id": int(time.time()),
        "started_at": int(time.time()),
        "config": config,
        "artifacts": [],
    }


def add_artifact(manifest: dict, name: str, path: str):
    manifest.setdefault("artifacts", []).append({"name": name, "path": path})


def save_manifest(manifest: dict, output_dir: str) -> str:
    manifest["finished_at"] = int(time.time())
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
