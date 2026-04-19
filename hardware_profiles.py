"""
hardware_profiles.py - hardware-aware profile selection utilities.
"""

import os
from typing import Any, Dict

from config import HardwareProfileConfig, hardware_profile_cfg
from core.logging import get_logger
from core.hardware_optim import detect_profile_name
from security.validator import get_allowed_data_roots, safe_load_json

LOGGER = get_logger("hardware_profile")


def _detect_profile_name() -> str:
    return detect_profile_name()


def load_profile(cfg: HardwareProfileConfig = hardware_profile_cfg) -> Dict[str, Any]:
    profile_name = cfg.active_profile
    if profile_name == "auto":
        profile_name = _detect_profile_name()

    if not profile_name.endswith(".json"):
        profile_name = profile_name + ".json"

    path = os.path.join(cfg.profile_dir, profile_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hardware profile not found: {path}")

    profile = safe_load_json(
        path,
        max_bytes=1_000_000,
        allowed_roots=get_allowed_data_roots(),
    )
    if not isinstance(profile, dict):
        raise ValueError(f"Hardware profile must be a JSON object: {path}")
    profile["path"] = path
    return profile


def show_profile(cfg: HardwareProfileConfig = hardware_profile_cfg) -> None:
    p = load_profile(cfg)
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("HARDWARE PROFILE")
    LOGGER.info("%s", "=" * 60)
    for k in sorted(p.keys()):
        LOGGER.info("%s: %s", k, p[k])
    LOGGER.info("%s", "=" * 60)


if __name__ == "__main__":
    show_profile()
