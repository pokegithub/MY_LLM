"""Shared logging setup for pipeline modules."""

from __future__ import annotations

import logging
import os
from typing import Optional

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_CONFIGURED = False


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger once with console and optional file handlers."""
    global _CONFIGURED

    numeric_level = _LEVEL_MAP.get(str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    if not _CONFIGURED:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
        _CONFIGURED = True

    if log_file:
        abs_log = os.path.abspath(log_file)
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == abs_log
            for handler in root.handlers
        ):
            parent = os.path.dirname(abs_log)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_handler = logging.FileHandler(abs_log, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )
            root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
