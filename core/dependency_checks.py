"""Small dependency-version checks for truthful readiness reporting."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import re
from typing import Any, Dict, List, Tuple


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(.*)$")
_SPEC_RE = re.compile(r"(<=|>=|==|<|>)\s*([A-Za-z0-9_.+-]+)")


def _version_tuple(value: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for piece in re.split(r"[.\-+]", value):
        match = re.match(r"^(\d+)", piece)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _compare_versions(left: str, right: str) -> int:
    a = _version_tuple(left)
    b = _version_tuple(right)
    width = max(len(a), len(b), 1)
    a = a + (0,) * (width - len(a))
    b = b + (0,) * (width - len(b))
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def _satisfies(installed: str, operator: str, required: str) -> bool:
    cmp = _compare_versions(installed, required)
    if operator == "==":
        return cmp == 0
    if operator == ">=":
        return cmp >= 0
    if operator == "<=":
        return cmp <= 0
    if operator == ">":
        return cmp > 0
    if operator == "<":
        return cmp < 0
    raise ValueError(f"unsupported requirement operator: {operator}")


def _parse_requirement(line: str) -> Dict[str, Any] | None:
    clean = line.split("#", 1)[0].strip()
    if not clean or clean.startswith("-"):
        return None
    match = _REQ_RE.match(clean)
    if not match:
        return None
    package = match.group(1)
    spec_text = match.group(2).split(";", 1)[0].strip()
    specs = _SPEC_RE.findall(spec_text)
    return {
        "package": package,
        "required": spec_text or "unspecified",
        "specs": specs,
    }


def check_requirement_file(path: str) -> Dict[str, Any]:
    """Return installed-version drift against this repo's requirements file.

    This intentionally supports only the simple requirement forms used here
    (`pkg>=a,<b`, `pkg==a`, etc.). Unsupported lines are marked unverified
    instead of silently treated as valid.
    """
    report: Dict[str, Any] = {
        "path": path,
        "exists": os.path.isfile(path),
        "ok": False,
        "items": [],
        "summary": {
            "pass": 0,
            "fail": 0,
            "unverified": 0,
        },
    }
    if not report["exists"]:
        report["items"].append({
            "package": "requirements_file",
            "status": "unverified",
            "required": path,
            "installed": None,
            "detail": "requirements file is missing",
        })
        report["summary"]["unverified"] = 1
        return report

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for line in lines:
        parsed = _parse_requirement(line)
        if parsed is None:
            continue
        package = parsed["package"]
        required = parsed["required"]
        specs = parsed["specs"]
        item = {
            "package": package,
            "required": required,
            "installed": None,
            "status": "pass",
            "detail": "installed version satisfies declared requirement",
        }
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            item.update({
                "status": "fail",
                "detail": "package is not installed",
            })
        else:
            item["installed"] = installed
            if not specs:
                item.update({
                    "status": "unverified",
                    "detail": "requirement has no supported version constraint",
                })
            else:
                failed = [
                    f"{op}{version}"
                    for op, version in specs
                    if not _satisfies(installed, op, version)
                ]
                if failed:
                    item.update({
                        "status": "fail",
                        "detail": (
                            "installed version does not satisfy "
                            + ", ".join(failed)
                        ),
                    })
        report["items"].append(item)
        report["summary"][item["status"]] += 1

    report["ok"] = (
        bool(report["items"])
        and report["summary"]["fail"] == 0
        and report["summary"]["unverified"] == 0
    )
    return report
