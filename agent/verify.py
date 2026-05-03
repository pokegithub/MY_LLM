"""Structured verifier for Phase 1 agent actions."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.types import CheckResult, PlanCheck, VerificationReport


def _write_stream(path: str, content: str) -> Optional[str]:
    if content == "":
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def run_verification(
    *,
    workspace_root: str,
    checks: List[PlanCheck],
    run_dir: str,
    exact_result: Optional[Dict[str, Any]] = None,
) -> VerificationReport:
    results: List[CheckResult] = []

    for index, check in enumerate(checks):
        stdout_path = os.path.join(run_dir, "checks", f"{index:02d}_{check.check_type}.stdout.txt")
        stderr_path = os.path.join(run_dir, "checks", f"{index:02d}_{check.check_type}.stderr.txt")

        if check.check_type == "exact_validation":
            passed = bool(exact_result and exact_result.get("validated") is True)
            summary = "deterministic exact-tool validation passed" if passed else "deterministic exact-tool validation failed"
            results.append(
                CheckResult(
                    check_type=check.check_type,
                    spec=check.spec,
                    passed=passed,
                    exit_code=None,
                    summary=summary,
                    evidence=exact_result or {},
                )
            )
            continue

        if check.check_type == "compileall":
            targets = [item.strip() for item in check.spec.split(",") if item.strip()]
            if not targets:
                targets = ["."]
            cmd = [sys.executable, "-m", "compileall", *targets]
        elif check.check_type == "pytest":
            cmd = [sys.executable, "-m", "pytest", *shlex.split(check.spec, posix=False)]
        elif check.check_type == "unittest":
            cmd = [sys.executable, "-m", "unittest", *shlex.split(check.spec, posix=False)]
        elif check.check_type == "path_exists":
            target = Path(workspace_root) / check.spec if not os.path.isabs(check.spec) else Path(check.spec)
            passed = target.exists()
            results.append(
                CheckResult(
                    check_type=check.check_type,
                    spec=check.spec,
                    passed=passed,
                    exit_code=None,
                    summary=f"path exists={passed}: {target}",
                    evidence={"resolved_path": str(target.resolve())},
                )
            )
            continue
        else:
            results.append(
                CheckResult(
                    check_type=check.check_type,
                    spec=check.spec,
                    passed=False,
                    exit_code=None,
                    summary="unsupported verification check type",
                )
            )
            continue

        completed = subprocess.run(
            cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout_saved = _write_stream(stdout_path, completed.stdout)
        stderr_saved = _write_stream(stderr_path, completed.stderr)
        summary_source = completed.stderr or completed.stdout or f"exit_code={completed.returncode}"
        summary_lines = summary_source.strip().splitlines()[:4]
        results.append(
            CheckResult(
                check_type=check.check_type,
                spec=check.spec,
                passed=completed.returncode == 0,
                exit_code=int(completed.returncode),
                summary=" | ".join(line[:240] for line in summary_lines)[:800],
                stdout_path=stdout_saved,
                stderr_path=stderr_saved,
                evidence={"command": cmd},
            )
        )

    meaningful = any(check.meaningful for check in checks)
    overall_passed = bool(results) and meaningful and all(item.passed for item in results)
    if not meaningful:
        summary = "no meaningful verification checks were available"
        quality_claim = "none"
    elif overall_passed:
        summary = "all verification checks passed"
        quality_claim = "verification_passed"
    else:
        summary = "one or more verification checks failed"
        quality_claim = "none"

    return VerificationReport(
        overall_passed=overall_passed,
        meaningful=meaningful,
        checks=tuple(results),
        summary=summary,
        quality_claim=quality_claim,
    )
