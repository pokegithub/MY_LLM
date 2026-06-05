"""Fail-closed Lightning AI free-credit readiness runner.

This module prepares local/cloud smoke execution plans only. It does not log in,
store credentials, start Lightning jobs, download models, or run training.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
POLICY_PATH = SCRIPT_DIR / "gpu_policy_v1.json"
DEFAULT_REPORT_ROOT = REPO_ROOT / "run_artifacts" / "cloud" / "lightning_ai"
DEFAULT_CANDIDATE_MODEL_PATH = "./run_artifacts/local_models/qwen2.5-coder-0.5b-instruct"
SCHEMA = "lightning_ai_cloud_readiness_report_v1"


def load_policy(path: Path = POLICY_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_env(env: Mapping[str, str], name: str) -> Optional[float]:
    value = env.get(name)
    if value is None or not str(value).strip():
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _bool_env(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().upper() == "YES"


def _safe_path_exists(path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.exists()


def _torch_summary(include_torch: bool = True) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "torch_error": None,
    }
    if not include_torch:
        summary["torch_error"] = "torch_probe_skipped"
        return summary
    try:
        import torch  # type: ignore

        summary["torch_available"] = True
        summary["cuda_available"] = bool(torch.cuda.is_available())
        if summary["cuda_available"]:
            device = torch.cuda.current_device()
            summary["gpu_name"] = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            summary["vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 2)
    except Exception as exc:  # pragma: no cover - environment dependent
        summary["torch_error"] = exc.__class__.__name__
    return summary


def detect_auth_status(env: Mapping[str, str]) -> Dict[str, Any]:
    manual_confirmed = _bool_env(env, "MYLLM_LIGHTNING_AUTH_CONFIRMED")
    cli_present = bool(shutil.which("lightning") or shutil.which("lightning-ai") or shutil.which("lit"))
    if manual_confirmed:
        status = "manual_confirmed"
        action = None
    else:
        status = "missing"
        action = "manual_login_required"
    return {
        "auth_status": status,
        "action_required": action,
        "manual_auth_confirmation_env": "MYLLM_LIGHTNING_AUTH_CONFIRMED",
        "lightning_cli_present": cli_present,
        "credentials_stored": False,
        "password_requested": False,
        "secrets_printed": False,
    }


def detect_credit_status(policy: Mapping[str, Any], env: Mapping[str, str]) -> Dict[str, Any]:
    max_env = _float_env(env, "MYLLM_CLOUD_MAX_CREDITS")
    balance = _float_env(env, "MYLLM_LIGHTNING_FREE_CREDITS_AVAILABLE")
    estimated = _float_env(env, "MYLLM_LIGHTNING_ESTIMATED_CREDITS")
    free_confirmed = _bool_env(env, "MYLLM_LIGHTNING_FREE_CREDITS_CONFIRMED")
    price_confirmed = _bool_env(env, "MYLLM_LIGHTNING_PRICE_CONFIRMED")
    budget_max = float(policy["credit_budget_max"])

    failures: List[str] = []
    if max_env is None:
        failures.append("max_credit_env_missing")
    elif max_env != budget_max:
        failures.append("max_credit_env_mismatch")
    if balance is None and not free_confirmed:
        failures.append("free_credit_balance_unknown")
    if estimated is None and not price_confirmed:
        failures.append("estimated_price_unknown")
    if estimated is not None and estimated > budget_max:
        failures.append("estimated_cost_exceeds_budget")
    if balance is not None and estimated is not None and estimated > balance:
        failures.append("estimated_cost_exceeds_free_credit_balance")

    return {
        "credit_budget_max": budget_max,
        "require_free_credit_only": bool(policy["require_free_credit_only"]),
        "allow_paid_overage": bool(policy["allow_paid_overage"]),
        "max_credit_env_value": max_env,
        "free_credit_balance_status": "known" if balance is not None else ("manual_confirmed" if free_confirmed else "unknown"),
        "free_credit_balance_value": balance,
        "estimated_credit_cost_status": "known" if estimated is not None else ("manual_confirmed" if price_confirmed else "unknown"),
        "estimated_credit_cost_value": estimated,
        "manual_free_credit_confirmation": free_confirmed,
        "manual_price_confirmation": price_confirmed,
        "budget_failures": failures,
        "budget_ok_for_execute": not failures,
    }


def build_plan(policy: Mapping[str, Any], task_class: str) -> Dict[str, Any]:
    allowed_commands = list(policy["allowed_commands"])
    return {
        "task_class": task_class,
        "commands": allowed_commands,
        "blocked_commands": list(policy["blocked_commands"]),
        "training_commands_blocked": True,
        "full_bakeoff_blocked": True,
        "production_serving_blocked": True,
        "quality_claim": "none",
    }


def evaluate_guards(
    *,
    policy: Mapping[str, Any],
    args: argparse.Namespace,
    env: Mapping[str, str],
    include_torch: bool = True,
) -> Dict[str, Any]:
    mode = args.mode
    execute_requested = bool(args.execute)
    task_class = args.task_class
    allowed = task_class in set(policy["allowed_task_classes"])
    disallowed = task_class in set(policy["disallowed_task_classes"])
    auth = detect_auth_status(env)
    credit = detect_credit_status(policy, env)
    torch_summary = _torch_summary(include_torch=include_torch)
    candidate_model_path_exists = _safe_path_exists(args.candidate_model_path)
    explicit_execute_confirmed = _bool_env(env, "MYLLM_CLOUD_CONFIRM_EXECUTE")
    runtime_minutes = int(args.max_runtime_minutes)
    runtime_ok = runtime_minutes <= int(policy["max_runtime_minutes_hard_limit"])

    failures: List[str] = []
    warnings: List[str] = []
    if disallowed or not allowed:
        failures.append("task_class_not_allowed")
    if runtime_minutes > int(policy["max_runtime_minutes_hard_limit"]):
        failures.append("runtime_minutes_exceeds_hard_limit")
    if not candidate_model_path_exists and task_class in {"backend_smoke", "candidate_readiness_smoke"}:
        warnings.append("candidate_model_path_missing_for_local_repo")

    if execute_requested:
        if not explicit_execute_confirmed:
            failures.append("manual_execute_confirmation_missing")
        if auth["auth_status"] != "manual_confirmed":
            failures.append("auth_missing")
        if not credit["budget_ok_for_execute"]:
            failures.extend(credit["budget_failures"])
    else:
        if auth["auth_status"] != "manual_confirmed":
            warnings.append("manual_login_required_before_execute")
        if not credit["budget_ok_for_execute"]:
            warnings.append("credit_or_price_confirmation_required_before_execute")

    execution_allowed = execute_requested and not failures
    return {
        "schema": SCHEMA,
        "mode": mode,
        "execute_requested": execute_requested,
        "execution_allowed": execution_allowed,
        "execution_blocked": not execution_allowed,
        "blocked_reasons": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "repo": {
            "repo_root": str(REPO_ROOT),
            "candidate_model_path": args.candidate_model_path,
            "candidate_model_path_exists": candidate_model_path_exists,
        },
        "dependency": {
            "torch": torch_summary,
        },
        "lightning": {
            **auth,
            "environment_detected": any(key.startswith("LIGHTNING") for key in env),
        },
        "budget": credit,
        "gpu_policy": {
            "selection_rule": policy["gpu_selection_rule"],
            "compatible_requirements": policy["compatible_gpu_requirements"],
            "selected_gpu": args.gpu_name or "not_selected_dry_run_price_required",
            "cheapest_compatible_required": True,
            "absolute_cheapest_not_sufficient": True,
        },
        "task": {
            "task_class": task_class,
            "allowed": allowed,
            "disallowed": disallowed,
            "runtime_minutes": runtime_minutes,
            "runtime_ok": runtime_ok,
        },
        "plan": build_plan(policy, task_class),
        "phase_b_started": False,
        "training_started": False,
        "cloud_job_started": False,
        "credits_spent": 0,
        "quality_claim": "none",
        "model_quality_proven": False,
    }


def write_report(report: Mapping[str, Any], report_path: Optional[str] = None) -> str:
    if report_path:
        path = Path(report_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        DEFAULT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        path = DEFAULT_REPORT_ROOT / f"lightning_cloud_readiness_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path.resolve())


def _print_summary(report: Mapping[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LIGHTNING AI CLOUD READINESS")
    print("=" * 60)
    print(f"  mode                  : {report.get('mode')}")
    print(f"  execute_requested     : {str(report.get('execute_requested')).lower()}")
    print(f"  execution_allowed     : {str(report.get('execution_allowed')).lower()}")
    print(f"  cloud_job_started     : {str(report.get('cloud_job_started')).lower()}")
    print(f"  credits_spent         : {report.get('credits_spent')}")
    print(f"  auth_status           : {(report.get('lightning') or {}).get('auth_status')}")
    print(f"  action_required       : {(report.get('lightning') or {}).get('action_required')}")
    print(f"  free_credit_status    : {(report.get('budget') or {}).get('free_credit_balance_status')}")
    print(f"  estimated_cost_status : {(report.get('budget') or {}).get('estimated_credit_cost_status')}")
    print(f"  budget_max            : {(report.get('budget') or {}).get('credit_budget_max')}")
    print(f"  paid_overage_allowed  : {str((report.get('budget') or {}).get('allow_paid_overage')).lower()}")
    print(f"  task_class            : {(report.get('task') or {}).get('task_class')}")
    print(f"  task_allowed          : {str((report.get('task') or {}).get('allowed')).lower()}")
    print(f"  selected_gpu          : {(report.get('gpu_policy') or {}).get('selected_gpu')}")
    print(f"  phase_b_started       : {str(report.get('phase_b_started')).lower()}")
    print(f"  model_quality_proven  : {str(report.get('model_quality_proven')).lower()}")
    if report.get("blocked_reasons"):
        print(f"  blocked_reasons       : {', '.join(report['blocked_reasons'])}")
    if report.get("warnings"):
        print(f"  warnings              : {', '.join(report['warnings'])}")
    if report.get("mode") == "print_plan":
        print("  plan_commands         :")
        for command in (report.get("plan") or {}).get("commands", []):
            print(f"    - {command}")
    if report.get("report_path"):
        print(f"  report_path           : {report.get('report_path')}")
    print("=" * 60)


def _run_execute_plan(report: Mapping[str, Any]) -> int:
    if not report.get("execution_allowed"):
        return 2
    commands = list((report.get("plan") or {}).get("commands") or [])
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, shell=True)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightning AI free-credit cloud readiness runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_const", const="dry_run", dest="mode")
    mode.add_argument("--check-env", action="store_const", const="check_env", dest="mode")
    mode.add_argument("--print-plan", action="store_const", const="print_plan", dest="mode")
    mode.add_argument("--run-smoke", action="store_const", const="run_smoke", dest="mode")
    parser.set_defaults(mode="dry_run")
    parser.add_argument("--execute", action="store_true", help="Execute allowed smoke commands only after all guards pass.")
    parser.add_argument("--task-class", default="candidate_readiness_smoke")
    parser.add_argument("--candidate-model-path", default=DEFAULT_CANDIDATE_MODEL_PATH)
    parser.add_argument("--gpu-name", default=None, help="Manually selected Lightning GPU label after price/credit review.")
    parser.add_argument("--max-runtime-minutes", type=int, default=30)
    parser.add_argument("--policy-path", default=str(POLICY_PATH))
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    policy = load_policy(Path(args.policy_path))
    report = evaluate_guards(policy=policy, args=args, env=os.environ)
    report["report_path"] = write_report(report, args.report_path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_summary(report)
    if args.mode == "run_smoke" and args.execute:
        return _run_execute_plan(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
