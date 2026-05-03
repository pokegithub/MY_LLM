import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

from agent.context import build_context
from agent.exact_tools import execute_exact_task
from agent.orchestrator import plan_task, solve_task, verify_task
from agent.planner import build_plan
from agent.router import route_task
from agent.types import (
    ROUTE_CODING,
    ROUTE_EXACT,
    ROUTE_UNSUPPORTED,
    PLAN_STATUS_READY,
    SOLVE_STATUS_BLOCKED,
    SOLVE_STATUS_FAILED,
    SOLVE_STATUS_UNSUPPORTED,
    SOLVE_STATUS_VERIFIED,
    TaskRequest,
)
from agent.verify import run_verification


@contextmanager
def writable_tempdir():
    base_dir = os.path.join(os.getcwd(), "run_artifacts", "test_agent_phase1_workspaces")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"case_{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class AgentRouterTests(unittest.TestCase):
    def test_router_classifies_exact_symbolic_tasks(self):
        route = route_task(
            TaskRequest(task_text='count substring "ana" in "banana"')
        )
        self.assertEqual(route.route, ROUTE_EXACT)
        self.assertTrue(route.deterministic_required)

    def test_router_classifies_coding_edit_tasks(self):
        route = route_task(
            TaskRequest(
                task_text="Fix the failing test in sample.py",
                file_hints=("sample.py",),
            )
        )
        self.assertEqual(route.route, ROUTE_CODING)
        self.assertTrue(route.supported)


class AgentExactToolTests(unittest.TestCase):
    def test_exact_tool_solves_count_task(self):
        result = execute_exact_task(
            'count substring "ana" in "banana"',
            workspace_root=".",
        )
        self.assertEqual(result["tool"], "count_substring")
        self.assertEqual(result["result"], 2)
        self.assertTrue(result["validated"])

    def test_exact_tool_solves_json_format(self):
        result = execute_exact_task(
            'json format {"b":2,"a":1}',
            workspace_root=".",
        )
        self.assertEqual(result["tool"], "json_format")
        self.assertEqual(result["result"], '{"a":1,"b":2}')
        self.assertTrue(result["validated"])

    def test_unsupported_exact_task_does_not_fallback_to_backend(self):
        result = solve_task(
            TaskRequest(task_text='count words in "banana split"')
        )
        self.assertEqual(result["status"], SOLVE_STATUS_UNSUPPORTED)
        self.assertEqual(result["route"]["route"], ROUTE_UNSUPPORTED)


class AgentPlanningTests(unittest.TestCase):
    def test_plan_schema_is_machine_readable(self):
        request = TaskRequest(
            task_text="Fix the module",
            file_hints=("tests/test_cli_truthfulness.py",),
            checks=("compileall:tests/test_cli_truthfulness.py",),
        )
        route = route_task(request)
        context = build_context(request)
        plan = build_plan(request, route, context)
        self.assertEqual(plan.route, ROUTE_CODING)
        self.assertEqual(plan.retry_budget, 3)
        self.assertTrue(plan.checks)

    def test_plan_command_writes_report(self):
        payload = plan_task(
            TaskRequest(task_text='count substring "ana" in "banana"')
        )
        self.assertEqual(payload["schema"], "agent_phase2_report_v1")
        self.assertEqual(payload["status"], PLAN_STATUS_READY)
        self.assertTrue(os.path.isfile(payload["report_path"]))

    def test_plan_registry_failure_is_explicitly_degraded(self):
        class FailingRegistry:
            def add(self, *_args, **_kwargs):
                raise PermissionError("registry locked")

        with patch("agent.orchestrator.RunRegistry", return_value=FailingRegistry()):
            payload = plan_task(TaskRequest(task_text='count substring "ana" in "banana"'))

        self.assertEqual(payload["status"], PLAN_STATUS_READY)
        self.assertTrue(payload["degraded_mode"])
        self.assertFalse(payload["run_registry"]["ok"])
        self.assertTrue(payload["artifact_warnings"])
        self.assertIn("registry locked", payload["artifact_warnings"][0]["message"])


class AgentSolveTests(unittest.TestCase):
    def test_coding_task_without_backend_fails_closed(self):
        result = solve_task(
            TaskRequest(
                task_text="Fix the bug in demo.py",
                file_hints=("demo.py",),
                checks=("compileall:demo.py",),
            )
        )
        self.assertEqual(result["status"], SOLVE_STATUS_BLOCKED)
        self.assertIn("no coding backend configured", result["blocked_reason"])

    def test_workspace_rolls_back_failed_candidate(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "candidate.json")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("VALUE = 1\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import unittest\n"
                    "from demo import VALUE\n\n"
                    "class DemoTests(unittest.TestCase):\n"
                    "    def test_value(self):\n"
                    "        self.assertEqual(VALUE, 1)\n"
                )
            with open(backend_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "broken",
                        "summary": "introduce failing value",
                        "edits": [{"path": "demo.py", "new_content": "VALUE = 2\n"}],
                    },
                    handle,
                )

            result = solve_task(
                TaskRequest(
                    task_text="Fix demo.py",
                    file_hints=("demo.py",),
                    checks=("unittest:discover -s . -p test_demo.py -v",),
                    workspace_root=td,
                ),
                backend_script_path=backend_path,
            )

            self.assertEqual(result["status"], SOLVE_STATUS_FAILED)
            self.assertIn("rolled back", result["blocked_reason"])
            with open(module_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "VALUE = 1\n")

    def test_scripted_backend_can_produce_verified_success(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
            backend_path = os.path.join(td, "candidate.json")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("def add(a, b):\n    return a - b\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import unittest\n"
                    "from demo import add\n\n"
                    "class DemoTests(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(2, 3), 5)\n"
                )
            with open(backend_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "candidate_id": "fix-add",
                        "summary": "repair add()",
                        "edits": [
                            {
                                "path": "demo.py",
                                "new_content": "def add(a, b):\n    return a + b\n",
                            }
                        ],
                    },
                    handle,
                )

            result = solve_task(
                TaskRequest(
                    task_text="Fix add()",
                    file_hints=("demo.py",),
                    checks=("unittest:discover -s . -p test_demo.py -v", "compileall:demo.py"),
                    workspace_root=td,
                ),
                backend_script_path=backend_path,
            )

            self.assertEqual(result["status"], SOLVE_STATUS_VERIFIED)
            self.assertEqual(result["final_origin"], "initial")
            self.assertEqual(result["quality_claim"], "verification_passed")
            with open(module_path, "r", encoding="utf-8") as handle:
                self.assertIn("return a + b", handle.read())


class AgentVerifierTests(unittest.TestCase):
    def test_verifier_report_schema_records_green_checks(self):
        with writable_tempdir() as td:
            report = run_verification(
                workspace_root=td,
                checks=[],
                run_dir=os.path.join(td, "report"),
            )
            self.assertFalse(report.overall_passed)
            self.assertFalse(report.meaningful)

    def test_verify_exact_task_returns_verified_success(self):
        payload = verify_task(
            TaskRequest(task_text='contains "ana" in "banana"')
        )
        self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
        self.assertEqual(payload["verification"]["quality_claim"], "verification_passed")

    def test_verified_success_requires_green_report(self):
        with writable_tempdir() as td:
            module_path = os.path.join(td, "demo.py")
            with open(module_path, "w", encoding="utf-8") as handle:
                handle.write("def broken(:\n")
            payload = verify_task(
                TaskRequest(
                    task_text="Check the workspace",
                    file_hints=("demo.py",),
                    checks=("compileall:demo.py",),
                    workspace_root=td,
                )
            )
            self.assertEqual(payload["status"], SOLVE_STATUS_FAILED)
            self.assertFalse(payload["verification"]["overall_passed"])


if __name__ == "__main__":
    unittest.main()
