import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from agent.backend import (
    BackendLoadError,
    STRUCTURED_CANDIDATE_CONTRACT_VERSION,
    backend_smoke_report,
    candidate_from_model_text,
    load_backend,
    structured_candidate_schema,
)
from agent.orchestrator import solve_task
from agent.trajectory import build_trajectory_record
from agent.trajectory_quality import SFT_POSITIVE, classify_trajectory
from agent.types import SOLVE_STATUS_BLOCKED, SOLVE_STATUS_VERIFIED, TaskRequest
from config import agent_cfg
from run import collect_backend_dependency_checks


@contextmanager
def agent_config_overrides(**overrides):
    old_values = {key: getattr(agent_cfg, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(agent_cfg, key, value)
        yield
    finally:
        for key, value in old_values.items():
            setattr(agent_cfg, key, value)


def fake_transformers_module(output_text: str):
    return fake_transformers_sequence([output_text])


def fake_transformers_sequence(output_texts):
    outputs = list(output_texts)
    state = {"index": 0}

    class FakeTokenizer:
        model_max_length = 128
        chat_template = None

        def __call__(self, prompt, return_tensors=None, **kwargs):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, generated, skip_special_tokens=True):
            index = min(state["index"], len(outputs) - 1)
            value = outputs[index]
            state["index"] += 1
            return value

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    return types.SimpleNamespace(
        __version__="fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )


def fake_chat_transformers_module(output_text: str, seen_prompts):
    class FakeTokenizer:
        model_max_length = 128
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            seen_prompts.append(messages[0]["content"])
            return "CHAT:" + messages[0]["content"]

        def __call__(self, prompt, return_tensors=None, **kwargs):
            seen_prompts.append(prompt)
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, generated, skip_special_tokens=True):
            return output_text

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    return types.SimpleNamespace(
        __version__="fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )


def fake_transformers_load_fails(message: str = "local files missing"):
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError(message)

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError(message)

    return types.SimpleNamespace(
        __version__="fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )


def valid_candidate_json(path="demo.py", content="def add(a, b):\n    return a + b\n"):
    return json.dumps(
        {
            "candidate_id": "model-fix",
            "summary": "repair add behavior",
            "edits": [{"path": path, "new_content": content}],
        }
    )


class TransformersBackendMvpTests(unittest.TestCase):
    def test_structured_candidate_contract_schema_is_canonical(self):
        schema = structured_candidate_schema()

        self.assertEqual(schema["contract_version"], STRUCTURED_CANDIDATE_CONTRACT_VERSION)
        self.assertEqual(schema["required"], ["candidate_id", "summary", "edits"])
        rejected = set(schema["rejected_cases"])
        self.assertIn("plain prose", rejected)
        self.assertIn("markdown fenced JSON", rejected)
        self.assertIn("empty new_content", rejected)

    def test_structured_candidate_contract_artifacts_exist(self):
        markdown_path = os.path.join("backend_integration", "structured_candidate_contract_v1.md")
        json_path = os.path.join("backend_integration", "structured_candidate_contract_v1.json")

        self.assertTrue(os.path.exists(markdown_path))
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertEqual(payload["schema"], "structured_candidate_contract_v1")
        self.assertEqual(payload["malformed_output_retry_policy"]["max_format_retries"], 2)
        self.assertFalse(payload["proves_real_coding_ability"])
        self.assertEqual(payload["quality_claim"], "none")

    def test_backend_dependency_surface_is_optional_and_machine_readable(self):
        report = collect_backend_dependency_checks()

        self.assertTrue(report["optional"])
        self.assertEqual(report["purpose"], "local_transformers_backend")
        self.assertTrue(report["exists"])
        packages = {item["package"] for item in report["items"]}
        self.assertIn("transformers", packages)
        self.assertIn("safetensors", packages)

    def test_backend_smoke_example_config_is_local_only_and_non_default(self):
        path = os.path.join("configs", "backend_smoke_local_example.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        agent = payload["agent"]
        self.assertEqual(agent["backend_kind"], "local_transformers_in_process")
        self.assertTrue(agent["backend_local_files_only"])
        self.assertFalse(agent["backend_trust_remote_code"])
        self.assertLessEqual(agent["backend_max_new_tokens"], 128)
        self.assertIn("run_artifacts/local_models", agent["backend_model_id_or_path"].replace("\\", "/"))

    def test_backend_smoke_guide_preserves_non_claiming_language(self):
        path = os.path.join("backend_integration", "backend_smoke_execution_guide_v1.md")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().lower()

        self.assertIn("backend_local_files_only: true", text)
        self.assertIn("proves_real_coding_ability: false", text)
        self.assertIn("quality_claim: none", text)
        self.assertIn("not proof of coding ability", text)
        self.assertNotIn("production ready", text)

    def test_no_backend_coding_task_still_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            with agent_config_overrides(
                backend_kind="none",
                backend_model_id_or_path=None,
                report_dir=os.path.join(td, "agent_reports"),
            ):
                payload = solve_task(
                    TaskRequest(
                        task_text="Fix the bug in demo.py",
                        file_hints=("demo.py",),
                        checks=("compileall:demo.py",),
                        workspace_root=td,
                    )
                )

        self.assertEqual(payload["status"], SOLVE_STATUS_BLOCKED)
        self.assertEqual(payload["backend_status"]["kind"], "none")
        self.assertEqual(payload["backend_status"]["failure_class"], "backend_not_configured")

    def test_exact_symbolic_task_bypasses_configured_backend(self):
        with tempfile.TemporaryDirectory() as td:
            with agent_config_overrides(
                backend_kind="local_transformers_in_process",
                backend_model_id_or_path=None,
                report_dir=os.path.join(td, "agent_reports"),
            ):
                payload = solve_task(
                    TaskRequest(
                        task_text='count substring "ana" in "banana"',
                        workspace_root=td,
                    )
                )

        self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
        self.assertEqual(payload["final_origin"], "deterministic_exact_tool")
        self.assertEqual(payload["backend_status"]["kind"], "deterministic_exact_tool")

    def test_configured_fake_transformers_backend_can_produce_verified_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            module_path = os.path.join(td, "demo.py")
            test_path = os.path.join(td, "test_demo.py")
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

            fake_module = fake_transformers_module(valid_candidate_json())
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                    report_dir=os.path.join(td, "agent_reports"),
                ):
                    payload = solve_task(
                        TaskRequest(
                            task_text="Fix add() in demo.py",
                            file_hints=("demo.py",),
                            checks=("unittest:discover -s . -p test_demo.py -v", "compileall:demo.py"),
                            workspace_root=td,
                        )
                    )

        self.assertEqual(payload["status"], SOLVE_STATUS_VERIFIED)
        self.assertEqual(payload["backend_status"]["kind"], "local_transformers_in_process")
        self.assertEqual(payload["final_origin"], "initial")
        self.assertEqual(payload["candidate"]["source"], "local_transformers_initial")
        self.assertEqual(
            payload["candidate"]["metadata"]["structured_contract_version"],
            STRUCTURED_CANDIDATE_CONTRACT_VERSION,
        )
        self.assertEqual(payload["candidate"]["metadata"]["generation_attempts"], 1)
        self.assertEqual(payload["quality_claim"], "verification_passed")
        trajectory = build_trajectory_record(payload)
        quality = classify_trajectory(trajectory)
        self.assertIn(SFT_POSITIVE, quality["quality_classes"])

    def test_malformed_backend_output_is_rejected_before_candidate_use(self):
        with tempfile.TemporaryDirectory() as td:
            fake_module = fake_transformers_module("Here is the fix in prose, not JSON.")
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                    report_dir=os.path.join(td, "agent_reports"),
                ):
                    payload = solve_task(
                        TaskRequest(
                            task_text="Fix demo.py",
                            file_hints=("demo.py",),
                            checks=("compileall:demo.py",),
                            workspace_root=td,
                        )
                    )

        self.assertEqual(payload["status"], SOLVE_STATUS_BLOCKED)
        self.assertIn("malformed_candidate_output", payload["blocked_reason"])
        self.assertEqual(payload["backend_status"]["last_failure_class"], "malformed_candidate_output")
        self.assertEqual(payload["backend_status"]["generation_attempts"], 3)
        self.assertEqual(payload["backend_status"]["malformed_retry_count"], 2)
        self.assertEqual(payload["attempts"], [])

    def test_unsafe_backend_edit_path_is_rejected_before_candidate_use(self):
        with tempfile.TemporaryDirectory() as td:
            unsafe_output = valid_candidate_json(path="../escape.py")
            fake_module = fake_transformers_module(unsafe_output)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                    report_dir=os.path.join(td, "agent_reports"),
                ):
                    payload = solve_task(
                        TaskRequest(
                            task_text="Fix demo.py",
                            file_hints=("demo.py",),
                            checks=("compileall:demo.py",),
                            workspace_root=td,
                        )
                    )

        self.assertEqual(payload["status"], SOLVE_STATUS_BLOCKED)
        self.assertIn("schema_validation_failed", payload["blocked_reason"])
        self.assertTrue(payload["backend_status"]["last_failure_details"])
        self.assertFalse(os.path.exists(os.path.join(td, "..", "escape.py")))

    def test_empty_backend_edit_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            empty_edit_output = valid_candidate_json(path="demo.py", content="")
            fake_module = fake_transformers_module(empty_edit_output)
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                    report_dir=os.path.join(td, "agent_reports"),
                ):
                    payload = solve_task(
                        TaskRequest(
                            task_text="Fix demo.py",
                            file_hints=("demo.py",),
                            checks=("compileall:demo.py",),
                            workspace_root=td,
                        )
                    )

        self.assertEqual(payload["status"], SOLVE_STATUS_BLOCKED)
        self.assertEqual(payload["backend_status"]["last_failure_class"], "schema_validation_failed")
        self.assertIn("new_content must be non-empty", payload["backend_status"]["last_failure_reason"])

    def test_backend_unavailable_surfaces_without_silent_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            with patch(
                "agent.backend._import_transformers_module",
                side_effect=BackendLoadError("backend_unavailable", "transformers package is not installed"),
            ):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                ):
                    backend, status = load_backend(workspace_root=td)

        self.assertIsNone(backend)
        self.assertFalse(status["available"])
        self.assertEqual(status["kind"], "local_transformers_in_process")
        self.assertEqual(status["failure_class"], "backend_unavailable")

    def test_backend_smoke_reports_schema_parse_success_without_claiming_coding_ability(self):
        with tempfile.TemporaryDirectory() as td:
            fake_module = fake_transformers_module(
                valid_candidate_json(path="backend_smoke_target.py", content="def value():\n    return 2\n")
            )
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                ):
                    report = backend_smoke_report(workspace_root=td)

        self.assertEqual(report["schema"], "agent_backend_smoke_v1")
        self.assertTrue(report["backend_available"])
        self.assertTrue(report["local_files_only"])
        self.assertTrue(report["transformers_available"])
        self.assertEqual(report["tokenizer_load_status"], "passed")
        self.assertEqual(report["model_load_status"], "passed")
        self.assertEqual(report["generation_status"], "passed")
        self.assertEqual(report["structured_contract_version"], STRUCTURED_CANDIDATE_CONTRACT_VERSION)
        self.assertEqual(report["generation_attempts"], 1)
        self.assertEqual(report["malformed_retry_count"], 0)
        self.assertEqual(report["structured_output_parse_status"], "passed")
        self.assertEqual(report["final_schema_validation_status"], "passed")
        self.assertTrue(report["structured_candidate_valid"])
        self.assertTrue(report["candidate_valid"])
        self.assertEqual(report["tiny_candidate_generation_status"], "candidate_generated")
        self.assertEqual(report["smoke_level"], "model_loaded_generation_succeeded_parse_passed")
        self.assertFalse(report["proves_real_coding_ability"])
        self.assertEqual(report["quality_claim"], "none")

    def test_backend_smoke_reports_model_missing_separately_from_runtime_missing(self):
        with tempfile.TemporaryDirectory() as td:
            missing_path = os.path.join(td, "missing-model")
            fake_module = fake_transformers_load_fails()
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path=missing_path,
                ):
                    report = backend_smoke_report(workspace_root=td)

        self.assertTrue(report["transformers_available"])
        self.assertFalse(report["backend_available"])
        self.assertFalse(report["model_path_exists"])
        self.assertEqual(report["tokenizer_load_status"], "failed")
        self.assertEqual(report["smoke_level"], "model_missing")
        self.assertEqual(report["failure_class"], "model_load_failed")

    def test_backend_smoke_reports_generation_succeeded_parse_retried_failed(self):
        with tempfile.TemporaryDirectory() as td:
            fake_module = fake_transformers_module("not json")
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                ):
                    report = backend_smoke_report(workspace_root=td)

        self.assertTrue(report["backend_available"])
        self.assertEqual(report["generation_status"], "passed")
        self.assertEqual(report["structured_output_parse_status"], "failed")
        self.assertFalse(report["structured_candidate_valid"])
        self.assertEqual(report["failure_class"], "malformed_candidate_output")
        self.assertEqual(report["generation_attempts"], 3)
        self.assertEqual(report["malformed_retry_count"], 2)
        self.assertEqual(report["smoke_level"], "model_loaded_generation_succeeded_parse_retried_failed")
        self.assertGreater(report["generated_text_chars"], 0)

    def test_candidate_parser_rejects_plain_prose(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(Exception) as caught:
                candidate_from_model_text("plain prose", source="test", workspace_root=td)
        self.assertIn("malformed_candidate_output", str(caught.exception))

    def test_candidate_parser_rejects_markdown_wrapped_json(self):
        with tempfile.TemporaryDirectory() as td:
            wrapped = "```json\n" + valid_candidate_json() + "\n```"
            with self.assertRaises(Exception) as caught:
                candidate_from_model_text(wrapped, source="test", workspace_root=td)
        self.assertIn("malformed_candidate_output", str(caught.exception))

    def test_bounded_retry_can_accept_valid_json_after_format_repair(self):
        with tempfile.TemporaryDirectory() as td:
            fake_module = fake_transformers_sequence(
                [
                    "not json",
                    "```json\nstill not strict\n```",
                    valid_candidate_json(path="backend_smoke_target.py", content="def value():\n    return 3\n"),
                ]
            )
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                ):
                    report = backend_smoke_report(workspace_root=td)

        self.assertEqual(report["structured_output_parse_status"], "passed")
        self.assertTrue(report["structured_candidate_valid"])
        self.assertEqual(report["generation_attempts"], 3)
        self.assertEqual(report["malformed_retry_count"], 2)
        self.assertEqual(report["smoke_level"], "model_loaded_generation_succeeded_parse_passed")

    def test_chat_template_is_used_when_available_without_weakening_schema(self):
        seen_prompts = []
        with tempfile.TemporaryDirectory() as td:
            fake_module = fake_chat_transformers_module(
                valid_candidate_json(path="backend_smoke_target.py", content="def value():\n    return 4\n"),
                seen_prompts,
            )
            with patch.dict(sys.modules, {"transformers": fake_module}):
                with agent_config_overrides(
                    backend_kind="local_transformers_in_process",
                    backend_model_id_or_path="./local-test-model",
                ):
                    report = backend_smoke_report(workspace_root=td)

        self.assertEqual(report["structured_output_parse_status"], "passed")
        self.assertTrue(any(str(item).startswith("CHAT:") for item in seen_prompts))


if __name__ == "__main__":
    unittest.main()
