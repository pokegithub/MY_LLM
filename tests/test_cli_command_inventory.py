import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "cli_command_inventory_v1.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _literal_string_list(node):
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return values


def extract_parser_command_choices():
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != "command":
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                choices = _literal_string_list(keyword.value)
                if choices is None:
                    raise AssertionError("run.py command choices are not a literal string list")
                return choices
    raise AssertionError("could not find run.py parser command choices")


def extract_dispatch_command_keys():
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    dispatches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "commands" for target in node.targets):
            continue
        keys = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise AssertionError("run.py dispatch keys are not literal strings")
            keys.append(key.value)
        dispatches.append(keys)
    if len(dispatches) != 1:
        raise AssertionError(f"expected one commands dispatch table, found {len(dispatches)}")
    return dispatches[0]


class CLICommandInventorySnapshotTests(unittest.TestCase):
    def test_fixture_freezes_expected_command_inventory(self):
        fixture = load_fixture()
        commands = fixture["commands"]
        self.assertEqual(fixture["schema"], "cli_command_inventory_v1")
        self.assertEqual(fixture["command_count"], 51)
        self.assertEqual(len(commands), 51)
        self.assertEqual(len(commands), len(set(commands)))

    def test_parser_choices_match_frozen_inventory(self):
        fixture = load_fixture()
        parser_choices = extract_parser_command_choices()
        self.assertEqual(parser_choices, fixture["commands"])

    def test_dispatch_keys_match_frozen_inventory(self):
        fixture = load_fixture()
        dispatch_keys = extract_dispatch_command_keys()
        self.assertEqual(dispatch_keys, fixture["commands"])

    def test_parser_choices_and_dispatch_keys_match_each_other(self):
        self.assertEqual(extract_parser_command_choices(), extract_dispatch_command_keys())

    def test_help_lists_frozen_commands_without_running_handlers(self):
        result = subprocess.run(
            [sys.executable, "run.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        help_text = result.stdout
        fixture = load_fixture()
        for command in fixture["commands"]:
            self.assertIn(command, help_text)

    def test_safety_critical_commands_remain_present(self):
        fixture = load_fixture()
        commands = set(fixture["commands"])
        for command in fixture["safety_critical_commands"]:
            self.assertIn(command, commands)

    def test_high_risk_commands_are_known_but_not_executed_by_inventory_tests(self):
        fixture = load_fixture()
        commands = set(fixture["commands"])
        for command in fixture["high_risk_commands"]:
            self.assertIn(command, commands)
        self.assertEqual(
            sorted(fixture["high_risk_commands"]),
            ["distill", "dpo", "full", "improve", "sft", "train"],
        )

    def test_cloud_or_download_commands_are_known_but_not_executed_by_inventory_tests(self):
        fixture = load_fixture()
        commands = set(fixture["commands"])
        for command in fixture["cloud_or_download_commands"]:
            self.assertIn(command, commands)
        self.assertEqual(
            sorted(fixture["cloud_or_download_commands"]),
            [
                "agent-backend-provision-small-candidate",
                "agent-backend-provision-tiny-model",
                "download",
                "download-core",
                "download-safe",
            ],
        )


if __name__ == "__main__":
    unittest.main()
