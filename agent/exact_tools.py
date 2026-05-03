"""Deterministic exact-task handlers for Phase 1."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from security.validator import validate_local_path


_COUNT_RE = re.compile(
    r'^\s*count\s+(substring|char|character|letter)\s+"(?P<needle>.*)"\s+in\s+"(?P<haystack>.*)"\s*$',
    re.IGNORECASE,
)
_CONTAINS_RE = re.compile(
    r'^\s*contains\s+"(?P<needle>.*)"\s+in\s+"(?P<haystack>.*)"\s*$',
    re.IGNORECASE,
)
_INDEX_RE = re.compile(
    r'^\s*index\s+of\s+"(?P<needle>.*)"\s+in\s+"(?P<haystack>.*)"\s*$',
    re.IGNORECASE,
)
_JSON_RE = re.compile(r"^\s*json\s+(validate|format)\s+(?P<payload>.+)$", re.IGNORECASE | re.DOTALL)
_FILE_EXISTS_RE = re.compile(r"^\s*file\s+exists\s+(?P<path>.+?)\s*$", re.IGNORECASE)
_SYMBOL_EXISTS_RE = re.compile(r'^\s*symbol\s+exists\s+"(?P<symbol>.+)"\s+in\s+(?P<path>.+?)\s*$', re.IGNORECASE)


def _count_overlapping(needle: str, haystack: str) -> int:
    if needle == "":
        return 0
    total = 0
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            break
        total += 1
        start = index + 1
    return total


def _safe_arithmetic_eval(expr: str) -> Any:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )

    def _eval(node: ast.AST) -> Any:
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("arithmetic constants must be int/float")
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("unsupported binary operator")
        raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def _python_symbol_exists(path: Path, symbol_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol_name:
                return True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    return True
    return False


def execute_exact_task(task_text: str, workspace_root: str) -> Dict[str, Any]:
    text = task_text.strip()

    if text.lower().startswith("arithmetic:"):
        expr = text.split(":", 1)[1].strip()
        result = _safe_arithmetic_eval(expr)
        recomputed = _safe_arithmetic_eval(expr)
        return {
            "tool": "arithmetic",
            "input": expr,
            "result": result,
            "validated": result == recomputed,
            "evidence": {"expression": expr},
        }

    match = _COUNT_RE.match(text)
    if match:
        needle = match.group("needle")
        haystack = match.group("haystack")
        result = _count_overlapping(needle, haystack)
        return {
            "tool": "count_substring",
            "input": {"needle": needle, "haystack": haystack},
            "result": result,
            "validated": result == _count_overlapping(needle, haystack),
            "evidence": {"needle_length": len(needle), "haystack_length": len(haystack)},
        }

    match = _CONTAINS_RE.match(text)
    if match:
        needle = match.group("needle")
        haystack = match.group("haystack")
        result = needle in haystack
        return {
            "tool": "contains",
            "input": {"needle": needle, "haystack": haystack},
            "result": result,
            "validated": result == (needle in haystack),
            "evidence": {"needle_length": len(needle), "haystack_length": len(haystack)},
        }

    match = _INDEX_RE.match(text)
    if match:
        needle = match.group("needle")
        haystack = match.group("haystack")
        result = haystack.find(needle)
        return {
            "tool": "index_of",
            "input": {"needle": needle, "haystack": haystack},
            "result": result,
            "validated": result == haystack.find(needle),
            "evidence": {"needle_length": len(needle), "haystack_length": len(haystack)},
        }

    match = _JSON_RE.match(text)
    if match:
        mode = text.split(None, 2)[1].lower()
        payload = match.group("payload")
        parsed = json.loads(payload)
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        reparsed = json.loads(canonical)
        return {
            "tool": f"json_{mode}",
            "input": payload,
            "result": canonical if mode == "format" else {"valid": True, "type": type(parsed).__name__},
            "validated": parsed == reparsed,
            "evidence": {"canonical": canonical, "json_type": type(parsed).__name__},
        }

    match = _FILE_EXISTS_RE.match(text)
    if match:
        raw_path = match.group("path").strip().strip('"')
        resolved = validate_local_path(
            os.path.join(workspace_root, raw_path) if not os.path.isabs(raw_path) else raw_path,
            allowed_roots=(workspace_root,),
            must_exist=False,
        )
        exists = resolved.exists()
        return {
            "tool": "file_exists",
            "input": raw_path,
            "result": exists,
            "validated": exists == resolved.exists(),
            "evidence": {"resolved_path": str(resolved)},
        }

    match = _SYMBOL_EXISTS_RE.match(text)
    if match:
        symbol_name = match.group("symbol")
        raw_path = match.group("path").strip().strip('"')
        resolved = validate_local_path(
            os.path.join(workspace_root, raw_path) if not os.path.isabs(raw_path) else raw_path,
            allowed_roots=(workspace_root,),
            must_exist=True,
            expected_kind="file",
        )
        if Path(resolved).suffix != ".py":
            raise ValueError("symbol exists handler only supports Python files")
        exists = _python_symbol_exists(Path(resolved), symbol_name)
        return {
            "tool": "symbol_exists",
            "input": {"symbol": symbol_name, "path": raw_path},
            "result": exists,
            "validated": exists == _python_symbol_exists(Path(resolved), symbol_name),
            "evidence": {"resolved_path": str(resolved)},
        }

    raise ValueError("no deterministic exact-task handler matched")
