"""
quant_utils.py - Practical quantization/export utilities for small model serving.
"""

import os
import json
import time
import torch

from config import QuantConfig, quant_cfg, model_cfg
from model import LLM
from tokenizer import BPETokenizer
from core.checkpoint_io import (
    atomic_torch_save,
    extract_state_dict,
    safe_torch_load,
)


SUPPORTED_QUANT_MODES = {"dynamic-int8"}


class QuantizationExportError(RuntimeError):
    """Raised when a requested quantization export cannot be produced."""


def validate_quant_mode(quant_mode: str) -> None:
    if quant_mode not in SUPPORTED_QUANT_MODES:
        supported = ", ".join(sorted(SUPPORTED_QUANT_MODES))
        raise ValueError(f"Unsupported quant mode: {quant_mode}. Supported: {supported}")


def validate_quant_report_payload(report: dict) -> None:
    if not isinstance(report, dict):
        raise TypeError("quantization report must be a dict")
    if not isinstance(report.get("success"), bool):
        raise ValueError("quantization report requires boolean 'success'")
    if report.get("quant_mode") not in SUPPORTED_QUANT_MODES:
        raise ValueError("quantization report has unsupported quant_mode")

    errors = report.get("errors", [])
    if errors is None:
        errors = []
    if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
        raise ValueError("quantization report 'errors' must be a list of strings")
    if "torchscript_error" in report or "onnx_error" in report:
        raise ValueError("legacy export error fields are not allowed in reports")
    if report["success"] and errors:
        raise ValueError("successful quantization report cannot contain errors")
    if not report["success"] and not errors:
        raise ValueError("failed quantization report must explain errors")


def _write_quant_report(report: dict, report_path: str) -> None:
    validate_quant_report_payload(report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _load_model(cfg: QuantConfig, device: str = "cpu") -> LLM:
    tok = BPETokenizer()
    tok.load(cfg.tokenizer_path)
    model_cfg.vocab_size = tok.vocab_size_
    model = LLM(model_cfg).to(device)

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(
            f"Quantization checkpoint path not found: {cfg.checkpoint_path}"
        )
    st, path = safe_torch_load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(st))
    print(f"  Loaded checkpoint: {path}")

    model.eval()
    return model


def _dynamic_int8_quantize(model: LLM) -> torch.nn.Module:
    q = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return q


def run_quantization(cfg: QuantConfig = quant_cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    t0 = time.time()

    validate_quant_mode(cfg.quant_mode)
    if cfg.export_onnx:
        raise QuantizationExportError("ONNX export is configured but not implemented")

    model = _load_model(cfg, device="cpu")

    qmodel = _dynamic_int8_quantize(model)
    qpath = os.path.join(cfg.output_dir, "model_dynamic_int8.pt")
    atomic_torch_save({"model_state_dict": qmodel.state_dict()}, qpath)

    meta = os.path.join(cfg.output_dir, "quant_report.json")
    report = {
        "success": True,
        "quant_mode": cfg.quant_mode,
        "output_path": qpath,
        "export_torchscript": cfg.export_torchscript,
        "export_onnx": cfg.export_onnx,
        "elapsed_sec": round(time.time() - t0, 3),
        "errors": [],
    }

    if cfg.export_torchscript:
        sample = torch.randint(
            0,
            model_cfg.vocab_size,
            (1, min(cfg.max_seq_len, 64)),
            dtype=torch.long,
        )
        try:
            ts = torch.jit.trace(qmodel, (sample,), strict=False)
            tspath = os.path.join(cfg.output_dir, "model_dynamic_int8.ts")
            ts.save(tspath)
            report["torchscript_path"] = tspath
        except (RuntimeError, TypeError, ValueError) as exc:
            message = f"TorchScript export failed: {str(exc)[:200]}"
            report["success"] = False
            report["errors"].append(message)
            report["elapsed_sec"] = round(time.time() - t0, 3)
            _write_quant_report(report, meta)
            raise QuantizationExportError(message) from exc

    report["elapsed_sec"] = round(time.time() - t0, 3)
    _write_quant_report(report, meta)

    print(f"Quantization complete: {qpath}")
    print(f"Report: {meta}")


if __name__ == "__main__":
    run_quantization()
