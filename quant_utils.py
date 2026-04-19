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

    model = _load_model(cfg, device="cpu")

    if cfg.quant_mode != "dynamic-int8":
        raise ValueError(f"Unsupported quant mode: {cfg.quant_mode}")

    qmodel = _dynamic_int8_quantize(model)
    qpath = os.path.join(cfg.output_dir, "model_dynamic_int8.pt")
    atomic_torch_save({"model_state_dict": qmodel.state_dict()}, qpath)

    out = {
        "quant_mode": cfg.quant_mode,
        "output_path": qpath,
        "export_torchscript": cfg.export_torchscript,
        "export_onnx": cfg.export_onnx,
        "elapsed_sec": round(time.time() - t0, 3),
    }

    if cfg.export_torchscript:
        sample = torch.randint(0, model_cfg.vocab_size, (1, min(cfg.max_seq_len, 64)), dtype=torch.long)
        try:
            ts = torch.jit.trace(qmodel, (sample,), strict=False)
            tspath = os.path.join(cfg.output_dir, "model_dynamic_int8.ts")
            ts.save(tspath)
            out["torchscript_path"] = tspath
        except Exception as e:
            out["torchscript_error"] = str(e)[:200]

    meta = os.path.join(cfg.output_dir, "quant_report.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Quantization complete: {qpath}")
    print(f"Report: {meta}")


if __name__ == "__main__":
    run_quantization()
