"""Training telemetry utilities for gradient, loss-spike, and hardware tracking."""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import torch

from observability.metrics import MetricsLogger


class _LossSpikeTracker:
    def __init__(self, threshold: float = 1.4, window: int = 32, warmup: int = 20):
        self.threshold = max(float(threshold), 1.05)
        self.window = max(int(window), 4)
        self.warmup = max(int(warmup), 1)
        self.history: Deque[float] = deque(maxlen=self.window)

    def update(self, step: int, loss: float) -> Dict[str, Any]:
        self.history.append(float(loss))
        baseline = sum(self.history) / max(len(self.history), 1)
        ratio = float(loss) / max(baseline, 1e-9)
        is_spike = len(self.history) >= self.warmup and ratio >= self.threshold
        return {
            "loss_baseline": round(baseline, 6),
            "loss_ratio": round(ratio, 6),
            "loss_spike": bool(is_spike),
            "loss_spike_threshold": self.threshold,
            "step": int(step),
        }


def _gradient_stats(model: torch.nn.Module) -> Dict[str, Any]:
    sum_sq = 0.0
    abs_sum = 0.0
    abs_max = 0.0
    elements = 0
    non_zero = 0
    tensors = 0

    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if grad.is_sparse:
            grad = grad.coalesce().values()

        g = grad.detach().float()
        if g.numel() == 0:
            continue

        tensors += 1
        elements += g.numel()
        sum_sq += float(torch.sum(g * g).item())
        g_abs = torch.abs(g)
        abs_sum += float(torch.sum(g_abs).item())
        abs_max = max(abs_max, float(torch.max(g_abs).item()))
        non_zero += int(torch.count_nonzero(g).item())

    if elements == 0:
        return {
            "grad_tensors": 0,
            "grad_l2": 0.0,
            "grad_abs_mean": 0.0,
            "grad_abs_max": 0.0,
            "grad_zero_fraction": 1.0,
        }

    grad_l2 = math.sqrt(max(sum_sq, 0.0))
    return {
        "grad_tensors": tensors,
        "grad_l2": round(grad_l2, 6),
        "grad_abs_mean": round(abs_sum / elements, 8),
        "grad_abs_max": round(abs_max, 8),
        "grad_zero_fraction": round(1.0 - (non_zero / elements), 6),
    }


def _cuda_stats(device: str) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device_type": "cpu"}

    idx = 0
    if isinstance(device, str) and ":" in device:
        try:
            idx = int(device.split(":", 1)[1])
        except ValueError:
            idx = 0

    with torch.cuda.device(idx):
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        max_allocated = torch.cuda.max_memory_allocated(idx)

    payload: Dict[str, Any] = {
        "device_type": "cuda",
        "device_index": idx,
        "cuda_memory_allocated_mb": round(allocated / (1024 ** 2), 3),
        "cuda_memory_reserved_mb": round(reserved / (1024 ** 2), 3),
        "cuda_memory_peak_mb": round(max_allocated / (1024 ** 2), 3),
    }

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if out.returncode == 0 and out.stdout.strip():
            rows = [r.strip() for r in out.stdout.strip().splitlines() if r.strip()]
            if idx < len(rows):
                util_gpu, mem_used, mem_total, temp = [
                    x.strip() for x in rows[idx].split(",")
                ]
                payload.update(
                    {
                        "gpu_util_percent": float(util_gpu),
                        "gpu_mem_used_mb": float(mem_used),
                        "gpu_mem_total_mb": float(mem_total),
                        "gpu_temp_c": float(temp),
                    }
                )
    except Exception:
        # Hardware sampling should never fail a training step.
        pass

    return payload


class TrainingTelemetry:
    """Write structured per-step telemetry events as JSONL."""

    def __init__(
        self,
        path: str,
        *,
        enabled: bool = True,
        gradient_every: int = 25,
        hardware_every: int = 10,
        loss_spike_threshold: float = 1.4,
        phase: str = "train",
    ):
        self.enabled = bool(enabled)
        self.phase = str(phase)
        self.gradient_every = max(int(gradient_every), 1)
        self.hardware_every = max(int(hardware_every), 1)
        self._spikes = _LossSpikeTracker(threshold=loss_spike_threshold)
        self._metrics = MetricsLogger(path) if self.enabled else None
        self._last_hw: Dict[str, Any] = {}

    def log_step(
        self,
        *,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float],
        model: Optional[torch.nn.Module],
        device: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled or self._metrics is None:
            return {}

        payload: Dict[str, Any] = {
            "kind": "training_step",
            "phase": self.phase,
            "step": int(step),
            "loss": round(float(loss), 6),
            "lr": round(float(lr), 12),
        }

        if grad_norm is not None:
            payload["grad_norm"] = round(float(grad_norm), 6)

        spike = self._spikes.update(step=step, loss=loss)
        payload.update(spike)

        if model is not None and step % self.gradient_every == 0:
            payload.update(_gradient_stats(model))

        if step % self.hardware_every == 0:
            self._last_hw = _cuda_stats(device)
        if self._last_hw:
            payload.update(self._last_hw)

        if extra:
            payload.update(extra)

        self._metrics.log(payload)
        return payload


class PerformanceMonitor:
    """Track runtime throughput and hardware utilization for training steps."""

    def __init__(self, tokens_per_step: int, smoothing: float = 0.9):
        self.tokens_per_step = max(int(tokens_per_step), 1)
        self.smoothing = min(max(float(smoothing), 0.0), 0.99)
        self._last_step: Optional[int] = None
        self._last_time: Optional[float] = None
        self._ema_tps: Optional[float] = None

    def sample(self, step: int, device: str) -> Dict[str, Any]:
        now = time.time()
        if self._last_step is None or self._last_time is None:
            self._last_step = int(step)
            self._last_time = now
            payload = _cuda_stats(device)
            payload["perf_tokens_per_sec"] = 0.0
            payload["perf_tokens_per_sec_ema"] = 0.0
            return payload

        step_delta = max(int(step) - self._last_step, 0)
        elapsed = max(now - self._last_time, 1e-9)
        inst_tps = float(step_delta * self.tokens_per_step) / elapsed

        if self._ema_tps is None:
            self._ema_tps = inst_tps
        else:
            self._ema_tps = (
                self.smoothing * self._ema_tps
                + (1.0 - self.smoothing) * inst_tps
            )

        self._last_step = int(step)
        self._last_time = now

        payload = _cuda_stats(device)
        payload["perf_tokens_per_sec"] = round(inst_tps, 3)
        payload["perf_tokens_per_sec_ema"] = round(float(self._ema_tps), 3)
        return payload


def default_telemetry_path(output_dir: str, phase: str) -> str:
    ts = int(time.time())
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{phase}_telemetry_{ts}.jsonl")
