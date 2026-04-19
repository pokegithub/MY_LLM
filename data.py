"""Data loading pipeline for mixed streaming and cached token sources."""

import gc
import os
import sys
import re
import hashlib
import traceback
import torch
import random
import queue
import threading
import numpy as np
from collections import defaultdict
from torch.utils.data import IterableDataset
from datasets import load_dataset
from config import TrainConfig
from core.logging import get_logger

LOGGER = get_logger("data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_OVERRIDES = {
    "HuggingFaceH4/ultrachat_200k": "train_sft",
    "truthful_qa": "validation",
}

TOKEN_CACHE_DIR = "./data_cache/tokens"
VAL_CACHE_DIR = "./data_cache/val_tokens"

CURRICULUM_BOOST = {
    "HuggingFaceFW/fineweb-edu",
    "wikimedia/wikipedia",
    "HuggingFaceTB/finemath",
    "open-phi/textbooks",
    "openai/gsm8k",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    "teknium/OpenHermes-2.5",
    "Open-Orca/SlimOrca",
}

ALL_SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 3.0,
     lambda x: x.get("text", "")),
    ("wikimedia/wikipedia", "20231101.en", 2.0,
     lambda x: x.get("text", "")),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 2.0,
     lambda x: x.get("text", "")),
    ("open-phi/textbooks", None, 1.5,
     lambda x: x.get("markdown", x.get("text", ""))),
    ("HuggingFaceTB/finemath", "finemath-3plus", 2.5,
     lambda x: x.get("text", "")),
    ("open-web-math/open-web-math", None, 1.5,
     lambda x: x.get("text", "")),
    ("openai/gsm8k", "main", 1.2,
     lambda x: "Q: " + str(x.get("question", "")) +
     "\nA: " + str(x.get("answer", ""))),
    ("microsoft/orca-math-word-problems-200k", None, 2.0,
     lambda x: "Q: " + str(x.get("question", "")) +
     "\nA: " + str(x.get("answer", ""))),
    ("TIGER-Lab/MathInstruct", None, 1.8,
     lambda x: (str(x.get("instruction", "")) + "\n" +
                str(x.get("output", ""))).strip()),
    ("lighteval/MATH-Hard", None, 1.5,
     lambda x: (str(x.get("problem", "")) + "\n" +
                str(x.get("solution", ""))).strip()),
    ("m-a-p/CodeFeedback-Filtered-Instruction", None, 1.8,
     lambda x: (str(x.get("query", "")) + "\n" +
                str(x.get("answer", ""))).strip()),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", None, 1.5,
     lambda x: (str(x.get("instruction", "")) + "\n" +
                str(x.get("response", ""))).strip()),
    ("iamtarun/python_code_instructions_18k_alpaca", None, 1.2,
     lambda x: (str(x.get("instruction", "")) + "\n" +
                str(x.get("output", ""))).strip()),
    ("codeparrot/github-code", None, 2.0,
     lambda x: x.get("code", "")),
    ("ajibawa-2023/Code-290k-ShareGPT", None, 1.5,
     lambda x: "\n".join(
         str(m.get("value", ""))
         for m in (x.get("conversations") or [])
         if isinstance(m, dict))),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, 2.0,
     lambda x: (str(x.get("instruction", "")) + "\n" +
                str(x.get("response", ""))).strip()),
    ("nickrosh/Evol-Instruct-Code-80k-v1", None, 2.0,
     lambda x: (str(x.get("instruction", "")) + "\n" +
                str(x.get("output", ""))).strip()),
    ("deepmind/code_contests", None, 2.0,
     lambda x: str(x.get("description", "")).strip()),
    ("code-search-net/code_search_net", None, 1.8,
     lambda x: (str(x.get("func_documentation_string", "")) + "\n" +
                str(x.get("whole_func_string", ""))).strip()),
    ("b-mc2/sql-create-context", None, 1.2,
     lambda x: (str(x.get("question", "")) + "\n" +
                str(x.get("answer", ""))).strip()),
    ("argilla/magpie-ultra-v0.1", None, 2.0,
     lambda x: "\n".join(
         m.get("content", "")
         for m in (x.get("messages") or [])
         if isinstance(m, dict))),
    ("HuggingFaceH4/ultrachat_200k", None, 1.5,
     lambda x: "\n".join(
         m.get("content", "")
         for m in (x.get("messages") or [])
         if isinstance(m, dict))),
    ("teknium/OpenHermes-2.5", None, 2.0,
     lambda x: "\n".join(
         t.get("value", "")
         for t in (x.get("conversations") or [])
         if isinstance(t, dict))),
    ("Open-Orca/SlimOrca", None, 1.8,
     lambda x: "\n".join(
         t.get("value", "")
         for t in (x.get("conversations") or [])
         if isinstance(t, dict))),
    ("google/boolq", None, 0.8,
     lambda x: (str(x.get("passage", "")) + "\nQ: " +
                str(x.get("question", "")) +
                "\nA: " + str(x.get("answer", ""))).strip()),
    ("truthful_qa", "generation", 1.5,
     lambda x: ("Q: " + str(x.get("question", "")) + "\nA: " +
                str(x.get("best_answer", ""))).strip()),
    ("Anthropic/hh-rlhf", None, 0.8,
     lambda x: str(x.get("chosen", "")).strip()),
    ("allenai/ai2_arc", "ARC-Challenge", 0.6,
     lambda x: (str(x.get("question", "")) + "\n" +
                str(x.get("answerKey", ""))).strip()),
    ("winogrande", "winogrande_xl", 0.5,
     lambda x: str(x.get("sentence", "")).strip()),
]


def _active_sources(cfg: TrainConfig):
    excluded = set(getattr(cfg, "excluded_benchmark_sources", ()))
    if not excluded:
        return list(ALL_SOURCES)
    return [s for s in ALL_SOURCES if s[0] not in excluded]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_label(path: str, sub) -> str:
    name = path.replace("/", "__")
    if sub:
        name += f"__{sub.replace('/', '_')}"
    return name


def _bin_path(path: str, sub) -> str:
    return os.path.join(TOKEN_CACHE_DIR, f"{_safe_label(path, sub)}.bin")


def _get_text(fn, example) -> str:
    try:
        t = fn(example)
        if isinstance(t, str) and len(t.strip()) > 30:
            return t.strip()
    except Exception as exc:
        LOGGER.debug(
            "Source extractor failed; using fallback fields: %s",
            exc,
        )
    for key in (
        "text", "content", "markdown", "output", "code",
        "abstract", "document", "chosen", "response",
    ):
        v = example.get(key, "")
        if isinstance(v, str) and len(v.strip()) > 30:
            return v.strip()
    return ""


def _eff_weight(path: str, base: float, step: int, csteps: int) -> float:
    if step < csteps and path in CURRICULUM_BOOST:
        return base * 2.0
    return base


# ---------------------------------------------------------------------------
# Token sources
# ---------------------------------------------------------------------------

class BinaryTokenSource:
    def __init__(
        self,
        bin_path: str,
        vocab_size: int,
        rank: int = 0,
        world: int = 1,
    ):
        self.path = bin_path
        self.dtype = np.uint16 if vocab_size <= 65535 else np.uint32
        self.rank = max(0, int(rank))
        self.world = max(1, int(world))

    def __iter__(self):
        data = np.fromfile(self.path, dtype=self.dtype)
        n = len(data)
        if n == 0:
            return
        start = random.randint(0, max(0, min(50000, n - 1)))
        for i in range(start + self.rank, n, self.world):
            yield int(data[i])


# ---------------------------------------------------------------------------
# Pipeline exception
# ---------------------------------------------------------------------------

class DataPipelineError(Exception):
    """Raised when the data pipeline encounters a fatal error."""
    pass


# ---------------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------------

class ValidationTokenDataset(IterableDataset):
    """
    Fixed-size validation dataset from held-out binary token files.
    If no validation files exist, yields from a small held-out portion
    of training data.
    """

    def __init__(self, tokenizer, cfg: TrainConfig):
        self.tok = tokenizer
        self.seq_len = cfg.seq_len
        self.vsz = tokenizer.vocab_size_
        self.val_dir = cfg.val_data_dir
        self.n_steps = cfg.val_steps
        self.sources = _active_sources(cfg)

    def __iter__(self):
        dtype = np.uint16 if self.vsz <= 65535 else np.uint32

        # Try to load from validation cache
        if os.path.exists(self.val_dir):
            files = [
                os.path.join(self.val_dir, f)
                for f in os.listdir(self.val_dir)
                if f.endswith(".bin")
            ]
            if files:
                buf = []
                yielded = 0
                for fp in files:
                    data = np.fromfile(fp, dtype=dtype)
                    buf.extend(int(x) for x in data)
                    while len(buf) >= self.seq_len + 1:
                        chunk = buf[: self.seq_len + 1]
                        buf = buf[self.seq_len:]
                        yield {
                            "input_ids": torch.tensor(
                                chunk[:-1], dtype=torch.long
                            ),
                            "targets": torch.tensor(
                                chunk[1:], dtype=torch.long
                            ),
                        }
                        yielded += 1
                        if yielded >= self.n_steps:
                            return
                return

        # Fallback: use last 5% of first training source
        for path, sub, _, fn in self.sources[:3]:
            bp = _bin_path(path, sub)
            if os.path.exists(bp):
                data = np.fromfile(bp, dtype=dtype)
                n = len(data)
                val_start = int(n * 0.95)
                if n - val_start < self.seq_len + 1:
                    continue
                buf = list(data[val_start:].astype(int))
                yielded = 0
                while len(buf) >= self.seq_len + 1 and yielded < self.n_steps:
                    chunk = buf[: self.seq_len + 1]
                    buf = buf[self.seq_len:]
                    yield {
                        "input_ids": torch.tensor(
                            chunk[:-1], dtype=torch.long
                        ),
                        "targets": torch.tensor(
                            chunk[1:], dtype=torch.long
                        ),
                    }
                    yielded += 1
                if yielded > 0:
                    return


# ---------------------------------------------------------------------------
# Main training dataset
# ---------------------------------------------------------------------------

class MixedTokenDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        cfg: TrainConfig,
        start_step: int = 0,
        rank: int = 0,
        world: int = 1,
    ):
        self.tok = tokenizer
        self.seq_len = cfg.seq_len
        self.max_open = cfg.max_open_sources
        self.vsz = tokenizer.vocab_size_
        self.csteps = cfg.curriculum_tokens // (
            cfg.batch_size * cfg.seq_len * cfg.grad_accum
        )
        self.start_step = start_step
        self.rank = max(0, int(rank))
        self.world = max(1, int(world))
        self.sources = _active_sources(cfg)

    def _classify(self, step):
        disk, stream = [], []
        for path, sub, bw, fn in self.sources:
            ew = _eff_weight(path, bw, step, self.csteps)
            bp = _bin_path(path, sub)
            if os.path.exists(bp):
                sz = os.path.getsize(bp) // (
                    2 if self.vsz <= 65535 else 4
                )
                if sz > 0:
                    disk.append((path, sub, ew, fn, bp, sz))
                else:
                    stream.append((path, sub, bw, fn))
            else:
                stream.append((path, sub, bw, fn))
        return disk, stream

    def __iter__(self):
        step = self.start_step
        buf: list = []
        eos = self.tok.eos_token_id
        consecutive_failures = 0

        def _record_failure(
            source: str,
            sample_index: int,
            current_count: int,
            exc: Exception,
        ) -> int:
            next_count = current_count + 1
            LOGGER.warning(
                "Data sample skipped source=%s index=%s failures=%s/50 error=%s",
                source,
                sample_index,
                next_count,
                exc,
            )
            if next_count > 50:
                raise DataPipelineError(
                    "Data pipeline aborted after >50 consecutive sample failures."
                ) from exc
            return next_count

        disk, stream = self._classify(step)
        print(
            f"\n  Sources: {len(disk)} disk  {len(stream)} streaming",
            flush=True,
        )
        print(
            f"  Phase: {'CURRICULUM' if step < self.csteps else 'FULL'}",
            flush=True,
        )

        # ---- Disk phase ----
        if disk:
            tw = sum(w for *_, w, _, _, _ in disk)
            probs = [w / tw for *_, w, _, _, _ in disk]
            iters = [
                iter(BinaryTokenSource(
                    bp,
                    self.vsz,
                    rank=self.rank,
                    world=self.world,
                ))
                for *_, bp, _ in disk
            ]
            labels = [
                path + (f"/{sub}" if sub else "")
                for path, sub, _, _, _, _ in disk
            ]
            dead: set = set()
            nd = len(disk)
            sample_index = [0] * nd
            print(
                f"  Disk: {sum(s for *_, s in disk) / 1e6:.0f}M tokens",
                flush=True,
            )

            while len(dead) < nd:
                alive = [i for i in range(nd) if i not in dead]
                ap = [probs[i] for i in alive]
                s = sum(ap)
                idx = random.choices(
                    alive, weights=[p / s for p in ap], k=1
                )[0]
                try:
                    token = next(iters[idx])
                    sample_index[idx] += 1
                    buf.append(token)
                    consecutive_failures = 0
                except StopIteration:
                    dead.add(idx)
                    continue
                except Exception as exc:
                    sample_index[idx] += 1
                    consecutive_failures = _record_failure(
                        labels[idx],
                        sample_index[idx],
                        consecutive_failures,
                        exc,
                    )
                    continue

                while len(buf) >= self.seq_len + 1:
                    chunk = buf[: self.seq_len + 1]
                    buf = buf[self.seq_len:]
                    step += 1
                    if step == self.csteps:
                        print(
                            f"  Curriculum ended step {step:,}",
                            flush=True,
                        )
                    yield {
                        "input_ids": torch.tensor(
                            chunk[:-1], dtype=torch.long
                        ),
                        "targets": torch.tensor(
                            chunk[1:], dtype=torch.long
                        ),
                    }

            del iters
            gc.collect()
            print("  Disk phase done.", flush=True)

        # ---- Stream phase ----
        if stream:
            sl = list(stream)
            random.shuffle(sl)
            groups = [
                sl[i: i + self.max_open]
                for i in range(0, len(sl), self.max_open)
            ]
            print(
                f"  Streaming {len(stream)} in {len(groups)} groups",
                flush=True,
            )

            for gi, group in enumerate(groups):
                loaded, wts = [], []
                for path, sub, bw, fn in group:
                    ew = _eff_weight(path, bw, step, self.csteps)
                    try:
                        sp = SPLIT_OVERRIDES.get(path, "train")
                        kw = dict(split=sp, streaming=True)
                        if sub:
                            kw["name"] = sub
                        ds = load_dataset(path, **kw)
                        loaded.append(
                            (ds, fn, path + (f"/{sub}" if sub else ""))
                        )
                        wts.append(ew)
                        print(f"    OK {loaded[-1][2]}", flush=True)
                    except Exception as e:
                        print(
                            f"    SKIP {path}: {str(e)[:50]}",
                            flush=True,
                        )

                if not loaded:
                    continue

                tw = sum(wts)
                probs = [w / tw for w in wts]
                iters = {
                    i: iter(loaded[i][0]) for i in range(len(loaded))
                }
                fns = [fn for _, fn, _ in loaded]
                labels = [label for _, _, label in loaded]
                dead: set = set()
                n = len(loaded)
                sample_index = [0] * n

                print(
                    f"  Group {gi + 1}/{len(groups)}: {n} src "
                    f"[{'curr' if step < self.csteps else 'full'}]",
                    flush=True,
                )

                while len(dead) < n:
                    alive = [i for i in range(n) if i not in dead]
                    ap = [probs[i] for i in alive]
                    s = sum(ap)
                    idx = random.choices(
                        alive, weights=[p / s for p in ap], k=1
                    )[0]
                    try:
                        ex = next(iters[idx])
                        sample_index[idx] += 1
                        consecutive_failures = 0
                    except StopIteration:
                        dead.add(idx)
                        iters.pop(idx, None)
                        continue
                    except Exception as exc:
                        sample_index[idx] += 1
                        consecutive_failures = _record_failure(
                            labels[idx],
                            sample_index[idx],
                            consecutive_failures,
                            exc,
                        )
                        continue

                    try:
                        text = _get_text(fns[idx], ex)
                        if not text:
                            continue
                        if self.world > 1:
                            shard = int(
                                hashlib.md5(text.encode("utf-8")).hexdigest(),
                                16,
                            ) % self.world
                            if shard != self.rank:
                                continue
                        ids = self.tok.encode(text, add_bos=False)
                    except Exception as exc:
                        consecutive_failures = _record_failure(
                            labels[idx],
                            sample_index[idx],
                            consecutive_failures,
                            exc,
                        )
                        continue

                    ids.append(eos)
                    buf.extend(ids)

                    while len(buf) >= self.seq_len + 1:
                        chunk = buf[: self.seq_len + 1]
                        buf = buf[self.seq_len:]
                        step += 1
                        yield {
                            "input_ids": torch.tensor(
                                chunk[:-1], dtype=torch.long
                            ),
                            "targets": torch.tensor(
                                chunk[1:], dtype=torch.long
                            ),
                        }

                del loaded, iters
                gc.collect()


# ---------------------------------------------------------------------------
# PrefetchDataLoader
# ---------------------------------------------------------------------------

class PrefetchDataLoader:
    """
    Background-thread prefetcher with bounded keep-alive semantics.
    """

    def __init__(self, dataloader, prefetch: int = 8):
        self.loader = dataloader
        self.prefetch = prefetch

    def _worker(self, q: queue.Queue):
        consecutive_failures = 0
        batch_index = 0
        iterator = iter(self.loader)

        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            except DataPipelineError as exc:
                q.put(("err", (exc, traceback.format_exc())))
                break
            except Exception as exc:
                consecutive_failures += 1
                LOGGER.warning(
                    "Prefetch worker skipped shard=loader index=%s failures=%s/50 error=%s",
                    batch_index,
                    consecutive_failures,
                    exc,
                )
                if consecutive_failures > 50:
                    q.put(("err", (exc, traceback.format_exc())))
                    break
                continue

            consecutive_failures = 0
            q.put(("ok", batch))
            batch_index += 1

        q.put(("end", None))

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        t = threading.Thread(target=self._worker, args=(q,), daemon=True)
        t.start()
        while True:
            tag, val = q.get()
            if tag == "ok":
                yield val
            elif tag == "err":
                exc, tb = val
                raise DataPipelineError(
                    f"Data pipeline worker failed:\n{tb}"
                ) from exc
            else:
                break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_data_workers(cfg: TrainConfig) -> int:
    configured = int(getattr(cfg, "data_num_workers", 0))
    if configured < 0:
        configured = 0

    # Conservative default for CUDA Linux; keep Windows stable by default.
    if configured == 0 and torch.cuda.is_available() and os.name != "nt":
        configured = 2

    if os.name == "nt" and configured > 2:
        configured = 2

    return configured


def _loader_kwargs(cfg: TrainConfig, is_train: bool) -> dict:
    workers = _resolve_data_workers(cfg)
    pin_memory = bool(
        getattr(cfg, "data_pin_memory", torch.cuda.is_available())
    )

    kwargs = {
        "num_workers": workers,
        "pin_memory": pin_memory,
    }

    if workers > 0:
        kwargs["persistent_workers"] = bool(
            getattr(cfg, "data_persistent_workers", True)
        )
        kwargs["prefetch_factor"] = max(
            1, int(getattr(cfg, "data_prefetch_factor", 2))
        )

    if not is_train and kwargs.get("persistent_workers") and workers <= 1:
        kwargs["persistent_workers"] = False

    return kwargs


def get_dataloader(
    tokenizer,
    cfg: TrainConfig,
    start_step: int = 0,
    rank: int = 0,
    world: int = 1,
):
    kwargs = _loader_kwargs(cfg, is_train=True)
    base = torch.utils.data.DataLoader(
        MixedTokenDataset(
            tokenizer,
            cfg,
            start_step=start_step,
            rank=rank,
            world=world,
        ),
        batch_size=cfg.batch_size,
        **kwargs,
    )
    prefetch_q = max(1, int(getattr(cfg, "data_prefetch_queue", 8)))
    return PrefetchDataLoader(base, prefetch=prefetch_q)


def get_val_dataloader(tokenizer, cfg: TrainConfig):
    kwargs = _loader_kwargs(cfg, is_train=False)
    base = torch.utils.data.DataLoader(
        ValidationTokenDataset(tokenizer, cfg),
        batch_size=cfg.batch_size,
        **kwargs,
    )
    return base