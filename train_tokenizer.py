"""
train_tokenizer.py — BPE tokenizer trainer (v3)

Changes from v2:
  - Improved error handling
  - Better progress reporting
  - Validation of all special tokens
"""

import os
import sys
import json
import random
import time
import tempfile
import shutil
import re
import logging
from typing import Dict, List, Tuple

from security.validator import safe_load_json
from core.logging import configure_logging, get_logger

LOGGER = get_logger("train_tokenizer")


def _check():
    miss = []
    try:
        from tokenizers import ByteLevelBPETokenizer  # noqa
    except ImportError:
        miss.append("tokenizers")
    try:
        from datasets import load_dataset  # noqa
    except ImportError:
        miss.append("datasets")
    if miss:
        LOGGER.error("Missing dependencies. Install with: pip install %s", " ".join(miss))
        sys.exit(1)


_check()

from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from config import train_cfg

VOCAB = 50_000
SAVE = "./tokenizer_data"
MAX_PROMOTED_MERGES = 2_048
PROMOTION_MIN_SCORE = 8
PROMOTION_MAX_SHIFT = 4_096

SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train",
     lambda x: x.get("text", ""), 25000),
    ("wikimedia/wikipedia", "20231101.en", "train",
     lambda x: x.get("text", ""), 15000),
    ("open-phi/textbooks", None, "train",
     lambda x: x.get("markdown", x.get("text", "")), 8000),
    ("HuggingFaceTB/finemath", "finemath-3plus", "train",
     lambda x: x.get("text", ""), 10000),
    ("openai/gsm8k", "main", "train",
     lambda x: str(x.get("question", "")) + " " +
     str(x.get("answer", "")),
     5000),
    ("iamtarun/python_code_instructions_18k_alpaca", None, "train",
     lambda x: str(x.get("instruction", "")) + " " +
     str(x.get("output", "")),
     8000),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, "train",
     lambda x: str(x.get("instruction", "")) + " " +
     str(x.get("response", "")),
     8000),
    ("teknium/OpenHermes-2.5", None, "train",
     lambda x: " ".join(
         t.get("value", "")
         for t in (x.get("conversations") or [])
         if isinstance(t, dict)
     ),
     10000),
    ("Open-Orca/SlimOrca", None, "train",
     lambda x: " ".join(
         t.get("value", "")
         for t in (x.get("conversations") or [])
         if isinstance(t, dict)
     ),
     8000),
    ("allenai/ai2_arc", "ARC-Challenge", "train",
     lambda x: str(x.get("question", "")), 3000),
    ("truthful_qa", "generation", "validation",
     lambda x: str(x.get("question", "")) + " " +
     str(x.get("best_answer", "")),
     1000),
]


def active_sources():
    excluded = set(getattr(train_cfg, "excluded_benchmark_sources", ()))
    if not excluded:
        return list(SOURCES)
    return [s for s in SOURCES if s[0] not in excluded]


def _quiet_external_loggers() -> None:
    """Keep tokenizer runs readable by suppressing chatty transport libraries."""
    noisy = (
        "httpx",
        "httpcore",
        "urllib3",
        "fsspec",
        "datasets",
        "huggingface_hub",
    )
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        from datasets.utils import logging as ds_logging  # type: ignore
        ds_logging.set_verbosity_warning()
    except Exception:
        pass


def _txt(fn, ex, mx=800):
    try:
        t = fn(ex)
        if isinstance(t, str) and len(t.strip()) > 20:
            return t.strip()[:mx]
    except Exception:
        pass
    for k in ("text", "content", "output", "abstract", "code"):
        v = ex.get(k, "")
        if isinstance(v, str) and len(v.strip()) > 20:
            return v.strip()[:mx]
    return ""


def _looks_like_structured_text(text: str) -> bool:
    return bool(
        re.search(
            r"\\(frac|sum|int|alpha|beta|gamma|theta|begin|end)"
            r"|\b(def|class|return|import|lambda|for|while|if|else|try|except)\b"
            r"|[{}\[\]();:=<>+\-*/\\^_]",
            text,
        )
    )


def _normalize_training_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if _looks_like_structured_text(text):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", "    ")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    return " ".join(text.split())


def _merge_priority_score(left: str, right: str) -> int:
    token = left + right
    score = 0
    if re.search(
        r"\b(def|class|return|import|lambda|async|await|for|while|if|else|try|except)\b",
        token,
    ):
        score += 6
    if "__" in token or "->" in token or "::" in token:
        score += 4
    if re.search(r"\\frac|\\sum|\\int|\\alpha|\\beta|\\gamma|\\theta", token):
        score += 6
    if sum(ch in "{}[]()=+-*/\\^_:" for ch in token) >= 2:
        score += 2
    if "_" in token and re.search(r"[A-Za-z]", token):
        score += 2
    if len(token) >= 24:
        score -= 2
    return score


def _boost_code_math_merges(
    merges: List[List[object]],
) -> Tuple[List[List[object]], int]:
    """Promote a bounded set of code/math merges without globally reordering all ranks."""
    scored: List[Tuple[int, List[str], int]] = []
    for rank, item in enumerate(merges):
        pair = item[0]
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        left = str(pair[0])
        right = str(pair[1])
        score = _merge_priority_score(left, right)
        scored.append((rank, [left, right], score))

    if not scored:
        return [], 0

    candidates = sorted(
        [row for row in scored if row[2] >= PROMOTION_MIN_SCORE],
        key=lambda row: (-row[2], row[0]),
    )
    selected_ranks = {row[0] for row in candidates[:MAX_PROMOTED_MERGES]}

    adjusted: List[Tuple[int, int, List[str]]] = []
    for old_rank, pair, score in scored:
        shift = (
            min(PROMOTION_MAX_SHIFT, score * 256)
            if old_rank in selected_ranks
            else 0
        )
        adjusted_rank = max(0, old_rank - shift)
        adjusted.append((adjusted_rank, old_rank, pair))

    adjusted.sort(key=lambda row: (row[0], row[1]))
    boosted: List[List[object]] = []
    promoted = 0
    for new_rank, (_, old_rank, pair) in enumerate(adjusted):
        if new_rank < old_rank:
            promoted += 1
        boosted.append([pair, new_rank])
    return boosted, promoted


def _chars_per_token(tok, text: str) -> float:
    ids = tok.encode(text, add_bos=False)
    return len(text) / max(len(ids), 1)


def collect():
    random.seed(42)
    texts = []
    sources = active_sources()
    LOGGER.info("Collecting from %s sources...", len(sources))
    for path, sub, split, fn, n in sources:
        lab = path.split("/")[-1] + (f"/{sub}" if sub else "")
        try:
            kw = dict(split=split, streaming=True)
            if sub:
                kw["name"] = sub
            ds = load_dataset(path, **kw)
            got = []
            for ex in ds:
                t = _txt(fn, ex)
                if t:
                    normalized = _normalize_training_text(t)
                    if normalized:
                        got.append(normalized)
                if len(got) >= n:
                    break
            texts.extend(got)
            LOGGER.info("OK %-45s %6s", lab, len(got))
        except Exception as e:
            LOGGER.warning("SKIP %-44s %s", lab, str(e)[:40])
    random.shuffle(texts)
    LOGGER.info(
        "%s samples, %.1fM chars",
        f"{len(texts):,}",
        sum(len(t) for t in texts) / 1e6,
    )
    return texts


def train_bpe(texts):
    td = tempfile.mkdtemp(prefix="tok_")
    cf = os.path.join(td, "corpus.txt")
    with open(cf, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    LOGGER.info("Training BPE (vocab=%s)...", f"{VOCAB:,}")
    t0 = time.time()
    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[cf],
        vocab_size=VOCAB,
        min_frequency=2,
        special_tokens=[
            "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
            "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|think|>", "<|answer|>", "<|verify|>", "<|unknown|>",
        ],
    )
    LOGGER.info("%ss | %s tokens", f"{time.time() - t0:.0f}", f"{bpe.get_vocab_size():,}")
    bpe.save_model(td)
    return bpe, td


def save(bpe, td):
    os.makedirs(SAVE, exist_ok=True)
    vocab_path = os.path.join(td, "vocab.json")
    enc = safe_load_json(
        vocab_path,
        max_bytes=64_000_000,
        allowed_roots=[td, os.getcwd()],
    )
    if not isinstance(enc, dict):
        raise ValueError(f"Expected vocab.json object, got {type(enc)!r}")
    with open(os.path.join(td, "merges.txt"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    merges = []
    for r, line in enumerate(lines):
        if line.startswith("#"):
            continue
        p = line.split(" ")
        if len(p) == 2:
            merges.append([[p[0], p[1]], r])

    boosted, promoted = _boost_code_math_merges(merges)

    ep = os.path.join(SAVE, "encoder.json")
    mp = os.path.join(SAVE, "merges.json")
    with open(ep, "w", encoding="utf-8") as f:
        json.dump(enc, f, ensure_ascii=False, indent=1)
    with open(mp, "w") as f:
        json.dump(boosted, f)
    shutil.rmtree(td, ignore_errors=True)
    LOGGER.info("%s (%sKB)", ep, f"{os.path.getsize(ep) / 1e3:.0f}")
    LOGGER.info("%s (%.1fMB)", mp, os.path.getsize(mp) / 1e6)
    LOGGER.info("Vocab: %s Merges: %s Promoted: %s", f"{len(enc):,}", f"{len(boosted):,}", f"{promoted:,}")


def _lcs_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    if n > 500 or m > 500:
        return 1.0 if (a in b or b in a) else 0.5
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            curr[j] = (
                prev[j - 1] + 1
                if a[i - 1] == b[j - 1]
                else max(curr[j - 1], prev[j])
            )
        prev = curr
    return prev[m] / max(n, m)


def verify():
    sys.path.insert(0, ".")
    if "tokenizer" in sys.modules:
        del sys.modules["tokenizer"]
    from tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.load(SAVE)

    cases = [
        ("English", "The quick brown fox jumps over the lazy dog.", True),
        ("Python", "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)", True),
        ("Math", "∫x² dx = x³/3 + C, π ≈ 3.14159", True),
        ("LaTeX", "\\frac{d}{dx}x^2=2x,\\;\\sum_{i=1}^{n}i=\\frac{n(n+1)}{2}", True),
        ("Chat", "<|system|>\nHelp.<|end|>\n<|user|>\nHi<|end|>", False),
        ("Numbers", "$1,234.56 | -273.15°C | 98.7%", True),
        ("Special", "<|think|>Let me reason.<|answer|>42<|verify|>", False),
    ]
    LOGGER.info("Verification:")
    ok_all = True
    warn_count = 0
    hard_cpt_targets: Dict[str, float] = {"Python": 1.4, "LaTeX": 1.4}
    stretch_cpt_targets: Dict[str, float] = {"Python": 3.5, "LaTeX": 3.5}
    for name, text, skip_special in cases:
        ids = tok.encode(text, add_bos=False)
        dec = tok.decode(ids, skip_special=skip_special)
        src = "".join(text.split())
        dst = "".join(dec.split())
        fidelity = _lcs_ratio(src, dst)
        cpt = _chars_per_token(tok, text)
        ok = len(ids) > 0 and fidelity > 0.85

        hard_target = hard_cpt_targets.get(name)
        stretch_target = stretch_cpt_targets.get(name)

        if hard_target is not None and cpt < hard_target:
            ok = False

        warn_only = False
        if ok and stretch_target is not None and cpt < stretch_target:
            warn_only = True
            warn_count += 1

        if not ok:
            ok_all = False

        status = "FAIL"
        if ok and warn_only:
            status = "WARN"
        elif ok:
            status = "OK"

        suffix = ""
        if hard_target is not None:
            suffix += f" (floor>={hard_target:.1f})"
        if stretch_target is not None:
            suffix += f" (goal>={stretch_target:.1f})"

        LOGGER.info(
            "%4s  %-10s %4stok  fidelity=%.2f  chars/token=%.2f%s",
            status,
            name,
            len(ids),
            fidelity,
            cpt,
            suffix,
        )

    summary = "PASS" if ok_all else "FAIL"
    if ok_all and warn_count:
        summary = f"PASS_WITH_WARNINGS ({warn_count} stretch targets)"
    LOGGER.info("%s (vocab=%s)", summary, f"{tok.vocab_size_:,}")


def main():
    configure_logging(level=os.environ.get("MYLLM_LOG_LEVEL", "INFO"))
    _quiet_external_loggers()
    if os.path.exists(os.path.join(SAVE, "encoder.json")):
        LOGGER.info("Exists: %s/", SAVE)
        if input("Retrain? (y/n): ").strip().lower() != "y":
            verify()
            return
    t0 = time.time()
    texts = collect()
    bpe, td = train_bpe(texts)
    save(bpe, td)
    verify()
    LOGGER.info("%.1f min total", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()