"""
download_data.py — Dataset downloader with resume support (v3)

Changes from v2:
  - Added validation split download
  - Better progress tracking
  - All exceptions caught as Exception (no bare except)
"""

import os
import sys
import math
import time
import json
import numpy as np
import random
sys.path.insert(0, ".")
from datasets import load_dataset
from tokenizer import BPETokenizer
from config import train_cfg
from security.validator import ValidationError, get_allowed_data_roots, safe_load_json

TOTAL_TOKENS = train_cfg.total_tokens
CACHE_DIR = os.environ.get(
    "DATA_CACHE_DIR", os.path.join(".", "data_cache", "tokens")
)
VAL_CACHE_DIR = os.path.join(".", "data_cache", "val_tokens")
MAX_DISK_GB = 22.0
MAX_PASSES = 5
MIN_KEEP_TOK = 10_000_000
VAL_FRACTION = 0.02  # 2% held out for validation
MAX_STREAM_RETRIES = 5
MAX_LOAD_RETRIES = 3
MANIFEST_PATH = os.path.join(".", "data_cache", "download_manifest.json")
NETWORK_SAFE_FLAKY_SOURCES = {
    "HuggingFaceTB/smollm-corpus",
    "codeparrot/github-code",
    "deepmind/code_contests",
}
CORE_FIRST_SOURCES = {
    "HuggingFaceFW/fineweb-edu",
    "wikimedia/wikipedia",
    "open-phi/textbooks",
    "HuggingFaceTB/finemath",
    "open-web-math/open-web-math",
    "microsoft/orca-math-word-problems-200k",
    "TIGER-Lab/MathInstruct",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    "Open-Orca/SlimOrca",
}

SPLIT_OVERRIDES = {
    "HuggingFaceH4/ultrachat_200k": "train_sft",
    "truthful_qa": "validation",
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


def active_sources():
    excluded = set(getattr(train_cfg, "excluded_benchmark_sources", ()))
    env_skips = {
        s.strip() for s in os.environ.get("DOWNLOAD_SKIP_SOURCES", "").split(",")
        if s.strip()
    }
    excluded |= env_skips
    network_safe = os.environ.get("DOWNLOAD_NETWORK_SAFE", "").strip().lower()
    if network_safe in {"1", "true", "yes", "y"}:
        excluded |= NETWORK_SAFE_FLAKY_SOURCES
    if not excluded:
        sources = list(ALL_SOURCES)
    else:
        sources = [s for s in ALL_SOURCES if s[0] not in excluded]

    core_first = os.environ.get("DOWNLOAD_CORE_FIRST", "").strip().lower()
    if core_first in {"1", "true", "yes", "y"}:
        sources = sorted(
            sources,
            key=lambda s: (0 if s[0] in CORE_FIRST_SOURCES else 1, -s[2], s[0]),
        )

    source_limit_raw = os.environ.get("DOWNLOAD_SOURCE_LIMIT", "").strip()
    if source_limit_raw:
        try:
            source_limit = max(1, int(source_limit_raw))
            sources = sources[:source_limit]
        except ValueError:
            print(
                f"  Warning: invalid DOWNLOAD_SOURCE_LIMIT={source_limit_raw}; "
                "ignoring"
            )

    return sources


def safe_label(path, sub):
    n = path.replace("/", "__")
    if sub:
        n += f"__{sub.replace('/', '_')}"
    return n


def bin_path(path, sub):
    return os.path.join(CACHE_DIR, f"{safe_label(path, sub)}.bin")


def val_bin_path(path, sub):
    return os.path.join(VAL_CACHE_DIR, f"{safe_label(path, sub)}.bin")


def existing_tokens(fp, db):
    if not os.path.exists(fp):
        return 0
    return os.path.getsize(fp) // db


def disk_gb(d):
    if not os.path.exists(d):
        return 0
    return sum(
        os.path.getsize(os.path.join(d, f))
        for f in os.listdir(d)
        if f.endswith(".bin")
    ) / 1e9


def get_text(fn, ex):
    try:
        t = fn(ex)
        if isinstance(t, str) and len(t.strip()) > 30:
            return t.strip()
    except Exception:
        pass
    for k in (
        "text", "content", "code", "output", "markdown",
        "chosen", "instruction", "question", "description",
    ):
        v = ex.get(k, "")
        if isinstance(v, str) and len(v.strip()) > 30:
            return v.strip()
    return ""


def load_ds(path, sub):
    sp = SPLIT_OVERRIDES.get(path, "train")
    kw = dict(split=sp, streaming=True)
    if sub:
        kw["name"] = sub
    last_err = None
    for i in range(MAX_LOAD_RETRIES):
        try:
            return load_dataset(path, **kw)
        except Exception as e:
            last_err = e
            wait_s = min(2 ** i, 8)
            print(
                f"     Dataset load error ({i + 1}/{MAX_LOAD_RETRIES}): "
                f"{str(e)[:100]} | retry in {wait_s}s"
            )
            time.sleep(wait_s)
    raise RuntimeError(f"Failed loading dataset {path}: {str(last_err)[:140]}")


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {"runs": []}
    try:
        obj = safe_load_json(
            MANIFEST_PATH,
            max_bytes=16_000_000,
            allowed_roots=get_allowed_data_roots(),
        )
        if isinstance(obj, dict) and isinstance(obj.get("runs"), list):
            return obj
    except (ValidationError, OSError, ValueError):
        pass
    return {"runs": []}


def _save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def summarize_manifest(max_runs: int = 5) -> dict:
    manifest = _load_manifest()
    runs = manifest.get("runs", [])
    if not runs:
        return {
            "total_runs": 0,
            "recent": [],
            "fail_counts": {},
        }

    recent = runs[-max_runs:]
    fail_counts = {}
    for run in recent:
        for src in run.get("failed_sources", []):
            fail_counts[src] = fail_counts.get(src, 0) + 1
    return {
        "total_runs": len(runs),
        "recent": recent,
        "fail_counts": dict(sorted(fail_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }


tok = None


def download_source(
    path, sub, fn, budget, dtype, out_path, val_path,
    already_have=0
):
    still_need = budget - already_have
    db = 2 if dtype == np.uint16 else 4
    label = path + (f"/{sub}" if sub else "")
    print(f"\n  {label}")
    print(
        f"     Budget:{budget / 1e6:.1f}M  Have:{already_have / 1e6:.1f}M  "
        f"Need:{still_need / 1e6:.1f}M"
    )

    try:
        ds = load_ds(path, sub)
    except Exception as e:
        print(f"     Cannot load: {str(e)[:100]}")
        return 0

    mode = "ab" if already_have > 0 else "wb"
    new_tokens = 0
    val_tokens = 0
    pass_num = 0
    one_pass = None
    t0 = time.time()
    stream_retries = 0

    val_budget = int(budget * VAL_FRACTION)
    val_collected = existing_tokens(val_path, db)

    with open(out_path, mode) as fout, \
            open(val_path, "ab" if val_collected > 0 else "wb") as fval:
        while new_tokens < still_need:
            pass_num += 1
            tpass = 0
            try:
                for ex in ds:
                    text = get_text(fn, ex)
                    if not text:
                        continue
                    ids = tok.encode(text, add_bos=False)
                    ids.append(tok.eos_token_id)

                    # Route some to validation
                    if (
                        val_collected + val_tokens < val_budget
                        and random.random() < VAL_FRACTION
                    ):
                        fval.write(
                            np.array(ids, dtype=dtype).tobytes()
                        )
                        val_tokens += len(ids)
                        continue

                    space = still_need - new_tokens
                    tw = ids[:space]
                    fout.write(np.array(tw, dtype=dtype).tobytes())
                    new_tokens += len(tw)
                    tpass += len(tw)
                    if new_tokens % 5_000_000 < len(tw):
                        el = time.time() - t0
                        rate = new_tokens / el / 1e3 if el > 0 else 0
                        pct = (
                            (already_have + new_tokens) / budget * 100
                        )
                        print(
                            f"     {pct:5.1f}% | +{new_tokens / 1e6:.1f}M | "
                            f"{rate:.0f}Kt/s | pass {pass_num}",
                            flush=True,
                        )
                    if new_tokens >= still_need:
                        break
            except Exception as e:
                stream_retries += 1
                msg = str(e)
                delay_s = min(2 ** stream_retries, 30)
                print(
                    f"     Stream error: {msg[:120]} | "
                    f"retry {stream_retries}/{MAX_STREAM_RETRIES} in {delay_s}s"
                )
                # SSL/cert or closed-client errors tend to be sticky for this stream.
                # Keep partial data and move on after a small number of attempts.
                if (
                    "CERTIFICATE_VERIFY_FAILED" in msg
                    or "client has been closed" in msg
                ) and stream_retries >= 2:
                    print("     Fatal stream state detected — keeping partial data")
                    break
                if stream_retries > MAX_STREAM_RETRIES:
                    print("     Max stream retries reached — keeping partial data")
                    break
                time.sleep(delay_s)
                try:
                    ds = load_ds(path, sub)
                except Exception as reload_e:
                    print(f"     Reload failed: {str(reload_e)[:120]}")
                    break
                continue
            else:
                stream_retries = 0

            if new_tokens < still_need:
                if tpass == 0:
                    print("     Empty — stop")
                    break
                if one_pass is None:
                    one_pass = tpass
                    if one_pass < MIN_KEEP_TOK:
                        print(f"     Only {one_pass / 1e6:.2f}M — as-is")
                        break
                    needed_p = math.ceil(still_need / one_pass)
                    if needed_p > MAX_PASSES:
                        still_need = one_pass * MAX_PASSES
                        print(
                            f"     Cap {MAX_PASSES}x="
                            f"{still_need / 1e6:.0f}M",
                            flush=True,
                        )
                    else:
                        print(f"     Loop {needed_p}x", flush=True)
                if pass_num >= MAX_PASSES:
                    break
                try:
                    ds = load_ds(path, sub)
                except Exception as e:
                    print(f"     Reload failed: {e}")
                    break

    total_now = already_have + new_tokens
    pct = total_now / budget * 100
    print(
        f"     {'OK' if pct >= 95 else 'PARTIAL'}: "
        f"{total_now / 1e6:.1f}M/{budget / 1e6:.1f}M ({pct:.0f}%) "
        f"val:{val_tokens / 1e6:.2f}M"
    )
    return new_tokens


def main():
    global tok
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)
    print("Loading tokenizer...")
    tok = BPETokenizer()
    tok.load(train_cfg.tokenizer_path)
    print(f"  Vocab: {tok.vocab_size_:,}")
    print(f"  Cache: {os.path.abspath(CACHE_DIR)}")
    print(f"  Val:   {os.path.abspath(VAL_CACHE_DIR)}")

    dtype = np.uint16 if tok.vocab_size_ <= 65535 else np.uint32
    db = 2 if dtype == np.uint16 else 4

    sources = active_sources()

    tw = sum(w for _, _, w, _ in sources)
    budgets = {
        safe_label(p, s): int(TOTAL_TOKENS * w / tw)
        for p, s, w, _ in sources
    }

    print(
        f"\nPlan: {TOTAL_TOKENS / 1e9:.1f}B | {len(sources)} sources | "
        f"cap {MAX_DISK_GB}GB"
    )

    skip, resume, new = [], [], []
    for path, sub, w, fn in sources:
        lab = safe_label(path, sub)
        bp = bin_path(path, sub)
        bud = budgets[lab]
        have = existing_tokens(bp, db)
        pct = have / bud * 100 if bud > 0 else 0
        disp = (
            path.split("/")[-1] + (f"/{sub}" if sub else "")
        )[:45]
        if pct >= 95:
            print(
                f"  OK     {disp:<45} "
                f"{bud / 1e6:>6.0f}M  {have / 1e6:>6.0f}M"
            )
            skip.append(lab)
        elif have > 0:
            print(
                f"  RESUME {disp:<45} "
                f"{bud / 1e6:>6.0f}M  {have / 1e6:>6.0f}M ({pct:.0f}%)"
            )
            resume.append((path, sub, w, fn, bud, bp, have))
        else:
            print(f"  NEW    {disp:<45} {bud / 1e6:>6.0f}M")
            new.append((path, sub, w, fn, bud, bp, 0))

    work = resume + new
    print(
        f"\n  Skip:{len(skip)}  Resume:{len(resume)}  New:{len(new)}"
    )
    if not work:
        print("\nAll cached — ready to train!")
        return

    cg = disk_gb(CACHE_DIR)
    ng = sum((b - h) * db for _, _, _, _, b, _, h in work) / 1e9
    print(f"\n  Current:{cg:.1f}GB  Needed:{ng:.1f}GB")
    auto_proceed = os.environ.get("DOWNLOAD_AUTO_PROCEED", "").strip().lower()
    if auto_proceed in {"1", "true", "yes", "y"}:
        print("  Proceed? (y/n): y  [auto]")
        proceed = "y"
    else:
        proceed = input("  Proceed? (y/n): ").strip().lower()
    if proceed != "y":
        print("  Aborted.")
        return

    random.seed(42)
    total_new = 0
    tt = time.time()
    failed_sources = []
    max_source_failures = int(
        os.environ.get("DOWNLOAD_MAX_SOURCE_FAILURES", "999")
    )

    manifest = _load_manifest()
    run_entry = {
        "run_id": int(time.time()),
        "started_at": int(time.time()),
        "auto_proceed": auto_proceed in {"1", "true", "yes", "y"},
        "max_source_failures": max_source_failures,
        "core_first": os.environ.get("DOWNLOAD_CORE_FIRST", "").strip().lower() in {"1", "true", "yes", "y"},
        "source_limit": os.environ.get("DOWNLOAD_SOURCE_LIMIT", "").strip() or None,
        "sources": [],
    }
    manifest["runs"].append(run_entry)
    _save_manifest(manifest)

    for i, (path, sub, w, fn, bud, bp, have) in enumerate(work, 1):
        print(f"\n[{i}/{len(work)}]")
        vp = val_bin_path(path, sub)
        src_t0 = time.time()
        src_disp = path.split("/")[-1] + (f"/{sub}" if sub else "")
        src_status = "unknown"
        src_error = ""
        added_tokens = 0
        try:
            added_tokens = download_source(
                path, sub, fn, bud, dtype, bp, vp,
                already_have=have,
            )
            total_new += added_tokens
            have_now = existing_tokens(bp, db)
            pct_now = have_now / bud * 100 if bud > 0 else 0
            src_status = "ok" if pct_now >= 95 else "partial"
        except Exception as e:
            src_error = str(e)[:140]
            print(f"  Source failed ({src_disp}): {src_error}")
            print("  Continuing with next source...")
            failed_sources.append(src_disp)
            src_status = "failed"

        run_entry["sources"].append({
            "source": path,
            "subset": sub,
            "display": src_disp,
            "status": src_status,
            "budget_tokens": bud,
            "tokens_before": have,
            "tokens_added": added_tokens,
            "tokens_after": existing_tokens(bp, db),
            "elapsed_sec": round(time.time() - src_t0, 2),
            "error": src_error,
        })
        _save_manifest(manifest)

        if len(failed_sources) >= max_source_failures:
            print(
                f"\nFailure budget reached ({len(failed_sources)}/"
                f"{max_source_failures}) — stopping early"
            )
            break

        if disk_gb(CACHE_DIR) > MAX_DISK_GB:
            print("\nDisk cap — rest will stream")
            break

    print(
        f"\nDone: {total_new / 1e9:.3f}B tokens | "
        f"{disk_gb(CACHE_DIR):.1f}GB | "
        f"{(time.time() - tt) / 60:.1f}min"
    )
    run_entry["finished_at"] = int(time.time())
    run_entry["total_new_tokens"] = int(total_new)
    run_entry["cache_gb"] = round(disk_gb(CACHE_DIR), 3)
    run_entry["failed_sources"] = failed_sources
    _save_manifest(manifest)
    print(f"  Manifest: {os.path.abspath(MANIFEST_PATH)}")
    if failed_sources:
        print("  Failed sources (partial/skip): " + ", ".join(failed_sources))


if __name__ == "__main__":
    main()