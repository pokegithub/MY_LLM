"""
run.py — Master orchestrator for the full SLM pipeline

Usage:
  python run.py tokenizer      Train the BPE tokenizer
  python run.py download       Download and cache training data
    python run.py download-safe  Download with network-safe defaults
        python run.py download-core  Download core-first bootstrap set
    python run.py download-status Show download manifest summary
  python run.py train          Pretrain the base model
  python run.py sft            Supervised fine-tuning
  python run.py dpo            Direct Preference Optimization
    python run.py distill        Distill student from teacher checkpoint
    python run.py quantize       Build quantized export artifacts
    python run.py hw-profile     Show active hardware profile
    python run.py eval-harness   Run evaluation with manifest harness
  python run.py improve        Self-improvement loop
  python run.py eval           Run evaluation suite
    python run.py benchmark-harness Run evaluation + 1x/10x/100x comparison report
  python run.py audit          Run production-readiness audit checks
  python run.py full           Run the entire pipeline end-to-end
  python run.py status         Show pipeline status
"""

import os
import sys
import time
import json
import argparse
import importlib.util
import subprocess

from core.config_manager import ConfigManager
from config import apply_overrides, runtime_config_dict
from security.validator import ValidationError, get_allowed_data_roots, safe_load_json


def apply_runtime_config(
    config_paths=None,
    include_env: bool = True,
    strict: bool = False,
):
    """Load and apply runtime config layers before command dispatch."""
    manager = ConfigManager()
    snapshot = manager.load(config_paths=config_paths or [], include_env=include_env)
    if not snapshot.sources:
        return None

    result = apply_overrides(snapshot.data, strict=strict)

    os.makedirs("./run_artifacts", exist_ok=True)
    stamp = int(time.time())
    snapshot_path = os.path.join(
        "./run_artifacts", f"runtime_config_snapshot_{stamp}.json"
    )
    state_path = os.path.join(
        "./run_artifacts", f"runtime_config_effective_{stamp}.json"
    )

    ConfigManager.save_json(snapshot, snapshot_path)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config_dict(), f, indent=2)

    print("\n" + "=" * 60)
    print("RUNTIME CONFIG")
    print("=" * 60)
    print(f"  sources      : {', '.join(snapshot.sources)}")
    print(f"  config_hash  : {snapshot.config_hash}")
    print(f"  applied_keys : {result['applied']}")
    if result["unknown_sections"]:
        print(
            "  unknown sections: "
            + ", ".join(result["unknown_sections"])
        )
    if result["unknown_fields"]:
        print(
            "  unknown fields  : "
            + ", ".join(result["unknown_fields"])
        )
    print(f"  snapshot     : {os.path.abspath(snapshot_path)}")
    print(f"  effective    : {os.path.abspath(state_path)}")
    print("=" * 60)

    return snapshot


def check_dependencies(command: str):
    """Check required packages for the selected command."""
    required_by_cmd = {
        "tokenizer": {"tokenizers": "tokenizers", "datasets": "datasets"},
        "download": {"numpy": "numpy", "datasets": "datasets"},
        "download-safe": {"numpy": "numpy", "datasets": "datasets"},
        "download-core": {"numpy": "numpy", "datasets": "datasets"},
        "train": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "sft": {"torch": "torch", "numpy": "numpy"},
        "dpo": {"torch": "torch", "numpy": "numpy"},
        "distill": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "quantize": {"torch": "torch", "numpy": "numpy"},
        "hw-profile": {"torch": "torch"},
        "eval-harness": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "improve": {"torch": "torch", "numpy": "numpy"},
        "eval": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "benchmark-harness": {"torch": "torch", "numpy": "numpy", "datasets": "datasets"},
        "full": {"torch": "torch", "numpy": "numpy", "datasets": "datasets", "tokenizers": "tokenizers"},
        "audit": {"torch": "torch", "numpy": "numpy", "tokenizers": "tokenizers"},
        "download-status": {},
        "status": {},
    }
    required = required_by_cmd.get(command, {})
    optional = {
        "regex": "regex",
    }
    missing = []

    def _is_available(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    for name, pkg in required.items():
        if not _is_available(pkg):
            missing.append(name)

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    for name, pkg in optional.items():
        if not _is_available(pkg):
            print(f"  TIP: pip install {name}  (for better tokenization)")


def show_status():
    """Show current pipeline status."""
    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    checks = [
        ("Tokenizer", "./tokenizer_data/encoder.json"),
        ("Data cache", "./data_cache/tokens"),
        ("Val cache", "./data_cache/val_tokens"),
        ("Pretrain ckpt", "./checkpoints"),
        ("SFT ckpt", "./sft_checkpoints"),
        ("DPO ckpt", "./dpo_checkpoints"),
        ("Improved ckpt", "./improved_checkpoints"),
        ("Eval results", "./eval_results/eval_report.json"),
    ]

    for name, path in checks:
        if os.path.exists(path):
            if os.path.isdir(path):
                files = os.listdir(path)
                n = len(files)
                size_mb = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in files
                    if os.path.isfile(os.path.join(path, f))
                ) / 1e6
                print(f"  ✓ {name:<20} {n} files, {size_mb:.1f}MB")
            else:
                size_mb = os.path.getsize(path) / 1e6
                print(f"  ✓ {name:<20} {size_mb:.1f}MB")
        else:
            print(f"  ✗ {name:<20} not found")

    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("\n  NEXT: python run.py tokenizer")
    elif not os.path.exists("./data_cache/tokens"):
        print("\n  NEXT: python run.py download")
    elif not os.path.exists("./checkpoints"):
        print("\n  NEXT: python run.py train")
    elif not os.path.exists("./sft_checkpoints"):
        print("\n  NEXT: python run.py sft  (need SFT data in ./sft_data/)")
    elif not os.path.exists("./dpo_checkpoints"):
        print("\n  NEXT: python run.py dpo  (need DPO data in ./dpo_data/)")
    else:
        print("\n  NEXT: python run.py eval")


def run_tokenizer():
    print("\n" + "=" * 60)
    print("STEP 1: Training Tokenizer")
    print("=" * 60)
    t0 = time.time()

    import train_tokenizer
    train_tokenizer.main()

    elapsed = time.time() - t0
    print(f"\nTokenizer training complete: {elapsed / 60:.1f} min")


def run_download():
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Data")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer not found. Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    import download_data
    download_data.main()

    elapsed = time.time() - t0
    print(f"\nData download complete: {elapsed / 60:.1f} min")


def run_download_safe():
    """Run downloader with safer defaults for flaky networks."""
    os.environ.setdefault("DOWNLOAD_AUTO_PROCEED", "1")
    os.environ.setdefault("DOWNLOAD_NETWORK_SAFE", "1")
    os.environ.setdefault("DOWNLOAD_MAX_SOURCE_FAILURES", "3")
    print("\nUsing network-safe download defaults:")
    print("  DOWNLOAD_AUTO_PROCEED=1")
    print("  DOWNLOAD_NETWORK_SAFE=1")
    print("  DOWNLOAD_MAX_SOURCE_FAILURES=3")
    run_download()


def run_download_core():
    """Run downloader in core-first mode to bootstrap usable data quickly."""
    os.environ.setdefault("DOWNLOAD_AUTO_PROCEED", "1")
    os.environ.setdefault("DOWNLOAD_NETWORK_SAFE", "1")
    os.environ.setdefault("DOWNLOAD_CORE_FIRST", "1")
    os.environ.setdefault("DOWNLOAD_SOURCE_LIMIT", "10")
    os.environ.setdefault("DOWNLOAD_MAX_SOURCE_FAILURES", "2")
    print("\nUsing core-first download defaults:")
    print("  DOWNLOAD_AUTO_PROCEED=1")
    print("  DOWNLOAD_NETWORK_SAFE=1")
    print("  DOWNLOAD_CORE_FIRST=1")
    print("  DOWNLOAD_SOURCE_LIMIT=10")
    print("  DOWNLOAD_MAX_SOURCE_FAILURES=2")
    run_download()


def run_download_status():
    """Print summary from download manifest for operational visibility."""
    manifest_path = os.path.join(".", "data_cache", "download_manifest.json")
    summary = {
        "total_runs": 0,
        "recent": [],
        "fail_counts": {},
    }
    if os.path.exists(manifest_path):
        try:
            manifest = safe_load_json(
                manifest_path,
                max_bytes=8_000_000,
                allowed_roots=get_allowed_data_roots(),
            )
            runs = manifest.get("runs", []) if isinstance(manifest, dict) else []
            recent = runs[-5:]
            fail_counts = {}
            for r in recent:
                for src in r.get("failed_sources", []):
                    fail_counts[src] = fail_counts.get(src, 0) + 1
            summary = {
                "total_runs": len(runs),
                "recent": recent,
                "fail_counts": dict(
                    sorted(fail_counts.items(), key=lambda kv: kv[1], reverse=True)
                ),
            }
        except (ValidationError, OSError, ValueError) as e:
            print(f"\nFailed reading manifest: {str(e)[:120]}")
            return

    print("\n" + "=" * 60)
    print("DOWNLOAD STATUS")
    print("=" * 60)
    print(f"  Manifest: {os.path.abspath(manifest_path)}")
    print(f"  Total runs tracked: {summary['total_runs']}")

    recent = summary.get("recent", [])
    if not recent:
        print("  No download runs recorded yet.")
    else:
        print("  Recent runs:")
        for r in recent:
            run_id = r.get("run_id", "?")
            new_tok = int(r.get("total_new_tokens", 0))
            cache_gb = r.get("cache_gb", "?")
            failed = len(r.get("failed_sources", []))
            print(
                f"    run_id={run_id} | new_tokens={new_tok:,} | "
                f"cache={cache_gb}GB | failed_sources={failed}"
            )

    fail_counts = summary.get("fail_counts", {})
    if fail_counts:
        print("  Top failing sources:")
        for src, cnt in list(fail_counts.items())[:10]:
            print(f"    {src}: {cnt}")
    print("=" * 60)


def run_train():
    print("\n" + "=" * 60)
    print("STEP 3: Pretraining")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer not found. Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    import train
    train.train()

    elapsed = time.time() - t0
    print(f"\nPretraining complete: {elapsed / 60:.1f} min")


def run_sft():
    print("\n" + "=" * 60)
    print("STEP 4: Supervised Fine-Tuning")
    print("=" * 60)

    if not os.path.exists("./sft_data"):
        print("SFT data directory not found: ./sft_data/")
        print("\nExpected format (JSONL):")
        print('  {"messages": [')
        print('    {"role": "system", "content": "You are helpful."},')
        print('    {"role": "user", "content": "Hello!"},')
        print('    {"role": "assistant", "content": "Hi there!"}')
        print("  ]}")
        print("\nCreate ./sft_data/ with your JSONL files and re-run.")

        os.makedirs("./sft_data", exist_ok=True)
        sample = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ]
            },
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful coding assistant.",
                    },
                    {
                        "role": "user",
                        "content": "Write hello world in Python.",
                    },
                    {
                        "role": "assistant",
                        "content": 'print("Hello, World!")',
                    },
                ]
            },
        ]
        with open("./sft_data/sample.jsonl", "w", encoding="utf-8") as f:
            for s in sample:
                f.write(json.dumps(s) + "\n")
        print("\nCreated sample: ./sft_data/sample.jsonl")
        return

    t0 = time.time()
    import sft_trainer
    sft_trainer.sft_train()

    elapsed = time.time() - t0
    print(f"\nSFT complete: {elapsed / 60:.1f} min")


def run_dpo():
    print("\n" + "=" * 60)
    print("STEP 5: Direct Preference Optimization")
    print("=" * 60)

    if not os.path.exists("./dpo_data"):
        print("DPO data directory not found: ./dpo_data/")
        print("\nExpected format (JSONL):")
        print('  {"prompt": "What is AI?",')
        print('   "chosen": "AI is artificial intelligence...",')
        print('   "rejected": "AI is magic..."}')
        print("\nCreate ./dpo_data/ with your JSONL files and re-run.")

        os.makedirs("./dpo_data", exist_ok=True)
        sample = [
            {
                "prompt": "Explain gravity simply.",
                "chosen": "Gravity is the force that pulls objects "
                          "toward each other. The more massive an "
                          "object, the stronger its gravitational pull.",
                "rejected": "Gravity is just what happens. Things fall "
                            "down because they want to.",
            },
        ]
        with open("./dpo_data/sample.jsonl", "w", encoding="utf-8") as f:
            for s in sample:
                f.write(json.dumps(s) + "\n")
        print("\nCreated sample: ./dpo_data/sample.jsonl")
        return

    t0 = time.time()
    import alignment
    alignment.dpo_train()

    elapsed = time.time() - t0
    print(f"\nDPO complete: {elapsed / 60:.1f} min")


def run_distill():
    print("\n" + "=" * 60)
    print("STEP 5.5: Distillation")
    print("=" * 60)
    t0 = time.time()

    import distillation_trainer
    distillation_trainer.distill_train()

    elapsed = time.time() - t0
    print(f"\nDistillation complete: {elapsed / 60:.1f} min")


def run_quantize():
    print("\n" + "=" * 60)
    print("STEP 6.5: Quantization")
    print("=" * 60)
    t0 = time.time()

    import quant_utils
    quant_utils.run_quantization()

    elapsed = time.time() - t0
    print(f"\nQuantization complete: {elapsed / 60:.1f} min")


def run_hw_profile():
    print("\n" + "=" * 60)
    print("STEP 0: Hardware Profile")
    print("=" * 60)
    import hardware_profiles
    hardware_profiles.show_profile()


def run_eval_harness():
    print("\n" + "=" * 60)
    print("STEP 7.5: Evaluation Harness")
    print("=" * 60)
    t0 = time.time()

    from eval_harness.runner import run
    run()

    elapsed = time.time() - t0
    print(f"\nEvaluation harness complete: {elapsed / 60:.1f} min")


def run_improve():
    print("\n" + "=" * 60)
    print("STEP 6: Self-Improvement Loop")
    print("=" * 60)

    t0 = time.time()
    import infinite_improver
    infinite_improver.improve()

    elapsed = time.time() - t0
    print(f"\nSelf-improvement complete: {elapsed / 60:.1f} min")


def run_eval():
    print("\n" + "=" * 60)
    print("STEP 7: Evaluation")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer artifacts not found.")
        print("Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    import eval_suite
    eval_suite.run_evaluation()

    elapsed = time.time() - t0
    print(f"\nEvaluation complete: {elapsed / 60:.1f} min")


def run_benchmark_harness():
    print("\n" + "=" * 60)
    print("STEP 7.2: Benchmark Harness")
    print("=" * 60)

    if not os.path.exists("./tokenizer_data/encoder.json"):
        print("ERROR: Tokenizer artifacts not found.")
        print("Run: python run.py tokenizer")
        sys.exit(1)

    t0 = time.time()
    from eval.benchmark_harness import run_benchmark_harness as _run
    _run()

    elapsed = time.time() - t0
    print(f"\nBenchmark harness complete: {elapsed / 60:.1f} min")


def run_audit():
    """Run quick production-readiness checks."""
    print("\n" + "=" * 60)
    print("PIPELINE AUDIT")
    print("=" * 60)

    checks = [
        (
            "Python compile check",
            [
                sys.executable,
                "-m",
                "compileall",
                "tokenizer.py",
                "alignment.py",
                "sft_trainer.py",
                "eval_suite.py",
                "infinite_improver.py",
                "model.py",
                "run.py",
            ],
        ),
        (
            "Tokenizer smoke test",
            [
                sys.executable,
                "-c",
                (
                    "from tokenizer import BPETokenizer; "
                    "t=BPETokenizer(); "
                    "t.encoder={t.PAD:0,t.BOS:1,t.EOS:2,t.UNK:3,t.USR:4,t.AST:5}; "
                    "t.decoder={v:k for k,v in t.encoder.items()}; "
                    "t.vocab_size_=len(t.encoder); "
                    "t._special_set=set(t.SPECIAL); t._compile_special_pattern(); "
                    "ids=t.encode('<|user|>x<|assistant|>y', add_bos=True); "
                    "print('OK', len(ids))"
                ),
            ],
        ),
    ]

    for name, cmd in checks:
        print(f"  Running: {name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            print(f"\nAUDIT FAILED: {name}")
            sys.exit(1)

    # Config integrity checks for data/eval trustworthiness.
    from config import train_cfg, eval_cfg

    expected_benches = {
        "truthful_qa",
        "google/boolq",
        "allenai/ai2_arc",
        "winogrande",
        "openai/gsm8k",
    }
    excluded = set(getattr(train_cfg, "excluded_benchmark_sources", ()))
    missing = sorted(expected_benches - excluded)
    if missing:
        print(
            "\nAUDIT FAILED: train_cfg.excluded_benchmark_sources "
            f"missing {missing}"
        )
        sys.exit(1)

    if not eval_cfg.strict_real_benchmarks:
        print("\nAUDIT FAILED: eval_cfg.strict_real_benchmarks must be True")
        sys.exit(1)

    if eval_cfg.allow_toy_fallback:
        print("\nAUDIT FAILED: eval_cfg.allow_toy_fallback must be False")
        sys.exit(1)

    threshold_checks = [
        ("min_real_mc_samples", eval_cfg.min_real_mc_samples),
        ("min_real_math_samples", eval_cfg.min_real_math_samples),
        ("min_real_code_samples", eval_cfg.min_real_code_samples),
    ]
    for name, value in threshold_checks:
        if not isinstance(value, int) or value <= 0:
            print(f"\nAUDIT FAILED: eval_cfg.{name} must be a positive integer")
            sys.exit(1)

    print("\nAUDIT PASSED")


def run_full():
    """Run the full pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)
    t_total = time.time()

    stages = [
        ("tokenizer", run_tokenizer),
        ("download", run_download),
        ("train", run_train),
        ("sft", run_sft),
        ("dpo", run_dpo),
        ("improve", run_improve),
        ("eval", run_eval),
    ]

    for name, fn in stages:
        try:
            fn()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            print("Pipeline stopped. Fix the issue and re-run.")
            sys.exit(1)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"FULL PIPELINE COMPLETE: {elapsed / 3600:.1f} hours")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="SLM Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  tokenizer    Train the BPE tokenizer
  download     Download and cache training data
    download-safe Download with network-safe defaults
        download-core Download core-first bootstrap set
    download-status Show download manifest summary
  train        Pretrain the base model
  sft          Supervised fine-tuning
  dpo          Direct Preference Optimization
    distill      Distill student from teacher checkpoint
    quantize     Build quantized export artifacts
    hw-profile   Show active hardware profile
    eval-harness Run evaluation with manifest harness
  improve      Self-improvement loop
  eval         Run evaluation suite
    benchmark-harness Run evaluation + 1x/10x/100x comparison report
  audit        Run production-readiness audit checks
  full         Run full pipeline (tokenizer -> download -> train -> sft -> dpo -> improve -> eval)
  status       Show pipeline status
        """,
    )
    parser.add_argument(
        "command",
        choices=[
            "tokenizer", "download", "download-safe", "download-core", "download-status",
            "train", "sft", "dpo", "distill", "quantize", "hw-profile", "eval-harness",
            "improve", "eval", "benchmark-harness", "audit", "full", "status",
        ],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML/JSON config file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-env-overrides",
        action="store_true",
        help="Disable MYLLM__ environment variable config overrides.",
    )
    parser.add_argument(
        "--strict-config",
        action="store_true",
        help="Fail on unknown config sections or fields.",
    )

    args = parser.parse_args()

    try:
        apply_runtime_config(
            config_paths=args.config,
            include_env=not args.no_env_overrides,
            strict=args.strict_config,
        )
    except Exception as e:
        print(f"ERROR: failed to apply runtime config: {e}")
        sys.exit(1)

    check_dependencies(args.command)

    commands = {
        "tokenizer": run_tokenizer,
        "download": run_download,
        "download-safe": run_download_safe,
        "download-core": run_download_core,
        "download-status": run_download_status,
        "train": run_train,
        "sft": run_sft,
        "dpo": run_dpo,
        "distill": run_distill,
        "quantize": run_quantize,
        "hw-profile": run_hw_profile,
        "eval-harness": run_eval_harness,
        "improve": run_improve,
        "eval": run_eval,
        "benchmark-harness": run_benchmark_harness,
        "audit": run_audit,
        "full": run_full,
        "status": show_status,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
