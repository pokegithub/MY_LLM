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
import hashlib
import numpy as np
import random
sys.path.insert(0, ".")
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
TOKEN_ARTIFACT_MANIFEST_PATH = os.path.join(
    ".", "data_cache", "token_artifacts_manifest.json"
)
SOURCE_LICENSE_METADATA_PATH = os.path.join(
    ".", "data_governance", "source_license_metadata.json"
)
ALLOWED_CORPUS_MANIFEST_PATH = os.path.join(
    ".", "data_governance", "allowed_corpus_manifest_v1.json"
)
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

SOURCE_GOVERNANCE_METADATA = {
    "HuggingFaceFW/fineweb-edu": {"category": "web", "domain": "general_web"},
    "wikimedia/wikipedia": {"category": "reference", "domain": "encyclopedic"},
    "HuggingFaceTB/smollm-corpus": {"category": "synthetic_mix", "domain": "mixed"},
    "open-phi/textbooks": {"category": "textbook", "domain": "education"},
    "HuggingFaceTB/finemath": {"category": "math", "domain": "education"},
    "open-web-math/open-web-math": {"category": "math", "domain": "web"},
    "openai/gsm8k": {"category": "benchmark", "domain": "math"},
    "microsoft/orca-math-word-problems-200k": {"category": "math", "domain": "synthetic_instruction"},
    "TIGER-Lab/MathInstruct": {"category": "math", "domain": "instruction"},
    "lighteval/MATH-Hard": {"category": "benchmark_like", "domain": "math"},
    "m-a-p/CodeFeedback-Filtered-Instruction": {"category": "code", "domain": "instruction"},
    "ise-uiuc/Magicoder-Evol-Instruct-110K": {"category": "code", "domain": "instruction"},
    "iamtarun/python_code_instructions_18k_alpaca": {"category": "code", "domain": "instruction"},
    "codeparrot/github-code": {"category": "code", "domain": "source_code"},
    "ajibawa-2023/Code-290k-ShareGPT": {"category": "code", "domain": "chat_instruction"},
    "bigcode/self-oss-instruct-sc2-exec-filter-50k": {"category": "code", "domain": "instruction"},
    "nickrosh/Evol-Instruct-Code-80k-v1": {"category": "code", "domain": "instruction"},
    "deepmind/code_contests": {"category": "benchmark_like", "domain": "code"},
    "code-search-net/code_search_net": {"category": "code", "domain": "source_code"},
    "b-mc2/sql-create-context": {"category": "code", "domain": "sql"},
    "argilla/magpie-ultra-v0.1": {"category": "instruction", "domain": "synthetic_chat"},
    "HuggingFaceH4/ultrachat_200k": {"category": "instruction", "domain": "chat"},
    "teknium/OpenHermes-2.5": {"category": "instruction", "domain": "chat"},
    "Open-Orca/SlimOrca": {"category": "instruction", "domain": "chat"},
    "google/boolq": {"category": "benchmark", "domain": "qa"},
    "truthful_qa": {"category": "benchmark", "domain": "truthfulness"},
    "Anthropic/hh-rlhf": {"category": "preference", "domain": "chat"},
    "allenai/ai2_arc": {"category": "benchmark", "domain": "science_qa"},
    "winogrande": {"category": "benchmark", "domain": "commonsense"},
}


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


def dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def config_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_artifact_record(
    *,
    path: str,
    source: str,
    subset,
    dataset_split: str,
    artifact_split: str,
    dtype,
    config_digest: str | None = None,
) -> dict:
    dt = np.dtype(dtype)
    exists = os.path.exists(path)
    size_bytes = os.path.getsize(path) if exists else 0
    if exists and size_bytes % dt.itemsize != 0:
        raise ValueError(
            f"Token artifact byte size is not divisible by dtype width: {path}"
        )
    record = {
        "path": path,
        "exists": exists,
        "source": source,
        "subset": subset,
        "dataset_split": dataset_split,
        "artifact_split": artifact_split,
        "dtype": dt.name,
        "itemsize": dt.itemsize,
        "size_bytes": size_bytes,
        "token_count": size_bytes // dt.itemsize,
    }
    if config_digest is not None:
        record["config_hash"] = config_digest
    return record


def write_token_artifact_manifest(
    records: list[dict],
    manifest_path: str,
    metadata: dict | None = None,
) -> str:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    payload = {
        "schema": "token_artifacts_manifest_v1",
        "created_at": int(time.time()),
        "artifacts": records,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def build_token_artifact_records(sources, dtype, config_digest: str | None = None):
    records = []
    for path, sub, _, _ in sources:
        dataset_split = SPLIT_OVERRIDES.get(path, "train")
        records.append(
            token_artifact_record(
                path=bin_path(path, sub),
                source=path,
                subset=sub,
                dataset_split=dataset_split,
                artifact_split="train",
                dtype=dtype,
                config_digest=config_digest,
            )
        )
        records.append(
            token_artifact_record(
                path=val_bin_path(path, sub),
                source=path,
                subset=sub,
                dataset_split=dataset_split,
                artifact_split="validation",
                dtype=dtype,
                config_digest=config_digest,
            )
        )
    return records


def _known_artifact_labels() -> dict[str, tuple[str, object, str]]:
    labels = {}
    for path, sub, _, _ in ALL_SOURCES:
        labels[safe_label(path, sub)] = (
            path,
            sub,
            SPLIT_OVERRIDES.get(path, "train"),
        )
    return labels


def _infer_token_dtype(tokenizer_path: str):
    tok = BPETokenizer()
    tok.load(tokenizer_path)
    return np.uint16 if tok.vocab_size_ <= 65535 else np.uint32, {
        "method": "tokenizer_vocab_size",
        "tokenizer_path": tokenizer_path,
        "vocab_size": tok.vocab_size_,
    }


def reconstruct_token_artifact_manifest(
    *,
    token_cache_dir: str = CACHE_DIR,
    val_cache_dir: str = VAL_CACHE_DIR,
    manifest_path: str = TOKEN_ARTIFACT_MANIFEST_PATH,
    tokenizer_path: str = "./tokenizer_data",
    dtype=None,
) -> dict:
    """Rebuild a truthful manifest from local token binaries.

    This records only file-derived facts and source IDs inferable from the
    artifact filename. It does not reconstruct original license decisions,
    source filtering decisions, shuffle order, or deduplication provenance.
    """
    if dtype is None:
        dtype, dtype_meta = _infer_token_dtype(tokenizer_path)
    else:
        dtype = np.dtype(dtype)
        dtype_meta = {
            "method": "caller_supplied",
            "dtype": dtype.name,
        }

    known = _known_artifact_labels()
    records: list[dict] = []
    unknown_source_count = 0

    for directory, artifact_split in (
        (token_cache_dir, "train"),
        (val_cache_dir, "validation"),
    ):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".bin"):
                continue
            label = name[:-4]
            source, subset, dataset_split = known.get(
                label,
                ("unknown", "unknown", "unknown"),
            )
            source_status = "matched_known_source"
            if source == "unknown":
                source_status = "unknown_filename"
                unknown_source_count += 1
            record = token_artifact_record(
                path=os.path.join(directory, name),
                source=source,
                subset=subset,
                dataset_split=dataset_split,
                artifact_split=artifact_split,
                dtype=dtype,
            )
            record["provenance_status"] = "reconstructed_from_local_cache"
            record["source_identifier_status"] = source_status
            record["provenance_limitations"] = [
                "file size and token count are verified from local binary size",
                "source/subset are inferred from filename only when recognized",
                "original filtering, license, dedup, shuffle, and contamination decisions are not reconstructed",
            ]
            records.append(record)

    metadata = {
        "provenance_status": "reconstructed_from_local_cache",
        "dtype_inference": dtype_meta,
        "artifact_count": len(records),
        "unknown_source_count": unknown_source_count,
        "limitations": [
            "manifest reconstructed after artifact creation",
            "does not prove dataset quality, licensing, deduplication, or benchmark decontamination",
        ],
    }
    written = write_token_artifact_manifest(records, manifest_path, metadata=metadata)
    return {
        "manifest_path": written,
        "artifact_count": len(records),
        "unknown_source_count": unknown_source_count,
        "metadata": metadata,
    }


def _sample_file_sha256(path: str, block_size: int = 65536) -> dict:
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    mode = "full" if size <= block_size * 2 else "head_tail_sample"
    with open(path, "rb") as handle:
        if mode == "full":
            digest.update(handle.read())
        else:
            digest.update(handle.read(block_size))
            handle.seek(max(0, size - block_size))
            digest.update(handle.read(block_size))
    return {
        "hash": digest.hexdigest(),
        "mode": mode,
        "block_size": block_size,
        "is_full_file_hash": mode == "full",
    }


def validate_token_artifact_manifest(
    *,
    manifest_path: str = TOKEN_ARTIFACT_MANIFEST_PATH,
    hash_block_size: int = 65536,
) -> dict:
    """Inspect token artifacts without claiming data quality or licensing proof."""
    report = {
        "ok": False,
        "manifest_path": manifest_path,
        "schema": "unknown",
        "artifact_count": 0,
        "existing_artifact_count": 0,
        "failed_artifact_count": 0,
        "missing_train_val_pairs": [],
        "hash_scope": "full for small files, head/tail sample for larger files",
        "quality_claim": "none",
        "artifacts": [],
        "errors": [],
    }
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"manifest_load_failed: {exc}")
        return report

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        report["errors"].append("manifest artifacts field is missing or not a list")
        return report

    report["schema"] = str(manifest.get("schema", "unknown"))
    report["artifact_count"] = len(artifacts)
    split_pairs: dict[tuple[str, str], set[str]] = {}

    for item in artifacts:
        result = {
            "path": item.get("path") if isinstance(item, dict) else None,
            "source": item.get("source", "unknown") if isinstance(item, dict) else "unknown",
            "subset": item.get("subset", "unknown") if isinstance(item, dict) else "unknown",
            "artifact_split": item.get("artifact_split", "unknown") if isinstance(item, dict) else "unknown",
            "status": "pass",
            "errors": [],
        }
        if not isinstance(item, dict):
            result["status"] = "fail"
            result["errors"].append("artifact record is not an object")
            report["artifacts"].append(result)
            continue

        path = str(item.get("path", ""))
        dtype_name_value = item.get("dtype", "uint16")
        try:
            dtype = np.dtype(dtype_name_value)
        except (TypeError, ValueError) as exc:
            result["status"] = "fail"
            result["errors"].append(f"invalid dtype: {exc}")
            report["artifacts"].append(result)
            continue

        if not os.path.isfile(path):
            result["status"] = "fail"
            result["errors"].append("artifact file is missing")
            report["artifacts"].append(result)
            continue

        report["existing_artifact_count"] += 1
        size_bytes = os.path.getsize(path)
        token_count = size_bytes // dtype.itemsize
        result.update({
            "size_bytes": size_bytes,
            "dtype": dtype.name,
            "itemsize": dtype.itemsize,
            "token_count": token_count,
        })
        if size_bytes % dtype.itemsize != 0:
            result["status"] = "fail"
            result["errors"].append("size is not divisible by dtype itemsize")
        if int(item.get("size_bytes", size_bytes)) != size_bytes:
            result["status"] = "fail"
            result["errors"].append("manifest size_bytes does not match file")
        if int(item.get("token_count", token_count)) != token_count:
            result["status"] = "fail"
            result["errors"].append("manifest token_count does not match file")

        checksum = _sample_file_sha256(path, block_size=hash_block_size)
        result["sha256"] = checksum

        key = (str(item.get("source", "unknown")), str(item.get("subset", "unknown")))
        split_pairs.setdefault(key, set()).add(str(item.get("artifact_split", "unknown")))
        report["artifacts"].append(result)

    for (source, subset), splits in sorted(split_pairs.items()):
        if {"train", "validation"} - splits:
            report["missing_train_val_pairs"].append({
                "source": source,
                "subset": subset,
                "present_splits": sorted(splits),
            })

    report["failed_artifact_count"] = sum(
        1 for item in report["artifacts"] if item.get("status") == "fail"
    )
    report["ok"] = report["failed_artifact_count"] == 0 and report["artifact_count"] > 0
    return report


MANUAL_LICENSE_ALLOWED = {
    "license_evidence_source": {
        "manual_repo_metadata",
        "manual_dataset_card_metadata",
        "manual_upstream_legal_page",
        "manual_dataset_card_and_upstream_legal_page",
        "unknown",
    },
    "review_basis": {
        "not_reviewed",
        "manual_source_page_review",
        "manual_dataset_card_review",
        "manual_repo_policy_decision",
    },
    "governance_classification": {
        "unknown",
        "needs_manual_review",
        "excluded_from_training",
        "documented_but_unreviewed",
        "policy_allowed_but_not_legally_cleared",
    },
    "repo_policy_status": {
        "unspecified",
        "excluded",
        "blocked_pending_review",
        "allowed_by_repo_policy",
        "restricted",
    },
    "training_blocker_level": {"hard_blocker", "caution", "informational"},
    "benchmark_risk_status": {
        "not_flagged",
        "benchmark_excluded",
        "benchmark_adjacent_needs_review",
    },
    "intended_training_role": {
        "none",
        "pending_review_only",
        "reference_only",
        "instruction_sft_candidate",
        "preference_optimization_only",
    },
    "legal_clearance_claim": {"none"},
}


def _source_metadata_key(source: str, subset) -> str:
    normalized_subset = "null" if subset is None else str(subset)
    return f"{source}::{normalized_subset}"


def load_manual_source_license_metadata(
    metadata_path: str = SOURCE_LICENSE_METADATA_PATH,
) -> dict:
    """Load manually curated source metadata without treating it as legal clearance."""
    report = {
        "schema": "manual_source_license_metadata_load_v1",
        "path": metadata_path,
        "exists": os.path.isfile(metadata_path),
        "ok": False,
        "record_count": 0,
        "records": [],
        "records_by_key": {},
        "errors": [],
        "legal_clearance_claim": "none",
    }
    if not report["exists"]:
        report["errors"].append("metadata file is missing")
        return report
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"metadata_load_failed: {exc}")
        return report
    if not isinstance(payload, dict):
        report["errors"].append("metadata payload is not an object")
        return report
    if payload.get("legal_clearance_claim") != "none":
        report["errors"].append("metadata file attempted non-none legal clearance claim")
    records = payload.get("records")
    if not isinstance(records, list):
        report["errors"].append("metadata records field is missing or not a list")
        return report

    normalized_records = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            report["errors"].append(f"record[{index}] is not an object")
            continue
        source_id = str(raw.get("source_id", "")).strip()
        if not source_id:
            report["errors"].append(f"record[{index}] has empty source_id")
            continue
        subset = raw.get("subset")
        declared_license = str(raw.get("declared_license", "unknown")).strip() or "unknown"
        record = {
            "source_id": source_id,
            "subset": subset,
            "source_category": str(raw.get("source_category", "unknown") or "unknown"),
            "source_domain": str(raw.get("source_domain", "unknown") or "unknown"),
            "declared_license": declared_license,
            "license_evidence_source": str(
                raw.get("license_evidence_source", "unknown") or "unknown"
            ),
            "review_basis": str(raw.get("review_basis", "not_reviewed") or "not_reviewed"),
            "governance_classification": str(
                raw.get("governance_classification", "unknown") or "unknown"
            ),
            "repo_policy_status": str(
                raw.get("repo_policy_status", "unspecified") or "unspecified"
            ),
            "training_blocker_level": str(
                raw.get("training_blocker_level", "hard_blocker") or "hard_blocker"
            ),
            "benchmark_risk_status": str(
                raw.get("benchmark_risk_status", "not_flagged") or "not_flagged"
            ),
            "intended_training_role": str(
                raw.get("intended_training_role", "none") or "none"
            ),
            "provenance_evidence": raw.get("provenance_evidence", []),
            "exclusion_rationale": str(raw.get("exclusion_rationale", "")),
            "notes": str(raw.get("notes", "")),
            "legal_clearance_claim": str(
                raw.get("legal_clearance_claim", "none") or "none"
            ),
            "metadata_evidence_status": "manual_metadata_present",
        }
        for field, allowed in MANUAL_LICENSE_ALLOWED.items():
            if record[field] not in allowed:
                report["errors"].append(
                    f"record[{index}] {field} has invalid value {record[field]!r}"
                )
        if declared_license != "unknown" and record["license_evidence_source"] == "unknown":
            report["errors"].append(
                f"record[{index}] documents a license without evidence source"
            )
        if not isinstance(record["provenance_evidence"], list):
            report["errors"].append(
                f"record[{index}] provenance_evidence must be a list"
            )
        elif not all(isinstance(item, str) and item.strip() for item in record["provenance_evidence"]):
            report["errors"].append(
                f"record[{index}] provenance_evidence must contain non-empty strings"
            )
        key = _source_metadata_key(source_id, subset)
        if key in report["records_by_key"]:
            report["errors"].append(f"duplicate metadata record for {key}")
        report["records_by_key"][key] = record
        normalized_records.append(record)

    report["records"] = normalized_records
    report["record_count"] = len(normalized_records)
    report["ok"] = not report["errors"] and report["record_count"] > 0
    return report


def load_allowed_corpus_manifest(
    manifest_path: str = ALLOWED_CORPUS_MANIFEST_PATH,
    *,
    metadata_path: str = SOURCE_LICENSE_METADATA_PATH,
) -> dict:
    """Load the tiny repo-policy-allowed subset without implying legal clearance."""
    report = {
        "schema": "allowed_corpus_manifest_load_v1",
        "path": manifest_path,
        "exists": os.path.isfile(manifest_path),
        "ok": False,
        "allowed_source_count": 0,
        "reviewed_candidate_source_count": 0,
        "allowed_sources": [],
        "reviewed_candidates": [],
        "errors": [],
        "legal_clearance_claim": "none",
    }
    if not report["exists"]:
        report["errors"].append("allowed corpus manifest is missing")
        return report

    try:
        payload = safe_load_json(
            manifest_path,
            max_bytes=2_000_000,
            allowed_roots=get_allowed_data_roots(),
        )
    except (ValidationError, OSError, ValueError) as exc:
        report["errors"].append(f"allowed_corpus_manifest_load_failed: {exc}")
        return report

    if not isinstance(payload, dict):
        report["errors"].append("allowed corpus manifest payload is not an object")
        return report
    if payload.get("schema") != "allowed_corpus_manifest_v1":
        report["errors"].append("allowed corpus manifest schema must be allowed_corpus_manifest_v1")
    if payload.get("legal_clearance_claim") != "none":
        report["errors"].append("allowed corpus manifest attempted non-none legal clearance claim")

    reviewed_candidates = payload.get("reviewed_candidates")
    allowed_sources = payload.get("allowed_sources")
    if not isinstance(reviewed_candidates, list):
        report["errors"].append("reviewed_candidates field is missing or not a list")
        return report
    if not isinstance(allowed_sources, list):
        report["errors"].append("allowed_sources field is missing or not a list")
        return report

    report["reviewed_candidates"] = reviewed_candidates
    report["allowed_sources"] = allowed_sources
    report["reviewed_candidate_source_count"] = len(reviewed_candidates)
    report["allowed_source_count"] = len(allowed_sources)

    metadata = load_manual_source_license_metadata(metadata_path)
    metadata_by_key = metadata.get("records_by_key", {})
    configured_exclusions = set(getattr(train_cfg, "excluded_benchmark_sources", ()))

    seen_reviewed = set()
    for index, item in enumerate(reviewed_candidates):
        if not isinstance(item, dict):
            report["errors"].append(f"reviewed_candidates[{index}] is not an object")
            continue
        source_id = str(item.get("source_id", "")).strip()
        subset = item.get("subset")
        if not source_id:
            report["errors"].append(f"reviewed_candidates[{index}] has empty source_id")
            continue
        key = _source_metadata_key(source_id, subset)
        if key in seen_reviewed:
            report["errors"].append(f"duplicate reviewed candidate for {key}")
        seen_reviewed.add(key)
        status = str(item.get("repo_policy_status", "")).strip()
        if status not in MANUAL_LICENSE_ALLOWED["repo_policy_status"]:
            report["errors"].append(
                f"reviewed_candidates[{index}] repo_policy_status has invalid value {status!r}"
            )
        metadata_record = metadata_by_key.get(key)
        if metadata_record is None:
            report["errors"].append(f"reviewed_candidates[{index}] missing metadata record for {key}")
            continue
        if metadata_record["repo_policy_status"] != status:
            report["errors"].append(
                f"reviewed_candidates[{index}] repo_policy_status does not match source metadata for {key}"
            )

    seen_allowed = set()
    for index, item in enumerate(allowed_sources):
        if not isinstance(item, dict):
            report["errors"].append(f"allowed_sources[{index}] is not an object")
            continue
        source_id = str(item.get("source_id", "")).strip()
        subset = item.get("subset")
        if not source_id:
            report["errors"].append(f"allowed_sources[{index}] has empty source_id")
            continue
        key = _source_metadata_key(source_id, subset)
        if key in seen_allowed:
            report["errors"].append(f"duplicate allowed source for {key}")
        seen_allowed.add(key)
        metadata_record = metadata_by_key.get(key)
        if metadata_record is None:
            report["errors"].append(f"allowed_sources[{index}] missing metadata record for {key}")
            continue
        if metadata_record["repo_policy_status"] != "allowed_by_repo_policy":
            report["errors"].append(
                f"allowed_sources[{index}] is not allowed_by_repo_policy in source metadata for {key}"
            )
        if metadata_record["declared_license"] == "unknown":
            report["errors"].append(
                f"allowed_sources[{index}] has unknown declared_license in source metadata for {key}"
            )
        if source_id in configured_exclusions:
            report["errors"].append(
                f"allowed_sources[{index}] uses configured benchmark exclusion source {source_id}"
            )
        if metadata_record["legal_clearance_claim"] != "none":
            report["errors"].append(
                f"allowed_sources[{index}] attempted non-none legal clearance claim for {key}"
            )

    report["ok"] = not report["errors"]
    return report


def _source_governance_metadata(source: str) -> dict:
    return dict(SOURCE_GOVERNANCE_METADATA.get(source, {}))


def _source_governance_classification(
    *,
    active_now: bool,
    benchmark_risk: bool,
    declared_license,
    metadata_classification: str = "unknown",
    repo_policy_status: str = "unspecified",
) -> str:
    if repo_policy_status == "excluded":
        return "excluded_from_training"
    if not active_now:
        return "excluded_from_training"
    if benchmark_risk:
        return "excluded_from_training"
    if repo_policy_status == "allowed_by_repo_policy":
        return "policy_allowed_but_not_legally_cleared"
    if repo_policy_status == "blocked_pending_review":
        return "needs_manual_review"
    if metadata_classification in MANUAL_LICENSE_ALLOWED["governance_classification"]:
        if metadata_classification != "unknown":
            return metadata_classification
    if declared_license and declared_license != "unknown":
        return "known_but_unreviewed"
    return "needs_manual_review"


def _training_blocker_level(
    *,
    active_now: bool,
    repo_policy_status: str,
    declared_license,
    metadata_level: str = "hard_blocker",
) -> str:
    if repo_policy_status in {"excluded", "blocked_pending_review"}:
        return "hard_blocker"
    if not active_now:
        return "hard_blocker"
    if declared_license in {None, "unknown", ""}:
        return "hard_blocker"
    if repo_policy_status == "allowed_by_repo_policy":
        return "caution"
    if repo_policy_status == "restricted":
        return "caution"
    if metadata_level in MANUAL_LICENSE_ALLOWED["training_blocker_level"]:
        return metadata_level
    return "hard_blocker"


def build_dataset_source_manifest(
    *,
    metadata_path: str = SOURCE_LICENSE_METADATA_PATH,
) -> dict:
    """Build source governance evidence from the local registry only."""
    manual_metadata = load_manual_source_license_metadata(metadata_path)
    metadata_by_key = manual_metadata.get("records_by_key", {})
    active = {
        (path, sub)
        for path, sub, _, _ in active_sources()
    }
    configured_exclusions = set(
        getattr(train_cfg, "excluded_benchmark_sources", ())
    )
    records = []
    missing_metadata_count = 0
    for source, subset, weight, _ in ALL_SOURCES:
        benchmark_risk = source in configured_exclusions
        active_now = (source, subset) in active
        fallback = _source_governance_metadata(source)
        metadata = metadata_by_key.get(_source_metadata_key(source, subset))
        metadata_status = "manual_metadata_present"
        if metadata is None:
            metadata_status = "missing_manual_metadata"
            missing_metadata_count += 1
            metadata = {
                "source_category": fallback.get("category", "unknown"),
                "source_domain": fallback.get("domain", "unknown"),
                "declared_license": "unknown",
                "license_evidence_source": "unknown",
                "review_basis": "not_reviewed",
                "governance_classification": "unknown",
                "repo_policy_status": "blocked_pending_review",
                "training_blocker_level": "hard_blocker",
                "notes": "manual metadata entry missing",
                "legal_clearance_claim": "none",
            }
        declared_license = metadata.get("declared_license", "unknown") or "unknown"
        repo_policy_status = metadata.get("repo_policy_status", "unspecified")
        governance_classification = _source_governance_classification(
            active_now=active_now,
            benchmark_risk=benchmark_risk,
            declared_license=declared_license,
            metadata_classification=metadata.get("governance_classification", "unknown"),
            repo_policy_status=repo_policy_status,
        )
        training_blocker_level = _training_blocker_level(
            active_now=active_now,
            repo_policy_status=repo_policy_status,
            declared_license=declared_license,
            metadata_level=metadata.get("training_blocker_level", "hard_blocker"),
        )
        license_status = (
            "known_but_unreviewed" if declared_license != "unknown" else "unknown"
        )
        records.append({
            "source": source,
            "subset": subset,
            "dataset_split": SPLIT_OVERRIDES.get(source, "train"),
            "weight": weight,
            "source_category": metadata.get("source_category", fallback.get("category", "unknown")),
            "source_domain": metadata.get("source_domain", fallback.get("domain", "unknown")),
            "active_in_current_config": active_now,
            "excluded_by_current_config": not active_now,
            "declared_license": declared_license,
            "license_status": license_status,
            "license_review_status": metadata.get("review_basis", "not_reviewed"),
            "license_evidence_source": metadata.get("license_evidence_source", "unknown"),
            "review_basis": metadata.get("review_basis", "not_reviewed"),
            "metadata_evidence_status": metadata_status,
            "governance_classification": governance_classification,
            "repo_policy_status": repo_policy_status,
            "training_blocker_level": training_blocker_level,
            "benchmark_risk_status": metadata.get(
                "benchmark_risk_status",
                "benchmark_excluded" if benchmark_risk else "not_flagged",
            ),
            "intended_training_role": metadata.get("intended_training_role", "none"),
            "provenance_evidence": list(metadata.get("provenance_evidence", [])),
            "exclusion_rationale": metadata.get("exclusion_rationale", ""),
            "notes": metadata.get("notes", ""),
            "legal_clearance_claim": "none",
            "benchmark_risk": benchmark_risk,
            "source_level_contamination_risk": "source_id_overlap"
            if benchmark_risk
            else "not_flagged_by_configured_source_ids",
            "benchmark_risk_basis": (
                "listed in train_cfg.excluded_benchmark_sources"
                if benchmark_risk
                else "not listed in train_cfg.excluded_benchmark_sources"
            ),
        })

    governance_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    repo_policy_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    declared_license_counts: dict[str, int] = {}
    evidence_counts: dict[str, int] = {}
    hard_blockers = []
    allowed_by_repo_policy = []
    for item in records:
        governance_counts[item["governance_classification"]] = (
            governance_counts.get(item["governance_classification"], 0) + 1
        )
        category_counts[item["source_category"]] = (
            category_counts.get(item["source_category"], 0) + 1
        )
        domain_counts[item["source_domain"]] = (
            domain_counts.get(item["source_domain"], 0) + 1
        )
        repo_policy_counts[item["repo_policy_status"]] = (
            repo_policy_counts.get(item["repo_policy_status"], 0) + 1
        )
        blocker_counts[item["training_blocker_level"]] = (
            blocker_counts.get(item["training_blocker_level"], 0) + 1
        )
        declared_license_counts[item["declared_license"]] = (
            declared_license_counts.get(item["declared_license"], 0) + 1
        )
        evidence_counts[item["license_evidence_source"]] = (
            evidence_counts.get(item["license_evidence_source"], 0) + 1
        )
        blocker_summary = {
            "source": item["source"],
            "subset": item["subset"],
            "source_category": item["source_category"],
            "repo_policy_status": item["repo_policy_status"],
            "governance_classification": item["governance_classification"],
            "training_blocker_level": item["training_blocker_level"],
            "reason": item["notes"],
        }
        if item["training_blocker_level"] == "hard_blocker":
            hard_blockers.append(blocker_summary)
        if item["repo_policy_status"] == "allowed_by_repo_policy":
            allowed_by_repo_policy.append(blocker_summary)

    known_license_count = sum(
        1 for item in records if item["license_status"] == "known_but_unreviewed"
    )
    manual_metadata_matched = sum(
        1 for item in records if item["metadata_evidence_status"] == "manual_metadata_present"
    )
    active_hard_blockers = [
        item for item in records
        if item["active_in_current_config"]
        and item["training_blocker_level"] == "hard_blocker"
    ]
    return {
        "schema": "dataset_source_governance_v1",
        "ok": not bool(manual_metadata.get("errors")),
        "source_count": len(records),
        "active_source_count": sum(
            1 for item in records if item["active_in_current_config"]
        ),
        "excluded_source_count": sum(
            1 for item in records if item["excluded_by_current_config"]
        ),
        "known_license_count": known_license_count,
        "documented_license_count": known_license_count,
        "unknown_license_count": len(records) - known_license_count,
        "manual_metadata": {
            "path": metadata_path,
            "exists": bool(manual_metadata.get("exists")),
            "ok": bool(manual_metadata.get("ok")),
            "record_count": int(manual_metadata.get("record_count", 0)),
            "matched_source_count": manual_metadata_matched,
            "missing_source_metadata_count": missing_metadata_count,
            "coverage_ratio": round(manual_metadata_matched / max(1, len(records)), 8),
            "errors": list(manual_metadata.get("errors", [])),
            "legal_clearance_claim": "none",
        },
        "governance_classification_counts": dict(sorted(governance_counts.items())),
        "repo_policy_status_counts": dict(sorted(repo_policy_counts.items())),
        "training_blocker_level_counts": dict(sorted(blocker_counts.items())),
        "declared_license_counts": dict(sorted(declared_license_counts.items())),
        "source_category_counts": dict(sorted(category_counts.items())),
        "source_domain_counts": dict(sorted(domain_counts.items())),
        "license_evidence_source_counts": dict(sorted(evidence_counts.items())),
        "license_evidence_sources": sorted(evidence_counts),
        "legal_clearance_claim": "none",
        "serious_training_blocked_by_policy": bool(active_hard_blockers),
        "active_hard_blocker_count": len(active_hard_blockers),
        "hard_blockers": hard_blockers,
        "allowed_by_repo_policy_sources": allowed_by_repo_policy,
        "quality_claim": "none",
        "sources": records,
        "limitations": [
            "Licenses are not fetched from remote dataset cards by this report.",
            "Repo-side source category metadata is not license evidence.",
            "Manual metadata documents repo-side review state; it is not legal advice.",
            "repo_policy_status=allowed_by_repo_policy would not mean legal clearance.",
            "license_status=unknown is not legal clearance.",
            "license_status=known_but_unreviewed would still require manual legal review.",
            "benchmark_risk is source-registry metadata, not content contamination detection.",
        ],
    }


def build_exact_chunk_dedup_report(
    *,
    manifest_path: str = TOKEN_ARTIFACT_MANIFEST_PATH,
    chunk_tokens: int = 2048,
    max_artifacts: int = 16,
    max_chunks_per_artifact: int = 256,
    include_validation_overlap: bool = True,
) -> dict:
    """Hash exact token chunks in a bounded sample; no near-dedup claim."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if max_artifacts <= 0:
        raise ValueError("max_artifacts must be > 0")
    if max_chunks_per_artifact <= 0:
        raise ValueError("max_chunks_per_artifact must be > 0")

    report = {
        "schema": "exact_token_chunk_dedup_v1",
        "ok": False,
        "manifest_path": manifest_path,
        "scope": "bounded_exact_chunk_sample",
        "chunk_tokens": int(chunk_tokens),
        "max_artifacts": int(max_artifacts),
        "max_chunks_per_artifact": int(max_chunks_per_artifact),
        "include_validation_overlap": bool(include_validation_overlap),
        "total_manifest_artifacts": 0,
        "eligible_train_artifact_count": 0,
        "selected_train_artifact_count": 0,
        "chunks_available_estimate": 0,
        "chunks_coverage_ratio": 0.0,
        "full_exact_chunk_scan": False,
        "artifacts_inspected": 0,
        "chunks_inspected": 0,
        "duplicate_chunks": 0,
        "duplicate_ratio": 0.0,
        "duplicates_by_source": {},
        "inspected_chunks_by_source": {},
        "train_val_exact_overlap": {
            "scope": "bounded_exact_chunk_sample",
            "enabled": bool(include_validation_overlap),
            "validation_artifacts_inspected": 0,
            "validation_chunks_inspected": 0,
            "overlap_chunks": 0,
            "overlap_ratio": 0.0,
            "overlap_examples": [],
            "claim": "bounded_exact_overlap_only",
        },
        "duplicate_examples": [],
        "near_dedup_claim": "none",
        "full_corpus_dedup_claim": "none",
        "errors": [],
        "limitations": [
            "Only exact byte-identical token chunks are detected.",
            "This is a bounded sample unless max_artifacts/max_chunks cover all artifacts.",
            "No semantic, fuzzy, MinHash, or near-duplicate detection is performed.",
        ],
    }
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"manifest_load_failed: {exc}")
        return report

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        report["errors"].append("manifest artifacts field is missing or not a list")
        return report

    report["total_manifest_artifacts"] = len(artifacts)
    eligible_train = [
        item for item in artifacts
        if isinstance(item, dict)
        and item.get("artifact_split") == "train"
        and os.path.isfile(str(item.get("path", "")))
    ]
    eligible_validation = [
        item for item in artifacts
        if isinstance(item, dict)
        and item.get("artifact_split") == "validation"
        and os.path.isfile(str(item.get("path", "")))
    ]
    report["eligible_train_artifact_count"] = len(eligible_train)

    def _chunk_count(item: dict) -> int:
        path = str(item.get("path", ""))
        try:
            dtype = np.dtype(item.get("dtype", "uint16"))
        except (TypeError, ValueError):
            return 0
        if not os.path.isfile(path):
            return 0
        return os.path.getsize(path) // dtype.itemsize // chunk_tokens

    total_available_chunks = sum(_chunk_count(item) for item in eligible_train)
    report["chunks_available_estimate"] = int(total_available_chunks)
    selected_train = eligible_train[:max_artifacts]
    report["selected_train_artifact_count"] = len(selected_train)

    seen: dict[str, dict] = {}

    def _scan_item_chunks(item: dict, *, target_seen: dict[str, dict] | None = None):
        path = str(item.get("path", ""))
        try:
            dtype = np.dtype(item.get("dtype", "uint16"))
        except (TypeError, ValueError) as exc:
            report["errors"].append(f"invalid dtype for {path}: {exc}")
            return []
        data = np.memmap(path, dtype=dtype, mode="r")
        chunks = min(len(data) // chunk_tokens, max_chunks_per_artifact)
        chunk_records = []
        for chunk_index in range(chunks):
            start = chunk_index * chunk_tokens
            end = start + chunk_tokens
            digest = hashlib.sha256(data[start:end].tobytes()).hexdigest()
            chunk_records.append((
                digest,
                {
                    "path": path,
                    "chunk_index": chunk_index,
                    "source": item.get("source", "unknown"),
                    "subset": item.get("subset", "unknown"),
                    "artifact_split": item.get("artifact_split", "unknown"),
                },
            ))
        del data
        if target_seen is not None:
            for digest, current in chunk_records:
                target_seen[digest] = current
        return chunk_records

    for item in selected_train:
        chunk_records = _scan_item_chunks(item)
        if not chunk_records:
            continue
        report["artifacts_inspected"] += 1
        for digest, current in chunk_records:
            source = str(current.get("source", "unknown"))
            report["chunks_inspected"] += 1
            report["inspected_chunks_by_source"][source] = (
                report["inspected_chunks_by_source"].get(source, 0) + 1
            )
            previous = seen.get(digest)
            if previous is not None:
                report["duplicate_chunks"] += 1
                report["duplicates_by_source"][source] = (
                    report["duplicates_by_source"].get(source, 0) + 1
                )
                if len(report["duplicate_examples"]) < 10:
                    report["duplicate_examples"].append({
                        "sha256": digest,
                        "first_seen": previous,
                        "duplicate": current,
                    })
            else:
                seen[digest] = current

    if report["chunks_inspected"]:
        report["duplicate_ratio"] = round(
            report["duplicate_chunks"] / report["chunks_inspected"],
            8,
        )
        report["chunks_coverage_ratio"] = round(
            report["chunks_inspected"] / max(1, total_available_chunks),
            8,
        )
    report["full_exact_chunk_scan"] = (
        report["selected_train_artifact_count"] == report["eligible_train_artifact_count"]
        and report["chunks_inspected"] == total_available_chunks
    )
    report["scope"] = (
        "full_exact_chunk_scan"
        if report["full_exact_chunk_scan"]
        else "bounded_exact_chunk_sample"
    )

    if include_validation_overlap and seen:
        overlap_report = report["train_val_exact_overlap"]
        for item in eligible_validation[:max_artifacts]:
            chunk_records = _scan_item_chunks(item)
            if not chunk_records:
                continue
            overlap_report["validation_artifacts_inspected"] += 1
            for digest, current in chunk_records:
                overlap_report["validation_chunks_inspected"] += 1
                previous = seen.get(digest)
                if previous is not None:
                    overlap_report["overlap_chunks"] += 1
                    if len(overlap_report["overlap_examples"]) < 10:
                        overlap_report["overlap_examples"].append({
                            "sha256": digest,
                            "train_chunk": previous,
                            "validation_chunk": current,
                        })
        if overlap_report["validation_chunks_inspected"]:
            overlap_report["overlap_ratio"] = round(
                overlap_report["overlap_chunks"] / overlap_report["validation_chunks_inspected"],
                8,
            )

    report["ok"] = report["artifacts_inspected"] > 0
    return report


def build_benchmark_source_risk_report(
    *,
    manifest_path: str = TOKEN_ARTIFACT_MANIFEST_PATH,
) -> dict:
    """Report benchmark-source artifacts; do not claim text contamination scan."""
    configured = set(getattr(train_cfg, "excluded_benchmark_sources", ()))
    report = {
        "schema": "benchmark_source_risk_v1",
        "ok": False,
        "manifest_path": manifest_path,
        "configured_benchmark_exclusions": sorted(configured),
        "configured_benchmark_exclusion_count": len(configured),
        "local_artifacts_checked": 0,
        "local_unique_sources_checked": [],
        "artifact_matches": [],
        "source_artifact_risk_count": 0,
        "source_level_risk": {
            "method": "exact_source_identifier_match",
            "status": "not_run",
            "risk_artifact_count": 0,
            "risk_source_count": 0,
            "claim": "source_level_risk_only",
        },
        "content_level_overlap": {
            "method": "none",
            "status": "unverified_no_local_benchmark_text_or_hashes",
            "overlap_count": None,
            "claim": "unverified",
        },
        "content_overlap_status": "unverified_no_local_benchmark_text_or_hashes",
        "contamination_claim": "none",
        "limitations": [
            "This only checks whether local token artifacts come from configured benchmark source IDs.",
            "It does not compare benchmark examples, n-grams, hashes, or generated outputs.",
            "Source-level risk is not content-level contamination detection.",
        ],
        "errors": [],
    }
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"manifest_load_failed: {exc}")
        return report

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        report["errors"].append("manifest artifacts field is missing or not a list")
        return report

    unique_sources = set()
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "unknown"))
        unique_sources.add(source)
        report["local_artifacts_checked"] += 1
        if source in configured:
            report["artifact_matches"].append({
                "source": source,
                "subset": item.get("subset"),
                "artifact_split": item.get("artifact_split"),
                "path": item.get("path"),
                "token_count": item.get("token_count"),
            })
    report["source_artifact_risk_count"] = len(report["artifact_matches"])
    report["local_unique_sources_checked"] = sorted(unique_sources)
    risk_sources = sorted({item["source"] for item in report["artifact_matches"]})
    report["source_level_risk"] = {
        "method": "exact_source_identifier_match",
        "status": "checked",
        "risk_artifact_count": len(report["artifact_matches"]),
        "risk_source_count": len(risk_sources),
        "risk_sources": risk_sources,
        "claim": "source_level_risk_only",
    }
    report["ok"] = True
    return report


def build_data_governance_report() -> dict:
    source_manifest = build_dataset_source_manifest()
    allowed_corpus = load_allowed_corpus_manifest()
    dedup = build_exact_chunk_dedup_report()
    benchmark_risk = build_benchmark_source_risk_report()
    summary = {
        "legal_clearance_claim": "none",
        "license_known_count": source_manifest.get("known_license_count", 0),
        "documented_license_count": source_manifest.get("documented_license_count", 0),
        "license_unknown_count": source_manifest.get("unknown_license_count", 0),
        "manual_metadata_coverage_ratio": source_manifest.get("manual_metadata", {}).get(
            "coverage_ratio",
            0.0,
        ),
        "repo_policy_status_counts": source_manifest.get(
            "repo_policy_status_counts",
            {},
        ),
        "training_blocker_level_counts": source_manifest.get(
            "training_blocker_level_counts",
            {},
        ),
        "serious_training_blocked_by_policy": source_manifest.get(
            "serious_training_blocked_by_policy",
            True,
        ),
        "active_hard_blocker_count": source_manifest.get(
            "active_hard_blocker_count",
            0,
        ),
        "allowed_by_repo_policy_count": len(
            source_manifest.get("allowed_by_repo_policy_sources", [])
        ),
        "allowed_corpus_manifest_ok": bool(allowed_corpus.get("ok")),
        "allowed_corpus_source_count": int(allowed_corpus.get("allowed_source_count", 0)),
        "reviewed_candidate_source_count": int(
            allowed_corpus.get("reviewed_candidate_source_count", 0)
        ),
        "declared_license_counts": source_manifest.get("declared_license_counts", {}),
        "governance_classification_counts": source_manifest.get(
            "governance_classification_counts",
            {},
        ),
        "dedup_scope": dedup.get("scope", "unknown"),
        "dedup_chunks_inspected": dedup.get("chunks_inspected", 0),
        "dedup_duplicate_ratio": dedup.get("duplicate_ratio", 0.0),
        "train_val_exact_overlap_chunks": dedup.get(
            "train_val_exact_overlap",
            {},
        ).get("overlap_chunks", 0),
        "source_level_risk_artifacts": benchmark_risk.get(
            "source_level_risk",
            {},
        ).get("risk_artifact_count", 0),
        "content_level_contamination_status": benchmark_risk.get(
            "content_level_overlap",
            {},
        ).get("status", "unknown"),
        "contamination_claim": "none",
        "training_readiness_changed": "no",
    }
    return {
        "schema": "data_governance_evidence_v1",
        "ok": bool(
            source_manifest["ok"]
            and allowed_corpus["ok"]
            and dedup["ok"]
            and benchmark_risk["ok"]
        ),
        "summary": summary,
        "source_manifest": source_manifest,
        "allowed_corpus_manifest": allowed_corpus,
        "exact_dedup": dedup,
        "benchmark_source_risk": benchmark_risk,
        "quality_claim": "none",
        "legal_clearance_claim": "none",
        "contamination_claim": "none",
    }


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
    except Exception as exc:
        print(f"     Source extractor failed; using fallback fields: {str(exc)[:100]}")
    for k in (
        "text", "content", "code", "output", "markdown",
        "chosen", "instruction", "question", "description",
    ):
        v = ex.get(k, "")
        if isinstance(v, str) and len(v.strip()) > 30:
            return v.strip()
    return ""


def load_ds(path, sub):
    from datasets import load_dataset

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
    except (ValidationError, OSError, ValueError) as exc:
        print(f"  Warning: failed to load download manifest: {str(exc)[:120]}")
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
    artifact_config_hash = config_hash(
        {
            "total_tokens": TOTAL_TOKENS,
            "val_fraction": VAL_FRACTION,
            "sources": [
                {
                    "source": path,
                    "subset": sub,
                    "weight": weight,
                    "dataset_split": SPLIT_OVERRIDES.get(path, "train"),
                }
                for path, sub, weight, _ in sources
            ],
            "dtype": dtype_name(dtype),
            "tokenizer_vocab_size": tok.vocab_size_,
        }
    )

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
        run_entry["token_artifacts"] = build_token_artifact_records(
            sources,
            dtype,
            artifact_config_hash,
        )
        write_token_artifact_manifest(
            run_entry["token_artifacts"],
            TOKEN_ARTIFACT_MANIFEST_PATH,
        )
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
    run_entry["token_artifacts"] = build_token_artifact_records(
        sources,
        dtype,
        artifact_config_hash,
    )
    write_token_artifact_manifest(
        run_entry["token_artifacts"],
        TOKEN_ARTIFACT_MANIFEST_PATH,
    )
    _save_manifest(manifest)
    print(f"  Manifest: {os.path.abspath(MANIFEST_PATH)}")
    print(f"  Token artifact manifest: {os.path.abspath(TOKEN_ARTIFACT_MANIFEST_PATH)}")
    if failed_sources:
        print("  Failed sources (partial/skip): " + ", ".join(failed_sources))


if __name__ == "__main__":
    main()
