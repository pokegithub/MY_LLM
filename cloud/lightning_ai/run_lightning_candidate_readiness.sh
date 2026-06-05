#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

MODE="--dry-run"
TASK_CLASS="candidate_readiness_smoke"
CANDIDATE_MODEL_PATH="./run_artifacts/local_models/qwen2.5-coder-0.5b-instruct"
GPU_NAME=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --execute)
      MODE="--run-smoke --execute"
      shift
      ;;
    --dry-run)
      MODE="--dry-run"
      shift
      ;;
    --task-class)
      TASK_CLASS="$2"
      shift 2
      ;;
    --candidate-model-path)
      CANDIDATE_MODEL_PATH="$2"
      shift 2
      ;;
    --gpu-name)
      GPU_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -n "$GPU_NAME" ]; then
  # shellcheck disable=SC2086
  exec "$PYTHON" "$SCRIPT_DIR/cloud_readiness_runner.py" $MODE --task-class "$TASK_CLASS" --candidate-model-path "$CANDIDATE_MODEL_PATH" --gpu-name "$GPU_NAME"
fi

# shellcheck disable=SC2086
exec "$PYTHON" "$SCRIPT_DIR/cloud_readiness_runner.py" $MODE --task-class "$TASK_CLASS" --candidate-model-path "$CANDIDATE_MODEL_PATH"
