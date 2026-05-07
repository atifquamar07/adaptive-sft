#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

load_hf_token() {
  local env_file
  for env_file in "$ROOT/.env.local" "$ROOT/.env"; do
    if [[ -f "$env_file" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$env_file"
      set +a
      break
    fi
  done
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  fi
}

load_hf_token
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib-cache}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

if [[ -x "$ROOT/.conda-env/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.conda-env/bin/python"
elif [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

MODE="${1:-smoke}"
CONFIG="${CONFIG:-configs/default.yaml}"

run_pipeline() {
  local smoke_flag="$1"
  "$PYTHON_BIN" -m src.data --config "$CONFIG" $smoke_flag --mode prepare
  "$PYTHON_BIN" -m src.data --config "$CONFIG" $smoke_flag --mode sanity
  "$PYTHON_BIN" -m src.train_warmup --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.collect_utility_labels --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_evaluator --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.diagnose_evaluator --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.check_gradient_signal --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_baseline_random --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_baseline_static_loss --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_baseline_static_gradient --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag --shuffled-scores
  "$PYTHON_BIN" -m src.train_oracle_gradient_sft --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.evaluate --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.plots --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.report --config "$CONFIG" $smoke_flag
}

run_adaptive_only() {
  local smoke_flag="$1"
  "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag --shuffled-scores
  "$PYTHON_BIN" -m src.evaluate --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.plots --config "$CONFIG" $smoke_flag
  "$PYTHON_BIN" -m src.report --config "$CONFIG" $smoke_flag
}

case "$MODE" in
  smoke)
    run_pipeline "--smoke"
    ;;
  adaptive)
    run_adaptive_only ""
    ;;
  adaptive-smoke)
    run_adaptive_only "--smoke"
    ;;
  smoke_noise)
    export ADAPTIVE_SFT_NOISE=1
    export ADAPTIVE_SFT_OUTPUT_DIR="${ADAPTIVE_SFT_OUTPUT_DIR:-outputs/smoke_noise}"
    run_pipeline "--smoke"
    ;;
  full_noise)
    echo "Running mandatory noisy smoke test before noisy full experiment."
    ADAPTIVE_SFT_NOISE=1 ADAPTIVE_SFT_OUTPUT_DIR=outputs/smoke_noise run_pipeline "--smoke"
    echo "Noisy smoke test passed. Running noisy full experiment."
    ADAPTIVE_SFT_NOISE=1 ADAPTIVE_SFT_OUTPUT_DIR="${ADAPTIVE_SFT_OUTPUT_DIR:-outputs_noise}" run_pipeline ""
    ;;
  full)
    echo "Running mandatory smoke test before full experiment."
    run_pipeline "--smoke"
    echo "Smoke test passed. Running full experiment."
    run_pipeline ""
    ;;
  *)
    echo "Usage: bash scripts/run_all.sh [smoke|full|smoke_noise|full_noise|adaptive|adaptive-smoke]" >&2
    exit 2
    ;;
esac
