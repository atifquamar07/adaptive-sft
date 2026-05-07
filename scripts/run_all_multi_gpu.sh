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

MODE="${1:-full}"
CONFIG="${CONFIG:-configs/default.yaml}"
BASE_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
BASE_GPUS=()
if [[ -n "$BASE_CUDA_VISIBLE_DEVICES" ]]; then
  IFS=',' read -r -a BASE_GPUS <<< "$BASE_CUDA_VISIBLE_DEVICES"
fi

default_gpu_ids() {
  local count=0
  if [[ "${#BASE_GPUS[@]}" -gt 0 ]]; then
    count="${#BASE_GPUS[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]' || true)"
  fi
  if [[ "${count:-0}" -le 0 ]]; then
    echo "0"
    return
  fi
  local ids=()
  local idx
  for ((idx = 0; idx < count; idx++)); do
    ids+=("$idx")
  done
  local IFS=,
  echo "${ids[*]}"
}

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  echo "$value"
}

GPU_IDS="${GPU_IDS:-$(default_gpu_ids)}"
RAW_GPUS=()
IFS=',' read -r -a RAW_GPUS <<< "$GPU_IDS"
GPUS=()
for gpu in "${RAW_GPUS[@]}"; do
  gpu="$(trim_spaces "$gpu")"
  if [[ -n "$gpu" ]]; then
    GPUS+=("$gpu")
  fi
done

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "No GPU ids provided. Set GPU_IDS, for example GPU_IDS=0,1,2,3." >&2
  exit 2
fi

resolve_gpu() {
  local requested="$1"
  if [[ "${#BASE_GPUS[@]}" -gt 0 && "$requested" =~ ^[0-9]+$ && "$requested" -lt "${#BASE_GPUS[@]}" ]]; then
    echo "${BASE_GPUS[$requested]}"
  else
    echo "$requested"
  fi
}

RESOLVED_GPUS=()
for gpu in "${GPUS[@]}"; do
  RESOLVED_GPUS+=("$(resolve_gpu "$gpu")")
done
GPUS=("${RESOLVED_GPUS[@]}")

join_by_comma() {
  local IFS=,
  echo "$*"
}

smoke_flag_for_mode() {
  if [[ "$1" == "smoke" ]]; then
    echo "--smoke"
  else
    echo ""
  fi
}

output_dir_for_mode() {
  if [[ -n "${ADAPTIVE_SFT_OUTPUT_DIR:-}" ]]; then
    echo "$ROOT/${ADAPTIVE_SFT_OUTPUT_DIR}"
    return
  fi
  if [[ "$1" == "smoke" ]]; then
    echo "$ROOT/outputs/smoke"
  else
    echo "$ROOT/outputs"
  fi
}

run_fg() {
  local gpu="$1"
  shift
  echo "[gpu $gpu] $*"
  CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

run_fg_all() {
  local gpus_csv="$1"
  shift
  echo "[gpus $gpus_csv] $*"
  CUDA_VISIBLE_DEVICES="$gpus_csv" ADAPTIVE_SFT_DATA_PARALLEL="${ADAPTIVE_SFT_DATA_PARALLEL:-1}" "$@"
}

run_bg() {
  local gpu="$1"
  local name="$2"
  local log_dir="$3"
  shift 3
  local log_file="$log_dir/${name}.log"
  echo "[gpu $gpu] starting $name; log=$log_file"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "$@"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

wait_group() {
  local failed=0
  for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
      echo "finished ${NAMES[$idx]}"
    else
      echo "FAILED ${NAMES[$idx]}" >&2
      failed=1
    fi
  done
  PIDS=()
  NAMES=()
  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
}

run_pipeline() {
  local mode="$1"
  local smoke_flag
  smoke_flag="$(smoke_flag_for_mode "$mode")"
  local out_dir
  out_dir="$(output_dir_for_mode "$mode")"
  local launch_logs="$out_dir/launcher_logs"
  mkdir -p "$launch_logs"

  local g0="${GPUS[0]}"
  local all_gpus_csv
  all_gpus_csv="$(join_by_comma "${GPUS[@]}")"
  if [[ -n "$BASE_CUDA_VISIBLE_DEVICES" ]]; then
    echo "Parent CUDA_VISIBLE_DEVICES=$BASE_CUDA_VISIBLE_DEVICES"
  fi
  echo "Using GPUs: ${GPUS[*]}"
  echo "Shared data/evaluator stages run on GPU $g0."
  echo "Shared model stages use DataParallel across GPUs: $all_gpus_csv."
  echo "Method stages fan out across GPUs."

  run_fg "$g0" "$PYTHON_BIN" -m src.data --config "$CONFIG" $smoke_flag --mode prepare
  run_fg "$g0" "$PYTHON_BIN" -m src.data --config "$CONFIG" $smoke_flag --mode sanity
  run_fg_all "$all_gpus_csv" "$PYTHON_BIN" -m src.train_warmup --config "$CONFIG" $smoke_flag
  run_fg_all "$all_gpus_csv" "$PYTHON_BIN" -m src.collect_utility_labels --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.train_evaluator --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.diagnose_evaluator --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.check_gradient_signal --config "$CONFIG" $smoke_flag

  PIDS=()
  NAMES=()
  local g1="${GPUS[1]:-${GPUS[0]}}"
  local g2="${GPUS[2]:-${GPUS[0]}}"
  local g3="${GPUS[3]:-${GPUS[0]}}"

  echo "[gpu ${GPUS[0]}] starting random_sft_then_static_loss_sft; log=$launch_logs/random_sft_then_static_loss_sft.log"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "$PYTHON_BIN" -m src.train_baseline_random --config "$CONFIG" $smoke_flag
    "$PYTHON_BIN" -m src.train_baseline_static_loss --config "$CONFIG" $smoke_flag
  ) >"$launch_logs/random_sft_then_static_loss_sft.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("random_sft_then_static_loss_sft")

  echo "[gpu $g1] starting static_gradient_then_oracle_gradient_sft; log=$launch_logs/static_gradient_then_oracle_gradient_sft.log"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$g1"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "$PYTHON_BIN" -m src.train_baseline_static_gradient --config "$CONFIG" $smoke_flag
    "$PYTHON_BIN" -m src.train_oracle_gradient_sft --config "$CONFIG" $smoke_flag
  ) >"$launch_logs/static_gradient_then_oracle_gradient_sft.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("static_gradient_then_oracle_gradient_sft")
  run_bg "$g2" adaptive_utility_sft "$launch_logs" "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag
  run_bg "$g3" adaptive_shuffled_scores "$launch_logs" "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag --shuffled-scores
  wait_group

  run_fg "$g0" "$PYTHON_BIN" -m src.evaluate --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.plots --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.report --config "$CONFIG" $smoke_flag
}

run_adaptive_only() {
  local mode="$1"
  local smoke_flag
  smoke_flag="$(smoke_flag_for_mode "$mode")"
  local out_dir
  out_dir="$(output_dir_for_mode "$mode")"
  local launch_logs="$out_dir/launcher_logs"
  mkdir -p "$launch_logs"

  local g0="${GPUS[0]}"
  local g1="${GPUS[1]:-${GPUS[0]}}"
  echo "Using GPUs: ${GPUS[*]}"
  echo "Adaptive-only mode reuses existing data, warmup, and evaluator artifacts."

  PIDS=()
  NAMES=()
  run_bg "$g0" adaptive_utility_sft "$launch_logs" "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag
  run_bg "$g1" adaptive_shuffled_scores "$launch_logs" "$PYTHON_BIN" -m src.train_adaptive_sft --config "$CONFIG" $smoke_flag --shuffled-scores
  wait_group

  run_fg "$g0" "$PYTHON_BIN" -m src.evaluate --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.plots --config "$CONFIG" $smoke_flag
  run_fg "$g0" "$PYTHON_BIN" -m src.report --config "$CONFIG" $smoke_flag
}

case "$MODE" in
  smoke)
    run_pipeline "smoke"
    ;;
  adaptive)
    run_adaptive_only "full"
    ;;
  adaptive-smoke)
    run_adaptive_only "smoke"
    ;;
  smoke_noise)
    ADAPTIVE_SFT_NOISE=1 ADAPTIVE_SFT_OUTPUT_DIR="${ADAPTIVE_SFT_OUTPUT_DIR:-outputs/smoke_noise}" run_pipeline "smoke"
    ;;
  full_noise)
    echo "Running mandatory noisy smoke test before noisy full multi-GPU experiment."
    ADAPTIVE_SFT_NOISE=1 ADAPTIVE_SFT_OUTPUT_DIR=outputs/smoke_noise run_pipeline "smoke"
    echo "Noisy smoke test passed. Running noisy full multi-GPU experiment."
    ADAPTIVE_SFT_NOISE=1 ADAPTIVE_SFT_OUTPUT_DIR="${ADAPTIVE_SFT_OUTPUT_DIR:-outputs_noise}" run_pipeline "full"
    ;;
  full)
    echo "Running mandatory smoke test before full multi-GPU experiment."
    run_pipeline "smoke"
    echo "Smoke test passed. Running full multi-GPU experiment."
    run_pipeline "full"
    ;;
  *)
    echo "Usage: GPU_IDS=0,1,2,3 bash scripts/run_all_multi_gpu.sh [smoke|full|smoke_noise|full_noise|adaptive|adaptive-smoke]" >&2
    exit 2
    ;;
esac
