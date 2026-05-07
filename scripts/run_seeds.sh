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

if [[ -x "$ROOT/.conda-env/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.conda-env/bin/python"
elif [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

MODE="${MODE:-full}"
SEEDS="${SEEDS:-1 2 3}"
ROOT_OUT="${ROOT_OUT:-outputs/seeds}"
THRESHOLD="${THRESHOLD:-}"

mkdir -p "$ROOT_OUT"
for seed in $SEEDS; do
  out_dir="$ROOT_OUT/seed_$seed"
  echo "Running seed $seed -> $out_dir"
  ADAPTIVE_SFT_SEED="$seed" ADAPTIVE_SFT_OUTPUT_DIR="$out_dir" bash scripts/run_all.sh "$MODE"
done

if [[ -n "$THRESHOLD" ]]; then
  "$PYTHON_BIN" -m src.aggregate_seeds --root "$ROOT_OUT" --threshold "$THRESHOLD"
else
  "$PYTHON_BIN" -m src.aggregate_seeds --root "$ROOT_OUT"
fi
