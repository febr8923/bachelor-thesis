#!/usr/bin/env bash
# submit-all.sh — Submit all benchmark jobs with dependency chaining.
#
# Jobs run one after another so each gets the full node.
# Usage:  bash jobs/submit-all.sh
#         bash jobs/submit-all.sh --dry-run   (prints sbatch commands without submitting)

set -euo pipefail
cd "$(dirname "$0")/.."          # benchmarks/

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

LOGDIR="jobs/logs"
mkdir -p "$LOGDIR"

submit() {
  local script="$1"
  local dep="${2:-}"

  local cmd="sbatch"
  [[ -n "$dep" ]] && cmd="$cmd --dependency=afterany:$dep"
  cmd="$cmd $script"

  if $DRY_RUN; then
    echo "[dry-run] $cmd"
    echo "0"          # fake job id
    return
  fi

  local out
  out=$($cmd)
  local jid
  jid=$(echo "$out" | awk '{print $NF}')
  echo "$out" >&2
  echo "$jid"
}

echo "=== Submitting benchmark jobs ==="

JID=""

# 1. vLLM GPU warm
JID=$(submit jobs/vllm-gpu.sbatch "$JID")
echo "  vllm-gpu      -> $JID"

# 2. vLLM GPU cold
JID=$(submit jobs/vllm-cold.sbatch "$JID")
echo "  vllm-cold     -> $JID"

# 3. vLLM CPU
JID=$(submit jobs/vllm-cpu.sbatch "$JID")
echo "  vllm-cpu      -> $JID"

# 4. DL resnet50
JID=$(submit jobs/dl-resnet50.sbatch "$JID")
echo "  dl-resnet50   -> $JID"

# 5. DL vgg19
JID=$(submit jobs/dl-vgg19.sbatch "$JID")
echo "  dl-vgg19      -> $JID"

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
