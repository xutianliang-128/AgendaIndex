#!/usr/bin/env bash
# Score val+test with Qwen3-4B zero-shot and with each LoRA checkpoint.
set -u
cd "$(dirname "$0")/.."
PY=/home/tianlix/.conda/envs/RAG_new/bin
OUT=results/sft_eval
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

run() {  # name adapter-or-empty
  for split in val test; do
    [ -s "$OUT/${1}_${split}.json" ] && continue
    echo "=== $1 $split $(date +%T)"
    $PY/torchrun --nproc_per_node "$NP" --master_port 29531 sft/score.py \
      --data results/sft_data/$split.jsonl --out "$OUT/${1}_${split}.json" \
      ${2:+--adapter $2} 2>&1 | grep -v -i warn | tail -3
  done
}

run qwen_zero ""
run qwen_sft200 results/sft_qwen4b/checkpoint-200
run qwen_sft808 results/sft_qwen4b/checkpoint-808
echo "=== ALL DONE $(date +%T)"
