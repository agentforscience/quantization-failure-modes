#!/bin/bash
# Usage: run_queue.sh GPU GPU_GB EXTRA_ARGS -- model1 model2 ...
# Runs generate.py for each model on one GPU, retrying (e.g. startup OOM on shared GPUs).
cd "$(dirname "$0")/.."; source .venv/bin/activate
GPU=$1; GB=$2; EXTRA=$3; shift 4
for m in "$@"; do
  for try in $(seq 1 30); do
    CUDA_VISIBLE_DEVICES=$GPU python src/generate.py --model $m --gpu_gb $GB $EXTRA >> logs/gen_$m.log 2>&1 && break
    echo "$(date) $m try $try failed" >> logs/gen_$m.log; sleep 60
  done
  echo "$(date) $m finished" >> logs/queue_status.txt
done
