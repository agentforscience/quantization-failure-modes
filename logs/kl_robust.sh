#!/bin/bash
cd "$(dirname "$0")/.."; source .venv/bin/activate
for m in qwen1.5b llama1b; do
  for try in 1 2 3 4 5 6; do
    gpu=""; while [ -z "$gpu" ]; do for g in 1 0 3 2; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 4000 ] && gpu=$g && break; done; [ -z "$gpu" ] && sleep 20; done
    CUDA_VISIBLE_DEVICES=$gpu python src/kl.py --model $m --group -1 --levels 16,8,6,5,4,3 >> logs/kl_robust.log 2>&1 && break; sleep 20
  done
done
