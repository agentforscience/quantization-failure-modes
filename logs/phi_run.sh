#!/bin/bash
# Phi-3 reduced sweep (4 and 3 bits; bf16 already done) on a GPU with >= 13.6 GB free, then its KL.
cd "$(dirname "$0")/.."; source .venv/bin/activate
for try in $(seq 1 10); do
  gpu=""; while [ -z "$gpu" ]; do for g in 2 0 1 3; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 13900 ] && gpu=$g && break; done; [ -z "$gpu" ] && sleep 15; done
  echo "$(date) phi reduced start GPU $gpu" >> logs/gen_phi3mini.log
  CUDA_VISIBLE_DEVICES=$gpu python src/generate.py --model phi3mini --levels 4,3 --gpu_gb 13 --max_num_seqs 32 --batched_tokens 1024 --eager >> logs/gen_phi3mini.log 2>&1 && break
  sleep 15
done
