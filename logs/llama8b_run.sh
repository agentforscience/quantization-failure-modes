#!/bin/bash
# After qwen7b KL: Llama-3.1-8B reduced sweep (16,8,4,3) on a GPU with >= 20.5 GB free, then KL.
cd "$(dirname "$0")/.."; source .venv/bin/activate
until [ -f results/kl/qwen7b_g128.csv ]; do sleep 20; done
for try in $(seq 1 10); do
  gpu=""; while [ -z "$gpu" ]; do for g in 2 0 1 3; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 20500 ] && gpu=$g && break; done; [ -z "$gpu" ] && sleep 15; done
  echo "$(date) llama8b reduced start GPU $gpu" >> logs/gen_llama8b.log
  CUDA_VISIBLE_DEVICES=$gpu python src/generate.py --model llama8b --levels 16,8,4,3 --gpu_gb 19.5 --max_num_seqs 64 --batched_tokens 2048 --eager >> logs/gen_llama8b.log 2>&1 && break
  sleep 15
done
for try in $(seq 1 10); do
  gpu=""; while [ -z "$gpu" ]; do for g in 2 0 1 3; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 19500 ] && gpu=$g && break; done; [ -z "$gpu" ] && sleep 15; done
  CUDA_VISIBLE_DEVICES=$gpu python src/kl.py --model llama8b --levels 16,8,4,3 >> logs/kl_llama8b.log 2>&1 && break
  sleep 15
done
