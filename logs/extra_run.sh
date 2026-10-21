#!/bin/bash
# Fill in the remaining precision levels for the 7-8B models, then full-level KL for llama8b.
cd "$(dirname "$0")/.."; source .venv/bin/activate
until [ -f results/kl/llama8b_g128.csv ]; do sleep 20; done
run() {  # model levels script
  for try in $(seq 1 8); do
    gpu=""; while [ -z "$gpu" ]; do for g in 2 0 1 3; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 20500 ] && gpu=$g && break; done; [ -z "$gpu" ] && sleep 15; done
    if [ $3 = gen ]; then CUDA_VISIBLE_DEVICES=$gpu python src/generate.py --model $1 --levels $2 --gpu_gb 19.5 --max_num_seqs 64 --batched_tokens 2048 --eager >> logs/gen_$1.log 2>&1 && return
    else CUDA_VISIBLE_DEVICES=$gpu python src/kl.py --model $1 --levels $2 >> logs/kl_$1.log 2>&1 && return; fi
    sleep 15
  done
}
run llama8b 6,5,2 gen
run llama8b 16,8,6,5,4,3,2 kl
run qwen7b 6,2 gen
echo "$(date) extra done" >> logs/queue_status.txt
