#!/bin/bash
# Usage: kl_queue.sh "GPU_LIST" NEED_GB -- model1 ...   (waits for bf16 generations + free GPU memory)
cd "$(dirname "$0")/.."; source .venv/bin/activate
GPUS=$1; GB=$2; shift 3
need=$(python -c "print(int($GB*1024))")
for m in "$@"; do
  [ -f results/kl/${m}_g128.csv ] && continue
  until [ -f results/gens/${m}_b16_g128.jsonl ]; do sleep 30; done
  for try in 1 2 3 4 5 6; do
    gpu=""
    while [ -z "$gpu" ]; do
      for g in $GPUS; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt "$need" ] && gpu=$g && break; done
      [ -z "$gpu" ] && sleep 30
    done
    echo "$(date) start on GPU $gpu" >> logs/kl_$m.log
    CUDA_VISIBLE_DEVICES=$gpu python src/kl.py --model $m >> logs/kl_$m.log 2>&1 && break
    echo "$(date) fail $try" >> logs/kl_$m.log; sleep 30
  done
done
