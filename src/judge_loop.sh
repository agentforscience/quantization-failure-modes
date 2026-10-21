#!/bin/bash
# Run src/judge.py when a config fits: 1 GPU x 7.5 GB or 2 GPUs x 4.4 GB (TP). Retries up to 12 times.
cd "$(dirname "$0")/.."; source .venv/bin/activate
pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v need=$2 '$2>need{print $2, $1}' \
    | sort -rn | head -n $1 | awk '{print $2}' | paste -sd, | awk -F, -v n=$1 'NF==n'; }
for try in $(seq 1 12); do
  gpus=""
  while [ -z "$gpus" ]; do
    for c in 1:7.5 2:4.4; do n=${c%%:*}; gb=${c##*:}; gpus=$(pick $n $(python -c "print(int(($gb+0.5)*1024))")); [ -n "$gpus" ] && break; done
    [ -z "$gpus" ] && sleep 15
  done
  echo "$(date) judge start on $gpus" >> logs/judge.log
  CUDA_VISIBLE_DEVICES=$gpus python src/judge.py --gpu_gb $gb --tp $n >> logs/judge.log 2>&1 && break
  echo "$(date) judge fail $try" >> logs/judge.log; sleep 15
done
