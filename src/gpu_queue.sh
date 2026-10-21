#!/bin/bash
# Usage: gpu_queue.sh "GPU_LIST" GPU_GB "EXTRA_ARGS" -- model1 model2 ...
# For each model: wait until one of the GPUs in GPU_LIST has > GPU_GB+0.7 GB free, then run
# generate.py there. Waiting does not consume retries; only real failures do (max 8).
cd "$(dirname "$0")/.."; source .venv/bin/activate
GPUS=$1; GB=$2; EXTRA=$3; shift 4
need=$(python -c "print(int(($GB+0.7)*1024))")
for m in "$@"; do
  fails=0
  while [ $fails -lt 8 ]; do
    gpu=""
    while [ -z "$gpu" ]; do
      for g in $GPUS; do
        f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$f" -gt "$need" ]; then gpu=$g; break; fi
      done
      [ -z "$gpu" ] && sleep 30
    done
    echo "$(date) $m start on GPU $gpu" >> logs/gen_$m.log
    CUDA_VISIBLE_DEVICES=$gpu python src/generate.py --model $m --gpu_gb $GB $EXTRA >> logs/gen_$m.log 2>&1 && break
    fails=$((fails+1)); echo "$(date) $m fail $fails" >> logs/gen_$m.log; sleep 20
  done
  echo "$(date) $m finished (fails=$fails)" >> logs/queue_status.txt
done
