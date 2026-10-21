#!/bin/bash
# Serial multi-GPU queue. Job line: KIND MODEL CONFIGS   (KIND = gen | kl; CONFIGS = "ngpu:gb,ngpu:gb,...")
# Waits until some config fits (ngpu GPUs each with > gb+0.6 GB free), runs the job sharded over them.
cd "$(dirname "$0")/.."; source .venv/bin/activate
pick() {  # $1 = n gpus, $2 = MiB needed; prints comma list or nothing
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v need=$2 '$2>need{print $2, $1}' \
    | sort -rn | head -n $1 | awk '{print $2}' | paste -sd, | awk -F, -v n=$1 'NF==n'
}
while read kind m configs; do
  [ -z "$kind" ] && continue
  [ $kind = kl ] && [ -f results/kl/${m}_g128.csv ] && continue
  log=logs/${kind}_$m.log
  for try in $(seq 1 12); do
    gpus=""
    while [ -z "$gpus" ]; do
      for c in ${configs//,/ }; do
        n=${c%%:*}; gb=${c##*:}; need=$(python -c "print(int(($gb+0.6)*1024))")
        gpus=$(pick $n $need); [ -n "$gpus" ] && break
      done
      [ -z "$gpus" ] && sleep 15
    done
    echo "$(date) $kind start on GPUs $gpus (n=$n gb=$gb)" >> $log
    if [ $kind = gen ]; then
      CUDA_VISIBLE_DEVICES=$gpus python src/generate.py --model $m --tp $n --gpu_gb $gb --eager --max_num_seqs 32 --batched_tokens 1024 >> $log 2>&1 && break
    else
      CUDA_VISIBLE_DEVICES=$gpus python src/kl.py --model $m --max_gb_per_gpu $(python -c "print($gb-0.8)") >> $log 2>&1 && break
    fi
    echo "$(date) $kind fail $try" >> $log; sleep 15
  done
  echo "$(date) $kind $m finished" >> logs/queue_status.txt
done < "$1"
