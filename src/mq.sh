#!/bin/bash
# Serial multi-GPU queue. Each job: KIND MODEL NGPU GB_PER_GPU  (KIND = gen | kl)
# Waits until NGPU GPUs each have > GB_PER_GPU+0.6 GB free, then runs the job sharded over them.
cd "$(dirname "$0")/.."; source .venv/bin/activate
pick() {  # $1 = n gpus, $2 = MiB needed; prints comma list or nothing
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v need=$2 '$2>need{print $2, $1}' \
    | sort -rn | head -n $1 | awk '{print $2}' | paste -sd, | awk -F, -v n=$1 'NF==n'
}
while read kind m n gb; do
  [ -z "$kind" ] && continue
  if [ $kind = gen ]; then extra="--tp $n --gpu_gb $gb --eager --max_num_seqs 64 --batched_tokens 2048"; log=logs/gen_$m.log
  else [ -f results/kl/${m}_g128.csv ] && continue; extra="--max_gb_per_gpu $gb"; log=logs/kl_$m.log; fi
  need=$(python -c "print(int(($gb+0.6)*1024))")
  for try in 1 2 3 4 5 6 7 8; do
    gpus=""; while [ -z "$gpus" ]; do gpus=$(pick $n $need); [ -z "$gpus" ] && sleep 20; done
    echo "$(date) $kind start on GPUs $gpus" >> $log
    if [ $kind = gen ]; then CUDA_VISIBLE_DEVICES=$gpus python src/generate.py --model $m $extra >> $log 2>&1 && break
    else CUDA_VISIBLE_DEVICES=$gpus python src/kl.py --model $m $extra >> $log 2>&1 && break; fi
    echo "$(date) $kind fail $try" >> $log; sleep 20
  done
  echo "$(date) $kind $m finished" >> logs/queue_status.txt
done < "$1"
