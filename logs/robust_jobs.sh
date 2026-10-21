#!/bin/bash
# Exp 4: per-output-channel RTN (group=-1) for the two smallest-but-capable models, then KL.
cd "$(dirname "$0")/.."
until ! pgrep -f "[g]pu_queue.sh 3 1 0 2" > /dev/null && ! pgrep -f "[g]pu_queue.sh 1 3 0 2" > /dev/null; do sleep 30; done
src/gpu_queue.sh "0 1 2 3" 4.2 "--eager --max_num_seqs 48 --batched_tokens 1024 --group -1 --levels 8,6,5,4,3" -- qwen1.5b llama1b
