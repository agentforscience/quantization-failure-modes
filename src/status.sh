#!/bin/bash
# Print generation/KL progress and GPU state.
cd "$(dirname "$0")/.."
date; echo "gens: $(ls results/gens/*.jsonl 2>/dev/null | wc -l)"; ls results/gens | tr '\n' ' '; echo
for f in logs/gen_*.log logs/kl_*.log; do [ -f $f ] && echo "== $f: $(grep -E '^\[|start on|fail |failed|Error' $f | tail -2 | tr '\n' ' ' | cut -c1-220)"; done
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
