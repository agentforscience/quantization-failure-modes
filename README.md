# Quantization Failure Modes: Task-Specific Accuracy Cliffs

We swept weight-only round-to-nearest (RTN, group-128) quantization from bf16 down to 8, 6, 5, 4, 3 and 2 bits on 7 instruction-tuned models from 3 families: Qwen2.5 0.5B/1.5B/3B/7B, Llama-3.2-1B / Llama-3.1-8B and Phi-3-mini. Each configuration was scored on 6 task types (≈1.6k prompts). For every model, task and precision level we also measured the token-level KL divergence from the bf16 model on that task's own outputs. Real AWQ-INT4 checkpoints serve as a reference point. Everything ran locally (vLLM + HF transformers) on shared A6000 GPUs.

**Full report: [REPORT.md](REPORT.md)**

## Key findings
- **The cliff is between 4 and 3 bits (3 and 2 for 7–8B models), not between 8 and 4.** 8 and 6 bits are essentially lossless on all six tasks (3 significant drops in 72 paired tests), and 5 bits nearly so. At 4 bits the mean loss is 3–16% depending on task. At 3 bits the ≤3B models collapse, keeping 0–30% on GSM8K, TriviaQA, HumanEval and SQuAD.
- **Code generation, not arithmetic, is the most INT4-sensitive task**, keeping ~0.7–0.77 of bf16 pass@1 for 4 of 7 models. Factual QA comes next. Instruction following (IFEval) is the most robust at INT4. Creative writing and reading comprehension are robust at INT4 for most models, but not for the smallest ones.
- **Accuracy hides churn.** At 5 bits, ~11% of GSM8K items the bf16 models solved flip to wrong while a similar number flip to right, so net accuracy looks flat. More than 90% of INT4 errors are fluent, confident wrong answers (e.g. "Jules Rimet" → "Jules Renard"). 3-bit errors are mostly degenerate, repetitive text.
- **Size helps, but not monotonically** (ρ = −0.53 between log size and critical precision, p = 0.0003). The 7–8B models keep 0.62–0.97 of bf16 at 3 bits, while 0.5–3B models fall to ≈0–0.3 on most tasks. Qwen2.5-3B was *not* more robust than 1.5B.
- **KL ≈ 0.2–0.3 nats separates failed from healthy configurations (AUC ≈ 0.97), but it is not an early warning.** KL crosses 0.3 at the last healthy precision in only 6/42 (14%) of cliffs, and the right threshold varies ~100× across tasks: code fails at KL ≈ 0.05, while factual QA tolerates ≈ 0.3.
- **"INT4" is not one thing.** Per-channel INT4 keeps 0.62 versus 0.92 for group-128. AWQ-INT4 recovers most of the RTN-INT4 loss.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync      # vllm 0.32, torch 2.13, transformers 5.19
python src/prepare_data.py                                           # builds datasets/suite.jsonl (needs HF hub access)
python src/generate.py --model qwen1.5b --gpu_gb 5 --eager           # generations for all levels (results/gens/)
python src/generate.py --model qwen1.5b --group -1 --levels 8,6,5,4,3 # per-channel robustness run
python src/kl.py --model qwen1.5b                                    # task-specific KL + WikiText-2 PPL (results/kl/)
python src/judge.py --gpu_gb 9                                       # creative-writing LLM judge (Qwen3-8B-AWQ)
python src/score.py && python src/analyze.py && python src/error_analysis.py && python src/make_tables.py
```
Model keys are in `src/models.py`. `src/gpu_queue.sh`, `src/mq3.sh` and `src/judge_loop.sh` are the retry/scheduling wrappers used on the shared GPUs.

## Layout
- `src/`: `quant.py` (RTN quantizer), `generate.py` (vLLM, in-place re-quantization per level), `kl.py`, `score.py`, `judge.py`, `analyze.py`, `error_analysis.py`, `make_tables.py`
- `datasets/`: `suite.jsonl` (GSM8K 250, TriviaQA 400, SQuAD 400, HumanEval 164, IFEval 300, WritingPrompts 100) and the WikiText-2 test split
- `code/instruction_following_eval/`: Google's IFEval checker
- `results/`: raw generations (`gens/`), KL (`kl/`), item-level scores, summaries, statistical tests, `hypothesis_tests.json`, `tables.md`
- `figures/`: fig1 (retained performance vs bits), fig2 (critical-bits heatmap), fig3 (KL vs accuracy)
- `planning.md`: pre-registered plan; `REPORT.md`: full write-up
