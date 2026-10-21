# Planning: Quantization Failure Modes — Task-Specific Accuracy Cliffs

## Motivation & Novelty Assessment

### Why This Research Matters
Weight-only post-training quantization (INT8/INT4, GPTQ, AWQ) is the default way to deploy LLMs
on commodity hardware. Practitioners usually pick a bit width from aggregate metrics (WikiText
perplexity, MMLU), but the damage may be very uneven across tasks: a model can look fine on
perplexity and still fail on multi-step arithmetic. A task-resolved map of *where* accuracy
collapses, and a cheap label-free signal that warns *before* it collapses, would let deployers
choose precision per use case.

### Gap in Existing Work
(No literature review was pre-gathered for this run; this is based on prior knowledge.)
GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2024) report perplexity plus a few zero-shot
multiple-choice tasks, mostly at 3–4 bits. Follow-up evaluations (e.g. "Evaluating Quantized
LLMs", Li et al. 2024; Jin et al. 2024; Marchisio et al. 2024 on multilingual quantization) show
uneven degradation, but (a) rarely sweep bit width finely enough to locate a cliff, (b) rarely
mix generative tasks (code, instruction following, creative writing) with QA under one protocol,
and (c) do not test whether distribution-level divergence (KL from the full-precision model)
anticipates the accuracy cliff. Dutta et al. (2024, "Accuracy is not all you need") argue for
KL/"flips" as complementary metrics, but do not test KL as a leading indicator of task cliffs.

### Our Novel Contribution
A controlled sweep: 8 instruction-tuned models from 3 families (Qwen2.5 0.5B–14B, Llama-3.2-1B /
Llama-3.1-8B, Phi-3-mini) × 7 weight precisions (bf16, 8, 6, 5, 4, 3, 2 bits; RTN, group 128) ×
6 task types, with task-specific token-level KL from the bf16 model measured on the
same task distributions. We explicitly test cliff location, task ordering, size dependence and
the "KL ≈ 0.3 nats" early-warning claim.

### Experiment Justification
- Exp 1 (accuracy sweep): needed to locate cliffs per task/model; directly tests H1, H2, H3.
- Exp 2 (task-specific KL + WikiText perplexity): tests H4 (KL as a leading indicator) and
  compares KL to the standard perplexity proxy.
- Exp 3 (real AWQ INT4 checkpoints for Qwen2.5-3B/7B/14B): checks that RTN-INT4 conclusions
  transfer to a production quantizer (RTN is a pessimistic lower bound).
- Exp 4 (robustness: per-channel vs group-128 at 4/3 bits on a subset): tests whether cliff
  location depends on quantization granularity rather than on "bits" per se.

## Research Question
Which task types lose accuracy first as weight precision is lowered, is the loss gradual or a
cliff, does the cliff move to higher precision for smaller models, and does KL divergence from the
full-precision model predict the cliff before accuracy degrades?

## Hypothesis Decomposition
- H1: Arithmetic (GSM8K) and factual QA (TriviaQA) show a >15% (relative) accuracy drop between
  INT8 and INT4.
- H2: Creative writing (judge score) and reading comprehension (SQuAD F1) retain ≥85% of bf16
  performance at INT4.
- H3: Within a family, the smaller the model, the higher the bit width at which the cliff occurs
  (critical bits negatively correlated with log parameter count).
- H4: Mean token-level KL(bf16 || quantized) crosses ≈0.3 nats at or before the precision at which
  the accuracy cliff appears; KL at level b predicts a cliff at level b or b−1.

Independent variables: bit width, task, model (size, family). Dependent: task score, retained
fraction (score_b / score_bf16), KL, top-1 agreement, perplexity.

## Proposed Methodology

### Approach
Local models only (no API keys on this host; Ollama not available — vLLM + HF transformers
instead). Weight-only RTN asymmetric min-max quantization with group size 128 applied in place to
all transformer linear layers (embeddings / LM head kept bf16), implemented once (`src/quant.py`)
and used by both the vLLM generation and the HF KL code. RTN is chosen because it allows any bit
width with one code path and no calibration data (no confound from calibration-set choice); real
AWQ-INT4 is added as a reference.

### Experimental Steps
1. Build fixed suite (seed 42): GSM8K 250, TriviaQA 400, SQuAD 400, HumanEval 164, IFEval 300,
   WritingPrompts 100.
2. Greedy generation for every model × level (vLLM, temperature 0).
3. Score: GSM8K exact numeric match; TriviaQA alias match (normalized containment); SQuAD token F1
   (and EM); HumanEval pass@1 via sandboxed execution; IFEval prompt-level strict accuracy;
   creative: LLM judge (Qwen local model, 1–10 rubric) + repetition statistics.
4. KL: teacher-force the bf16 model's own responses (first 60 items per task) through each
   quantized model; KL over response tokens using bf16 top-128 + lumped tail bucket (a lower bound
   on true KL); also WikiText-2 perplexity.
5. Analysis: cliffs, critical bits, correlations, logistic / threshold analysis for KL.

### Baselines
bf16 model (reference for every metric); AWQ-INT4 (production quantizer); 8-bit as the "safe"
reference level for cliff definitions.

### Evaluation Metrics
Task scores above; retained fraction; relative drop between adjacent levels; cliff = relative drop
>15% between adjacent tested precisions (with absolute drop also reported); critical bits = highest
precision with retained fraction < 0.85.

### Statistical Analysis Plan
Bootstrap 95% CIs (1000 resamples over items) for scores; paired McNemar (binary tasks) / paired
Wilcoxon (F1, judge scores) between bf16 and each level, Benjamini–Hochberg FDR at 0.05; Spearman
correlation of critical bits with log size; for H4, ROC-AUC of KL for predicting "retained < 0.85",
best threshold by Youden's J with bootstrap CI, and lead-time analysis (KL at the last "healthy"
level vs at the cliff level).

## Expected Outcomes
Supporting H1: GSM8K/TriviaQA retained < 0.85 at 4 bits. Refuting: retained ≥ 0.85 at 4 bits
(cliff at 3 bits or lower). H4 supported if KL threshold ≈0.3 with high AUC and KL elevated at the
level preceding the cliff.

## Timeline and Milestones
Setup+data 30 min; generation ~2–3 h (shared GPUs); KL ~1 h; scoring+judge ~1 h; analysis+report 1 h.

## Potential Challenges
Shared GPUs (memory fluctuates) → retry queues, small slices for small models. 2-bit outputs are
garbage and slow (max tokens) → accepted cost. Judge bias toward fluent text → also report
repetition/degeneration statistics. Note: "INT8 vs INT4" in the hypothesis is interpreted for
RTN-g128; we add 6/5/3/2 bits to localize cliffs.

## Success Criteria
All model × level × task cells scored, KL computed for each, every hypothesis given a clear
supported / partially supported / refuted verdict with CIs.
