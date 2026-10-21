"""Model registry: short key -> (HF repo id, family, params in billions)."""
MODELS = {
    "qwen0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5", 0.49),
    "qwen1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5", 1.54),
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5", 3.09),
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5", 7.62),
    "qwen14b": ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5", 14.7),
    "llama1b": ("unsloth/Llama-3.2-1B-Instruct", "Llama-3", 1.24),
    "llama8b": ("unsloth/Meta-Llama-3.1-8B-Instruct", "Llama-3", 8.03),
    "phi3mini": ("microsoft/Phi-3-mini-4k-instruct", "Phi-3", 3.82),
    # Real (production) INT4 checkpoints, used as a reference point for RTN-INT4.
    "qwen3b-awq": ("Qwen/Qwen2.5-3B-Instruct-AWQ", "Qwen2.5", 3.09),
    "qwen7b-awq": ("Qwen/Qwen2.5-7B-Instruct-AWQ", "Qwen2.5", 7.62),
    "qwen14b-awq": ("Qwen/Qwen2.5-14B-Instruct-AWQ", "Qwen2.5", 14.7),
}
LEVELS = [16, 8, 6, 5, 4, 3, 2]
