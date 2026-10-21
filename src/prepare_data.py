"""Build the fixed evaluation suite (6 task types) and save it to datasets/suite.jsonl.

Each item: {task, id, messages (chat), max_tokens, gold (task-specific)}.
Sampling is seeded (42) so the suite is reproducible.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
N = {"arith": 250, "factqa": 400, "reading": 400, "code": 164, "instr": 300, "creative": 100}
OUT = Path(__file__).resolve().parent.parent / "datasets"


def sample(ds, n, seed=SEED):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def main():
    OUT.mkdir(exist_ok=True)
    items = []

    # (1) Arithmetic reasoning: GSM8K test, zero-shot chain of thought.
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i, ex in enumerate(sample(gsm, N["arith"])):
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append(dict(task="arith", id=f"gsm{i}", max_tokens=512, gold=gold, messages=[{"role": "user", "content":
            ex["question"] + "\n\nSolve the problem step by step. End your response with the line "
            "'The answer is N' where N is the final number."}]))

    # (2) Factual QA: TriviaQA (closed book), short answer, alias match.
    tqa = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    for i, ex in enumerate(sample(tqa, N["factqa"])):
        aliases = sorted(set(ex["answer"]["normalized_aliases"] + [ex["answer"]["normalized_value"]]))
        items.append(dict(task="factqa", id=f"tqa{i}", max_tokens=32, gold=aliases, messages=[{"role": "user", "content":
            "Answer the trivia question with only the answer (a few words), no explanation.\n\nQuestion: "
            + ex["question"] + "\nAnswer:"}]))

    # (3) Reading comprehension: SQuAD v1.1 validation, extractive span.
    sq = load_dataset("rajpurkar/squad", split="validation")
    for i, ex in enumerate(sample(sq, N["reading"])):
        items.append(dict(task="reading", id=f"squad{i}", max_tokens=32, gold=ex["answers"]["text"], messages=[{"role": "user", "content":
            "Read the passage and answer the question with a short span copied from the passage. "
            "Reply with only the answer.\n\nPassage: " + ex["context"] + "\n\nQuestion: " + ex["question"] + "\nAnswer:"}]))

    # (4) Code generation: HumanEval (all 164), executed with unit tests.
    he = load_dataset("openai/openai_humaneval", split="test")
    for ex in he:
        items.append(dict(task="code", id=ex["task_id"], max_tokens=512,
                          gold={"prompt": ex["prompt"], "test": ex["test"], "entry_point": ex["entry_point"]},
                          messages=[{"role": "user", "content":
            "Complete the following Python function. Return the full function (including the signature) "
            "in a single ```python code block.\n\n```python\n" + ex["prompt"] + "```"}]))

    # (5) Instruction following: IFEval (verifiable constraints), random 300 of 541.
    ife = load_dataset("google/IFEval", split="train")
    for ex in sample(ife, N["instr"]):
        items.append(dict(task="instr", id=f"ifeval{ex['key']}", max_tokens=768,
                          gold={"key": ex["key"], "instruction_id_list": ex["instruction_id_list"],
                                "kwargs": ex["kwargs"], "prompt": ex["prompt"]},
                          messages=[{"role": "user", "content": ex["prompt"]}]))

    # (6) Creative writing: Reddit WritingPrompts (test split), short stories judged by an LLM.
    wp = load_dataset("euclaise/writingprompts", split="test")
    wp = wp.filter(lambda e: e["prompt"].startswith("[ WP ]") and 40 < len(e["prompt"]) < 300)
    for i, ex in enumerate(sample(wp, N["creative"])):
        p = ex["prompt"].replace("[ WP ]", "").strip()
        items.append(dict(task="creative", id=f"wp{i}", max_tokens=400, gold=None, messages=[{"role": "user", "content":
            "Write a short story (about 200 words) based on this prompt:\n\n" + p}]))

    with open(OUT / "suite.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    from collections import Counter
    print(Counter(it["task"] for it in items))
    for t in N:
        ex = next(it for it in items if it["task"] == t)
        print("----", t, "\n", ex["messages"][0]["content"][:300], "\nGOLD:", str(ex["gold"])[:150])


if __name__ == "__main__":
    main()
