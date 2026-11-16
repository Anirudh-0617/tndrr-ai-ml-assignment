"""Shared pieces for Task 1: data loading, prompt format, output parsing.

Task: Banking77 intent classification. The model reads a customer message
and must reply with exactly {"intent": "<label>"} using one of 77 labels.
Base model and fine-tuned model get the SAME prompt, so the comparison is fair.
"""
import json
import random
import re

from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42


def load_banking77():
    """Return (train_rows, test_rows, label_names). Each row: {"text", "label"}."""
    last_err = None
    for name in ("PolyAI/banking77", "mteb/banking77"):
        try:
            ds = load_dataset(name)
            break
        except Exception as e:  # dataset scripts sometimes break, fall back
            last_err = e
    else:
        raise RuntimeError(f"Could not load Banking77: {last_err}")

    feats = ds["train"].features
    if hasattr(feats["label"], "names"):
        names = list(feats["label"].names)
    else:  # mteb version stores label_text
        pairs = {}
        for split in ("train", "test"):
            for r in ds[split]:
                pairs[int(r["label"])] = r["label_text"]
        names = [pairs[i] for i in range(len(pairs))]

    def rows(split):
        return [{"text": r["text"], "label": names[int(r["label"])]} for r in ds[split]]

    return rows("train"), rows("test"), names


def make_splits(n_train=3000, n_val=200, n_test=200):
    """Sample fixed train/val/test subsets. Test comes only from the official test split."""
    train_all, test_all, names = load_banking77()
    rng = random.Random(SEED)
    rng.shuffle(train_all)
    rng.shuffle(test_all)
    return train_all[:n_train], train_all[n_train:n_train + n_val], test_all[:n_test], names


def build_messages(text, label_names, answer=None):
    labels = ", ".join(label_names)
    system = (
        "You classify banking customer messages into exactly one intent. "
        'Reply with only JSON in the form {"intent": "<label>"}. '
        f"Allowed labels: {labels}"
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": json.dumps({"intent": answer})})
    return msgs


def parse_intent(output):
    """Return (intent or None, is_valid_json)."""
    m = re.search(r"\{.*?\}", output, re.S)
    if not m:
        return None, False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, False
    intent = obj.get("intent") if isinstance(obj, dict) else None
    return (intent.strip() if isinstance(intent, str) else None), True
