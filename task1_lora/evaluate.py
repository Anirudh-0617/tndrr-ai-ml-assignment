"""Evaluate base model vs fine-tuned model on the same 200 held-out test examples.

    python evaluate.py --which base
    python evaluate.py --which finetuned --adapter outputs/adapter

Writes outputs/preds_<which>.jsonl (every prediction) and prints:
accuracy, valid-JSON rate, and how often the answer is a real label.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_ID, build_messages, make_splits, parse_intent

p = argparse.ArgumentParser()
p.add_argument("--which", choices=["base", "finetuned"], required=True)
p.add_argument("--adapter", default="outputs/adapter")
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--batch", type=int, default=16)
args = p.parse_args()

_, _, test_rows, names = make_splits(n_test=args.n_test)
label_set = set(names)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"  # left padding for batched generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
if args.which == "finetuned":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
model.eval()

os.makedirs("outputs", exist_ok=True)
out_path = f"outputs/preds_{args.which}.jsonl"
correct = valid_json = in_labels = 0

with open(out_path, "w") as f:
    for i in range(0, len(test_rows), args.batch):
        batch = test_rows[i:i + args.batch]
        prompts = [tok.apply_chat_template(build_messages(r["text"], names),
                                           add_generation_prompt=True, tokenize=False)
                   for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, out in zip(batch, outs):
            pred, ok = parse_intent(out)
            valid_json += ok
            in_labels += pred in label_set
            hit = pred == r["label"]
            correct += hit
            f.write(json.dumps({"text": r["text"], "gold": r["label"], "pred": pred,
                                "raw": out, "correct": hit}) + "\n")
        print(f"{min(i + args.batch, len(test_rows))}/{len(test_rows)}", end="\r")

n = len(test_rows)
summary = {"which": args.which, "n": n, "accuracy": round(correct / n, 4),
           "valid_json_rate": round(valid_json / n, 4),
           "label_in_set_rate": round(in_labels / n, 4)}
print("\n", summary)
with open(f"outputs/summary_{args.which}.json", "w") as f:
    json.dump(summary, f, indent=2)
