"""Fine-tune Qwen2.5-0.5B-Instruct with LoRA (or QLoRA) on Banking77.

Run on a free Colab/Kaggle T4:
    python train.py                 # LoRA, base weights in fp32 + fp16 autocast
    python train.py --qlora         # QLoRA, base weights in 4-bit NF4

Loss is computed only on the assistant answer tokens, not on the prompt.
"""
import argparse
import json
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

from common import MODEL_ID, SEED, build_messages, make_splits

p = argparse.ArgumentParser()
p.add_argument("--qlora", action="store_true")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--n_train", type=int, default=3000)
p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--out", default="outputs/adapter")
args = p.parse_args()

torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

train_rows, val_rows, _, names = make_splits(n_train=args.n_train)


class SFTData(Dataset):
    def __init__(self, rows):
        self.items = []
        for r in rows:
            prompt_txt = tok.apply_chat_template(
                build_messages(r["text"], names), add_generation_prompt=True, tokenize=False)
            full_txt = tok.apply_chat_template(
                build_messages(r["text"], names, answer=r["label"]), tokenize=False)
            assert full_txt.startswith(prompt_txt)  # prompt must be a prefix of the full text
            prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
            full_ids = tok(full_txt, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[: args.max_len]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.items.append({"input_ids": full_ids, "labels": labels[: len(full_ids)]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        att.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}


train_ds, val_ds = SFTData(train_rows), SFTData(val_rows)
avg_len = sum(len(x["input_ids"]) for x in train_ds.items) / len(train_ds)
print(f"train={len(train_ds)} val={len(val_ds)} avg_tokens_per_example={avg_len:.0f}")

if args.qlora:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32,
                                                 device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

lora = LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

total_steps = math.ceil(len(train_ds) / 16) * args.epochs
targs = TrainingArguments(
    output_dir="outputs/run",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch 16
    per_device_eval_batch_size=8,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, int(0.03 * total_steps)),  # 3% warmup (warmup_ratio was removed in transformers 5)
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)
trainer.train()
minutes = (time.time() - t0) / 60

os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out)
tok.save_pretrained(args.out)

stats = {
    "mode": "qlora" if args.qlora else "lora",
    "rank": args.rank, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
    "n_train": len(train_ds), "avg_tokens_per_example": round(avg_len),
    "train_minutes": round(minutes, 1),
    "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if torch.cuda.is_available() else None,
    "log_history": trainer.state.log_history,
}
with open(os.path.join(args.out, "train_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print({k: v for k, v in stats.items() if k != "log_history"})
