"""Answer a question end to end: retrieve top chunks, then let a small LLM answer
using only those chunks.

    python ask.py --pdf data/llama2.pdf --size 500 "How many tokens was Llama 2 pretrained on?"
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import DenseIndex, Embedder, chunk_pages, load_pdf_pages

LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"

p = argparse.ArgumentParser()
p.add_argument("question")
p.add_argument("--pdf", required=True)
p.add_argument("--size", type=int, default=500)
p.add_argument("--k", type=int, default=5)
args = p.parse_args()

chunks = chunk_pages(load_pdf_pages(args.pdf), args.size, int(args.size * 0.2))
emb = Embedder()
index = DenseIndex(emb.docs([c["text"] for c in chunks]))
ids, scores = index.search(emb.query(args.question), args.k)

context = "\n\n".join(f"[{n + 1}] (page {chunks[i]['page']}) {chunks[i]['text']}"
                      for n, i in enumerate(ids))
messages = [
    {"role": "system", "content": "Answer using only the numbered context. Cite sources like [2]. "
                                  "If the context does not contain the answer, say you don't know."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {args.question}"},
]

tok = AutoTokenizer.from_pretrained(LLM_ID)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=dtype, device_map="auto")
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=256, do_sample=False)

print("Retrieved:", [(chunks[i]["page"], round(s, 3)) for i, s in zip(ids, scores)])
print("\nAnswer:\n" + tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
