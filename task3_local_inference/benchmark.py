"""Benchmark one model at two precision levels with Ollama (works on a MacBook).

Setup (once):
    brew install ollama        # or download from ollama.com
    ollama serve               # leave running in another terminal
    ollama pull qwen2.5:1.5b-instruct-fp16
    ollama pull qwen2.5:1.5b-instruct-q4_K_M

Run:
    python benchmark.py qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q4_K_M

For each model it:
  1. unloads anything else, so memory numbers are for one model only
  2. does one warm-up call (not counted, it includes load time)
  3. runs every prompt in prompts.txt with temperature 0
  4. records generation speed (tokens/sec), prompt speed, and loaded size from /api/ps
Writes results/<model>.json and results/comparison.md (outputs side by side).
"""
import json
import os
import statistics
import sys

import requests

HOST = "http://localhost:11434"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 300, "num_ctx": 4096}


def generate(model, prompt, keep_alive="10m"):
    r = requests.post(f"{HOST}/api/generate", timeout=600, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": OPTIONS, "keep_alive": keep_alive})
    r.raise_for_status()
    return r.json()


def unload_all():
    for m in requests.get(f"{HOST}/api/ps").json().get("models", []):
        requests.post(f"{HOST}/api/generate", json={"model": m["name"], "keep_alive": 0})


def loaded_size(model):
    for m in requests.get(f"{HOST}/api/ps").json().get("models", []):
        if m["name"] == model or m["model"] == model:
            return {"size_gb": round(m["size"] / 1e9, 2),
                    "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 2)}
    return None


def run(model, prompts):
    unload_all()
    generate(model, "Say hi.")  # warm-up, loads weights
    mem = loaded_size(model)
    rows = []
    for i, prompt in enumerate(prompts, 1):
        r = generate(model, prompt)
        gen_tps = r["eval_count"] / (r["eval_duration"] / 1e9)
        pp_tps = r.get("prompt_eval_count", 0) / max(r.get("prompt_eval_duration", 1) / 1e9, 1e-9)
        rows.append({"prompt": prompt, "output": r["response"],
                     "output_tokens": r["eval_count"],
                     "gen_tokens_per_sec": round(gen_tps, 1),
                     "prompt_tokens_per_sec": round(pp_tps, 1)})
        print(f"  [{i}/{len(prompts)}] {gen_tps:.1f} tok/s, {r['eval_count']} tokens")
    return {"model": model, "memory": mem,
            "median_gen_tokens_per_sec": statistics.median(x["gen_tokens_per_sec"] for x in rows),
            "median_prompt_tokens_per_sec": statistics.median(x["prompt_tokens_per_sec"] for x in rows),
            "rows": rows}


def main():
    models = sys.argv[1:]
    if len(models) != 2:
        sys.exit("Usage: python benchmark.py <model_a> <model_b>")
    prompts = [l.strip() for l in open("prompts.txt") if l.strip() and not l.startswith("#")]
    os.makedirs("results", exist_ok=True)

    results = []
    for m in models:
        print(f"\n== {m}")
        res = run(m, prompts)
        results.append(res)
        json.dump(res, open(f"results/{m.replace(':', '_')}.json", "w"), indent=2)
        print(f"  median generation speed: {res['median_gen_tokens_per_sec']} tok/s, memory: {res['memory']}")
    unload_all()

    a, b = results
    with open("results/comparison.md", "w") as f:
        f.write("| | " + a["model"] + " | " + b["model"] + " |\n|---|---|---|\n")
        f.write(f"| Median generation tok/s | {a['median_gen_tokens_per_sec']} | {b['median_gen_tokens_per_sec']} |\n")
        f.write(f"| Median prompt tok/s | {a['median_prompt_tokens_per_sec']} | {b['median_prompt_tokens_per_sec']} |\n")
        size = lambda r: r["memory"]["size_gb"] if r["memory"] else "n/a"
        f.write(f"| Loaded size (GB) | {size(a)} | {size(b)} |\n\n")
        for i, (ra, rb) in enumerate(zip(a["rows"], b["rows"]), 1):
            f.write(f"## Prompt {i}\n{ra['prompt']}\n\n")
            f.write(f"### {a['model']}\n{ra['output']}\n\n### {b['model']}\n{rb['output']}\n\n")
            f.write("**My notes:** \n\n---\n\n")
    print("\nWrote results/comparison.md. Add your notes under each prompt.")


if __name__ == "__main__":
    main()
