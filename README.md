# tndrr AI/ML Internship Assignment

Anirudh Annaboina, B.Tech CSE (AI & ML), CMR Technical Campus

| Folder | Task | Where it runs |
|---|---|---|
| `task1_lora/` | LoRA / QLoRA fine-tune of Qwen2.5-0.5B-Instruct on Banking77 intent classification | Colab or Kaggle T4 |
| `task2_rag/` | Question answering over the Llama 2 paper (77 pages) with embeddings + FAISS, no frameworks | Colab or Kaggle T4 |
| `task3_local_inference/` | Qwen2.5-1.5B-Instruct in fp16 vs 4-bit (Q4_K_M) with Ollama | Colab T4 (also runs on a Mac) |

Results from the submitted run: Task 1 accuracy 12.5% -> 87.0%; Task 2 recall@5 0.80 (150-word chunks, dense) and 0.93 (hybrid); Task 3 63.6 -> 139.9 tok/s and 3.28 -> 1.17 GB. Raw outputs are in each task's `outputs/` or `results/` folder.

```bash
pip install -r requirements.txt
```

## Task 1: fine-tuning

The model reads a bank customer message and must reply `{"intent": "<label>"}` with one of 77 labels. Base and fine-tuned models get the same prompt (instructions plus the full label list), so the only difference is the LoRA training.

- Train: 3,000 examples from the Banking77 train split
- Validation: 200 other train examples (eval loss during training)
- Test: 200 examples from the official test split, never seen in training
- Metric: exact-match accuracy of the predicted label. Also reported: valid JSON rate and how often the answer is a real label.

```bash
cd task1_lora
python evaluate.py --which base                 # baseline
python train.py                                  # LoRA (add --qlora for QLoRA)
python evaluate.py --which finetuned --adapter outputs/adapter
python show_failures.py                          # pick 3 failures for the report
```

Training stats (time, peak GPU memory, loss curve) are saved to `outputs/adapter/train_stats.json`. The 35 MB adapter weights are not committed; `python train.py` recreates them in about 20 minutes on a T4.

## Task 2: RAG from scratch

```bash
cd task2_rag
mkdir -p data && wget -O data/llama2.pdf https://arxiv.org/pdf/2307.09288
python evaluate_retrieval.py --pdf data/llama2.pdf --sizes 150 500
python ask.py --pdf data/llama2.pdf --size 500 "your question here"
```

- Chunking: fixed word windows with 20% overlap, page number kept per chunk
- Embeddings: `BAAI/bge-small-en-v1.5`, normalised, searched with FAISS `IndexFlatIP` (cosine)
- A retrieved chunk counts as correct if it contains at least half of the gold passage's word 3-grams
- Also compared: BM25 written in NumPy, and dense + BM25 merged with reciprocal rank fusion
- Answers: Qwen2.5-1.5B-Instruct, told to answer only from the retrieved chunks and cite them

## Task 3: local inference

On Colab: install with `curl -fsSL https://ollama.com/install.sh | sh`, start `ollama serve` in the background, then the same commands. Locally:

```bash
cd task3_local_inference
ollama serve   # separate terminal
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python benchmark.py qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q4_K_M
```

Results go to `results/`. `comparison.md` shows both models' answers to the same 10 prompts side by side.

## Colab quick start

```
!git clone https://github.com/<you>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt
%cd task1_lora
!python evaluate.py --which base
```

Use Runtime > Change runtime type > T4 GPU.
