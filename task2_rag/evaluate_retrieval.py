"""Measure retrieval quality for two chunk sizes.

    python evaluate_retrieval.py --pdf data/llama2.pdf --questions questions.json

For each chunk size it reports recall@1, recall@5 and MRR for:
  dense  : embedding search only (the required system)
  bm25   : keyword search only (optional comparison)
  hybrid : dense + bm25 combined with reciprocal rank fusion (optional)
and saves every miss to results/failures_<size>.txt so you can pick 2 to discuss.
"""
import argparse
import json
import os

from rag import BM25, DenseIndex, Embedder, chunk_pages, is_hit, load_pdf_pages, rrf

p = argparse.ArgumentParser()
p.add_argument("--pdf", required=True)
p.add_argument("--questions", default="questions.json")
p.add_argument("--sizes", type=int, nargs="+", default=[150, 500])
p.add_argument("--overlap_ratio", type=float, default=0.2)
p.add_argument("--k", type=int, default=5)
args = p.parse_args()

pages = load_pdf_pages(args.pdf)
print(f"{len(pages)} pages, {sum(len(x['text'].split()) for x in pages)} words")
questions = json.load(open(args.questions))
assert all("question" in q and "gold_passage" in q for q in questions)
if any("REPLACE" in q["question"] for q in questions):
    raise SystemExit("questions.json still has placeholder entries. Write your 15 questions first.")

emb = Embedder()
qvecs = [emb.query(q["question"]) for q in questions]
os.makedirs("results", exist_ok=True)
report = {}


def metrics(ranked_lists, chunks):
    r1 = r5 = mrr = 0.0
    first_hits = []
    for q, ranked in zip(questions, ranked_lists):
        hit_rank = next((i for i, cid in enumerate(ranked)
                         if is_hit(chunks[cid]["text"], q["gold_passage"])), None)
        first_hits.append(hit_rank)
        if hit_rank is not None:
            r1 += hit_rank == 0
            r5 += hit_rank < args.k
            mrr += 1 / (hit_rank + 1)
    n = len(questions)
    return {"recall@1": round(r1 / n, 3), f"recall@{args.k}": round(r5 / n, 3),
            "mrr": round(mrr / n, 3)}, first_hits


for size in args.sizes:
    chunks = chunk_pages(pages, size, int(size * args.overlap_ratio))
    vecs = emb.docs([c["text"] for c in chunks])
    dense, bm25 = DenseIndex(vecs), BM25([c["text"] for c in chunks])

    dense_ranked = [dense.search(qv, args.k)[0] for qv in qvecs]
    bm25_ranked = [bm25.search(q["question"], args.k)[0] for q in questions]
    wide = 20  # fuse over deeper lists, then cut to k
    hybrid_ranked = [rrf([dense.search(qv, wide)[0], bm25.search(q["question"], wide)[0]], args.k)
                     for q, qv in zip(questions, qvecs)]

    report[size] = {"num_chunks": len(chunks)}
    for name, ranked in [("dense", dense_ranked), ("bm25", bm25_ranked), ("hybrid", hybrid_ranked)]:
        m, first = metrics(ranked, chunks)
        report[size][name] = m
        if name == "dense":
            with open(f"results/failures_{size}.txt", "w") as f:
                for q, ranked_ids, fh in zip(questions, ranked, first):
                    if fh is None:
                        f.write(f"Q: {q['question']}\nGOLD: {q['gold_passage']}\n")
                        for cid in ranked_ids:
                            f.write(f"  got p{chunks[cid]['page']}: {chunks[cid]['text'][:200]}...\n")
                        f.write("\n")
    print(f"\nchunk size {size} words ({len(chunks)} chunks)")
    for name in ("dense", "bm25", "hybrid"):
        print(f"  {name:7s} {report[size][name]}")

json.dump(report, open("results/retrieval_report.json", "w"), indent=2)
print("\nSaved results/retrieval_report.json and results/failures_<size>.txt")
