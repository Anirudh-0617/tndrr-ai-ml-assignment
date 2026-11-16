"""RAG building blocks with no LangChain / LlamaIndex.

Pieces:
  load_pdf_pages   PDF -> list of {"page", "text"}
  chunk_pages      fixed-size word windows with overlap, page number kept
  Embedder         sentence-transformers model, returns L2-normalised vectors
  DenseIndex       cosine search with FAISS (falls back to plain NumPy)
  BM25             keyword scoring written in NumPy (optional experiment)
  rrf              reciprocal rank fusion to combine dense + BM25 rankings
  is_hit           does a retrieved chunk contain the gold passage?
"""
import math
import re
from collections import Counter

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------- loading and chunking ----------

def clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)   # join words hyphenated across lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf_pages(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return [{"page": i + 1, "text": clean(p.extract_text() or "")}
            for i, p in enumerate(reader.pages)]


def chunk_pages(pages, size, overlap):
    """Slide a window of `size` words over the whole document, step = size - overlap.
    Each chunk records the page its first word came from."""
    words, word_page = [], []
    for p in pages:
        w = p["text"].split()
        words += w
        word_page += [p["page"]] * len(w)
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append({"id": len(chunks), "page": word_page[start], "text": " ".join(piece)})
        if start + size >= len(words):
            break
    return chunks


# ---------- dense retrieval ----------

class Embedder:
    def __init__(self, name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def docs(self, texts):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=True).astype("float32")

    def query(self, text):
        return self.model.encode([QUERY_PREFIX + text],
                                 normalize_embeddings=True).astype("float32")


class DenseIndex:
    """Inner product on normalised vectors = cosine similarity."""

    def __init__(self, vectors):
        self.vectors = vectors
        try:
            import faiss
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        except ImportError:
            self.index = None

    def search(self, qvec, k):
        if self.index is not None:
            scores, ids = self.index.search(qvec, k)
            return ids[0].tolist(), scores[0].tolist()
        sims = self.vectors @ qvec[0]
        top = np.argsort(-sims)[:k]
        return top.tolist(), sims[top].tolist()


# ---------- BM25 (keyword) ----------

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean()
        n = len(self.docs)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avg)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=float)
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s

    def search(self, query, k):
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return top.tolist(), s[top].tolist()


def rrf(rankings, k, c=60):
    """Combine several ranked id lists: score(id) = sum 1 / (c + rank)."""
    score = Counter()
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            score[doc_id] += 1.0 / (c + rank + 1)
    return [d for d, _ in score.most_common(k)]


# ---------- evaluation helper ----------

def shingles(text, n=3):
    w = tokenize(text)
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def is_hit(chunk_text, gold_passage, threshold=0.5):
    """Hit if at least half of the gold passage's word 3-grams appear in the chunk.
    Fuzzy on purpose: PDF text extraction rarely matches a copy-pasted quote exactly,
    and a passage can be split across two chunks."""
    g = shingles(gold_passage)
    if not g:
        return False
    return len(g & shingles(chunk_text)) / len(g) >= threshold
