"""Retrievers. All share one interface: fit(chunks) then search(query, k) -> list of chunk indices, best first.
BM25 and TF-IDF need nothing but numpy/scikit-learn. Dense retrievers and the reranker use sentence-transformers
(downloaded on first use)."""
import math, re, unicodedata
from collections import Counter
import numpy as np

def tokenize(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))      # perché -> perche
    return re.findall(r"[a-z0-9]+", t)

class BM25:
    name = "bm25"
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
    def fit(self, chunks):
        docs = [tokenize(c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in docs]; self.len = np.array([len(d) for d in docs], float)
        self.avg = self.len.mean(); N = len(docs); df = Counter()
        for d in self.tf:
            df.update(d.keys())
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.inv = {}
        for i, d in enumerate(self.tf):
            for w, c in d.items():
                self.inv.setdefault(w, []).append((i, c))
        return self
    def scores(self, q):
        s = np.zeros(len(self.tf))
        for w in set(tokenize(q)):
            if w not in self.inv:
                continue
            idf = self.idf[w]
            for i, c in self.inv[w]:
                s[i] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s
    def search(self, q, k=10):
        return list(np.argsort(-self.scores(q))[:k])

class TFIDF:
    name = "tfidf"
    def fit(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True, token_pattern=None)
        self.X = self.v.fit_transform([c["text"] for c in chunks]); return self
    def scores(self, q):
        return (self.X @ self.v.transform([q]).T).toarray().ravel()
    def search(self, q, k=10):
        return list(np.argsort(-self.scores(q))[:k])

class Dense:
    """Embedding retriever, e.g. 'intfloat/multilingual-e5-small', 'BAAI/bge-m3', 'sentence-transformers/all-MiniLM-L6-v2'."""
    def __init__(self, model, query_prefix="", doc_prefix=""):
        self.model_name, self.qp, self.dp = model, query_prefix, doc_prefix
        self.name = "dense:" + model.split("/")[-1]
    def fit(self, chunks):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(self.model_name)
        self.m.max_seq_length = min(self.m.max_seq_length or 512, 512)   # chunks are <= ~550 tokens; caps memory
        self.E = self.m.encode([self.dp + c["text"] for c in chunks], batch_size=8, normalize_embeddings=True,
                               show_progress_bar=False)
        return self
    def scores(self, q):
        return self.E @ self.m.encode([self.qp + q], normalize_embeddings=True)[0]
    def search(self, q, k=10):
        return list(np.argsort(-self.scores(q))[:k])

class Hybrid:
    """Reciprocal rank fusion of several retrievers (no score calibration needed)."""
    def __init__(self, *rs, c=60, depth=50):
        self.rs, self.c, self.depth = rs, c, depth
        self.name = "hybrid(" + "+".join(r.name for r in rs) + ")"
    def fit(self, chunks):
        for r in self.rs:
            r.fit(chunks)
        return self
    def search(self, q, k=10):
        s = {}
        for r in self.rs:
            for rank, i in enumerate(r.search(q, self.depth)):
                s[i] = s.get(i, 0.0) + 1.0 / (self.c + rank + 1)
        return sorted(s, key=lambda i: -s[i])[:k]

class Rerank:
    """Re-sort the first `depth` hits of a retriever with a cross-encoder, e.g. 'BAAI/bge-reranker-v2-m3'."""
    def __init__(self, base, model, depth=30):
        self.base, self.model_name, self.depth = base, model, depth
        self.name = base.name + "+rerank:" + model.split("/")[-1]
    def fit(self, chunks):
        from sentence_transformers import CrossEncoder
        self.base.fit(chunks); self.chunks = chunks; self.ce = CrossEncoder(self.model_name, max_length=512); return self
    def search(self, q, k=10):
        cand = self.base.search(q, self.depth)
        s = self.ce.predict([(q, self.chunks[i]["text"]) for i in cand], batch_size=4)
        return [cand[j] for j in np.argsort(-s)[:k]]
