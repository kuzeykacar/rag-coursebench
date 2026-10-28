"""Retrieval metrics against page-level gold labels, with bootstrap confidence intervals."""
import json, random
import numpy as np

def load_questions(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def is_hit(chunk, q):
    return chunk["doc"] == q["doc"] and bool(set(chunk["pages"]) & set(q["pages"]))

def rank_of_first_hit(chunks, ids, q):
    for r, i in enumerate(ids):
        if is_hit(chunks[i], q):
            return r + 1
    return None

def evaluate(retriever, chunks, questions, ks=(1, 3, 5, 10)):
    ranks = [rank_of_first_hit(chunks, retriever.search(q["question"], max(ks)), q) for q in questions]
    out = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in ks}
    out["mrr@10"] = float(np.mean([1.0 / r if r else 0.0 for r in ranks]))
    return out, ranks

def bootstrap(ranks, k=5, n=2000, seed=0):
    """95% interval for recall@k by resampling questions."""
    rng = random.Random(seed); v = [r is not None and r <= k for r in ranks]; m = []
    for _ in range(n):
        s = [v[rng.randrange(len(v))] for _ in v]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]

def paired_diff(ranks_a, ranks_b, k=5, n=2000, seed=0):
    """Bootstrap interval of recall@k(A) - recall@k(B) on the same questions."""
    rng = random.Random(seed)
    a = [r is not None and r <= k for r in ranks_a]; b = [r is not None and r <= k for r in ranks_b]; d = []
    for _ in range(n):
        idx = [rng.randrange(len(a)) for _ in a]
        d.append(sum(a[i] - b[i] for i in idx) / len(idx))
    d.sort(); return sum(x - y for x, y in zip(a, b)) / len(a), (d[int(0.025 * n)], d[int(0.975 * n)])
