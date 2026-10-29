"""Run every (chunking x retriever) combination and write results/retrieval.csv + per-question ranks.

  python scripts/run_benchmark.py --pages work/pages.jsonl --questions data/questions.jsonl [--dense] [--rerank]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from rcb.chunk import STRATEGIES
from rcb.retrieve import BM25, TFIDF, Dense, Hybrid, Rerank
from rcb.evaluate import load_questions, evaluate, bootstrap

DENSE = {"minilm-en": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
         "e5-small-multi": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
         "bge-m3": ("BAAI/bge-m3", "", "")}
RERANKER = "BAAI/bge-reranker-v2-m3"

def retrievers(args):
    yield "bm25", lambda: BM25()
    yield "tfidf", lambda: TFIDF()
    if args.dense:
        for k, (m, qp, dp) in DENSE.items():
            yield k, (lambda m=m, qp=qp, dp=dp: Dense(m, qp, dp))
        yield "hybrid(bm25+e5)", lambda: Hybrid(BM25(), Dense(*DENSE["e5-small-multi"]))
        yield "hybrid(bm25+bge-m3)", lambda: Hybrid(BM25(), Dense(*DENSE["bge-m3"]))
    if args.rerank:
        yield "hybrid(bm25+bge-m3)+rerank", lambda: Rerank(Hybrid(BM25(), Dense(*DENSE["bge-m3"])), RERANKER)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="work/pages.jsonl"); ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--chunking", nargs="*", default=list(STRATEGIES)); ap.add_argument("--dense", action="store_true")
    ap.add_argument("--rerank", action="store_true"); ap.add_argument("--out", default="results")
    ap.add_argument("--resume", help="CSV of rows already computed; those (chunking, retriever) pairs are skipped")
    args = ap.parse_args()
    pages = [json.loads(l) for l in open(args.pages)]; Q = load_questions(args.questions)
    docs = {p["doc"] for p in pages}; Q = [q for q in Q if q["doc"] in docs]
    rows, ranks = [], {}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and Path(args.resume).exists():
        prev = pd.read_csv(args.resume); rows = prev.to_dict("records"); done = set(zip(prev.chunking, prev.retriever))
    for cname in args.chunking:
        chunks = STRATEGIES[cname](pages)
        for rname, make in retrievers(args):
            if (cname, rname) in done:
                continue
            t0 = time.time()
            try:
                r = make().fit(chunks); m, rk = evaluate(r, chunks, Q)
            except RuntimeError as e:                       # e.g. out of GPU memory: record it and go on
                print("FAILED", cname, rname, str(e)[:200], flush=True)
                continue
            lo, hi = bootstrap(rk, 5)
            row = dict(chunking=cname, retriever=rname, n_chunks=len(chunks), **{k: round(v, 3) for k, v in m.items()},
                       r5_lo=round(lo, 3), r5_hi=round(hi, 3), seconds=round(time.time() - t0, 1))
            for kind in sorted({q["kind"] for q in Q}):
                idx = [i for i, q in enumerate(Q) if q["kind"] == kind]
                row[f"recall@5[{kind}]"] = round(sum(rk[i] is not None and rk[i] <= 5 for i in idx) / len(idx), 3)
            rows.append(row); ranks[f"{cname}|{rname}"] = rk
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(f"{args.out}/retrieval.csv", index=False)      # save after every row
            try:
                import gc, torch; gc.collect(); torch.mps.empty_cache()
            except Exception:
                pass
    pd.DataFrame(rows).to_csv(f"{args.out}/retrieval.csv", index=False)
    json.dump({"questions": [q["id"] for q in Q], "ranks": ranks}, open(f"{args.out}/ranks.json", "w"))
