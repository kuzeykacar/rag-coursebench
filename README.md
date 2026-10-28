# rag-coursebench

**What actually matters when you build RAG over real university course material?**

Retrieval-augmented generation (RAG) answers a question in two steps: first *retrieve* the passages that contain
the answer, then let a language model write the answer from them. If retrieval misses the right passage, the best
model in the world cannot answer. So this project measures **retrieval** carefully, one design choice at a time,
on the material I actually study with: lecture slides and notes from Politecnico di Torino and Università di Torino,
in **English and Italian**.

## The corpus and the test set

| Document | Language | Pages with text |
|---|---|---|
| Physics II, electrostatics slides | en | 82 |
| Computer Sciences, Unit T2 (computer architecture) | en | 61 |
| Macroeconomics slides (goods market → extended IS-LM) | en | 173 |
| Italian A2 course notes | it | 79 |

Scanned PDFs without a text layer (handwritten Analysis notes, chemistry slides) are skipped by design.
The PDFs themselves are **not** in this repo (they belong to the lecturers).

`data/questions.jsonl` holds **84 hand-checked questions**. Each has the document and the page(s) where the answer
is, a reference answer, and a type:

- **paraphrase** (76): asked in my own words, not copied from the slide. 61 in English, 15 in Italian.
- **crosslingual** (8): asked in **English** about the **Italian** notes. This is what you do when you study a
  language course: you think in one language and the notes are in another.

A retrieved chunk counts as correct if it comes from the right document and covers a gold page.

## What is compared

| Axis | Options |
|---|---|
| Chunking | fixed windows of 100 / 200 / 400 words (20% overlap), one chunk per page, paragraph-based |
| Retriever | BM25, TF-IDF, dense embeddings (MiniLM-en, multilingual-E5-small, BGE-M3), hybrid (reciprocal rank fusion) |
| Reranker | BGE-reranker-v2-m3 on the top 30 |

Metrics: recall@1/3/5/10 and MRR@10, with **bootstrap 95% intervals**, because 84 questions is a small sample and
differences of 2–3 points are often noise.

## First results: lexical retrieval (BM25 / TF-IDF)

Full table in [`results/lexical/retrieval.csv`](results/lexical/retrieval.csv).

| Chunking | Retriever | recall@1 | recall@5 | 95% CI (r@5) | MRR@10 | r@5 cross-lingual |
|---|---|---|---|---|---|---|
| fixed 200 | BM25 | 0.667 | 0.893 | 0.82–0.95 | 0.753 | **0.000** |
| fixed 100 | TF-IDF | 0.619 | 0.893 | 0.82–0.95 | 0.718 | **0.000** |
| fixed 400 | TF-IDF | 0.679 | 0.881 | 0.81–0.94 | 0.760 | 0.125 |
| page | BM25 | 0.643 | 0.845 | 0.76–0.92 | 0.728 | **0.000** |
| paragraph | BM25 | 0.655 | 0.810 | 0.73–0.89 | 0.726 | **0.000** |

What this already shows:

1. **Keyword search is strong on same-language questions** (about 0.9 recall@5) even when the question is
   paraphrased. Course slides use a small, consistent vocabulary.
2. **Keyword search is useless across languages**: 0 of 8 English questions about the Italian notes found the right
   page in the top 5. This is the gap embeddings are supposed to close, and the next experiment measures by how much.
3. **Paragraph chunks are the worst** here. Slides have no real paragraphs, so the splitter makes tiny, context-poor
   chunks. Fixed 200-word windows that cross slide boundaries do better.

Next: dense and hybrid retrievers, the reranker, and an end-to-end answer accuracy test with a local LLM.

## Run it

```bash
# lexical baselines only (no downloads)
pip install pypdf numpy pandas scikit-learn
python -m rcb.ingest data/manifest.json work/pages.jsonl
python scripts/run_benchmark.py --out results/lexical

# everything, including embedding models and the reranker
./scripts/run_all.sh
```

Use your own PDFs: put them in `pdfs/`, list them in `data/manifest.json` (doc id, path, language), and write
questions in the same format as `data/questions.jsonl`.

## Layout

```
rcb/ingest.py     PDF -> one record per page (text layer only)
rcb/chunk.py      chunking strategies; every chunk remembers its document and pages
rcb/retrieve.py   BM25 (from scratch), TF-IDF, dense, hybrid (RRF), cross-encoder reranker
rcb/evaluate.py   recall@k, MRR, bootstrap intervals, paired differences
scripts/          benchmark runner
data/             question set + example manifest
results/          CSV results per run
```
