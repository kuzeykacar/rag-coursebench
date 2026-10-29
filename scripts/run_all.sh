#!/usr/bin/env bash
# Full benchmark on your own machine (macOS / Linux). Put your PDFs in pdfs/ and list them in data/manifest.json.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt
[ -f data/manifest.json ] || cp data/manifest.example.json data/manifest.json
mkdir -p work
python -m rcb.ingest data/manifest.json work/pages.jsonl
python scripts/run_benchmark.py --dense --rerank --out results/full ${RESUME:+--resume "$RESUME"}
