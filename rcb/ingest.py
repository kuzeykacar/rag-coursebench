"""PDF -> one record per page: {doc, page (1-based), lang, text}. Scanned PDFs without a text layer are skipped."""
import json, re, sys
from pathlib import Path
import pypdf

def clean(t):
    t = t.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"-\n(\w)", r"\1", t)              # re-join words split over lines
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def ingest(manifest, out):
    """manifest: json list of {doc, path, lang}."""
    n = 0
    with open(out, "w") as f:
        for m in json.load(open(manifest)):
            r = pypdf.PdfReader(m["path"])
            for i, p in enumerate(r.pages):
                t = clean(p.extract_text() or "")
                if len(t) < 40:                       # empty or image-only page
                    continue
                f.write(json.dumps(dict(doc=m["doc"], page=i + 1, lang=m["lang"], text=t), ensure_ascii=False) + "\n"); n += 1
    print("pages written:", n)

if __name__ == "__main__":
    ingest(sys.argv[1], sys.argv[2])
