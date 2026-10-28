"""Chunking strategies. Every chunk keeps (doc, page) of the page where it starts, so retrieval can be scored by page."""
import re

def _words(t):
    return t.split()

def fixed(pages, size=200, overlap=40):
    """Sliding windows of `size` words over each document (crossing page breaks), stride size-overlap."""
    out = []
    by_doc = {}
    for p in pages:
        by_doc.setdefault(p["doc"], []).append(p)
    for doc, ps in by_doc.items():
        toks, pg = [], []
        for p in ps:
            w = _words(p["text"]); toks += w; pg += [p["page"]] * len(w)
        step = max(1, size - overlap)
        for s in range(0, max(1, len(toks) - overlap), step):
            e = min(len(toks), s + size)
            out.append(dict(doc=doc, page=pg[s], pages=sorted(set(pg[s:e])), lang=ps[0]["lang"], text=" ".join(toks[s:e])))
            if e == len(toks):
                break
    return out

def page(pages):
    """One chunk per page (slides map naturally to this)."""
    return [dict(doc=p["doc"], page=p["page"], pages=[p["page"]], lang=p["lang"], text=p["text"]) for p in pages]

def paragraph(pages, min_words=30, max_words=250):
    """Split each page on blank lines and bullet starts, then merge pieces until a chunk has at least min_words."""
    out = []
    for p in pages:
        parts = [x.strip() for x in re.split(r"\n\s*\n|\n(?=[\u2022\-\*\u25aa\u25cf]\s)", p["text"]) if x.strip()]
        texts, buf = [], []
        for part in parts:
            if buf and len(" ".join(buf + [part]).split()) > max_words:
                texts.append(" ".join(buf)); buf = []
            buf.append(part)
            if len(" ".join(buf).split()) >= min_words:
                texts.append(" ".join(buf)); buf = []
        if buf:
            if texts and len(" ".join(buf).split()) < min_words:
                texts[-1] += " " + " ".join(buf)
            else:
                texts.append(" ".join(buf))
        out += [dict(doc=p["doc"], page=p["page"], pages=[p["page"]], lang=p["lang"], text=t) for t in texts]
    return out

STRATEGIES = {
    "fixed100": lambda P: fixed(P, 100, 20),
    "fixed200": lambda P: fixed(P, 200, 40),
    "fixed400": lambda P: fixed(P, 400, 80),
    "page": page,
    "paragraph": paragraph,
}
