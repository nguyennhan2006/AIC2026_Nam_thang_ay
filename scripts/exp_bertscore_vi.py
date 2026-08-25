"""EXP-METEOR phan 2: soft alignment (BERTScore) tren tieng Viet.

exp_meteor_lexical.py cho thay METEOR thua BM25 vi alignment cua no la
exact-match roi rac va WordNet chi co tieng Anh. Cau hoi tiep: neu thay
module alignment bang similarity embedding (BERTScore) -- van la "cham
diem theo do phu query" nhung mem va KHA VI -- thi co thang khong?

Neu KHONG thang: ca huong "dung do do kieu METEOR de neo query" nen bo.
Neu THANG: day moi la ung vien lam loss/soft-label, khong phai METEOR.

Chay: python -m scripts.exp_bertscore_vi
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_meteor_lexical import (  # noqa: E402
    _TOKEN_RE,
    BM25,
    is_hit,
    load_corpus,
    load_queries,
)

MODEL = "intfloat/multilingual-e5-small"
KS = (1, 5, 10, 20, 50)


@torch.inference_mode()
def encode_tokens(texts: list[str], prefix: str, batch: int = 32):
    """Tra ve list embedding token da chuan hoa L2 cho tung text."""
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).eval()
    out = []
    for start in range(0, len(texts), batch):
        chunk = [prefix + t for t in texts[start : start + batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
        hidden = model(**enc).last_hidden_state  # (B, T, H)
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        mask = enc["attention_mask"].bool()
        for row, m in zip(hidden, mask):
            # bo [CLS]/[EOS]: giu token noi dung
            vecs = row[m][1:-1]
            out.append(vecs.numpy().astype(np.float32))
        print(f"  encoded {min(start + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def bertscore(q_vecs, d_vecs, alpha=0.5, q_idf=None):
    """BERTScore: greedy soft alignment, F thien recall theo alpha (w_R=alpha)."""
    if len(q_vecs) == 0 or len(d_vecs) == 0:
        return 0.0
    sim = q_vecs @ d_vecs.T  # (Tq, Td)
    r_per = sim.max(axis=1)  # moi token query khop tot nhat den dau
    p_per = sim.max(axis=0)
    if q_idf is not None:
        w = q_idf / max(q_idf.sum(), 1e-9)
        r = float((r_per * w).sum())
    else:
        r = float(r_per.mean())
    p = float(p_per.mean())
    denom = alpha * p + (1 - alpha) * r
    return (p * r / denom) if denom > 0 else 0.0


def evaluate(name, score_matrix, queries, scenes):
    """score_matrix[qi][si]."""
    recall = {k: 0 for k in KS}
    rr = 0.0
    for qi, q in enumerate(queries):
        order = np.argsort(-score_matrix[qi])
        for pos, i in enumerate(order, 1):
            if is_hit(scenes[i], q):
                rr += 1 / pos
                for k in KS:
                    if pos <= k:
                        recall[k] += 1
                break
    n = len(queries)
    return {"name": name, "mrr": rr / n, **{f"R@{k}": recall[k] / n for k in KS}}


def main() -> None:
    scenes = load_corpus()
    queries = load_queries()
    print(f"corpus {len(scenes)} scene | {len(queries)} query KIS | model {MODEL}")

    print("encoding scenes...")
    doc_vecs = encode_tokens([s.text for s in scenes], "passage: ")
    print("encoding queries...")
    q_vecs = encode_tokens([q["query_vi"] for q in queries], "query: ")

    # IDF cap token cho query (BERTScore ban co idf-weighting)
    bm25 = BM25(scenes)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    n_doc = len(scenes)
    df: Counter[str] = Counter()
    for s in scenes:
        ids = tok(("passage: " + s.text), truncation=True, max_length=256)["input_ids"]
        df.update(set(ids[1:-1]))
    idf_map = defaultdict(
        lambda: math.log(1 + (n_doc + 0.5) / 0.5),
        {t: math.log(1 + (n_doc - c + 0.5) / (c + 0.5)) for t, c in df.items()},
    )
    q_idfs = []
    for q in queries:
        ids = tok(("query: " + q["query_vi"]), truncation=True, max_length=256)["input_ids"][1:-1]
        q_idfs.append(np.array([idf_map[i] for i in ids], dtype=np.float32))

    variants = {
        "BERTScore a=.5": (0.5, False),
        "BERTScore a=.9 (recall)": (0.9, False),
        "BERTScore a=.5 +idf": (0.5, True),
        "BERTScore a=.9 +idf": (0.9, True),
    }
    mats: dict[str, np.ndarray] = {}
    for name, (alpha, use_idf) in variants.items():
        mat = np.zeros((len(queries), len(scenes)), dtype=np.float32)
        for qi in range(len(queries)):
            qv = q_vecs[qi]
            qw = q_idfs[qi][: len(qv)] if use_idf else None
            for si in range(len(scenes)):
                mat[qi, si] = bertscore(qv, doc_vecs[si], alpha=alpha, q_idf=qw)
        mats[name] = mat
        print(f"  done {name}")

    # baseline BM25 de so sanh tren cung bang
    bm = np.zeros((len(queries), len(scenes)), dtype=np.float32)
    for qi, q in enumerate(queries):
        qt = [t.casefold() for t in _TOKEN_RE.findall(q["query_vi"])]
        for si in range(len(scenes)):
            bm[qi, si] = bm25.score(qt, si)
    mats["BM25 (baseline)"] = bm

    hdr = f"{'config':<26}{'MRR':>7}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'AUC':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    labels = np.array(
        [[1 if is_hit(s, q) else 0 for s in scenes] for q in queries]
    ).ravel()
    for name, mat in mats.items():
        row = evaluate(name, mat, queries, scenes)
        auc = roc_auc_score(labels, mat.ravel())
        line = f"{row['name']:<26}{row['mrr']:>7.3f}"
        line += "".join(f"{row['R@' + str(k)]:>8.3f}" for k in KS)
        print(line + f"{auc:>8.4f}")


if __name__ == "__main__":
    main()
