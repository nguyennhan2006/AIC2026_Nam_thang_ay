"""EXP-METEOR: METEOR-style alignment co thang lexical coverage hien tai khong?

Cau hoi: co the dung METEOR lam tin hieu neo query trong retrieval/huan luyen?
Truoc khi ban toi RL/distillation, phai tra loi cau re nhat: co che cham diem
cua METEOR (alignment + F-mean thien recall + phat fragmentation) co tach duoc
scene dung khoi scene sai tren chinh du lieu cua minh khong.

Corpus: storage/exports_pack_L21test (614 scene, caption VI, OCR 0%).
Query : examples/gold_all3.jsonl, task=KIS, dung query_vi.
Hit   : scene cung video VA giao khoang voi target_intervals.

Chay:  python -m scripts.exp_meteor_lexical
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online.services.lexical_coverage import (  # noqa: E402
    STOPWORDS,
    compute_coverage,
    strip_accents,
)

PACK = ROOT / "storage/exports_pack_L21test"
DISTRACTOR = ROOT / "storage/exports_pack_L23"  # 621 scene khong bao gio la gold
GOLD = ROOT / "examples/gold_all3.jsonl"
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


# --------------------------------------------------------------- du lieu
@dataclass(slots=True)
class Scene:
    scene_id: str
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    tokens: list[str]


def _load_pack(pack: Path) -> list[Scene]:
    caps: dict[str, list[str]] = defaultdict(list)
    with (pack / "keyframes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            for c in d.get("captions") or []:
                if c.get("text"):
                    caps[d["scene_id"]].append(c["text"])
            for o in d.get("ocr_instances") or []:
                if o.get("text"):
                    caps[d["scene_id"]].append(o["text"])
    scenes = []
    with (pack / "scenes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            text = " ".join(caps.get(d["scene_id"], []))
            scenes.append(
                Scene(
                    d["scene_id"],
                    d["video_id"],
                    float(d["start_sec"]),
                    float(d["end_sec"]),
                    text,
                    [t.casefold() for t in _TOKEN_RE.findall(text)],
                )
            )
    return scenes


def load_corpus(with_distractors: bool = True) -> list[Scene]:
    scenes = _load_pack(PACK)
    if with_distractors and DISTRACTOR.exists():
        scenes += _load_pack(DISTRACTOR)
    return scenes


def load_queries() -> list[dict]:
    out = []
    with GOLD.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("task") == "KIS" and d.get("target_intervals"):
                out.append(d)
    return out


def is_hit(scene: Scene, q: dict) -> bool:
    if scene.video_id != q["target_video"]:
        return False
    return any(
        scene.start_sec < iv["end_sec"] and scene.end_sec > iv["start_sec"]
        for iv in q["target_intervals"]
    )


# --------------------------------------------------------------- scorers
def build_idf(scenes: list[Scene]) -> dict[str, float]:
    n = len(scenes)
    df: Counter[str] = Counter()
    for s in scenes:
        df.update(set(s.tokens))
    return {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}


class BM25:
    def __init__(self, scenes: list[Scene], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(s.tokens) for s in scenes]
        self.lens = [max(len(s.tokens), 1) for s in scenes]
        self.avgdl = sum(self.lens) / len(self.lens)
        self.idf = build_idf(scenes)

    def score(self, q_tokens: list[str], i: int) -> float:
        tf, dl, total = self.docs[i], self.lens[i], 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return total


def _norm_tokens(text: str) -> list[str]:
    """Token da bo dau, bo hu tu -- 'stem' kha di duy nhat cho tieng Viet."""
    return [
        strip_accents(t)
        for t in _TOKEN_RE.findall(text.casefold())
        if len(t) > 1 and strip_accents(t) not in STOPWORDS
    ]


def meteor(
    ref_tokens: list[str],
    hyp_tokens: list[str],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
    weights: dict[str, float] | None = None,
) -> tuple[float, float, float, float]:
    """METEOR(ref=query, hyp=caption). Tra (score, P, R, penalty).

    alpha la TRONG SO RECALL (w_R = alpha) theo cong thuc goc
    F = P*R / (alpha*P + (1-alpha)*R). alpha=1.0 => recall thuan.
    gamma=0 => tat phat fragmentation.
    weights != None => METEOR-IDF: P/R can theo IDF thay vi dem token deu nhau.
    METEOR goc KHONG co cho nay -- trong dich may ref/hyp cung do dai cung noi
    dung nen khong can, con trong retrieval thi "nguoi" va "Copenhagen" khong
    the dang gia nhu nhau.
    """
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0, 0.0

    # alignment greedy exact-match, 1-1, uu tien vi tri trai nhat
    used_hyp: dict[int, int] = {}  # idx_hyp -> idx_ref
    hyp_pos: dict[str, list[int]] = defaultdict(list)
    for j, t in enumerate(hyp_tokens):
        hyp_pos[t].append(j)
    taken: set[int] = set()
    for i, t in enumerate(ref_tokens):
        for j in hyp_pos.get(t, ()):
            if j not in taken:
                taken.add(j)
                used_hyp[j] = i
                break

    m = len(used_hyp)
    if m == 0:
        return 0.0, 0.0, 0.0, 0.0

    if weights is None:
        p = m / len(hyp_tokens)
        r = m / len(ref_tokens)
    else:
        w = lambda t: weights.get(t, 0.0)  # noqa: E731
        matched_w = sum(w(ref_tokens[i]) for i in used_hyp.values())
        hyp_w = sum(w(t) for t in hyp_tokens)
        ref_w = sum(w(t) for t in ref_tokens)
        if hyp_w <= 0 or ref_w <= 0:
            return 0.0, 0.0, 0.0, 0.0
        p = matched_w / hyp_w
        r = matched_w / ref_w
    denom = alpha * p + (1 - alpha) * r
    fmean = (p * r / denom) if denom > 0 else 0.0

    # dem chunk: cap aligned lien ke o CA hyp va ref moi cung mot chunk
    pairs = sorted(used_hyp.items())
    chunks = 1
    for (jp, ip), (jc, ic) in zip(pairs, pairs[1:]):
        if not (jc == jp + 1 and ic == ip + 1):
            chunks += 1
    penalty = gamma * (chunks / m) ** beta
    return fmean * (1 - penalty), p, r, penalty


# --------------------------------------------------------------- danh gia
KS = (1, 5, 10, 20, 50)


def evaluate(name: str, ranker, queries, scenes) -> dict:
    recall = {k: 0 for k in KS}
    vrecall = {k: 0 for k in KS}
    rr = 0.0
    for q in queries:
        scored = ranker(q)
        order = sorted(range(len(scenes)), key=lambda i: -scored[i])
        rank = None
        vrank = None
        for pos, i in enumerate(order, 1):
            if rank is None and is_hit(scenes[i], q):
                rank = pos
            if vrank is None and scenes[i].video_id == q["target_video"]:
                vrank = pos
            if rank is not None and vrank is not None:
                break
        if rank:
            rr += 1 / rank
            for k in KS:
                if rank <= k:
                    recall[k] += 1
        if vrank:
            for k in KS:
                if vrank <= k:
                    vrecall[k] += 1
    n = len(queries)
    return {
        "name": name,
        "mrr": rr / n,
        **{f"R@{k}": recall[k] / n for k in KS},
        **{f"vR@{k}": vrecall[k] / n for k in KS},
    }


def main() -> None:
    scenes = load_corpus()
    queries = load_queries()
    empty = sum(1 for s in scenes if not s.tokens)
    print(f"corpus: {len(scenes)} scene ({empty} rong caption) | queries KIS: {len(queries)}")
    print(f"avg caption tokens: {sum(len(s.tokens) for s in scenes) / len(scenes):.1f}")

    bm25 = BM25(scenes)
    scene_norm = [_norm_tokens(s.text) for s in scenes]
    idf = bm25.idf

    def r_bm25(q):
        qt = [t.casefold() for t in _TOKEN_RE.findall(q["query_vi"])]
        return [bm25.score(qt, i) for i in range(len(scenes))]

    def r_coverage(q):
        qv = q["query_vi"]
        qt = [t.casefold() for t in _TOKEN_RE.findall(qv)]
        out = []
        for i, s in enumerate(scenes):
            cov = compute_coverage(qv, s.text, idf)
            out.append(bm25.score(qt, i) + 2.0 * cov.group + 1.0 * cov.idf_weighted)
        return out

    # IDF tren khong gian token da chuan hoa (bo dau, bo hu tu)
    n_doc = len(scenes)
    df_norm: Counter[str] = Counter()
    for toks in scene_norm:
        df_norm.update(set(toks))
    idf_norm = {
        t: math.log(1 + (n_doc - c + 0.5) / (c + 0.5)) for t, c in df_norm.items()
    }
    default_idf = math.log(1 + (n_doc + 0.5) / 0.5)
    idf_norm = defaultdict(lambda: default_idf, idf_norm)

    def make_meteor(alpha, gamma, fuse_bm25=0.0, use_idf=False):
        def r(q):
            qn = _norm_tokens(q["query_vi"])
            qt = [t.casefold() for t in _TOKEN_RE.findall(q["query_vi"])]
            w = idf_norm if use_idf else None
            out = []
            for i in range(len(scenes)):
                sc, *_ = meteor(qn, scene_norm[i], alpha=alpha, gamma=gamma, weights=w)
                if fuse_bm25:
                    sc = sc + fuse_bm25 * bm25.score(qt, i)
                out.append(sc)
            return out

        return r

    configs = [
        ("BM25 (baseline)", r_bm25),
        ("BM25 + coverage (hien tai)", r_coverage),
        ("METEOR full a=.9 g=.5", make_meteor(0.9, 0.5)),
        ("METEOR no-penalty a=.9", make_meteor(0.9, 0.0)),
        ("METEOR recall-thuan a=1", make_meteor(1.0, 0.0)),
        ("METEOR a=.5 g=0 (can bang)", make_meteor(0.5, 0.0)),
        ("METEOR-IDF a=.9 g=.5", make_meteor(0.9, 0.5, use_idf=True)),
        ("METEOR-IDF a=.9 g=0", make_meteor(0.9, 0.0, use_idf=True)),
        ("METEOR-IDF a=.5 g=.5", make_meteor(0.5, 0.5, use_idf=True)),
        ("METEOR-IDF + 0.3*BM25", make_meteor(0.9, 0.5, fuse_bm25=0.3, use_idf=True)),
    ]

    # Tien de cua idea A (METEOR lam reward neo query cho rewriter): chi dang
    # phat drift NEU expansion that su gay drift. Do truc tiep.
    try:
        from online.services.query_expansion import expand_query

        def r_expanded(q):
            qt = [t.casefold() for t in _TOKEN_RE.findall(expand_query(q["query_vi"]))]
            return [bm25.score(qt, i) for i in range(len(scenes))]

        configs.append(("BM25 + query_expansion", r_expanded))
    except Exception as exc:  # pragma: no cover
        print(f"(bo qua query_expansion: {exc})")

    rows = [evaluate(n, r, queries, scenes) for n, r in configs]
    hdr = f"{'config':<30}{'MRR':>7}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'vR@1':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for row in rows:
        line = f"{row['name']:<30}{row['mrr']:>7.3f}"
        line += "".join(f"{row['R@' + str(k)]:>8.3f}" for k in KS)
        line += f"{row['vR@1']:>8.3f}"
        print(line)

    # chan doan: penalty co phan biet gold khoi non-gold khong?
    print("\n--- chan doan fragmentation penalty (idea B: METEOR lam soft label) ---")
    gold_pen, neg_pen, gold_sc, neg_sc = [], [], [], []
    for q in queries:
        qn = _norm_tokens(q["query_vi"])
        for i, s in enumerate(scenes):
            sc, p, r, pen = meteor(qn, scene_norm[i], alpha=0.9, gamma=0.5)
            (gold_pen if is_hit(s, q) else neg_pen).append(pen)
            (gold_sc if is_hit(s, q) else neg_sc).append(sc)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(
        f"penalty  gold={mean(gold_pen):.4f}  non-gold={mean(neg_pen):.4f}  "
        f"(n_gold={len(gold_pen)}, n_neg={len(neg_pen)})"
    )
    print(
        f"score    gold={mean(gold_sc):.4f}  non-gold={mean(neg_sc):.4f}  "
        f"ti so={mean(gold_sc) / max(mean(neg_sc), 1e-9):.2f}x"
    )

    # Phep thu quyet dinh cho idea B (METEOR lam soft label cho reranker):
    # muon distill duoc, METEOR phai la teacher TOT HON tin hieu san co.
    # Trung binh cao hon chua du -- do ROC-AUC tren toan bo cap (query, scene).
    print("\n--- ROC-AUC: diem nao du bao relevance tot hon? (idea B) ---")
    from sklearn.metrics import roc_auc_score

    labels: list[int] = []
    cols: dict[str, list[float]] = defaultdict(list)
    for q in queries:
        qn = _norm_tokens(q["query_vi"])
        qt = [t.casefold() for t in _TOKEN_RE.findall(q["query_vi"])]
        qv = q["query_vi"]
        for i, s in enumerate(scenes):
            labels.append(1 if is_hit(s, q) else 0)
            cols["BM25"].append(bm25.score(qt, i))
            cov = compute_coverage(qv, s.text, idf)
            cols["BM25+coverage"].append(
                bm25.score(qt, i) + 2.0 * cov.group + 1.0 * cov.idf_weighted
            )
            cols["METEOR a=.9 g=.5"].append(
                meteor(qn, scene_norm[i], alpha=0.9, gamma=0.5)[0]
            )
            cols["METEOR-IDF a=.5 g=.5"].append(
                meteor(qn, scene_norm[i], alpha=0.5, gamma=0.5, weights=idf_norm)[0]
            )
    for name, vals in cols.items():
        print(f"  {name:<24} AUC = {roc_auc_score(labels, vals):.4f}")


if __name__ == "__main__":
    main()
