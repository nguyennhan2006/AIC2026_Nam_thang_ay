"""Chấm bộ tách keyword TRỰC TIẾP với `sparse_terms` của gold.

Vì sao tách riêng khỏi `eval_tasks.py`: đo qua retrieval thì một thay đổi ở bộ
tách phải đi qua BM25 → fusion → rerank → dedup mới thấy được ở R@1, và lúc đó
không biết cải thiện đến từ đâu. Ở đây so thẳng đầu ra của bộ tách với keyword
người gán, nên vòng lặp sửa-đo tính bằng giây và quy trách nhiệm được.

12 truy vấn KIS trong gold có `sparse_terms`, dạng `a | b | c`. Chúng trộn
tiếng Việt và tiếng Anh (`"cảnh báo sạt lở nguy hiểm | riverbank erosion
warning | Cần Thơ"`) vì được gán khi caption còn tiếng Anh. Caption nay là
tiếng Việt, nên **chỉ chấm phần tiếng Việt** — đòi bộ tách sinh ra
`"riverbank erosion warning"` từ một câu tiếng Việt là đòi nó dịch, việc của
`FptQueryTranslator` chứ không phải của nó.

Ba chỉ số, tách bạch vì chúng hỏng theo ba cách khác nhau:

``token_recall``
    Bao nhiêu token của keyword gold xuất hiện trong đầu ra. Thấp = bỏ sót.
``token_precision``
    Bao nhiêu token đầu ra có trong keyword gold. Thấp = lôi thêm rác.
``phrase_hit``
    Tỉ lệ cụm gold được phủ >= 60% token. Đây là chỉ số sát BM25 nhất, vì
    BM25 khớp theo token chứ không theo cụm nguyên vẹn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from online.adapters.ocr_fuzzy import normalize_vi
from online.services.keyword_extraction import STOPWORDS, CorpusIdf, extract_keywords

# Chữ cái tiếng Việt có dấu — dùng để nhận diện cụm gold nào là tiếng Việt.
_VI_MARKS = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
                       re.IGNORECASE)


def is_vietnamese(phrase: str) -> bool:
    """Cụm có dấu tiếng Việt, hoặc toàn từ ngắn kiểu tiếng Việt không dấu.

    Không hoàn hảo — `"Cần Thơ"` là địa danh và cũng là tiếng Việt, còn
    `"ambulance"` thì không. Đủ để tách nhóm cần chấm.
    """

    if _VI_MARKS.search(phrase):
        return True
    words = phrase.split()
    return bool(words) and all(len(w) <= 5 for w in words)


def tokens_of(text: str) -> set[str]:
    return {
        token
        for token in normalize_vi(text).split()
        if token and token not in STOPWORDS and len(token) > 1
    }


def score_query(query, gold_terms, max_terms, idf=None, mode="content") -> dict:
    predicted = extract_keywords(query, max_terms=max_terms, idf=idf, mode=mode)
    predicted_tokens = tokens_of(" ".join(predicted))

    vi_terms = [term for term in gold_terms if is_vietnamese(term)]
    gold_tokens = tokens_of(" ".join(vi_terms))

    hit = sum(1 for token in gold_tokens if token in predicted_tokens)
    phrase_hits = 0
    for term in vi_terms:
        term_tokens = tokens_of(term)
        if not term_tokens:
            continue
        covered = sum(1 for token in term_tokens if token in predicted_tokens)
        if covered / len(term_tokens) >= 0.6:
            phrase_hits += 1

    return {
        "predicted": predicted,
        "gold_vi": vi_terms,
        "token_recall": hit / len(gold_tokens) if gold_tokens else 0.0,
        "token_precision": (
            sum(1 for token in predicted_tokens if token in gold_tokens) / len(predicted_tokens)
            if predicted_tokens else 0.0
        ),
        "phrase_hit": phrase_hits / len(vi_terms) if vi_terms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chấm bộ tách keyword với sparse_terms")
    parser.add_argument("--gold", type=Path,
                        default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl"))
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument("--metadata", type=Path, default=None,
                        help="scenes.jsonl de tinh IDF corpus; bo trong = khong dung IDF")
    parser.add_argument("--mode", default="content", choices=("content","head","idf"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    idf = None
    if args.metadata:
        import asyncio
        from online.adapters.json_metadata import JsonlSceneRepository

        repo = asyncio.run(JsonlSceneRepository.load(args.metadata))
        idf = CorpusIdf.from_scenes(asyncio.run(repo.all()))
        print(f"IDF tu {idf.total} scene, {len(idf.counts)} token")

    rows = [
        json.loads(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = []
    for row in rows:
        raw = row.get("sparse_terms")
        if not raw:
            continue
        query = row.get("query_vi") or row.get("question_vi") or ""
        gold_terms = [part.strip() for part in str(raw).split("|") if part.strip()]
        result = score_query(query, gold_terms, args.max_terms, idf, args.mode)
        result["query_id"] = row["query_id"]
        scored.append(result)

    if not scored:
        raise SystemExit("không có truy vấn nào có `sparse_terms`")

    if args.verbose:
        for item in scored:
            print(f"\n[{item['query_id']}] recall={item['token_recall']:.2f} "
                  f"prec={item['token_precision']:.2f} phrase={item['phrase_hit']:.2f}")
            print(f"   gold VI : {item['gold_vi']}")
            print(f"   tách ra : {item['predicted']}")

    n = len(scored)
    print(f"\n=== {n} truy vấn có sparse_terms ===")
    for key in ("token_recall", "token_precision", "phrase_hit"):
        print(f"  {key:18s} {sum(item[key] for item in scored) / n:.3f}")


if __name__ == "__main__":
    main()
