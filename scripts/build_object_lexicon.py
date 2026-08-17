"""Dựng từ điển Việt→Anh cho nhãn vật thể Open Images, chạy MỘT LẦN offline.

VẤN ĐỀ ĐO ĐƯỢC (EDA ngày 2026-08-17, 96/120 truy vấn gold):

    ti le tu trong truy van xuat hien o CANH DUNG
      task     caption     asr    object
      KIS        67.2%   35.6%      0.0%
      VQA        47.8%   35.9%      0.0%
      AVS        76.5%   78.4%      0.0%

`object` bằng **0,0% ở mọi task, mọi truy vấn**. Không phải thiếu dữ liệu —
export có 696.738 phát hiện trên 514 nhãn. Nguyên nhân là nhãn Open Images viết
bằng tiếng Anh (`Bicycle`, `Human face`) còn truy vấn thi đấu bằng tiếng Việt,
nên `bm25_object` không bao giờ khớp một token nào. Kiểm chứng trực tiếp: câu
"xe đạp và người đi đường" cho 0 ứng viên, câu "Bicycle Person Street" cho 100.

`AIC_ENABLE_QUERY_TRANSLATION` KHÔNG cứu được: nó bọc text encoder của nhánh
dense, các nhánh BM25 vẫn nhận nguyên câu tiếng Việt.

VÌ SAO LÀ TỪ ĐIỂN, KHÔNG PHẢI PROMPT LÚC TRUY VẤN

Từ vựng nhãn là **đóng** — 514 chuỗi, biết trước, không đổi giữa các truy vấn.
Dịch chúng một lần rồi tra bảng thì đường truy vấn không tốn thêm lời gọi LLM
nào, không thêm độ trễ, không thêm phụ thuộc mạng, và kết quả tái lập được. Gọi
LLM mỗi truy vấn để sinh lại cùng một ánh xạ là trả phí lặp cho một hàm hằng.

Đây cũng là lý do `EXPAND_QUERY` (sinh từ đồng nghĩa tự do mỗi truy vấn) đã đo
được là GÂY HẠI — KIS MRR 0.547 -> 0.512, `docs/20` § FPT-WIRE-01: đầu ra không
ràng buộc thì trôi nghĩa. Ở đây đầu ra bị ép về đúng 514 nhãn có thật.

    python -m scripts.build_object_lexicon --export storage/exports_competition
    python -m scripts.build_object_lexicon --labels storage/state/object_labels.json --dry-run
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import unicodedata

DEFAULT_OUT = Path("online/resources/object_lexicon_vi.json")

SYSTEM = (
    "Ban la tu dien doi chieu Anh-Viet cho nhan phat hien vat the Open Images, "
    "dung cho tim kiem video tin tuc tieng Viet."
)

INSTRUCTION = """Voi MOI nhan tieng Anh duoi day, liet ke cac tu/cum tu TIENG VIET
ma nguoi Viet se dung khi mo ta canh quay chua vat the do.

Quy tac:
- Chi tu thong dung trong loi noi hang ngay va ban tin. Khong dinh nghia, khong giai thich.
- 1 den 4 cum tu moi nhan. Uu tien cum ngan (1-3 tu).
- Khong them dau cau, khong danh so.
- Neu nhan qua chuyen nganh va khong co tu tieng Viet thong dung, tra mang rong [].

Tra ve DUNG mot object JSON, khoa la nhan tieng Anh nguyen van, gia tri la mang chuoi.
Khong bao boc trong markdown."""


def strip_accents(text: str) -> str:
    """Bỏ dấu + hạ chữ. Cùng phép chuẩn hoá với `online/services/lexical_coverage.py`
    để bảng tra khớp được cả khi người dùng gõ không dấu."""

    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d")


def labels_from_export(export: Path) -> list[str]:
    vocab: collections.Counter[str] = collections.Counter()
    path = export / "keyframes.jsonl" if export.is_dir() else export
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"objects"' not in line:
                continue
            for item in json.loads(line).get("objects") or []:
                label = str(item.get("label") or "").strip()
                if label:
                    vocab[label] += 1
    return sorted(vocab)


def _parse_json_object(text: str) -> dict[str, list[str]]:
    """Bóc object JSON khỏi câu trả lời, kể cả khi model bọc trong ```json."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        cleaned = cleaned.split("\n", 1)[1] if cleaned.lower().startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"khong tim thay object JSON trong: {text[:200]!r}")
    raw = json.loads(cleaned[start : end + 1])
    return {
        str(key): [str(v).strip() for v in value if str(v).strip()]
        for key, value in raw.items()
        if isinstance(value, list)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--export", help="thu muc export (doc nhan tu keyframes.jsonl)")
    parser.add_argument("--labels", help="file JSON chua san danh sach nhan")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch", type=int, default=40, help="so nhan moi lan goi LLM")
    parser.add_argument("--dry-run", action="store_true", help="chi in nhan, khong goi LLM")
    arguments = parser.parse_args()

    if arguments.labels:
        labels = json.loads(Path(arguments.labels).read_text(encoding="utf-8"))
    elif arguments.export:
        labels = labels_from_export(Path(arguments.export))
    else:
        raise SystemExit("can --export hoac --labels")
    print(f"  {len(labels)} nhan")

    if arguments.dry_run:
        print("  " + ", ".join(labels[:30]) + " …")
        return

    from online.config import Settings
    from online.adapters.fpt_client import FptClient

    settings = Settings.from_env()
    client = FptClient.from_settings(settings)
    model = settings.fpt_fast_llm_model or settings.fpt_llm_model
    print(f"  model: {model}")

    lexicon: dict[str, list[str]] = {}
    tokens_in = tokens_out = 0
    for start in range(0, len(labels), arguments.batch):
        chunk = labels[start : start + arguments.batch]
        result = client.chat_completion(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": INSTRUCTION + "\n\n" + json.dumps(chunk, ensure_ascii=False)},
            ],
            model=model,
            temperature=0.0,
            max_tokens=3000,
        )
        try:
            part = _parse_json_object(result.text or "")
        except ValueError as exc:
            print(f"    LOI o lo {start}: {exc}")
            continue
        # Chỉ nhận khoá CÓ THẬT trong danh sách: model đôi khi tự bịa thêm nhãn,
        # và một khoá lạ sẽ nằm im trong từ điển mà không ai đối chiếu.
        known = set(chunk)
        lexicon.update({k: v for k, v in part.items() if k in known})
        tokens_in += result.usage.input_tokens
        tokens_out += result.usage.output_tokens
        print(
            f"    {min(start + arguments.batch, len(labels))}/{len(labels)}  "
            f"da co {len(lexicon)} nhan  (token {tokens_in}+{tokens_out})",
            flush=True,
        )

    missing = [label for label in labels if label not in lexicon]
    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "model": model,
                    "label_count": len(labels),
                    "translated": len(lexicon),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
                "labels": lexicon,
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    empties = sum(1 for v in lexicon.values() if not v)
    print()
    print(f"  dich duoc      {len(lexicon)}/{len(labels)}")
    print(f"  tra mang rong  {empties}")
    print(f"  thieu han      {len(missing)}  {missing[:8]}")
    print(f"  token          {tokens_in} vao + {tokens_out} ra")
    print(f"  -> {out_path}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
