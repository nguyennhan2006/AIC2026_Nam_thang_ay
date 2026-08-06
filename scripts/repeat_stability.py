"""REPEAT-STABILITY-01 — cùng cấu hình, chạy N lần, xem số có đứng yên không.

Vì sao cần: FPT text rerank và FPT QA LLM nằm TRONG đường xếp hạng, mà chúng
là dịch vụ ngoài. Một lần đo `evidence_recall = 0.750` trong khi bốn lần khác
cùng cấu hình đều cho 0.833 — chưa giải thích được. Không biết biên độ dao
động thì không đọc nổi chênh lệch 1 query, mà bộ gold này 1 query đã là
0.083–0.125.

Tách BẮT BUỘC hai chế độ, không được trộn:

``local``
    `AIC_FPT_ENABLED=false`. Không gọi mạng. Dao động ở đây là lỗi tất định
    của chính hệ thống (thứ tự lặp, cold-start, tie-break) và phải bằng 0.

``fpt``
    Cấu hình thật. Dao động ở đây trừ đi dao động của `local` chính là phần
    **do nhà cung cấp gây ra**, không phải do thuật toán.

Gộp hai thứ lại rồi báo cáo một con số "độ lệch chuẩn" là cách chắc chắn để
quy nhầm nguyên nhân.

Chạy::

    python -m scripts.repeat_stability --tasks AVS --repeats 5 \\
        --variant hard_gate:AIC_AVS_GRADE_MODE=hard_gate \\
        --variant soft:AIC_AVS_GRADE_MODE=soft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

# Chỉ theo dõi các metric quyết định giữ/bỏ. Thêm bừa chỉ làm bảng khó đọc.
TRACKED = (
    "nDCG@100", "P@100", "event_coverage", "zero_result_rate",
    "MRR", "R@1", "answer_accuracy", "joint_top1", "evidence_recall",
    "mean_r_score", "frame_selection_accuracy", "top1_pairwise_accuracy",
)


def fingerprint(payload: dict) -> str:
    """Dấu vân tay của THỨ HẠNG, không phải của điểm số.

    Hai lần chạy có thể cho cùng metric tổng mà thứ tự bên trong đã khác —
    trường hợp đó vẫn là bất ổn định và phải nhìn thấy được.
    """

    rows = []
    for record in payload.get("per_query", []):
        rows.append(
            (
                record.get("query_id"),
                record.get("first_frame_hit_rank"),
                record.get("evidence_rank"),
                tuple(record.get("predicted_frames") or ()),
                tuple(record.get("predicted_answers") or ()),
            )
        )
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def run_once(args: argparse.Namespace, env_overrides: dict[str, str]) -> dict:
    out = Path(tempfile.mkdtemp()) / "run.json"
    env = os.environ | env_overrides | {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    command = [
        sys.executable, "-m", "scripts.eval_tasks",
        "--pipeline", "container",
        "--gold", str(args.gold),
        "--metadata", str(args.metadata),
        "--max-per-video", "0",
        "--tasks", args.tasks,
        "--json-out", str(out),
        *args.extra_arg,
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if not out.exists():
        raise SystemExit(
            f"eval_tasks không tạo được {out}\n"
            f"--- stderr ---\n{completed.stderr[-2000:]}"
        )
    return json.loads(out.read_text(encoding="utf-8"))


def branch_health(payloads: list[dict]) -> dict:
    """Nhánh nào KHÔNG chạy được, và ở bao nhiêu lượt truy vấn.

    Tách khỏi phần metric vì nó trả lời câu hỏi khác: metric dao động có thể do
    thuật toán, mà cũng có thể chỉ vì một nhánh thỉnh thoảng timeout. Gộp hai
    thứ lại là quy nhầm nguyên nhân.
    """

    degraded: dict[str, int] = {}
    total = 0
    for payload in payloads:
        for record in payload.get("per_query", []):
            total += 1
            for name in record.get("degraded_branches", []) or []:
                degraded[name] = degraded.get(name, 0) + 1
    return {"query_runs": total, "degraded_branch_hits": dict(sorted(degraded.items()))}


def query_deltas(payloads: list[dict]) -> dict[str, list]:
    """Truy vấn nào ĐỔI HẠNG giữa các lần chạy, kèm dải giá trị.

    Báo cáo mean/sd cho biết CÓ dao động; cái này cho biết dao động nằm ở ĐÂU.
    Với bộ gold 8–12 truy vấn thì biết đúng query nào bất ổn đáng giá hơn nhiều
    so với một con số độ lệch chuẩn.
    """

    fields = ("first_frame_hit_rank", "evidence_rank", "r_score", "ndcg")
    seen: dict[str, dict[str, set]] = {}
    for payload in payloads:
        for record in payload.get("per_query", []):
            bucket = seen.setdefault(record["query_id"], {})
            for field in fields:
                if field in record:
                    bucket.setdefault(field, set()).add(json.dumps(record[field]))
    unstable: dict[str, list] = {}
    for query_id, bucket in sorted(seen.items()):
        changed = {
            field: sorted(json.loads(v) if v != "null" else None for v in values)
            for field, values in bucket.items()
            if len(values) > 1
        }
        if changed:
            unstable[query_id] = changed
    return unstable


def summarize(payloads: list[dict]) -> dict:
    """mean/min/max/stdev cho từng metric + tập dấu vân tay thứ hạng."""

    flat: list[dict[str, float]] = []
    for payload in payloads:
        row: dict[str, float] = {}
        for task_block in payload.values():
            if isinstance(task_block, dict):
                for key, value in task_block.items():
                    if key in TRACKED and isinstance(value, (int, float)):
                        row[key] = float(value)
        flat.append(row)

    stats: dict[str, dict] = {}
    for key in sorted({name for row in flat for name in row}):
        values = [row[key] for row in flat if key in row]
        stats[key] = {
            "mean": round(statistics.fmean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        }
    return {
        "runs": len(payloads),
        "metrics": stats,
        "ranking_fingerprints": sorted({fingerprint(p) for p in payloads}),
        "branch_health": branch_health(payloads),
        "unstable_queries": query_deltas(payloads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo độ ổn định khi lặp lại cùng cấu hình")
    parser.add_argument("--gold", type=Path,
                        default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_l21_enriched/scenes.jsonl"))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variant", action="append", default=[],
                        help="TÊN:BIẾN=GIÁ_TRỊ[,BIẾN=GIÁ_TRỊ...]")
    parser.add_argument("--modes", default="local,fpt",
                        help="local = tắt FPT (phải tất định); fpt = cấu hình thật")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Tham so truyen thang cho eval_tasks, lap lai duoc")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/evaluation/stability/report.json"))
    args = parser.parse_args()

    variants: dict[str, dict[str, str]] = {}
    for spec in args.variant or ["default:"]:
        name, _, assignments = spec.partition(":")
        overrides = {}
        for pair in assignments.split(","):
            if "=" in pair:
                key, _, value = pair.partition("=")
                overrides[key.strip()] = value.strip()
        variants[name] = overrides

    report: dict = {"repeats": args.repeats, "tasks": args.tasks, "results": {}}
    for mode in args.modes.split(","):
        mode = mode.strip()
        # `local` phải tắt MỌI thứ cần mạng, không chỉ `AIC_FPT_ENABLED`:
        # dịch query và mở rộng bằng LLM đều đòi có LLM, và container fail-fast
        # đúng như thiết kế nếu bật chúng mà không có provider.
        #
        # Lưu ý khi đọc: `local` KHÔNG đo cùng cấu hình với `fpt`. Nó đo độ tất
        # định của phần máy móc chạy trên máy — dùng để biết dao động thấy ở
        # `fpt` là do nhà cung cấp hay do chính hệ thống.
        base = (
            {
                "AIC_FPT_ENABLED": "false",
                "AIC_ENABLE_QUERY_TRANSLATION": "false",
                "AIC_ENABLE_LLM_EXPANSION": "false",
                "AIC_RERANK_VLM_ENABLED": "false",
            }
            if mode == "local"
            else {}
        )
        for name, overrides in variants.items():
            label = f"{mode}/{name}"
            print(f"--- {label} x{args.repeats} ---", flush=True)
            payloads = [run_once(args, base | overrides) for _ in range(args.repeats)]
            summary = summarize(payloads)
            report["results"][label] = summary
            unstable = [
                key for key, value in summary["metrics"].items() if value["stdev"] > 0
            ]
            health = summary["branch_health"]["degraded_branch_hits"]
            print(
                f"    dấu vân tay: {len(summary['ranking_fingerprints'])}"
                f" | metric dao động: {unstable or 'không có'}"
                f" | query bất ổn: {list(summary['unstable_queries']) or 'không có'}"
                f" | nhánh hỏng: {health or 'không có'}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nchi tiết -> {args.out}")


if __name__ == "__main__":
    main()
