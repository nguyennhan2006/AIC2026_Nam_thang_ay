"""Kiểm tra file gold query TRƯỚC khi chạy eval.

Vì sao cần: `scripts/eval_tasks.py` đọc gold rồi mới dựng container và chạy 40
truy vấn. Một trường sai tên hay một `frame_idx` ngoài phạm vi chỉ lộ ra sau
khi đã chạy xong phần tốn thời gian nhất — hoặc tệ hơn, không lộ ra mà lặng lẽ
làm mọi metric của truy vấn đó bằng 0.

Kiểm ba nhóm, tách bạch vì chúng cần ba cách sửa khác nhau:

1. **Cấu trúc** — thiếu trường, sai kiểu, task lạ.
2. **Tham chiếu** — `target_video` có trong export không, `frame_idx` có nằm
   trong phạm vi video không, có keyframe nào gần đó không.
3. **Đo được hay không** — bao nhiêu truy vấn mỗi task, có đủ để phân biệt cải
   thiện thật với một query may mắn không.

Chạy::

    python -m scripts.validate_gold --gold examples/queries.jsonl \\
        --metadata storage/exports_multivideo/scenes.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
from pathlib import Path
import sys

TASKS = {"TEXTUAL_KIS", "KIS", "VQA", "QA", "TRAKE", "AVS"}

# Dưới ngưỡng này, một truy vấn đổi kết quả đã vượt quá phần lớn khác biệt giữa
# các cấu hình — tức không kết luận được gì. 12 truy vấn KIS cho 1/12 = 0.083.
MIN_QUERIES_FOR_SIGNAL = 20


def load_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append({"_line": number, **json.loads(line)})
        except json.JSONDecodeError as exc:
            raise SystemExit(f"dòng {number}: JSON hỏng — {exc}")
    return rows


def check_structure(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        line = row["_line"]
        query_id = row.get("query_id")
        if not query_id:
            problems.append(f"dòng {line}: thiếu `query_id`")
        elif query_id in seen_ids:
            problems.append(f"dòng {line}: `query_id` trùng: {query_id}")
        else:
            seen_ids.add(query_id)

        task = row.get("task")
        if task not in TASKS:
            problems.append(f"dòng {line}: `task` không hợp lệ: {task!r} (cần một trong {sorted(TASKS)})")
        # VQA dùng `question_vi` thay cho `query_vi`.
        if not (row.get("query_vi") or row.get("question_vi") or row.get("query")):
            problems.append(f"dòng {line}: thiếu `query_vi` (VQA dùng `question_vi`)")
        if not row.get("target_video"):
            problems.append(f"dòng {line}: thiếu `target_video`")

        if task == "TRAKE":
            events = row.get("events")
            if not isinstance(events, list) or len(events) < 2:
                problems.append(f"dòng {line}: TRAKE cần `events` với ít nhất 2 phần tử")
            else:
                for index, event in enumerate(events):
                    if not isinstance(event, dict):
                        problems.append(f"dòng {line}: events[{index}] không phải object")
                        continue
                    if "representative_frame" not in event:
                        problems.append(f"dòng {line}: events[{index}] thiếu `representative_frame`")
        elif task == "AVS":
            # AVS dùng `relevant_intervals` (kèm `relevance_grade` mỗi khoảng),
            # KHÔNG phải `target_intervals` như KIS/VQA. Hai task hai tên
            # trường khác nhau — bản đầu của validator này giả định chung một
            # tên và báo sai 8 lỗi trên file gold đang chạy tốt.
            if not row.get("relevant_intervals"):
                problems.append(f"dòng {line}: AVS cần `relevant_intervals`")
            else:
                for index, interval in enumerate(row["relevant_intervals"]):
                    if not isinstance(interval, dict):
                        problems.append(f"dòng {line}: relevant_intervals[{index}] không phải object")
                    elif "relevance_grade" not in interval:
                        problems.append(
                            f"dòng {line}: relevant_intervals[{index}] thiếu `relevance_grade`"
                        )
        elif task in ("KIS", "TEXTUAL_KIS", "VQA", "QA"):
            if not row.get("target_intervals"):
                problems.append(f"dòng {line}: {task} cần `target_intervals`")
            if task in ("VQA", "QA") and not row.get("accepted_answers"):
                problems.append(f"dòng {line}: {task} cần `accepted_answers`")
    return problems


def check_references(rows: list[dict], metadata: Path) -> list[str]:
    if not metadata.exists():
        return [f"không tìm thấy {metadata} — bỏ qua phần kiểm tham chiếu"]

    frames_by_video: dict[str, list[int]] = collections.defaultdict(list)
    bounds: dict[str, int] = {}
    keyframes_path = metadata.with_name("keyframes.jsonl")
    if keyframes_path.exists():
        for line in keyframes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                frames_by_video[row["video_id"]].append(row["frame_idx"])
    for frames in frames_by_video.values():
        frames.sort()

    videos_path = metadata.with_name("videos.jsonl")
    if videos_path.exists():
        for line in videos_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                bounds[row["video_id"]] = int(row.get("frame_count") or 0)

    problems: list[str] = []
    for row in rows:
        line, video = row["_line"], row.get("target_video")
        if video and video not in frames_by_video:
            problems.append(f"dòng {line}: `target_video` {video!r} không có trong export")
            continue

        wanted: list[int] = []
        for event in row.get("events") or []:
            if isinstance(event, dict) and "representative_frame" in event:
                wanted.append(int(event["representative_frame"]))
        for interval in (row.get("target_intervals") or []) + (row.get("relevant_intervals") or []):
            if isinstance(interval, dict):
                for key in ("start_frame", "end_frame"):
                    if key in interval:
                        wanted.append(int(interval[key]))

        limit = bounds.get(video or "", 0)
        frames = frames_by_video.get(video or "", [])
        for frame in wanted:
            if limit and not (0 <= frame < limit):
                problems.append(
                    f"dòng {line}: frame {frame} ngoài phạm vi {video} (0..{limit - 1})"
                )
                continue
            if frames:
                index = bisect.bisect_left(frames, frame)
                nearest = min(
                    (abs(frame - frames[j]) for j in (index - 1, index, index + 1)
                     if 0 <= j < len(frames)),
                    default=None,
                )
                # Cảnh báo, KHÔNG phải lỗi: gold có quyền trỏ vào frame không
                # được trích. Nhưng nếu keyframe gần nhất quá xa thì truy vấn đó
                # gần như chắc chắn không thể ăn điểm, và biết trước tốt hơn là
                # phát hiện sau khi đã tinh chỉnh nhầm.
                if nearest is not None and nearest > 150:
                    problems.append(
                        f"dòng {line}: CẢNH BÁO frame {frame} cách keyframe gần nhất "
                        f"{nearest} frame — truy vấn này khó ăn điểm"
                    )
    return problems


def report_power(rows: list[dict]) -> None:
    by_task = collections.Counter(row.get("task") for row in rows)
    by_video = collections.Counter(row.get("target_video") for row in rows)
    print("\n--- Số lượng ---")
    for task, count in sorted(by_task.items()):
        note = ""
        if count < MIN_QUERIES_FOR_SIGNAL:
            note = f"  <== 1 truy vấn = {1 / count:.3f}, chênh dưới 2 truy vấn KHÔNG đọc được"
        print(f"  {str(task):12s} {count:3d}{note}")
    print("\n--- Theo video ---")
    for video, count in sorted(by_video.items()):
        print(f"  {str(video):12s} {count:3d}")
    if len(by_video) < 2:
        print("\n  CẢNH BÁO: mọi truy vấn trỏ về cùng một video, nên không tách được")
        print("  holdout và không đo được khả năng tổng quát sang video chưa tinh chỉnh.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra file gold trước khi chạy eval")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_multivideo/scenes.jsonl"))
    args = parser.parse_args()

    rows = load_rows(args.gold)
    print(f"đọc {len(rows)} truy vấn từ {args.gold}")

    problems = check_structure(rows)
    problems += check_references(rows, args.metadata)
    errors = [p for p in problems if "CẢNH BÁO" not in p]
    warnings = [p for p in problems if "CẢNH BÁO" in p]

    if errors:
        print(f"\n--- {len(errors)} LỖI ---")
        for item in errors:
            print(f"  {item}")
    if warnings:
        print(f"\n--- {len(warnings)} cảnh báo ---")
        for item in warnings[:20]:
            print(f"  {item}")
        if len(warnings) > 20:
            print(f"  ... và {len(warnings) - 20} cảnh báo nữa")

    report_power(rows)

    if errors:
        print(f"\nKHÔNG dùng được: {len(errors)} lỗi cần sửa trước.")
        sys.exit(1)
    print("\nDùng được cho eval.")


if __name__ == "__main__":
    main()
