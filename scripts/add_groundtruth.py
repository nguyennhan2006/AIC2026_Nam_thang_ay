"""Công cụ nhỏ hỗ trợ mở rộng ground-truth cho scripts/eval_kis.py.

KHÔNG tự sinh/đoán dữ liệu — chỉ hỏi/nhận thông tin người dùng đã tự xem video xác
nhận, rồi append đúng schema `GroundTruthItem` (xem docstring `scripts/eval_kis.py`)
vào file JSONL. Ground-truth hiện chỉ có 14 query (`examples/kis_groundtruth*.jsonl`)
— xem docs/13_PRODUCTION_READINESS_INFO.md mục 4 và docs/15_RESEARCH_AGENDA.md.

CÁCH DÙNG (non-interactive, dùng trong script/CI):
    python -m scripts.add_groundtruth --query 'Người áo vàng đang bơi' \
        --video-id L01_V001 --start-sec 12.5 --end-sec 18.0 \
        --scene-ids L01_V001_S0002

CÁCH DÙNG (interactive, hỏi lần lượt từng field):
    python -m scripts.add_groundtruth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VIDEO_ID_RE = re.compile(r"^L\d{2}_V\d{3}$")
DEFAULT_PATH = Path("examples/kis_groundtruth.jsonl")


def _next_query_id(existing: list[dict]) -> str:
    numbers = []
    for item in existing:
        match = re.fullmatch(r"q(\d+)", str(item.get("query_id", "")))
        if match:
            numbers.append(int(match.group(1)))
    return f"q{max(numbers, default=0) + 1}"


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _prompt(label: str, *, required: bool = True, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print("  (bắt buộc, nhập lại)")


def build_entry(
    *,
    query: str,
    video_id: str,
    start_sec: float | None,
    end_sec: float | None,
    scene_ids: list[str],
    query_id: str,
) -> dict:
    if not VIDEO_ID_RE.match(video_id):
        raise ValueError(f"video_id phải khớp ^L\\d{{2}}_V\\d{{3}}$, nhận: {video_id!r}")
    if (start_sec is None) != (end_sec is None):
        raise ValueError("start_sec/end_sec phải cùng có hoặc cùng không có")
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise ValueError("end_sec phải lớn hơn start_sec")
    entry: dict = {"query_id": query_id, "query": query, "video_id": video_id}
    if start_sec is not None:
        entry["start_sec"] = start_sec
        entry["end_sec"] = end_sec
    if scene_ids:
        entry["scene_ids"] = scene_ids
    return entry


def append_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--query-id", default=None, help="mặc định tự tăng qN+1")
    parser.add_argument("--query", default=None)
    parser.add_argument("--video-id", default=None, help="dạng L01_V001")
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    args = parser.parse_args()

    existing = _load_existing(args.path)
    interactive = args.query is None and args.video_id is None

    if interactive:
        print(f"Đang thêm vào {args.path} ({len(existing)} query hiện có). Ctrl+C để huỷ.")
        query = _prompt("Query (nội dung truy vấn)")
        video_id = _prompt("video_id (vd L01_V001)")
        has_interval = _prompt("Có interval thời gian? (y/n)", default="y").lower().startswith("y")
        start_sec = end_sec = None
        if has_interval:
            start_sec = float(_prompt("start_sec"))
            end_sec = float(_prompt("end_sec"))
        scene_ids_raw = _prompt("scene_ids (cách nhau bởi dấu phẩy, để trống nếu không có)", required=False)
        scene_ids = [item.strip() for item in scene_ids_raw.split(",") if item.strip()]
        query_id = _prompt("query_id", default=_next_query_id(existing))
    else:
        if not args.query or not args.video_id:
            raise SystemExit("--query và --video-id là bắt buộc ở chế độ non-interactive")
        query = args.query
        video_id = args.video_id
        start_sec = args.start_sec
        end_sec = args.end_sec
        scene_ids = args.scene_ids or []
        query_id = args.query_id or _next_query_id(existing)

    entry = build_entry(
        query=query, video_id=video_id, start_sec=start_sec, end_sec=end_sec, scene_ids=scene_ids, query_id=query_id
    )
    append_entry(args.path, entry)
    print(f"Đã thêm {query_id} vào {args.path} (tổng {len(existing) + 1} query).")


if __name__ == "__main__":
    main()
