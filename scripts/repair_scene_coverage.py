"""SCENE-COVERAGE-01 — sửa: gộp scene không keyframe vào láng giềng.

Chẩn đoán (docs/20_EXPERIMENT_LOG.md § SCENE-COVERAGE-01): 119/336 scene bị
quarantine vì không có keyframe nào — keyframe trích theo stride cố định ~123
frame nên scene ngắn hơn stride rơi trọn qua lưới. Kết quả: coverage 78.6%,
84 gap, 5–7 bước TRAKE có frame gold không thuộc scene nào.

**Vì sao gộp thay vì trích lại keyframe:** keyframe do BTC cung cấp đã chứa
câu trả lời — vấn đề không phải thiếu ảnh mà là scene chứa mốc gold bị xoá.
Gộp khoảng trống vào scene láng giềng ĐANG CÓ keyframe khôi phục được cả hai
thứ, không tốn thêm lần trích hay lần caption nào.

Có một tác dụng phụ CÓ LỢI và đúng luật: cửa sổ chấm của TRAKE suy từ độ dài
scene (`clamp(duration * 0.5, 2, 7)`). Frame nằm trong gap hiện không có
scene nên rơi về fallback tối thiểu ±1.0s — chính những mốc cần cửa sổ rộng
nhất lại bị chấm ngặt nhất. Sau khi gộp, chúng thuộc một scene dài hơn nên
nhận cửa sổ đúng theo quy tắc.

Quy tắc sở hữu (tất định, không phụ thuộc thứ tự duyệt):

    Mỗi gap thuộc về láng giềng có KEYFRAME GẦN TÂM GAP NHẤT.
    Hoà -> ưu tiên scene phía trước (chỉ số nhỏ hơn).

Chọn theo khoảng cách keyframe chứ không phải theo bên nào cũng được, vì
keyframe của scene nhận gap chính là ảnh sẽ đại diện cho nội dung trong gap.

Script này KHÔNG sửa file gốc. Nó ghi ra một export mới để so sánh A/B.

Chạy::

    python -m scripts.repair_scene_coverage \\
        --metadata storage/exports_l21/scenes.jsonl \\
        --out storage/exports_l21_repaired
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_scenes(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def keyframe_positions(scene: dict) -> list[int]:
    return [int(item["frame_idx"]) for item in scene.get("keyframes", []) if "frame_idx" in item]


def repair(scenes: list[dict], frame_count: int | None) -> tuple[list[dict], list[dict]]:
    """Trả `(scene đã sửa, nhật ký từng lần gộp)`."""

    ordered = sorted(scenes, key=lambda item: item["start_frame"])
    journal: list[dict] = []
    if not ordered:
        return ordered, journal

    # Đầu video.
    if ordered[0]["start_frame"] > 0:
        journal.append({
            "kind": "missing_head", "length": ordered[0]["start_frame"],
            "assigned_to": ordered[0]["scene_id"],
        })
        ordered[0]["start_frame"] = 0
        ordered[0]["start_sec"] = 0.0

    # Khoảng trống giữa các scene.
    for previous, following in zip(ordered, ordered[1:], strict=False):
        gap_start = previous["end_frame_exclusive"]
        gap_end = following["start_frame"]
        if gap_end <= gap_start:
            continue
        centre = (gap_start + gap_end) / 2
        before = keyframe_positions(previous)
        after = keyframe_positions(following)
        distance_before = min((abs(k - centre) for k in before), default=float("inf"))
        distance_after = min((abs(k - centre) for k in after), default=float("inf"))
        # Hoà -> ưu tiên scene phía trước (`<=`), giữ tính tất định.
        to_previous = distance_before <= distance_after
        if to_previous:
            previous["end_frame_exclusive"] = gap_end
            previous["end_sec"] = following["start_sec"]
        else:
            following["start_frame"] = gap_start
            following["start_sec"] = previous["end_sec"]
        journal.append({
            "kind": "gap", "start_frame": gap_start, "end_frame_exclusive": gap_end,
            "length": gap_end - gap_start,
            "assigned_to": previous["scene_id"] if to_previous else following["scene_id"],
            "keyframe_distance_sec": round(min(distance_before, distance_after), 1),
        })

    # Đuôi video.
    if frame_count is not None and ordered[-1]["end_frame_exclusive"] < frame_count:
        journal.append({
            "kind": "missing_tail",
            "length": frame_count - ordered[-1]["end_frame_exclusive"],
            "assigned_to": ordered[-1]["scene_id"],
        })
        ordered[-1]["end_frame_exclusive"] = frame_count

    return ordered, journal


def main() -> None:
    parser = argparse.ArgumentParser(description="Gộp gap scene vào láng giềng có keyframe")
    parser.add_argument("--metadata", type=Path, default=Path("storage/exports_l21/scenes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("storage/exports_l21_repaired"))
    args = parser.parse_args()

    source_dir = args.metadata.parent
    scenes = load_scenes(args.metadata)

    frame_counts: dict[str, int] = {}
    videos_path = source_dir / "videos.jsonl"
    if videos_path.exists():
        for line in videos_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                frame_counts[record["video_id"]] = int(record["frame_count"])

    by_video: dict[str, list[dict]] = {}
    for scene in scenes:
        by_video.setdefault(scene["video_id"], []).append(scene)

    repaired: list[dict] = []
    journal: list[dict] = []
    for video_id, items in sorted(by_video.items()):
        fixed, log = repair(items, frame_counts.get(video_id))
        repaired.extend(fixed)
        for entry in log:
            entry["video_id"] = video_id
        journal.extend(log)

    args.out.mkdir(parents=True, exist_ok=True)
    # Copy nguyên các file khác của export để thư mục mới dùng được ngay.
    for sibling in source_dir.glob("*.jsonl"):
        if sibling.name != args.metadata.name:
            shutil.copy2(sibling, args.out / sibling.name)
    for sibling in source_dir.glob("*.json"):
        shutil.copy2(sibling, args.out / sibling.name)

    (args.out / args.metadata.name).write_text(
        "\n".join(json.dumps(scene, ensure_ascii=False) for scene in repaired) + "\n",
        encoding="utf-8",
    )
    (args.out / "coverage_repair_journal.json").write_text(
        json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    total = sum(entry["length"] for entry in journal)
    print(f"gộp {len(journal)} khoảng trống, {total} frame")
    print(f"  về scene trước: {sum(1 for e in journal if e['kind'] == 'gap' and e.get('keyframe_distance_sec') is not None)} gap")
    print(f"-> {args.out}")
    print("Kiểm tra lại bằng: python -m scripts.check_scene_coverage --metadata "
          f"{args.out / args.metadata.name}")


if __name__ == "__main__":
    main()
