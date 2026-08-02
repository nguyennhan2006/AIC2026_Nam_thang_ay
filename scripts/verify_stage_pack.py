"""Kiểm tra một stage pack TRƯỚC khi assemble.

Chạy được ngay trên Kaggle ở cuối notebook, hoặc ở local sau khi tải zip về.
Bắt lỗi contract sớm rẻ hơn nhiều so với phát hiện lúc assemble cả corpus.

    python -m scripts.verify_stage_pack storage/packs/01_scene_detection --stage scene
    python -m scripts.verify_stage_pack storage/packs --all

Kiểm tra: `_SUCCESS.json` status, `model_info.json`, manifest tồn tại và
parse được, trường bắt buộc theo `contracts/stage_pack.schema.json`, và các
bất biến không diễn đạt được bằng JSON Schema (scene liền mạch, frame_idx
không trùng, `end_frame >= start_frame`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from offline.stagepack import STAGE_MANIFESTS, StagePack, StagePackError, discover_packs, open_pack

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "video": ("video_id", "source_path", "fps", "frame_count", "width", "height"),
    "scene": ("video_id", "scene_index", "start_frame", "end_frame"),
    "keyframe": ("video_id", "frame_idx", "image_path", "width", "height"),
    "asr": ("video_id", "start_sec", "end_sec", "text"),
    "embedding": ("video_id", "frame_idx"),
    "color": ("video_id", "frame_idx"),
    "ocr": ("video_id", "frame_idx", "instances"),
    "object": ("video_id", "frame_idx", "objects"),
    "caption": ("video_id", "frame_idx", "captions"),
}


def _check_rows(pack: StagePack) -> list[str]:
    problems: list[str] = []
    required = REQUIRED_FIELDS[pack.stage]
    seen_frames: dict[str, set[int]] = defaultdict(set)
    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for index, row in enumerate(pack.rows(), start=1):
        count += 1
        missing = [field for field in required if field not in row]
        if missing:
            problems.append(f"dòng {index}: thiếu trường {missing}")
            continue
        if pack.stage in {"keyframe", "embedding", "color", "ocr", "object", "caption"}:
            frame_idx = int(row["frame_idx"])
            if frame_idx in seen_frames[row["video_id"]]:
                problems.append(
                    f"dòng {index}: (video_id, frame_idx)=({row['video_id']}, {frame_idx}) "
                    "trùng — khóa join phải là duy nhất"
                )
            seen_frames[row["video_id"]].add(frame_idx)
        if pack.stage == "scene":
            if int(row["end_frame"]) < int(row["start_frame"]):
                problems.append(f"dòng {index}: end_frame < start_frame")
            scenes[row["video_id"]].append(row)
        if pack.stage == "asr" and float(row["end_sec"]) < float(row["start_sec"]):
            problems.append(f"dòng {index}: end_sec < start_sec")

    for video_id, rows in scenes.items():
        rows.sort(key=lambda item: int(item["start_frame"]))
        for previous, current in zip(rows, rows[1:]):
            if int(current["start_frame"]) <= int(previous["end_frame"]):
                problems.append(
                    f"{video_id}: scene chồng lấn tại frame {current['start_frame']}"
                )
            elif int(current["start_frame"]) != int(previous["end_frame"]) + 1:
                problems.append(
                    f"{video_id}: khoảng hở giữa frame {previous['end_frame']} và "
                    f"{current['start_frame']} — scene phải liền mạch"
                )
    if count == 0:
        problems.append("manifest rỗng")
    return problems


def verify(pack: StagePack) -> dict[str, Any]:
    problems = _check_rows(pack)
    return {
        "stage": pack.stage,
        "root": str(pack.root),
        "model": pack.model_info.get("model"),
        "ok": not problems,
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra stage pack theo contract")
    parser.add_argument("path", type=Path)
    parser.add_argument("--stage", choices=sorted(STAGE_MANIFESTS), default=None)
    parser.add_argument("--all", action="store_true", help="`path` là thư mục chứa nhiều pack")
    args = parser.parse_args()

    try:
        if args.all:
            packs = discover_packs(args.path)
        elif args.stage:
            packs = {args.stage: open_pack(args.path, args.stage)}
        else:
            parser.error("cần --stage hoặc --all")
            return
    except StagePackError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    reports = [verify(pack) for _stage, pack in sorted(packs.items())]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    failed = [item for item in reports if not item["ok"]]
    if failed:
        print(
            f"\nFAIL: {len(failed)}/{len(reports)} pack sai contract",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"\nOK: {len(reports)} pack hợp lệ")


if __name__ == "__main__":
    main()
