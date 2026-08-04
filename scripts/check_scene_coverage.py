"""SCENE-COVERAGE-01: scene có lát kín video không?

Vì sao cần (docs/20_EXPERIMENT_LOG.md § DENSE-TEXT-01): 5/35 bước TRAKE có
frame gold KHÔNG thuộc bất kỳ scene nào. Khi đó candidate tương ứng không tồn
tại trong corpus, nên mọi cải tiến caption/dense/BM25/reranker đều bất lực.

Quy ước interval của hệ thống là NỬA MỞ `[start_frame, end_frame_exclusive)`.
Lát kín nghĩa là:

    scene[0].start_frame            == 0
    scene[i].end_frame_exclusive    == scene[i+1].start_frame
    scene[-1].end_frame_exclusive   == frame_count

Script chỉ CHẨN ĐOÁN, không sửa. Không tự tạo scene phủ gap trước khi biết
gap phát sinh ở tầng nào (TransNet -> post-process -> merge -> export -> load).

Chạy::

    python -m scripts.check_scene_coverage --metadata storage/exports_l21/scenes.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from online.adapters.json_metadata import JsonlSceneRepository


def load_frame_counts(metadata: Path) -> dict[str, int]:
    path = metadata.with_name("videos.jsonl")
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "frame_count" in record:
            counts[record["video_id"]] = int(record["frame_count"])
    return counts


def analyse(video_id: str, scenes: list, frame_count: int | None) -> dict:
    ordered = sorted(scenes, key=lambda item: (item.start_frame, item.end_frame_exclusive))
    report: dict = {
        "video_id": video_id,
        "frame_count": frame_count,
        "scene_count": len(ordered),
        "gaps": [],
        "overlaps": [],
        "zero_length": [],
        "out_of_range": [],
        "missing_head": None,
        "missing_tail": None,
    }
    if not ordered:
        return report

    for scene in ordered:
        if scene.end_frame_exclusive <= scene.start_frame:
            report["zero_length"].append(scene.scene_id)
        if frame_count is not None and scene.end_frame_exclusive > frame_count:
            report["out_of_range"].append(
                {"scene_id": scene.scene_id, "end_frame_exclusive": scene.end_frame_exclusive}
            )

    if ordered[0].start_frame > 0:
        report["missing_head"] = {"start_frame": 0, "end_frame_exclusive": ordered[0].start_frame,
                                  "length": ordered[0].start_frame}

    for current, following in zip(ordered, ordered[1:], strict=False):
        if following.start_frame > current.end_frame_exclusive:
            report["gaps"].append({
                "after_scene": current.scene_id,
                "before_scene": following.scene_id,
                "start_frame": current.end_frame_exclusive,
                "end_frame_exclusive": following.start_frame,
                "length": following.start_frame - current.end_frame_exclusive,
            })
        elif following.start_frame < current.end_frame_exclusive:
            report["overlaps"].append({
                "scene_a": current.scene_id, "scene_b": following.scene_id,
                "length": current.end_frame_exclusive - following.start_frame,
            })

    if frame_count is not None and ordered[-1].end_frame_exclusive < frame_count:
        report["missing_tail"] = {
            "start_frame": ordered[-1].end_frame_exclusive,
            "end_frame_exclusive": frame_count,
            "length": frame_count - ordered[-1].end_frame_exclusive,
        }

    covered = sum(
        max(0, scene.end_frame_exclusive - scene.start_frame) for scene in ordered
    )
    report["covered_frames"] = covered
    report["coverage_ratio"] = round(covered / frame_count, 6) if frame_count else None
    return report


async def main_async(args: argparse.Namespace) -> None:
    repository = await JsonlSceneRepository.load(args.metadata)
    scenes = await repository.all()
    counts = load_frame_counts(args.metadata)

    by_video: dict[str, list] = {}
    for scene in scenes:
        by_video.setdefault(scene.video_id, []).append(scene)

    reports = [analyse(video, items, counts.get(video)) for video, items in sorted(by_video.items())]
    for report in reports:
        gaps = report["gaps"]
        lost = sum(item["length"] for item in gaps)
        print(f"\n=== {report['video_id']} ===")
        print(f"  scene={report['scene_count']}  frame_count={report['frame_count']}  "
              f"coverage={report['coverage_ratio']}")
        print(f"  gap={len(gaps)} (mất {lost} frame)  overlap={len(report['overlaps'])}  "
              f"zero_length={len(report['zero_length'])}  out_of_range={len(report['out_of_range'])}")
        if report["missing_head"]:
            print(f"  THIẾU ĐẦU : {report['missing_head']}")
        if report["missing_tail"]:
            print(f"  THIẾU ĐUÔI: {report['missing_tail']}")
        for gap in gaps[:10]:
            print(f"    gap {gap['start_frame']}..{gap['end_frame_exclusive']} "
                  f"({gap['length']} frame) giữa {gap['after_scene']} và {gap['before_scene']}")
        if len(gaps) > 10:
            print(f"    … còn {len(gaps) - 10} gap nữa")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra scene có lát kín video không")
    parser.add_argument("--metadata", type=Path, default=Path("storage/exports_l21/scenes.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
