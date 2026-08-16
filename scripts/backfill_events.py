"""Bù `events.jsonl` + `extensions.event_id` cho video còn thiếu.

Đo được trên export hiện tại::

    events.jsonl        69 event, TOÀN BỘ thuộc L21_V001
    event_id trên scene 217/765 (28.4%)  -> V002 0/262, V003 0/286

Hai thứ hỏng theo, cả hai đều im lặng:

- **dedup theo event** chỉ chạy trên 1/3 corpus. Hai scene liền nhau của cùng
  một tin ở V002/V003 được tính là hai kết quả độc lập, còn ở V001 thì gộp một —
  tức metric giữa các video KHÔNG so sánh được với nhau.
- **nhánh `event_search`** (nếu bật) chỉ trả về V001, không bao giờ trả V002/V003.

Vì sao lệch: `events.jsonl` được sinh từ lần assemble khi corpus mới có V001.
Hai video sau thêm vào bằng đường khác và không đi qua bước gom event.

Script này KHÔNG chạy lại assemble. Assemble dựng lại từ stage pack, mà pack thì
không có phần bù OCR (`scripts/ocr_backfill.py`) lẫn chyron
(`scripts/chyron_backfill.py`) — cả hai đều ghi thẳng vào export. Chạy lại là
mất trắng chúng.

Dùng ĐÚNG `offline/event_grouping.py` mà assemble dùng, cùng tham số mặc định
của `OfflineSettings`, nên event của V002/V003 sinh ra giống hệt như thể chúng
đã có mặt từ lần assemble đầu.

    python -m scripts.backfill_events --dry-run
    python -m scripts.backfill_events
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import dataclasses

from datasection.schemas import ModelProvenance, Scene
from offline.config import OfflineSettings
from offline.event_grouping import build_event, group_scenes_into_events, link_event_neighbors

# `OfflineSettings` là dataclass `slots=True`, nên `OfflineSettings.event_max_gap_sec`
# trả về `member_descriptor` chứ KHÔNG phải giá trị mặc định — đưa thẳng vào
# argparse thì lỗi chỉ nổ ra sâu trong `group_scenes_into_events`. Lấy default
# đúng cách qua `dataclasses.fields`.
_DEFAULTS = {field.name: field.default for field in dataclasses.fields(OfflineSettings)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bu events.jsonl cho video con thieu")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--max-gap-sec", type=float, default=_DEFAULTS["event_max_gap_sec"])
    parser.add_argument("--max-duration-sec", type=float, default=_DEFAULTS["event_max_duration_sec"])
    parser.add_argument("--min-text-overlap", type=float, default=_DEFAULTS["event_min_text_overlap"])
    parser.add_argument("--config-id", default=_DEFAULTS["event_config_id"])
    parser.add_argument("--rebuild-all", action="store_true",
                        help="Dung lai event cho MOI video, ke ca video da co")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes_path = args.export / "scenes.jsonl"
    events_path = args.export / "events.jsonl"
    raw_scenes = [
        json.loads(line)
        for line in scenes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if events_path.exists() else []

    have = {event["video_id"] for event in existing}
    by_video: dict[str, list[dict]] = defaultdict(list)
    for scene in raw_scenes:
        by_video[scene["video_id"]].append(scene)

    targets = sorted(by_video) if args.rebuild_all else sorted(set(by_video) - have)
    print(f"video da co event: {sorted(have) or '(khong)'}")
    print(f"se dung event cho: {targets or '(khong con thieu)'}")
    if not targets:
        return

    provenance = ModelProvenance(
        model_name="event-grouping:backfill",
        model_revision="events-backfill-v1",
        pipeline_version="aic-v1.0.0",
        device="cpu",
    )

    built: list[dict] = []
    event_id_by_scene: dict[str, str] = {}
    for video_id in targets:
        rows = sorted(by_video[video_id], key=lambda item: int(item["scene_idx"]))
        # Validate qua pydantic: `build_event` đọc `scene.captions[].text`,
        # `scene.keywords[].normalized_text`, `scene.keyframes[0].keyframe_id` —
        # đưa dict thô vào sẽ nổ ở chỗ khó lần ra.
        scenes = [Scene.model_validate(row) for row in rows]
        groups = group_scenes_into_events(
            scenes, args.max_gap_sec, args.max_duration_sec, args.min_text_overlap
        )
        events = link_event_neighbors([
            build_event(video_id, index, group, args.config_id, provenance)
            for index, group in enumerate(groups)
        ])
        print(f"  {video_id}: {len(scenes)} scene -> {len(events)} event")
        for event in events:
            built.append(event.model_dump(mode="json"))
            for scene_id in event.scene_ids:
                event_id_by_scene[scene_id] = event.event_id

    if args.dry_run:
        print("(--dry-run: khong ghi gi)")
        return

    kept = [
        event for event in existing
        if args.rebuild_all is False and event["video_id"] not in set(targets)
    ]
    merged = sorted(kept + built, key=lambda item: (item["video_id"], item["event_idx"]))
    events_path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in merged),
        encoding="utf-8",
    )

    # `extensions.event_id` trên scene là bản sao nướng sẵn của quan hệ này —
    # online đọc nó để dedup mà không phải nạp events.jsonl mỗi request. Quên
    # bước này thì events.jsonl đúng nhưng dedup vẫn hỏng.
    stamped = 0
    for scene in raw_scenes:
        event_id = event_id_by_scene.get(scene["scene_id"])
        if event_id:
            scene.setdefault("extensions", {})["event_id"] = event_id
            stamped += 1
    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in raw_scenes),
        encoding="utf-8",
    )
    print(f"xong: {len(merged)} event tong cong, {stamped} scene duoc gan event_id moi")


if __name__ == "__main__":
    main()
