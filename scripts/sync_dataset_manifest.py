"""Đồng bộ `dataset_manifest.json` với các file export THẬT đang có trên đĩa.

`dataset_manifest.json` không tự cập nhật khi export bị sửa bằng script rời
(`ocr_backfill`, `chyron_backfill`, `backfill_color_quality`, `backfill_events`
đều ghi thẳng vào export). Hệ quả đo được::

    verify_export()  ->  checksum mismatch: videos.jsonl
    clip_count 426   ->  clips.jsonl KHÔNG tồn tại
    /v1/health       ->  bao build_id cu, khong mot canh bao nao

Script này KHÔNG dựng lại dữ liệu — nó chỉ đếm lại, băm lại, và ghi `build_id`
mới, để checksum quay về đúng vai trò "biết export có bị sửa hay không".

`clips.jsonl` vắng mặt được ghi nhận TRUNG THỰC là `clip_count: 0` thay vì giữ
con số 426 của một file không còn tồn tại. Online không đọc clip
(xem docs/29 §4.5) nên điều này không đổi hành vi tìm kiếm; nó chỉ khiến
`verify_export()` chạy được trở lại.

    python -m scripts.sync_dataset_manifest --dry-run
    python -m scripts.sync_dataset_manifest
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from datasection.exporter import sha256_file

FILES = ("videos.jsonl", "scenes.jsonl", "keyframes.jsonl", "clips.jsonl", "events.jsonl")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Dong bo dataset_manifest.json voi export that")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--keep-build-id", action="store_true",
                        help="Giu nguyen build_id cu (chi sua checksum/count)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.export / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    counts = {name: _count_lines(args.export / name) for name in FILES}
    keyframe_count = 0
    with (args.export / "scenes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keyframe_count += len(json.loads(line).get("keyframes") or [])

    checksums = {
        name: sha256_file(args.export / name)
        for name in FILES
        if (args.export / name).exists()
    }

    updated = dict(manifest)
    updated["video_count"] = counts["videos.jsonl"]
    updated["scene_count"] = counts["scenes.jsonl"]
    updated["keyframe_count"] = keyframe_count
    updated["clip_count"] = counts["clips.jsonl"]
    updated["event_count"] = counts["events.jsonl"]
    updated["export_checksums"] = checksums
    if not args.keep_build_id:
        updated["build_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"{'truong':18s} {'cu':>12s} {'moi':>12s}")
    for key in ("build_id", "video_count", "scene_count", "keyframe_count", "clip_count", "event_count"):
        old, new = manifest.get(key), updated.get(key)
        flag = "" if old == new else "   <- doi"
        print(f"{key:18s} {str(old):>12s} {str(new):>12s}{flag}")
    changed = [
        name for name in FILES
        if manifest.get("export_checksums", {}).get(name) != checksums.get(name)
    ]
    print(f"checksum doi: {changed or '(khong)'}")
    missing = [name for name in FILES if not (args.export / name).exists()]
    if missing:
        print(f"file VANG MAT (bo khoi checksum): {missing}")

    if args.dry_run:
        print("(--dry-run: khong ghi gi)")
        return

    manifest_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"da ghi {manifest_path}")

    from datasection.exporter import verify_export

    try:
        verify_export(args.export)
        print("verify_export: PASS")
    except Exception as exc:  # noqa: BLE001 - bao nguyen van ly do that bai
        print(f"verify_export: VAN FAIL -> {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
