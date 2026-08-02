"""Stage pack contract — cầu nối notebook Kaggle → canonical dataset.

Bối cảnh: mỗi giai đoạn trích xuất chạy trong MỘT notebook độc lập (scene
detection, ASR, keyframe, embedding, color... trên Kaggle T4; OCR/object/
caption trên A100). Mỗi notebook ghi ra một "stage pack" tự chứa, không
notebook nào ghi đè output của notebook khác, và có thể chạy lại riêng lẻ.

Trước module này các pack không có contract nào cả — `offline/pipeline.py`
tự cắt scene đều 8 giây và không đọc pack nào, nên toàn bộ dữ liệu
TransNetV2/faster-whisper đang chạy trên Kaggle không có đường vào hệ thống.

Cấu trúc một pack::

    <pack_root>/
      _SUCCESS.json      status=success + số lượng + thời gian chạy
      model_info.json    component, model, version, pack_version
      manifests/*.jsonl  payload chính (một dòng một record)
      videos/<vid>/...   shard theo video, để rerun lẻ từng video

Quy ước khóa join (quan trọng)::

    stage mức video     -> khóa `video_id`
    stage mức scene     -> khóa (`video_id`, `scene_index`)
    stage mức keyframe  -> khóa (`video_id`, `frame_idx`)

Stage mức keyframe KHÔNG được tự đặt `keyframe_id`: id canonical nhúng
`scene_idx` bên trong (`{video}_S{scene:04d}_F{frame:06d}`), mà notebook
trích OCR/caption không biết — và không nên biết — scene nào sẽ nhận frame
đó sau khi assemble. `offline/assemble.py` là nơi duy nhất dựng
`keyframe_id`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator
import zipfile

PACK_CONTRACT_VERSION = "1.0.0"

# Tên stage canonical -> tên file manifest chính của stage đó.
STAGE_MANIFESTS: dict[str, str] = {
    "video": "video_manifest.jsonl",
    "scene": "scene_manifest.jsonl",
    "keyframe": "keyframe_manifest.jsonl",
    "asr": "asr_segments.jsonl",
    "embedding": "embedding_manifest.jsonl",
    "color": "color_manifest.jsonl",
    "ocr": "ocr_manifest.jsonl",
    "object": "object_manifest.jsonl",
    "caption": "caption_manifest.jsonl",
}

# Không có ba stage này thì không dựng nổi một Scene canonical hợp lệ.
REQUIRED_STAGES = ("video", "scene", "keyframe")


class StagePackError(Exception):
    """Pack sai contract. Luôn nêu rõ pack nào và thiếu/sai cái gì."""


@dataclass(slots=True)
class StagePack:
    """Một stage pack đã mở và đã kiểm tra contract tối thiểu."""

    stage: str
    root: Path
    success: dict[str, Any]
    model_info: dict[str, Any]
    _extracted: Path | None = field(default=None, repr=False)

    @property
    def manifest_name(self) -> str:
        return STAGE_MANIFESTS[self.stage]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifests" / self.manifest_name

    def rows(self, manifest_name: str | None = None) -> Iterator[dict[str, Any]]:
        """Duyệt từng dòng của một manifest; bỏ qua dòng trắng."""

        path = self.root / "manifests" / (manifest_name or self.manifest_name)
        if not path.exists():
            raise StagePackError(
                f"stage {self.stage!r}: thiếu manifest {path.name} trong {self.root}"
            )
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StagePackError(
                        f"stage {self.stage!r}: {path.name} dòng {line_number} không phải JSON hợp lệ"
                    ) from exc

    def has_manifest(self, manifest_name: str) -> bool:
        return (self.root / "manifests" / manifest_name).exists()

    def provenance_fields(self) -> dict[str, str]:
        """Trích thông tin model để gắn vào ModelProvenance của record."""

        return {
            "model_name": str(self.model_info.get("model") or f"unknown:{self.stage}"),
            "model_revision": str(self.model_info.get("pack_version") or PACK_CONTRACT_VERSION),
        }


def _read_json(path: Path, *, stage_hint: str) -> dict[str, Any]:
    if not path.exists():
        raise StagePackError(f"stage {stage_hint!r}: thiếu {path.name} trong {path.parent}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StagePackError(f"stage {stage_hint!r}: {path.name} không phải JSON hợp lệ") from exc


def _resolve_root(root: Path) -> Path:
    """Notebook đóng gói zip có thể có hoặc không có thư mục bọc ngoài."""

    if (root / "_SUCCESS.json").exists():
        return root
    children = [item for item in root.iterdir() if item.is_dir()]
    if len(children) == 1 and (children[0] / "_SUCCESS.json").exists():
        return children[0]
    return root


def open_pack(source: Path, stage: str, *, extract_dir: Path | None = None) -> StagePack:
    """Mở một pack (thư mục hoặc .zip) và kiểm tra contract tối thiểu."""

    if stage not in STAGE_MANIFESTS:
        raise StagePackError(
            f"stage {stage!r} không nằm trong contract; hợp lệ: {sorted(STAGE_MANIFESTS)}"
        )
    extracted: Path | None = None
    if source.is_file() and source.suffix == ".zip":
        target = Path(extract_dir or tempfile.mkdtemp(prefix=f"aic_pack_{stage}_"))
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target)
        extracted = target
        root = _resolve_root(target)
    elif source.is_dir():
        root = _resolve_root(source)
    else:
        raise StagePackError(f"stage {stage!r}: {source} không phải thư mục hay file .zip")

    success = _read_json(root / "_SUCCESS.json", stage_hint=stage)
    if success.get("status") != "success":
        raise StagePackError(
            f"stage {stage!r}: _SUCCESS.json báo status={success.get('status')!r} — "
            "pack chưa chạy xong, không được đưa vào assemble"
        )
    model_info = _read_json(root / "model_info.json", stage_hint=stage)
    pack = StagePack(
        stage=stage, root=root, success=success, model_info=model_info, _extracted=extracted
    )
    if not pack.manifest_path.exists():
        raise StagePackError(
            f"stage {stage!r}: thiếu manifests/{pack.manifest_name} trong {root}"
        )
    return pack


def discover_packs(packs_dir: Path, *, extract_dir: Path | None = None) -> dict[str, StagePack]:
    """Tìm pack cho từng stage trong `packs_dir`.

    Nhận diện theo tên: thư mục/zip có chứa tên stage (vd `01_scene_detection`,
    `scene`, `03_asr_output.zip`). Nhiều pack cùng stage -> lỗi tường minh,
    không tự chọn bừa một cái.
    """

    if not packs_dir.is_dir():
        raise StagePackError(f"thư mục pack không tồn tại: {packs_dir}")
    found: dict[str, list[Path]] = {}
    for entry in sorted(packs_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir() and entry.suffix != ".zip":
            continue
        name = entry.name.casefold()
        # Khớp stage dài trước để "keyframe" không bị "frame" hay "key" nuốt.
        for stage in sorted(STAGE_MANIFESTS, key=len, reverse=True):
            if stage in name:
                found.setdefault(stage, []).append(entry)
                break
    duplicates = {stage: paths for stage, paths in found.items() if len(paths) > 1}
    if duplicates:
        detail = "; ".join(
            f"{stage}: {[item.name for item in paths]}" for stage, paths in duplicates.items()
        )
        raise StagePackError(
            f"nhiều pack cho cùng một stage ({detail}) — giữ đúng một bản cho mỗi stage"
        )
    return {
        stage: open_pack(paths[0], stage, extract_dir=extract_dir)
        for stage, paths in found.items()
    }


__all__ = [
    "PACK_CONTRACT_VERSION",
    "REQUIRED_STAGES",
    "STAGE_MANIFESTS",
    "StagePack",
    "StagePackError",
    "discover_packs",
    "open_pack",
]
