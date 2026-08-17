"""Nạp `AIC2026_competition_clean_v3` vào format export canonical của repo.

Pack thi đấu và export của repo là HAI lược đồ khác nhau, không chỉ khác tên
trường mà khác cả cách định danh:

    pack   scene_id     L21_V001_S00000      (5 chữ số)
    repo   scene_id     L21_V001_S0000       (4 — `SCENE_ID_PATTERN` ép)
    pack   keyframe_id  L21_V001_F000000090  (không mang scene)
    repo   keyframe_id  L21_V001_S0002_F000090

Nên bước này KHÔNG phải đổi tên trường: nó đúc lại id theo regex của contract,
gộp bằng chứng từ 6 file rời của pack về đúng một `Scene` lồng `Keyframe`, và
kiểm mọi bất biến mà `datasection.schemas` sẽ kiểm lại lúc server nạp — kiểm ở
đây để lỗi hiện ra lúc convert (có tên video, có số dòng) chứ không phải lúc
khởi động server giữa buổi thi.

Id gốc của pack được giữ trong `extensions.pack` của từng bản ghi: sau khi đúc
lại id thì không còn đường nào truy ngược về pack, mà báo cáo `reports/` của
pack lại đánh số theo id cũ.

QUYẾT ĐỊNH CÓ HỆ QUẢ, ghi ở đây vì không đọc được từ code:

1.  **Scene không có keyframe bị BỎ.** `Scene.keyframes` khai `min_length=1` —
    không phải chỗ này chọn, contract chọn. Pack có 13.719/101.461 scene như
    vậy. Đo được phần mất thật: 1.948/135.970 đoạn ASR (1,4%) chỉ xuất hiện ở
    những scene đó; phần còn lại vẫn nằm trong scene có keyframe vì một đoạn
    ASR thường phủ nhiều scene.

2.  **KHÔNG ghi caption mức scene.** Trong pack `scene_evidence.caption_vi`
    đúng bằng caption của keyframe ĐẦU TIÊN, chép nguyên văn. Mà
    `project_scene()` đã cộng caption của mọi keyframe lên scene rồi, nên ghi
    thêm sẽ khiến caption của keyframe đầu được đếm hai lần trong BM25 — một
    thiên lệch xếp hạng không ai khai báo và không nhìn thấy được từ kết quả.

3.  **`end_sec` tính lại từ frame, không lấy của pack.** Pack ghi `end_sec` của
    frame CUỐI (bao gồm), còn contract đòi `start_sec <= ts < end_sec`. Lấy
    nguyên thì mọi keyframe rơi đúng frame cuối scene sẽ trượt validate.

4.  **`color` để TRỐNG dù pack có HSV 100%.** `color_search` khớp CHÍNH XÁC
    theo chuỗi giữa `COLOR_LEXICON` và `dominant_colors[].name`, mà tên màu
    được đếm trên phân bố ĐỒNG THỜI của (hue, sat, val) — xem
    `scripts/backfill_color_quality.py`. Pack chỉ có ba histogram BIÊN, không
    khôi phục được phân bố đồng thời. Suy ra tên màu từ đó là bịa một tín hiệu
    xếp hạng. Cần `color` thì phải có ảnh rồi chạy backfill.

5.  **`image_path` vẫn được ghi dù pack KHÔNG kèm ảnh nào.** Đường dẫn trỏ tới
    chỗ ảnh PHẢI nằm; `--report` đếm bao nhiêu ảnh thật sự có trên đĩa. Không
    ghi thì lúc ảnh về phải convert lại toàn bộ.

Vector giữ nguyên bố cục của pack (một `.npy` cho mỗi video, float16) và được
trỏ tới bằng `vector_uri` dạng `<đường dẫn>.npy#<hàng>`. Không nổ ra 168.960
file rời: trên NTFS đó là 168.960 lần mở file mỗi lần khởi động.
`online/adapters/frame_vector_store.py::_read_vector_file` hiểu cú pháp `#hàng`.

    python -m scripts.import_competition_pack --pack "D:/.../AIC2026_competition_clean_v3.zip"
    python -m scripts.import_competition_pack --pack ... --batch L22 --batch L23
    python -m scripts.import_competition_pack --pack ... --limit-videos 5 --dry-run
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator
import zipfile

PIPELINE_VERSION = "aic-v1.0.0"

# Đọc từ pack. `online/*` cố tình không có trong danh sách: cả ba file ở đó
# trùng byte-for-byte với bản `canonical/*` (đã đối chiếu CRC), đọc thêm chỉ
# tốn thời gian.
F_MAPPING = "canonical/keyframe_scene_mapping.csv"
F_SCENES = "canonical/scene_manifest.jsonl"
F_FRAME_EVIDENCE = "canonical/frame_evidence.jsonl"
F_SCENE_EVIDENCE = "canonical/scene_evidence.jsonl"
F_ASR = "canonical/asr_segments.jsonl"
F_ASR_STATUS = "canonical/asr_video_status.jsonl"
F_EMBEDDINGS = "canonical/embedding_manifest.jsonl"
F_VIDEO_MANIFEST = "online/video_manifest.jsonl"
F_INDEX_VERSION = "index_version.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PackSource:
    """Đọc pack từ file .zip HOẶC thư mục đã giải nén, cùng một giao diện.

    Đọc thẳng trong .zip là mặc định vì bản v3 nở 570 MB -> 2,44 GB khi giải
    nén, mà 636 MB trong đó là ba cặp file trùng nhau.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._zip: zipfile.ZipFile | None = None
        if root.is_dir():
            self._dir: Path | None = root
        elif root.is_file():
            self._dir = None
            self._zip = zipfile.ZipFile(root)
        else:
            raise SystemExit(f"khong tim thay pack: {root}")

    def exists(self, name: str) -> bool:
        if self._dir is not None:
            return (self._dir / name).exists()
        assert self._zip is not None
        try:
            self._zip.getinfo(name)
        except KeyError:
            return False
        return True

    def lines(self, name: str) -> Iterator[str]:
        if self._dir is not None:
            with (self._dir / name).open(encoding="utf-8") as handle:
                yield from handle
            return
        assert self._zip is not None
        with self._zip.open(name) as raw:
            yield from io.TextIOWrapper(raw, encoding="utf-8")

    def read_json(self, name: str) -> Any:
        if self._dir is not None:
            return json.loads((self._dir / name).read_text(encoding="utf-8"))
        assert self._zip is not None
        return json.loads(self._zip.read(name).decode("utf-8"))

    def copy_to(self, name: str, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self._dir is not None:
            shutil.copyfile(self._dir / name, destination)
        else:
            assert self._zip is not None
            with self._zip.open(name) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
        return destination.stat().st_size


class VideoStream:
    """Một file của pack, đọc theo NHÓM video, mỗi lần một video.

    Cả 5 file lớn của pack v3 đều đã sắp xếp tăng dần và liền mạch theo
    `video_id` (đã kiểm trước khi viết hàm này), nên phép trộn dưới đây không
    cần giữ gì trong RAM ngoài đúng một video.

    Thứ tự được kiểm TẠI CHỖ chứ không tin: một bản pack sau mà không còn sắp
    xếp thì phép trộn sẽ lặng lẽ bỏ qua các video lệch chỗ, và export ra vẫn
    hợp lệ — chỉ thiếu dữ liệu. Sai thứ tự là dừng hẳn.
    """

    def __init__(self, source: PackSource, name: str, *, parse=json.loads) -> None:
        self.name = name
        self._iterator = self._groups(source, name, parse)
        self._head: tuple[str, list[dict]] | None = next(self._iterator, None)

    @classmethod
    def from_groups(cls, name: str, groups: Iterator[tuple[str, list[dict]]]) -> "VideoStream":
        """Cùng giao diện nhưng nhóm do caller dựng — dùng cho CSV, thứ không
        đọc được theo từng dòng độc lập vì còn dòng header."""

        stream = cls.__new__(cls)
        stream.name = name
        stream._iterator = groups
        stream._head = next(groups, None)
        return stream

    @staticmethod
    def _groups(source: PackSource, name: str, parse) -> Iterator[tuple[str, list[dict]]]:
        current: str | None = None
        bucket: list[dict] = []
        seen: set[str] = set()
        for line in source.lines(name):
            if not line.strip():
                continue
            record = parse(line)
            video_id = record["video_id"]
            if video_id != current:
                if current is not None:
                    yield current, bucket
                if video_id in seen:
                    raise SystemExit(
                        f"{name}: video {video_id} xuat hien lai sau khi da doi sang video "
                        "khac. File khong con nhom lien mach theo video_id — phep tron "
                        "trong script nay se bo sot du lieu. Dung lai."
                    )
                seen.add(video_id)
                current, bucket = video_id, []
            bucket.append(record)
        if current is not None:
            yield current, bucket

    def take(self, video_id: str) -> list[dict]:
        """Bản ghi của `video_id`, hoặc [] nếu file này không có video đó."""

        if self._head is not None and self._head[0] == video_id:
            records = self._head[1]
            self._head = next(self._iterator, None)
            return records
        return []

    def leftover(self) -> str | None:
        return None if self._head is None else self._head[0]


def _rows_from_csv(source: PackSource, name: str) -> Iterator[dict]:
    yield from csv.DictReader(source.lines(name))


@dataclass
class Report:
    videos_in_pack: int = 0
    videos_written: int = 0
    scenes_written: int = 0
    scenes_dropped_no_keyframe: int = 0
    keyframes_written: int = 0
    keyframes_with_caption: int = 0
    keyframes_with_ocr: int = 0
    keyframes_with_vector: int = 0
    keyframes_vector_from_donor: int = 0
    keyframes_image_on_disk: int = 0
    asr_segments_written: int = 0
    asr_segments_skipped: int = 0
    scene_bounds_widened: int = 0
    max_bounds_widening_sec: float = 0.0
    vector_files_copied: int = 0
    vector_bytes_copied: int = 0
    frame_size_source: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in self.__dict__.items()
        }
        return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _image_size(path: Path) -> tuple[int, int] | None:
    """Kích thước thật của ảnh, hoặc None nếu ảnh chưa có trên đĩa.

    Chỉ đọc header (PIL mở lazy), không giải mã pixel.
    """

    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as handle:
            return int(handle.width), int(handle.height)
    except (FileNotFoundError, OSError):
        return None


def _load_donor_embeddings(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """`embedding_refs` của một export CÓ SẴN, khoá theo (video_id, frame_idx).

    Lý do tồn tại: pack v3 có vector cho 95,6% keyframe nhưng riêng batch L21
    chỉ 43/7.790 — mà toàn bộ 120 truy vấn gold nằm trên L21_V001..V003. Nhập
    pack không thôi thì corpus đầy đủ có đúng một lỗ hổng dense ngay chỗ duy
    nhất đo được. Trên máy này 855 vector cho ba video đó đã có sẵn từ trước.

    Chỉ BÙ vào chỗ trống: `embedding_name` nào keyframe đã có từ pack thì giữ
    của pack. Không đảo thứ tự ưu tiên giữa hai nguồn cùng tên.
    """

    donor: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            references = raw.get("embedding_refs") or []
            if references:
                donor[(str(raw["video_id"]), int(raw["frame_idx"]))] = references
    return donor


def _provenance(model_name: str, revision: str | None, **parameters: Any) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_revision": revision,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": _now(),
        "parameters": parameters,
    }


def _build_keyframe(
    *,
    video_id: str,
    scene_id: str,
    frame_idx: int,
    timestamp_sec: float,
    evidence: dict[str, Any],
    mapping: dict[str, Any],
    embedding_refs: list[dict[str, Any]],
    frame_size: tuple[int, int],
    frame_size_source: str,
    caption_provenance: dict[str, Any],
    ocr_provenance: dict[str, Any],
    keep_hsv: bool,
) -> dict[str, Any]:
    captions: list[dict[str, Any]] = []
    caption_text = (evidence.get("caption_vi") or "").strip()
    if caption_text:
        captions.append(
            {
                "language": "vi",
                "caption_type": "detailed",
                "text": caption_text,
                "provenance": caption_provenance,
            }
        )

    # Pack v3 có OCR 0% trên cả 873 video. Đường ánh xạ vẫn viết đủ để khi bản
    # pack sau kèm OCR thì chỉ cần chạy lại script này, không phải sửa code.
    ocr_instances: list[dict[str, Any]] = []
    for item in evidence.get("ocr_instances") or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        instance: dict[str, Any] = {"text": text, "provenance": ocr_provenance}
        box = item.get("bbox")
        if isinstance(box, dict) and {"x1", "y1", "x2", "y2"} <= set(box):
            instance["bbox"] = {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")}
        if item.get("confidence") is not None:
            instance["confidence"] = float(item["confidence"])
        ocr_instances.append(instance)

    width, height = frame_size
    extensions: dict[str, Any] = {
        "pack": {
            "keyframe_id": evidence.get("keyframe_id") or mapping.get("keyframe_id"),
            "source_keyframe_index": mapping.get("source_keyframe_index"),
            "mapping_status": mapping.get("mapping_status"),
            "clean_status": evidence.get("clean_status"),
        },
        "frame_size_source": frame_size_source,
    }
    if keep_hsv and evidence.get("hsv_features"):
        extensions["hsv_features"] = evidence["hsv_features"]

    keyframe: dict[str, Any] = {
        "schema_version": "1.0.0",
        "keyframe_id": f"{scene_id}_F{frame_idx:06d}",
        "video_id": video_id,
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "image_path": f"processed/keyframes/{video_id}/frame_{frame_idx:06d}.jpg",
        "width": width,
        "height": height,
        "roles": ["representative"],
        "captions": captions,
        "ocr_instances": ocr_instances,
        "objects": [],
        "action_tags": [],
        "color": None,
        "embedding_refs": embedding_refs,
        "extensions": extensions,
    }
    return keyframe


def _asr_segments_for_scene(
    *,
    scene_id: str,
    scene_start_sec: float,
    scene_end_sec: float,
    segment_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    report: Report,
) -> list[dict[str, Any]]:
    """`ASRSegment` của contract là HÌNH CHIẾU của đoạn ASR lên scene, đã cắt
    theo biên scene — không phải đoạn gốc.

    Pack gắn cho mỗi scene danh sách đoạn ASR GIAO với nó, và mốc thời gian
    trong đó là mốc trên trục video, thường tràn ra ngoài scene. `Scene` từ chối
    đúng trường hợp đó (`ASR segment ... is outside scene time interval`), nên
    phải cắt tại đây. Không cắt thì cả 873 video đều không nạp được.

    Chỉ mốc thời gian bị cắt, `text` giữ nguyên cả câu: một câu nói vắt qua hai
    scene thì cả hai scene đều nên tìm được bằng câu đó.
    """

    out: list[dict[str, Any]] = []
    for index, pack_id in enumerate(segment_ids):
        raw = by_id.get(pack_id)
        if raw is None:
            report.asr_segments_skipped += 1
            continue
        text = (raw.get("text") or "").strip()
        start = max(float(raw.get("start_sec") or 0.0), scene_start_sec)
        end = min(float(raw.get("end_sec") or 0.0), scene_end_sec)
        if not text or end <= start:
            # Phần giao rỗng (hoặc mỏng tới mức làm tròn về 0) — đoạn này không
            # thật sự thuộc scene.
            report.asr_segments_skipped += 1
            continue
        # `A000000` của pack -> `ASR000000` mà `ASR_SOURCE_ID_PATTERN` đòi.
        suffix = pack_id.rsplit("_A", 1)[-1]
        if not suffix.isdigit():
            report.asr_segments_skipped += 1
            continue
        out.append(
            {
                "segment_id": f"{scene_id}_A{index:04d}",
                "source_segment_id": f"{raw['video_id']}_ASR{int(suffix):06d}",
                "start_sec": start,
                "end_sec": end,
                "text": text,
                "language": raw.get("language"),
                "provenance": _provenance(
                    str(raw.get("model") or "faster-whisper:large-v3"),
                    str(raw.get("compute_type") or "") or None,
                ),
            }
        )
        report.asr_segments_written += 1
    return out


def convert(arguments: argparse.Namespace) -> Report:
    source = PackSource(Path(arguments.pack))
    for name in (F_MAPPING, F_SCENES, F_FRAME_EVIDENCE, F_SCENE_EVIDENCE, F_EMBEDDINGS):
        if not source.exists(name):
            raise SystemExit(f"pack thieu file bat buoc: {name}")

    index_version = source.read_json(F_INDEX_VERSION) if source.exists(F_INDEX_VERSION) else {}
    fingerprint = str(index_version.get("fingerprint_sha256") or "unknown")[:16]
    pack_version = str(index_version.get("pack_version") or "unknown")
    embedding_meta = index_version.get("embedding") or {}
    embedding_name = str(embedding_meta.get("name") or "jina_clip_v2")
    embedding_model = str(embedding_meta.get("model") or "jinaai/jina-clip-v2")
    embedding_dim = int(embedding_meta.get("dimension") or 1024)
    embedding_normalized = bool(embedding_meta.get("normalized", True))

    caption_provenance = _provenance(f"competition_pack_v{pack_version}:caption", fingerprint)
    ocr_provenance = _provenance(f"competition_pack_v{pack_version}:ocr", fingerprint)

    out_dir = Path(arguments.out)
    data_root = Path(arguments.data_root)
    vectors_rel = arguments.vectors_dir.strip("/")
    vectors_dir = data_root / vectors_rel

    width, _, height = arguments.assume_frame_size.lower().partition("x")
    assumed_size = (int(width), int(height))

    batches = {item.upper() for item in (arguments.batch or [])}
    wanted_videos = set(arguments.video or [])

    donor_embeddings: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if arguments.merge_embeddings_from:
        donor_path = Path(arguments.merge_embeddings_from)
        if donor_path.is_dir():
            donor_path = donor_path / "keyframes.jsonl"
        if not donor_path.exists():
            raise SystemExit(f"--merge-embeddings-from: khong thay {donor_path}")
        donor_embeddings = _load_donor_embeddings(donor_path)
        print(f"  nap {len(donor_embeddings)} keyframe co vector tu {donor_path}")

    # Bảng nhỏ, nạp trọn: 873 dòng mỗi cái.
    asr_status: dict[str, dict[str, Any]] = {}
    if source.exists(F_ASR_STATUS):
        for line in source.lines(F_ASR_STATUS):
            if line.strip():
                record = json.loads(line)
                asr_status[record["video_id"]] = record
    video_manifest: dict[str, dict[str, Any]] = {}
    if source.exists(F_VIDEO_MANIFEST):
        for line in source.lines(F_VIDEO_MANIFEST):
            if line.strip():
                record = json.loads(line)
                video_manifest[record["video_id"]] = record

    report = Report()
    streams = {
        "scenes": VideoStream(source, F_SCENES),
        "frames": VideoStream(source, F_FRAME_EVIDENCE),
        "scene_evidence": VideoStream(source, F_SCENE_EVIDENCE),
        "embeddings": VideoStream(source, F_EMBEDDINGS),
    }
    asr_stream = VideoStream(source, F_ASR) if source.exists(F_ASR) else None

    # CSV cần reader riêng: `VideoStream` parse theo từng dòng độc lập, còn csv
    # phải đọc header một lần. Gom theo video bằng cùng cơ chế nhóm liền mạch.
    def _mapping_groups() -> Iterator[tuple[str, list[dict]]]:
        current: str | None = None
        bucket: list[dict] = []
        for row in _rows_from_csv(source, F_MAPPING):
            video_id = row["video_id"]
            if video_id != current:
                if current is not None:
                    yield current, bucket
                current, bucket = video_id, []
            bucket.append(row)
        if current is not None:
            yield current, bucket

    mapping_stream = VideoStream.from_groups(F_MAPPING, _mapping_groups())

    if not arguments.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        vectors_dir.mkdir(parents=True, exist_ok=True)

    scenes_path = out_dir / "scenes.jsonl"
    keyframes_path = out_dir / "keyframes.jsonl"
    videos_path = out_dir / "videos.jsonl"

    handles: dict[str, Any] = {}
    if not arguments.dry_run:
        handles = {
            "scenes": scenes_path.open("w", encoding="utf-8", newline="\n"),
            "keyframes": keyframes_path.open("w", encoding="utf-8", newline="\n"),
            "videos": videos_path.open("w", encoding="utf-8", newline="\n"),
            "clips": (out_dir / "clips.jsonl").open("w", encoding="utf-8", newline="\n"),
            "events": (out_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n"),
        }

    def emit(kind: str, record: dict[str, Any]) -> None:
        if handles:
            handles[kind].write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        # Danh sách video chủ đạo lấy từ mapping CSV: nó là định nghĩa "frame nào
        # là canonical" của chính pack.
        while mapping_stream.leftover() is not None:
            video_id = mapping_stream.leftover()
            assert video_id is not None
            mapping_rows = mapping_stream.take(video_id)
            scene_rows = streams["scenes"].take(video_id)
            frame_rows = streams["frames"].take(video_id)
            scene_evidence_rows = streams["scene_evidence"].take(video_id)
            embedding_rows = streams["embeddings"].take(video_id)
            asr_rows = asr_stream.take(video_id) if asr_stream else []
            report.videos_in_pack += 1

            batch_id = video_id.split("_")[0]
            if batches and batch_id.upper() not in batches:
                continue
            if wanted_videos and video_id not in wanted_videos:
                continue
            if arguments.limit_videos and report.videos_written >= arguments.limit_videos:
                # Dừng hẳn chứ không chạy nốt: `--limit-videos` tồn tại để thử
                # nhanh, mà quét hết 873 video chỉ để bỏ qua thì mất vài phút.
                # Đổi lại `leftover()` bên dưới sẽ báo dư — nên bỏ qua cảnh báo
                # đó khi có limit.
                break

            written = _convert_video(
                video_id=video_id,
                mapping_rows=mapping_rows,
                scene_rows=scene_rows,
                frame_rows=frame_rows,
                scene_evidence_rows=scene_evidence_rows,
                embedding_rows=embedding_rows,
                asr_rows=asr_rows,
                asr_status=asr_status.get(video_id) or {},
                video_manifest=video_manifest.get(video_id) or {},
                emit=emit,
                report=report,
                data_root=data_root,
                vectors_rel=vectors_rel,
                assumed_size=assumed_size,
                caption_provenance=caption_provenance,
                ocr_provenance=ocr_provenance,
                embedding_name=embedding_name,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                embedding_normalized=embedding_normalized,
                keep_hsv=arguments.keep_hsv,
                pack_version=pack_version,
                donor_embeddings=donor_embeddings,
            )
            if not written:
                continue
            report.videos_written += 1

            vector_member = f"dense/vectors/{video_id}.npy"
            if not arguments.dry_run and embedding_rows and source.exists(vector_member):
                size = source.copy_to(vector_member, vectors_dir / f"{video_id}.npy")
                report.vector_files_copied += 1
                report.vector_bytes_copied += size
    finally:
        for handle in handles.values():
            handle.close()

    for stream in (*streams.values(), asr_stream):
        if arguments.limit_videos:
            break
        if stream is not None and stream.leftover() is not None:
            report.warnings.append(
                f"{stream.name}: con du lieu chua doc tu video {stream.leftover()} — "
                "file nay chua nhieu video hon mapping CSV"
            )

    if not arguments.dry_run:
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": f"competition-pack-v{pack_version}",
            "build_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "pipeline_version": PIPELINE_VERSION,
            "video_count": report.videos_written,
            "scene_count": report.scenes_written,
            "keyframe_count": report.keyframes_written,
            "clip_count": 0,
            "event_count": 0,
            "models": [
                {
                    "task": "scene",
                    "model_name": "transnetv2_pytorch",
                    "revision": fingerprint,
                    "config_checksum": None,
                },
                {
                    "task": "caption",
                    "model_name": f"competition_pack_v{pack_version}",
                    "revision": fingerprint,
                    "config_checksum": None,
                },
                {
                    "task": "asr",
                    "model_name": "faster-whisper:large-v3",
                    "revision": fingerprint,
                    "config_checksum": None,
                },
                {
                    "task": "embedding",
                    "model_name": embedding_model,
                    "revision": fingerprint,
                    "config_checksum": None,
                },
            ],
            "indexes": [],
            "export_checksums": {
                name: _sha256(out_dir / name)
                for name in ("videos.jsonl", "scenes.jsonl", "keyframes.jsonl", "clips.jsonl", "events.jsonl")
            },
            "created_at": _now(),
        }
        (out_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "import_report.json").write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def _convert_video(
    *,
    video_id: str,
    mapping_rows: list[dict],
    scene_rows: list[dict],
    frame_rows: list[dict],
    scene_evidence_rows: list[dict],
    embedding_rows: list[dict],
    asr_rows: list[dict],
    asr_status: dict[str, Any],
    video_manifest: dict[str, Any],
    emit,
    report: Report,
    data_root: Path,
    vectors_rel: str,
    assumed_size: tuple[int, int],
    caption_provenance: dict[str, Any],
    ocr_provenance: dict[str, Any],
    embedding_name: str,
    embedding_model: str,
    embedding_dim: int,
    embedding_normalized: bool,
    keep_hsv: bool,
    pack_version: str,
    donor_embeddings: dict[tuple[str, int], list[dict[str, Any]]],
) -> bool:
    if not mapping_rows or not scene_rows:
        return False

    fps_values = [float(row["fps"]) for row in mapping_rows if row.get("fps")]
    fps = fps_values[0] if fps_values else 0.0
    if fps <= 0:
        report.warnings.append(f"{video_id}: khong co fps trong mapping CSV — bo video")
        return False

    evidence_by_frame = {int(row["frame_idx"]): row for row in frame_rows}
    mapping_by_frame = {int(row["frame_idx"]): row for row in mapping_rows}
    embedding_by_frame = {int(row["frame_idx"]): row for row in embedding_rows}
    asr_by_id = {row["asr_segment_id"]: row for row in asr_rows}
    scene_evidence_by_id = {row["scene_id"]: row for row in scene_evidence_rows}

    # Kích thước frame: đọc từ ảnh THẬT nếu có, đó là con số duy nhất đúng.
    # Pack không mang width/height ở bất kỳ file nào.
    frame_size = assumed_size
    frame_size_source = "assumed"
    keyframe_dir = data_root / "processed" / "keyframes" / video_id
    if keyframe_dir.is_dir():
        for row in mapping_rows[:8]:
            probe = keyframe_dir / f"frame_{int(row['frame_idx']):06d}.jpg"
            measured = _image_size(probe)
            if measured:
                frame_size, frame_size_source = measured, "measured"
                break
    report.frame_size_source[frame_size_source] += 1

    scenes_written = 0
    keyframes_written = 0
    max_frame = 0

    for scene_row in scene_rows:
        pack_scene_id = scene_row["scene_id"]
        scene_idx = int(scene_row["scene_index"])
        scene_id = f"{video_id}_S{scene_idx:04d}"
        start_frame = int(scene_row["start_frame"])
        end_frame_exclusive = int(scene_row["end_frame"]) + 1
        max_frame = max(max_frame, end_frame_exclusive)
        # Bao giờ cũng dựng lại từ frame: xem quyết định 3 ở đầu file.
        start_sec = start_frame / fps
        end_sec = end_frame_exclusive / fps

        evidence_row = scene_evidence_by_id.get(pack_scene_id) or {}
        frame_indices = [
            index
            for index in (evidence_row.get("frame_indices") or [])
            if int(index) in mapping_by_frame
        ]
        if not frame_indices:
            report.scenes_dropped_no_keyframe += 1
            continue

        keyframes: list[dict[str, Any]] = []
        for frame_idx in sorted(int(item) for item in frame_indices):
            mapping = mapping_by_frame[frame_idx]
            evidence = evidence_by_frame.get(frame_idx, {})
            timestamp = float(mapping.get("timestamp_sec") or frame_idx / fps)

            embedding_refs: list[dict[str, Any]] = []
            embedding_row = embedding_by_frame.get(frame_idx)
            if embedding_row:
                for reference in embedding_row.get("embedding_refs") or []:
                    locations = reference.get("storage_locations") or []
                    if not locations:
                        continue
                    uri = str(locations[0].get("vector_uri") or "")
                    _, _, row_index = uri.rpartition("#")
                    if not row_index.isdigit():
                        continue
                    embedding_refs.append({
                        "embedding_name": str(reference.get("embedding_name") or embedding_name),
                        "modality": "image",
                        "model_name": str(reference.get("model_name") or embedding_model),
                        "model_revision": None,
                        "dimension": int(reference.get("dimension") or embedding_dim),
                        "normalized": bool(reference.get("normalized", embedding_normalized)),
                        "storage_locations": [
                            {
                                "backend": "file",
                                "vector_id": f"{video_id}_F{frame_idx:06d}",
                                "index_name": str(
                                    reference.get("embedding_name") or embedding_name
                                ),
                                "vector_uri": f"{vectors_rel}/{video_id}.npy#{int(row_index)}",
                            }
                        ],
                    })
                    break

            # Bù từ export có sẵn, chỉ cho `embedding_name` mà pack không có.
            donated = donor_embeddings.get((video_id, frame_idx))
            if donated:
                have = {item["embedding_name"] for item in embedding_refs}
                extra = [item for item in donated if item.get("embedding_name") not in have]
                if extra:
                    embedding_refs.extend(extra)
                    report.keyframes_vector_from_donor += 1

            keyframe = _build_keyframe(
                video_id=video_id,
                scene_id=scene_id,
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                evidence=evidence,
                mapping=mapping,
                embedding_refs=embedding_refs,
                frame_size=frame_size,
                frame_size_source=frame_size_source,
                caption_provenance=caption_provenance,
                ocr_provenance=ocr_provenance,
                keep_hsv=keep_hsv,
            )
            keyframes.append(keyframe)
            if keyframe["captions"]:
                report.keyframes_with_caption += 1
            if keyframe["ocr_instances"]:
                report.keyframes_with_ocr += 1
            if embedding_refs:
                report.keyframes_with_vector += 1
            if (data_root / keyframe["image_path"]).exists():
                report.keyframes_image_on_disk += 1

        # `pts_time` của pack đến từ container video, không phải frame_idx/fps,
        # nên có thể lệch quá biên scene một chút. Nới biên đúng phần lệch —
        # dưới một frame — và ĐẾM, thay vì vứt keyframe hoặc để validate nổ.
        timestamps = [item["timestamp_sec"] for item in keyframes]
        low, high = min(timestamps), max(timestamps)
        if low < start_sec or high >= end_sec:
            widening = max(start_sec - low, high - end_sec + 1.0 / fps, 0.0)
            report.scene_bounds_widened += 1
            report.max_bounds_widening_sec = max(report.max_bounds_widening_sec, widening)
            start_sec = min(start_sec, low)
            end_sec = max(end_sec, high + 1.0 / fps)

        scene = {
            "schema_version": "1.0.0",
            "scene_id": scene_id,
            "video_id": video_id,
            "scene_idx": scene_idx,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame_exclusive,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "segmentation_provenance": _provenance(
                str(scene_row.get("detector") or "transnetv2_pytorch"),
                str(scene_row.get("stage_signature") or "")[:16] or None,
                threshold=scene_row.get("detector_threshold"),
                boundary_repaired=scene_row.get("boundary_repaired"),
            ),
            "keyframes": keyframes,
            # Cố ý KHÔNG có "captions": xem quyết định 2 ở đầu file.
            "asr_segments": _asr_segments_for_scene(
                scene_id=scene_id,
                scene_start_sec=start_sec,
                scene_end_sec=end_sec,
                segment_ids=list(evidence_row.get("asr_segment_ids") or []),
                by_id=asr_by_id,
                report=report,
            ),
            "keywords": [],
            "action_tags": [],
            "extensions": {"pack": {"scene_id": pack_scene_id, "pack_version": pack_version}},
        }
        emit("scenes", scene)
        for keyframe in keyframes:
            emit("keyframes", keyframe)
        scenes_written += 1
        keyframes_written += len(keyframes)

    if scenes_written == 0:
        return False

    report.scenes_written += scenes_written
    report.keyframes_written += keyframes_written

    frame_count = max(max_frame, max(mapping_by_frame) + 1 if mapping_by_frame else 0)
    duration = float(asr_status.get("video_duration_sec") or 0.0) or frame_count / fps
    emit(
        "videos",
        {
            "schema_version": "1.0.0",
            "video_id": video_id,
            "source_path": f"raw/videos/{video_id}.mp4",
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
            "width": frame_size[0],
            "height": frame_size[1],
            "audio_present": bool(asr_status.get("segment_count")),
            "probe_provenance": _provenance(
                f"competition_pack_v{pack_version}:probe", None, frame_size_source=frame_size_source
            ),
            "extensions": {
                "pack": {
                    "asr_status": asr_status.get("status"),
                    "scene_count": video_manifest.get("scene_count"),
                    "keyframe_count": video_manifest.get("keyframe_count"),
                    "embedding_count": video_manifest.get("embedding_count"),
                }
            },
        },
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", required=True, help="file .zip hoac thu muc pack da giai nen")
    parser.add_argument("--out", default="storage/exports_competition")
    parser.add_argument("--data-root", default="storage")
    parser.add_argument(
        "--vectors-dir",
        default="processed/embeddings_pack",
        help="thu muc chua .npy, TUONG DOI voi --data-root (di vao vector_uri)",
    )
    parser.add_argument("--batch", action="append", help="chi lay batch nay, vd L22 (lap lai duoc)")
    parser.add_argument("--video", action="append", help="chi lay video nay (lap lai duoc)")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument(
        "--merge-embeddings-from",
        help=(
            "export co san (thu muc hoac keyframes.jsonl) de BU vector vao cho pack "
            "khong co. Can cho L21: pack chi co 43/7790 vector o batch do, ma toan bo "
            "bo gold nam tren L21_V001..V003"
        ),
    )
    parser.add_argument(
        "--assume-frame-size",
        default="1280x720",
        help="kich thuoc dung khi ANH CHUA CO tren dia; co anh thi do that",
    )
    parser.add_argument(
        "--keep-hsv",
        action="store_true",
        help="chep hsv_features vao extensions (nang them ~100MB, pack van con ban goc)",
    )
    parser.add_argument("--dry-run", action="store_true", help="chi dem, khong ghi file")
    arguments = parser.parse_args()

    report = convert(arguments)

    print()
    print(f"  video trong pack           {report.videos_in_pack}")
    print(f"  video da ghi               {report.videos_written}")
    print(f"  scene da ghi               {report.scenes_written}")
    print(f"  scene bo (khong keyframe)  {report.scenes_dropped_no_keyframe}")
    print(f"  keyframe da ghi            {report.keyframes_written}")
    print(f"    co caption               {report.keyframes_with_caption}")
    print(f"    co OCR                   {report.keyframes_with_ocr}")
    print(f"    co vector                {report.keyframes_with_vector}")
    print(f"      trong do bu tu export  {report.keyframes_vector_from_donor}")
    print(f"    co ANH tren dia          {report.keyframes_image_on_disk}")
    print(f"  doan ASR da ghi            {report.asr_segments_written}")
    print(f"  doan ASR bo qua            {report.asr_segments_skipped}")
    print(f"  scene phai noi bien        {report.scene_bounds_widened} (toi da {report.max_bounds_widening_sec:.4f}s)")
    print(f"  file vector da chep        {report.vector_files_copied} ({report.vector_bytes_copied/1e6:.1f} MB)")
    print(f"  nguon kich thuoc frame     {dict(report.frame_size_source)}")
    for warning in report.warnings:
        print(f"  CANH BAO: {warning}")

    if report.keyframes_written and report.keyframes_image_on_disk == 0:
        print()
        print("  LUU Y: khong mot keyframe nao co anh tren dia. Hệ van chay va van xep")
        print("  hang duoc (dense/caption/ASR khong can anh), nhung UI se khong hien")
        print("  duoc gi va VLM rerank khong dung duoc. Xem README_BRANCH.md §4.1.")


if __name__ == "__main__":
    main()
