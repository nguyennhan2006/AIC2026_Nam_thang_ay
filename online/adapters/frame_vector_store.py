"""Vector rows mức FRAME cho dense_visual thật, xây từ export đã nạp.

Cố tình song song (không import) với `offline/indexing.py::frame_rows` —
`online/` và `offline/` không cross-import nhau ở bất kỳ chỗ nào khác trong
repo, giữ nguyên ranh giới đó ở đây.

`FrameEvidence` (online/domain/candidate.py) CHỦ Ý chỉ giữ `embedding_names`
(tên) chứ không giữ vị trí lưu vector thật — theo đúng comment tại đó "vector
thật nằm trong vector store". Vì vậy hàm này đọc lại MỘT LẦN `keyframes.jsonl`
thô, chỉ để lấy `embedding_refs[].storage_locations`; phần còn lại của payload
(scene bounds, event_id, has_ocr/has_asr...) lấy từ repository đã nạp sẵn,
không parse lại toàn bộ export.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Sequence

from online.adapters.json_metadata import JsonlSceneRepository

# Ma trận .npy đang mở, khoá theo đường dẫn. Nhỏ có chủ ý: export sắp xếp theo
# video nên vòng lặp dưới đây chỉ chạm 1-2 file cùng lúc; giữ nhiều hơn chỉ tốn
# handle chứ không tránh thêm lần mở nào.
_MATRIX_CACHE_SIZE = 4
_matrix_cache: "OrderedDict[str, Any]" = OrderedDict()


def _read_npy_row(path: Path, row: int) -> Any:
    """Một hàng của ma trận .npy nhiều vector, mở bằng mmap.

    Pack thi đấu gom vector theo VIDEO (`dense/vectors/L21_V001.npy`) thay vì
    một file cho mỗi keyframe. Ở quy mô 168.960 keyframe thì đó là khác biệt
    giữa 873 lần mở file và 168.960 lần — trên NTFS, khác biệt đó tính bằng
    phút ở mỗi lần khởi động.

    `mmap_mode="r"` để cả ma trận không bị nạp vào RAM chỉ vì cần một hàng;
    trang nào đọc tới thì OS mới nạp trang đó.
    """

    import numpy

    key = str(path)
    matrix = _matrix_cache.get(key)
    if matrix is None:
        matrix = numpy.load(path, mmap_mode="r")
        _matrix_cache[key] = matrix
        while len(_matrix_cache) > _MATRIX_CACHE_SIZE:
            _matrix_cache.popitem(last=False)
    else:
        _matrix_cache.move_to_end(key)
    if row >= len(matrix):
        raise IndexError(
            f"{path.name}: xin hang {row} nhung file chi co {len(matrix)} vector. "
            "Export va thu muc vector lech nhau — sinh lai export."
        )
    # float32 vì `InMemoryVectorStore` xếp mọi thứ về float32; trả thẳng mảng
    # thay vì list[float] cắt RAM trung gian 6 lần (24.1 KB -> 4.1 KB mỗi vector,
    # tức 4.1 GB -> 0.7 GB ở 168.960 vector) và bỏ được vòng ép kiểu Python.
    return numpy.asarray(matrix[row], dtype=numpy.float32)


def _release_matrix_cache() -> None:
    """Đóng mọi mmap đang mở.

    Cache chỉ có ích TRONG lúc dựng rows: mỗi hàng đọc ra đã được sao thành
    mảng float32 riêng, xong việc thì không ai cần ma trận nữa. Trên Windows,
    giữ mmap là giữ KHOÁ file — server đang chạy sẽ chặn việc ghi đè chính
    export mà nó đang phục vụ, và `TemporaryDirectory` trong test không xoá
    được. Nên thả ngay khi dựng xong thay vì để bộ thu gom rác quyết định.
    """

    while _matrix_cache:
        _, matrix = _matrix_cache.popitem()
        handle = getattr(matrix, "_mmap", None)
        if handle is not None:
            handle.close()


def _read_vector_file(path: Path) -> Sequence[float]:
    """Vector tại `path`. Hậu tố `#<số>` chọn một hàng trong file nhiều vector."""

    raw = str(path)
    base, separator, fragment = raw.rpartition("#")
    if separator and fragment.isdigit():
        return _read_npy_row(Path(base), int(fragment))
    if path.suffix == ".npy":
        import numpy  # local: chỉ cần khi embedding lưu dạng .npy

        return numpy.asarray(numpy.load(path), dtype=numpy.float32)
    return [float(value) for value in json.loads(path.read_text(encoding="utf-8"))]


def _vector_uri_index(
    keyframes_path: Path, *, embedding_names: set[str] | None = None
) -> dict[tuple[str, int], dict[str, str]]:
    """`(video_id, frame_idx)` -> `{embedding_name: vector_uri}`, đọc theo dòng.

    Bản trước giữ NGUYÊN bản ghi keyframe đã parse của MỌI frame trong một dict
    rồi mới duyệt. Với pack thi đấu (`keyframes.jsonl` 515 MB, 176 707 frame,
    696 738 instance object) đó là vài GB sống suốt lúc khởi động, trên máy chỉ
    còn ~2 GB trống — tức chết vì hết bộ nhớ, hoặc thrashing, đúng lúc khởi
    động server trước giờ thi.

    Thứ duy nhất cần từ file thô là `vector_uri`. Giữ mỗi chuỗi đó thì chỉ tốn
    ~40 MB, và caption/object/color của cùng bản ghi được thả ngay sau khi
    `json.loads` trả về.

    Giữ nguyên luật chọn của bản cũ: theo THỨ TỰ `embedding_refs`, mỗi tên lấy
    `storage_location` dạng file ĐẦU TIÊN. Dict giữ thứ tự chèn nên "ref đầu
    tiên thắng" vẫn đúng ở phía caller.
    """

    index: dict[tuple[str, int], dict[str, str]] = {}
    with keyframes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            uris: dict[str, str] = {}
            for reference in raw.get("embedding_refs", []):
                name = str(reference.get("embedding_name") or "")
                if embedding_names is not None and name not in embedding_names:
                    continue
                for location in reference.get("storage_locations", []):
                    if location.get("backend") == "file" and location.get("vector_uri"):
                        uris.setdefault(name, str(location["vector_uri"]))
                        break
            if uris:
                index[(str(raw["video_id"]), int(raw["frame_idx"]))] = uris
    return index


async def _build_frame_vector_rows(
    repository: JsonlSceneRepository,
    data_root: Path,
    *,
    embedding_name: str | None = None,
) -> tuple[list[tuple[str, str, Sequence[float], dict[str, Any]]], bool]:
    """Trả `(rows, has_real_embeddings)`.

    `has_real_embeddings=False` khi KHÔNG có frame nào từng qua enrichment
    embedding (vd fixture demo nhỏ) — caller phải rơi về nhánh lexical fallback
    thay vì dựng một vector store rỗng.
    """

    scenes = await repository.all()
    if not any(frame.embedding_names for scene in scenes for frame in scene.keyframes):
        return [], False

    keyframes_path = repository.path.with_name("keyframes.jsonl")
    uri_by_key = _vector_uri_index(
        keyframes_path,
        embedding_names={embedding_name} if embedding_name else None,
    )

    rows: list[tuple[str, str, Sequence[float], dict[str, Any]]] = []
    for scene in scenes:
        for frame in scene.keyframes:
            uris = uri_by_key.get((frame.video_id, frame.frame_idx))
            if not uris:
                continue
            vector = _read_vector_file(data_root / next(iter(uris.values())))
            payload = {
                "entity_type": "keyframe",
                "keyframe_id": frame.keyframe_id,
                "scene_id": scene.scene_id,
                "video_id": scene.video_id,
                "event_id": scene.event_id,
                "frame_idx": frame.frame_idx,
                "timestamp_sec": frame.timestamp_sec,
                "image_path": frame.image_path,
                "start_frame": scene.start_frame,
                "end_frame": scene.end_frame_exclusive - 1,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "has_ocr": bool(frame.ocr_texts),
                "has_asr": bool(scene.asr_texts),
            }
            rows.append((frame.keyframe_id, scene.video_id, vector, payload))
    return rows, True




async def _build_frame_vector_rows_by_index(
    repository: JsonlSceneRepository,
    data_root: Path,
    *,
    embedding_names: list[str] | None = None,
) -> dict[str, list[tuple[str, str, Sequence[float], dict[str, Any]]]]:
    """Như `build_frame_vector_rows` nhưng TÁCH theo `embedding_name`.

    Một keyframe có thể mang nhiều `embedding_refs` (CLIP + Jina + SigLIP…).
    Gộp chúng vào một vector store là sai về bản chất: mỗi model có không gian
    riêng, cosine giữa hai không gian khác nhau chỉ là một con số vô nghĩa —
    và không có gì báo lỗi vì phép nhân vẫn chạy.

    Trả `{embedding_name: rows}`; caller dựng MỘT vector store và MỘT text
    encoder cho mỗi tên.

    `embedding_names=None` -> lấy mọi tên gặp trong export.
    """

    scenes = await repository.all()
    if not any(frame.embedding_names for scene in scenes for frame in scene.keyframes):
        return {}

    wanted = set(embedding_names) if embedding_names else None
    keyframes_path = repository.path.with_name("keyframes.jsonl")
    uri_by_key = _vector_uri_index(keyframes_path, embedding_names=wanted)

    out: dict[str, list[tuple[str, str, Sequence[float], dict[str, Any]]]] = {}
    for scene in scenes:
        for frame in scene.keyframes:
            uris = uri_by_key.get((frame.video_id, frame.frame_idx))
            if not uris:
                continue
            payload = {
                "entity_type": "keyframe",
                "keyframe_id": frame.keyframe_id,
                "scene_id": scene.scene_id,
                "video_id": scene.video_id,
                "event_id": scene.event_id,
                "frame_idx": frame.frame_idx,
                "timestamp_sec": frame.timestamp_sec,
                "image_path": frame.image_path,
                "start_frame": scene.start_frame,
                "end_frame": scene.end_frame_exclusive - 1,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "has_ocr": bool(frame.ocr_texts),
                "has_asr": bool(scene.asr_texts),
            }
            for name, uri in uris.items():
                if not name:
                    continue
                out.setdefault(name, []).append(
                    (
                        frame.keyframe_id,
                        scene.video_id,
                        _read_vector_file(data_root / uri),
                        dict(payload),
                    )
                )
    return out


async def build_frame_vector_rows(
    repository: JsonlSceneRepository,
    data_root: Path,
    *,
    embedding_name: str | None = None,
) -> tuple[list[tuple[str, str, Sequence[float], dict[str, Any]]], bool]:
    """`_build_frame_vector_rows` + đóng mmap, kể cả khi thân hàm vỡ giữa chừng.

    Tách vỏ ra thay vì bọc `try/finally` quanh vòng lặp: thân hàm là phần dễ đọc
    nhầm nhất của module này, và một tầng thụt đầu dòng nữa không đáng.
    """

    try:
        return await _build_frame_vector_rows(
            repository, data_root, embedding_name=embedding_name
        )
    finally:
        _release_matrix_cache()


async def build_frame_vector_rows_by_index(
    repository: JsonlSceneRepository,
    data_root: Path,
    *,
    embedding_names: list[str] | None = None,
) -> dict[str, list[tuple[str, str, Sequence[float], dict[str, Any]]]]:
    """Như trên, cho biến thể tách theo `embedding_name`."""

    try:
        return await _build_frame_vector_rows_by_index(
            repository, data_root, embedding_names=embedding_names
        )
    finally:
        _release_matrix_cache()


__all__ = ["build_frame_vector_rows", "build_frame_vector_rows_by_index"]
