"""In-memory cosine baseline and Qdrant Query API adapter."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from online.domain.models import Candidate, Modality, SearchFilters
from online.errors import DependencyUnavailableError

# numpy nằm trong extra `faiss`/`all` của pyproject.toml, KHÔNG phải core
# dependency (core chỉ có pydantic). Import mềm để bản cài tối thiểu vẫn chạy;
# đường Python thuần bên dưới cho ra cùng thứ hạng, chỉ chậm hơn vài trăm lần.
try:
    import numpy as _np
except ImportError:  # pragma: no cover - phụ thuộc môi trường cài đặt
    _np = None


def qdrant_point_id(entity_id: str) -> str:
    """Map business IDs to Qdrant-compatible deterministic UUID strings."""

    return str(uuid5(NAMESPACE_URL, f"aic2026:v1:{entity_id}"))


def candidate_from_payload(
    payload: dict,
    *,
    source: str,
    score: float,
    rank: int,
    index_id: str | None = None,
    model_id: str | None = None,
) -> Candidate | None:
    """Dựng Candidate từ payload của vector store, giữ `frame_idx` nếu có.

    Một collection có thể ở mức scene (`aic_scenes_*`) hoặc mức frame
    (`aic_frames_*`, PR-02). Payload frame mang thêm `frame_idx`/`keyframe_id`,
    và khi đó candidate được đánh dấu `entity_type="frame"` để tầng trên biết
    nó đã neo đúng tọa độ submission chứ không phải cả một scene.
    """

    video_id = payload.get("video_id")
    scene_id = payload.get("scene_id")
    if not video_id or not scene_id:
        return None
    frame_idx = payload.get("frame_idx")
    is_frame = frame_idx is not None and payload.get("keyframe_id")
    return Candidate(
        candidate_id=str(payload.get("keyframe_id") or scene_id),
        entity_type="frame" if is_frame else "scene",
        scene_id=str(scene_id),
        video_id=str(video_id),
        event_id=payload.get("event_id"),
        frame_idx=int(frame_idx) if frame_idx is not None else None,
        timestamp_sec=payload.get("timestamp_sec"),
        start_frame=payload.get("start_frame"),
        end_frame=payload.get("end_frame"),
        source=source,
        modality=Modality.VISUAL,
        raw_score=float(score),
        score_kind="cosine",
        rank=rank,
        model_id=model_id,
        index_id=index_id,
        payload=payload,
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class VectorStoreTooLargeError(RuntimeError):
    """Corpus vượt ngưỡng mà đường Python thuần còn kịp trả lời."""


class InMemoryVectorStore:
    """Quét tuyến tính toàn bộ vector; có numpy thì bằng MỘT phép nhân ma trận.

    Bản trước gọi `cosine()` Python thuần cho từng hàng, và còn tính lại chuẩn
    của CẢ HAI vector ở mỗi hàng. Đo được trên máy này: 69 µs/hàng, tức

        855 vector   ->  0.06 s   (không ai để ý)
        250 000      -> 17.3 s    (`AIC_BRANCH_TIMEOUT_MS` mặc định 8000)

    Nghĩa là ở quy mô thi đấu, `dense_visual` — nhánh mạnh nhất của hệ — sẽ bị
    cắt vì quá hạn và trả rỗng, mà `branch_status` báo `empty` chứ không báo
    `broken`, còn `empty` là trạng thái hợp lệ. Đúng kiểu hỏng trong im lặng đã
    dính ba lần (docs/30 §9).

    Con số "quét 250k vector chỉ mất 6.4 ms" trong docs/30 §6 đo bằng
    `scripts/bench_ann.py` — script đó dùng numpy, đường chạy thật thì không.
    Kết luận "chưa cần ANN" vẫn đúng, nhưng chỉ sau khi có lớp này.

    Chuẩn hoá MỘT LẦN lúc dựng thay vì mỗi lần so: cosine của hai vector đã
    chuẩn hoá chính là tích vô hướng, nên tìm kiếm rút về `matrix @ query`.

    `float32` không mất mát so với nguồn: `scripts/embed_keyframes_local.py`
    ghi vector đã qua `.float()` sau khi chuẩn hoá, tức dữ liệu vốn là float32.
    Nó cũng cắt RAM 8 lần — 250k vector 768 chiều là 5.74 GB dưới dạng
    `list[float]` (24.1 KB/vector) so với 0.72 GB dưới dạng ma trận float32.
    """

    # Không có numpy mà corpus lớn hơn ngần này thì mỗi truy vấn mất hàng chục
    # giây. Thà chặn lúc khởi động với thông báo đọc được còn hơn để nhánh chết
    # âm thầm vì timeout giữa buổi thi. Ngưỡng đặt ở 20k vì tại đó đường Python
    # đã mất ~1.4 s/truy vấn — vẫn chạy nhưng đã đủ chậm để phải biết.
    PYTHON_FALLBACK_MAX_ROWS = 20_000

    def __init__(self, rows: list[tuple[str, str, list[float], dict]]) -> None:
        self._ids = [str(row[0]) for row in rows]
        self._video_ids = [str(row[1]) for row in rows]
        self._payloads = [row[3] for row in rows]
        self.dim = len(rows[0][2]) if rows else 0

        # Chỉ mục phụ dựng sẵn: lọc theo video là đường nóng của TRAKE (mỗi
        # bước một lần tìm, khoá trong `AIC_TRAKE_VIDEO_TOP_K=10` video), và
        # quét cả corpus rồi vứt 99.9% là lãng phí thấy rõ.
        self._video_rows: dict[str, list[int]] = {}
        for index, video_id in enumerate(self._video_ids):
            self._video_rows.setdefault(video_id, []).append(index)
        self._has_ocr = [bool(payload.get("has_ocr")) for payload in self._payloads]
        self._has_asr = [bool(payload.get("has_asr")) for payload in self._payloads]

        if _np is not None:
            self.backend_impl = "numpy"
            matrix = _np.asarray([row[2] for row in rows], dtype=_np.float32).reshape(
                len(rows), self.dim
            )
            norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
            # Vector 0 (gặp ở fixture và ở scene không có text cho
            # HashingTextEncoder) phải cho điểm 0, không phải NaN — đúng như
            # `cosine()` cũ trả 0.0 khi một trong hai chuẩn bằng 0.
            self._matrix = _np.divide(matrix, norms, out=_np.zeros_like(matrix), where=norms > 0)
            self._vectors: list[list[float]] | None = None
        else:
            if len(rows) > self.PYTHON_FALLBACK_MAX_ROWS:
                raise VectorStoreTooLargeError(
                    f"{len(rows)} vector nhưng không có numpy: mỗi truy vấn sẽ mất "
                    f"~{len(rows) * 69e-6:.0f}s và bị AIC_BRANCH_TIMEOUT_MS cắt thành "
                    "kết quả rỗng. Cài numpy (`pip install .[all]`) hoặc dùng backend qdrant."
                )
            self.backend_impl = "python"
            self._matrix = None
            self._vectors = []
            for row in rows:
                norm = math.sqrt(sum(item * item for item in row[2]))
                self._vectors.append(
                    [item / norm for item in row[2]] if norm else [0.0] * self.dim
                )

    def __len__(self) -> int:
        return len(self._ids)

    async def health(self) -> bool:
        return True

    def _selected_rows(self, filters: SearchFilters) -> list[int] | None:
        """Chỉ số các hàng qua được bộ lọc; `None` = mọi hàng (khỏi vật chất hoá).

        Giữ nguyên ngữ nghĩa lọc của bản cũ, kể cả một chỗ khó chịu:
        `filters.scene_ids` được so với phần tử THỨ NHẤT của hàng, mà ở đường
        frame (`build_frame_vector_rows`) phần tử đó là `keyframe_id` chứ không
        phải `scene_id`. Sửa ở đây sẽ đổi kết quả mà không ai yêu cầu — ghi lại
        để sửa thành một thay đổi riêng, có đo.
        """

        if filters.video_ids:
            selected: list[int] = []
            for video_id in set(filters.video_ids):
                selected.extend(self._video_rows.get(video_id, []))
            selected.sort()
        elif filters.scene_ids or filters.has_ocr is not None or filters.has_asr is not None:
            selected = list(range(len(self._ids)))
        else:
            return None

        if filters.scene_ids:
            allowed = set(filters.scene_ids)
            selected = [index for index in selected if self._ids[index] in allowed]
        if filters.has_ocr is not None:
            selected = [index for index in selected if self._has_ocr[index] == filters.has_ocr]
        if filters.has_asr is not None:
            selected = [index for index in selected if self._has_asr[index] == filters.has_asr]
        return selected

    def _score(self, vector: Sequence[float], selected: list[int] | None):
        """Điểm cosine của các hàng đã chọn, cùng thứ tự với `selected`."""

        if _np is not None:
            query = _np.asarray(vector, dtype=_np.float32)
            norm = float(_np.linalg.norm(query))
            matrix = self._matrix if selected is None else self._matrix[selected]
            if norm == 0.0:
                return _np.zeros(len(matrix), dtype=_np.float32)
            return matrix @ (query / norm)
        norm = math.sqrt(sum(item * item for item in vector))
        unit = [item / norm for item in vector] if norm else None
        indices = range(len(self._ids)) if selected is None else selected
        if unit is None:
            return [0.0] * len(list(indices))
        return [
            sum(a * b for a, b in zip(unit, self._vectors[index], strict=True))
            for index in indices
        ]

    def _top_indices(self, scores, selected: list[int] | None, limit: int) -> list[int]:
        """`limit` hàng điểm cao nhất, hoà thì id nhỏ hơn đứng trước.

        Sắp toàn bộ 250k hàng mất ~63 ms chỉ để lấy 100 hàng đầu, nên cắt bằng
        `argpartition` (2.8 ms) rồi mới sắp đúng trên tập nhỏ. Nhưng
        `argpartition` KHÔNG ổn định: khi nhiều hàng cùng điểm ở đúng ranh giới
        top-k, nó chọn tuỳ ý, còn bản cũ luôn chọn theo id tăng dần. Vì vậy nới
        tập ứng viên xuống hết mọi hàng bằng điểm ranh giới trước khi sắp —
        điểm hoà nhau có thật ở nhánh `lexical_hash_fallback`.
        """

        count = len(scores)
        if limit <= 0 or count == 0:
            return []
        rows = list(range(count)) if selected is None else selected
        if _np is not None and count > limit:
            boundary = float(_np.partition(scores, count - limit)[count - limit])
            pool = _np.flatnonzero(scores >= boundary).tolist()
        else:
            pool = list(range(count))
        pool.sort(key=lambda position: (-float(scores[position]), self._ids[rows[position]]))
        return pool[:limit]

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[Candidate]:
        selected = self._selected_rows(filters)
        if selected is not None and not selected:
            return []
        scores = self._score(vector, selected)
        rows = list(range(len(self._ids))) if selected is None else selected
        candidates: list[Candidate] = []
        for rank, position in enumerate(self._top_indices(scores, selected, limit), start=1):
            candidate = candidate_from_payload(
                self._payloads[rows[position]],
                source="local_dense",
                score=float(scores[position]),
                rank=rank,
                index_id="inmemory_hashing_v1",
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates


def _qdrant_filter(filters: SearchFilters) -> dict | None:
    must: list[dict] = []
    if filters.video_ids:
        must.append({"key": "video_id", "match": {"any": filters.video_ids}})
    if filters.scene_ids:
        must.append({"key": "scene_id", "match": {"any": filters.scene_ids}})
    if filters.has_ocr is not None:
        must.append({"key": "has_ocr", "match": {"value": filters.has_ocr}})
    if filters.has_asr is not None:
        must.append({"key": "has_asr", "match": {"value": filters.has_asr}})
    if filters.start_sec_gte is not None:
        must.append({"key": "start_sec", "range": {"gte": filters.start_sec_gte}})
    if filters.end_sec_lte is not None:
        must.append({"key": "end_sec", "range": {"lte": filters.end_sec_lte}})
    return {"must": must} if must else None


class QdrantVectorStore:
    """Minimal stdlib HTTP adapter for Qdrant's universal Query API."""

    def __init__(
        self,
        url: str,
        collection: str,
        vector_name: str,
        *,
        api_key: str | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.collection = collection
        self.endpoint = f"{self.base_url}/collections/{collection}/points/query"
        self.vector_name = vector_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    async def health(self) -> bool:
        def perform() -> bool:
            headers = {"api-key": self.api_key} if self.api_key else {}
            request = Request(f"{self.base_url}/collections/{self.collection}", headers=headers, method="GET")
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    return response.status == 200 and payload.get("status") == "ok"
            except (HTTPError, URLError, TimeoutError):
                return False
        return await asyncio.to_thread(perform)

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[Candidate]:
        body: dict = {
            "query": list(vector),
            "using": self.vector_name,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        query_filter = _qdrant_filter(filters)
        if query_filter:
            body["filter"] = query_filter

        def perform() -> dict:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["api-key"] = self.api_key
            request = Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as exc:
                raise DependencyUnavailableError(f"Qdrant unavailable: {exc}") from exc

        response = await asyncio.to_thread(perform)
        result = response.get("result", {})
        if isinstance(result, dict):
            points = result.get("points", [])
        elif isinstance(result, list):
            points = result
        else:
            points = []
        candidates: list[Candidate] = []
        for rank, point in enumerate(points, start=1):
            candidate = candidate_from_payload(
                point.get("payload") or {},
                source="qdrant_dense",
                score=float(point["score"]),
                rank=rank,
                index_id=f"{self.collection}:{self.vector_name}",
                model_id=self.vector_name,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates
