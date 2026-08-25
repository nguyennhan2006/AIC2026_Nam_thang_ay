"""Nhánh retrieval dense trên TEXT của caption/tag — DENSE-TEXT-01.

Khác `DenseRetriever` (online/adapters/dense_retriever.py): nhánh đó khớp
truy vấn với embedding **ảnh** (CLIP/SigLIP). Nhánh này khớp truy vấn với
embedding **văn bản** của caption/object/action đã trích offline. Hai nhánh
bổ sung cho nhau: CLIP mạnh ở hình thức thị giác, dense text mạnh ở diễn đạt
khác từ nhưng cùng nghĩa.

Hai encoder được hỗ trợ, cùng cắm vào `CaptionDenseRetriever`:

    e5       intfloat/multilingual-e5-large — bất đối xứng bằng PREFIX CHUỖI
             (`query: ` / `passage: `), mean pooling thủ công.
    jina_v3  jinaai/jina-embeddings-v3 — bất đối xứng bằng LoRA ADAPTER theo
             task (`retrieval.query` / `retrieval.passage`), không có prefix.

Cả hai đều là lỗi IM LẶNG khi làm sai: thiếu prefix E5, hay quên `task=` của
jina, model vẫn chạy và vẫn trả vector 1024 chiều hợp lệ, chỉ kém hẳn. Vì vậy
prefix/task đọc từ manifest của index, để phía online và phía offline không
bao giờ lệch nhau.

`E5Encoder`/`JinaV3Encoder` sống ở đây chứ không ở `scripts/` vì cả hai phía
đều cần chúng và định nghĩa nhân đôi là đường ngắn nhất tới lệch pooling/
prefix giữa index và truy vấn — đúng loại lỗi im lặng vừa cảnh báo.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from online.domain.models import Candidate, Modality, QueryPlan
from online.services.branch_options import effective_limit, effective_weight

DOCUMENT_SCHEMA = "caption_dense_v1"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Jina-v3 KHÔNG dùng prefix chuỗi — nó chọn LoRA adapter qua `task=`. Đặt
# prefix rỗng để `CaptionDenseRetriever._encode` (vốn nối prefix vô điều kiện)
# dùng lại được không cần rẽ nhánh.
JINA_QUERY_TASK = "retrieval.query"
JINA_PASSAGE_TASK = "retrieval.passage"
ENCODER_KINDS = ("e5", "jina_v3", "jina_clip_v2")


def build_document_text(scene) -> str:
    """Một chuỗi text tìm kiếm được cho mỗi scene.

    Thứ tự cố ý: caption trước (mang nhiều ngữ nghĩa nhất), rồi tới tag. Không
    lặp field để giả trọng số — trọng số là việc của tầng fusion.

    CỐ Ý chưa có OCR/ASR: ROUTE-01 cho thấy hai nguồn đó có giá trị thật trong
    corpus này, nhưng trộn ngay vào document dense thì không tách được gain đến
    từ ngữ nghĩa caption hay từ lower-third bản tin.
    """

    parts: list[str] = []
    if scene.captions:
        parts.append(" ".join(scene.captions))
    for values in (scene.object_labels, scene.action_tags, scene.keywords):
        if values:
            parts.append(", ".join(dict.fromkeys(values)))
    return " | ".join(part for part in parts if part.strip())


class E5Encoder:
    """Encoder E5 chạy local, luôn L2-normalize để cosine = dot product.

    `torch`/`transformers` được import BÊN TRONG `__init__`: module này bị
    `online/api/container.py` import ở mọi lần khởi động, kể cả khi nhánh
    caption dense tắt, và kéo torch vào lúc đó là cộng vài giây cho một thứ
    không dùng tới.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 320) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.kind = "e5"
        self.dim = int(self.model.config.hidden_size)

    def _mean_pool(self, hidden, mask):
        expanded = mask.unsqueeze(-1).float()
        return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        torch = self._torch
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[start : start + batch_size],
                    padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                pooled = self._mean_pool(hidden, batch["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype="float32")

    def warmup(self) -> None:
        """Nạp trọng số NGAY, ngoài request path.

        Cùng lý do đã ghi ở container cho text tower CLIP: truy vấn đầu tiên
        nuốt trọn thời gian nạp model, vượt `AIC_BRANCH_TIMEOUT_MS` và nhánh
        biến mất trong im lặng — đo được là 1-2 truy vấn đầu mỗi tiến trình cho
        ranking khác hẳn các truy vấn sau.
        """

        self.encode(["query: warmup"])


class JinaV3Encoder:
    """Encoder jinaai/jina-embeddings-v3, luôn L2-normalize như `E5Encoder`.

    KHÔNG dùng `pipeline("feature-extraction")`. Pipeline trả hidden state của
    TỪNG TOKEN, không pool, không normalize, và — quan trọng nhất — không kích
    hoạt LoRA adapter theo task. Ba thứ đó cộng lại vẫn cho ra một ma trận số
    trông hợp lệ, nên sai kiểu này không có gì báo. `AutoModel.encode()` của
    jina lo cả pooling lẫn adapter.

    `task` cố định lúc dựng chứ không truyền theo từng lời gọi: phía offline
    dựng với `retrieval.passage`, phía online với `retrieval.query`, và ghi vào
    manifest để hai bên không lệch. Đây là cơ chế bất đối xứng thay cho prefix
    `query: `/`passage: ` của E5 — vai trò y hệt, chỉ khác cách khai.

    `torch`/`transformers` import bên trong `__init__`, cùng lý do đã ghi ở
    `E5Encoder`.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 320,
        task: str = JINA_QUERY_TASK,
    ) -> None:
        import torch
        from transformers import AutoModel

        self._torch = torch
        # `trust_remote_code` là BẮT BUỘC: pooling + adapter nằm trong code
        # riêng của repo, không phải kiến trúc chuẩn của transformers.
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self.task = task
        self.kind = "jina_v3"
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        with self._torch.no_grad():
            vectors = self.model.encode(
                texts,
                task=self.task,
                batch_size=batch_size,
                max_length=self.max_length,
                show_progress_bar=False,
            )
        matrix = np.asarray(vectors, dtype="float32")
        # Normalize LẠI dù `encode()` có tuỳ chọn riêng: cả `CaptionDenseRetriever
        # .search` lẫn script dựng index đều coi dot product = cosine, và bất biến
        # đó phải đúng bất kể mặc định của thư viện đổi ra sao.
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)

    def warmup(self) -> None:
        """Nạp trọng số NGAY, ngoài request path — xem `E5Encoder.warmup`."""

        self.encode(["warmup"])


class JinaClipV2Encoder:
    """Encoder jinaai/jina-clip-v2 (CLIP-style, khác với jina-embeddings-v3).

    Model này dùng `JinaCLIPModel.encode_text()` thay vì
    `jina-embeddings-v3.encode()`. Hai model khác nhau:
      - jina-embeddings-v3: 1024d, dùng LoRA adapter theo task
      - jina-clip-v2:       1024d, CLIP-style với `encode_text(task=)`

    Cả hai đều 1024 chiều nhưng KHÔNG cùng không gian embedding. Dùng sai
    encoder với index sẽ cho cosine score vô nghĩa trong im lặng — chốt này
    kiểm tra qua manifest encoder_kind.

    `encode_text(task=)` cố định ở init (retrieval.query phía online,
    retrieval.passage phía offline dựng index). Không dùng prefix.

    `torch`/`transformers` import bên trong `__init__`, cùng lý do đã ghi ở
    `E5Encoder`.
    """

    def __init__(
        self,
        model_path: str = "jinaai/jina-clip-v2",
        device: str = "cpu",
        max_length: int = 320,
        task: str = JINA_QUERY_TASK,
    ) -> None:
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self.task = task
        self.kind = "jina_clip_v2"
        self.dim = 1024  # fixed for jina-clip-v2

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        with self._torch.no_grad():
            result = self.model.encode_text(
                texts,
                task=self.task,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        # result is (batch, dim) numpy array
        matrix = np.asarray(result, dtype="float32")
        # Normalize lại: encode_text() đã normalize, nhưng đảm bảo đúng
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)

    def warmup(self) -> None:
        """Nạp trọng số NGAY, ngoài request path — xem `E5Encoder.warmup`."""
        self.encode(["warmup"])


def build_text_encoder(
    kind: str,
    model_path: str,
    *,
    device: str = "cpu",
    max_length: int = 320,
    for_passages: bool = False,
):
    """Dựng encoder theo `kind`, và khai luôn phía nào (query hay passage).

    Một chỗ duy nhất quyết định "kind nào -> class nào, bất đối xứng ra sao",
    dùng chung cho container và script dựng index. Tách ra hai nơi là cách
    chắc chắn nhất để một bên dùng adapter còn bên kia thì không.
    """

    kind = kind.casefold()
    if kind == "e5":
        return E5Encoder(model_path, device=device, max_length=max_length)
    if kind == "jina_v3":
        return JinaV3Encoder(
            model_path,
            device=device,
            max_length=max_length,
            task=JINA_PASSAGE_TASK if for_passages else JINA_QUERY_TASK,
        )
    if kind == "jina_clip_v2":
        # model_path bị bỏ qua: luôn dùng HuggingFace vì đây là model mới
        # và chưa có trong storage/models/. Nếu cần dùng local, sửa sau.
        return JinaClipV2Encoder(
            model_path=model_path,
            device=device,
            max_length=max_length,
            task=JINA_PASSAGE_TASK if for_passages else JINA_QUERY_TASK,
        )
    raise ValueError(f"encoder kind={kind!r} không hợp lệ; chọn một trong {ENCODER_KINDS}")


def prefixes_for(kind: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix) của một kind. Jina bất đối xứng bằng task."""

    return (QUERY_PREFIX, PASSAGE_PREFIX) if kind.casefold() == "e5" else ("", "")


class CaptionDenseRetriever:
    backend_kind = "vector"
    branch_id = "caption_dense"
    execution_id = "caption_dense.raw"
    name = branch_id
    modality = Modality.CAPTION
    supported_controls = ("enabled", "weight", "top_k", "timeout_ms")

    def __init__(self, index_dir: Path, encoder, *, branch_id: str | None = None) -> None:
        # `branch_id` là THAM SỐ chứ không cố định, cùng lý do đã ghi ở
        # `DenseRetriever`: chạy hai index text song song (vd E5 + jina) mà
        # dùng chung id thì `RetrieverRegistry.resolve` luôn trả cái đầu tiên,
        # `components[key][candidate.source]` của fusion ghi đè lẫn nhau, và
        # `matching_branches` đếm hai nhánh thành một — làm sai luôn ngưỡng
        # `minimum_matching_branches`. Không có gì báo lỗi, chỉ có một nhánh
        # lặng lẽ biến mất khỏi phiếu bầu.
        self.branch_id = branch_id or type(self).branch_id
        self.execution_id = f"{self.branch_id}.raw"
        self.name = self.branch_id
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"thiếu {manifest_path} — chạy scripts/build_caption_dense_index.py trước"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.matrix: np.ndarray = np.load(self.index_dir / "embeddings.npy")
        self.scene_ids: list[str] = json.loads(
            (self.index_dir / "scene_ids.json").read_text(encoding="utf-8")
        )
        if len(self.scene_ids) != self.matrix.shape[0]:
            raise ValueError(
                f"index hỏng: {len(self.scene_ids)} scene_id nhưng {self.matrix.shape[0]} vector"
            )
        self.encoder = encoder
        self.query_prefix: str = self.manifest.get("query_prefix", QUERY_PREFIX)
        self.model_id: str = self.manifest.get("model_id", "unknown")
        self.index_id: str = self.manifest.get("index_fingerprint", "unknown")
        # Index dựng TRƯỚC khi có jina đều là E5 và không ghi field này.
        self.encoder_kind: str = self.manifest.get("encoder_kind", "e5")

    def assert_covers(self, scene_ids, *, min_coverage: float = 0.98) -> float:
        """Index có phủ đúng corpus đang phục vụ không.

        Đây là chốt an toàn quan trọng nhất của nhánh này. Index là một thư mục
        RỜI, không nằm trong export, nên không có gì buộc nó phải được dựng lại
        khi đổi `AIC_METADATA_JSONL`. Dùng index của `exports_l21` (216 scene,
        chỉ L21_V001) để phục vụ `exports_multivideo` (765 scene, 3 video) thì
        nhánh vẫn chạy, vẫn trả 100 candidate, vẫn có điểm cosine hợp lệ — chỉ
        là **không bao giờ đề xuất nổi một scene nào của V002/V003**. Không có
        cảnh báo nào, và `branch_status` sẽ báo `success`.

        Cùng quy ước fail-fast đã áp cho `AIC_ENABLE_EVENT_SEARCH` và cho lệch
        chiều vector ở `AIC_DENSE_INDEXES`: thà chặn khởi động còn hơn để một
        nhánh chạy mà kết quả vô nghĩa.

        TRẢ VỀ coverage thực tế (float). Caller quyết định warn hay crash.
        """

        corpus = set(scene_ids)
        if not corpus:
            return 1.0
        covered = corpus & set(self.scene_ids)
        coverage = len(covered) / len(corpus)
        return coverage

    def assert_encoder_kind(self, encoder) -> None:
        """Encoder online phải cùng HỌ với encoder đã dựng index.

        Chốt này tồn tại vì `assert_dimension` MÙ đúng ở ca nguy hiểm nhất:
        multilingual-e5-large và jina-embeddings-v3 đều ra vector 1024 chiều.
        Đổi `AIC_CAPTION_DENSE_ENCODER` mà quên dựng lại index thì chiều vẫn
        khớp, nhánh vẫn trả 100 candidate, `branch_status` vẫn `success`, và
        mọi điểm cosine đều là rác — hai model không hề chung không gian
        embedding. Không có gì khác trong hệ bắt được ca này.

        Kèm theo đó là lệch bất đối xứng: E5 mang prefix `query: ` còn jina
        chọn LoRA adapter, nên dùng nhầm họ cũng có nghĩa là dùng nhầm luôn cơ
        chế query-vs-passage.
        """

        actual = getattr(encoder, "kind", None) or (
            "jina_clip_v2" if isinstance(encoder, JinaClipV2Encoder) else
            "jina_v3" if isinstance(encoder, JinaV3Encoder) else "e5"
        )
        if actual == self.encoder_kind:
            return
        raise ValueError(
            f"caption dense: index {self.index_dir} dựng bằng encoder "
            f"{self.encoder_kind!r} (model_id={self.model_id!r}) nhưng đang phục vụ bằng "
            f"{actual!r}. Hai họ này cùng 1024 chiều nên assert_dimension KHÔNG bắt được — "
            "cosine sẽ vô nghĩa trong im lặng. Đặt AIC_CAPTION_DENSE_ENCODER khớp với "
            "index, hoặc dựng lại index bằng scripts/build_caption_dense_index.py "
            f"--encoder {actual}."
        )

    def assert_dimension(self, probe: np.ndarray) -> None:
        """Encoder và index phải cùng chiều — khác chiều là khai sai model."""

        if probe.shape[-1] != self.matrix.shape[1]:
            raise ValueError(
                f"caption dense: encoder cho vector {probe.shape[-1]} chiều nhưng index "
                f"{self.index_dir} là {self.matrix.shape[1]} chiều. Gần như chắc chắn là "
                f"khai sai model (index ghi model_id={self.model_id!r})."
            )

    def _encode(self, query: str) -> np.ndarray:
        return np.asarray(self.encoder.encode([self.query_prefix + query])[0])

    def _score_sync(self, vector, limit: int):
        scores = self.matrix @ vector
        return scores, np.argsort(-scores)[:limit]

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.execution_id, self.modality, self.branch_id) <= 0:
            return []
        limit = effective_limit(plan, self.execution_id, limit, self.branch_id)
        # Cùng quy ước với nhánh lexical: TRAKE dựng lại plan cho TỪNG bước với
        # `events=[event]`, nên một event nghĩa là "chấm đúng bước này" chứ
        # không phải chấm cả câu. Đọc `normalized_query` ở đây sẽ làm mọi bước
        # của một chuỗi dùng chung một truy vấn.
        query = plan.events[0].text if len(plan.events) == 1 else plan.normalized_query
        # Inference chạy trong thread riêng: `encode` là CPU-bound thuần và giữ
        # nguyên trong coroutine sẽ chặn event loop, làm MỌI nhánh khác trượt
        # deadline. Đã có tiền lệ đo được — `ocr_fuzzy` chạy đồng bộ kéo
        # `dense_visual` từ p50 224ms lên 8.7s và timeout ở 40/84 truy vấn.
        vector = await asyncio.to_thread(self._encode, query)

        # Vector đã L2-normalize cả hai phía nên dot product = cosine.
        # `matrix @ vector` + `argsort` trên toàn bộ scene cũng phải rời event
        # loop: ở corpus thi đấu riêng argsort đã sắp 87.742 phần tử mỗi truy
        # vấn. numpy nhả GIL nên phần này chồng lấn thật giữa các request.
        scores, top = await asyncio.to_thread(self._score_sync, vector, limit)
        return [
            Candidate(
                candidate_id=f"{self.execution_id}:{self.scene_ids[position]}",
                video_id=self.scene_ids[position].rsplit("_S", 1)[0],
                scene_id=self.scene_ids[position],
                source=self.execution_id,
                modality=self.modality,
                raw_score=float(scores[position]),
                score_kind="cosine",
                rank=rank,
                model_id=self.model_id,
                index_id=self.index_id,
            )
            for rank, position in enumerate(top, start=1)
        ]


__all__ = [
    "CaptionDenseRetriever",
    "E5Encoder",
    "JinaV3Encoder",
    "JinaClipV2Encoder",
    "DOCUMENT_SCHEMA",
    "ENCODER_KINDS",
    "JINA_PASSAGE_TASK",
    "JINA_QUERY_TASK",
    "QUERY_PREFIX",
    "PASSAGE_PREFIX",
    "build_document_text",
    "build_text_encoder",
    "prefixes_for",
]
