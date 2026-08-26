"""Local deterministic and remote production text encoders."""

from __future__ import annotations

import asyncio
import threading
import hashlib
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from online.errors import DependencyUnavailableError


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class HashingTextEncoder:
    """Dependency-free deterministic encoder for smoke tests, never a quality model."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    async def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_RE.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(item * item for item in vector))
        return [item / norm for item in vector] if norm else vector


# Ba họ model có API text tower KHÁC NHAU. Không trừu tượng hoá được bằng một
# lời gọi duy nhất, nên khai báo tường minh thay vì thử-và-bắt-lỗi lần lượt —
# thử lần lượt sẽ nuốt mất lỗi thật (sai đường dẫn, thiếu trust_remote_code) và
# biến nó thành "không họ nào khớp".
#
#   clip    CLIPModel.get_text_features(**processor(text=...))
#   siglip  SiglipModel.get_text_features(...)  — padding="max_length" BẮT BUỘC
#   jina    AutoModel(trust_remote_code=True).encode_text([...])
_ENCODER_KINDS = ("clip", "siglip", "jina")


def infer_encoder_kind(model_path: str) -> str:
    """Đoán họ model từ đường dẫn/tên repo. Không đoán được -> `clip`.

    Chỉ là mặc định tiện tay; `AIC_DENSE_INDEXES` luôn khai báo được tường minh
    khi tên thư mục không nói lên điều gì.
    """

    lowered = model_path.replace("\\", "/").casefold()
    if "jina" in lowered:
        return "jina"
    if "siglip" in lowered:
        return "siglip"
    return "clip"


class LocalTextEncoder:
    """Text tower chạy tại chỗ, CÙNG không gian embedding với vector ảnh.

    Hỗ trợ ba họ (`clip`/`siglip`/`jina`) để chạy nhiều index song song. Mỗi
    index phải dùng ĐÚNG model đã sinh vector ảnh của nó — trộn hai model là
    cosine vô nghĩa, và không có gì báo lỗi vì phép nhân vẫn ra số.

    GPU: Tự động phát hiện CUDA và dùng GPU nếu có, tăng tốc inference đáng kể.
    Mặc định auto-detect; đặt device="cpu" để ép CPU.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        kind: str | None = None,
        device: str | None = None,  # "cuda", "cpu", hoặc None (auto-detect)
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self.kind = (kind or infer_encoder_kind(model_path)).casefold()
        if self.kind not in _ENCODER_KINDS:
            raise ValueError(
                f"kind={self.kind!r} không hợp lệ; chọn một trong {_ENCODER_KINDS}"
            )
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

        # Auto-detect GPU
        import torch as _torch
        self._torch = _torch
        if device is None:
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        with self._lock:
            if self._model is not None:
                return self._model, self._processor
            # Dựng vào biến cục bộ trước: nếu bị huỷ/lỗi giữa chừng thì
            # `self._model` vẫn None và lần sau nạp lại từ đầu.
            device = self._device
            if self.kind == "jina":
                from transformers import AutoModel

                model = AutoModel.from_pretrained(
                    self.model_path, revision=self.revision, trust_remote_code=True
                ).to(device).eval()
                processor = None
            elif self.kind == "siglip":
                from transformers import AutoProcessor, SiglipModel

                model = SiglipModel.from_pretrained(
                    self.model_path, revision=self.revision
                ).to(device).eval()
                processor = AutoProcessor.from_pretrained(
                    self.model_path, revision=self.revision
                )
            else:
                from transformers import CLIPModel, CLIPProcessor

                model = CLIPModel.from_pretrained(
                    self.model_path, revision=self.revision
                ).to(device).eval()
                processor = CLIPProcessor.from_pretrained(
                    self.model_path, revision=self.revision
                )
            self._model, self._processor = model, processor
        return self._model, self._processor

    def warmup(self) -> None:
        """Nạp model NGAY, ngoài request path.

        Gọi ở lúc dựng container/service. Không gọi thì truy vấn đầu tiên sẽ
        nuốt trọn thời gian nạp và nhánh dense bị timeout bỏ qua."""

        self._encode_sync("warmup")

    async def encode(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode_sync, text)

    def _encode_sync(self, text: str) -> list[float]:
        torch = self._torch

        model, processor = self._load()
        with torch.inference_mode():
            if self.kind == "jina":
                # Jina tự lo tokenise; trả sẵn numpy đã chuẩn hoá.
                raw = model.encode_text([text])[0]
                return [float(value) for value in raw]
            if self.kind == "siglip":
                # SigLIP huấn luyện với chuỗi độ dài CỐ ĐỊNH; dùng
                # `padding=True` như CLIP sẽ cho vector lệch.
                inputs = processor(
                    text=[text], return_tensors="pt",
                    padding="max_length", truncation=True,
                )
            else:
                inputs = processor(
                    text=[text], return_tensors="pt", padding=True, truncation=True
                )
            vector = model.get_text_features(**inputs)[0]
            vector = vector / vector.norm(p=2)
        return vector.detach().cpu().float().tolist()


# Tên cũ, giữ để không phải sửa mọi chỗ đang gọi. `LocalTextEncoder` mặc định
# suy ra họ `clip` từ đường dẫn nên hành vi y hệt bản trước.
LocalClipTextEncoder = LocalTextEncoder


class RemoteTextEncoder:
    """HTTP adapter for a separately deployed CLIP/SigLIP text encoder."""

    def __init__(self, url: str, timeout_sec: float = 10.0, api_key: str | None = None) -> None:
        self.url = url
        self.timeout_sec = timeout_sec
        self.api_key = api_key

    async def encode(self, text: str) -> list[float]:
        def request_vector() -> list[float]:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = Request(
                self.url,
                data=json.dumps({"text": text}).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as exc:
                raise DependencyUnavailableError(
                    f"embedding service unavailable: {exc}"
                ) from exc
            vector = payload.get("vector")
            if not isinstance(vector, list) or not vector:
                raise DependencyUnavailableError(
                    "embedding service must return {'vector': [float, ...]}"
                )
            return [float(item) for item in vector]

        return await asyncio.to_thread(request_vector)
