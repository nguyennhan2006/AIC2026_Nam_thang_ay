"""Optional production GPU engine loaded only on the Vast/Kaggle worker."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import os
import re
from pathlib import Path


_CAPTION_PROMPT = (
    "Mô tả khách quan, cụ thể bằng tiếng Việt (1-2 câu) những gì nhìn thấy trong khung hình "
    "video này: người, hành động, bối cảnh, vật thể nổi bật. Chỉ trả lời phần mô tả, không "
    "thêm giải thích hay tiêu đề."
)

_OCR_PROMPT = (
    "Tìm và đọc TOÀN BỘ chữ/văn bản xuất hiện rõ trong ảnh (biển hiệu, banner, phụ đề, chữ "
    "trên giấy tờ, tiêu đề...), giữ nguyên chính xác từng ký tự kể cả dấu tiếng Việt và số. "
    "Trả lời DUY NHẤT một JSON array, mỗi phần tử dạng "
    '{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}, toạ độ là pixel thực trên ảnh gốc. '
    "Nếu ảnh không có chữ nào, trả về []."
)


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


class TransformersGpuEngine:
    """Lazy model registry: Qwen2.5-VL (caption + semantic OCR), OWLv2, CLIP and Whisper."""

    def __init__(self) -> None:
        self.device = int(os.getenv("AIC_GPU_DEVICE", "0"))
        self._qwen = self._object = self._clip = self._asr = None

    @staticmethod
    def _image(request):
        from PIL import Image
        return Image.open(BytesIO(base64.b64decode(request.image_base64))).convert("RGB")

    def _load_qwen(self):
        if self._qwen is None:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            name = os.getenv("AIC_CAPTION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            cuda = torch.cuda.is_available()
            # float16 (not bf16) for broad GPU compat (e.g. Kaggle T4 has no fast bf16);
            # device_map="auto" shards across all visible GPUs (e.g. Kaggle T4x2 ~32GB combined).
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                name, torch_dtype=torch.float16 if cuda else torch.float32,
                device_map="auto" if cuda else "cpu",
            )
            processor = AutoProcessor.from_pretrained(name)
            self._qwen = (model, processor, name)
        return self._qwen

    def _qwen_generate(self, image, prompt: str, max_new_tokens: int) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        model, processor, _ = self._load_qwen()
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _caption_sync(self, request) -> dict:
        image = self._image(request)
        text = self._qwen_generate(image, _CAPTION_PROMPT, int(os.getenv("AIC_CAPTION_MAX_TOKENS", "120"))).strip()
        return {"captions": [{"text": text, "language": "vi", "confidence": None}]}

    def _ocr_sync(self, request) -> dict:
        image = self._image(request)
        width, height = image.size
        raw = _strip_code_fence(self._qwen_generate(image, _OCR_PROMPT, 512).strip())
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = []
        instances = []
        for item in items if isinstance(items, list) else []:
            text = str(item.get("text_content", "")).strip()
            box = item.get("bbox_2d")
            if not text or not (isinstance(box, list) and len(box) == 4):
                continue
            x1, y1, x2, y2 = box
            instances.append({
                "text": text, "normalized_text": text.casefold(), "language": None, "confidence": 0.75,
                "bbox": {
                    "x1": max(0.0, min(1.0, x1 / width)), "y1": max(0.0, min(1.0, y1 / height)),
                    "x2": max(0.0, min(1.0, x2 / width)), "y2": max(0.0, min(1.0, y2 / height)),
                },
            })
        return {"instances": instances}

    def _object_sync(self, request) -> dict:
        from transformers import pipeline
        labels = request.candidate_labels or ["person", "vehicle", "building", "animal", "sign", "food"]
        if self._object is None:
            self._object = pipeline("zero-shot-object-detection", model=os.getenv("AIC_OBJECT_MODEL", "google/owlv2-base-patch16-ensemble"), device=self.device)
        image = self._image(request)
        width, height = image.size
        rows = self._object(image, candidate_labels=labels, threshold=float(os.getenv("AIC_OBJECT_THRESHOLD", "0.15")))
        objects = []
        for row in rows:
            box = row["box"]
            x1 = max(0, min(width, box.get("xmin", 0)))
            y1 = max(0, min(height, box.get("ymin", 0)))
            x2 = max(0, min(width, box.get("xmax", width)))
            y2 = max(0, min(height, box.get("ymax", height)))
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append({"label": row["label"], "confidence": float(row["score"]), "bbox": {"x1": x1/width, "y1": y1/height, "x2": x2/width, "y2": y2/height}, "attributes": {"label_source": "caption+fallback"}})
        return {"objects": objects}

    def _image_embedding_sync(self, request) -> dict:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if self._clip is None:
            name = os.getenv("AIC_EMBEDDING_MODEL", "openai/clip-vit-large-patch14")
            self._clip = (CLIPModel.from_pretrained(name).to(self.device), CLIPProcessor.from_pretrained(name), name)
        model, processor, name = self._clip
        inputs = processor(images=self._image(request), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vector = model.get_image_features(**inputs)[0]
            vector = vector / vector.norm(p=2)
        values = vector.detach().cpu().float().tolist()
        return {"vector": values, "dimension": len(values), "model": name}

    def _text_embedding_sync(self, text: str) -> dict:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if self._clip is None:
            name = os.getenv("AIC_EMBEDDING_MODEL", "openai/clip-vit-large-patch14")
            self._clip = (CLIPModel.from_pretrained(name).to(self.device), CLIPProcessor.from_pretrained(name), name)
        model, processor, name = self._clip
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            vector = model.get_text_features(**inputs)[0]
            vector = vector / vector.norm(p=2)
        values = vector.detach().cpu().float().tolist()
        return {"vector": values, "dimension": len(values), "model": name}

    def _asr_sync(self, uri: str) -> dict:
        from transformers import pipeline
        root = Path(os.getenv("AIC_DATA_ROOT", "storage")).resolve()
        target = (root / uri).resolve()
        if root != target and root not in target.parents:
            raise ValueError("video_uri escapes AIC_DATA_ROOT")
        if self._asr is None:
            self._asr = pipeline("automatic-speech-recognition", model=os.getenv("AIC_ASR_MODEL", "openai/whisper-large-v3-turbo"), device=self.device)
        result = self._asr(str(target), return_timestamps=True, chunk_length_s=30)
        segments = []
        for chunk in result.get("chunks", []):
            start, end = chunk.get("timestamp", (None, None))
            if start is not None and end is not None and end > start:
                segments.append({"start_sec": float(start), "end_sec": float(end), "text": chunk["text"].strip(), "language": None, "confidence": None})
        return {"segments": segments}

    async def image(self, task: str, request) -> dict:
        functions = {"caption": self._caption_sync, "ocr": self._ocr_sync, "object": self._object_sync, "embedding": self._image_embedding_sync}
        return await asyncio.to_thread(functions[task], request)

    async def text_embedding(self, text: str) -> dict:
        return await asyncio.to_thread(self._text_embedding_sync, text)

    async def video(self, task: str, uri: str) -> dict:
        return await asyncio.to_thread(self._asr_sync, uri)
