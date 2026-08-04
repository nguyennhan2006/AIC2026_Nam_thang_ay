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
            name = os.getenv("AIC_CAPTION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            # AIC_CAPTION_MODEL is a generic-looking name but this loader only ever
            # instantiates Qwen2_5_VLForConditionalGeneration — pointing it at a
            # different model family (BLIP, Qwen3-VL, Florence, ...) crashes on
            # from_pretrained with a confusing shape/config mismatch. Fail fast with
            # a clear message instead (checked before the heavy torch/transformers
            # import so this is testable without a GPU environment installed).
            # Qwen3-VL captioning is a separate code path (scripts/caption_qwen3vl.py,
            # OpenAI-compatible HTTP client), not this one.
            if "qwen2.5-vl" not in name.casefold() and "qwen2_5-vl" not in name.casefold():
                raise ValueError(
                    f"AIC_CAPTION_MODEL={name!r} does not look like a Qwen2.5-VL checkpoint. "
                    "TransformersGpuEngine._load_qwen only supports the Qwen2.5-VL family "
                    "(Qwen2_5_VLForConditionalGeneration). For Qwen3-VL captioning, use "
                    "scripts/caption_qwen3vl.py instead (a separate HTTP-based path)."
                )
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            revision = os.getenv("AIC_CAPTION_MODEL_REVISION") or None
            cuda = torch.cuda.is_available()
            # float16 (not bf16) for broad GPU compat (e.g. Kaggle T4 has no fast bf16);
            # device_map="auto" shards across all visible GPUs (e.g. Kaggle T4x2 ~32GB combined).
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                name, revision=revision, torch_dtype=torch.float16 if cuda else torch.float32,
                device_map="auto" if cuda else "cpu",
            )
            processor = AutoProcessor.from_pretrained(name, revision=revision)
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
            self._object = pipeline(
                "zero-shot-object-detection",
                model=os.getenv("AIC_OBJECT_MODEL", "google/owlv2-base-patch16-ensemble"),
                revision=os.getenv("AIC_OBJECT_MODEL_REVISION") or None,
                device=self.device,
            )
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

    # Baseline color_search (Search Mixing Console W1) — pure CPU (PIL+numpy, không
    # cần model/GPU). Hue bucket cố định, mean_hsv là trung bình số học đơn giản (không
    # phải circular mean) — đủ cho baseline, tinh chỉnh sau qua calibration, không đổi
    # contract ColorFeature (datasection/schemas/keyframe.py).
    _HUE_NAMES = (
        ("red", 0.0, 15.0), ("orange", 15.0, 45.0), ("yellow", 45.0, 65.0),
        ("green", 65.0, 170.0), ("cyan", 170.0, 200.0), ("blue", 200.0, 255.0),
        ("purple", 255.0, 290.0), ("pink", 290.0, 330.0), ("red", 330.0, 360.0),
    )

    @classmethod
    def _name_pixels(cls, hue, sat, val):
        """Trả về mảng tên màu (str) cho từng pixel: đen/trắng/xám theo value khi
        saturation thấp, ngược lại theo hue bucket."""
        import numpy as np
        names = np.empty(hue.shape, dtype=object)
        low_sat = sat < 0.15
        names[low_sat & (val < 0.2)] = "black"
        names[low_sat & (val >= 0.85)] = "white"
        names[low_sat & (val >= 0.2) & (val < 0.85)] = "gray"
        colored = ~low_sat
        for name, lo, hi in cls._HUE_NAMES:
            names[colored & (hue >= lo) & (hue < hi)] = name
        return names

    def _color_sync(self, request) -> dict:
        import numpy as np
        image = self._image(request)
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
        hue = hsv[..., 0].ravel() / 255.0 * 360.0
        sat = hsv[..., 1].ravel() / 255.0
        val = hsv[..., 2].ravel() / 255.0

        bins = int(os.getenv("AIC_COLOR_HIST_BINS", "16"))
        raw_hist, _ = np.histogram(hue, bins=bins, range=(0.0, 360.0))
        total = float(raw_hist.sum())
        hsv_histogram = (raw_hist / total).tolist() if total > 0 else [0.0] * bins

        names = self._name_pixels(hue, sat, val)
        unique, counts = np.unique(names, return_counts=True)
        pixel_count = float(names.size)
        dominant_colors = sorted(
            ({"name": str(name), "ratio": round(float(count) / pixel_count, 4)} for name, count in zip(unique, counts)),
            key=lambda item: -item["ratio"],
        )[:8]

        height = hsv.shape[0]
        band = max(1, height // 3)
        regions = {}
        for region_name, band_slice in (("upper", slice(0, band)), ("center", slice(band, 2 * band)), ("lower", slice(2 * band, height))):
            band_names = self._name_pixels(
                hsv[band_slice, :, 0].ravel() / 255.0 * 360.0,
                hsv[band_slice, :, 1].ravel() / 255.0,
                hsv[band_slice, :, 2].ravel() / 255.0,
            )
            if band_names.size == 0:
                continue
            band_unique, band_counts = np.unique(band_names, return_counts=True)
            top = str(band_unique[np.argmax(band_counts)])
            regions[region_name] = [top]

        return {
            "dominant_colors": dominant_colors,
            "hsv_histogram": hsv_histogram,
            "mean_hsv": [round(float(hue.mean()), 2), round(float(sat.mean()), 4), round(float(val.mean()), 4)],
            "regions": regions,
        }

    def _image_embedding_sync(self, request) -> dict:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if self._clip is None:
            name = os.getenv("AIC_EMBEDDING_MODEL", "openai/clip-vit-large-patch14")
            revision = os.getenv("AIC_EMBEDDING_MODEL_REVISION") or None
            self._clip = (
                CLIPModel.from_pretrained(name, revision=revision).to(self.device),
                CLIPProcessor.from_pretrained(name, revision=revision),
                name,
            )
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
            revision = os.getenv("AIC_EMBEDDING_MODEL_REVISION") or None
            self._clip = (
                CLIPModel.from_pretrained(name, revision=revision).to(self.device),
                CLIPProcessor.from_pretrained(name, revision=revision),
                name,
            )
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
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=os.getenv("AIC_ASR_MODEL", "openai/whisper-large-v3-turbo"),
                revision=os.getenv("AIC_ASR_MODEL_REVISION") or None,
                device=self.device,
            )
        result = self._asr(str(target), return_timestamps=True, chunk_length_s=30)
        segments = []
        for chunk in result.get("chunks", []):
            start, end = chunk.get("timestamp", (None, None))
            if start is not None and end is not None and end > start:
                segments.append({"start_sec": float(start), "end_sec": float(end), "text": chunk["text"].strip(), "language": None, "confidence": None})
        return {"segments": segments}

    async def image(self, task: str, request) -> dict:
        functions = {"caption": self._caption_sync, "ocr": self._ocr_sync, "object": self._object_sync, "embedding": self._image_embedding_sync, "color": self._color_sync}
        return await asyncio.to_thread(functions[task], request)

    async def text_embedding(self, text: str) -> dict:
        return await asyncio.to_thread(self._text_embedding_sync, text)

    async def video(self, task: str, uri: str) -> dict:
        return await asyncio.to_thread(self._asr_sync, uri)
