"""Optional production GPU engine loaded only on the Vast worker."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path


class TransformersGpuEngine:
    """Lazy model registry for caption, OCR, OWLv2, CLIP and Whisper."""

    def __init__(self) -> None:
        self.device = int(os.getenv("AIC_GPU_DEVICE", "0"))
        self._caption = self._ocr = self._object = self._clip = self._asr = None

    @staticmethod
    def _image(request):
        from PIL import Image
        return Image.open(BytesIO(base64.b64decode(request.image_base64))).convert("RGB")

    def _caption_sync(self, request) -> dict:
        from transformers import pipeline
        if self._caption is None:
            self._caption = pipeline("image-to-text", model=os.getenv("AIC_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"), device=self.device)
        rows = self._caption(self._image(request), max_new_tokens=int(os.getenv("AIC_CAPTION_MAX_TOKENS", "80")))
        return {"captions": [{"text": row["generated_text"], "language": "en", "confidence": None} for row in rows]}

    def _ocr_sync(self, request) -> dict:
        import easyocr
        import numpy as np
        image = self._image(request)
        if self._ocr is None:
            self._ocr = easyocr.Reader(os.getenv("AIC_OCR_LANGUAGES", "vi,en").split(","), gpu=True)
        width, height = image.size
        instances = []
        for polygon, text, score in self._ocr.readtext(np.asarray(image)):
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            instances.append({
                "text": text, "normalized_text": text.casefold(), "language": None,
                "confidence": float(score),
                "bbox": {"x1": max(0, min(xs))/width, "y1": max(0, min(ys))/height, "x2": min(width, max(xs))/width, "y2": min(height, max(ys))/height},
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
