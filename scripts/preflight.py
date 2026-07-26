"""Fail-fast deployment checks before starting Online."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import tempfile

from datasection.exporter import verify_export
from online.adapters.vector_stores import QdrantVectorStore
from online.config import Settings

# 1x1 PNG, đủ để model thật (PIL.Image.open) mở được mà không cần ảnh video thật —
# warmup chỉ cần biết model/worker có TRẢ LỜI được không, không quan tâm nội dung.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
GPU_WARMUP_TASKS = ("caption", "ocr", "object", "embedding")


async def check_gpu_warmup() -> list[dict]:
    """Gọi thử từng loại inference (caption/ocr/object/embedding) qua provider đang cấu
    hình (mock hoặc remote worker thật) với 1 ảnh mẫu — model warmup gap ở doc 11 §4.G25/
    doc 14 Phase 5. Trả về list kết quả từng task; KHÔNG raise ở đây, để main() quyết định
    fail-fast theo policy "không silent degradation".
    """
    from offline.config import OfflineSettings
    from offline.providers import MockInferenceProvider, RemoteInferenceProvider

    settings = OfflineSettings.from_env()
    provider = (
        MockInferenceProvider()
        if settings.provider == "mock"
        else RemoteInferenceProvider(settings.gpu_url or "", settings.gpu_api_key, settings.timeout_sec, settings.retries)
    )

    results: list[dict] = []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(base64.b64decode(_TINY_PNG_B64))
        sample_path = Path(handle.name)
    try:
        for task in GPU_WARMUP_TASKS:
            entry = {"task": task, "provider": settings.provider}
            try:
                await asyncio.wait_for(provider.image(task, sample_path), timeout=settings.timeout_sec)
                entry["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 - báo lỗi rõ ràng, không nuốt exception
                entry["status"] = "error"
                entry["error"] = repr(exc)
            results.append(entry)
    finally:
        sample_path.unlink(missing_ok=True)
    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=Path("storage/exports"))
    parser.add_argument(
        "--check-gpu-warmup",
        action="store_true",
        help="Gọi thử caption/ocr/object/embedding qua provider offline hiện tại (mock hoặc worker thật) trước khi coi là ready",
    )
    args = parser.parse_args()
    manifest = verify_export(args.export_dir)
    settings = Settings.from_env()
    if settings.metadata_jsonl.resolve() != (args.export_dir / "scenes.jsonl").resolve():
        raise ValueError("AIC_METADATA_JSONL does not point at the verified export")
    if settings.backend == "qdrant":
        matches = [item for item in manifest.indexes if item.backend == "qdrant" and item.name == settings.qdrant_scene_collection and item.vector_name == settings.qdrant_vector_name]
        if not matches:
            raise ValueError("manifest does not publish the configured Qdrant index")
        store = QdrantVectorStore(settings.qdrant_url or "", settings.qdrant_scene_collection, settings.qdrant_vector_name, api_key=settings.qdrant_api_key, timeout_sec=settings.request_timeout_sec)
        if not await store.health():
            raise RuntimeError("Qdrant collection is not ready")

    report = {"status": "ready", "backend": settings.backend, "build_id": manifest.build_id, "videos": manifest.video_count, "scenes": manifest.scene_count, "keyframes": manifest.keyframe_count}
    if args.check_gpu_warmup:
        warmup = await check_gpu_warmup()
        report["gpu_warmup"] = warmup
        failed = [item["task"] for item in warmup if item["status"] != "ok"]
        if failed:
            print(json.dumps(report, ensure_ascii=False))
            raise RuntimeError(f"GPU warmup failed for task(s): {', '.join(failed)}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
