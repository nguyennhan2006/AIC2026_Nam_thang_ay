"""Create a canonical three-scene dataset used by local smoke tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from datasection.exporter import export_dataset
from datasection.schemas import (
    BoundingBox, CaptionRecord, DatasetManifest, Keyframe, ModelProvenance,
    OCRInstance, Scene, SceneCaptionRecord, Video,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "storage"


def provenance(name: str) -> ModelProvenance:
    return ModelProvenance(model_name=name, model_revision="demo", pipeline_version="aic-v1.0.0")


def main(verbose: bool = True) -> None:
    media = DATA / "processed/keyframes/L01_V001"
    media.mkdir(parents=True, exist_ok=True)
    labels = ["Người đang cào muối trên cánh đồng", "Đoàn người vẫy tay phía sau bảng chữ", "Nhóm người đứng trước căn nhà"]
    ocrs = [None, "Hẹn ngày gặp lại", "Gừng cay muối mặn xin đừng quên nhau"]
    frames = [150, 450, 750]
    scenes = []
    for index, (label, ocr, frame_idx) in enumerate(zip(labels, ocrs, frames, strict=True)):
        scene_id = f"L01_V001_S{index:04d}"
        image = media / f"frame_{frame_idx:06d}.svg"
        image.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"><rect width="100%" height="100%" fill="#16324f"/><text x="50" y="270" fill="white" font-size="30">{label}</text></svg>',
            encoding="utf-8",
        )
        ocr_items = [] if not ocr else [OCRInstance(text=ocr, normalized_text=ocr.casefold(), language="vi", confidence=0.99, bbox=BoundingBox(x1=.1, y1=.2, x2=.9, y2=.4), provenance=provenance("demo-ocr"))]
        frame = Keyframe(
            keyframe_id=f"{scene_id}_F{frame_idx:06d}", video_id="L01_V001", scene_id=scene_id,
            frame_idx=frame_idx, timestamp_sec=frame_idx / 30, image_path=image.relative_to(DATA).as_posix(),
            width=960, height=540, roles=["representative"],
            captions=[CaptionRecord(caption_type="detailed", language="vi", text=label, confidence=.9, provenance=provenance("demo-caption"))],
            ocr_instances=ocr_items,
        )
        scenes.append(Scene(
            scene_id=scene_id, video_id="L01_V001", scene_idx=index,
            start_frame=index * 300, end_frame_exclusive=(index + 1) * 300,
            start_sec=index * 10, end_sec=(index + 1) * 10,
            segmentation_provenance=provenance("demo-segmentation"), keyframes=[frame],
            captions=[SceneCaptionRecord(caption_type="visual", language="vi", text=label, evidence_keyframe_ids=[frame.keyframe_id], provenance=provenance("demo-scene-caption"))],
        ))
    video = Video(
        video_id="L01_V001", source_path="raw/videos/L01_V001.mp4", fps=30, frame_count=900,
        duration_sec=30, width=960, height=540, codec="h264", audio_present=True,
        probe_provenance=provenance("demo-ffprobe"), scenes=scenes,
    )
    manifest = DatasetManifest(
        dataset_id="aic-demo-v1", build_id="demo-001", pipeline_version="aic-v1.0.0",
        video_count=1, scene_count=3, keyframe_count=3,
    )
    result = export_dataset([video], DATA / "exports", manifest)
    if verbose:
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
