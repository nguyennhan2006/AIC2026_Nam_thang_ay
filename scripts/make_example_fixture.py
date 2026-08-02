"""Sinh lại fixture `examples/scenes.jsonl` + `examples/videos.jsonl`.

Vì sao cần script này thay vì sửa tay JSON: fixture cũ được viết tay và
KHÔNG hợp lệ theo `datasection.schemas.Scene` (thiếu `start_frame`,
`end_frame_exclusive`, keyframe thiếu `frame_idx`/`timestamp_sec`/
`width`/`height`/`roles`/provenance). `JsonlSceneRepository.load` có gọi
`Scene.model_validate`, nên fixture sai làm test online fail và — tệ hơn —
biến "fixture không khớp contract" thành chuyện bình thường.

Fixture phải là dữ liệu canonical thật thu nhỏ, nếu không nó sẽ không bắt
được lỗi contract nào cả. Nội dung giữ nguyên như bản viết tay (cùng
scene_id, cùng caption/OCR/ASR/keyword) để test hiện có vẫn nói về cùng
một câu chuyện.

    python -m scripts.make_example_fixture
"""

from __future__ import annotations

from pathlib import Path

from datasection.exporter import atomic_jsonl
from datasection.schemas import (
    ASRSegment,
    BoundingBox,
    CaptionRecord,
    ModelProvenance,
    OCRInstance,
    Scene,
    SceneCaptionRecord,
    SceneKeyword,
    Keyframe,
    Video,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FPS = 30.0


def provenance(name: str) -> ModelProvenance:
    return ModelProvenance(
        model_name=name, model_revision="fixture", pipeline_version="aic-v1.0.0"
    )


def build_scene(
    *,
    video_id: str,
    scene_idx: int,
    start_sec: float,
    end_sec: float,
    frame_idx: int,
    keyframe_caption: str,
    scene_caption: str,
    ocr_text: str | None,
    asr_text: str | None,
    keywords: list[str],
) -> Scene:
    scene_id = f"{video_id}_S{scene_idx:04d}"
    keyframe = Keyframe(
        keyframe_id=f"{scene_id}_F{frame_idx:06d}",
        video_id=video_id,
        scene_id=scene_id,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / FPS,
        image_path=f"processed/keyframes/{video_id}/frame_{frame_idx:06d}.jpg",
        width=960,
        height=540,
        roles=["representative"],
        captions=[
            CaptionRecord(
                caption_type="detailed",
                language="vi",
                text=keyframe_caption,
                confidence=0.9,
                provenance=provenance("fixture-caption"),
            )
        ],
        ocr_instances=(
            []
            if not ocr_text
            else [
                OCRInstance(
                    text=ocr_text,
                    normalized_text=ocr_text.casefold(),
                    language="vi",
                    confidence=0.98,
                    bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.9, y2=0.4),
                    provenance=provenance("fixture-ocr"),
                )
            ]
        ),
    )
    return Scene(
        scene_id=scene_id,
        video_id=video_id,
        scene_idx=scene_idx,
        start_frame=round(start_sec * FPS),
        end_frame_exclusive=round(end_sec * FPS),
        start_sec=start_sec,
        end_sec=end_sec,
        segmentation_provenance=provenance("fixture-segmentation"),
        keyframes=[keyframe],
        captions=[
            SceneCaptionRecord(
                caption_type="visual",
                language="en",
                text=scene_caption,
                evidence_keyframe_ids=[keyframe.keyframe_id],
                provenance=provenance("fixture-scene-caption"),
            )
        ],
        asr_segments=(
            []
            if not asr_text
            else [
                ASRSegment(
                    segment_id=f"{scene_id}_A0000",
                    source_segment_id=f"{video_id}_ASR000000",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=asr_text,
                    normalized_text=asr_text.casefold(),
                    language="vi",
                    confidence=0.9,
                    provenance=provenance("fixture-asr"),
                )
            ]
        ),
        keywords=[
            SceneKeyword(
                text=item,
                normalized_text=item.casefold(),
                sources=["caption"],
                provenance=provenance("fixture-keyword"),
            )
            for item in keywords
        ],
    )


def build_videos() -> list[Video]:
    l01 = [
        build_scene(
            video_id="L01_V001", scene_idx=1, start_sec=10.0, end_sec=15.0, frame_idx=300,
            keyframe_caption="Một số người đang cào muối trên cánh đồng muối trắng.",
            scene_caption="Workers rake salt into piles in a salt field.",
            ocr_text=None, asr_text=None, keywords=["cào muối", "salt field"],
        ),
        build_scene(
            video_id="L01_V001", scene_idx=2, start_sec=15.0, end_sec=20.0, frame_idx=450,
            keyframe_caption="Một đoàn người vẫy tay phía sau bảng chữ.",
            scene_caption="A group of people waves behind a sign.",
            ocr_text="Đoàn kết", asr_text=None, keywords=["vẫy tay", "bảng chữ"],
        ),
        build_scene(
            video_id="L01_V001", scene_idx=3, start_sec=20.0, end_sec=26.0, frame_idx=600,
            keyframe_caption="Một nhóm người đứng trước căn nhà màu trắng.",
            scene_caption="People pose in front of a white house.",
            ocr_text="Gừng cay muối mặn xin đừng quên nhau",
            asr_text="Xin đừng quên nhau", keywords=["căn nhà"],
        ),
    ]
    l02 = [
        build_scene(
            video_id="L02_V001", scene_idx=1, start_sec=5.0, end_sec=9.0, frame_idx=150,
            keyframe_caption="Người dẫn chương trình phát biểu trong trường quay.",
            scene_caption="A presenter speaks in a television studio.",
            ocr_text=None, asr_text="Bản tin hôm nay", keywords=["trường quay"],
        )
    ]
    return [
        Video(
            video_id="L01_V001", source_path="raw/videos/L01_V001.mp4", fps=FPS,
            frame_count=900, duration_sec=900 / FPS, width=960, height=540, codec="h264",
            audio_present=True, probe_provenance=provenance("fixture-ffprobe"), scenes=l01,
        ),
        Video(
            video_id="L02_V001", source_path="raw/videos/L02_V001.mp4", fps=FPS,
            frame_count=300, duration_sec=300 / FPS, width=960, height=540, codec="h264",
            audio_present=True, probe_provenance=provenance("fixture-ffprobe"), scenes=l02,
        ),
    ]


def main() -> None:
    videos = build_videos()
    scenes = [scene for video in videos for scene in video.scenes]
    atomic_jsonl(EXAMPLES / "scenes.jsonl", [item.model_dump(mode="json") for item in scenes])
    atomic_jsonl(EXAMPLES / "videos.jsonl", [item.model_dump(mode="json") for item in videos])
    print(f"wrote {len(scenes)} scenes / {len(videos)} videos to {EXAMPLES}")


if __name__ == "__main__":
    main()
