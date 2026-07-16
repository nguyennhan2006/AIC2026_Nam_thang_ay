# Datasection integration contract

## Required input

Default path:

```text
datasection/exports/scenes.jsonl
```

Each line must contain one accepted canonical Scene document. Online reads:

```text
scene_id, video_id, scene_idx, start_sec, end_sec
keyframes[].keyframe_id
keyframes[].image_path
keyframes[].captions[].text
keyframes[].ocr_instances[].text
captions[].text
asr_segments[].text
keywords[].normalized_text or keywords[].text
```

All remaining canonical fields are preserved in the dataset but are not loaded
into the Online read projection.

## Export responsibility

Datasection should validate every Scene with its Pydantic/JSON Schema contract
before writing JSONL. Online intentionally performs only projection-level
validation; it must not become a second implementation of the canonical schema.

## Atomic publishing

Recommended flow:

1. Write `scenes.jsonl.tmp`.
2. Validate every line and count unique `scene_id` values.
3. Build vector/lexical indexes against that exact manifest revision.
4. Rename atomically to `scenes.jsonl`.
5. Restart or reload Online.

Do not edit a published file in place while Online is reading it.

## Identity rules

Business IDs remain values such as `L01_V001_S0003`. Qdrant point IDs must be
unsigned integers or UUIDs, so indexing must use:

```python
from online.adapters.vector_stores import qdrant_point_id

point_id = qdrant_point_id("L01_V001_S0003")
```

Always store the original `scene_id` and `video_id` in payload. Official Qdrant
point documentation: https://qdrant.tech/documentation/manage-data/points/

## Reload behavior

V1 loads JSONL into memory during application lifespan. Updating the file does
not update a running process. Restart the API after publishing a new manifest.
A later version can introduce an explicit versioned hot-reload endpoint.

