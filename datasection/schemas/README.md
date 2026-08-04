# AIC 2026 metadata conventions

## ID hierarchy

IDs are fixed-width, case-sensitive, and encode their parent relationship:

| Entity | Format | Example |
|---|---|---|
| Video | `Ldd_Vddd` | `L01_V001` |
| Scene | `<video_id>_Sdddd` | `L01_V001_S0003` |
| Keyframe | `<scene_id>_Fdddddd` | `L01_V001_S0003_F001234` |
| Source ASR | `<video_id>_ASRdddddd` | `L01_V001_ASR000123` |
| Scene ASR projection | `<scene_id>_Adddd` | `L01_V001_S0003_A0001` |

For a keyframe, the six digits after `F` must equal `frame_idx`. IDs must not
be regenerated when metadata is reprocessed with a new model.

Because scene and keyframe IDs encode scene membership, a published scene
manifest is immutable. If re-segmentation changes scene boundaries or ordering,
publish a new dataset revision/namespace instead of silently reusing old IDs.
The fixed-width `Ldd_Vddd` convention is the V1 dataset contract; supporting a
different organizer naming scheme requires an explicit schema revision.

## Paths and URIs

- Media paths such as `image_path` are POSIX paths relative to the environment
  variable `AIC_DATA_ROOT`.
- Store `processed/keyframes/frame.jpg`, not a machine-specific absolute path.
- Absolute paths and `..` traversal are invalid.
- Artifact references may use a relative path or one of these URI schemes:
  `az`, `file`, `gs`, `https`, `qdrant`, `s3`.

## Checksums

Checksums use lowercase SHA-256 with an algorithm prefix:

```text
sha256:<64 lowercase hexadecimal characters>
```

The checksum is calculated from the exact keyframe image bytes referenced by
`image_path`, not from the source video or decoded pixel array.

## Frame and time semantics

- `frame_idx` is the zero-based frame index in the original video and is the
  identity source used in `keyframe_id`.
- `timestamp_sec` is the navigation/search time in seconds.
- Scene intervals are half-open: `[start_frame, end_frame_exclusive)` and
  `[start_sec, end_sec)`. A keyframe at the end boundary belongs to the next
  scene, not the current scene.
- `Video` cross-checks `frame_idx / fps` against `timestamp_sec` with a
  tolerance of at most two frames.

## Scene aggregation

- A canonical `Scene` embeds its complete `keyframes` children.
- OCR remains canonical at keyframe level. `Scene.ocr_text` is derived from
  children and must not be stored as a second editable value.
- ASR segments stored in a scene are scene-clipped projections. Preserve
  `source_segment_id` so the original ASR segment remains traceable when it
  crosses a scene boundary.

## Python and JSON behavior

Python models retain enum instances, for example `KeyframeRole.OCR_RICH`.
`model_dump(mode="json")` and JSON Schema expose their string values for APIs
and non-Python consumers.

## Contract generation

Run from the project root:

```bash
python scripts/export_schemas.py
python -m unittest discover -s tests -v
```

The generated contracts for Keyframe, Scene, Video and DatasetManifest are
committed. Tests fail if any JSON Schema is no longer synchronized.
