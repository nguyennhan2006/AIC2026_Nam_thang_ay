# AIC 2026 metadata conventions

> Quick reference only. Full documentation (features, usage guide, error
> reference — Vietnamese) lives in [`docs/data_section.md`](../docs/data_section.md).

## ID hierarchy

IDs are fixed-width, case-sensitive, and encode their parent relationship:

| Entity | Format | Example |
|---|---|---|
| Video | `Ldd_Vddd` | `L01_V001` |
| Scene | `<video_id>_Sdddd` | `L01_V001_S0003` |
| Keyframe | `<scene_id>_Fdddddd` | `L01_V001_S0003_F001234` |

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
- Cross-checking `frame_idx / fps` against `timestamp_sec` belongs to a later
  validator that has access to the parent `Video` entity and its FPS policy.

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

The generated `contracts/keyframe.schema.json` is committed as the portable
contract. Tests fail if it is no longer synchronized with the Python model.
