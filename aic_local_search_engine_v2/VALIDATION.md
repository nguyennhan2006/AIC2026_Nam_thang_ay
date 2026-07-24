# Validation report — v2

Validated on the supplied AIC sample (10 scenes, 33 keyframes, 512-dimensional
OpenCLIP vectors).

## Input compatibility

- Rich metadata ZIP: `scene_docs.jsonl`, `frame_docs.jsonl`, scene/frame visual
  vectors and validation report.
- Compact metadata: `scene_metadata.jsonl` and `scene_embeddings.npy`.
- Output 05 schema: captions, subjects, objects, actions, relations, keywords,
  visible text and temporal events.
- Output 06 schema: global `passed/errors/warnings` quality gate.
- ZIP and extracted-folder discovery both exercised.

## Executed checks

1. Built from rich metadata + output 05 + output 06 without outputs 01-04:
   10 scenes, 33 keyframes, both vector indexes.
2. Built from compact metadata + outputs 01-04 + output 05 + output 06.
3. Queried all lexical branches: semantic, OCR, speech, tags and event.
4. Queried scene and frame vector branches with known indexed vectors; the
   source scene/frame ranked first with cosine > 0.99.
5. Queried ordered temporal sequences, including event order within one scene
   and progression across later scenes.
6. Verified Vietnamese accent-folding (`địa` ↔ `dia`).
7. Verified `needs_review` receives a 0.75 score multiplier.
8. Verified an output 06 report with `passed=false` stops the build.
9. Executed every code cell in `08_local_multibranch_search.ipynb` end-to-end.
10. Verified missing PyTorch/open_clip causes vector text encoding to be skipped
    while multi-branch lexical search still returns results.

## Environment note

The validation container did not include FAISS, so vector execution used the
Numpy exact-search fallback. The FAISS path uses normalized float32 vectors,
`IndexHNSWFlat` with inner-product metric, and the same row mappings; the target
machine can use this path with its existing FAISS installation.
