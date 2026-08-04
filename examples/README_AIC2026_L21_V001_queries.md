# AIC 2026 — L21_V001 Four-Task Query Benchmark

## Scope
This package contains a high-quality mini benchmark for four query types:
- Textual KIS
- VQA / Q&A
- AVS
- TRAKE

Video metadata:
- video_id: `L21_V001`
- duration: `1261.726` seconds
- fps: `30`
- total frames: `37849`
- resolution: `1280x720`

## Official-rule alignment
The supplied preliminary-round document defines:
- KIS output: `<video_id>, <frame_id>`
- Q&A output: `<video_id>, <frame_id>, <answer>`
- TRAKE output: `<video_id>, <frame_id_1>, ..., <frame_id_n>`
- TRAKE first retrieves one video and then aligns one semantic keyframe for every ordered event.
- TRAKE semantic GT windows are usually very short; this dataset uses 9-frame windows (±4 frames).

AVS is included as the fourth system-evaluation task requested for the current project. Its positives use relevance grades 0–3 and require event-level deduplication.

## Package contents
- `AIC2026_L21_V001_queries_4tasks.xlsx`: human-readable workbook.
- `AIC2026_L21_V001_queries_4tasks.jsonl`: machine-readable nested records.
- `AIC2026_L21_V001_query_schema.json`: dataset and output contract.
- `README.md`: this guide.

## Quality policy
- Ground truth was built from direct visual and readable OCR inspection of the supplied video.
- No ASR-only query was generated because the audio transcript was not available for manual verification.
- KIS and VQA include hard-negative notes.
- AVS includes multiple positive intervals, relevance grades and deduplication requirements.
- TRAKE contains exact ordered semantic moments with narrow frame windows.

## Recommended evaluation
- KIS: R@1/5/20/50/100, MRR, hard-negative confusion.
- VQA: evidence hit, answer EM/F1, end-to-end correctness.
- AVS: mAP, nDCG, distinct-event coverage, redundancy rate.
- TRAKE: correct-video rate, per-event alignment accuracy, mean R-Score, complete-chain accuracy.
