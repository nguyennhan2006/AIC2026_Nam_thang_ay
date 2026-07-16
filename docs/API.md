# HTTP API

Base prefix: `/v1`

## Search request

```json
{
  "query": "Một nhóm người đứng trước căn nhà có dòng chữ...",
  "top_k": 20,
  "filters": {
    "video_ids": [],
    "scene_ids": [],
    "has_ocr": true,
    "has_asr": null,
    "start_sec_gte": null,
    "end_sec_lte": null
  },
  "debug": false
}
```

Routes force their own task type even if a client includes a different `task`:

- `POST /search/kis`
- `POST /search/avs`
- `POST /search/sequence`

## Search hit

```json
{
  "scene_id": "L01_V001_S0003",
  "video_id": "L01_V001",
  "scene_idx": 3,
  "start_sec": 20.0,
  "end_sec": 26.0,
  "score": 0.08,
  "keyframe_ids": ["L01_V001_S0003_F000600"],
  "keyframe_paths": ["processed/keyframes/...jpg"],
  "matched_modalities": ["visual", "ocr"],
  "evidence": [],
  "component_scores": {}
}
```

`score` is a fusion score and should only be compared within the same request
and pipeline version.

## VQA

`POST /vqa`

```json
{
  "question": "Dòng chữ trên căn nhà là gì?",
  "top_k_evidence": 5,
  "filters": {},
  "debug": false
}
```

Response includes answer, optional confidence, evidence scenes, and latency.

## Debug mode

`debug=true` returns the `QueryPlan`. Do not expose debug mode publicly if
future planners include private prompts or internal operational metadata.

## Error behavior

- `404`: scene not found.
- `422`: request validation failure.
- `503`: metadata, Qdrant, or embedding dependency unavailable.

