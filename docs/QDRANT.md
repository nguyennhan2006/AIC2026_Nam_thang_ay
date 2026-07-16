# Qdrant design and operations

## V1 role

Qdrant supplements FAISS for online operation:

- persistent service API;
- payload filters;
- updates/deletes;
- named vectors;
- filtered ANN search;
- operational scaling.

FAISS remains useful for offline benchmarks and static local experiments.

## Collection recommendation

```text
aic_scenes_v1
  point UUID = uuid5("aic2026:" + scene_id)
  named vector = visual
  payload = scene_id, video_id, scene_idx, start/end, has_ocr, has_asr

aic_keyframes_v1
  point UUID = uuid5("aic2026:" + keyframe_id)
  named vector = visual
  payload = keyframe_id, scene_id, video_id, timestamp
```

Online V1 queries `aic_scenes_v1`. A later fine-grained retriever can query the
keyframe collection and aggregate candidates to scenes before fusion.

## Named vectors

Qdrant supports multiple named vectors in one point, each with its own size and
distance. Do not place embeddings from different models under the same vector
name. Official collection documentation:

https://qdrant.tech/documentation/manage-data/collections/

## Payload indexes

Create payload indexes only for fields used in filters:

```text
video_id   keyword
scene_id   keyword
scene_idx  integer
has_ocr    bool
has_asr    bool
start_sec  float
end_sec    float
```

Payload indexes consume resources; Qdrant recommends indexing fields that
actually constrain searches:

https://qdrant.tech/documentation/manage-data/indexing/

## Query API

The adapter calls:

```text
POST /collections/{collection}/points/query
```

with `query`, `using`, `filter`, `limit`, and payload enabled. Qdrant documents
the Query API and hybrid/multi-stage queries here:

- https://qdrant.tech/documentation/search/search/
- https://qdrant.tech/documentation/search/hybrid-queries/

## Encoder compatibility

The query encoder must exactly match the model, preprocessing, dimension, and
normalization used by the index builder. Online requires an external endpoint:

```http
POST /embed/text
Content-Type: application/json

{"text": "một nhóm người cào muối"}
```

Response:

```json
{"vector": [0.01, -0.03, 0.2]}
```

Dimension mismatch is a deployment error, not something Online should pad or
truncate.

