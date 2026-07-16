# Configuration and deployment

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AIC_ONLINE_BACKEND` | `local` | `local` or `qdrant` |
| `AIC_METADATA_JSONL` | `datasection/exports/scenes.jsonl` | Canonical Scene export |
| `AIC_CANDIDATE_LIMIT` | `100` | Per-stage candidate pool |
| `AIC_RRF_K` | `60` | RRF stabilizing constant |
| `AIC_REQUEST_TIMEOUT_SEC` | `10` | External HTTP timeout |
| `AIC_QDRANT_URL` | unset | Required in Qdrant mode |
| `AIC_QDRANT_API_KEY` | unset | Qdrant Cloud/self-host auth |
| `AIC_QDRANT_SCENE_COLLECTION` | `aic_scenes_v1` | Scene collection |
| `AIC_QDRANT_VECTOR_NAME` | `visual` | Named vector queried |
| `AIC_EMBEDDING_URL` | unset | Required query encoder endpoint |

## Local process

From repository root:

```bash
python -m pip install -e ./online
cp online/.env.example .env
export AIC_METADATA_JSONL=datasection/exports/scenes.jsonl
uvicorn online.api.app:app --host 0.0.0.0 --port 8000
```

The code intentionally does not auto-read `.env`; load it through your shell,
Docker Compose, systemd, or deployment platform.

## Docker local baseline

From `repo/online`:

```bash
docker compose -f docker-compose.local.yml up --build
```

The compose file mounts `../datasection` read-only.

## Production checklist

1. Pin the exact embedding model revision.
2. Confirm query/index dimensions and normalization.
3. Create Qdrant payload indexes.
4. Validate JSONL and collection manifest revisions match.
5. Set Qdrant API key and TLS endpoint.
6. Disable public debug output.
7. Place API behind authentication/rate limiting.
8. Record P50/P95 latency and retrieval Recall@K.
9. Run smoke queries after every index deployment.

## Security scope

V1 contains no JWT or user accounts because competition search is typically a
single-team application. Authentication should be added at the gateway or as a
separate API dependency when deployment requirements are known.
