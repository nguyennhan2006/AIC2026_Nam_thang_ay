# File-by-file review guide

This document supports the same ACCEPT/REVISE process used for datasection.

| File/group | Objective | Main advantage | Cost or caution |
|---|---|---|---|
| `config.py` | Validate environment configuration | One explicit runtime contract | New options require dataclass update |
| `errors.py` | Stable typed failures | API can map infrastructure errors | Error taxonomy is intentionally small |
| `domain/models.py` | Query, candidate, hit, VQA read models | Strict boundary between API/services | Not canonical dataset models |
| `ports/interfaces.py` | Dependency inversion protocols | Swap Qdrant/FAISS/models | Protocol errors appear at integration time |
| `json_metadata.py` | Project canonical Scene JSONL | No duplicated persisted metadata | Loads entire projection into RAM |
| `encoders.py` | Local smoke and remote production encoders | Same TextEncoder interface | Hashing encoder has no benchmark value |
| `bm25.py` | Local field-specific lexical retrieval | Transparent and testable | Basic tokenizer; not large-scale storage |
| `vector_stores.py` | Local cosine and Qdrant Query API | Qdrant without SDK lock-in | REST response/version must be monitored |
| `dense_retriever.py` | Compose encoder and vector store | Infrastructure remains replaceable | Only scene-level dense retrieval in V1 |
| `query_planner.py` | Normalize, weight, split events | Deterministic Vietnamese fallback | Rules cannot capture every paraphrase |
| `fusion.py` | Weighted RRF | No invalid score-scale mixing | Weights require evaluation |
| `temporal.py` | Link event candidates | Simple ordered sequence baseline | Beam/gap hyperparameters require tuning |
| `search.py` | KIS/AVS/sequence orchestration | One service for every delivery layer | No cross-encoder enabled yet |
| `vqa.py` | Retrieve evidence and generate answer | Grounded default behavior | Fluent LLM answer is an external adapter |
| `api/container.py` | Select local or Qdrant components | Vendor choices isolated in one file | Startup rebuilds local indexes |
| `api/routes.py` | Versioned HTTP endpoints | Thin, testable delivery layer | Requires FastAPI runtime dependency |
| `api/app.py` | Lifespan and error handling | Dependencies loaded once | Startup fails fast if metadata is missing |
| `ui/*` | Minimal KIS/AVS/Sequence/VQA browser UI | Immediately usable search surface | Media serving remains external |
| `cli.py` | Local smoke runner | Reuses production service graph | Not an interactive UI |
| `examples/scenes.jsonl` | Executable example dataset | Reproduces key workflows | Not a benchmark dataset |
| `tests/test_online_core.py` | Regression checks | Tests full core without Qdrant | Does not test a live external service |
| `Dockerfile` | Reproducible API image | Clear Python/runtime boundary | Model service remains external |
| `docker-compose.local.yml` | Local container startup | Read-only datasection mount | Local mode only |
| `example_requests.http` | Ready API requests | Easy IDE/manual verification | Host/port may need editing |
| `pyproject.toml` | Dependencies and CLI entrypoint | `pip install -e ./online` | Must merge carefully with monorepo tooling |

## Decision notes

The package is deliberately a complete V1 baseline, not a claim that every
default is optimal. Recommended review order:

1. `domain/models.py`
2. `ports/interfaces.py`
3. `query_planner.py`
4. `fusion.py`
5. `temporal.py`
6. `search.py`
7. adapters
8. API and deployment files
