# Online architecture

## Dependency direction

```mermaid
flowchart TD
    API[FastAPI routes] --> SVC[Search and VQA services]
    SVC --> DOM[Online read models]
    SVC --> PORTS[Infrastructure ports]
    ADAPTERS[JSONL, BM25, Qdrant, encoders] --> PORTS
    ADAPTERS --> DATA[datasection exports]
```

Business services depend on protocols, not Qdrant, files, or FastAPI. Concrete
selection happens only in `api/container.py`.

## Request path

```mermaid
flowchart LR
    Q[Query] --> P[Rule planner]
    P --> R[Parallel retrievers]
    R --> F[Weighted RRF]
    F --> H[Metadata hydration]
    H --> T{Task}
    T -->|KIS| K[Top moments]
    T -->|AVS| A[Diversified scenes]
    T -->|Sequence| O[Temporal linker]
    T -->|VQA| V[Evidence answer]
```

## Read-model boundary

`datasection` owns the canonical `Scene -> Keyframe[]` contract. Online creates
an in-memory `SceneDocument` projection containing only fields needed for
retrieval and display. It is not saved back to the dataset and is therefore not
a second canonical source.

## Why separate retrievers?

- Dense visual: semantic objects, actions, settings.
- Caption BM25: rare explicit terms in generated descriptions.
- OCR BM25: exact on-screen text and names.
- ASR BM25: spoken content.
- Keyword BM25: compact expanded lexical features.

Scores are not directly comparable. Fusion uses rank positions rather than
adding cosine and BM25 values.

## FastAPI lifecycle

The application uses FastAPI lifespan to load metadata and indexes once before
serving traffic, then releases the container on shutdown. Routes obtain the
container through dependency injection. This follows the current FastAPI
recommendation:

- https://fastapi.tiangolo.com/advanced/events/
- https://fastapi.tiangolo.com/tutorial/dependencies/

