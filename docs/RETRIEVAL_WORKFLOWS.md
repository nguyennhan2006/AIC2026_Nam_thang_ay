# Retrieval workflows

## Query planning

The V1 planner normalizes Unicode/whitespace, extracts quoted phrases, detects
speech/text hints, and splits ordered sequence queries on Vietnamese/English
temporal connectives. It returns a validated `QueryPlan`.

Quoted text raises OCR weight. Speech cues raise ASR weight. The planner is
deterministic and safe to use as fallback even if an LLM planner is added later.

## Candidate generation

Retrievers run concurrently for ordinary queries:

```text
dense scene vector
caption BM25
OCR BM25
ASR BM25
keyword BM25
```

Each returns a ranked `Candidate[]` with business IDs, source, modality, score,
rank, and optional evidence.

## Weighted RRF

For scene `d`:

```text
RRF(d) = sum_m weight(m) / (k + rank_m(d))
```

Default `k=60`. This avoids incorrectly adding cosine, BM25, or future
cross-encoder scores on incompatible scales.

## KIS

1. Plan one query.
2. Retrieve candidates from all modalities.
3. Fuse.
4. Hydrate scene timestamps/keyframes.
5. Return the best exact moments.

## AVS

Uses the same hybrid search, then limits repeated results from one video. V1
allows at most three results per video in the final diversified list.

## Ordered sequence

1. Split query into ordered events.
2. Retrieve and fuse each event independently.
3. Beam-link candidates with the same `video_id` and increasing `scene_idx`.
4. Penalize large temporal gaps.
5. Return complete scene sequences.

V1 does not require consecutive scenes; relevant events may be separated by
short irrelevant scenes.

## VQA

1. Search question as a hybrid retrieval query.
2. Select top evidence scenes.
3. Load caption, OCR, and ASR evidence.
4. Pass contexts to `AnswerGenerator`.

The bundled answer generator is evidence-only: it refuses to invent a polished
answer and returns retrieved snippets. Replace it with a grounded LLM/VLM
adapter after defining the external service contract and evaluation set.

