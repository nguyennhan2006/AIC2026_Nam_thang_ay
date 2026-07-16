# Testing, limitations, and extension points

## Current automated coverage

Run:

```bash
python -m unittest discover -s online/tests -v
```

Coverage includes:

- nested Scene JSONL projection;
- Vietnamese sequence splitting;
- OCR weight boosting;
- weighted RRF;
- same-video/increasing-scene temporal constraints;
- deterministic Qdrant UUID mapping;
- end-to-end KIS;
- end-to-end three-event sequence retrieval.

## Known V1 limitations

1. Local hashing embeddings are smoke-test only.
2. BM25 tokenizer is whitespace/Unicode-word based, not Vietnamese word
   segmentation.
3. No fuzzy OCR correction yet.
4. No cross-encoder reranker is enabled.
5. VQA default returns grounded evidence rather than a fluent generated answer.
6. Metadata reload requires process restart.
7. AVS diversity uses a simple per-video cap.
8. Sequence linking uses beam search and a linear gap penalty.
9. API authentication, caching, distributed tracing, and rate limiting are not
   included until deployment requirements are known.

## Recommended experiments before replacing defaults

- Compare BM25 tokenizer with underthesea/VnCoreNLP segmentation.
- Compare RRF weights on a held-out KIS/AVS query set.
- Measure keyframe-first versus scene-first Qdrant retrieval.
- Add OCR normalization/fuzzy matching and run ablation.
- Evaluate temporal gap penalties on sequence queries.
- Add a cross-encoder only if Recall@100 is already high.
- Add an LLM/VLM answer generator with evidence-faithfulness metrics.

## Extension ports

`ports/interfaces.py` defines replaceable boundaries for:

- metadata repository;
- text encoder;
- vector store;
- retriever;
- reranker;
- answer generator.

New infrastructure should implement a port and be wired in `api/container.py`;
do not add vendor-specific code inside SearchService or routes.

