# Research alignment and deliberate V1 scope

## Sparse lexical image retrieval

The provided paper *Rethinking Sparse Lexical Representations for Image
Retrieval in the Age of Rising Multi-Modal Large Language Models* converts
visual content into captions/tags and retrieves with sparse lexical
representations such as BM25. Online V1 applies the deployable part of that
idea through separate caption, OCR, ASR, and keyword BM25 retrievers.

The paper's extensive fixed-pattern crop expansion is not performed online.
Crop captioning belongs to Offline/Data processing; Online only consumes the
resulting text. This prevents query latency and GPU cost from growing with the
number of crops.

## Dense + sparse hybrid

The paper does not imply that dense retrieval should be removed. V1 keeps:

```text
dense semantic candidates + sparse lexical candidates -> weighted RRF
```

This is important for Vietnamese natural-language queries where exact OCR and
names benefit from sparse matching, while paraphrased actions/settings benefit
from visual-language embeddings.

## Composed image retrieval

The provided CoLLM paper (arXiv:2503.19910) targets a reference image plus
modification text. Online V1 defines image/vector extension boundaries but does
not include CoLLM training or inference. A later endpoint can accept:

```text
reference_keyframe_id + modification_text
```

and implement a composed-query encoder behind the existing `TextEncoder` /
`Retriever` boundary, without changing fusion, temporal linking, or response
models.

## Scope decision

V1 prioritizes an executable, explainable baseline:

- deterministic query planning;
- dense and lexical candidate generation;
- rank fusion;
- temporal coherence;
- grounded VQA evidence.

Fine-tuned composed retrieval, learned fusion, ColBERT, SPLADE, and MLLM
reranking remain measured experiments rather than mandatory runtime dependencies.

