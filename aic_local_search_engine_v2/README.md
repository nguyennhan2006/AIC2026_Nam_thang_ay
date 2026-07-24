# AIC Local Multi-branch Search v2

Search engine chạy hoàn toàn trên máy local cho pipeline AIC 2026. Engine dùng
các kỹ thuật tương ứng với Elasticsearch nhưng không cần Docker, service nền
hoặc RAM dành riêng cho server.

| Kỹ thuật Elastic | Bản local |
|---|---|
| Inverted index + BM25 | SQLite FTS5 |
| Nhiều field/boost | 5 FTS index tách biệt |
| `dense_vector` + HNSW kNN | FAISS `IndexHNSWFlat` |
| Metadata/filter | Bảng SQLite + SQL index |
| RRF retriever | Weighted RRF trong Python |
| Temporal retrieval | Event index + beam search |
| Ingest/quality gate | Adapter output 05/06/metadata |

Đây là search engine, không phải neural model cần train.

## Input

Đặt các ZIP hoặc thư mục sau dưới cùng một `input-root`:

- output 05: `scene_semantics_qwen3vl.jsonl`;
- output 06: `component_validation_report.json`;
- metadata: engine hỗ trợ cả hai schema:
  - rich: `scene_docs.jsonl`, `frame_docs.jsonl`,
    `scene_visual_embeddings.npy`, `frame_visual_embeddings.npy`;
  - compact: `scene_metadata.jsonl`, `scene_embeddings.npy`.

Với schema rich, chỉ cần 05 + 06 + metadata để build đủ scene/frame index. Với
schema compact, output 01 là tùy chọn; thêm output 01 nếu muốn có frame-vector
search, còn scene-vector search vẫn hoạt động từ metadata.

Output 06 là quality gate toàn pipeline. Nếu `passed=false`, build dừng ngay.
Scene có status `needs_review` trong output 05 vẫn được index nhưng mặc định chỉ
nhận 75% điểm RRF. Scene invalid bị loại.

## Các nhánh search

Engine không nối mọi nội dung vào một chuỗi lớn. Mỗi loại bằng chứng có index và
trọng số riêng:

1. `semantic`: caption Việt/Anh và mô tả event từ output 05;
2. `ocr`: OCR scene và `visible_text`;
3. `speech`: ASR transcript và `speech_summary`;
4. `tags`: keyword, subject/object, action, attribute, relation, scene type;
5. `event`: từng temporal event có timestamp riêng;
6. `scene_vector`: FAISS trên scene embedding;
7. `frame_vector`: FAISS trên keyframe embedding.

Query planner tự tăng trọng số OCR cho câu hỏi về chữ/logo/số, tăng ASR cho
“nói/phát biểu”, và tăng event/action cho câu hỏi trình tự. Kết quả các nhánh
được gộp bằng weighted RRF, sau đó áp quality penalty.

## Cài local

Python 3.10 hoặc 3.11 được khuyến nghị.

```powershell
conda create -n aic-search python=3.10 -y
conda activate aic-search
cd C:\Users\LOQ\Documents\AIC2026\aic_local_search
pip install -e .
```

Bạn đã có FAISS thì không cần cài lại. Kiểm tra:

```powershell
python -c "import faiss; print('FAISS', faiss.__version__)"
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('create virtual table t using fts5(x)'); print('FTS5 OK')"
```

PyTorch **không bắt buộc** để build index, BM25/semantic/OCR/ASR/tag/event
search hoặc tìm bằng vector đã có. Chỉ cài PyTorch + `open_clip_torch` khi muốn
encode một câu mô tả mới trực tiếp sang OpenCLIP visual vector. Nếu chưa cài,
API tự bỏ qua nhánh encode đó và vẫn trả kết quả multi-branch lexical.

## Build

```powershell
aic-local-search build `
  --input-root C:\Users\LOQ\Documents\AIC2026\component_outputs `
  --index-dir C:\Users\LOQ\Documents\AIC2026\08_local_search_index `
  --vector-backend auto
```

Khi FAISS có sẵn, output index chỉ gồm:

```text
08_local_search_index/
├── aic_search.db
├── scene_hnsw.faiss
├── frame_hnsw.faiss       # nếu metadata có frame vector
└── index_manifest.json
```

Nếu không có FAISS, engine tự lưu `scene_embeddings.npy` và
`frame_embeddings.npy` để exact-search bằng Numpy.

## Query không cần PyTorch

```powershell
aic-local-search query `
  --index-dir C:\Users\LOQ\Documents\AIC2026\08_local_search_index `
  --text "Công viên địa chất Lạng Sơn được UNESCO công nhận" `
  --task scene --no-vector --top-k 10
```

OCR-oriented query:

```powershell
aic-local-search query `
  --index-dir C:\Users\LOQ\Documents\AIC2026\08_local_search_index `
  --text "chữ CỤC THÚ Y dấu vuông trên màn hình" `
  --task frame --no-vector --top-k 10
```

Temporal search:

```powershell
aic-local-search sequence `
  --index-dir C:\Users\LOQ\Documents\AIC2026\08_local_search_index `
  --step "Cục Thú y nói về thịt heo" `
  --step "Lạng Sơn nhận bằng UNESCO" `
  --no-vector --top-k 5
```

## Python API

```python
from aic_local_search import LocalHybridSearchEngine

with LocalHybridSearchEngine(r"C:\AIC2026\08_local_search_index") as engine:
    hits = engine.search(
        "Công viên địa chất Lạng Sơn được UNESCO công nhận",
        use_vector=False,
        task="scene",
        top_k=10,
    )
```

Nếu đã có query vector 512 chiều tương thích OpenCLIP, truyền thẳng
`query_vector=...`; thao tác này dùng FAISS nhưng không dùng PyTorch.

## Giới hạn

Thiết kế phù hợp cho một máy local, demo và dataset cỡ hàng trăm nghìn scene.
Nó không có sharding, replication hoặc multi-user concurrency như
Elasticsearch. Khi dataset lên hàng triệu frame hoặc cần nhiều worker cập nhật
đồng thời, nên chuyển lại Elasticsearch/OpenSearch/Qdrant nhưng giữ nguyên lớp
query planner và RRF.
