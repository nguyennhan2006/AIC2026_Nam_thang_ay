# 03. Offline pipeline

## Stage và đầu ra

| Stage | Input | Output | Có thể retry |
|---|---|---|---|
| probe | video | fps/frame/codec/audio | có |
| scene | probe | half-open boundaries | có |
| keyframe | boundaries | JPEG/SVG + quality | có |
| caption | image/crop | short/detailed/tags | có |
| OCR | image | text + bbox + confidence | có |
| object | image + caption labels | label + bbox | có |
| ASR | audio/video URI | timestamp segments | có |
| embedding | image/text | normalized vector | có |
| export/index | validated entities | JSONL/FAISS/Qdrant | atomic |

`JobLedger` ghi trạng thái từng video bằng atomic rename. Remote client gửi
`Idempotency-Key`, retry lỗi timeout/429/5xx với exponential backoff; lỗi 4xx
không retry. Caption/OCR/object của cùng keyframe chạy song song nhưng concurrency
thật được giới hạn ở worker.

## Thay model thật

`AIC_GPU_PROVIDER=transformers` bật engine production mẫu trong
`offline/gpu_engine.py`; `offline/worker.py::InferenceEngine` là smoke engine.
Adapter model thay thế phải trả:

- caption: `{captions:[{text,language,confidence}]}`
- OCR: `{instances:[{text,normalized_text,language,confidence,bbox}]}`
- object: `{objects:[{label,confidence,bbox,attributes}]}`
- embed text: `{vector:[float],dimension,model}`
- ASR: `{segments:[{start_sec,end_sec,text,...}]}`

`bbox` luôn normalized XYXY. Ghi model name, revision, prompt/config và pipeline
version vào provenance. Không đổi model encoder mà tái dùng index cũ.

## Object từ caption và visual

Caption/tags sinh candidate labels; visual detector hoặc open-vocabulary
detector định vị label trên ảnh. Caption là soft evidence, visual score quyết
định bbox. Không tạo bbox giả chỉ từ text. Gộp object bằng label normalization
và NMS; giữ provenance của từng nguồn.

## Baseline hiện thực

Pipeline mặc định dùng uniform scene và mock model để chạy plumbing ở mọi máy.
Khi worker dùng CLIP, chạy index với `--encoder remote` để scene vector là mean
pool của image embeddings và Online query dùng text embedding cùng model. Không
được ghép local hashing index với CLIP query encoder.

Production cần thay scene detector/model inference; đây là cấu hình có chủ ý,
không phải tuyên bố chất lượng retrieval.
