# 11. Server implementation — profile THI ĐẤU / THỰC TẾ trên GPU A100

Branch: `server_implementation`. Nguồn spec: sơ đồ `aic2026-tong-quan.html`, profile
**THI ĐẤU / THỰC TẾ** — goal "Tối đa chất lượng truy hồi và độ tin cậy evidence, vẫn
đáp ứng latency và khả năng khôi phục"; compute "GPU workers tách biệt · storage/index
versioned · API server · job queue/resume"; policy **"Cascaded retrieval, task-aware
rerank, provenance đầy đủ, timeout có trạng thái; không silent degradation."**

Tài liệu ánh xạ từng node sơ đồ (zone A–G, cột production) sang code hiện có: giữ /
thay / thêm, kèm ngân sách GPU và kế hoạch phase. Nguyên tắc xuyên suốt: mọi thay đổi
retrieval chỉ giữ nếu tăng Recall@K/MRR đo bằng `scripts/eval_kis.py`; các mục sơ đồ
đánh dấu *planned* (tracking, queue, resilience, monitoring) là hạng mục bắt buộc nối
production trong kế hoạch này, không để nằm trên giấy (đúng flow "Khoảng trống ưu
tiên" của sơ đồ).

## 1. Topology triển khai

```
┌─────────────────────────── Vast.ai — node GPU ────────────────────────────────┐
│  GPU 0 (A100 80GB)                                                             │
│    [vllm-32b]  Qwen3-VL-32B-Instruct bf16 — caption đa khung, semantic OCR     │
│                verify, VQA answer, evidence rerank top-20 (cổng 8001, internal)│
│  GPU 1 (A100 40/80GB)                                                          │
│    [vllm-14b]  Qwen3-14B-Instruct (AWQ) — query parser strict JSON (8002)      │
│    [worker]    offline/worker.py — TransNetV2, SigLIP2+OpenCLIP, PaddleOCR,    │
│                YOLOv8 + Grounding DINO, Whisper large-v3 + WhisperX,           │
│                Silero VAD, BGE reranker (cổng 8010)                            │
│    [rq-worker] job queue consumer — enrich theo stage, resume/retry            │
│  CPU                                                                           │
│    [qdrant]        HNSW tuned, named vectors frame+scene (6333)                │
│    [elasticsearch] BM25 caption/OCR/ASR/keyword, analyzer vi/en (9200)         │
│    [redis]         queue broker + query/embed cache + rate limit (6379)        │
│    [backend]       online/api FastAPI — cổng PUBLIC duy nhất (8000)            │
│    [prometheus]+[grafana]  metrics + dashboard (nội bộ, profile "ops")         │
└────────────────────────────────────────────────────────────────────────────────┘
                             ▲ HTTPS + Bearer + rate limit
              UI React/Vite local: rank/evidence/latency, export frame_idx
```

- Giữ kiến trúc 3 tầng (datasection / offline / online) và contract worker HTTP —
  worker đổi engine bên trong + thêm endpoint; `RemoteInferenceProvider` giữ nguyên.
- "GPU workers tách biệt" của sơ đồ = tách **model serving (vLLM)** khỏi **worker
  enrich**; mọi model nặng đứng sau HTTP, cho phép scale từng phần hoặc chuyển
  vLLM-32B sang node riêng khi thuê được nhiều máy.
- Enrich toàn corpus là batch job qua **queue** (không còn vòng for tuần tự);
  online serving dùng chung vLLM-32b cho VQA/rerank nên node phải trụ được cả hai
  giai đoạn — xem §3.

## 2. Bảng model theo sơ đồ (cột THI ĐẤU / THỰC TẾ)

| Node sơ đồ | Spec production | Trạng thái code | Việc phải làm |
|---|---|---|---|
| Nguồn video | manifest + quarantine lỗi, incremental ingest | validate + raise, chưa quarantine | §4.A1 |
| Tách scene | TransNetV2 tuned, merge/split, incremental version | uniform 8s | §4.A2 |
| Chọn keyframe | cosine + blur + **motion**, OCR/action-aware sampling, thumbnail, không null metadata | 1 frame giữa scene | §4.A3 |
| Audio | ffmpeg 16kHz + **Silero VAD** + overlap-resolve, resume per chunk, speaker-ready timestamps | ASR đọc thẳng video | §4.A4 |
| Visual embedding | **SigLIP2 + OpenCLIP ViT-L/14 ensemble**, frame + clip-level vectors, calibrated score | CLIP đơn, scene mean-pool | §4.B5 |
| Caption | **Qwen3-VL-32B** multi-frame temporal, uncertainty, hard-case recheck, prompt registry | Qwen2.5-VL-7B đơn frame | §4.B6 |
| OCR | PaddleOCR exact + Qwen3-VL verify frame khó, bbox merge, numeric candidates | chỉ semantic Qwen | §4.B7 |
| Object/action | YOLOv8/11 + **Grounding DINO**; ByteTrack + OSNet re-ID (*planned*, gate theo metric) | chỉ OWLv2 | §4.B8 |
| ASR | **Whisper large-v3 + WhisperX** word alignment, pyannote optional | large-v3-turbo, chunk-level | §4.B9 |
| Metadata | Parquet + object storage, schema migration, reverse map index→frame_idx | JSONL + schema strict | §4.C10 |
| Quality gate | calibrated confidence gate, quarantine + audit reason | không có | §4.C11 |
| Orchestrate | **queue + retries + GPU slots, incremental index + rollback** (*planned*) | for tuần tự + JobLedger | §4.C12 |
| Dense index | Qdrant **HNSW tuned**, snapshot + atomic publish, frame/scene named vectors | Qdrant default + FAISS Flat | §4.D13 |
| Sparse index | **Elasticsearch/OpenSearch** BM25, analyzer vi/en riêng, SPLADE/BGE-M3 chỉ khi tăng metric | BM25 in-memory | §4.D14 |
| Evidence store | object storage + metadata DB (DuckDB/Postgres), evidence pack cache, audit API | JSONL in-RAM | §4.D15 |
| Query parser | **Qwen3-14B** task-aware, must/nice/negative/temporal, query version + confidence | rule-based; `PreparedQueryPlanner` (target/ocr/context) wired sau cờ `AIC_ENABLE_QUERY_PREP`, mặc định tắt, chờ ablation | §4.E16 |
| Search | dense SigLIP/CLIP + Elastic sparse, object/temporal refine, **per-branch deadline/status** | gather all-or-fail; OCR-fuzzy/expansion/rules wired sau cờ env (`AIC_ENABLE_OCR_FUZZY`/`_EXPANSION`/`_RULES`), mặc định tắt — xem docs/15_RESEARCH_AGENDA.md | §4.E17 |
| Fusion | dynamic weights theo query type, RRF + calibrated components, temporal neighborhood | RRF weight tĩnh | §4.E18 |
| Rerank | cross-encoder **top-300→50**, Qwen3-VL evidence rerank **top-20**, rubric riêng KIS/VQA/AVS, temporal consistency | protocol rỗng | §4.E19 |
| Resilience | **Redis cache, circuit breaker, required-branch fail-fast, rate limit** (*planned*) | không có | §4.E20 |
| KIS | precise frame neighborhood, temporal/must-match rerank, top result explainable | scene→best keyframe overlap | §4.F21 |
| VQA | evidence selector + answer + **independent verifier**, OCR/ASR exact rule checks | EvidenceOnly | §4.F22 |
| AVS | **0–3 evidence grading**, MMR + duplicate cluster, coverage/redundancy | per-video cap 3 | §4.F23 |
| UI | rank/evidence/latency view, **submission export frame_idx**, auth + operator controls | vanilla JS | §4.F24 |
| Serving | separate GPU workers, queue + GPU scheduling, **model warmup/preflight** (*planned*) | 1 worker in-process | §4.G25 |
| Eval | offline benchmark gate, per-query audit/error taxonomy, leaderboard | eval_kis + load_test | §4.G26 |
| Monitoring | **Prometheus/Grafana, trace query→evidence, feedback replay** (*planned*) | logging thường | §4.G27 |

## 3. Ngân sách GPU

| Thành phần | VRAM (~) | Enrich | Serving |
|---|---|---|---|
| Qwen3-VL-32B bf16 (vLLM, GPU 0) | ~66GB + KV | ✔ | ✔ |
| Qwen3-14B AWQ (vLLM, GPU 1) | ~9–10GB | — | ✔ |
| SigLIP2 + OpenCLIP ViT-L/14 | ~3.5GB | ✔ | ✔ (text) |
| Whisper large-v3 + WhisperX align | ~4GB | ✔ | — |
| PaddleOCR vi+en | ~1GB | ✔ | — |
| YOLOv8 + Grounding DINO | ~2.5GB | ✔ | — |
| TransNetV2 + Silero VAD | <1GB | ✔ | — |
| BGE reranker-v2-m3 | ~2GB | — | ✔ |
| (ByteTrack + OSNet, khi bật) | ~1.5GB | ✔ | — |

- **Cấu hình khuyến nghị: 1× A100 80GB (GPU 0, dành riêng vLLM-32B bf16) + 1× A100
  40GB (GPU 1, mọi thứ còn lại ~13GB enrich / ~15GB serving)**. Thi đấu ưu tiên chất
  lượng → 32B chạy bf16, không lượng tử. Phương án tiết kiệm khi khan máy: 1× 80GB
  duy nhất + 32B AWQ (~20GB) — chấp nhận rủi ro chất lượng, phải benchmark trước.
- A100 không có FP8 native (Hopper+); nếu dùng checkpoint FP8 chính chủ của Qwen thì
  vLLM chạy weight-only qua Marlin — xác minh throughput lúc dựng máy.
- Elasticsearch + Redis + Qdrant + backend chạy CPU; node thuê cần ≥64GB RAM và
  NVMe đủ cho corpus + index + model cache (~100GB+).

## 4. Thiết kế chi tiết theo zone

### A. Dữ liệu & tiền xử lý

1. **Ingest + quarantine** — thay raise-dừng-cả-run trong
   [pipeline.py](../offline/pipeline.py) bằng: video lỗi (probe fail, tên sai, codec
   không decode) chuyển vào `storage/quarantine/` + record lý do trong ledger; run
   tiếp tục các video còn lại. Ingest incremental: video đã `complete` trong ledger
   thì skip (sửa gap `run()` hiện không skip).
2. **TransNetV2 tuned** — endpoint worker `POST /v1/inference/scene`
   (`{video_uri}` → boundaries + confidence + transition). Threshold tune trên dev
   set của BTC; hậu xử lý merge scene <1s, split >60s, ghi
   `transition_in/out`, `boundary_confidence_*` (schema
   [scene.py](../datasection/schemas/scene.py) có sẵn field). Scene version ghi vào
   manifest để index có thể invalidate đúng phần.
3. **Keyframe production** — `offline/keyframe_select.py`: candidate đều k=8/scene →
   embed (GPU 1) → cosine dedup 0.92 → Laplacian blur + **motion score** (optical
   flow farneback giữa frame kề — đủ tốt, không cần model riêng) → giữ 1–3
   frame/scene + **frame phụ cận cho OCR** khi PaddleOCR phát hiện text (spec
   "OCR/action-aware sampling"). Sinh **thumbnail** 480px cho UI (đóng gap UI load
   ảnh gốc). Điền đầy đủ `QualitySignals` + `selection_score` — "không null
   metadata" đúng nghĩa đen.
4. **Audio** — `extract_audio()` 16kHz mono wav; **Silero VAD** cắt vùng có tiếng
   trước khi đưa Whisper (giảm hallucination đoạn im lặng); chunk 30s overlap 5s,
   resolve overlap bằng interval join; ledger ghi tiến độ theo chunk để resume.

### B. Multimodal workers

5. **Embedding ensemble** — hai encoder cùng chạy: **SigLIP2** (chính) và
   **OpenCLIP ViT-L/14** (đối chứng); Qdrant lưu **named vectors**:
   `frame_siglip`, `frame_clip`, `scene_siglip` (mean-pool). Online dùng
   `scene_siglip` mặc định; nhánh `frame_*` phục vụ KIS frame-precise (§4.F21).
   Encoder nào vào fusion với trọng số bao nhiêu → quyết bằng ablation eval_kis,
   không quyết trước. Đổi encoder = đổi collection version, ghi manifest — không
   ghép index cũ với encoder mới (quy tắc sẵn có của repo).
6. **Caption Qwen3-VL-32B multi-frame** — mỗi scene gửi **chùm keyframe đã dedup**
   (tối đa 4 ảnh) trong 1 request vLLM, trả JSON
   `{caption, objects[], actions[], texts[], uncertainty: 0..1}`; caption cấp scene
   thật thay vì nối chuỗi caption frame (thay `_keyframe`-concat hiện tại).
   `uncertainty > 0.6` → đánh dấu hard-case, chạy recheck 1 lần với prompt yêu cầu
   liệt kê evidence từng ý; vẫn cao → giữ cờ trong `extensions` cho audit. Prompt
   đặt ở `offline/prompts.py` có `PROMPT_VERSION`, ghi vào `ModelProvenance`
   (spec "prompt/version registry").
7. **OCR hai tầng** — PaddleOCR vi+en là nguồn **exact** (`ocr_instances`, lọc
   conf ≥ 0.5, bbox merge các mảnh cùng dòng, tách **numeric candidates** — biển
   số, tỷ số, giá tiền — thành keyword riêng vì hay là đáp án VQA/KIS). Frame khó
   (Paddle conf thấp nhưng Qwen `texts[]` có nội dung) → Qwen3-VL verify, lưu
   caption_type `ocr_semantic` provenance riêng. Hai nguồn không trộn — nhánh BM25
   ocr và OCR-fuzzy chỉ ăn exact; semantic vào caption field.
8. **Detection** — YOLOv8/11 closed-set mọi keyframe; **Grounding DINO** thay OWLv2
   cho open-vocab (label từ caption Qwen + từ query online khi cần refine); merge
   casefold + NMS IoU 0.5, giữ provenance từng nguồn. **ByteTrack + OSNet re-ID để
   sau cờ `AIC_ENABLE_TRACKING`** — chỉ bật khi bộ câu hỏi VQA count/follow chứng
   minh cần (đúng trạng thái *planned* của sơ đồ, gate theo metric).
9. **ASR** — Whisper **large-v3** (bản đầy đủ, không turbo — thi đấu ưu tiên độ
   chính xác) + **WhisperX** forced alignment ra word-level timestamps (KIS cần trỏ
   đúng giây); pyannote diarization sau cờ, chỉ khi câu hỏi speaker xuất hiện.
   `ASRSegment.normalized_text` điền bản bỏ dấu casefold phục vụ BM25.

### C. Chuẩn hoá metadata & điều phối

10. **Storage layout** — JSONL canonical giữ nguyên làm source of truth (schema
    strict + checksum đã tốt); **thêm export Parquet** (scenes/keyframes phẳng hoá)
    cho DuckDB đọc trực tiếp phục vụ audit/debug/analytics; `embedding_refs`
    (schema có sẵn, đang bỏ trống) điền reverse map scene→collection/point-id —
    đóng gap "muốn biết scene dùng vector nào phải suy từ manifest".
11. **Quality gate + quarantine** — trước export: keyframe duplicate>0.98/blur quá
    ngưỡng → loại, ghi `storage/quarantine/audit.jsonl` với lý do máy-đọc-được;
    ngưỡng calibrate trên mẫu human-audit (200 frame/đợt dữ liệu).
12. **Job queue** — **Redis + RQ** (chọn RQ vì Redis đã có mặt cho cache, đơn giản
    hơn Celery, đủ cho 1–2 node; Ray chỉ khi scale nhiều node). Stage
    `probe → scene → keyframe → enrich(caption/ocr/detect) → asr → export` thành
    job idempotent riêng, key theo `(video_id, stage, input_checksum)`; JobLedger
    giữ nguyên làm state cục bộ, RQ quản retry/GPU-slot (queue riêng cho job cần
    GPU 0 vs GPU 1). **Incremental index**: chỉ upsert scene của video mới/đổi;
    Qdrant snapshot trước mỗi đợt publish → rollback được.

### D. Kho chỉ mục

13. **Qdrant HNSW tuned** — cấu hình tường minh (m=32, ef_construct=256, ef search
    tune theo p95) thay vì default; named vectors theo §4.B5; alias collection
    `aic_scenes_active` trỏ version đang phục vụ → publish atomic = đổi alias,
    rollback = trỏ lại. FAISS artifact giữ làm phương án đối chứng offline eval.
14. **Elasticsearch** — index `scenes_v{n}`, field caption/ocr/asr/keyword với
    analyzer riêng: vi (ICU folding + bản bỏ dấu shadow-field) và en (standard).
    Thay 4 `LexicalRetriever` in-memory bằng adapter ES giữ nguyên interface
    `Retriever` — [container.py](../online/api/container.py) chỉ đổi wiring.
    SPLADE/BGE-M3 **không** làm mặc định; chỉ thêm nếu ablation thắng BM25.
15. **Evidence store** — DuckDB đọc Parquet (§4.C10) phục vụ
    `GET /v1/scenes/{id}` + API audit mới `GET /v1/audit/query/{query_id}` (trace
    query→branch→candidates→rerank→result, spec "audit/debug API"); media serve
    thumbnail trước, ảnh gốc theo yêu cầu.

### E. Online retrieval

16. **Query parser Qwen3-14B** — `online/services/llm_planner.py`, output strict
    JSON: `{intent, must[], nice[], negative[], temporal[], ocr_query,
    exact_phrases[], modality_weights, rewrites[≤3], confidence}`; Pydantic
    validate + retry 1; fail → degrade **có cờ** `parser="rule_fallback"` về
    `RuleBasedQueryPlanner` (policy cấm *silent* degradation — degrade hiển thị là
    hợp lệ, giữa trận không được chết vì parser). Query plan cache Redis theo
    normalized query.
17. **Cascaded search** — nhánh: dense (siglip/clip theo §4.B5), 4×ES BM25,
    OCR-fuzzy (module C sẵn có — wire vào), object/temporal refine (lọc candidate
    theo object label + khoảng thời gian từ parser). Mỗi nhánh **deadline riêng**
    (`asyncio.wait_for`), response luôn kèm
    `branches: [{name, status, took_ms, count}]`; nhánh required (dense hoặc ES
    theo intent) fail → 503 rõ ràng; nhánh phụ fail → kết quả vẫn về, status lỗi
    hiển thị. Sửa dứt điểm gap `gather` all-or-fail.
18. **Fusion** — RRF k=60 làm nền + **dynamic weights theo intent** từ parser
    (ocr_query đậm → nhánh ocr tăng; temporal → sequence linking ưu tiên);
    component score calibrate min-max trên dev set; **temporal neighborhood
    bonus**: scene kề (±1 idx) cùng video có mặt trong candidates → cộng nhẹ.
    Mọi trọng số phải thắng ablation trước khi thành mặc định.
19. **Rerank cascade** — tầng 1: BGE reranker-v2-m3 (endpoint worker
    `POST /v1/rerank`) **top-300→50** trên text pack (caption+ocr+asr). Tầng 2:
    Qwen3-VL-32B **top-20** với ảnh keyframe + rubric riêng từng task (KIS: đúng
    khoảnh khắc; VQA: chứa đáp án; AVS: mức liên quan 0–3) + **temporal consistency
    score** cho SEQUENCE. Số lần gọi tầng 2 có trần cứng để giữ p95.
20. **Resilience** — Redis: cache query-plan + text-embedding + kết quả (TTL ngắn,
    key gồm build_id); **circuit breaker** quanh vLLM/ES/Qdrant (fail N lần → mở
    mạch, trả branch-status degraded hiển thị); rate limit theo token bucket trên
    backend; auth Bearer giữ nguyên.

### F. Task & UI

21. **KIS frame-precise** — sau rerank scene: match lại query trên nhánh
    `frame_*` vectors trong scene thắng → `best_timestamp_sec` từ frame thật thay
    vì token-overlap; must-match penalty (module E sẵn có) wire vào; response giữ
    field giải thích (matched_modalities, component_scores, rule_adjustments) —
    "top result explainable".
22. **VQA answer + verifier độc lập** — `QwenVLAnswerGenerator` (32B, evidence
    ảnh + text, bắt buộc dẫn scene_id, abstain khi thiếu); **verifier là lượt gọi
    riêng** (32B, prompt đối kháng: "answer có được evidence chống lưng không?") →
    không pass thì abstain; câu hỏi dạng đọc số/chữ → đối chiếu **exact OCR/ASR
    rule check** trước khi tin model. Count/tracking tools chỉ khi bật §4.B8.
23. **AVS grading** — retrieve rộng 1000 → BGE grade 0–3 (prompt rubric) →
    threshold ≥2 → **MMR + cluster dedup** trên scene vectors (λ tune) → coverage
    check theo video. Per-video cap cũ giữ làm baseline ablation.
24. **UI React/Vite** — tabs KIS/VQA/AVS; bảng rank + evidence + latency từng
    branch (ăn `branches[]` §4.E17); video seek tới `best_timestamp_sec`;
    **export submission đúng format BTC (frame_idx)** — dùng `fps` từ
    videos.jsonl để đổi timestamp→frame chính xác; auth token + nút operator
    (clear cache, xem health/build_id).

### G. Hạ tầng

25. **Serving & warmup** — compose `infra/docker-compose.production.yml` (kèm
    branch): tách vllm-32b / vllm-14b / worker / rq-worker theo GPU; **preflight
    warmup**: backend chỉ nhận traffic sau khi mọi model trả lời request thử +
    manifest khớp collection alias (mở rộng `scripts/preflight.py`).
26. **Eval gate + error taxonomy** — mở rộng ground-truth (≥50 query/loại task
    trên dev set BTC); `eval_kis` chạy trong CI trước mỗi merge vào
    `server_implementation`; kết quả từng query ghi kèm nhãn lỗi (missed-recall /
    lost-at-rerank / wrong-video / ocr-miss...) → leaderboard nội bộ theo build_id.
27. **Monitoring** — Prometheus metrics từ backend/worker (latency từng branch,
    queue depth, GPU util qua DCGM exporter), Grafana dashboard, alert khi
    build_id backend ≠ manifest hoặc encoder mismatch; log JSON có `query_id`
    xuyên suốt → trace query→evidence; feedback: log click/chọn kết quả của
    operator thành replay set cho ablation sau.

## 5. Kế hoạch theo phase

| Phase | Nội dung | Nghiệm thu |
|---|---|---|
| 1 | Enrich production: A1–A4, B5–B9, C10–C11 chạy queue (C12) trên 2 GPU | Enrich trọn dev set; quarantine/audit hoạt động; caption/OCR soi mắt ≥90% đúng; resume giữa chừng không hỏng dữ liệu |
| 2 | Index + eval: D13–D15, mở rộng ground-truth, baseline mới | eval_kis: R@1 fusion > 0 và video-R@100 cải thiện rõ so với bản mock; alias publish/rollback demo được |
| 3 | Online: E16–E20, F21–F23 | Ablation từng mục thắng baseline; VQA verifier bắt được case sai do thiếu evidence; giả lập ES/vLLM chết → branch-status degraded hiển thị, không 503 toàn phần khi nhánh phụ chết |
| 4 | F24 UI + G25–G27 ops | Demo end-to-end UI local → backend; export submission đúng format frame_idx; dashboard + alert chạy; load test p95 đạt ngưỡng đề ra |

## 6. Rủi ro & điểm mở

- **2 GPU là cấu hình tối thiểu** cho 32B bf16 + phần còn lại; nếu chỉ thuê được
  1× 80GB thì phải hạ 32B xuống AWQ và benchmark lại chất lượng caption/VQA trước
  khi chấp nhận.
- **Checkpoint + phiên bản phải pin lúc dựng máy**: Qwen3-VL-32B revision, Qwen3-14B
  AWQ, SigLIP2 checkpoint, TransNetV2 weights (port PyTorch cộng đồng), plugin
  analyzer tiếng Việt cho Elasticsearch — tất cả xác minh trên máy thuê, ghi vào
  manifest.
- **PaddleOCR chung CUDA context với torch** đôi khi xung đột — fallback CPU mode
  hoặc process riêng nếu gặp.
- **Latency serving khi vLLM-32B bận rerank/VQA đồng thời**: trần số call tầng-2 +
  đo p95 bằng load_test là bắt buộc trước thi; nếu nghẽn → tách vLLM-32B sang node
  riêng (kiến trúc đã cho phép).
- **WhisperX/pyannote cần HF token và license gate** (pyannote) — chuẩn bị trước
  khi dựng máy.
- Elasticsearch vs giữ BM25 in-memory: corpus thi đấu lớn (nghìn video) mới phát
  huy ES; nếu dev set nhỏ thì Phase 2 có thể chạy song song cả hai và so eval_kis
  trước khi cắt hẳn.
