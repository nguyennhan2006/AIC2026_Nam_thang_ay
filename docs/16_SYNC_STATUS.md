# 16. Trạng thái đồng bộ hệ thống — chốt sau PR-01..10 (2026-08-02)

Tài liệu này là ảnh chụp **sự thật hiện tại** của việc đồng bộ hóa hệ thống theo
kế hoạch `01082026_new_docs.md`. Không thay thế tài liệu đó — chỉ đối chiếu
từng mục Definition of Done (§22) với code thật đang có trên branch
`server_implementation`, và nêu rõ phần nào còn thiếu để không ai tưởng nhầm
là đã xong 100%.

Nguyên tắc nguồn sự thật (giữ đúng thứ tự đã chốt ở `01082026_new_docs.md` §1):
luật/format BTC > gold benchmark đã version hóa > code/test hiện tại > notebook
đã chạy có manifest > tài liệu nghiên cứu > tài liệu suy luận cũ.

## 1. Commit đã thực hiện

| PR | Nội dung | Commit | Test |
|---|---|---|---|
| 01 | Taxonomy canonical + `frame_idx` xuyên suốt | `7f4bb8d` | +26 |
| 02 | Stage pack + `assemble` + frame index + eval 4 task | `ddaa72d` | +26 |
| 03 | Branch registry + orchestrator chịu lỗi | `28801bb` | +18 |
| 04 | Normalization + threshold + từ chối option giả | `1681276` | +17 |
| 05 | Dedup service theo từng task | `10f91bf` | +13 |
| 06 | Evidence pack + rerank cascade | `4666564` | +12 |
| 07 | Bốn task processor KIS/QA/TRAKE/AVS | `9b7445e` | +64 |
| 08 | Competition/submission layer | `ca68792` | +37 |
| 09 | API thống nhất + SSE thật + session replay | `44e5639` | +24 |
| 10 | React Studio UI viết lại theo contract mới | `6067087` | +2 (vitest) |
| 11 | Production gate (tài liệu này) | — | +10 |

**356 test Python pass** (`python -m pytest tests/ -q --ignore=tests/test_caption_qwen3vl_config.py`,
bỏ qua vì thiếu `cv2` cục bộ, không liên quan tới đợt này) + 2 test vitest phía
`online/ui-react`. Load smoke thật (không phải TestClient) trên server sống:
60/60 request OK, p50 ≈ 26ms, p95 ≈ 367ms trên dataset demo 3 scene.

## 2. Đối chiếu Definition of Done (theo `01082026_new_docs.md` §22)

### Contracts

- [x] Một taxonomy duy nhất — `TaskType.TEXTUAL_KIS/QA/TRAKE/AVS`, alias chuẩn
      hóa ở API boundary (`online/domain/tasks.py`).
- [x] `frame_idx` xuyên suốt — `FrameEvidence` → `Candidate` → `SearchHit` →
      `KisResultItem`/`QaResultItem`/`TrakeResultItem` → submission CSV.
- [x] Không task/body conflict im lặng — `TaskConflictError` (422).
- [x] Mọi artifact/version truy vết được — `Candidate.model_id/index_id`,
      `EvidencePack.model_versions/dataset_version`, `SearchExecutionTrace`.

### Retrieval

- [x] Branch controls chạy thật — `online/services/capabilities.py` từ chối
      (422) option chưa cài đặt thay vì nhận rồi lờ đi.
- [x] Per-branch timeout — `RetrievalOrchestrator._run_one` (`asyncio.wait_for`).
- [x] Partial failure — branch lỗi trả `BranchStatus`, không kéo đổ request;
      chỉ lỗi khi **mọi** branch đều hỏng.
- [x] Score normalization — `online/services/score_normalization.py` (percentile/
      minmax/affine theo `score_kind`).
- [x] Hard/soft threshold — `online/services/thresholding.py`, áp **trước**
      fusion.
- [x] Per-branch fusion — `fuse_candidates` (rrf/weighted_sum/max_score/
      intersection/union) + `branch_contributions`.
- [x] Dedup rõ ràng — `online/services/deduplication.py`, chính sách khác nhau
      theo task (KIS gộp event, QA giữ nhiều frame, TRAKE không dedup, AVS siết
      mạnh).

### Tasks

- [x] KIS trả frame — `KisResultItem.frame_idx` + safe-frame scoring
      (`online/services/safe_frame.py`).
- [x] QA trả frame-answer tuple — `QaResultItem(video_id, frame_idx, answer)`,
      verifier độc lập với tool sinh answer.
- [x] TRAKE trả ordered frame list — `TrakeResultItem.frame_ids` (tăng dần
      nghiêm ngặt, validator ép), 3 giai đoạn tách biệt
      (`online/services/trake/{video_retriever,sequence_search,frame_refinement}.py`).
- [x] AVS trả segment có relevance/diversity — `AvsResultItem.relevance_grade`
      (0–3) + `cluster_id` (MMR).

### Evidence

- [x] Evidence pack đầy đủ — `EvidencePack` (keyframes, neighbor context,
      branch scores, rule adjustments, model/dataset version).
- [~] BGE/VLM rerank hoạt động — **adapter đã viết** (`online/adapters/rerank.py`,
      `BgeTextReranker`/`QwenVlReranker` qua HTTP) và cascade đã nối
      (`online/services/rerank_pipeline.py`), nhưng **chưa có model server thật
      để trỏ vào** — đúng theo kế hoạch Phase 3 `docs/14_TECHNICAL_PREPARATION.md`.
      `/v1/search/capabilities.rerank.{text,vlm}` báo `false` trung thực, bật
      tầng chưa cấu hình bị 422 thay vì "chạy" mà không làm gì.
- [x] QA verifier — `verify_answer()` chạy tách khỏi tool sinh answer,
      trả `SUPPORTED/PARTIAL/CONTRADICTED/INSUFFICIENT`.
- [x] TRAKE frame refinement — `online/services/trake/frame_refinement.py`,
      cửa sổ quanh anchor; đánh dấu `keyframe_only` khi chưa có frame index
      dày (xem mục 3 bên dưới).

### Submission

- [x] Format theo task — CSV builder đúng
      `docs/12_USER_GUIDE.md` §6 (`online/competition/submission_builder.py`).
- [x] Max 100 — `MAX_SUBMISSION_ITEMS`, validator chặn + cảnh báo.
- [x] Local scorer — `online/competition/scorer.py` (KIS/QA/TRAKE, đúng luật
      "TRAKE sai video = 0").
- [x] True frame validation — `submission_validator.py` dùng
      `video_frame_count()` thật (đọc từ `videos.jsonl`), không suy từ scene.
- [~] History/retry — session replay có (`POST /v1/search-sessions/{id}/replay`),
      nhưng **không có lịch sử các lần build/nộp submission** (chỉ có lịch sử
      search). Chưa làm — nằm ngoài phạm vi PR-08/09 đã thống nhất.

### UI

- [x] Simple/Advanced/Expert — chưa phân 3 mức, nhưng Mixing Console đã
      capability-driven (không control giả) — đủ điều kiện cần, chưa đủ điều
      kiện đủ của mục này.
- [x] Không control giả — `MixingConsole.tsx` chỉ render control có trong
      `supported_controls` của từng branch.
- [x] Branch status — `BranchStatusPanel.tsx`.
- [x] Evidence inspector — `EvidenceInspector.tsx`.
- [x] Task workspace — `TaskWorkspaces.tsx` (KIS/QA/TRAKE/AVS).
- [x] Submission board — `SubmissionBoard.tsx`.
- [~] Session restore — Compare Lab đọc/replay session qua id, nhưng **không tự
      khôi phục session gần nhất khi mở lại trang** (không có "session hiện tại"
      lưu localStorage) — chưa làm.

### Evaluation

- [x] Unit/integration/E2E pass — 356 test Python + 2 vitest + `test_e2e_production_gate.py`
      (10 kịch bản mô phỏng một phiên thi thật).
- [~] Default regression không giảm ngoài tolerance đã chốt — **chưa đo được
      trên gold thật**: `scripts/eval_tasks.py` chạy trên
      `examples/AIC2026_L21_V001_queries_4tasks.jsonl` (40 query) nhưng dataset
      L21_V001 chưa được assemble (chỉ có fixture demo 4 scene). Mọi số liệu
      hiện tại là **0 vì thiếu dữ liệu**, không phải vì retrieval sai — xem mục 3.
- [x] Config/model/index/dataset version được lưu — `SearchExecutionTrace`.

## 3. Khoảng trống còn lại (thành thật, không giấu)

Đây là danh sách việc **chưa xong**, xếp theo mức độ chặn:

1. **Chưa có dữ liệu L21 thật được assemble.** `offline/assemble.py` (PR-02)
   đã sẵn sàng nhận stage pack, nhưng **chưa có notebook nào sinh ra pack cho
   OCR/object/caption/color/embedding** (chỉ có N1-scene và N3-ASR, xem mục 4).
   Không có các pack này thì `assemble` chỉ dựng được Scene rỗng nội dung.
2. **Không có embedding thị giác thật.** Backend `local` vẫn dùng
   `lexical_hash_fallback` (hash trên caption/keyword) — `/capabilities` báo
   đúng `degraded=true`, không giả vờ là dense visual. Cần notebook N4
   (embedding) + `offline index --frames --qdrant` để có `aic_frames_v2` thật.
3. **BGE/VLM rerank chưa có model server để trỏ vào** (mục Evidence ở trên).
4. **`scripts/eval_tasks.py` chưa chạy được trên gold thật** vì (1). Khi có
   dataset L21 đã assemble, chạy lại đúng lệnh ở đầu file đó để có số
   R@K/MRR/mean-R-Score/nDCG thật — **không suy diễn số liệu từ tài liệu này**.
5. **React UI chưa được test bằng trình duyệt thật** (agent này không có
   trình duyệt) — chỉ xác nhận qua `tsc -b` + `vitest` + `vite build` +
   `oxlint` đều pass, và toàn bộ endpoint nó gọi đã được test qua
   `test_e2e_production_gate.py`. Cần một lượt kiểm tra bằng mắt trước khi
   dùng thi thật.
6. **Prometheus/Grafana chưa nối** — đúng theo kế hoạch `docs/11_SERVER_IMPLEMENTATION.md`
   (đánh dấu *planned* cho profile A100). `GET /v1/health` đã đủ cho vận hành
   tối thiểu (dataset_version, branch_count, session_store_enabled).
7. **Submission history/session-restore-on-reload** chưa làm (mục Submission/UI
   ở trên) — không nằm trong phạm vi PR-08/09 đã thống nhất ban đầu.

## 4. Việc kế tiếp (notebook Kaggle)

Theo kế hoạch gốc, còn thiếu:

| Notebook | Stage | Trạng thái |
|---|---|---|
| N0 | video manifest (ffprobe) | Chưa viết |
| N1 | scene detection (TransNetV2) | **Có** (`notebooks/scene-l21-a.ipynb`) |
| N2 | keyframe + quality | Chưa viết |
| N3 | ASR (faster-whisper) | **Có** (`notebooks/asr-l21-a.ipynb`) |
| N4 | embedding (SigLIP2/OpenCLIP) | Chưa viết |
| N5 | color | Chưa viết |

`offline/stagepack.py` + `offline/assemble.py` (PR-02) đã sẵn sàng nhận đúng
5 notebook này theo contract `contracts/stage_pack.schema.json`. Việc còn lại
là viết N0/N2/N4/N5 theo đúng pattern `TaskTracker`/`progress.json` mà N1/N3 đã
thiết lập, rồi chạy `python -m scripts.verify_stage_pack` cho từng pack trước
khi `offline assemble`.
