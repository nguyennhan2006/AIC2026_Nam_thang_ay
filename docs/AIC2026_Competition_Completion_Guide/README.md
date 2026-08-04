# AIC 2026 — Bộ hướng dẫn hoàn thiện hệ thống thi đấu

Bộ tài liệu này dùng để đưa hệ thống AIC 2026 từ trạng thái **pipeline nghiên cứu/backend retrieval** thành một **competition system hoàn chỉnh**, bao gồm:

- Offline enrichment và indexing có thể tái lập.
- Online hybrid retrieval cho KIS, VQA và AVS.
- Giao diện thao tác nhanh, evidence-centric và hỗ trợ exact-frame selection.
- Progressive clue, event/temporal navigation và relevance feedback.
- Submission proxy, validation, log và diễn tập thi đấu.
- Production readiness, reliability, monitoring và fallback.
- Research agenda có hypothesis, cách đo và điều kiện chấp nhận.

## 1. Nguyên tắc sử dụng

1. **Không bật mặc định tính năng chưa thắng ablation.** Mọi module mới phải đứng sau feature flag, có baseline và có kết quả đo.
2. **Không silent degradation.** Nhánh timeout, model lỗi, index stale hoặc fallback đều phải xuất hiện trong response và UI.
3. **Raw query luôn được giữ lại.** Query parser/expansion chỉ tạo thêm biến thể, không được thay thế truy vấn gốc.
4. **Mọi kết quả phải map ngược được.** Từ vector/document phải truy ra `video_id`, `segment_id`, `frame_idx`, timestamp, source path và model/index version.
5. **UI và submission là một phần của thuật toán thi đấu.** Không để tới cuối mới triển khai.
6. **KIS, VQA, AVS là ba task chính.** Sequence/temporal là năng lực phụ trợ bên trong KIS hoặc VQA.
7. **Không chốt theo cảm giác.** KIS dùng Recall/MRR; VQA dùng answer + evidence; AVS dùng mAP/nDCG/diversity; UI dùng time-to-correct/time-to-submit.

## 2. Thứ tự đọc khuyến nghị

1. `docs/01_SYSTEM_SCOPE_AND_SUCCESS.md`
2. `docs/02_TARGET_ARCHITECTURE.md`
3. `docs/03_COMPLETE_FUNCTION_CATALOG.md`
4. `docs/04_ONLINE_WORKFLOW_AND_UI.md`
5. `docs/05_QUERY_RETRIEVAL_RERANKING.md`
6. `docs/06_API_AND_DATA_CONTRACTS.md`
7. `docs/07_SUBMISSION_AND_COMPETITION_OPERATIONS.md`
8. `docs/08_RELIABILITY_SECURITY_OBSERVABILITY.md`
9. `docs/09_EVALUATION_GROUNDTRUTH_ABLATION.md`
10. `docs/10_IMPLEMENTATION_ROADMAP.md`
11. `docs/11_UI_ACCEPTANCE_AND_E2E.md`
12. `docs/12_TEAM_RUNBOOK.md`
13. `docs/13_PRODUCTION_READINESS_INFO.md`
14. `docs/14_TECHNICAL_PREPARATION.md`
15. `docs/15_RESEARCH_AGENDA.md`
16. `docs/16_MASTER_CHECKLIST.md`
17. `docs/REFERENCES_AND_DESIGN_LESSONS.md`

## 3. Cấu trúc package

```text
AIC2026_Competition_Completion_Guide/
├── README.md
├── docs/
│   ├── 01_SYSTEM_SCOPE_AND_SUCCESS.md
│   ├── 02_TARGET_ARCHITECTURE.md
│   ├── 03_COMPLETE_FUNCTION_CATALOG.md
│   ├── 04_ONLINE_WORKFLOW_AND_UI.md
│   ├── 05_QUERY_RETRIEVAL_RERANKING.md
│   ├── 06_API_AND_DATA_CONTRACTS.md
│   ├── 07_SUBMISSION_AND_COMPETITION_OPERATIONS.md
│   ├── 08_RELIABILITY_SECURITY_OBSERVABILITY.md
│   ├── 09_EVALUATION_GROUNDTRUTH_ABLATION.md
│   ├── 10_IMPLEMENTATION_ROADMAP.md
│   ├── 11_UI_ACCEPTANCE_AND_E2E.md
│   ├── 12_TEAM_RUNBOOK.md
│   ├── 13_PRODUCTION_READINESS_INFO.md
│   ├── 14_TECHNICAL_PREPARATION.md
│   ├── 15_RESEARCH_AGENDA.md
│   ├── 16_MASTER_CHECKLIST.md
│   └── REFERENCES_AND_DESIGN_LESSONS.md
└── templates/
    ├── competition_release_manifest.yaml
    ├── search_profiles.yaml
    ├── submission_contract.yaml
    ├── experiment_record.yaml
    ├── env.production.example
    ├── groundtruth_examples.jsonl
    └── competition_drill_report.md
```

## 4. Quy ước trạng thái

- `TODO`: chưa bắt đầu.
- `IN_PROGRESS`: đang triển khai.
- `BLOCKED`: thiếu dữ liệu, rule, checkpoint hoặc hạ tầng.
- `EXPERIMENTAL`: đã code nhưng chưa thắng ablation.
- `READY`: đạt nghiệm thu kỹ thuật.
- `COMPETITION_READY`: đã qua full drill và submission test.

## 5. Definition of Competition Ready

Một release chỉ được đánh dấu `COMPETITION_READY` khi đồng thời đạt:

- Preflight tất cả model/index pass.
- UI restore session và exact-frame selection pass E2E.
- Submission endpoint test thành công với payload mẫu.
- Không có index/model build mismatch.
- Có fallback cho ES/Qdrant/VLM/competition server.
- Có ít nhất một full competition drill không có lỗi P0.
- Metric không thấp hơn baseline đã freeze.
- Release manifest đã ký duyệt bởi technical lead và operator lead.
