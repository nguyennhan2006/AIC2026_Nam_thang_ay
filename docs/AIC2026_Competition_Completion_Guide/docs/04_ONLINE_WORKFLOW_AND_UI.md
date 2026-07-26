# 04. Workflow online và thiết kế giao diện thi đấu

## 1. Ba chế độ giao diện

### Competition Mode

- Tối giản.
- Keyboard-first.
- Model/index profile bị khóa.
- Query, results, evidence, submission luôn hiện.
- Autosave và countdown.

### Research Mode

- Parsed query.
- Branch scores.
- Model/index selector.
- Compare runs.
- Raw metadata và error tagging.

### Practice Mode

- Replay câu hỏi.
- Reveal ground truth sau khi hoàn thành.
- Đo time-to-correct/time-to-submit.
- Báo cáo lỗi người dùng và hệ thống.

## 2. Bố cục chính

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ AIC 2026 | KIS VQA AVS | profile | API● GPU● INDEX● | countdown         │
├────────────────┬────────────────────────────────┬────────────────────────┤
│ QUERY PANEL    │ RESULTS                        │ EVIDENCE INSPECTOR     │
│ raw query      │ grid/list/timeline/event       │ video player           │
│ clue history   │ candidate cards                │ frame strip             │
│ parsed chips   │ group by video/event           │ metadata tabs           │
│ filters        │ pin/hide/select                │ exact frame/select crop │
├────────────────┴────────────────────────────────┴────────────────────────┤
│ TASK BASKET / SUBMISSION | validate | send | history                    │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. Query Panel

- Task selector.
- Raw query textarea.
- Progressive clue append.
- Search mode: fast/balanced/quality.
- Top-k.
- Filters.
- Parsed entity chips.
- Must-match/nice-to-have/negative.
- Variant list Q0/Q1/Qn.
- Branch routing switches.
- Search, cancel, restore default.

### Quy tắc UX

- Q0 raw query luôn hiển thị.
- Expansion mới phải phân biệt bằng màu/label.
- Parser confidence thấp phải cảnh báo.
- Không yêu cầu người thi nhập quá nhiều field từ đầu.

## 4. Result Cards

Mỗi card có:

- Rank.
- Video/scene/frame/timestamp.
- Thumbnail.
- Overall score.
- VIS/CAP/OCR/ASR/OBJ/TMP badges.
- Short evidence.
- Deep-rerank status.
- Pin/hide/add/open/more-like-this.

## 5. Result Views

- Grid: scan nhanh.
- List: metadata chi tiết.
- Timeline: temporal query.
- Group by video.
- Event view: cluster và trước/sau.
- Storyboard: ordered sequence.

## 6. Evidence Inspector

Tabs:

- Summary.
- Caption.
- OCR.
- ASR.
- Objects/actions.
- Event/temporal.
- Scores.
- Raw.

Video player:

- Seek segment.
- Loop.
- ±2/5/10/30 giây.
- Frame step.
- Current frame index.
- Select current frame.
- Search from frame/crop.

## 7. KIS workflow

```text
Nhập clue đầu
→ search Q0 + variants
→ pin candidates
→ clue mới được append
→ rank history cập nhật
→ mở video/neighbor event
→ chọn exact frame
→ validate
→ submit
```

UI phải giữ pin và selected evidence khi clue mới tới.

## 8. VQA workflow

```text
Question
→ parse answer/evidence type
→ retrieve evidence
→ evidence table
→ rule/tool/VLM answer
→ verifier
→ human review
→ submit answer + evidence
```

Không hiển thị reasoning dài; chỉ hiển thị tool/evidence trace có thể kiểm chứng.

## 9. AVS workflow

```text
Query + criteria
→ broad retrieval
→ relevance grade
→ dedup/diversify
→ cluster review
→ bulk select
→ basket validation
→ submit/export
```

## 10. Keyboard shortcuts đề xuất

| Phím | Hành động |
|---|---|
| Ctrl+Enter | Search |
| 1–9 | Mở result tương ứng |
| P | Pin |
| H | Hide |
| A | Add to basket |
| J/L | Lùi/tiến video |
| ←/→ | Frame trước/sau |
| Shift+←/→ | ±1 giây |
| E | Evidence panel |
| S | Validate/submit dialog |

## 11. Trạng thái bắt buộc

- Idle.
- Parsing.
- Retrieving per branch.
- Fusion.
- Reranking.
- Verifying.
- Completed.
- Completed with warnings.
- No results.
- Cancelled.
- Timeout.
- Backend unavailable.
- Index mismatch.
- Submission queued/failed/succeeded.

## 12. Không được làm

- Autoplay nhiều video cùng lúc.
- Ẩn branch fail.
- Chỉ hiển thị một score tổng.
- Để token competition ở frontend.
- Cho submit timestamp khi rule yêu cầu frame index.
- Mất session khi refresh.
