# 11. UI acceptance và E2E tests

## 1. Công cụ

- Unit/component: Vitest + Testing Library.
- E2E: Playwright.
- API contract: generated TypeScript types/OpenAPI check.
- Visual regression chỉ dùng cho layout quan trọng, không thay functional test.

## 2. Core E2E

### Search

- Nhập query và nhận results.
- Hiển thị branch progress.
- Branch timeout hiển thị warning.
- Cancel request.
- Restore session sau refresh.

### Result/evidence

- Mở candidate.
- Video seek đúng start time.
- Frame step đúng.
- Frame index hiển thị đúng.
- Neighbor frame/scene/event hoạt động.
- OCR/ASR highlight đúng.

### KIS

- Progressive clue append.
- Pin không mất khi rerun.
- Select current frame.
- Validate and submit mock.
- Duplicate submit bị chặn.

### VQA

- Evidence table.
- Unsupported answer hiển thị abstain.
- Human replace evidence.
- Submit answer mock.

### AVS

- Grade candidates.
- Toggle diversity.
- Bulk add/remove.
- Validate max count.

## 3. Accessibility/UX

- Keyboard navigation.
- Focus states.
- Không chỉ dùng màu để biểu diễn modality/status.
- Dark mode.
- Font/readability trong phòng thi.
- Không loading spinner vô hạn.

## 4. Performance budgets cần đo

- First paint.
- Search interaction responsiveness.
- 500 result virtualization.
- Thumbnail load.
- Video seek.
- Session autosave.

## 5. Acceptance checklist

- Không mất pinned result.
- Không submit sai frame do rounding.
- Không expose token.
- Không che warning/degraded state.
- Không crash khi result thiếu OCR/ASR.
- Không phụ thuộc chuột cho thao tác chính.
