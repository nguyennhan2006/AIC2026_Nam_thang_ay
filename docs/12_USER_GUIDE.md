# 12. Chạy hệ thống — hướng dẫn đã kiểm chứng thực tế

Mọi lệnh trong tài liệu này **đã được chạy thật** trên máy Windows của dự án
(branch `server_implementation`, commit `dfe5c86`) và kết quả in ra ở đây là
output thật, không phải ví dụ minh hoạ. Nếu bạn chạy đúng thứ tự mà ra khác,
đó là sự cố cần tra ở §7 chứ không phải "tài liệu viết chung chung".

Bản trước của file này mô tả API và UI đã lỗi thời nên làm người dùng không
chạy đúng được — bản này viết lại từ đầu theo đúng những gì đang chạy.

---

## 0. TL;DR — 4 lệnh để có UI chạy

Mở **hai** terminal PowerShell tại thư mục gốc repo.

**Terminal 1 — backend:**

```powershell
$env:AIC_METADATA_JSONL = "storage/exports_l21/scenes.jsonl"
$env:AIC_DATA_ROOT = "storage"
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — giao diện:**

```powershell
cd online/ui-react
npm install          # chỉ lần đầu
npm run dev
```

Mở `http://127.0.0.1:5173`. Gõ truy vấn, bấm **Tìm kiếm** (hoặc Ctrl+Enter).

Muốn thấy ngay giao diện ở trạng thái có dữ liệu mà không phải gõ gì:
`http://127.0.0.1:5173/?demo=1` — nó tự chạy **một truy vấn thật** lên backend
và hiện badge `demo` trên thanh nav. Không có dữ liệu giả nào được nhúng.

---

## 1. Chuẩn bị môi trường (chỉ làm một lần)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[api,faiss,test]"
```

Node.js ≥ 20 cho giao diện (máy dự án đang dùng v24.11.0, npm 11.6.3).

**Không có dotenv loader trong code.** File `.env`/`.env.fpt.local` chỉ để tra
cứu tên biến — muốn có tác dụng phải `$env:TÊN = "giá trị"` trong đúng shell
sẽ chạy lệnh. Đây là lỗi hay gặp nhất khi "sửa .env mà không thấy gì đổi".

**Đừng** copy nguyên `.env.fpt.local` thành `.env`: file đó trỏ
`AIC_METADATA_JSONL` vào `storage/exports/fpt_acceptance/scenes.jsonl` — một
đường dẫn của thí nghiệm cũ, hiện **không tồn tại**, nên backend sẽ khởi động
với 0 scene. Chỉ lấy đúng các biến `AIC_FPT_*` bạn cần từ đó.

---

## 2. Chọn dataset

| Dataset | `AIC_METADATA_JSONL` | Khi nào dùng |
|---|---|---|
| **L21_V001 thật** (khuyến nghị) | `storage/exports_l21/scenes.jsonl` | Có caption/OCR/object thật + embedding CLIP → search ra kết quả có nghĩa |
| Demo 3 scene | `storage/exports/scenes.jsonl` | Chỉ smoke test plumbing, caption là placeholder |

Kiểm tra dataset thật còn nguyên:

```powershell
python -m datasection.cli storage/exports_l21
```

Nếu `storage/exports_l21/` không còn, dựng lại theo
[17_MANUAL_TEST_GUIDE_L21_V001.md](17_MANUAL_TEST_GUIDE_L21_V001.md) §1.

---

## 3. Chạy backend

```powershell
$env:AIC_METADATA_JSONL = "storage/exports_l21/scenes.jsonl"
$env:AIC_DATA_ROOT = "storage"
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

Xác nhận (output thật của lệnh này trên dataset L21):

```powershell
curl.exe -s http://127.0.0.1:8000/v1/health
```

```json
{"status":"ok","backend":"local","scene_count":217,
 "dataset":"storage\\exports_l21\\scenes.jsonl",
 "dataset_version":"20260803T081508Z","branch_count":5,
 "session_store_enabled":true,
 "video_count":1,"keyframe_count":307,"asr_segment_count":383}
```

Bốn số cuối chính là 4 thẻ thống kê hiện ở góc phải giao diện — nếu thẻ hiện
`—` thì `dataset_manifest.json` thiếu cạnh file scenes, không phải lỗi UI.

### Khởi động lại đúng cách (đọc kỹ — đây là bẫy đã mất thời gian thật)

Backend chỉ đọc `scenes.jsonl` và biến môi trường **một lần lúc khởi động**.
Trên Windows, `uvicorn --reload` có thể để lại **tiến trình cũ vẫn giữ cổng
8000 dù `Stop-Process` báo thành công** — càng dễ bị che nếu dùng
`-ErrorAction SilentlyContinue`. Luôn xác nhận cổng đã trống:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
# Còn PID nào thì kill rồi KIỂM TRA LẠI, đừng tin lệnh kill là đã xong:
Stop-Process -Id <PID> -Force
```

---

## 4. Chạy giao diện

```powershell
cd online/ui-react
npm install      # lần đầu
npm run dev      # Vite, mặc định http://127.0.0.1:5173
```

Bản build tĩnh (ổn định hơn dev server khi trình diễn):

```powershell
npm run build
npx serve dist -l 5173
```

### CORS

UI mặc định gọi backend ở `http://localhost:8000` (đổi được trong ô **API
base**, chế độ **Advanced**). `AIC_CORS_ORIGINS` mặc định đã whitelist **cả
hai** origin `http://localhost:5173` và `http://127.0.0.1:5173`, nên mở UI bằng
tên nào cũng chạy — miễn backend thật sự nghe ở cổng 8000.

Chỉ khi bạn đổi cổng/tên miền của UI mới cần thêm origin vào
`AIC_CORS_ORIGINS`, và **phải khởi động lại backend** vì biến này không
hot-reload. Nếu backend nghe ở địa chỉ khác (máy xa, cổng khác), sửa ô **API
base** cho khớp chứ không sửa CORS.

`online/ui/` (bản vanilla cũ) vẫn được backend phục vụ tại
`http://127.0.0.1:8000/ui/`, dùng khi cần smoke test không cài Node.js. Bản này
**không** được cập nhật theo contract mới — mọi việc thật dùng `online/ui-react/`.

---

## 5. Dùng giao diện

Bố cục ba cột, khoá trong đúng một màn hình (không có scrollbar ở trang; mỗi
panel tự cuộn):

```
Task | Top-K | Simple/Advanced          [ 4 thẻ dataset ]
┌ ô truy vấn ──────────────────────┐
├ Trọng số 292 ┬ Kết quả ─────────┬ Preview & Details 372 ┤
```

**Truy vấn** — chọn task bằng chip KIS/QA/TRAKE/AVS, gõ, Ctrl+Enter. Chế độ
**Advanced** mới hiện API base / API token / stream SSE / nút kiểm tra server.

**Panel Trọng số** (cột trái) — mọi control đọc từ
`GET /v1/search/capabilities`, không hard-code: nhánh nào server không có thì
không hiện. Kéo slider **không** tự chạy lại search; giá trị chỉ được gửi lên
khi bạn bấm Tìm kiếm (badge `chưa áp dụng` cho biết còn thay đổi chưa gửi).

> **Top-K trả về ít hơn số bạn đặt?** Đây là hành vi đúng chứ không phải bug:
> dedup của task KIS giữ tối đa **5 kết quả mỗi video**, mà dataset hiện chỉ
> có 1 video. Đặt **Max / video** trong nhóm *Fusion & Ranking* (vd 20) để nới.
> Giao diện tự nói điều này dưới lưới kết quả khi phát hiện bị cắt.

**Kết quả** (cột giữa) — 3 tab: *Lưới ảnh* (card kèm đóng góp từng nhánh),
tab theo task (KIS frames / QA answers / Sequences / Segments), *Submission*.
Bấm một card để đưa nó sang cột phải.

**Preview & Details** (cột phải) — *Preview* (video tua đúng thời điểm, hoặc
keyframe), *Evidence* (gọi `GET /v1/evidence/{id}`), *Trace* (trạng thái từng
nhánh, modality weights, tải JSON đầy đủ). Sau mỗi lần search, kết quả đầu
được chọn sẵn nên cột này không bao giờ rỗng khi đã có dữ liệu.

Màn hẹp: <1280 ẩn cột Preview, <1024 ẩn thêm cột Trọng số, mobile xếp dọc.

---

## 6. Gọi API trực tiếp (không cần UI)

`top_k` mặc định 20. Convenience endpoint tự điền `task` theo path; endpoint
thống nhất `/v1/search` **bắt buộc** có `task` trong body.

**Cách truyền body trên PowerShell** — đừng nhét JSON thẳng vào `-d`:
PowerShell 5.1 nuốt mất dấu nháy và curl nhận được JSON hỏng
(`json_invalid ... Unterminated string`), lại thêm rủi ro hỏng dấu tiếng Việt.
Ghi body ra file UTF-8 rồi `--data-binary "@file"`. Ba lệnh dưới đây là output
thật đã chạy:

```powershell
# Danh sách nhánh + tầng rerank thật sự có — đọc trước khi set search_options
curl.exe -s http://127.0.0.1:8000/v1/search/capabilities

# KIS
Set-Content "$env:TEMP\q.json" -Encoding utf8 `
  '{"query":"cảnh báo sạt lở nguy hiểm ven sông","top_k":20}'
curl.exe -s -X POST http://127.0.0.1:8000/v1/search/kis `
  -H "Content-Type: application/json" --data-binary "@$env:TEMP\q.json"
# -> results=5  (bị dedup cắt, xem ghi chú Top-K ở §5)

# Nới giới hạn kết quả mỗi video (chính là ô "Max / video" trên UI)
Set-Content "$env:TEMP\q2.json" -Encoding utf8 `
  '{"task":"TEXTUAL_KIS","query":"cảnh báo sạt lở nguy hiểm ven sông","top_k":20,"search_options":{"fusion":{"max_results_per_video":20}}}'
curl.exe -s -X POST http://127.0.0.1:8000/v1/search `
  -H "Content-Type: application/json" --data-binary "@$env:TEMP\q2.json"
# -> results=20
```

Trên Git Bash / Linux thì `-d '{"query":"..."}'` bình thường, không cần file.

Các endpoint còn lại: `/v1/search/qa|trake|avs`, `/v1/search/stream` (SSE),
`/v1/evidence/{candidate_id}`, `/v1/scenes/{scene_id}`,
`/v1/search-sessions/{id}` + `/replay`, `/v1/submissions/build|validate|
evaluate-local`, `/v1/media/{path}`. `/v1/vqa` và `/v1/search/sequence` vẫn
còn nhưng đã `deprecated` — dùng `/v1/search/trake` thay cho sequence.

Đặt một field chưa được cài đặt thật trong `search_options` sẽ nhận **422** kèm
lý do, thay vì bị nhận rồi lờ đi. Danh sách ở
`online/services/capabilities.py::UNSUPPORTED`, và UI hiển thị nó trong mục
"N option server chưa chạy thật".

---

## 7. Bật FPT AI Marketplace (rerank + QA answer bằng LLM)

Không có GPU rời/máy thuê thì FPT thay tạm. Set **trước khi** khởi động backend:

```powershell
$env:AIC_FPT_ENABLED = "true"
$env:AIC_FPT_API_KEY = "<api key thật>"
$env:AIC_FPT_RERANK_MODEL = "bge-reranker-v2-m3"   # bật rerank.text
$env:AIC_FPT_LLM_MODEL = "Qwen3.6-27B"             # bật QA answer bằng LLM
```

Xác nhận rerank: `GET /v1/search/capabilities` phải trả `"rerank":{"text":true,…}`.
Xác nhận QA LLM: xem field `source` trong `qa[]` (`fpt_llm` thay vì `ocr_exact`
…), hoặc `warnings` nếu key/model sai.

⚠ `.env.example` còn khai báo `AIC_FPT_QUERY_LLM_MODEL` /
`AIC_FPT_FAST_LLM_MODEL` / `AIC_FPT_DEEP_LLM_MODEL` — **code hiện chưa đọc ba
biến này**. Chỉ `AIC_FPT_LLM_MODEL` có tác dụng (dùng cho QA).

`AIC_FPT_QA_TOP_N` (mặc định 5) giới hạn số evidence pack được LLM xử lý mỗi
câu, để chi phí/độ trễ đoán trước được. Rule-based `ANSWER_TOOLS` vẫn luôn
chạy làm nền, vì luật chấm QA tính bất kỳ dòng nào đúng cả ba
(video/frame/answer), không riêng dòng đầu.

---

## 8. Đo chất lượng (bắt buộc trước khi kết luận cải tiến nào tốt hơn)

```powershell
# KIS: Recall@K / MRR trên 4 mode retrieval
python -m scripts.eval_kis --metadata storage/exports_l21/scenes.jsonl `
  --groundtruth examples/kis_groundtruth.jsonl --mode all

# Cả 4 task trên gold set L21_V001
python -m scripts.eval_tasks --metadata storage/exports_l21/scenes.jsonl
```

Cờ ablation cho cả hai: `--use-query-prep --use-rules --use-expansion
--use-rerank`. `--use-rerank` bật **cả** text rerank lẫn QA answer qua FPT
(cần env ở §7); thiếu env thì nó báo lỗi rõ ràng chứ không âm thầm chạy không
rerank. Đừng kết luận gì nếu chưa chạy bảng này trước **và** sau khi đổi.

---

## 9. Lệnh kiểm tra

```powershell
python -m pytest tests/ -q                    # 465 pass
cd online/ui-react
npx tsc -b                                    # typecheck
npx vitest run                                # 9 pass
npx oxlint                                    # lint
npm run build                                 # production build
node scripts/visual-check.mjs                 # xem §10
```

---

## 10. Kiểm tra giao diện tự động (`scripts/visual-check.mjs`)

Cần backend + dev server đang chạy. Script mở Chromium thật ở
**1920 / 1708 / 1440 / 1280 / mobile 390**, chụp cả trạng thái rỗng lẫn sau khi
chạy một search thật, lưu vào `online/ui-react/screenshots/`, và **tự fail** nếu:

- `body`/`#root` sinh scrollbar dọc hoặc ngang,
- phần tử nào tràn ra ngoài chiều ngang viewport (trừ vùng `.scroll-x` cố ý),
- nhãn trong Weight Panel / stat card / tab bị wrap,
- slider đè lên ô số trong hàng trọng số,
- panel nào rỗng trơn (không nội dung, không empty state),
- cột Preview vẫn rỗng sau khi đã có kết quả.

Lần chạy gần nhất: **5/5 viewport ĐẠT**. Sửa CSS/layout xong hãy chạy lại lệnh
này trước khi nói là xong — nó đã bắt được ba lỗi thật mà đọc code không thấy
(xem commit `dfe5c86`).

---

## 11. Sự cố thường gặp

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| Sửa code/env, restart, backend vẫn như cũ | `uvicorn --reload` để lại tiến trình cũ giữ cổng 8000 dù `Stop-Process` báo OK | `Get-NetTCPConnection -LocalPort 8000 -State Listen` → kill PID còn lại → **kiểm tra lại** rồi mới start |
| `scene_count` = 0 hoặc rất nhỏ | `AIC_METADATA_JSONL` trỏ sai (hay gặp: copy `.env.fpt.local` trỏ vào `exports/fpt_acceptance/` không tồn tại) | Set lại đúng `storage/exports_l21/scenes.jsonl` rồi restart |
| Top-K = 20 nhưng chỉ ra 5 kết quả | Dedup KIS giữ tối đa 5 kết quả/video, dataset chỉ có 1 video | Đặt **Max / video** ở panel Trọng số (hoặc `fusion.max_results_per_video` qua API) |
| `Access-Control-Allow-Origin` | Origin đang mở ≠ ô API base, hoặc chưa whitelist | §4; sửa `AIC_CORS_ORIGINS` xong **phải restart backend** |
| `404 no scene matched the query` | Query không khớp field nào đã index | Dùng truy vấn khớp `examples/AIC2026_L21_V001_queries_4tasks.jsonl`, hoặc kiểm tra caption/OCR đã sinh chưa |
| `422 task is required` | `POST /v1/search` bắt buộc có `task` trong body | Thêm `"task":"TEXTUAL_KIS"` hoặc gọi `/v1/search/kis` |
| `422` khi set `search_options` | Field đó chưa có consumer thật, bị chặn ở `UNSUPPORTED` để không "cấu hình giả vờ có tác dụng" | Đọc `/v1/search/capabilities` trước khi set |
| `.env` sửa mà không có tác dụng | Không có dotenv loader | Set `$env:` trực tiếp trong shell chạy lệnh |
| `curl` trả `json_invalid` / `Unterminated string` | PowerShell 5.1 nuốt dấu nháy trong `-d '{...}'` | Ghi body ra file UTF-8 rồi `--data-binary "@file"` — xem §6 |
| QA vẫn trả answer kiểu regex dù đã bật FPT | Thiếu `AIC_FPT_LLM_MODEL`, hoặc set nhầm 3 biến `QUERY/FAST/DEEP` chưa được wire | §7 |
| `ssl.SSLCertVerificationError: Basic Constraints of CA cert not marked critical` | Python 3.13+/OpenSSL bật `VERIFY_X509_STRICT`; CA trên máy này không tuân thủ chi tiết RFC đó | Gọi qua `FptClient` (đã vá đúng một cờ này), đừng dùng `urllib.request.urlopen` trực tiếp |
| Best Sequence (TRAKE) trống dù có kết quả | Đã sửa ở `dfe5c86` — trước đó phụ thuộc `query_plan` chỉ có khi `debug=true` | Cập nhật lên commit mới nhất |

---

## 12. Đọc tiếp

- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) — kiến trúc tổng thể
- [04_ONLINE_RETRIEVAL.md](04_ONLINE_RETRIEVAL.md) — lõi retrieval + 4 task processor
- [17_MANUAL_TEST_GUIDE_L21_V001.md](17_MANUAL_TEST_GUIDE_L21_V001.md) — dựng lại dataset L21 từ đầu
- [11_SERVER_IMPLEMENTATION.md](11_SERVER_IMPLEMENTATION.md) — profile thi đấu A100 (thiết kế, chưa code)
- [KAGGLE_OFFLINE_GUIDE.md](KAGGLE_OFFLINE_GUIDE.md) — chạy model thật khi máy local không đủ VRAM
