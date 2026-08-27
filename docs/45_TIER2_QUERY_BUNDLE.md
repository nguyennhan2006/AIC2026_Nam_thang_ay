# Tier 2 — LLM soạn dữ liệu riêng cho từng search engine

## Vấn đề

Mỗi engine mạnh ở một loại dữ liệu khác nhau, nhưng trước đây cả bốn nhận
chung một chuỗi:

| engine | cơ chế | thứ nó THỰC SỰ cần |
|---|---|---|
| Jina CLIP | so ảnh với câu | một câu mô tả khung hình trông ra sao |
| BM25 caption | khớp từ | danh từ/động từ cụ thể + cách gọi khác |
| OCR | đọc chữ trên hình | chuỗi ký tự thật sự hiện trên màn hình |
| ASR | tìm trong lời nói | câu như người ta sẽ nói ra |

## Bằng chứng: cùng một frame, lệch 35 hạng

Gold `L21_V023` frame `25995`. Caption của chính keyframe đó:

> "một bàn tay đeo đồng hồ đang đổ chất lỏng vào bát trắng đặt trên **cân điện
> tử**, trong khi một **con cá** nhỏ màu tối nổi bật trong bát"

Hai truy vấn trỏ đúng frame này:

| truy vấn đưa cho CLIP | rank |
|---|---|
| "Bàn tay đeo đồng hồ đổ chất lỏng vào bát trắng đặt trên cân điện tử" | **1** (Δ=0) |
| "một con cá được đặt lên cân, sau đó… Con số hiển thị cuối cùng trên cân" | 35 |

Cùng index, cùng frame. Khác biệt là **mô tả khung hình** so với **kể chuyện +
hỏi**. Tầng rule cắt được phần hỏi nhưng không viết lại được câu kể thành mô
tả thị giác — đó là việc của Tier 2.

## Kiến trúc

```
truy vấn
   |
   +-- Tier 1  rule (LUÔN chạy, <1ms, tất định)
   |             cắt phần hỏi, tách event, phân loại intent
   |
   +-- Tier 2  LLM (tuỳ chọn)
                 VIẾT LẠI từng trường cho đúng thế mạnh mỗi engine
                 hỏng -> giữ nguyên bundle của Tier 1
```

`FptQueryBundlePreparer.refine()` chỉ ghi đè trường nào LLM thật sự cải thiện:

- text ngắn hơn `_MIN_TEXT_CHARS` bị từ chối (chặn đúng kiểu output `"ao sau"`)
- `answer_type` lạ bị bỏ qua
- cụm trong ngoặc kép do rule trích **luôn** đứng trước phần LLM suy đoán
- `ProviderError` / JSON hỏng / trường rỗng → giữ nguyên bundle rule

Cache hai tầng (bộ nhớ + `storage/cache/query_bundle/`) nên eval lặp không gọi lại.

## Chạy

```powershell
.\run_server_full.ps1     # baseline .env  (thuần rule)
.\run_server_tier2.ps1    # Tier 2 .env.tier2.local
```

```powershell
python test_10_queries.py
```

## So sánh cho đúng

`.env.tier2.local` sao từ `.env.fpt.local` và **chỉ** khác:

```
AIC_ENABLE_LLM_QUERY_BUNDLE=true     # công tắc Tier 2
AIC_RUNTIME_PROFILE / AIC_PIPELINE_VERSION / AIC_STATE_DIR   # tách cache
```

Muốn quy chênh lệch về đúng Tier 2 thì baseline phải là `.env.fpt.local`,
**không phải** `.env` mặc định — `.env.fpt.local` còn bật
`QUERY_TRANSLATION`, `OCR_FUZZY`, `EVENT/OBJECT/ACTION/COLOR_SEARCH`,
`EXPANSION`, `RULES`. So với `.env` là trộn hai nguyên nhân vào một con số.

```powershell
$env:AIC_ENV_FILE=".env.fpt.local";   .\run_server_full.ps1   # baseline đúng
$env:AIC_ENV_FILE=".env.tier2.local"; .\run_server_full.ps1   # tier 2
```

## Preflight (1 lệnh gọi LLM, không cần boot server)

```powershell
$env:AIC_ENV_FILE=".env.tier2.local"
python -c "from online.config import Settings; from online.adapters.fpt_client import FptClient; from online.adapters.fpt_query_bundle import FptQueryBundlePreparer; from online.services.query.router import QueryRouter; from online.domain.models import SearchRequest, TaskType; s=Settings.from_env(); r=QueryRouter().prepare_sync(SearchRequest(query='Con cá trên cân nặng bao nhiêu?', task=TaskType.QA)); print(FptQueryBundlePreparer(FptClient.from_settings(s), model_id=s.fpt_fast_llm_model).refine(r, task='QA').visual_query)"
```

Kết quả preflight đã chạy (2026-08-27, `gemma-4-31B-it`), cả 7 trường được áp:

```
rule    visual : một con cá được đặt lên cân, sau đó có cảnh… cầm đuôi. Con số hiển thị cuối cùng trên cân
tier2   visual : Cận cảnh màn hình điện tử của một chiếc cân hiển thị con số khi có con cá đặt lên trên.
tier2   vis_en : Close-up of a digital scale display showing a number with a fish placed on it.
tier2   caption: cân điện tử, số cân, trọng lượng, con cá, cá
tier2   asr    : Con cá này nặng bao nhiêu cân nhỉ?
```

## Bẫy: boot treo ở phase `encoder`

Triệu chứng: `/v1/startup` đứng ở `{"phase":"encoder"}` hàng chục phút, tiến
trình chiếm ~3.7GB RAM nhưng **CPU = 0** (đo bằng `Get-Process` hai lần cách
nhau 6s). CPU bằng 0 nghĩa là nó không tính toán mà đang **chờ I/O mạng**.

Nguyên nhân: `storage/models/clip-vit-large-patch14` đã có đủ file, nhưng
`transformers` vẫn gọi HuggingFace Hub để đối chiếu revision trước khi dùng
bản local. Máy này chặn SSL tới HF nên lệnh đó không lỗi ngay mà treo tới
timeout.

Cách sửa — và **chỗ đặt mới là phần quan trọng**:

```powershell
# ĐÚNG: biến môi trường, trước khi python khởi động (run_server_tier2.ps1)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m uvicorn online.api.app:app --port 8001
```

```
# SAI: đặt trong .env — KHÔNG có tác dụng
HF_HUB_OFFLINE=1
```

`huggingface_hub` đọc `HF_HUB_OFFLINE` **một lần lúc import module** rồi đóng
băng thành hằng số. `Settings.from_env()` nạp file `.env` *sau* khi
`transformers` đã import xong, nên biến đặt trong `.env` tới quá muộn. Kiểm
chứng trực tiếp:

```
import transformers.utils.hub as hub
os.environ["HF_HUB_OFFLINE"] = "1"
hub.constants.HF_HUB_OFFLINE   # vẫn là False
```

Đo được khi đặt đúng chỗ: model nạp trong **0.5 giây** thay vì treo 15+ phút.

```powershell
# kiểm nhanh, không cần boot server
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('storage/models/clip-vit-large-patch14', local_files_only=True); print('OK')"
```

`.env.fpt.local` gốc cũng không có hai biến này (mà kể cả có cũng vô tác dụng
vì lý do trên), nên khi chạy baseline để so sánh, phải export chúng ở shell —
nếu không nó treo y hệt.

## Bẫy: một CẢNH BÁO làm sập server (đã sửa ở nguồn)

`container.py` in cảnh báo coverage caption_dense bằng `print()` tiếng Việt.
stdout trên Windows mặc định là `cp1258`, không encode nổi `ẫ` trong "vẫn" →
`UnicodeEncodeError` giết cả tiến trình boot:

```
{"status":"failed","phase":"failed",
 "error":"UnicodeEncodeError: 'charmap' codec can't encode character '\\u1eab'"}
```

Nhánh này chỉ chạy khi coverage < 98%, nên nó nằm im cho tới đúng lúc dữ liệu
có vấn đề — tức là sập đúng lúc cần đọc cảnh báo nhất.

Đã sửa: đổi sang `logger.warning` (logging tự nuốt lỗi encode của handler thay
vì ném lên caller). Script cũng đặt `PYTHONIOENCODING=utf-8` làm lớp phòng thủ
thứ hai. Quét lại toàn bộ `online/`: không còn `print()` non-ASCII nào khác.

## Bảo mật

`.env.tier2.local` chứa API key thật. `.gitignore:42` (`.env.*.local`) đã bắt
nó — kiểm lại bằng `git check-ignore -v .env.tier2.local`.
