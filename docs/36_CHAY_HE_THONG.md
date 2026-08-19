# 36 — Chạy hệ thống trên corpus thi đấu

Lập 2026-08-18. Dành cho **đồng đội chạy lần đầu** trên máy của mình.
Thay thế phần runbook của [docs/34 §7](34_COMPETITION_PACK_IMPORT.md), vốn
thiếu một bước và vì thế **không chạy được**.

---

## 1. Khởi động

Cần **hai terminal**. Backend và UI là hai tiến trình riêng biệt.

```powershell
# Terminal 1 - backend (cong 8000)
cd "<thu-muc-repo>"
.\scripts\run_competition.ps1

# Terminal 2 - UI thi dau (cong 5173)
cd "<thu-muc-repo>"
.\scripts\run_ui.ps1
```

Mở **http://localhost:5173** — KHÔNG phải 8000. Xem §8.

Script tự kiểm tra file thiếu, tự vá `config.json` của jina (xem §3), cảnh báo
nếu RAM không đủ, rồi khởi động uvicorn. **Đợi ~4 phút** tới dòng
`Application startup complete.` Đừng bấm gì trong lúc đó — nó đang nạp
87.742 scene, 176.707 vector và text tower jina.

Kiểm ngay sau khi lên:

```powershell
$key = (Select-String -Path .env.fpt.local -Pattern '^AIC_ONLINE_API_KEY=(.+)$').Matches.Groups[1].Value
curl.exe -s localhost:8000/v1/health | python -m json.tool
curl.exe -s -H "Authorization: Bearer $key" localhost:8000/v1/search/capabilities | python -m json.tool
```

Kỳ vọng `scene_count` **87742**, `video_count` **873**, `keyframe_count`
**176707**, và `dense_visual` với `backend_kind: "vector"`.

> Thấy tên nhánh là `lexical_hash_fallback` chứ không phải `dense_visual` nghĩa
> là **không đọc được vector** — hệ vẫn trả kết quả, vẫn 200, chỉ là tầng ngữ
> nghĩa đã tắt. Không có cảnh báo nào cho ca này, phải tự nhìn.

---

## 2. Cần có sẵn những gì

| Thứ | Đường dẫn | Dung lượng |
|---|---|---:|
| Export | `storage/exports_competition/` (5 file `.jsonl` + manifest) | 1,1 GB |
| Vector | `storage/processed/embeddings_pack/*.npy` (873 file) | 362 MB |
| Ảnh keyframe | `storage/processed/keyframes/<video>/frame_*.jpg` | ~28 GB |
| Model | `storage/models/jina-clip-v2` + `storage/models/jina-embeddings-v3` | ~4 GB |
| Khoá | `.env.fpt.local` | — |

**Cả 5 file export phải nằm CÙNG một thư mục**: `AIC_METADATA_JSONL` trỏ
`scenes.jsonl`, bốn file kia được tìm bằng `with_name()`.

Ảnh thiếu thì UI không hiện thumbnail và `/v1/media` trả 404, nhưng tìm kiếm
vẫn chạy. Vector thiếu thì mất nhánh mạnh nhất — xem cảnh báo ở §1.

---

## 3. Cái bẫy đã làm hỏng runbook cũ

`storage/models/jina-clip-v2/config.json` khai text tower là
`"hf_model_name_or_path": "jinaai/jina-embeddings-v3"` — một **repo trên
HuggingFace**. Hệ quả:

- `HF_HUB_OFFLINE=1` → `model_info` **ném lỗi** thay vì rơi về cache;
- bỏ cờ đó → máy bị chặn SSL tới `huggingface.co`.

Tức là **cả hai đường đều chết**, dù bản local đã nằm sẵn ở
`storage/models/jina-embeddings-v3`. `run_competition.ps1` tự sửa trường đó
thành đường dẫn tương đối và giữ bản gốc ở `config.json.orig`.

Tải lại model từ HuggingFace hay từ gói Drive sẽ ghi đè và làm lỗi quay lại —
script vá lại mỗi lần chạy nên không cần nhớ.

---

## 4. Vì sao mỗi biến môi trường lại đặt như vậy

| Biến | Giá trị | Lý do |
|---|---|---|
| `AIC_VISUAL_EMBEDDING_NAME` | `jina_clip_v2` | Pack chỉ có vector jina 1024 chiều. **Đừng dùng `AIC_DENSE_INDEXES`** — biến đó đổi tên nhánh thành `dense_jina_clip_v2`, làm mọi trọng số nhánh và mọi `--disable-branch` đã lưu trỏ sai chỗ |
| `AIC_ENABLE_QUERY_TRANSLATION` | `false` | jina có text tower đa ngữ; dịch VI→EN cho nó là mất thông tin. Cũng nhờ vậy đường truy vấn không cần mạng và không tốn lời gọi FPT nào |
| `AIC_BRANCH_TIMEOUT_MS` | `30000` | 8000 là con số chọn cho 765 scene. Nhánh vượt hạn biến mất **trong im lặng** (`branch_status=timeout`, API vẫn 200) |
| `AIC_ENABLE_OCR_BRANCH`, `AIC_ENABLE_OCR_FUZZY` | `true` | Pack có **OCR 0%** nên tắt nghe hợp lý, nhưng tắt làm mọi truy vấn từ UI trả **422** — xem §9. Hai nhánh rỗng tốn p50 0 ms |

---

## 5. Nhánh nào thật sự sống

Đo trên 120 truy vấn gold, corpus đầy đủ:

| Nhánh | success | p50 | |
|---|---:|---:|---|
| `dense_visual` | 120/120 | 2382 ms | max 4525 ms |
| `bm25_caption` | 120/120 | 787 ms | |
| `bm25_asr` | 96/120 | 241 ms | 24 lượt bị cổng modality bỏ qua |
| `bm25_object` | 88/120 | 17 ms | từ điển VI→EN đang chạy |
| `color_search` | 15/120 | 0 ms | |
| `bm25_ocr`, `ocr_fuzzy` | 0 | — | pack không có OCR |
| `bm25_keyword` | 0 | — | pack để `keywords: []` |
| `bm25_action` | 0 | — | pack để `action_tags: []` |
| `event_search` | 0 | — | 11.079 event nhưng `event_caption: null` |

5/10 nhánh chết. Ba nhánh cuối vá được từ dữ liệu đã có trong export, nhưng
docs/20 đã đo: thêm nhánh **không tự động tốt lên** — `bm25_ocr` hồi sinh còn
làm tệ đi. Đừng bật thêm nếu chưa đo.

---

## 6. Chạy eval

```powershell
python -m scripts.eval_tasks --pipeline container `
    --gold examples/gold_all3.jsonl `
    --metadata storage/exports_competition/scenes.jsonl `
    --json-out outputs/eval.json
```

**`--metadata` là bắt buộc.** Thiếu nó, script vẫn tìm kiếm đúng trên corpus
đầy đủ (container dựng từ env) nhưng nạp repository chấm điểm từ mặc định
`storage/exports/scenes.jsonl` — 3 scene. Repository đó nuôi `_SCENE_BOUNDS`,
và `_scene_in_gold` trả `False` cho mọi scene không có trong đó. Kết quả:
**TRAKE/QA/AVS tụt về gần 0 mà không có lỗi nào**. Đã dính bẫy này một lần.

Dấu hiệu nhận biết: dòng `gold=120 query  scenes=...` phải in **87742**.

Thêm `PYTHONHASHSEED=0` nếu cần tái lập.

---

## 7. Một tiến trình một lúc

Container chiếm ~5 GB, đỉnh 5,1 GB. Máy 15,6 GB chạy được **một** eval hoặc
**một** server, không phải cả hai.

Dừng bằng Ctrl+C ở terminal chưa chắc giết tiến trình python con — đã gặp ca
hai eval "đã dừng" vẫn giữ 8,67 GB, làm lượt chạy sau chậm **110 s/truy vấn**
thay vì 6,6 s. Kiểm và dọn:

```powershell
Get-Process python | Select-Object Id, @{n='GB';e={[math]::Round($_.WorkingSet64/1GB,2)}}
Stop-Process -Id <id> -Force
```

---

## 8. Mở cổng 8000 sẽ ra NHẦM UI

Repo có **ba** bản giao diện, và bản hiện ra mặc định không phải bản thi đấu:

| Đường dẫn | Là gì | |
|---|---|---|
| `online/ui/` | HTML/JS thuần. API **tự mount** tại `/ui`, và `/` chuyển hướng vào đó | ❌ demo cũ |
| `ui/` (gốc repo) | bản sao của trên | ❌ |
| `online/ui-react/` | React + TS, `<title>AIC 2026 Search</title>` | ✅ **bản thi đấu** |

Mở `http://localhost:8000` là ra bản demo cũ — không có bảng trọng số nhánh,
không có tab chỉnh frame, không có đường nộp bài. Nó **vẫn tìm kiếm được**, nên
rất dễ tưởng là đúng.

### Lần đầu mở UI

Điền hai ô trong QueryStudio (lưu `localStorage`, chỉ một lần):

- **API base**: `http://localhost:8000`
- **Token**: `AIC_ONLINE_API_KEY` trong `.env.fpt.local` — `run_ui.ps1` in sẵn.

Thiếu token thì mọi `/v1/*` trả 401 (trừ `/v1/health`).

### Đừng phục vụ `online/ui-react/dist/`

Bản build sẵn cũ hơn `src/` (07/08 so với 13/08), thiếu cả tab "Chỉnh frame"
thêm ngày 10/08. `run_ui.ps1` chạy `npm run dev` từ nguồn và cảnh báo nếu
`dist/` cũ hơn. Muốn dùng bản tĩnh phải `npm run build` lại.

---

## 9. Lỗi 422 "search_options chứa cấu hình backend chưa chạy thật"

```
branches['bm25_ocr']: không có branch/execution nào tên này đang chạy
```

UI gửi kèm trọng số nhánh đã lưu, mà `AIC_BRANCH_WEIGHTS` trong `.env.fpt.local`
khai `bm25_ocr:1.0`. Nếu backend chạy với `AIC_ENABLE_OCR_BRANCH=false` thì nhánh
đó không tồn tại, và `/v1/search/capabilities` **từ chối bằng 422 thay vì lờ đi**
— nên MỌI truy vấn từ UI đều hỏng.

Vì vậy `run_competition.ps1` gán **tường minh** `AIC_ENABLE_OCR_BRANCH=true` và
`AIC_ENABLE_OCR_FUZZY=true`, dù pack có OCR 0%. Hai nhánh rỗng tốn p50 **0 ms**
— tắt chúng không được gì mà làm hỏng mọi cấu hình trọng số đã lưu.

Gặp lỗi này: kiểm `GET /v1/search/capabilities` xem nhánh nào đang chạy, rồi
hoặc bật lại nhánh thiếu, hoặc bỏ nó khỏi `AIC_BRANCH_WEIGHTS` và xoá
`localStorage` của UI. **Đổi topology nhánh là đổi hợp đồng với UI.**

> `$env:` sống suốt phiên PowerShell. Đã lỡ chạy với `false` thì mở terminal
> MỚI, hoặc chạy lại script (nay nó gán "true" tường minh).

---

## 10. `ValueError: không nạp được text encoder 'storage/models/jina-clip-v2'` vì phiên bản `huggingface-hub`

```
huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this
module, but found huggingface-hub==1.28.0.
```

Thông điệp lỗi của container gợi ý "trỏ vào model đã tải sẵn" — **ở ca này lời
khuyên đó sai đường**. Model vẫn nằm đúng chỗ; cái chết là `transformers` 4.x
kiểm phiên bản `huggingface-hub` ngay lúc import và ném `ImportError` khi gặp
hub 1.x. Lỗi đó bị `warmup()` bắt lại rồi bọc thành lỗi cấu hình, nên nhìn
giống thiếu model.

Hay gặp trên **máy GPU thuê** (Vast.ai/RunPod): image gốc đã cài sẵn hub 1.x ở
`/venv/main`, còn `.venv` của dự án dựng kèm system-site-packages nên thừa
hưởng luôn.

Xem cái nào đang được nạp, rồi ghim hub về nhánh 0.x **vào đúng venv đang chạy
uvicorn**:

```bash
cd /workspace/AIC2026_Nam_thang_ay
.venv/bin/python -c "import transformers, huggingface_hub as h; print(transformers.__version__, transformers.__file__); print(h.__version__, h.__file__)"
.venv/bin/pip install "huggingface_hub>=0.34,<1"
```

Cài vào `.venv` là đủ để che bản 1.x của `/venv/main` — không cần đụng vào
image gốc. Ghim này nay nằm sẵn trong extra `gpu` của `pyproject.toml`, nên máy
mới chỉ cần `pip install -e '.[gpu]'`.

**Đừng chữa theo hướng ngược lại** (giữ hub 1.x, nâng `transformers` lên 5.x):
`jina-clip-v2` chạy bằng `trust_remote_code`, code mô hình đó viết cho API
`transformers` 4.x — đổi lên 5 là đổi một lỗi dependency lấy một lỗi runtime
khó đoán hơn, ngay giữa lúc thi.

---

## 11. `OSError: We couldn't connect to 'https://huggingface.co'` dù model đã có sẵn trên đĩa

```
File ".../transformers/dynamic_module_utils.py", line 406, in get_cached_module_file
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

Đây **không phải** cái bẫy text tower ở §3 (cái đó đã vá bằng `config.json`), và
cũng không phải thiếu model. Cái thiếu là **code mô hình**:

```json
"auto_map": {"AutoModel": "jinaai/jina-clip-implementation--modeling_clip.JinaCLIPModel"}
```

Dấu `--` nghĩa là file `.py` nằm ở **repo khác**. Dù `storage/models/jina-clip-v2`
đủ 1,7 GB trọng số, `transformers` vẫn gọi `cached_file()` lên repo đó —
`HF_HUB_OFFLINE=1` biến lời gọi ấy thành `OSError` ngay. `jina-embeddings-v3`
cũng thế, qua `jinaai/xlm-roberta-flash-implementation`.

**Chép `hf_modules/` (gói Drive) hay giải nén `04_hf_modules.zip` (Kaggle) KHÔNG
đủ.** `~/.cache/huggingface/modules/` là bản sao dẫn xuất mà `transformers` sinh
ra *sau* khi `cached_file()` chạy xong; còn `cached_file()` chỉ đọc
`$HF_HOME/hub`. Thiếu `hub/models--jinaai--*-implementation` là chết y hệt.

Chạy một lần trên máy còn mạng (~350 KB, hai repo code):

```bash
python -m scripts.prepare_jina_offline
```

Script vá `config.json` (như §3), kéo về đúng hai repo code, rồi **tự kiểm lại
bằng một tiến trình con bật `HF_HUB_OFFLINE=1`** — xanh nghĩa là startup sẽ qua.
Kiểm suông, không đụng mạng: `--verify-only`.

Máy **hoàn toàn không ra được** huggingface.co thì chép hai thư mục này từ máy
đã chạy được sang `$HF_HOME/hub` (mặc định `~/.cache/huggingface/hub`):

```
models--jinaai--jina-clip-implementation
models--jinaai--xlm-roberta-flash-implementation
```

> `HF_HOME` trên Vast.ai thường bị trỏ sang `/workspace/.hf_home`. Chép vào
> `~/.cache` trong khi uvicorn đọc `$HF_HOME` là hỏng âm thầm — `bootstrap_vast_from_kaggle.py
> --verify-only` in ra đúng đường dẫn nó đang kiểm, đối chiếu ở đó.

---

## 12. `ImportError: This modeling file requires ... einops`

Lỗi tiếp ngay sau §11, và là **dấu hiệu tốt**: cache đã có code mô hình, nên
`transformers` mới đọc tới `import` của nó. `check_imports` chặn ở đây, **mỗi
lần một gói** — `einops`, rồi `timm`, rồi `torchvision`... mỗi lần một lượt khởi
động nữa. Cài trọn một phát:

```bash
pip install einops timm torchvision click
```

Toàn bộ gói mà code jina cần (quét bằng `get_imports` trên hai repo code):

| Repo code | Gói bên thứ ba |
|---|---|
| `jinaai/jina-clip-implementation` | `einops`, `timm`, `torchvision`, `pillow`, `numpy`, `requests`, `torch`, `transformers` |
| `jinaai/xlm-roberta-flash-implementation` | `einops`, `click`, `numpy`, `torch`, `transformers` |

`flash_attn` và `xformers` **không** có trong bảng: code jina bọc chúng trong
`try/except` nên thiếu vẫn chạy (chậm hơn). Đừng cài `flash-attn` chỉ vì thấy
`use_flash_attn: true` trong `config.json` — nó biên dịch hàng chục phút.

`python -m scripts.prepare_jina_offline` nay quét sẵn danh sách này và in đúng
một dòng `pip install` cho những gói còn thiếu, trước khi thử nạp. Ghim cũng đã
nằm trong extra `gpu` của `pyproject.toml`, nên máy mới chỉ cần
`pip install -e '.[gpu]'`.

> `torchvision` phải khớp bản `torch` đang có. Image GPU thuê hầu như luôn cài
> sẵn cả hai — kiểm `python -c "import torch, torchvision"` trước, đừng để pip
> kéo về một cặp torch khác rồi hỏng CUDA.

---

## 13. UI chạy trên máy mình, backend chạy trên Vast.ai

UI mặc định gọi `http://localhost:8000`, còn backend nằm trong container Vast —
hai máy khác nhau, chưa có gì nối chúng lại. `run_ui.ps1` nay dò
`/v1/health` (đường duy nhất không đòi token) và in ra cách nối khi không thấy
backend.

**Cách 1 — SSH tunnel (khuyên dùng).** Không cần map cổng lúc tạo máy, và API
base trong QueryStudio giữ nguyên `http://localhost:8000` nên không phải sửa gì:

```powershell
# Terminal riêng, để chạy suốt phiên. Lệnh ssh lấy ở nút Connect của Vast.
ssh -p <SSH_PORT> root@<HOST> -L 8000:localhost:8000 -N
```

Backend trên box bind `127.0.0.1` hay `0.0.0.0` đều được — tunnel đi vào
`localhost` phía box.

**Cách 2 — cổng công khai.** Chỉ được nếu lúc tạo máy đã mở cổng (ô "Open Ports"
hoặc `-p 8000:8000`). Backend PHẢI bind `0.0.0.0` (`scripts/run_backend.sh` đã
làm sẵn), rồi điền `API base = http://<IP>:<cổng đã map>`.

Kiểm nhanh, theo thứ tự — dừng ở chỗ đầu tiên im lặng:

```bash
# trên box
ss -ltnp | grep 8000
curl -s localhost:8000/v1/health
```

```powershell
# trên máy mình, sau khi mở tunnel
curl.exe -s http://localhost:8000/v1/health
```

CORS **không** phải thứ chặn bạn ở đây: `AIC_CORS_ORIGINS` mặc định đã có
`http://localhost:5173`, và origin tính theo nơi phục vụ TRANG (UI), không phải
nơi API chạy. Đổi cổng dev của Vite thì mới phải thêm origin.

> Ảnh keyframe đi qua `/v1/media` của backend, tức cũng qua tunnel. Box không có
> 28 GB ảnh thì UI vẫn tìm kiếm được, chỉ là thumbnail 404 — xem §2.
