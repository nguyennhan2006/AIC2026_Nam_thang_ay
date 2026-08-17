# Branch `full-runnable` — clone về, thêm `.env`, chạy

Branch này gom toàn bộ code + tài liệu + metadata đã enrich của hệ thống tìm
kiếm video AIC2026. Mục tiêu: người khác clone về, bỏ file `.env` của mình vào,
dựng lại phần dữ liệu nặng bằng script, và chạy được hệ thật — không phải bản
demo giả lập.

Đọc hết mục **"Ba thứ repo KHÔNG có"** trước khi bắt đầu. Bỏ qua nó thì hệ vẫn
khởi động, vẫn trả kết quả, chỉ là nhánh mạnh nhất đã âm thầm chết — đúng kiểu
hỏng mà cả dự án này dành phần lớn công sức để chống.

---

## 1. Repo có sẵn những gì

| Thứ | Ở đâu | Ghi chú |
|---|---|---|
| Toàn bộ code online + offline | `online/`, `offline/`, `datasection/` | |
| 36 tài liệu thiết kế & nhật ký thí nghiệm | `docs/` | bắt đầu từ `docs/08_FILE_GUIDE.md` |
| **Metadata đã enrich của 3 video** | `storage/exports_multivideo/*.jsonl` | 15MB — caption/OCR/ASR do Qwen2.5-VL sinh. **Đắt nhất, và đã làm sẵn.** |
| Bộ gold 120 truy vấn | `examples/gold_all3.jsonl` | 36 KIS / 36 VQA / 24 AVS / 24 TRAKE |
| Notebook Kaggle theo từng stage | `notebooks/` | caption, OCR, ASR, scene, embedding |
| Kết quả đo đã chốt | `outputs/evaluation/` | |

## 2. Ba thứ repo KHÔNG có

Đều **sinh lại được**, và đều là lý do file `.gitignore` từ chối chúng.

| Thiếu | Dung lượng | Hậu quả nếu bỏ qua | Cách dựng lại |
|---|---:|---|---|
| `storage/processed/keyframes/L21_V00{1,2,3}/` | 238MB | UI không có ảnh; không embed lại được | §4.1 |
| `storage/processed/embeddings/` | 428MB | **`dense_visual` chết âm thầm** → rơi về `lexical_hash_fallback` | §4.2 |
| `storage/models/` | 9.2GB | script tự tải từ HuggingFace | tự động |

⚠️ Cái nguy hiểm là dòng thứ hai. Thiếu vector ảnh thì container **không báo
lỗi** — nó đổi tên nhánh thành `lexical_hash_fallback` và chạy tiếp. API vẫn
200, UI vẫn có kết quả, chỉ là tầng ngữ nghĩa đã biến mất. §6 nói cách kiểm.

---

## 3. Cài đặt

Yêu cầu **Python 3.11+**, và `ffmpeg` nếu cần tách keyframe từ video.

```bash
git clone -b full-runnable https://github.com/nguyennhan2006/AIC2026_Nam_thang_ay.git
cd AIC2026_Nam_thang_ay
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[api,test]"
```

Kiểm tra ngay, chưa cần dữ liệu gì:

```bash
python -m pytest tests/ -q      # phải PASS toàn bộ
python -m scripts.smoke_e2e     # chạy trên fixture nhỏ có sẵn trong repo
```

### File `.env` của bạn

Repo **không** kèm khoá nào. `.gitignore` chặn `.env` và `.env.*.local`, và
`scripts/check_secret_leak.py` chặn lần nữa lúc commit.

Copy `.env.example` rồi điền. Điểm khác biệt dễ vấp: file env được nạp
**tường minh** qua biến `AIC_ENV_FILE`, KHÔNG tự dò `.env` ở thư mục hiện tại —
cố ý, để việc khoá thật có được nạp hay không là một lựa chọn nhìn thấy trên
dòng lệnh. Gõ sai đường dẫn là lỗi dừng hẳn, không phải bỏ qua âm thầm.

```bash
cp .env.example .env.fpt.local   # rồi điền AIC_FPT_API_KEY, ...
export AIC_ENV_FILE=.env.fpt.local
```

Tối thiểu để chạy được nhánh dense và rerank:

```ini
AIC_FPT_ENABLED=true
AIC_FPT_API_KEY=<khoá của bạn>
AIC_FPT_BASE_URL=https://mkp-api.fptcloud.com
AIC_FPT_FAST_LLM_MODEL=gemma-4-31B-it
AIC_ENABLE_QUERY_TRANSLATION=true
AIC_FUSION_METHOD=norm_max
AIC_METADATA_JSONL=storage/exports_multivideo/scenes.jsonl
```

`AIC_ENABLE_QUERY_TRANSLATION=true` **không phải tuỳ chọn**. Text tower của
CLIP chỉ biết tiếng Anh; không dịch thì nó dồn mọi truy vấn tiếng Việt về gần
một điểm (đo được: cosine giữa 10 câu khác nghĩa hẳn nhau = 0.912 so với 0.448
của cùng model trên tiếng Anh). Xem `docs/33_RETRIEVAL_TECHNIQUES.md` §2.1.

---

## 4. Dựng lại phần dữ liệu nặng

### 4.1 Ảnh keyframe

Ba video đến từ hai nguồn khác nhau:

- **L21_V001** — có video gốc. Tách bằng pipeline offline
  (`python -m offline run`), hoặc lấy sẵn từ archive `Keyframes_L21` của ban tổ
  chức. Ảnh nằm ở `storage/processed/keyframes/L21_V001/frame_%06d.jpg`.
- **L21_V002 / L21_V003** — distractor, ban tổ chức **chỉ phát ảnh keyframe +
  CSV** (`n, pts_time, fps, frame_idx`), không có video, không có ASR. Chép
  thẳng ảnh vào `storage/processed/keyframes/L21_V00{2,3}/`.

Tên file phải đúng `frame_%06d.jpg` với `frame_idx` khớp
`storage/exports_multivideo/keyframes.jsonl` — đó là khoá nối duy nhất giữa ảnh
và metadata.

Muốn dựng lại export cho video distractor khác:

```bash
python -m scripts.build_distractor_export --video L21_V002 --video L21_V003 \
    --base storage/exports_l21_enriched --out storage/exports_multivideo
```

### 4.2 Vector ảnh cho `dense_visual`

```bash
python -m scripts.embed_export_keyframes \
    --export storage/exports_multivideo \
    --model-path storage/models/clip-vit-large-patch14 \
    --model-id openai/clip-vit-large-patch14
```

Ghi vector vào `storage/processed/embeddings/` **và** cập nhật `embedding_refs`
trong cả `keyframes.jsonl` lẫn `scenes.jsonl`. Cập nhật cả hai là bắt buộc:
repository đọc keyframe **lồng trong scene**, nên sửa một file mà quên file kia
thì vector nằm trên đĩa mà nhánh dense không thấy gì.

Trên CPU mất khoảng 15 phút cho 855 keyframe. Script có checkpoint và bỏ qua
keyframe đã có vector, nên đứt giữa chừng thì chạy lại là tiếp.

**Thử encoder đa ngữ** (`jina-clip-v2`, không cần bước dịch) — thêm bộ vector
thứ hai song song, không đè lên CLIP:

```bash
pip install timm einops
python -m scripts.embed_export_keyframes --kind jina \
    --model-path jinaai/jina-clip-v2 --model-id jinaai/jina-clip-v2 \
    --embedding-name jina_clip_v2
```

Đọc `docs/20_EXPERIMENT_LOG.md` § VISUAL-01 trước khi dùng kết quả này để quyết
định gì: phép so vẫn còn dở, và jina-clip-v2 là CC-BY-NC-4.0 (phi thương mại).

### 4.3 Index dense trên caption (tuỳ chọn, mặc định TẮT)

```bash
python -m scripts.build_caption_dense_index \
    --metadata storage/exports_multivideo/scenes.jsonl \
    --out storage/indexes_multivideo/caption_dense
```

Nhánh này đã đo và **DROP** trên bộ gold hiện tại (`docs/20` § DENSE-TEXT-02).
Để `AIC_CAPTION_DENSE_INDEX` rỗng là tắt; bật thì phải khai
`AIC_CAPTION_DENSE_ENCODER` khớp `encoder_kind` trong manifest của index.

---

### 4.4 Corpus thi đấu đầy đủ — 873 video

Ba mục trên nói về corpus 3 video. Muốn chạy trên **toàn bộ 873 video** thì
nguồn dữ liệu là `AIC2026_competition_clean_v3.zip`, và nó dùng một lược đồ
KHÁC repo (id khác, file khác) nên phải chuyển đổi:

```bash
python -m scripts.import_competition_pack \
    --pack /duong/dan/AIC2026_competition_clean_v3.zip \
    --out storage/exports_competition \
    --merge-embeddings-from storage/exports_multivideo
```

Ra 873 video / 87.742 scene / 176.707 keyframe, mất ~6 phút.

⚠️ Có **hai** bản zip cùng tên. Bản build 07:25 UTC chỉ có 43/7.790 vector cho
batch L21 — mà toàn bộ bộ gold nằm trên L21_V001..V003, nên nhập bản đó là mất
tầng dense ngay chỗ duy nhất đo được, và không có gì báo lỗi. Kiểm trước:

```bash
python -c "import zipfile,json;print(json.loads(zipfile.ZipFile('<zip>').read('index_version.json'))['dense_vector_count'])"
```

**176707** là bản đúng; 168960 thì tải lại. Chi tiết `docs/34` §3.

**`docs/34_COMPETITION_PACK_IMPORT.md` là tài liệu đầy đủ**: pack thiếu gì
(OCR 0%, không một file ảnh nào), script quyết định những gì và giá phải trả,
số đo nạp/RAM/độ trễ, và lệnh chạy server trên corpus này.

Ảnh keyframe lấy riêng từ mirror ban tổ chức — `docs/35_KAGGLE_MEDIA.md`:

```bash
python -m scripts.fetch_aic_data --what keyframes --pack <pack.zip> --csv <link.csv>
```

28,69 GB, tải + giải nén + đổi tên trong một lệnh. **Đừng chép thẳng thư mục
ảnh**: nguồn đặt tên theo số thứ tự keyframe (`002.jpg`) còn export trỏ theo
frame index (`frame_000090.jpg`) — chép thẳng là 176.707 ảnh nằm sai chỗ và
không có gì báo lỗi.

Một cảnh báo không được bỏ qua: ở 87.742 scene, `dense_visual` mất 5,2–11,8 s
mỗi truy vấn, nên **`AIC_BRANCH_TIMEOUT_MS=8000` mặc định sẽ cắt nhánh đó trong
im lặng**. Đặt `30000` cho corpus đầy đủ.

---

## 5. Chạy

```bash
export AIC_ENV_FILE=.env.fpt.local
uvicorn online.api.app:app --host 0.0.0.0 --port 8000
```

Terminal thứ hai cho UI:

```bash
./scripts/run_local_ui.sh        # mở http://localhost:5173
```

Trong UI giữ Backend là `http://localhost:8000`, bấm **Kiểm tra server**, rồi
tìm thử. Bộ gold `examples/gold_all3.jsonl` có sẵn 120 truy vấn để thử.

---

## 6. Xác nhận hệ thật sự chạy đúng, không phải chạy nửa vời

Đây là phần quan trọng nhất của README này. Ba phép kiểm, theo thứ tự:

> **Mọi endpoint nằm dưới `/v1`** (`online/api/routes.py`:
> `APIRouter(prefix="/v1")`). Và khi `AIC_ONLINE_API_KEY` có giá trị — mặc định
> trong `.env.fpt.local` là có — thì mọi đường `/v1/*` **trừ `/v1/health`** đòi
> header `Authorization: Bearer <khoá>`, thiếu thì 401. Bản trước của README này
> ghi thiếu `/v1`, nên các lệnh curl trong đó trả 404.

**1. Dataset có đúng cái mình tưởng không**

```bash
curl -s localhost:8000/v1/health | python -m json.tool
```

`scene_count` phải là **765**, `video_count` **3**, `keyframe_count` **855**.
Lệch nghĩa là `AIC_METADATA_JSONL` đang trỏ export khác.

**2. Nhánh dense còn sống không — phép kiểm quan trọng nhất**

```bash
curl -s -H "Authorization: Bearer $AIC_ONLINE_API_KEY" \
    localhost:8000/v1/search/capabilities | python -m json.tool
```

Phải thấy `dense_visual` với `backend_kind: "vector"`.

Nếu thấy **`lexical_hash_fallback`** thì §4.2 chưa xong: export không có vector
ảnh nào, và hệ đang chạy bằng hash text giả làm dense. Nó **không** báo lỗi —
container cố ý đổi tên nhánh để `/capabilities` không quảng cáo nhầm, nhưng nếu
không nhìn vào đây thì không cách nào biết.

**3. Nhánh nào thật sự đóng góp cho một truy vấn**

Response của `/search` có `branch_status` cho từng nhánh. `success` với
`candidate_count > 0` mới là chạy; `timeout` nghĩa là nhánh đó đã biến mất khỏi
kết quả trong im lặng.

Đo lại toàn bộ chỉ số:

```bash
python -m scripts.eval_tasks --pipeline container \
    --gold examples/gold_all3.jsonl \
    --metadata storage/exports_multivideo/scenes.jsonl
```

⚠️ Dùng `--pipeline container`. Không có cờ đó, harness dựng một pipeline
**thứ hai** thiếu vài nhánh và **không bọc `TranslatingTextEncoder`** — số in ra
trông hợp lệ nhưng không phải số của server. Xem `docs/20` § "Lỗ hổng hạ tầng đo".

---

## 7. Đọc tiếp

| Tài liệu | Nội dung |
|---|---|
| `docs/08_FILE_GUIDE.md` | file nào làm gì — đọc trước tiên |
| `docs/34_COMPETITION_PACK_IMPORT.md` | nạp pack thi đấu 873 video, và nó còn thiếu gì |
| `docs/35_KAGGLE_MEDIA.md` | tải keyframe/video từ Kaggle — và vì sao KHÔNG được chép thẳng |
| `docs/04_ONLINE_RETRIEVAL.md` | kiến trúc tầng tìm kiếm |
| `docs/33_RETRIEVAL_TECHNIQUES.md` | mỗi nhánh dùng kỹ thuật gì, và điểm mù của chúng |
| `docs/20_EXPERIMENT_LOG.md` | mọi thí nghiệm đã chạy, gồm cả những cái **DROP** |
| `docs/27_SYSTEM_ISSUES.md` | bảng cấu hình đã đo — cái nào bật, cái nào tắt, vì sao |
| `docs/12_USER_GUIDE.md` | biến môi trường đầy đủ |
| `docs/KAGGLE_OFFLINE_GUIDE.md` | chạy stage nặng trên Kaggle T4x2 |

`docs/20_EXPERIMENT_LOG.md` ghi cả kết quả âm. Trước khi thử một ý tưởng, tra
xem nó đã bị đo và loại chưa — vài ý tưởng nghe rất hợp lý trong đó đã được đo
là **làm hệ kém đi**.
