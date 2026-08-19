# 38 — Chạy hệ thống trên máy thuê Vast.ai

Từ một máy trống tới UI tìm kiếm được. Bản Windows/local là `docs/36`; file này
chỉ nói phần **khác biệt của máy thuê**, và mọi mục dưới đây đều là một lần đã
hỏng thật chứ không phải phòng xa.

Thứ tự bắt buộc: **code → venv → dữ liệu → cache code mô hình → chạy → nối UI**.
Đảo bước là mất một lượt khởi động 4 phút để biết mình sai.

---

## 0. Đường ngắn nhất

```bash
# --- trên máy Vast (SSH vào rồi) ---
apt-get update && apt-get install -y libarchive-tools git
echo 'export HF_HOME=/workspace/.hf_home' >> ~/.bashrc && export HF_HOME=/workspace/.hf_home

cd /workspace
git clone <repo-url> AIC2026_Nam_thang_ay && cd AIC2026_Nam_thang_ay
git checkout full-runnable

python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[api]' "transformers>=4.49,<5" "huggingface_hub<1" einops timm click numpy requests tqdm

export KAGGLE_USERNAME=... KAGGLE_KEY=...
.venv/bin/python -m scripts.bootstrap_vast_from_kaggle --plan          # xem trước
.venv/bin/python -m scripts.bootstrap_vast_from_kaggle --skip-keyframes
.venv/bin/python -m scripts.prepare_jina_offline
./scripts/run_competition.sh
```

```powershell
# --- trên máy mình, hai terminal ---
ssh -p <SSH_PORT> root@<HOST> -L 8000:localhost:8000 -N   # terminal 1: tunnel
.\scripts\run_ui.ps1                                      # terminal 2: UI
```

---

## 1. Tạo máy

| Mục | Chọn | Vì sao |
|---|---|---|
| GPU | bất kỳ ≥ 8 GB VRAM (đã chạy RTX 5060) | backend chỉ chạy text tower jina; VRAM không phải nút thắt |
| RAM | ≥ 16 GB | đỉnh RSS lúc dựng container **5,1 GB**, ổn định 3,9 GB |
| Disk | **≥ 40 GB** không ảnh, **≥ 80 GB** có ảnh keyframe | xem bảng dung lượng ở §5 |
| Image | PyTorch chính chủ của Vast | có sẵn torch khớp CUDA của GPU — thứ khó cài lại nhất |
| Ports | mở `8000` **chỉ khi** không dùng SSH tunnel | §8 |

**Khoá SSH**: dán `~/.ssh/id_ed25519.pub` vào Vast → *Account* → *SSH Keys*
**trước khi** tạo máy. Thêm sau thì máy đang chạy không nhận khoá mới cho tới
lần khởi động lại.

GPU mới (Blackwell, sm_120 — 5060/5070/5090) cần `torch` build **cu128** trở
lên. Kiểm ngay sau khi vào máy, đừng đợi tới lúc warmup:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

Báo `no kernel image is available for execution on the device` nghĩa là image có
torch quá cũ → đổi image, **đừng** `pip install -U torch` đè lên (§4).

---

## 2. Nối vào máy

Lệnh SSH lấy ở nút *Connect* của instance. Vast hay đi qua máy trung gian
(`ssh5.vast.ai`), nên cổng là cổng lạ chứ không phải 22:

```powershell
ssh -p <SSH_PORT> root@<HOST>
```

`/workspace` là volume bền vững, mọi thứ khác trong container **mất khi máy bị
huỷ**. Luôn để repo và dữ liệu dưới `/workspace`.

---

## 3. Lấy code

```bash
cd /workspace
git clone <repo-url> AIC2026_Nam_thang_ay
cd AIC2026_Nam_thang_ay && git checkout full-runnable
```

Branch `full-runnable` là branch có `scripts/run_competition.sh` và
`scripts/prepare_jina_offline.py`. Thiếu hai file này thì mọi bước sau vô nghĩa.

---

## 4. venv — chỗ dễ tự bắn vào chân nhất

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[api]' "transformers>=4.49,<5" "huggingface_hub<1" einops timm click numpy requests tqdm
```

**`--system-site-packages` là cố ý**: để thừa hưởng `torch`/`torchvision` đã
khớp CUDA của image. Cài lại torch bằng pip trên máy thuê rất dễ ra bản CPU hoặc
lệch CUDA, và đó là hàng GB tải về để hỏng.

**Ba ràng buộc phiên bản, vi phạm cái nào cũng chết lúc dựng container:**

| Ghim | Vì sao |
|---|---|
| `huggingface_hub < 1` | image thường cài sẵn hub 1.x, mà `transformers` 4.x kiểm phiên bản **lúc import** và ném `ImportError`. Cài vào `.venv` là đủ để che bản 1.x của `/venv/main` — xem `docs/36` §10 |
| `transformers >= 4.49, < 5` | code `trust_remote_code` của jina viết cho API 4.x. Nâng lên 5 để "chữa" lỗi hub là đổi một lỗi dependency lấy một lỗi runtime khó đoán hơn |
| `einops`, `timm`, `torchvision`, `click` | không phải phụ thuộc của dự án mà là **import của code mô hình** jina tải về. Thiếu thì `check_imports` chặn **mỗi lần một gói**, mỗi lần tốn một lượt khởi động — `docs/36` §12 |

Máy **không** có sẵn torch thì dùng `pip install -e '.[gpu]'` — extra đó đã gói
đủ cả ba nhóm trên. Có sẵn torch thì đừng, vì nó sẽ đụng vào torch.

Đặt `HF_HOME` **một lần, trước mọi bước sau**, và để nó nằm trên `/workspace`:

```bash
echo 'export HF_HOME=/workspace/.hf_home' >> ~/.bashrc
export HF_HOME=/workspace/.hf_home
```

Đặt lệch giữa lúc tải và lúc chạy là kiểu hỏng âm thầm khó chịu nhất: file có
mặt đầy đủ, chỉ là `transformers` nhìn sang chỗ khác.

---

## 5. Tải dữ liệu — `bootstrap_vast_from_kaggle.py`

```bash
apt-get update && apt-get install -y libarchive-tools   # bsdtar, BẮT BUỘC
export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>          # hoặc ~/.kaggle/kaggle.json
```

Script **stream thẳng từ Kaggle vào `bsdtar`**, không lưu zip trung gian — nên
disk chỉ cần đủ chứa dữ liệu đã giải nén, không cần gấp đôi.

```bash
.venv/bin/python -m scripts.bootstrap_vast_from_kaggle --plan            # in bảng file -> đích rồi dừng
.venv/bin/python -m scripts.bootstrap_vast_from_kaggle --skip-keyframes  # 6,8 GB, đủ chạy backend + eval
.venv/bin/python -m scripts.bootstrap_vast_from_kaggle                   # 35,4 GB, có ảnh
.venv/bin/python -m scripts.bootstrap_vast_from_kaggle --verify-only     # kiểm layout, không tải
```

Hai dataset (`trongnhantran25/aic-nam-thang-ay`,
`nguyenchonnhan/data-for-namthangay-competition`), sáu archive, đích tính theo
**tên file** chứ không theo cây thư mục Kaggle:

| Archive | Dung lượng | Về đâu |
|---|---:|---|
| `01_export.zip` | 1,10 GB | `storage/exports_competition/` |
| `02_vectors.zip` | 0,34 GB | `storage/processed/embeddings_pack/` (873 `.npy`) |
| `03_models.zip` | 5,39 GB | `storage/models/` (jina-clip-v2 + e-v3) |
| `04_hf_modules.zip` | 1,2 MB | **`$HF_HOME/modules/`** — ngoài repo |
| `05_config.zip` | ~60 KB | `.env.fpt.local` (đã bỏ khoá) + docs |
| `06_keyframes.zip` | 28,62 GB | `storage/processed/keyframes/` |

Đứt mạng giữa chừng thì chạy lại — script tải tiếp từng archive, không làm lại
từ đầu. File lạ không khớp luật nào sẽ **dừng script** kèm tên file, thay vì
đoán bừa một chỗ đổ.

Xong thì `--verify-only` phải xanh **mọi dòng**, gồm cả hai dòng `hf code ...`
(§6). Nó kiểm đúng những đường dẫn container sẽ mở, không kiểm thư mục suông.

> `.env.fpt.local` trong `05_config.zip` đã **bị bỏ mọi khoá** trước khi lên
> Kaggle. Cần khoá FPT hay `AIC_ONLINE_API_KEY` thì tự điền lại trên máy —
> đừng upload bản có khoá lên Kaggle, kể cả dataset riêng tư.

---

## 6. Cache code mô hình — bước hay bị bỏ quên

```bash
.venv/bin/python -m scripts.prepare_jina_offline
```

Một lệnh làm ba việc: vá `config.json` (text tower → bản local), kéo **code mô
hình** của jina về `$HF_HOME/hub`, và kiểm lại bằng một tiến trình con bật
`HF_HUB_OFFLINE=1`.

Vì sao cần dù model đã nằm đủ trên đĩa: `config.json` khai
`auto_map: "jinaai/jina-clip-implementation--modeling_clip.JinaCLIPModel"` —
dấu `--` nghĩa là file `.py` nằm ở **repo khác**, và `transformers` luôn hỏi
HuggingFace về nó. Chi tiết và cách chữa khi máy không ra được mạng:
`docs/36` §11.

**`04_hf_modules.zip` không thay được bước này.** Thư mục `modules/` là bản sao
dẫn xuất, sinh ra *sau* khi lượt tải code chạy xong; lượt tải đó chỉ đọc
`$HF_HOME/hub`.

Script cũng quét luôn phụ thuộc của code mô hình và in đúng một dòng
`pip install` cho những gói còn thiếu — thay vì để `check_imports` chặn từng gói
một qua từng lượt khởi động.

---

## 7. Chạy backend

```bash
./scripts/run_competition.sh                 # bind 0.0.0.0:8000
HOST=127.0.0.1 ./scripts/run_competition.sh  # chỉ localhost, dùng với SSH tunnel
```

Script gom sẵn 9 biến môi trường đã kiểm chứng (ý nghĩa từng biến: `docs/36`
§4), chặn trước khi tốn 4 phút nếu thiếu file, và chạy
`prepare_jina_offline --verify-only` như một bước preflight không cần mạng.

**Cảnh báo bảo mật script sẽ in ra**: `.env.fpt.local` lấy từ Kaggle có
`AIC_ONLINE_API_KEY` **rỗng**, mà khoá rỗng nghĩa là `api_key_guard` tắt hẳn
(`online/api/app.py`). Bind `0.0.0.0` với khoá rỗng là mở toàn bộ API ra
Internet cho bất kỳ ai quét trúng cổng. Hoặc đặt khoá, hoặc chạy
`HOST=127.0.0.1` + SSH tunnel.

Đợi `Application startup complete.` (~4 phút). Chạy nền thì mất SSH không giết
server:

```bash
nohup ./scripts/run_competition.sh > /workspace/backend.log 2>&1 &
tail -f /workspace/backend.log
```

---

## 8. Nối UI trên máy mình

Cách nối, cách kiểm, và vì sao CORS **không** phải thứ chặn bạn: `docs/36` §13.
Tóm tắt: mở tunnel `ssh -p <SSH_PORT> root@<HOST> -L 8000:localhost:8000 -N`
rồi `.\scripts\run_ui.ps1` — API base giữ nguyên `http://localhost:8000`.
`run_ui.ps1` tự dò `/v1/health` và in hướng dẫn nếu không thấy backend.

Token trong QueryStudio phải là `AIC_ONLINE_API_KEY` **của box**. Lệch nhau thì
`/v1/health` xanh mà mọi truy vấn trả 401. Khoá đang **rỗng** (mặc định của
`.env.fpt.local` hiện tại) thì để trống ô Token — `api_key_guard` tắt hẳn, và
đó chính là lý do §7 bắt chọn giữa "đặt khoá" và "chỉ mở qua tunnel".

---

## 9. Kiểm nhanh

```bash
curl -s localhost:8000/v1/health | python3 -m json.tool
curl -s -H "Authorization: Bearer $KEY" localhost:8000/v1/search/capabilities | python3 -m json.tool
```

Con số phải khớp, lệch là dữ liệu nạp thiếu chứ không phải "chạy được rồi":

| Chỉ số | Kỳ vọng |
|---|---:|
| `video_count` | 873 |
| `scene_count` | 87.742 |
| `keyframe_count` | 176.707 |
| nhánh dense | `dense_visual` với `backend_kind: "vector"` |

Thấy `lexical_hash_fallback` thay vì `dense_visual` là **không đọc được vector**
— hệ vẫn trả 200, chỉ là tầng ngữ nghĩa đã tắt (`docs/36` §1).

---

## 10. Lỗi thường gặp

| Triệu chứng | Đọc |
|---|---|
| `huggingface-hub>=0.34.0,<1.0 is required ... found 1.x` | `docs/36` §10 |
| `We couldn't connect to 'https://huggingface.co'` dù model có sẵn | `docs/36` §11 |
| `This modeling file requires ... einops` | `docs/36` §12 |
| UI không gọi được backend | `docs/36` §13 |
| 422 `search_options chứa cấu hình backend chưa chạy thật` | `docs/36` §9 |
| `no kernel image is available for execution on the device` | §1 ở trên — image có torch quá cũ cho GPU Blackwell |
| `Không tìm thấy bsdtar` | `apt-get install -y libarchive-tools` |
| Kaggle trả 401/403 | thiếu `KAGGLE_USERNAME`/`KAGGLE_KEY` |

---

## 11. Tắt máy

Vast tính tiền **theo giờ máy chạy**, cộng phí lưu trữ khi máy *stopped*. Huỷ
instance là **mất sạch `/workspace`** — 35 GB tải lại từ đầu.

Trước khi huỷ, kéo về những thứ sinh ra trên máy mà Kaggle không có:

```powershell
scp -P <SSH_PORT> root@<HOST>:/workspace/AIC2026_Nam_thang_ay/outputs/*.json .\outputs\
```

Dữ liệu gốc thì không cần cứu: nó nằm trên Kaggle, và `bootstrap` dựng lại được
trong một lệnh.
