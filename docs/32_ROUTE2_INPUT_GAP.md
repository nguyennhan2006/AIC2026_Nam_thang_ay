# 32 — Route2 (VastAI) ↔ contract online: còn thiếu gì

Lập 2026-08-10. Đối chiếu **spec Route2 `aic-multikeyframe-v2.0`** với
[contract đã chốt](29_DATA_CONTRACT.md) và với **những trường online thực sự đọc**.

Đọc §1 trước mọi thứ khác — nó quyết định phần lớn Route2 sinh ra có vào được hệ
thống hay không.

---

## 1. Sự thật khó chịu: online chỉ đọc `.text` và `.label`

Route2 sinh entity/attribute/action/relation có cấu trúc, certainty, song ngữ.
Nhưng đường nạp online hiện tại
([json_metadata.py](../online/adapters/json_metadata.py)) **chỉ lấy**:

```
captions[].text          objects[].label          asr_segments[].text
ocr_instances[].text     keywords[].normalized_text
```

Mọi thứ còn lại — bbox, confidence, certainty, language, relation, attribute,
evidence — **bị bỏ hoàn toàn** ở tầng xếp hạng. Xem [docs/29 §4.1](29_DATA_CONTRACT.md).

Nên có đúng hai lựa chọn, và phải chọn TRƯỚC khi chạy VastAI:

| | Cách | Chi phí | Được gì |
|---|---|---|---|
| **A** | Làm phẳng Route2 vào `index_text` rồi nạp như một caption/keyword | Gần bằng 0 — Route2 **đã sinh sẵn** `index_text` | BM25 đọc được toàn bộ entity/action/relation ngay |
| **B** | Mở rộng online đọc cấu trúc (nhánh entity, nhánh relation, lọc theo attribute) | Lớn — nhánh mới, contract mới, phải đo lại | Truy vấn kiểu *"người mặc áo đỏ cầm chai"* mới thật sự khớp cấu trúc |

**Khuyến nghị: làm A ngay, B sau và chỉ khi đo được.** Lý do: `index_text` đã có
sẵn, không tốn gì; còn thêm nhánh mới thì [Phase D](31_TRAKE_EXPERIMENT_PLAN.md)
vừa cho thấy thêm nhánh không tự động tốt lên — `bm25_ocr` sống lại còn **làm
tệ đi**, và `color_search` sống mà đóng góp bằng 0.

---

## 2. BẮT BUỘC bổ sung vào lần chạy VastAI

Ba thứ Route2 không sinh mà thiếu nó thì một tính năng online **chết hẳn**.

### 2.1 🔴 Dense embedding — thiếu là mất nhánh mạnh nhất

`dense_visual` là nhánh mạnh nhất của hệ (xem [docs/30 §3.1](30_SYSTEM_DIAGNOSIS.md):
nó cứu được cả một caption bịa). Không có vector thì container **tự động rơi về**
`lexical_hash_fallback` — vẫn chạy, vẫn trả kết quả, và **không có cảnh báo nào**.

Cần sinh trong lần chạy VastAI:

```
{data_root}/processed/embeddings/{video_id}/frame_{frame_idx:06d}.json   (hoặc .npy)
```

và pack `embedding` với **cả hai**:

```json
{"video_id": "L21_V001", "frame_idx": 9987,
 "vector": [0.013, -0.072, ...],
 "embedding_refs": [{
   "embedding_name": "clip_vit_l14_v1", "dimension": 768, "modality": "image",
   "model_name": "openai/clip-vit-large-patch14", "normalized": true,
   "storage_locations": [{"backend": "file",
     "vector_uri": "processed/embeddings/L21_V001/frame_009987.json"}]}]}
```

`vector` inline cho clip pooling, `embedding_refs` cho online dựng vector store —
hai đường đọc khác nhau, thiếu một là hỏng một.

⚠️ **Model phải TRÙNG với text tower online.** Danh tính model là
`openai/clip-vit-large-patch14`; máy local nạp từ `storage/models/clip-vit-large-patch14`
(`AIC_VISUAL_EMBEDDING_MODEL` là đường dẫn thư mục, không phải tên repo HF — máy
này bị chặn SSL khi tải từ HuggingFace). Trên VastAI dùng tên repo cũng được,
miễn **cùng model**: khác model thì cosine vô nghĩa mà không ai báo lỗi.

Đã có `scripts/embed_keyframes_local.py` chạy CPU; trên VastAI có GPU nên nhanh hơn nhiều.

### 2.2 🔴 `dominant_colors[].name` — HSV thô không đủ

`color_search` khớp **chuỗi tên màu** với từ vựng cố định:

```
red · orange · yellow · green · cyan · blue · purple · pink · black · white · gray
```

Route2 cho `hue_hist`/`sat_hist`/`val_hist`/`dominant_hue_deg` — **không có tên**.
Adapter hiện chuyển histogram sang nhưng để `dominant_colors` rỗng, nên
`color_search` sẽ **rỗng trên toàn corpus Route2**.

Cần: chạy `_name_pixels` (bucket hue, đã có ở
[gpu_engine.py:156](../offline/gpu_engine.py#L156)) rồi ghi `dominant_colors`:

```json
"dominant_colors": [{"name": "green", "ratio": 0.42}, {"name": "gray", "ratio": 0.31}]
```

⚠️ **Kèm `ratio` và lọc ngưỡng.** Đo trên corpus hiện tại: giữ top-8 bất kể tỉ lệ
thì "đỏ" xuất hiện ở **99.6%** scene, xám 99.1% — không phân biệt được gì. Ngưỡng
0.15–0.20 đưa về 1.7–2.2 màu/scene.

> Đo được: ngay cả khi lọc ngưỡng, `color_search` **vẫn đóng góp 0** vì fusion.
> Nên đây là điều kiện CẦN, chưa đủ. Xem [docs/30 §3.4](30_SYSTEM_DIAGNOSIS.md).

### 2.3 🟡 `quality.*` — safe_frame đang chạy mù

`safe_frame` (chọn frame đại diện cho KIS) đọc `sharpness`, `brightness`,
`black_frame_ratio`. Route2 không sinh. Rẻ, CPU thuần, đã có ở
[backfill_color_quality.py](../scripts/backfill_color_quality.py) — chạy được ngay
trong lần VastAI:

```json
"quality": {"sharpness": 488.0, "brightness": 0.51, "contrast": 0.33,
            "black_frame_ratio": 0.0, "duplicate_score": null}
```

⚠️ **Thang sharpness của online không khớp corpus này**: đo thực tế p50 = 488
nhưng `safe_frame` chuẩn hoá trong `[40, 300]` → **87% keyframe vượt trần**, bị kẹp
về 1.0 và mất khả năng phân biệt. Cần hiệu chuẩn lại `SHARPNESS_CEILING` sau khi
có phân bố của corpus thật.

---

## 3. Route2 CÓ mà ta đang bỏ phí — lấy được ngay, gần như miễn phí

### 3.1 ✅ `frame_selection` → `selection_score` + `roles`

`selection_score` hiện **0/855**, và tôi cố ý để trống vì bịa ra một điểm "frame
này đại diện tốt đến đâu" là nhét tín hiệu chưa đo vào scoring.

**Route2 sinh nó có căn cứ**: `evidence_strength` + `is_scene_best`. Đây là nguồn
hợp lệ. Map:

```
frame_selection.is_scene_best   -> roles: ["representative"] hoặc ["support"]
frame_selection.evidence_strength -> selection_score   (PHẢI chuẩn hoá về [0,1],
                                     schema ràng buộc ge=0 le=1)
```

`safe_frame` đọc thẳng `selection_score`, nên đây là tính năng bật được mà không
viết thêm dòng code online nào.

### 3.2 ✅ `actions` → `action_tags` thật thay vì suy từ caption

`action_tags` hiện **57–58%**, và nó được `extract_action_tags()` suy ra từ **văn
bản caption** chứ không phải từ model. Route2 cho action label thật kèm
`motion_verified` và `certainty`.

Map `actions[].label_vi` → `action_tags[]`. Nhánh `bm25_action` sẽ đi từ 57% lên
gần 100% và nội dung có nghĩa hơn hẳn.

### 3.3 ✅ `keywords_vi` → `keywords` mức scene

`assemble` hiện tự sinh `keywords` từ `objects[].label` (`_scene_keywords`).
Route2 cho keywords trực tiếp, chất lượng cao hơn. Nhánh `bm25_keyword` hưởng lợi.

### 3.4 ✅ `index_text` → đường ngắn nhất để không phí entity/relation

Đây là **lựa chọn A của §1**. Route2 đã ghép caption + keywords + entity names +
descriptions + actions + attributes + relations + OCR thành một chuỗi. Nạp nó vào
như một caption:

```json
{"caption_type": "tags", "language": "vi", "text": "<index_text>"}
```

`bm25_caption` đọc được ngay. Không cần nhánh mới, không cần đổi contract.

⚠️ Cân nhắc: `index_text` rất dài, có thể loãng BM25 (cùng bệnh với chữ lớp phủ
OCR — tín hiệu có mặt khắp nơi thì không phân biệt được). **Phải đo** trước/sau,
và nên là một `caption_type` riêng để tắt/bật được mà không đụng caption thường.

---

## 4. Route2 có mà contract KHÔNG có chỗ chứa

Những thứ này hiện sẽ bị `offline/assemble.py` **vứt im lặng**. Chọn một trong ba:
mở rộng contract, nhét vào `extensions` (online không đọc), hoặc bỏ.

| Route2 | Đề nghị |
|---|---|
| `entities[].attributes` (color/clothing/material/pose…) | **Đáng đầu tư nhất.** Đây chính là thứ truy vấn *"người đàn ông áo đỏ"* cần, mà `color_search` mức-frame không bao giờ giải được — nó không phân biệt "áo đỏ" với "nền đỏ" ([color_search.py](../online/adapters/color_search.py) docstring nói đúng điều này). Cần trường mới + nhánh mới |
| `relations[]` (holding/wearing/riding…) | Cần nhánh mới. Giá trị cao cho truy vấn kiểu *"người cầm chai cạnh dòng nước"* — đúng một bước TRAKE trong gold |
| `actions[].certainty`, `motion_verified` | Có thể lọc ngưỡng lúc adapter, chỉ giữ action đủ chắc → không cần contract mới |
| `ocr_regions[].difficulty_flags`, `model_retry_recommended` | **Dùng ở tầng OFFLINE**, không cần vào contract: chạy lại OCR cho đúng vùng được đánh dấu. Rẻ và trúng đích |
| `uncertainties`, `parse_ok`, `schema_error` | Không vào contract. Dùng để **lọc record hỏng trước khi assemble** — hiện `assemble` chỉ quarantine record sai cấu trúc, không biết record nào model tự nhận là không chắc |
| `scene.environment` / `setting` / `media_type` | Chưa có bộ lọc nào dùng. Để `extensions` chờ khi có nhu cầu thật |
| caption EN (`short_caption_en`…) | Có thể dùng cho **text tower CLIP** thay vì dịch VI→EN mỗi request. Hiện `AIC_ENABLE_QUERY_TRANSLATION=true` gọi LLM cho mỗi truy vấn — nếu index cả bản EN thì bớt một phụ thuộc mạng ở đường request |
| `visual_evidence`, `best_keyframe_reason` | Hiển thị/audit, không vào scoring |

---

## 5. Adapter hiện tại đang bám TÊN TRƯỜNG CŨ

[scripts/route2_to_stage_packs.py](../scripts/route2_to_stage_packs.py) viết ngày
06/08, trước spec `v2.0`. Đối chiếu:

| Adapter đang đọc | Spec v2.0 | |
|---|---|:-:|
| `entity.label_vi` | `entities[].name_vi` | ❌ đổi |
| `entity.bbox` | `entities[].bbox_2d` | ❌ đổi |
| `region.text` | `ocr_regions[].text_raw` | ❌ đổi |
| `region.bbox` | `ocr_regions[].bbox_2d` | ❌ đổi |
| `short_caption_vi` / `detailed_caption_vi` | giống | ✅ |
| `keywords_vi` | giống | ✅ |
| `hsv_features` | giống | ✅ |
| — | `frame_selection` | ➕ thêm (§3.1) |
| — | `actions[]` | ➕ thêm (§3.2) |
| — | `index_text` | ➕ thêm (§3.4) |
| — | `dominant_colors` có tên | ➕ thêm (§2.2) |
| — | `quality` | ➕ thêm (§2.3) |

**Chưa sửa** — sửa mù theo spec chữ mà không có một file Route2 thật để chạy thử
thì rủi ro cao hơn lợi. Cần một mẫu `scene_index_ready.json` của bản v2.0 để
đối chiếu; đưa tôi một file là tôi cập nhật adapter và chạy `verify_stage_pack` ngay.

---

## 6. Bất biến của contract — sai là hỏng âm thầm

Bốn chỗ đã cắn thật, ghi lại để lần chạy VastAI không lặp lại:

1. **`end_frame` của scene là INCLUSIVE.** `assemble` tự cộng 1 thành
   `end_frame_exclusive`. Đưa sẵn exclusive vào là lệch một frame ở mọi scene.

2. **Stage mức keyframe KHÔNG được tự đặt `keyframe_id`.** Id canonical nhúng
   `scene_idx` mà notebook không biết trước. `assemble` là nơi duy nhất dựng nó.

3. **`timestamp_sec` của pack bị BỎ QUA** — `assemble` luôn tính lại `frame_idx / fps`.
   Đã có ca thật: `pts_time` từ ffmpeg lệch `frame_idx/fps` một frame và làm
   validator scene false-fail.

4. **bbox phải chuẩn hoá [0,1].** Route2 trả pixel (`bbox_2d`), adapter chia theo
   `width`/`height`. Đừng kẹp bằng `min()` — bbox pixel bị kẹp về 1.0 là hỏng hết;
   phải phát hiện đơn vị rồi mới chia.

Và một bất biến của tầng online: **cả 5 file export phải nằm CÙNG một thư mục** —
`AIC_METADATA_JSONL` trỏ `scenes.jsonl`, bốn file kia tìm bằng `with_name()`.

---

## 7. Việc nên chuẩn bị TRƯỚC khi chạy VastAI, xếp theo giá trị

| # | Việc | Thuộc | Vì sao |
|---|---|---|---|
| 1 | **Sinh dense embedding** (cùng model CLIP với online) | VastAI | Thiếu là mất nhánh mạnh nhất, và hỏng **im lặng** |
| 2 | **`dominant_colors` có tên + ratio** | VastAI | Không có thì `color_search` rỗng toàn corpus |
| 3 | **`quality` + `selection_score` từ `frame_selection`** | VastAI | Bật `safe_frame` mà không cần code online mới |
| 4 | **`actions[]` → `action_tags`** | adapter | 57% → ~100%, nội dung thật thay vì suy từ caption |
| 5 | **`index_text` thành `caption_type` riêng** | adapter | Không phí entity/relation; **phải đo** vì có thể loãng BM25 |
| 6 | Cập nhật adapter theo tên trường v2.0 | adapter | Cần một file mẫu thật trước |
| 7 | Lọc record theo `parse_ok`/`uncertainties` trước assemble | adapter | `assemble` không biết record nào model tự nhận là không chắc |
| 8 | Chạy lại OCR theo `model_retry_recommended` | VastAI | Trúng đích, rẻ hơn OCR lại toàn bộ |

Mục **1–3 là điều kiện cần**: thiếu chúng thì corpus Route2 chạy được nhưng ba
tính năng online chết hoặc chạy mù, và không cái nào tự báo lỗi.

---

## 8. Kiểm tra bắt buộc sau khi có pack Route2

```powershell
python -m scripts.verify_stage_pack storage/packs --all
python -m offline assemble --packs storage/packs --out storage/exports_full
python -c "from datasection.exporter import verify_export; from pathlib import Path; verify_export(Path('storage/exports_full'))"
curl -s http://127.0.0.1:8000/v1/search/capabilities
```

Rồi **bắt buộc** kiểm ba điều dưới đây bằng một truy vấn thật — cả ba đều là lỗi
đã xảy ra và đều **không tự báo**:

| Kiểm | Cách | Hỏng thì thấy gì |
|---|---|---|
| dense_visual có sống không | `branch_status` của một truy vấn thật | Tên nhánh là `lexical_hash_fallback` chứ không phải `dense_visual` |
| nhánh nào thực sự trả kết quả | `branch_status`, **không phải** `/capabilities` | `/capabilities` liệt kê nhánh ĐÃ ĐĂNG KÝ, kể cả nhánh không có dữ liệu |
| OCR có vào index không | đếm scene còn chữ sau bộ lọc lớp phủ | Từng có ca `AIC_OCR_OVERLAY_DF` xoá sạch **0/765** scene trong im lặng |

---

## 9. ĐÃ CÀI 10/08 — hạ tầng nhiều index dense

Phần này không còn là đề nghị; nó chạy được rồi. Mặc định **giữ nguyên hành vi cũ**.

### Cấu hình

```bash
# Rỗng (mặc định) = một nhánh `dense_visual` như trước.
AIC_DENSE_INDEXES=

# Nhiều index song song:
AIC_DENSE_INDEXES=clip_vit_l14_v1:storage/models/clip-vit-large-patch14,jina_v2:jinaai/jina-clip-v2:jina
```

Định dạng `<embedding_name>:<model_path>[:<kind>]`, ngăn cách bằng dấu phẩy.
`kind` ∈ `clip | siglip | jina`, bỏ trống thì suy từ đường dẫn.

`<embedding_name>` phải **khớp `embedding_refs[].embedding_name`** trong export —
đó là khoá nối giữa vector ảnh offline và text encoder online.

### Ba họ model, ba API khác nhau

| kind | Nạp | Encode |
|---|---|---|
| `clip` | `CLIPModel` + `CLIPProcessor` | `get_text_features`, `padding=True` |
| `siglip` | `SiglipModel` + `AutoProcessor` | `get_text_features`, **`padding="max_length"`** |
| `jina` | `AutoModel(trust_remote_code=True)` | `model.encode_text([...])` |

⚠️ SigLIP huấn luyện với chuỗi **độ dài cố định**; dùng `padding=True` như CLIP
sẽ cho vector lệch mà không báo lỗi.

### Đặt tên nhánh

Bật cờ này thì nhánh đổi tên thành `dense_<embedding_name>` — **kể cả khi chỉ khai
một index**. Không giữ lẫn `dense_visual` cho cái đầu rồi `dense_x` cho phần còn
lại: bật cờ là đổi topology có chủ đích, tên lẫn lộn sẽ làm mọi cấu hình trọng số
và mọi `--disable-branch` đã lưu trỏ sai chỗ mà không báo.

```
AIC_DENSE_INDEXES rỗng   ->  dense_visual
khai clip_vit_l14_v1     ->  dense_clip_vit_l14_v1
```

### Hai chốt an toàn — đều chặn lỗi "vẫn chạy nhưng vô nghĩa"

**1. Tên index không có trong export → chặn khởi động**, kèm liệt kê tên đang có:

```
AIC_DENSE_INDEXES khai index ['jina_v2'] nhưng export không có vector nào mang
`embedding_name` đó. Tên CÓ trong export: ['clip_vit_l14_v1'].
```

**2. Lệch chiều vector → chặn khởi động.** Text encoder 512 chiều gặp vector ảnh
768 chiều nghĩa là khai sai model. Không chặn thì cosine vẫn ra số, thứ hạng vẫn
có, và không ai biết nhánh đó đang cho kết quả rác:

```
index dense 'x': text encoder cho vector 512 chiều nhưng vector ảnh trong export
là 768 chiều. Gần như chắc chắn là khai sai model.
```

### Kiểm chứng

Chạy với một index đúng tên, so với đường mặc định trên 84 truy vấn gold
(KIS + TRAKE + AVS): **khớp từng số**, không hồi quy.

```
                          mac dinh   1 index
KIS R@1                      0.750     0.750
KIS MRR                      0.852     0.852
TRAKE video_recall@1         1.000     1.000
TRAKE mean_r                 0.354     0.354
AVS nDCG@100                 0.565     0.565
AVS event_coverage           0.884     0.884
```

Test: `tests/test_dense_multi_index.py` (13 ca) — parse spec kể cả đường dẫn
Windows có dấu hai chấm, suy `kind`, và cả hai chốt an toàn.

### Chưa đo được, và vì sao

Không có vector Jina/SigLIP trong export nên **chưa biết nhiều index có tốt hơn
một index không**. Hạ tầng sẵn sàng; câu trả lời phải chờ dữ liệu.

Hai điều đã biết trước, đừng bỏ qua khi đo:

- **Thêm nhánh không tự động tốt lên.** `bm25_ocr` sống lại còn làm KIS R@1 tụt
  0.583 → 0.500; `color_search` sống mà đóng góp bằng 0. Xem [docs/30 §3](30_SYSTEM_DIAGNOSIS.md).
- **Fusion đang là `norm_max`**, không phải RRF. Nhiều nhánh dense cùng modality
  `visual` sẽ chia nhau trọng số visual — cần đo lại trọng số, không chỉ bật thêm.

---

## 10. Tài liệu liên quan

- [docs/29](29_DATA_CONTRACT.md) — hợp đồng dữ liệu đầy đủ, trường nào online thực sự đọc
- [docs/30](30_SYSTEM_DIAGNOSIS.md) — chẩn đoán toàn hệ thống, §2 tầng dữ liệu
- [docs/31](31_TRAKE_EXPERIMENT_PLAN.md) — vì sao "thêm nhánh" không tự động tốt lên
- [scripts/route2_to_stage_packs.py](../scripts/route2_to_stage_packs.py) — adapter, cần cập nhật theo §5
