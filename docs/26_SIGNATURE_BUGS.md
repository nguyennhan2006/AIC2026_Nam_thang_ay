# 26 — Sáu lỗi trong tầng chấm điểm KIS và AVS, và hai kết luận cũ phải sửa

Tiếp §7 của `docs/25`: "phân tích 8 cặp top-2 chọn sai — rẻ nhất, và nó quyết
định mọi thứ sau". Phân tích đó làm được **hoàn toàn từ dữ liệu đã lưu**
(`outputs/evaluation/p2/kw_false.json`), không cần chạy lại pipeline. Và nó
cho thấy vấn đề lớn hơn 8 cặp.

---

## 1. Kết luận cũ sai một nửa

`docs/25` §2 viết: *"Tầng chấm điểm task đang gánh toàn bộ. `KisProcessor` kéo
từ 2/32 lên 18/36."* Con số đó đúng nhưng nó là số TỔNG. Nhìn theo từng truy
vấn, thứ hạng của gold TRƯỚC và SAU `KisProcessor`:

```
đẩy LÊN 10 truy vấn    đẩy XUỐNG 15 truy vấn    giữ nguyên 2
```

| bị đẩy xuống | trước → sau | | được đẩy lên | trước → sau |
|---|---|---|---|---|
| V002_KIS_E01 | 7 → **53** | | V002_KIS_M01 | 60 → **5** |
| V002_KIS_H02 | 14 → **57** | | V002_KIS_H03 | 92 → **40** |
| V003_KIS_H02 | 67 → 80 | | V001_KIS_H02 | 64 → **22** |
| V002_KIS_M04 | 5 → 11 | | V003_KIS_E04 | 22 → 11 |
| V003_KIS_M01 | 2 → 8 | | V002_KIS_E03 | 6 → **1** |
| V001_KIS_M02 | **1** → 4 | | V002_KIS_H04 | 12 → 8 |
| V003_KIS_E01 | **1** → 3 | | (4 truy vấn còn lại +1..+2) | |
| V001_KIS_M04 | 2 → 5 | | | |
| (7 truy vấn còn lại −1..−4) | | | | |

Hai điều đọc ra:

**`KisProcessor` phá nhiều hơn nó sửa, tính theo số truy vấn.** Nó lãi tổng vì
vài cú kéo rất mạnh (+55, +52, +42) bù cho mười lăm cú tụt. Trong đó có hai
truy vấn mà **fusion đã đặt gold ở hạng 1** rồi bị đẩy xuống hạng 3 và 4.

**Hai ca tụt nặng (−46, −43) không phải nhiễu.** Cơ chế: `must_coverage` là
phép AND từ vựng giữa vài từ đầu của truy vấn và văn bản scene. Khi caption mô
tả cùng cảnh bằng từ khác, gold được 0 còn hàng chục scene chung chung được
0.67 — mất trắng 0.6 điểm. Nó cộng thêm một lần nữa đúng thứ BM25 đã đo, và
phạt đúng thứ dense visual vừa tìm ra.

---

## 2. Lỗi 1 — `PROPER_NOUN_RE` nhận mọi từ thường có dấu là danh từ riêng

`online/services/kis.py`. Lớp ký tự cũ:

```python
r"[A-ZĐÀ-Ỹ][a-zà-ỹA-ZĐÀ-Ỹ]{1,}"
```

Dải Unicode `À`(U+00C0)–`Ỹ`(U+1EF8) **xen kẽ hoa và thường**: `à á ạ đ ê ô` đều
nằm trong đó. Nên mọi từ thường mở đầu bằng nguyên âm có dấu bị nhận là tên
riêng. `rare_cues` trên 36 truy vấn KIS thực tế chứa:

```
đỏ  đặt  đứng  đất  đêm  đường  đó  đi  đạp  đáy  đàn  được  đóng  đầy  đục
```

Đây là những từ PHỔ BIẾN NHẤT tiếng Việt, trong khi `rare_cues` được định nghĩa
là "khớp được thì gần như chắc chắn đúng" và ăn trọng số 0.25. Hệ quả: gần như
scene nào cũng khớp vài cue, `rare_score` bão hòa, và cue thật (`Copenhagen`)
bị chia đều trọng số với `đi`, `đạp`.

Sửa: liệt kê tường minh tập chữ hoa tiếng Việt. Sau khi sửa, `rare_cues` còn
lại đúng thứ nó phải là:

```
Copenhagen · BMX · Kingfoodmart · Earth Rover · Andrew Bailey
Intel Ocotillo Campus · Amsterdam · Hòa Bình · 42,3 · National Park Service
```

23/36 truy vấn không còn cue nào — đúng, vì chúng thật sự không có dấu hiệu
hiếm nào, và `rare_score = 0` trung thực hơn là một con số bịa.

## 3. Lỗi 2 — `must_match` lấy ba từ đầu câu, nên toàn là từ chỉ "một cảnh"

`must = quoted + content[:3]`, ăn trọng số 0.6 — lớn nhất sau retrieval. Thực tế:

| truy vấn | `must_match` cũ |
|---|---|
| "Tìm **phóng sự** cháy rừng…" | `phóng`, `cháy`, `rừng` |
| "Tìm **đoạn** đội y tế vận chuyển…" | `đoạn`, `đội`, `vận` |
| "Tìm **hiện trường** ban đêm…" | `hiện`, `trường`, `ban` |

`phóng sự`, `đoạn`, `hiện trường` là từ chỉ THỂ LOẠI, không chỉ nội dung.
`keyword_extraction.py` đã có sẵn `SCENE_NOUNS` cho đúng việc này nhưng
`kis.py` giữ bản `STOPWORDS` riêng, thiếu 7 từ: `bang clip doan hien phong su
truong`.

Kèm theo, dòng khử trùng so sai vế:

```python
if normalized not in content:   # `normalized` không dấu, `content` có dấu
    content.append(token)
```

nên nó gần như không khử được gì; một từ lặp lại vẫn chiếm chỗ trong
`content[:3]` và đẩy dấu hiệu thật ra ngoài.

Sau khi sửa: `cháy rừng bắt`, `đội vận chuyển`, `ban đêm năm`, `quả bưởi xanh`,
`thợ lặn khảo`.

## 4. Lỗi 3 — `không` bị đọc là phủ định kể cả khi nó là danh từ ghép

`online/services/negative_constraints.py`. Trên 120 truy vấn gold, **3/5**
constraint trích ra là dương tính giả, cả ba cùng kiểu:

| truy vấn | cụm bị cấm | thực tế trong câu |
|---|---|---|
| V001_KIS_H02 | `gian bao tang` | "**không gian** bảo tàng" |
| V001_TRAKE_H01 | `gian ben trong bao tang` | "**không gian** bên trong bảo tàng" |
| V001_VQA_M03 | `va mang tui nuoc de ho tro dap lua` | "bay **trên không** và mang túi nước" |

Ca thứ ba nguy hiểm nhất: cụm bị cấm chính là đặc điểm nhận dạng của đáp án
đúng (trực thăng mang túi nước chữa cháy) — mà đây là **lọc cứng**, xóa
candidate khỏi pool chứ không phải hạ điểm.

Sửa: `không` chỉ là phủ định khi nó không đứng trước một âm tiết ghép
(`gian, khí, quân, trung, tặc, vận, lực, phận`) và không đứng sau một từ dẫn
(`trên, dưới, giữa, hàng, bầu, phòng, vùng`).

## 5. Lỗi 4 — nhưng cả tính năng negative-constraint đang vô hiệu

Kiểm tra thẳng trên 765 scene, đếm số scene mà mỗi constraint thật sự loại bỏ:

```
   0 scene · gian bao tang
   0 scene · va mang tui nuoc de ho tro dap lua
   0 scene · gian ben trong bao tang
   0 scene · thay phuong tien hay con nguoi          <- constraint ĐÚNG
   0 scene · tinh canh phuong tien chi chay binh thuong tren duong   <- ĐÚNG
```

**Không constraint nào loại được scene nào**, kể cả hai cái đúng. Vì
`_NEGATION_RE` bắt tới dấu câu gần nhất nên cụm dài 5–9 từ, mà điều kiện lọc
là "MỌI từ đều có mặt trong scene" — không bao giờ thỏa.

Nghĩa là lỗi 3 sửa đúng nhưng **chắc chắn không đổi điểm** trên bộ gold này.
Ghi lại để không ai đi tìm phần tăng điểm không tồn tại. Muốn tính năng này
thật sự hoạt động thì phải rút ngắn cụm — và đó là hành vi MỚI, có rủi ro xóa
nhầm gold, nên phải đo riêng chứ không gộp vào đây.

---

## 6. Lỗi 5 — nDCG của AVS vượt 1, tức không xếp hạng được gì

Phát hiện khi lượt đo SIG-01a trả về `nDCG@100 = 1.185`. nDCG theo định nghĩa
không thể vượt 1.

```python
grades = [_gold_grade(row) for row in response.avs]      # tới 100 phần tử
ideal  = [iv.relevance_grade for iv in item.intervals]   # 2–7 phần tử
return dcg(grades) / dcg(sorted(ideal, reverse=True))
```

Một interval gold trải qua **15–64 scene** (đo trên bộ gold thật). Trả về cả
chùm scene của cùng một sự kiện thì mỗi scene đều được chấm điểm, nên tử số
cộng trên nhiều vị trí hơn mẫu số.

| | nDCG > 1 | max |
|---|---|---|
| baseline P2 (`docs/25`) | 2/24 truy vấn | 1.429 |
| lượt sau (cap AVS = 20) | 14/24 truy vấn | 2.357 |

**Hệ quả: mọi kết luận AVS trước đây đều dựa trên một chỉ số không phải thứ
tự** — gồm AVS-GRADE-01, việc chọn `grade_mode`, và con số 0.522 ghi trong
`docs/25`. Phải đo lại toàn bộ.

Sửa ở **tử số**, không phải mẫu số. Gold ghi rõ `dedup_requirement: at most one
representative segment per news report`, nên scene thứ hai của cùng một sự
kiện là trùng lặp chứ không phải phát hiện mới: `dedup_grades_by_event()` cho
nó điểm 0. Sau đó `grades` là hoán vị con của `ideal` có chèn 0, nên
`dcg(grades) ≤ dcg(ideal)` — chặn trên 1.0 là hệ quả toán học, không phải kẹp
bằng `min()`.

Bỏ hướng "mẫu số = mọi scene chạm gold" vì nó **phạt việc khử trùng**, đúng
thứ đề bài yêu cầu phải làm.

## 7. Và một confound vẫn còn nằm trong `.env.fpt.local`

Cùng lượt đó, `result_count` của AVS nhảy từ 9 lên 35–60 — thoạt nhìn giống
bất định giữa các lượt. Không phải:

```
pre_grade       p50 88.5   (hai lượt GIỐNG HỆT)
post_grade      p50 42.5   (giống hệt)
prefusion_total p50 475     (giống hệt)
result_count    9  ->  42   (khác)
```

Retrieval và cổng grade tất định tuyệt đối. Chỉ khác ở chặn đầu ra:
`.env.fpt.local` (dòng 418 sau khi thêm chú thích) đang là `AIC_AVS_MAX_RESULTS_PER_VIDEO=20` (60 = 20×3
video), còn baseline `docs/25` chạy ở mặc định 3 (9 = 3×3).

Đây đúng là confound đã ghi nhận một lần trước đây và **chưa được gỡ**. Với
metric cũ, nới cap là tăng điểm miễn phí vì scene trùng cũng được cộng. Với
metric đã sửa, scene trùng ăn 0 và còn chiếm chỗ — nên cap mới thật sự là một
đánh đổi precision/recall đo được. Giờ mới đáng chạy ablation cap.

## 8. Vì sao cả năm lỗi sống sót qua 561 test

Cùng một nguyên nhân: test dùng dữ liệu do test tự dựng, sạch hơn dữ liệu thật.

- `test_numbers_and_proper_nouns_are_rare_cues` kiểm `UNESCO` và `14` — cả hai
  đều đúng ở bản cũ. Không có test nào hỏi *"cái gì KHÔNG được là rare cue"*.
- `test_extracts_bare_khong_phrase` dùng `"không mưa"`, không có danh từ ghép.
- Không test nào chấm `build_signature` trên truy vấn thi đấu thật.
- `NdcgTests` có `test_perfect_ranking_scores_one` với `grades == ideal`, tức
  đúng trường hợp KHÔNG có trùng lặp. Không test nào hỏi *"nDCG có thể vượt 1
  không"*.

Đã thêm test khóa cho từng lỗi, viết theo hướng **phủ định** (cái gì không được
xuất hiện), vì đó là hướng mà bộ test cũ bỏ trống hoàn toàn.

---

## 9. Kết quả: hai bản sửa, và chúng chuyển được sang video khác

Đo trên `examples/gold_all3.jsonl`, 36 truy vấn KIS, `PYTHONHASHSEED=0`:

| | R@1 | MRR | `top1_pairwise` | V001 | V002 | V003 |
|---|---|---|---|---|---|---|
| baseline P2 | 0.500 | 0.671 | 0.692 (n=26) | 5/12 | 6/12 | 7/12 |
| + SIG-01a | 0.500 | 0.678 | 0.667 (n=27) | 5/12 | 6/12 | 7/12 |
| + SIG-01b | **0.583** | **0.725** | **0.778** (n=27) | **7/12** | **7/12** | 7/12 |

SIG-01a không đổi R@1 nhưng kéo thêm một gold vào top-2 (n 26→27, và vẫn đúng
18 lần) — đúng như kỳ vọng: nó chỉ dọn nhiễu khỏi `rare_cues`, và 23/36 truy
vấn vốn không có cue nào.

SIG-01b được +3 truy vấn, **trải trên hai video**, không video nào tụt. Đây là
tiêu chuẩn nhận, chứ không phải con số tổng — xem mục sau để biết vì sao.

---

## 10. Chỉnh trọng số: DROP, và đây là lý do

Sau khi có công cụ sweep, dò 72 cấu hình quanh `KisConfig`. Cấu hình dẫn đầu
trên toàn bộ 36 truy vấn:

```
must=0.2  rare=0.9  safe=0.3  agreement=0.0     R@1 23/36 (0.639)   MRR 0.771
                                    so với BASE  R@1 21/36 (0.583)   MRR 0.725
```

Hướng đi còn có vẻ hợp lý về cơ chế: hạ `must` vì nó là heuristic vị trí độ
chính xác thấp, nâng `rare` vì sau SIG-01a nó mới thật sự hiếm.

**Nhưng cả +2 đều nằm ở V001.** Kiểm bằng cách chọn trọng số trên một video
rồi đo trên hai video còn lại:

```
chọn trên V001  →  hai video kia  14/24 → 12/24   (-2)
chọn trên V002  →  hai video kia  14/24 → 14/24   (+0)
chọn trên V003  →  hai video kia  14/24 → 14/24   (+0)
```

Và trong **cả 72 cấu hình, không cấu hình nào cải thiện được video nào ngoài
V001**. Cái +2 ở V001 phải trả bằng −2 ở nơi khác.

Đây đúng là kiểu hỏng đã ghi nhận ở TRAKE (`video_recall@1` 1.000 trên V001 so
với 0.375 trên holdout). Trọng số bị khớp vào đặc thù của một video.

Giữ nguyên `KisConfig` mặc định. Ghi lại để không ai dò lại vòng này.

> Đối chiếu: SIG-01b cũng chỉ là một thay đổi nhỏ, nhưng nó tăng ở **hai**
> video và không đánh đổi ở đâu. Khác biệt không nằm ở độ lớn con số tổng, mà
> ở chỗ nó sửa một lỗi ngôn ngữ chung hay khớp vào một bộ dữ liệu cụ thể.

---

## 11. Sau khi sửa, tầng chấm điểm KIS còn đóng góp gì?

Câu hỏi tự nhiên tiếp theo, và nay trả lời được trong một giây nhờ bản dump.
Đặt toàn bộ trọng số signature về 0, chỉ xếp theo điểm retrieval:

| | R@1 | MRR | V001 | V002 | V003 |
|---|---|---|---|---|---|
| chỉ retrieval | **21/36** | **0.746** | 9/12 | 5/12 | 7/12 |
| BASE (đủ signature) | **21/36** | 0.725 | 7/12 | 7/12 | 7/12 |
| bỏ `must` | 20/36 | 0.729 | 10/12 | 5/12 | 5/12 |
| bỏ `rare` | 20/36 | 0.711 | 7/12 | 6/12 | 7/12 |
| bỏ `safe` | 21/36 | 0.727 | 7/12 | 7/12 | 7/12 |
| bỏ `agreement` | 21/36 | 0.730 | 7/12 | 7/12 | 7/12 |

**R@1 bằng nhau, MRR của bản không-signature còn cao hơn.** Tầng chấm điểm
task hiện chỉ dịch chuyển điểm giữa các video (V001 9→7, V002 5→7), không tạo
ra điểm mới.

Đây là chỗ phải nói thẳng: kết luận của `docs/25` §2 — *"tầng chấm điểm task
đang gánh toàn bộ, kéo từ 2/32 lên 18/36"* — được đo trên chính bộ signature
đang hỏng. Sửa xong hai lỗi thì retrieval một mình đã ngang bằng.

Nhìn lại quỹ đạo của tầng này qua ba lần đo:

| | đẩy lên | đẩy xuống | giữ nguyên | tụt nặng nhất |
|---|---|---|---|---|
| trước khi sửa | 10 | 15 | 2 | **−46, −43** |
| sau SIG-01a+01b | 5 | 8 | 23 | −5 |

Nó đã thôi phá (hai ca −46/−43 nay là 1→2 và 2→**1**), nhưng cũng thôi đóng
góp. Chưa đủ cơ sở để gỡ bỏ — đánh đổi giữa các video là hoà, không phải
thắng — nhưng đủ cơ sở để **ngừng đầu tư vào đây**. Phần xếp hạng thật sự đang
nằm ở retrieval + fusion + rerank, và đó mới là chỗ đáng bỏ công tiếp.

---

## 12. Lỗi 6 — 73% tiêu chí AVS không thể khớp được scene nào

Cùng lớp lỗi với SIG-01b, nhưng ở `online/services/avs.py` thì hậu quả nặng
hơn vì tiêu chí này là **cổng chặn**, không phải điểm cộng.

`extract_criteria` lọc token của mỗi mệnh đề rồi **nối các token còn sót lại
thành một cụm**, sau đó `_matches` khớp cụm nhiều từ bằng **chuỗi con**:

```python
options.append(" ".join(terms))          # dựng cụm từ các mảnh còn sót
...
return term_norm in normalized_text      # rồi đòi nó xuất hiện nguyên văn
```

Bộ lọc bỏ token có `len(normalize_vi(token)) < 3`, mà rất nhiều âm tiết tiếng
Việt chỉ 2 ký tự sau khi bỏ dấu. Từ ghép bị xé:

```
cứu hộ    -> cứu        bảo vệ    -> bảo
hỗ trợ    -> trợ        tìm kiếm  -> kiếm
```

Kết quả là những cụm chưa từng tồn tại trong bất kỳ caption nào:
`'phóng hoạt động bảo môi trường'`, `'chó trợ kiếm'`, `'dùng động vật trợ bảo
tồn'`. Đếm trên 765 scene:

```
tổng option inclusion: 67  (57 là cụm nhiều từ)
không khớp được scene NÀO:  49/67  (73%)
```

Và không sửa được ở phía trích. Thử giữ lại âm tiết 2 ký tự, thử đổi sang bộ
stopword hợp nhất — tỉ lệ chết vẫn 71–77%, vì stopword cũng cắt từ giữa cụm.
**Không thể dựng một cụm khớp được bằng cách xoá từ khỏi câu rồi nối lại**:
mọi phép xoá đều phá tính liền mạch.

Phải sửa ở phía KHỚP. Option là một túi token nội dung, nên phải chấm theo độ
phủ token chứ không phải chuỗi con. Đo mức tách giữa scene thuộc gold và scene
ngoài gold, trên toàn bộ 24 truy vấn AVS:

| cách chấm | scene ĐÚNG | scene SAI | tách |
|---|---|---|---|
| chuỗi con (hiện tại) | 0.053 | 0.027 | +0.026 |
| độ phủ token | 0.215 | 0.144 | **+0.071** |

Gấp 2.7 lần. Và con số 0.053 mới là điều đáng nói: **tiêu chí AVS hiện gần như
không kích hoạt ngay cả trên scene đúng** — cổng đang chặn bằng một điều kiện
mà chính đáp án cũng không thỏa.

### Đã sửa — biến thể C, chọn bằng ablation 5 cách trên cùng 2098 pack

`scripts/dump_avs_candidates.py` ghi nguyên vẹn `EvidencePack` nên
`scripts/replay_avs_grading.py` gọi được CHÍNH `AvsProcessor.rank` thật và chỉ
thay `AvsCriteria.grade`. Chênh lệch giữa các biến thể vì thế quy được về đúng
một nguyên nhân.

| | nDCG@100 | P@100 | event_cov | gold bị gate loại |
|---|---|---|---|---|
| A chuỗi con (cũ) | 0.545 | 0.085 | 0.752 | 3.75 |
| B độ phủ token | 0.589 | 0.051 | 0.831 | 0.42 |
| **C + trọng số IDF** | **0.598** | 0.052 | **0.841** | **0.42** |
| D + thưởng cụm/khoảng cách | 0.589 | 0.051 | 0.833 | 0.42 |
| E + max theo trường | 0.567 | 0.052 | 0.841 | 0.42 |

C tăng trên **cả ba video** (0.643 / 0.654 / 0.497) và **13 truy vấn tăng, 3
giảm, 8 hoà** — khác hẳn kiểu overfit ở mục 10. Hard-negative tuyệt đối tăng
36.9 → 55.4 nhưng **theo tỉ lệ chỉ 0.932 → 0.948**; nó phình vì C trả về nhiều
kết quả hơn (39.6 → 58.4), không phải vì lọc kém đi.

Hai hướng phức tạp hơn đều DROP: thưởng cụm nguyên văn + khoảng cách gần (D)
tụt về 0.589, và chấm theo từng trường rồi lấy max (E) còn 0.567.

Xác nhận end-to-end trên đủ 4 task: nDCG 0.598, P@100 0.052, event_coverage
0.841 — **trùng dự đoán của replay đến ba chữ số thập phân**. Và
`correct_candidate_dropped_by_grade` **1.000 → 0.083**: từ 100% truy vấn AVS có
candidate đúng bị chính cổng loại, xuống 8%.

Hai phép gộp phải giữ tách bạch, và đây là chỗ dễ sai nhất:

- giữa các *option* trong một nhóm (tách bởi "hoặc"/"hay"): `max` — chúng là
  các cách nói của cùng một ý;
- giữa các *nhóm*: **trung bình** — truy vấn đòi cả "người cứu hộ" lẫn "đưa
  nạn nhân lên xe" thì khớp một nửa không phải khớp.

Trọng số IDF không phải trang trí: bộ tiêu chí đầy `người`, `hoạt động`,
`đang`, `cảnh`, và đếm token thô cho chúng ngang với `thợ lặn` hay `rùa biển`.
Chênh lệch B → C (0.589 → 0.598) chính là phần đó.

---

## 13. Công cụ: sweep trọng số KIS không cần chạy lại pipeline

Mỗi lượt `eval_tasks` trên 120 truy vấn mất ~43 phút, gần hết là chờ API. Thử
một bộ trọng số mà tốn 43 phút thì không thể dò được gì — 72 cấu hình ở mục
trên sẽ mất 52 giờ.

```bash
python -m scripts.dump_kis_features --disable-branch event_search \
    --disable-branch ocr_fuzzy --out outputs/evaluation/kis_features.json
python -m scripts.sweep_kis_weights --holdout
```

`dump_kis_features.py` chạy retrieval **một lần** và ghi sáu thành phần điểm
của từng candidate (`retrieval, must, rare, nice, agreement, safe,
contradicted`). Nó chộp đầu vào bằng cách bọc `KisProcessor.rank` chứ không
dựng lại đường fusion — bản dựng lại sẽ trôi khỏi bản thật lúc nào không biết.

`sweep_kis_weights.py` xếp hạng lại offline, 72 cấu hình dưới một giây. Trước
khi in bất cứ con số nào, nó tính lại thứ hạng ở đúng bộ trọng số đang chạy và
so với `online_rank` đã ghi kèm; lệch một truy vấn là dừng hẳn, vì khi đó bản
dump thiếu thành phần nào đó và mọi kết luận phía sau đều sai.

`--holdout` là phần đáng dùng nhất: nó chọn trọng số trên một video rồi đo
trên hai video kia. Không có nó thì mục 10 đã kết luận ngược.
