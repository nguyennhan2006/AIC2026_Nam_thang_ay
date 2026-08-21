# 22 — TRAKE: sinh chuỗi theo n điểm neo và cách chấm điểm chuỗi

Thiết kế cho tầng chọn frame — tầng đã ba lần bị tối ưu nhầm chỗ.

**Mục 4b đã đo và loại hai trong ba giả thuyết.** Phần đề xuất còn lại (điểm
tuyệt đối từ VLM + quy hoạch động n điểm neo) chưa cài, chưa đo.

---

## 1. Ba dữ kiện định hình toàn bộ thiết kế

**R-score cho điểm TỪNG BƯỚC, không phải được-ăn-cả.**

```python
# online/competition/scorer.py:88
hits = sum(1 for frame_idx, window in zip(item.frame_ids, gold.step_windows)
           if window.contains(frame_idx))
r_score = hits / total
```

Chuỗi đúng 3/4 bước được 0.75. Hệ quả trực tiếp: hàm mục tiêu nội bộ phải là
**tổng/trung bình** xác suất trúng từng bước, **không phải `min`**. `min`
(mắt xích yếu nhất) chỉ đúng cho metric all-or-nothing. Dùng `min` ở đây sẽ
vứt bỏ những chuỗi ăn chắc 3/4 bước để đổi lấy chuỗi "cân bằng" mà không bước
nào chắc.

**Không được bỏ trống bước nào.** Bước sai và bước bỏ trống đều được 0 điểm,
nhưng bước đoán bừa còn có cơ hội trúng. Luôn điền đủ n frame kể cả khi độ tin
cậy thấp.

**Candidate luôn tồn tại.** Đo trên bộ gold: khoảng cách xa nhất từ frame gold
tới keyframe gần nhất là 92 frame = 3.07s. Không bước nào "không tìm được".

---

## 2. Luật TRAKE, diễn giải thành ràng buộc kỹ thuật

Theo luật đã chốt: **thứ tự phải đúng**, **ngữ nghĩa từng keyframe phải khớp
chặt với hành động được mô tả**, còn **lệch so với frame answer thì chấp nhận
được miễn nằm trong range của hành động**.

Ba hệ quả, trong đó hai cái đi ngược cài đặt hiện tại:

| Luật | Ràng buộc | Hiện tại |
|---|---|---|
| Thứ tự phải đúng | **Cổng cứng**: `f_1 < f_2 < ... < f_n` | ĐÃ là cổng cứng (`sequence_search.py:87`) — `ordering_weight` là chuyện khác, nó xếp hạng *video* |
| Ngữ nghĩa phải chặt | Điểm chuỗi do **độ khớp ngữ nghĩa tuyệt đối** quyết định | điểm retrieval (mang tính so sánh) |
| Lệch vị trí được tha | **Không phạt khoảng cách** | phạt dead-zone + trần — và ĐO ĐƯỢC là cần, mục 4c |

`gap_penalty_per_sec` không có cơ sở nào trong luật: không điều khoản nào
thưởng cho chuỗi gọn về thời gian. Nghi vấn "nó đang lấn át độ liên quan" đã
được **kiểm chứng và bác bỏ** trên đường `processor` — xem mục 4b — nhưng phép
đo đó KHÔNG nói gì về đường đang chạy thật (`link_event_hits`), nơi công thức
phạt khác hẳn. Mục 4c ghi lại chỗ đó.

---

## 3. Điểm từng bước phải TUYỆT ĐỐI, không phải tương đối

Đây là chỗ dễ sai nhất. Điểm retrieval trả lời *"frame này tốt hơn frame kia
không"*. R-score hỏi *"frame này có nằm trong cửa sổ của hành động không"* —
một câu hỏi tuyệt đối. Chuẩn hoá điểm retrieval thành [0,1] **không** biến nó
thành xác suất: frame tốt nhất trong một tập toàn frame sai vẫn được điểm cao.

Nguồn tín hiệu tuyệt đối duy nhất đang có là VLM verifier:

```
s_i(f) = relevance × must_match_coverage        (từ FptVlmReranker)
s_i(f) = 0                                       nếu contradictions ≠ []
```

Đúng là "đảm bảo ngữ nghĩa của keyframe khớp chặt với hành động".

> **Điều này cứu lại nhánh VLM.** FPT-WIRE-01 cho thấy VLM rerank vô dụng với
> KIS — trùng baseline đúng từng chữ số. Lý do: KIS chỉ cần *xếp hạng*, mà
> text rerank đã xếp tốt rồi. TRAKE cần *phán quyết tuyệt đối* từng bước, và
> đó chính là việc VLM làm được còn retrieval thì không. Cùng một công cụ, vô
> dụng ở chỗ này và cần thiết ở chỗ kia.

---

## 4. Cách chấm đề xuất

### 4.1 Nguyên tắc

**Hàm mục tiêu nội bộ phải là ước lượng của chính R-score chính thức**, không
phải một hàm tự nghĩ ra. Mọi số hạng thêm vào mà không tương ứng với điều
khoản chấm điểm nào đều là chỗ để tối ưu chệch hướng.

```
S(C) = (1/n) · Σ s_i(f_i)          ← chính là kỳ vọng của R-score
```

Ràng buộc cứng:

```
f_1 < f_2 < ... < f_n              thứ tự nghiêm ngặt
f_i đôi một khác nhau              chặn chuỗi suy biến dồn vào một chỗ
```

### 4.2 Cải tiến đề xuất của bạn: n điểm neo là BỘ SINH, không phải bộ chọn

Đề xuất gốc: với n hành động, lần lượt lấy hành động thứ *i* làm gốc, lan ra
hai phía để lấy frame cho các hành động còn lại → n chuỗi. Rồi chọn một.

Chọn nguyên một chuỗi là bỏ phí. Chuỗi neo ở hành động 2 có thể chọn bước 2 và
3 rất tốt nhưng bước 4 tệ, trong khi chuỗi neo ở hành động 4 lại có bước 4
hoàn hảo. Không lý do gì phải chọn một trong hai.

**Gộp lại rồi để quy hoạch động chọn tổ hợp tối ưu:**

```
F_i  = hợp của các frame ứng viên cho bước i, gom từ CẢ n lần neo
mục tiêu: chọn f_1 < f_2 < ... < f_n sao cho Σ s_i(f_i) lớn nhất
```

Quy hoạch động, sắp mọi frame ứng viên theo `frame_idx` tăng dần:

```
dp[1][k] = s_1(f_k)
dp[i][k] = s_i(f_k) + max( dp[i-1][j] : j < k )
```

Giữ prefix-max của `dp[i-1]` thì mỗi bước là O(|F|), tổng **O(n · |F|)** — với
n=4 và ~40 ứng viên thì không đáng kể so với chi phí gọi VLM.

Kết quả của DP **luôn ≥ chuỗi neo tốt nhất**, vì mỗi chuỗi neo bản thân nó là
một nghiệm khả thi của DP. Không có trường hợp nào DP thua.

Vậy n điểm neo vẫn giữ nguyên giá trị — nhưng ở vai trò **quyết định frame nào
lọt vào `F_i`**, không phải vai trò chọn chuỗi cuối. Neo vào hành động đặc
trưng nhất (ví dụ "cầm chai") thu hẹp vùng tìm cho các hành động mơ hồ hơn
("quay từ xa") rất tốt; neo vào hành động mơ hồ thì ngược lại. Chạy cả n lần
neo là cách rẻ để không phải đoán trước hành động nào đặc trưng.

### 4.3 Ngân sách gọi VLM

`n_bước × ứng_viên_mỗi_bước` lệnh gọi cho một truy vấn. Với n=4, 10 ứng
viên/bước → 40 lệnh. So với 1400 lệnh của VLM rerank cho KIS (đổi lại 0 cải
thiện), đây là chỗ đáng tiêu tiền hơn nhiều.

Giảm chi phí: chỉ chấm VLM cho top-k ứng viên theo điểm retrieval, vì retrieval
đủ tốt để **lọc thô** — nó chỉ không đủ để **phán quyết**.

---

## 4b. ĐÃ ĐO — hai giả thuyết bị loại, còn lại đúng một

Chạy trên nền cấu hình B (dịch bật, expansion tắt, VLM tắt), `PYTHONHASHSEED=0`.

### Ràng buộc hình thức KHÔNG lấn át độ liên quan — DROP

| cấu hình | mean_r_score |
|---|---|
| mặc định (`gap=0.002`, `order=0.6`) | 0.263 |
| `gap_penalty_per_sec=0` | 0.263 |
| `order_weight=0` | 0.263 |
| cả hai = 0 | **0.231** |

Tắt riêng từng cái không đổi gì; tắt cả hai thì *tệ hơn*. Nghi vấn ghi trong
`20_EXPERIMENT_LOG.md` ("ràng buộc hình thức đang lấn át độ liên quan") **sai**.

Lý do nhìn lại thì rõ: thứ tự vốn ĐÃ là cổng cứng
(`sequence_search.py:87` bỏ qua hit có frame nhỏ hơn frame trước), và
`max_gap_sec=300s` không bó khi khoảng cách bước thực chỉ ~12s. Hai tham số
này chưa bao giờ là thứ đang quyết định.

## 4c. Nới thời gian trên đường ĐANG CHẠY THẬT — ĐÃ ĐO, giả thuyết SAI

Mục 4b đo `SequenceConfig.gap_penalty_per_sec`, tham số của `TrakeProcessor`.
Nhưng deployment mặc định là `AIC_TRAKE_ENGINE=sequences`, tức chuỗi thật do
`online/services/temporal.py::link_event_hits` dựng, và ở đó phạt là **tuyến
tính từ gap = 0** với `lambda = 0.002`:

| gap giữa hai bước gold | p10 5s | p50 10s | p90 21s | max 36s |
|---|---|---|---|---|
| phạt (0.002/s) | 0.010 | **0.020** | 0.042 | 0.072 |
| điểm một scene sau RRF | ~0.04 | ~0.04 | ~0.04 | ~0.04 |

Đọc bảng đó rất dễ kết luận "chuỗi có nhịp đúng bằng trung vị gold đang bị lấy
mất nửa điểm một scene, vậy phạt phải yếu đi". **Kết luận đó sai**, và
GAP-RELAX-01 ([docs/20](20_EXPERIMENT_LOG.md)) đo ra điều ngược lại.

### Hình dạng mới

`online/services/temporal_gap.py`, dùng chung cho beam, dp và `sequence_search`:

```
penalty(gap) = min( lambda * max(0, gap - W),  cap )
```

| tham số | giá trị | căn cứ |
|---|---|---|
| `W` (`free_gap_sec`) | 60s | gold max 36s → chuỗi đúng không mất điểm. Đo: trung tính (.354 → .355) |
| `lambda` | 0.002 | **giữ nguyên**. Hạ xuống 2e-5 đo ra tệ hơn cả tắt hẳn |
| `cap` | 1.0 | ngưỡng gãy giữa 0.3 và 0.5; 1.0 nằm giữa hai điểm đo giống hệt nhau |

Ràng buộc **cứng** rút còn đúng hai: cùng video, và `(scene_idx, best_frame_idx)`
tăng nghiêm ngặt — hai bước được phép nằm trong cùng một scene miễn frame tiến
lên. Đo: trung tính, giữ vì đúng luật chứ không vì ăn điểm.

### Vì sao trần thấp lại hỏng

`cap=0.01` (¼ điểm một scene, đúng theo lập luận "phạt chỉ nên đủ phá hoà") làm
`video_recall@1` rơi 1.000 → 0.958 và `mean_r` 0.355 → 0.330, **tệ hơn cả tắt
hẳn phạt** (0.338).

Phạt khoảng cách không phải cái phá hoà giữa hai chuỗi gần bằng nhau trong cùng
một video. Nó là thứ **dìm chuỗi trải rộng ở video SAI** xuống để video đúng nổi
lên — và để làm việc đó nó cần thẩm quyền cỡ 0.5, hơn 12 lần điểm của một scene.
Bảng đầy đủ ở GAP-RELAX-01.

Chỗ nhầm trong lập luận của mục 2 bên trên: luật không thưởng chuỗi gọn về thời
gian, đúng — nhưng tính cục bộ thời gian vẫn là một **prior đúng về việc diễn
biến của một video trông như thế nào**, và trên corpus này prior đó đang gánh
việc chọn đúng video. Một tham số retrieval không cần có điều khoản tương ứng
trong luật chấm để có ích.

---

### Cửa sổ chấm KHÔNG phải ràng buộc đang bó — chỉ nới thì gần như vô ích

| nửa cửa sổ | mean_r_score | trần lý thuyết | khoảng cách tới trần |
|---|---|---|---|
| 2s (mặc định) | 0.263 | 0.800 | 0.537 |
| 3s | 0.294 | 0.971 | 0.677 |
| 4s | 0.319 | 1.000 | 0.681 |
| 5s | 0.350 | 1.000 | 0.650 |

Nới 2s → 4s đẩy trần lên **+0.200** nhưng điểm thật chỉ được **+0.056**.
Khoảng cách tới trần *rộng ra* chứ không hẹp lại.

**Kết luận: hệ đang chọn nhầm frame, không phải bị cửa sổ chặn.** Mọi ứng viên
đúng đều nằm sẵn trong tầm với — chỉ là không được chọn.

### Giả thuyết 3 — beam bỏ sót chuỗi tốt hơn: SAI (đã cài DP để kiểm)

Cài `search_sequences_dp` — quy hoạch động chính xác, **cùng hàm mục tiêu** với
beam. Test property trên 60 input ngẫu nhiên xác nhận DP không bao giờ thua
beam *trên hàm mục tiêu*, và có test dựng riêng một trường hợp beam rộng 1 cắt
nhầm mà DP tìm lại được.

Trên R-score thật:

| | mean_r_score |
|---|---|
| beam (mặc định) | 0.263 |
| DP, thiếu `max_gap_sec` | **0.094** |
| DP, đầy đủ ràng buộc | 0.231 |

Hai điều rút ra, cái thứ nhất quan trọng hơn:

**Lần cài đầu tôi bỏ mất `max_gap_sec` và điểm rơi xuống 0.094.** Chẩn đoán:
TRAKE_E02 cho chuỗi trải **980 giây**, nhảy sang tận frame 36764. `max_gap_sec`
là chặn **cứng**, và nó đang làm việc thật — khác hẳn `gap_penalty_per_sec`
(phạt **mềm**), thứ mà mục trên đo được là không ảnh hưởng gì. Trước đó tôi đã
gộp hai tham số này làm một nhóm "ràng buộc hình thức"; chúng không cùng loại.

**Với ràng buộc đầy đủ, DP và beam gần như trùng nhau: 7/8 query giống hệt**,
chỉ TRAKE_H01 khác (0.250 vs 0.000). Chênh 1 query trên bộ 8 query là dưới
ngưỡng kết luận được. Beam rộng 50 đã đủ chính xác cho quy mô bài toán này.

> **Tối ưu hàm mục tiêu tốt hơn KHÔNG kéo theo điểm thật tốt hơn.** DP tối ưu
> chính xác cái mà beam chỉ xấp xỉ, và vẫn không hơn. Khi proxy sai, tìm kiếm
> giỏi hơn chỉ giúp bạn tới đích sai nhanh hơn.

### Về phần "n điểm neo" trong đề xuất gốc

`step_hits` **đã** là danh sách ứng viên riêng cho từng bước — mỗi hành động
đã có lượt retrieval riêng của nó. Nên `F_i` vốn tồn tại sẵn, và việc "neo rồi
lan ra" không thêm ứng viên mới nào.

Ý tưởng neo chỉ còn giá trị nếu việc lan ra **có điều kiện** theo điểm neo —
ví dụ chỉ tìm bước *i+1* trong một cửa sổ thời gian sau điểm neo, thay vì tìm
toàn video. Đó là một ý tưởng KHÁC và chưa thử. Nhưng ba kết quả trên cho thấy
khâu ghép chuỗi không phải chỗ đang mất điểm, nên nó không nên là việc tiếp theo.

### Giả thuyết còn lại

Sau khi loại BA giả thuyết trên — ràng buộc hình thức, cửa sổ chấm, và thuật
toán ghép chuỗi — chỉ còn đúng một điều chưa thử: **điểm từng bước đang là
tương đối chứ không tuyệt đối** (mục 3).

Ba giả thuyết bị loại đều nằm ở khâu *tìm kiếm* và *chấm điểm cuối*. Cái còn
lại nằm ở khâu *đánh giá từng ứng viên*. Đó giờ là giả thuyết sống duy nhất.

---

## 5. Cửa sổ chấm — thay đổi phép ĐO, phải tách bạch

Cận dưới 2s hiện tại đang chặn trần ở 0.800. Luật bạn nêu ("chỉ cần thuộc
range của hành động") cho phép nới, nhưng đây là **thay đổi cách đo, không
phải thay đổi hệ thống** — nới cửa sổ làm điểm tăng mà chẳng cần cải tiến gì.

Nên tách làm hai và báo cáo cả hai:

```
--trake-window-min-sec 2   (giữ nguyên)   → so được với mọi số cũ
--trake-window-min-sec 4   (theo luật)    → trần thật sự đạt tới được
```

Và **không nới cửa sổ nếu không có xác minh ngữ nghĩa đi kèm**. Cửa sổ ±4s ở
30fps là 240 frame; trong ngần ấy thời gian cảnh có thể đã đổi hẳn. Cửa sổ
rộng cộng với `s_i` tuyệt đối từ VLM là hợp lệ — vì VLM chặn ở khâu ngữ nghĩa.
Cửa sổ rộng mà chấm bằng điểm retrieval thì chỉ là tự thổi điểm.

---

## 6. Đo thế nào để biết có hiệu quả

Giữ nguyên ba tầng đã tách sẵn, thêm một chẩn đoán:

| Chỉ số | Hiện tại | Ý nghĩa |
|---|---|---|
| `frame_oracle_coverage` | 23/35 | frame đúng có nằm trong tập ứng viên không |
| `frame_selection_accuracy_given_oracle` | 19/23 | có sẵn rồi thì có chọn đúng không |
| `mean_r_score` | 0.263 | chỉ số chính |
| **hit-rate theo VỊ TRÍ bước** | *chưa có* | bước đầu/giữa/cuối, bước nào hỏng |

Chỉ số thứ tư là mới và cần: nếu lỗi dồn vào bước cuối thì vấn đề là lan truyền
từ điểm neo; nếu rải đều thì vấn đề ở `s_i`. Hai nguyên nhân này cần hai cách
sửa khác hẳn nhau, và `mean_r_score` gộp chung không phân biệt được — đúng lỗi
"một metric trộn hai tầng" đã mắc ở CAPTION-ENRICH-01.

Thứ tự thí nghiệm, mỗi lần một biến:

1. **Tắt `gap_penalty_per_sec` và `ordering_weight`**, đổi thứ tự thành cổng
   cứng. Không thêm VLM. Đây là phép thử rẻ nhất, và log đã nghi ngờ sẵn.
2. **Thêm `s_i` từ VLM** cho top-k ứng viên, giữ cách chọn chuỗi cũ. Đo riêng
   phần đóng góp của tín hiệu tuyệt đối.
3. **Thay bằng DP n-neo**. Đo riêng phần đóng góp của cách chọn tổ hợp.
4. Chỉ sau đó mới đụng cửa sổ chấm, và báo cáo song song cả 2s lẫn 4s.

---

## 7. Một câu hỏi phải xác minh với BTC trước khi chốt

`scorer.py:82` giả định **chỉ một chuỗi được chấm**:

> *"TRAKE không có khái niệm 'dòng nào trong top-K đúng', nộp bài chỉ có một
> chuỗi được chấm thật."*

Đây là giả định trong code, **chưa đối chiếu với luật BTC**. Nó quyết định
chiến lược:

- Nếu **chỉ một chuỗi** được chấm → nghiệm DP là bài nộp duy nhất, và toàn bộ
  giá trị nằm ở việc chọn đúng.
- Nếu **nhiều dòng** được chấm và lấy dòng tốt nhất → nộp cả n chuỗi neo cộng
  nghiệm DP, xếp theo `S(C)`. Giới hạn là 100 dòng còn n thường là 4–6, nên
  việc này **miễn phí** và chỉ có lợi.

Khác biệt giữa hai kịch bản đủ lớn để đáng đi hỏi trước khi tối ưu thêm.
