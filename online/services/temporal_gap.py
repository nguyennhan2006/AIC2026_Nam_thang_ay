"""Hàm phạt khoảng cách thời gian dùng chung cho mọi bộ ghép chuỗi.

Tồn tại để `temporal.py` (beam), `temporal_dp.py` (DP) và `trake/sequence_search.py`
không trôi lệch nhau: ba nơi từng có ba công thức phạt khác nhau, nên mọi so sánh
beam-vs-dp trước đây đều so nhầm hai hàm mục tiêu chứ không phải hai cách tìm.

**Hình dạng: dead-zone + tuyến tính + trần.**::

    penalty(gap) = min( lambda * max(0, gap - W),  cap )

`W` (`free_gap_sec`) là vùng miễn phạt. Ràng buộc CỨNG chỉ còn hai điều — cùng
video và thứ tự xuất hiện tăng dần — còn thời gian chỉ là tín hiệu MỀM, và chỉ
lên tiếng khi hai bước cách nhau xa tới mức khó còn là một diễn biến.

`cap` là thứ giữ cho phạt không bao giờ lấn át độ liên quan: dù hai bước cách
nhau 30 giây hay 30 phút, phạt tối đa vẫn là `cap`, nên một chuỗi có step khớp
thật sự tốt hơn không thể bị một chuỗi kém hơn vượt mặt chỉ vì nó gọn thời gian.

**Ba tham số này ĐÃ ĐO, không suy luận.** GAP-RELAX-01, 24 query TRAKE trên
`exports_multivideo` (3 video), `--pipeline container`, `PYTHONHASHSEED=0`::

    W    lambda   cap     video_recall@1   mean_r   frame_sel
    0    0.002    inf     1.000            0.354    0.419     <- ban CU
    60   0.002    inf     1.000            0.355    0.420
    60   0.002    1.0     1.000            0.355    0.420     <- MAC DINH
    60   0.002    0.5     1.000            0.355    0.420
    60   0.002    0.3     0.958            0.347    0.428
    60   0.002    0.2     0.958            0.330    0.407
    60   0.002    0.1     0.958            0.330    0.407
    60   2e-5     0.01    0.958            0.330    0.407
    60   0        --      0.958            0.338    0.417     <- tat han

Hai kết luận, cái thứ hai đi NGƯỢC giả thuyết đã dựng ra module này:

1. **Dead-zone là miễn phí.** `W=0 -> 60` gần như không đổi gì (0.354 -> 0.355).
   Chuỗi đúng có nhịp p50=10s nên vùng miễn phạt 60s chưa từng phải làm việc.
   Giữ nó vì nó đúng về mặt luật — không điều khoản nào thưởng chuỗi gọn về
   thời gian — và vì nó không tốn gì, chứ không phải vì nó ăn điểm.

2. **Trần thấp thì HỎNG.** Bản đầu đặt `cap=0.01` (1/4 điểm một scene) theo lập
   luận "phạt chỉ nên đủ để phá hoà, không được lấn át độ liên quan". Đo ra
   `video_recall@1` rơi 1.000 -> 0.958 và `mean_r` 0.355 -> 0.330, tức TỆ HƠN
   cả việc tắt hẳn phạt. Phạt khoảng cách ở đây không phải cái phá hoà: nó là
   thứ dìm chuỗi trải rộng ở video SAI xuống để video ĐÚNG nổi lên, và để làm
   được việc đó nó cần thẩm quyền cỡ **0.5 — hơn 12 lần điểm của một scene**.
   Ngưỡng gãy nằm giữa 0.3 và 0.5.

`cap=1.0` được chọn vì nó nằm giữa hai điểm đo GIỐNG HỆT nhau (0.5 và không có
trần), tức có biên an toàn mà vẫn chặn được trường hợp bệnh lý ở video rất dài.

Các mặc định dưới đây ở thang điểm RRF thô (~0.04/scene). `trake/sequence_search.py`
chuẩn hoá điểm về ~1.0/step nên phải dùng bộ số RIÊNG, xem `SequenceConfig`.
"""

from __future__ import annotations

#: Vùng miễn phạt, tính bằng giây. Gold max 36s -> 60s phủ trọn với biên.
DEFAULT_FREE_GAP_SEC = 60.0
#: Phạt mỗi giây VƯỢT `free_gap_sec`. GIỮ NGUYÊN giá trị vẫn dùng từ trước: hạ
#: xuống 2e-5 đo ra tệ hơn cả tắt hẳn phạt.
DEFAULT_GAP_PENALTY_PER_SEC = 0.002
#: Trần phạt. Chạm trần ở gap = 60 + 1.0/0.002 = 560s.
DEFAULT_MAX_GAP_PENALTY = 1.0


def gap_penalty_value(
    gap_sec: float,
    *,
    penalty_per_sec: float = DEFAULT_GAP_PENALTY_PER_SEC,
    free_gap_sec: float = DEFAULT_FREE_GAP_SEC,
    max_penalty: float = DEFAULT_MAX_GAP_PENALTY,
) -> float:
    """`min(lambda * max(0, gap - W), cap)`, luôn >= 0.

    `gap_sec` âm (hai bước chồng lấn) coi như 0 — thứ tự đã được ràng buộc cứng
    ở nơi gọi, nên chồng lấn chỉ là hai khoảnh khắc trong cùng một scene.
    """

    excess = gap_sec - free_gap_sec
    if excess <= 0.0 or penalty_per_sec <= 0.0:
        return 0.0
    return min(penalty_per_sec * excess, max_penalty)


def gap_penalty_ceiling_sec(
    *,
    penalty_per_sec: float = DEFAULT_GAP_PENALTY_PER_SEC,
    free_gap_sec: float = DEFAULT_FREE_GAP_SEC,
    max_penalty: float = DEFAULT_MAX_GAP_PENALTY,
) -> float:
    """Gap nhỏ nhất mà phạt đã chạm trần. `inf` khi không có trần hiệu lực.

    `temporal_dp` cần mốc này để biết từ đâu trở đi phạt là HẰNG SỐ, nhờ đó cắt
    được phần lớn không gian tìm kiếm mà vẫn cho nghiệm đúng.
    """

    if penalty_per_sec <= 0.0:
        return free_gap_sec
    if max_penalty <= 0.0:
        return free_gap_sec
    return free_gap_sec + max_penalty / penalty_per_sec


__all__ = [
    "DEFAULT_FREE_GAP_SEC",
    "DEFAULT_GAP_PENALTY_PER_SEC",
    "DEFAULT_MAX_GAP_PENALTY",
    "gap_penalty_ceiling_sec",
    "gap_penalty_value",
]
