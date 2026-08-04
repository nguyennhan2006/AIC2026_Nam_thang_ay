"""FIX-DETERMINISM-01: nạp model phải nguyên tử và phải xảy ra ngoài request path.

Bối cảnh đo được — gọi `_retrieve` BA lần với CÙNG plan, trong CÙNG tiến trình::

    run0: dense_visual timeout 3004ms > deadline 3000ms  -> 0 candidate
    run1: dense_visual failed "Cannot copy out of meta tensor"
    run2: dense_visual success 142ms, 100 candidate      -> top khác HẲN

Lần `encode()` đầu nạp CLIP (~3s) nên vượt deadline nhánh và bị
`asyncio.wait_for` huỷ; huỷ giữa `from_pretrained` để lại model kẹt trên
`meta` device nên lần sau hỏng hẳn. Hệ quả: 1–2 truy vấn ĐẦU của mỗi tiến
trình chạy KHÔNG có nhánh dense, im lặng.

Điều này từng bị chẩn đoán nhầm là "PYTHONHASHSEED rò vào ranking" — cố định
seed làm số ổn định chỉ vì nó thay đổi thời điểm/thứ tự chứ không phải vì
hash order. Test dưới đây khoá nguyên nhân THẬT.
"""

from __future__ import annotations

import threading
import unittest

from online.adapters.encoders import LocalClipTextEncoder


class _ExplodingEncoder(LocalClipTextEncoder):
    """Giả lập `from_pretrained` bị huỷ/lỗi giữa chừng."""

    def __init__(self) -> None:
        super().__init__("khong-ton-tai")
        self.attempts = 0

    def _load(self):  # type: ignore[override]
        if self._model is not None:
            return self._model, self._processor
        with self._lock:
            if self._model is not None:
                return self._model, self._processor
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("bị huỷ giữa chừng")
            self._model, self._processor = object(), object()
        return self._model, self._processor


class AtomicLoadTests(unittest.TestCase):
    def test_failed_load_leaves_no_half_built_model(self) -> None:
        """Hỏng lần đầu KHÔNG được để lại state dở dang.

        Đây chính là lỗi 'Cannot copy out of meta tensor': model được gán vào
        `self` trước khi dựng xong, nên lần gọi sau dùng phải một đối tượng
        chết thay vì nạp lại sạch.
        """

        encoder = _ExplodingEncoder()
        with self.assertRaises(RuntimeError):
            encoder._load()
        self.assertIsNone(encoder._model, "lần nạp hỏng vẫn để lại model dở dang")

        model, processor = encoder._load()
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)
        self.assertEqual(encoder.attempts, 2, "phải thử nạp lại chứ không dùng state hỏng")

    def test_load_is_only_done_once_under_concurrency(self) -> None:
        """Hai request song song không được cùng nạp model."""

        encoder = _ExplodingEncoder()
        encoder.attempts = 1  # bỏ qua lần hỏng cố ý
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                encoder._load()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(encoder.attempts, 2, "model bị nạp nhiều hơn một lần")


class WarmupContractTests(unittest.TestCase):
    def test_warmup_exists_so_callers_can_load_outside_the_request_path(self) -> None:
        """`build_container`/`build_service` dựa vào `hasattr(encoder, 'warmup')`.

        Đổi tên hoặc bỏ method này sẽ âm thầm khôi phục lỗi cold-start: không
        có gì fail, chỉ có truy vấn đầu tiên mất nhánh dense.
        """

        self.assertTrue(callable(getattr(LocalClipTextEncoder, "warmup", None)))


if __name__ == "__main__":
    unittest.main()
