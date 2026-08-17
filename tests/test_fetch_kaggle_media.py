"""scripts/fetch_kaggle_media.py — bảng đổi tên keyframe.

Phần đáng test duy nhất của script đó không phải việc tải, mà là phép ánh xạ
`source_keyframe_index` -> `frame_idx`. Sai phép này thì ảnh vẫn về đủ, vẫn mở
được, chỉ là gắn nhầm frame — và không có gì báo lỗi vì `/media` vẫn trả ra một
JPEG hợp lệ.

Kiểm chứng ngoài test này: chạy thật trên 855 ảnh L21_V001..V003 đã có sẵn trên
máy, bảng sinh ra trùng KHỚP TUYỆT ĐỐI tên file thật (307/262/286).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.fetch_kaggle_media import load_index_map

_HEADER = (
    "video_id,keyframe_id,source_keyframe_index,frame_idx,timestamp_sec,pts_time,fps,"
    "scene_id,scene_index,scene_start_frame,scene_end_frame,relative_position_in_scene,"
    "mapping_status,source_file\n"
)


def _row(video_id: str, source_index: int, frame_idx: int) -> str:
    return (
        f"{video_id},{video_id}_F{frame_idx:09d},{source_index},{frame_idx},"
        f"{frame_idx/30:.4f},{frame_idx/30:.4f},30.0,{video_id}_S00000,0,0,100,0.0,matched,x\n"
    )


class KaggleIndexMapTests(unittest.TestCase):
    def _pack(self, directory: Path, rows: str) -> Path:
        target = directory / "canonical"
        target.mkdir(parents=True, exist_ok=True)
        (target / "keyframe_scene_mapping.csv").write_text(_HEADER + rows, encoding="utf-8")
        return directory

    def test_maps_source_index_to_frame_idx(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            pack = self._pack(
                Path(raw),
                _row("L21_V001", 1, 0) + _row("L21_V001", 2, 90) + _row("L21_V001", 3, 261),
            )
            mapping = load_index_map(pack)
            self.assertEqual(mapping["L21_V001"], {1: 0, 2: 90, 3: 261})

    def test_gap_in_source_index_must_not_shift_the_rest(self) -> None:
        """192/873 video có chỉ số nguồn KHÔNG liên tục — ghép theo THỨ TỰ là sai.

        L21_V006 là ca thật: thư mục Kaggle có 257 ảnh, export chỉ dùng 256 vì
        chỉ số 2 bị loại. Ghép theo thứ tự thì ảnh `003.jpg` rơi vào ô của
        `002.jpg` và MỌI ảnh sau đó lệch một nấc — ảnh vẫn hiện, chỉ là sai
        ảnh, nên không cách nào phát hiện từ giao diện.
        """

        with tempfile.TemporaryDirectory() as raw:
            pack = self._pack(
                Path(raw),
                _row("L21_V006", 1, 0) + _row("L21_V006", 3, 120) + _row("L21_V006", 4, 330),
            )
            mapping = load_index_map(pack)["L21_V006"]

            self.assertNotIn(2, mapping)
            self.assertEqual(mapping[3], 120)

            # Kaggle phát hành LIÊN TỤC 1..max, kể cả chỉ số mà pack đã loại.
            tren_kaggle = list(range(1, max(mapping) + 1))
            self.assertEqual(tren_kaggle, [1, 2, 3, 4])

            # Ghép theo thứ tự: file thứ i <- frame thứ i. Lệch từ chỗ khuyết trở đi.
            theo_thu_tu = dict(zip(tren_kaggle, sorted(mapping.values())))
            self.assertEqual(theo_thu_tu[3], 330)  # sai
            self.assertEqual(mapping[3], 120)  # đúng
            self.assertNotEqual(theo_thu_tu[3], mapping[3])

    def test_missing_mapping_file_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(SystemExit):
                load_index_map(Path(raw))


if __name__ == "__main__":
    unittest.main()
