"""FB-003: bản nháp sắp xếp phải DÙNG CHUNG được cả đội và sống qua restart.

Trước đây thứ tự sắp tay + đáp án sửa tay chỉ nằm trong state React của một
tab: F5 là mất, và người ngồi máy bên cạnh không thấy được bản đã soát của
người kia — hai người soát trùng một câu, bỏ trắng câu khác.

Hợp đồng khoá ở đây:

* lưu rồi đọc lại được từ MỘT tiến trình khác (mô phỏng bằng store thứ hai
  trỏ vào cùng file) — đó chính là "của mọi người";
* lưu lại cùng `draft_id` là GHI ĐÈ, không đẻ bản trùng tên;
* một dòng hỏng trong file không được làm mất những bản còn lại.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from online.adapters.draft_store import JsonlDraftStore
from online.api.app import create_app
from online.config import Settings
from online.domain.drafts import DraftRow, DraftSaveRequest


def run(coro):
    return asyncio.run(coro)


def save_request(name: str, **overrides) -> DraftSaveRequest:
    payload = {
        "name": name,
        "author": "nhan",
        "task": "QA",
        "query": "biển báo ghi gì",
        "rows": [
            DraftRow(video_id="L01_V001", frame_idx=120, answer="cấm dừng"),
            DraftRow(video_id="L01_V002", frame_idx=340, answer="cấm dừng"),
        ],
    }
    payload.update(overrides)
    return DraftSaveRequest(**payload)


class DraftStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "drafts.jsonl"

    def test_luu_roi_doc_lai_duoc_tu_tien_trinh_khac(self):
        writer = JsonlDraftStore(self.path)
        saved = run(writer.save(save_request("bản của Nhân")))
        self.assertTrue(saved.draft_id)

        # Store thứ hai = người khác, tab khác, sau khi restart server.
        reader = JsonlDraftStore(self.path)
        drafts = run(reader.list())
        self.assertEqual([draft.name for draft in drafts], ["bản của Nhân"])
        self.assertEqual(drafts[0].rows[1].video_id, "L01_V002")
        self.assertEqual(drafts[0].rows[0].answer, "cấm dừng")

    def test_luu_lai_cung_id_la_ghi_de_chu_khong_de_ban_moi(self):
        store = JsonlDraftStore(self.path)
        first = run(store.save(save_request("nháp 1")))
        second = run(store.save(save_request("nháp 1 đã sửa", draft_id=first.draft_id)))

        self.assertEqual(second.draft_id, first.draft_id)
        self.assertEqual(second.created_at, first.created_at, "created_at phải giữ của bản gốc")
        drafts = run(store.list())
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].name, "nháp 1 đã sửa")

    def test_khong_co_id_thi_la_ban_moi(self):
        store = JsonlDraftStore(self.path)
        run(store.save(save_request("a")))
        run(store.save(save_request("b")))
        self.assertEqual(len(run(store.list())), 2)

    def test_moi_nhat_dung_truoc(self):
        store = JsonlDraftStore(self.path)
        run(store.save(save_request("cũ")))
        run(store.save(save_request("mới")))
        names = [draft.name for draft in run(store.list())]
        self.assertEqual(names[0], "mới")

    def test_xoa(self):
        store = JsonlDraftStore(self.path)
        draft = run(store.save(save_request("bỏ đi")))
        self.assertTrue(run(store.delete(draft.draft_id)))
        self.assertEqual(run(store.list()), [])
        self.assertFalse(run(store.delete(draft.draft_id)), "xoá lần hai phải báo không có")

    def test_mot_dong_hong_khong_lam_mat_cac_ban_con_lai(self):
        store = JsonlDraftStore(self.path)
        run(store.save(save_request("còn sống")))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{ vỡ dở\n")
            handle.write(json.dumps({"draft_id": "x"}) + "\n")  # thiếu field bắt buộc

        drafts = run(store.list())
        self.assertEqual([draft.name for draft in drafts], ["còn sống"])


class DraftRouteTest(unittest.TestCase):
    """Vòng đời đầy đủ qua HTTP — đúng đường mà UI đi."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings = dataclasses.replace(
            Settings.from_env(),
            draft_store_path=Path(self.tmp.name) / "drafts.jsonl",
        )
        self.client = TestClient(create_app(settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def payload(self, name: str, **overrides) -> dict:
        body = {
            "name": name,
            "author": "nhan",
            "task": "TEXTUAL_KIS",
            "query": "cào muối",
            "rows": [{"video_id": "L01_V001", "frame_idx": 150, "frame_ids": [], "answer": None}],
        }
        body.update(overrides)
        return body

    def test_vong_doi_luu_liet_ke_xoa(self):
        self.assertEqual(self.client.get("/v1/submission-drafts").json()["drafts"], [])

        created = self.client.post("/v1/submission-drafts", json=self.payload("thứ tự của Nhân"))
        self.assertEqual(created.status_code, 200, created.text)
        draft_id = created.json()["draft_id"]

        listed = self.client.get("/v1/submission-drafts").json()["drafts"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["author"], "nhan")
        self.assertEqual(listed[0]["rows"][0]["frame_idx"], 150)

        removed = self.client.delete(f"/v1/submission-drafts/{draft_id}")
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(self.client.get("/v1/submission-drafts").json()["drafts"], [])

    def test_ten_rong_bi_tu_choi(self):
        """Nháp không tên thì người khác không tìm lại được — vô dụng."""

        response = self.client.post("/v1/submission-drafts", json=self.payload("   "))
        self.assertEqual(response.status_code, 422)

    def test_xoa_ban_khong_ton_tai_tra_404(self):
        self.assertEqual(self.client.delete("/v1/submission-drafts/khong-co").status_code, 404)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
