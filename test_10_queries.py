"""Chấm 10 truy vấn trên server local (cổng 8001, dataset thi đấu).

Gold cho truy vấn cá/cân: L21_V023 frame 25995 (caption của chính keyframe đó:
"một bàn tay đeo đồng hồ đang đổ chất lỏng vào bát trắng đặt trên cân điện tử,
trong khi một con cá nhỏ màu tối nổi bật trong bát").

Chấm hai mức, vì chúng cần hai cách sửa khác nhau:
  - video_hit: video đúng có nằm trong top-K không  (recall của Stage A)
  - frame_hit: có frame nào rơi trong ±tolerance của gold không (độ chính xác)
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
from pathlib import Path

import httpx

SERVER = "http://127.0.0.1:8001"
FRAME_TOLERANCE = 250  # ~8s ở 30fps; scene-level là đủ để nói "đúng khoảnh khắc"

QUERIES = [
    {
        "id": "01_ca_can_goc",
        "task": "QA",
        "query": "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loài bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?",
        "gold_video": "L21_V023",
        "gold_frame": 25995,
    },
    {
        "id": "02_ca_can_ngan",
        "task": "QA",
        "query": "Con cá được đặt trên cân điện tử, số hiển thị trên cân là bao nhiêu?",
        "gold_video": "L21_V023",
        "gold_frame": 25995,
    },
    {
        "id": "03_ca_can_trake",
        "task": "TRAKE",
        "query": "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loài bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?",
        "gold_video": "L21_V023",
        "gold_frame": 25995,
    },
    {
        "id": "04_ban_tay_do_chat_long",
        "task": "TEXTUAL_KIS",
        "query": "Bàn tay đeo đồng hồ đổ chất lỏng vào bát trắng đặt trên cân điện tử",
        "gold_video": "L21_V023",
        "gold_frame": 25995,
    },
    {
        "id": "05_ca_map_be_nuoc",
        "task": "TEXTUAL_KIS",
        "query": "Một con cá mập nhỏ thân sọc đen trắng bơi trong bể nước trong suốt nhìn từ trên cao",
        "gold_video": "L21_V023",
        "gold_frame": 26090,
    },
    {
        "id": "06_dan_chuong_trinh",
        "task": "TEXTUAL_KIS",
        "query": "Hai người dẫn chương trình đứng sau bàn trong studio truyền hình, phía sau là phông nền thành phố lúc hoàng hôn",
        "gold_video": "L21_V023",
        "gold_frame": 120,
    },
    {
        "id": "07_ocr_toa_an_paris",
        "task": "TEXTUAL_KIS",
        "query": 'Cảnh có dòng chữ "TÒA ÁN PHÚC THẨM PARIS BÁC BỎ ĐƠN KIỆN CỦA BÀ TRẦN TỐ NGA"',
        "gold_video": "L21_V023",
        "gold_frame": 716,
    },
    {
        "id": "08_asr_60_giay",
        "task": "QA",
        "query": "Người dẫn chương trình nói chương trình 60 giây buổi sáng phát sóng vào ngày nào?",
        "gold_video": "L21_V023",
        "gold_frame": 120,
    },
    {
        "id": "09_phong_xu_an_ve",
        "task": "TEXTUAL_KIS",
        "query": "Bức vẽ đen trắng mô tả phòng xử án với nhiều người ngồi ở các hàng ghế",
        "gold_video": "L21_V023",
        "gold_frame": 675,
    },
    {
        "id": "10_ran_soc_be_nhua",
        "task": "TEXTUAL_KIS",
        "query": "Một con rắn vằn sọc đen trắng nằm trong bể nhựa, phần đuôi uốn cong",
        "gold_video": "L21_V023",
        "gold_frame": 26244,
    },
]


def _frames_of(response: dict) -> list[tuple[str, int]]:
    """(video_id, frame_idx) theo thứ hạng, gộp mọi khối kết quả của mọi task."""

    found: list[tuple[str, int]] = []

    for item in response.get("trake") or []:
        video = item.get("video_id", "")
        for step in item.get("steps") or []:
            found.append((video, step.get("frame_idx", -1)))

    for key in ("kis", "qa", "avs", "results"):
        for item in response.get(key) or []:
            video = item.get("video_id", "")
            frame = (
                item.get("frame_idx")
                if item.get("frame_idx") is not None
                else item.get("best_frame_idx", -1)
            )
            found.append((video, frame if frame is not None else -1))

    return found


async def run_one(client: httpx.AsyncClient, case: dict) -> dict:
    payload = {"query": case["query"], "task": case["task"], "top_k": 20}
    try:
        # 900s: server cho MỖI NHÁNH tới AIC_BRANCH_TIMEOUT_MS=800000 (800s),
        # nên client đặt thấp hơn thế là tự bỏ cuộc trước server và ghi nhận
        # "lỗi" cho truy vấn thật ra vẫn chạy. Đã xảy ra: 3/10 query báo
        # ReadTimeout ở client trong khi server log trả 200 cho cả 10.
        response = await client.post(
            f"{SERVER}/v1/search/stream", json=payload, timeout=900.0
        )
    except Exception as exc:  # noqa: BLE001 - báo lỗi mạng như một kết quả
        return {"id": case["id"], "error": f"{type(exc).__name__}: {exc}"}

    bundle: dict = {}
    final: dict = {}
    for line in response.text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if event.get("type") == "query_bundle":
            bundle = event
        elif event.get("type") == "search_completed":
            final = event.get("response") or {}

    frames = _frames_of(final)
    videos = [video for video, _ in frames]

    video_rank = next(
        (i + 1 for i, video in enumerate(videos) if video == case["gold_video"]), None
    )
    frame_hit = any(
        video == case["gold_video"] and abs(frame - case["gold_frame"]) <= FRAME_TOLERANCE
        for video, frame in frames
    )
    nearest = min(
        (abs(frame - case["gold_frame"]) for video, frame in frames if video == case["gold_video"]),
        default=None,
    )

    return {
        "id": case["id"],
        "task": case["task"],
        "visual": bundle.get("visual_query", ""),
        "ocr": bundle.get("ocr_query", ""),
        "intent": bundle.get("intent", ""),
        "n_results": len(frames),
        "top_video": videos[0] if videos else None,
        "video_rank": video_rank,
        "frame_hit": frame_hit,
        "nearest_frame_delta": nearest,
    }


async def main() -> None:
    # Nhãn cho lần chạy này (vd: python test_10_queries.py tier2). Kết quả được
    # ghi ra JSON để so hai lần chạy — nếu không thì chạy baseline sẽ xoá mất
    # số của Tier 2 và phải đo lại từ đầu (mỗi lần ~30 phút).
    label = sys.argv[1] if len(sys.argv) > 1 else "run"

    print("=" * 78)
    print(f"10 QUERIES vs {SERVER}   [{label}]")
    print(f"gold video L21_V023, tolerance ±{FRAME_TOLERANCE}f")
    print("=" * 78)

    results = []
    async with httpx.AsyncClient() as client:
        for index, case in enumerate(QUERIES, 1):
            print(f"\n[{index:02d}/10] {case['id']}  ({case['task']})")
            print(f"   query : {case['query'][:72]}")
            outcome = await run_one(client, case)
            results.append(outcome)

            if "error" in outcome:
                print(f"   ERROR : {outcome['error']}")
                continue

            print(f"   visual: {outcome['visual'][:72]}")
            if outcome["ocr"]:
                print(f"   ocr   : {outcome['ocr'][:72]}")
            print(f"   intent: {outcome['intent']}   results={outcome['n_results']}")
            rank = outcome["video_rank"]
            print(f"   video : rank={rank if rank else 'MISS'}   top={outcome['top_video']}")
            delta = outcome["nearest_frame_delta"]
            print(
                f"   frame : {'HIT' if outcome['frame_hit'] else 'miss'}"
                + (f"   nearest Δ={delta}f" if delta is not None else "")
            )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    video_hits = sum(1 for r in results if r.get("video_rank"))
    frame_hits = sum(1 for r in results if r.get("frame_hit"))
    for r in results:
        if "error" in r:
            print(f"  ERR  {r['id']}")
            continue
        rank = r["video_rank"]
        mark = "HIT " if rank else "miss"
        frame = "F-HIT" if r["frame_hit"] else "     "
        print(f"  {mark} {frame}  {r['id']:<26} rank={rank if rank else '-':<4} top={r['top_video']}")
    print(f"\nvideo recall@20 : {video_hits}/{len(QUERIES)}")
    print(f"frame hit@20    : {frame_hits}/{len(QUERIES)}")

    out_path = Path(f"eval_10q_{label}.json")
    out_path.write_text(
        json.dumps(
            {
                "label": label,
                "video_recall_at_20": f"{video_hits}/{len(QUERIES)}",
                "frame_hit_at_20": f"{frame_hits}/{len(QUERIES)}",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nda luu: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
