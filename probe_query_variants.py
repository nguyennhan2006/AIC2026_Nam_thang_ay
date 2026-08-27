"""Dò từng cách diễn đạt cho MỘT khoảnh khắc đích, xem cách nào tìm ra nó.

Câu hỏi cần trả lời: bài toán nằm ở CÁCH VIẾT TRUY VẤN hay ở RETRIEVAL?

Biến thể quan trọng nhất là `gold_caption` — đưa thẳng caption của chính
keyframe gold làm truy vấn. Đó là trần trên tuyệt đối: không cách viết nào
mô tả khung hình đó sát hơn chính lời chú thích của nó.

    gold_caption -> rank 1   => cách viết CÓ tác dụng, đáng tinh chỉnh prompt
    gold_caption -> rank xấu => cách viết KHÔNG cứu được, phải sửa fusion/dedup

Lưu ý: nếu server bật Tier 2, LLM sẽ VIẾT LẠI truy vấn dò. Script in ra
`query_bundle` để thấy thứ thật sự được đem đi tìm.

    python probe_query_variants.py            # tất cả nhóm
    python probe_query_variants.py ca_can      # một nhóm
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json

import httpx

SERVER = "http://127.0.0.1:8001"

# Caption thật của keyframe gold, lấy từ storage/exports_competition/scenes.jsonl
GOLD_CAPTION_25995 = (
    "Góc nhìn từ trên xuống, một bàn tay đeo đồng hồ đang đổ chất lỏng vào bát "
    "trắng đặt trên cân điện tử, trong khi một con cá nhỏ màu tối nổi bật trong bát."
)
GOLD_CAPTION_26090 = (
    "Một con cá mập nhỏ có vây và thân màu đen trắng sọc rõ nét đang bơi trong bể "
    "nước trong suốt, góc nhìn từ trên cao."
)

GROUPS = {
    "ca_can": {
        "gold_video": "L21_V023",
        "gold_frame": 25995,
        "variants": [
            ("00_gold_caption", GOLD_CAPTION_25995),
            ("01_query_goc", "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loài bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?"),
            ("02_mo_ta_frame", "Bàn tay đeo đồng hồ đổ chất lỏng vào bát trắng đặt trên cân điện tử"),
            ("03_chi_tu_hiem", "cân điện tử bát trắng đồng hồ đeo tay"),
            ("04_bo_het_chu_ca", "bàn tay đổ chất lỏng vào bát trắng trên cân điện tử"),
            ("05_them_dang_boi", "con cá đang bơi trong bát trắng đặt trên cân điện tử"),
            ("06_sieu_ngan", "cân điện tử"),
        ],
    },
    "ca_map": {
        "gold_video": "L21_V023",
        "gold_frame": 26090,
        "variants": [
            ("00_gold_caption", GOLD_CAPTION_26090),
            ("01_query_goc", "Một con cá mập nhỏ thân sọc đen trắng bơi trong bể nước trong suốt nhìn từ trên cao"),
            ("02_bo_chu_ca", "sinh vật sọc đen trắng bơi trong bể nước trong suốt"),
            ("03_chi_tu_hiem", "cá mập sọc đen trắng bể nước trong suốt"),
        ],
    },
}


def _frames(resp: dict) -> list[tuple[str, int]]:
    out = []
    for item in resp.get("trake") or []:
        for step in item.get("steps") or []:
            out.append((item.get("video_id", ""), step.get("frame_idx", -1)))
    for key in ("kis", "qa", "avs", "results"):
        for item in resp.get(key) or []:
            f = item.get("frame_idx")
            if f is None:
                f = item.get("best_frame_idx", -1)
            out.append((item.get("video_id", ""), f if f is not None else -1))
    return out


async def probe(client, text, gold_video, gold_frame, task="TEXTUAL_KIS"):
    try:
        r = await client.post(
            f"{SERVER}/v1/search/stream",
            json={"query": text, "task": task, "top_k": 20},
            timeout=900.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"err": f"{type(exc).__name__}"}

    sent = final = None
    for line in r.text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "query_bundle":
            sent = ev.get("visual_query", "")
        elif ev.get("type") == "search_completed":
            final = ev.get("response") or {}

    frames = _frames(final or {})
    vids = [v for v, _ in frames]
    rank = next((i + 1 for i, v in enumerate(vids) if v == gold_video), None)
    near = min(
        (abs(f - gold_frame) for v, f in frames if v == gold_video), default=None
    )
    return {
        "rank": rank,
        "near": near,
        "top": vids[0] if vids else None,
        "sent": sent or "",
    }


async def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    for name, g in GROUPS.items():
        if only and name != only:
            continue
        print("=" * 96)
        print(f"NHOM {name}   gold={g['gold_video']} frame={g['gold_frame']}")
        print("=" * 96)
        print(f"{'bien the':<20} {'rank':>6} {'nearest':>9}  {'top video':<14} {'da gui di':<32}")
        print("-" * 96)

        async with httpx.AsyncClient() as client:
            for label, text in g["variants"]:
                res = await probe(client, text, g["gold_video"], g["gold_frame"])
                if "err" in res:
                    print(f"{label:<20} {res['err']}")
                    continue
                rank = res["rank"] or "-"
                near = res["near"] if res["near"] is not None else "-"
                sent = res["sent"][:30] + ("…" if len(res["sent"]) > 30 else "")
                mark = "  <<<" if res["rank"] == 1 else ""
                print(
                    f"{label:<20} {str(rank):>6} {str(near):>9}  "
                    f"{str(res['top']):<14} {sent:<32}{mark}"
                )
        print()


if __name__ == "__main__":
    asyncio.run(main())
