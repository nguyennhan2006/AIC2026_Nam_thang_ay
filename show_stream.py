"""Xem một truy vấn đi qua hệ: mỗi engine nhận query gì, trả về bao nhiêu.

    python show_stream.py "truy van cua ban"
    python show_stream.py "..." TRAKE          # doi task
    python show_stream.py "..." QA L21_V023    # danh dau video gold trong ket qua

Trả lời ba câu mà log cũ không trả lời được:

  1. Mỗi nhánh THẬT SỰ đem chuỗi nào đi tìm (`query_sent`), lấy từ trường nào?
  2. Bundle do rule dựng hay LLM viết lại — và LLM đã ghi đè trường nào?
  3. TRAKE tách ra những step nào? (chỗ hỏng thường gặp nhất của TRAKE)
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json

import httpx

SERVER = "http://127.0.0.1:8001"
W = 96


def line(ch="-"):
    print(ch * W)


async def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    query = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else "TEXTUAL_KIS"
    gold = sys.argv[3] if len(sys.argv) > 3 else None

    line("=")
    print(f"QUERY : {query[:W - 8]}")
    print(f"TASK  : {task}" + (f"   |   gold: {gold}" if gold else ""))
    line("=")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SERVER}/v1/search/stream",
            json={"query": query, "task": task, "top_k": 20},
            timeout=900.0,
        )
        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")

        branches = {}
        final = None

        for raw in r.text.split("\n"):
            if not raw.startswith("data: "):
                continue
            try:
                ev = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")

            if kind == "query_bundle":
                src = ev.get("source", "?")
                print(f"\nBUNDLE  (nguon: {src.upper()})")
                if ev.get("llm"):
                    print(f"  prompt {ev['llm'].get('prompt')}  ghi de: {ev['llm'].get('fields')}")
                print(f"  intent={ev.get('intent')}  answer={ev.get('answer_type')}  "
                      f"complexity={ev.get('complexity')}")
                for label, key in (
                    ("visual   ", "visual_query"),
                    ("visual_en", "visual_query_en"),
                    ("caption  ", "caption_query"),
                    ("ocr      ", "ocr_query"),
                    ("asr      ", "asr_query"),
                ):
                    print(f"  {label}: {(ev.get(key) or '(rong)')[:W - 14]}")
                for i, e in enumerate(ev.get("events") or []):
                    print(f"  event[{i}] : {e[:W - 14]}")

            elif kind == "trake_step":
                if ev["step"] == 1:
                    print(f"\nTRAKE STEPS ({ev['total_steps']})")
                print(f"  [{ev['step']}/{ev['total_steps']}] {ev['text'][:W - 12]}")

            elif kind == "branch_started":
                bid = ev.get("branch_id", "?")
                branches[ev.get("execution_id", bid)] = {
                    "sent": ev.get("query_sent", ""),
                    "src": ev.get("query_source", "?"),
                }

            elif kind in ("branch_completed", "branch_failed"):
                b = branches.setdefault(ev.get("execution_id", "?"), {})
                b["n"] = ev.get("candidate_count", 0)
                b["ms"] = ev.get("latency_ms", 0)
                b["state"] = ev.get("state", "")

            elif kind == "search_completed":
                final = ev.get("response") or {}

        print("\nTUNG NHANH NHAN GI")
        line()
        print(f"{'nhanh':<26}{'n':>6}{'ms':>7}  {'tu truong':<16} query")
        line()
        for eid in sorted(branches):
            b = branches[eid]
            flag = "" if b.get("state") == "success" else f"  [{b.get('state', '?')}]"
            print(
                f"{eid:<26}{b.get('n', 0):>6}{int(b.get('ms', 0)):>7}  "
                f"{b.get('src', '?'):<16} {b.get('sent', '')[:34]}{flag}"
            )

        if final:
            print("\nKET QUA")
            line()
            rows = []
            for item in final.get("trake") or []:
                for st in item.get("steps") or []:
                    rows.append((item.get("video_id"), st.get("frame_idx")))
            for key in ("kis", "qa", "avs", "results"):
                for item in final.get(key) or []:
                    f = item.get("frame_idx")
                    rows.append((item.get("video_id"),
                                 f if f is not None else item.get("best_frame_idx")))
            for i, (v, f) in enumerate(rows[:12], 1):
                mark = "  <<<== GOLD" if gold and v == gold else ""
                print(f"  {i:>2}. {v}  f{f}{mark}")
            if gold:
                rank = next((i for i, (v, _) in enumerate(rows, 1) if v == gold), None)
                print(f"\n  rank cua {gold}: {rank if rank else 'KHONG CO trong top'}")


if __name__ == "__main__":
    asyncio.run(main())
