"""Test script for fish scale TRAKE query."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import json
import httpx

async def test_fish_scale():
    query = "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loài bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?"

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            "http://127.0.0.1:8001/v1/search/stream",
            json={"query": query, "task": "TRAKE"},
            headers={"Content-Type": "application/json"},
        )

        print(f"Status: {response.status_code}")
        print(f"\n{'='*60}")
        print(f"QUERY: {query[:80]}...")
        print(f"{'='*60}\n")

        content = response.text
        found_l21 = False
        results_data = None

        for line in content.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type", "unknown")

                    if event_type == "query_bundle":
                        print(f"[QUERY BUNDLE] intent={data.get('intent')}")
                        print(f"  visual: {data.get('visual_query', '')[:100]}...")
                        print(f"  visual_en: {data.get('visual_query_en', '')[:100]}...")
                        print()

                    elif event_type == "alignment_completed":
                        print(f"[ALIGNMENT COMPLETED]")
                        print(f"  sequence_count: {data.get('sequence_count', 0)}")
                        print()

                    elif event_type == "search_completed":
                        print(f"[SEARCH COMPLETED]")
                        trake = data.get("response", {}).get("trake", [])
                        results = data.get("response", {}).get("results", [])

                        print(f"  TRAKE sequences: {len(trake)}")
                        print(f"  Total results: {len(results)}")
                        print()

                        # Check for L21_V023
                        for i, seq in enumerate(trake[:10]):
                            video_id = seq.get('video_id', '')
                            if 'L21' in video_id or 'L22' in video_id:
                                found_l21 = True

                            print(f"  [{i+1}] Video: {video_id}")
                            print(f"       Steps: {len(seq.get('steps', []))}")

                            for step in seq.get('steps', [])[:5]:
                                print(f"         Step {step.get('step')}: frame={step.get('frame_idx')}, conf={step.get('confidence', 0):.2f}")

                        results_data = data.get("response", {})

                except json.JSONDecodeError:
                    pass

        print(f"\n{'='*60}")
        if found_l21:
            print("✅ FOUND L21_V023 or L22 in results!")
        else:
            print("❌ L21/L22 NOT found in results")
        print(f"{'='*60}")

        return found_l21

if __name__ == "__main__":
    asyncio.run(test_fish_scale())
