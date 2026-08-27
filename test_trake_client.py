"""Test script for TRAKE partial-chain."""

import asyncio
import json
import httpx

async def test_trake():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Test TRAKE query - partial chain
        response = await client.post(
            "http://127.0.0.1:8001/v1/search/stream",
            json={"query": "con cá", "task": "TRAKE"},
            headers={"Content-Type": "application/json"},
        )

        print(f"Status: {response.status_code}")

        # Parse SSE stream
        content = response.text
        print("\n=== TRAKE Stream Events ===")

        for line in content.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type", "unknown")

                    if event_type == "query_bundle":
                        print(f"\n[QUERY BUNDLE] intent={data.get('intent')}")
                        print(f"  events will be: {data.get('expected_units')}")
                    elif event_type == "query_prepared":
                        print(f"\n[QUERY PREPARED]")
                        print(f"  events: {len(data.get('events', [])) if 'events' in data else 'N/A'}")
                    elif event_type == "alignment_completed":
                        print(f"\n[ALIGNMENT COMPLETED]")
                        print(f"  sequence_count: {data.get('sequence_count', 0)}")
                        print(f"  note: {data.get('note', '')}")
                    elif event_type == "search_completed":
                        print(f"\n[SEARCH COMPLETED]")
                        results = data.get("response", {}).get("results", [])
                        print(f"  Total results: {len(results)}")
                        # Print TRAKE details
                        trake = data.get("response", {}).get("trake", [])
                        print(f"  TRAKE sequences: {len(trake)}")
                        for i, seq in enumerate(trake[:3]):
                            print(f"    [{i+1}] video={seq.get('video_id')}, steps={len(seq.get('steps', []))}")
                            steps = seq.get('steps', [])
                            for step in steps[:5]:
                                print(f"         step={step.get('step')}, frame={step.get('frame_idx')}, conf={step.get('confidence', 0):.2f}")
                    elif event_type == "error":
                        print(f"\n[ERROR] {data.get('message', '')}")
                except json.JSONDecodeError:
                    pass

if __name__ == "__main__":
    asyncio.run(test_trake())
