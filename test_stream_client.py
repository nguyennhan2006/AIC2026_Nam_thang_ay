"""Test script for Query Routing V2."""

import asyncio
import json
import httpx

async def test_query():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Test query
        response = await client.post(
            "http://127.0.0.1:8001/v1/search/stream",
            json={"query": "con cá", "task": "QA"},
            headers={"Content-Type": "application/json"},
        )

        print(f"Status: {response.status_code}")

        # Parse SSE stream
        content = response.text
        print("\n=== Stream Events ===")

        for line in content.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type", "unknown")

                    # Print key events
                    if event_type == "query_bundle":
                        print(f"\n[QUERY BUNDLE] intent={data.get('intent')}, answer_type={data.get('answer_type')}")
                        print(f"  visual: {data.get('visual', '')[:80]}...")
                        print(f"  visual_en: {data.get('visual_en', '')[:80]}...")
                        print(f"  ocr: {data.get('ocr', '')[:80]}...")
                    elif event_type == "query_prepared":
                        print(f"\n[QUERY PREPARED] normalized={data.get('normalized_query', '')[:80]}")
                    elif event_type == "branch_started":
                        print(f"  [BRANCH] {data.get('branch_id')}")
                    elif event_type == "branch_completed":
                        print(f"  [/BRANCH] {data.get('branch_id')} - {data.get('candidate_count', 0)} candidates")
                    elif event_type == "search_completed":
                        print(f"\n[COMPLETED]")
                        results = data.get("response", {}).get("results", [])
                        print(f"  Total results: {len(results)}")
                except json.JSONDecodeError:
                    pass

if __name__ == "__main__":
    asyncio.run(test_query())
