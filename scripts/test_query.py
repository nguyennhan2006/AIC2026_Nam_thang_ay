import httpx, json

payload = {
    "query": "Find the scene with a yellow-red warning sign beside a collapsed riverbank",
    "task": "KIS",
    "top_k": 5
}

try:
    resp = httpx.post("http://localhost:8000/v1/search/kis", json=payload, timeout=30.0)
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Error:", resp.text[:500])
    else:
        result = resp.json()
        print("Results:", len(result.get("results", [])))
        for r in result.get("results", [])[:3]:
            vid = r.get("video_id", "")
            frame = r.get("frame_idx", 0)
            score = r.get("score", 0)
            print(f"  - {vid}:{frame} score={score:.3f}")
except Exception as e:
    print("Error:", type(e).__name__, str(e))
