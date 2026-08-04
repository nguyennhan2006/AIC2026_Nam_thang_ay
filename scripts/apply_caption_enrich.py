"""Áp caption đã qua gate vào một export MỚI (không sửa file gốc)."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("storage/exports_l21_repaired/scenes.jsonl"))
    p.add_argument("--results", type=Path, default=Path("outputs/evaluation/caption_enrich_results.json"))
    p.add_argument("--out", type=Path, default=Path("storage/exports_l21_enriched"))
    a = p.parse_args()

    accepted = {
        r["scene_id"]: r["caption_new"]
        for r in json.loads(a.results.read_text(encoding="utf-8"))
        if r.get("accepted") and r.get("caption_new")
    }
    src = a.metadata.parent
    a.out.mkdir(parents=True, exist_ok=True)
    for f in list(src.glob("*.jsonl")) + list(src.glob("*.json")):
        if f.name != a.metadata.name:
            shutil.copy2(f, a.out / f.name)

    applied = 0
    lines = []
    for line in a.metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scene = json.loads(line)
        new = accepted.get(scene["scene_id"])
        if new:
            # Caption mới đứng TRƯỚC caption cũ: caption cũ vẫn giữ lại làm
            # bằng chứng và để rollback so sánh được, nhưng phần đặc trưng
            # nhất phải nằm trong cửa sổ token của encoder.
            scene["captions"] = [
                {"caption_type": "visual", "text": new, "language": "vi",
                 "confidence": None, "evidence_keyframe_ids": [],
                 "provenance": {"created_at": "2026-08-04T00:00:00Z", "device": "unknown",
                                "model_name": "Qwen2.5-VL-7B-Instruct:caption-enrich",
                                "model_revision": "caption_event_factual_v1",
                                "parameters": {}, "pipeline_version": "aic-v1.0.0",
                                "prompt_version": "caption_event_factual_v1"}},
                *scene.get("captions", []),
            ]
            applied += 1
        lines.append(json.dumps(scene, ensure_ascii=False))
    (a.out / a.metadata.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"áp {applied}/{len(accepted)} caption mới -> {a.out}")

if __name__ == "__main__":
    main()
