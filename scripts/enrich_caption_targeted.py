"""CAPTION-ENRICH-01: sinh lại caption cho ĐÚNG những scene bị chẩn đoán sai.

Bối cảnh (docs/20_EXPERIMENT_LOG.md): sau SCENE-COVERAGE-01, nút thắt còn lại
của TRAKE là caption tả không khí chung mà bỏ mất chủ thể của sự kiện::

    event   "xe cứu hỏa bật đèn xanh"
    caption "đám cháy rừng dữ dội với ánh sáng đỏ rực và khói bốc lên cao"
            -> đúng hiện trường, thiếu hẳn chiếc xe

    event   "rùa được thả từ thuyền xuống biển"
    caption "những người đang giúp đỡ một người khác lên tàu trên biển"
            -> mất chủ thể chính

Chỉ chạy trên danh sách mục tiêu (`scripts/list_caption_mismatch.py`), KHÔNG
chạy lại toàn corpus: 11 scene / 19 keyframe thay vì 217 scene.

`event_text` đưa vào prompt CHỈ như gợi ý vùng chú ý. Prompt cấm model xác
nhận sự kiện nếu không nhìn thấy — nếu không, ta chỉ đang dạy model chép lại
câu hỏi, và caption sẽ khớp gold một cách giả tạo.

Caption mới KHÔNG ghi đè caption cũ. Nó phải qua gate rồi mới được chấp nhận;
trượt gate thì giữ nguyên caption cũ.

Chạy::

    python -m scripts.enrich_caption_targeted --env-file .env.fpt.local
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from online.adapters.fpt_client import FptClient, image_to_data_url
from online.adapters.provider_errors import ProviderError

PROMPT_VERSION = "caption_event_factual_v1"

SYSTEM_PROMPT = """Bạn tạo metadata sự kiện cho hệ thống tìm kiếm video.

Xem TẤT CẢ các khung hình được cung cấp — chúng thuộc cùng một cảnh.

Chỉ mô tả những gì NHÌN THẤY ĐƯỢC. Không suy diễn sự kiện không hiện trên hình.

Trả về ĐÚNG MỘT object JSON, không kèm văn bản nào khác:
{
  "global_scene": "một câu mô tả tổng thể",
  "people": [],
  "roles_or_uniforms": [],
  "vehicles": [],
  "animals": [],
  "important_objects": [],
  "actions": [],
  "interactions": [],
  "scene_context": [],
  "text_on_screen": [],
  "uncertain_items": [],
  "languages_detected": []
}

BẮT BUỘC:
- Kiểm tra riêng từng loại: phương tiện, động vật, đồng phục, dụng cụ, và vật
  thể nhỏ ở tiền cảnh.
- Nêu tên chủ thể chính KỂ CẢ khi nó chỉ chiếm một vùng nhỏ trong khung hình.
- KHÔNG thay một vật thể cụ thể bằng mô tả không khí chung chung.
- Bằng chứng không rõ thì đưa vào "uncertain_items", không đoán bừa.
- Viết TIẾNG VIỆT, trừ chữ xuất hiện nguyên văn trong hình.
"""

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")


def build_messages(
    images: list[Path], old_caption: str, focus: str,
    ocr: str = "", asr: str = "",
) -> list[dict]:
    """FPT VLM chỉ nhận ĐÚNG MỘT ảnh mỗi prompt (HTTP 400: "At most 1 image(s)
    may be provided in one prompt"), nên caller phải gọi từng ảnh rồi hợp nhất
    — xem `merge_payloads`. Giữ chữ ký nhận list để chỗ gọi không phải đổi khi
    provider nới ràng buộc."""

    lines = [f"Caption hiện tại (có thể thiếu sót): {old_caption[:300]}"]
    if ocr.strip():
        lines.append(f"CHỮ TRÊN MÀN HÌNH (OCR): {ocr[:400]}")
    if asr.strip():
        lines.append(f"LỜI DẪN TRONG ĐOẠN (ASR): {asr[:600]}")
    lines.append(f"Vùng cần chú ý: {focus}")
    if ocr.strip() or asr.strip():
        lines.append(
            "OCR và lời dẫn cho biết đoạn này nói về chủ đề gì. Hãy dùng chúng "
            "để GỌI ĐÚNG TÊN và MÔ TẢ CHI TIẾT HƠN thứ bạn nhìn thấy — ví dụ "
            "biết đây là phóng sự về rùa biển thì gọi 'rùa biển' thay vì 'con "
            "vật', biết đang nói về chữa cháy thì gọi đúng 'xe cứu hoả', "
            "'lính cứu hoả'. Vẫn phải nhìn thấy mới được liệt kê; thứ chỉ nghe "
            "được mà không thấy thì đưa vào uncertain_items."
        )
    lines.append(
        "Nếu KHÔNG nhìn thấy nội dung ở 'vùng cần chú ý' thì đừng nhắc tới nó — "
        "chỉ mô tả những gì thực sự có trong hình."
    )
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


STRUCTURED_LISTS = (
    "people", "roles_or_uniforms", "vehicles", "animals",
    "important_objects", "actions", "interactions", "scene_context",
    "text_on_screen", "uncertain_items", "languages_detected",
)


def flatten_item(item) -> str:
    """Phần tử danh sách có thể là chuỗi HOẶC object.

    Model thường trả `{"name": "xe cứu thương", "description": "đang tới gần"}`
    dù prompt xin danh sách chuỗi. `str()` thẳng sẽ nhét nguyên dict repr
    (`{'name': ...}`) vào caption — rác cho cả BM25 lẫn embedding. Rút lấy
    phần chữ có nghĩa thay vì siết prompt, vì siết prompt không bảo đảm được.
    """

    if isinstance(item, dict):
        parts = [
            str(item[key]).strip()
            for key in ("name", "label", "text", "object", "value", "description")
            if item.get(key)
        ]
        return " ".join(dict.fromkeys(parts)) if parts else ""
    return str(item).strip()


def merge_payloads(payloads: list[dict]) -> dict:
    """Hợp nhất kết quả của nhiều khung hình trong CÙNG một scene.

    Đây chính là lý do phải xem nhiều frame: chủ thể nhỏ (con rùa, xe cứu hoả,
    chiếc xe máy) thường chỉ xuất hiện ở đúng một keyframe. Lấy hợp của các
    danh sách để không mất chúng; `global_scene` ghép lại theo thứ tự frame.
    """

    merged: dict = {key: [] for key in STRUCTURED_LISTS}
    scenes: list[str] = []
    for payload in payloads:
        text = str(payload.get("global_scene", "")).strip()
        if text and text not in scenes:
            scenes.append(text)
        for key in STRUCTURED_LISTS:
            for item in payload.get(key) or []:
                value = flatten_item(item)
                if value and value not in merged[key]:
                    merged[key].append(value)
    merged["global_scene"] = " ".join(scenes)
    return merged


def to_caption(payload: dict) -> str:
    """Ghép JSON có cấu trúc thành một chuỗi tìm kiếm được (variant C)."""

    parts = [str(payload.get("global_scene", "")).strip()]
    for key in ("people", "roles_or_uniforms", "vehicles", "animals",
                "important_objects", "actions", "interactions", "scene_context"):
        values = [text for text in (flatten_item(item) for item in payload.get(key) or []) if text]
        if values:
            parts.append(", ".join(dict.fromkeys(values)))
    caption = " | ".join(part for part in parts if part)
    # E5 cắt ở 320 token; giữ caption gọn để phần đầu (thông tin đặc trưng
    # nhất) không bị đẩy ra ngoài cửa sổ.
    return caption[:1500]


def validate(payload: dict, caption: str, focus: str) -> tuple[bool, list[str]]:
    """Gate — trượt thì giữ caption cũ."""

    problems: list[str] = []
    if not caption.strip():
        problems.append("caption rỗng")
    if CJK.search(caption):
        problems.append("lẫn ký tự CJK")
    if not str(payload.get("global_scene", "")).strip():
        problems.append("thiếu global_scene")
    concrete = sum(
        len(payload.get(key) or [])
        for key in ("people", "vehicles", "animals", "important_objects", "actions")
    )
    if concrete == 0:
        problems.append("không liệt kê được vật thể/hành động cụ thể nào")
    # Không phạt độ dài ở đây: gộp 4 frame thì dài là chuyện đương nhiên và
    # không làm caption sai. Cắt bớt là việc của `to_caption`.
    return (not problems), problems


def load_env(path: Path) -> None:
    import os

    if not path.exists():
        raise SystemExit(f"thiếu {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh lại caption cho scene bị chẩn đoán sai")
    parser.add_argument("--targets", type=Path,
                        default=Path("outputs/evaluation/caption_enrich_targets.json"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--env-file", type=Path, default=Path(".env.fpt.local"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/evaluation/caption_enrich_results.json"))
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()

    load_env(args.env_file)
    from online.config import Settings

    settings = Settings.from_env()
    if not settings.fpt_enabled or not settings.fpt_vlm_model:
        raise SystemExit("cần AIC_FPT_ENABLED=true và AIC_FPT_VLM_MODEL")
    client = FptClient.from_settings(settings)

    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    results: list[dict] = []
    accepted = 0
    for index, target in enumerate(targets, start=1):
        images = [args.data_root / path for path in target["keyframe_paths"]][: args.max_images]
        images = [path for path in images if path.exists()]
        record = {
            "used_ocr": bool(target.get("ocr_old", "").strip()),
            "used_asr": bool(target.get("asr_old", "").strip()),
            "scene_id": target["scene_id"], "query_id": target["query_id"],
            "event_text": target["event_text"], "caption_old": target["caption_old"],
            "prompt_version": PROMPT_VERSION, "image_count": len(images),
        }
        if not images:
            record.update(accepted=False, rejected_reasons=["không có ảnh"])
            results.append(record)
            continue
        print(f"[{index}/{len(targets)}] {target['scene_id']} ({len(images)} ảnh) …")

        # Một lần gọi cho MỖI ảnh (ràng buộc provider), rồi hợp nhất.
        payloads: list[dict] = []
        failures: list[str] = []
        tokens_in = tokens_out = 0
        for image in images:
            try:
                response = client.chat_completion(
                    build_messages(
                        [image], target["caption_old"], target["event_text"],
                        ocr=target.get("ocr_old", ""), asr=target.get("asr_old", ""),
                    ),
                    model=settings.fpt_vlm_model, temperature=0.0, max_tokens=900,
                )
            except ProviderError as exc:
                failures.append(f"provider: {exc}")
                continue
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens
            parsed = parse_json(response.text)
            if parsed is None:
                failures.append("JSON không hợp lệ")
            else:
                payloads.append(parsed)

        if not payloads:
            record.update(accepted=False, rejected_reasons=failures or ["không có kết quả"])
            results.append(record)
            continue
        payload = merge_payloads(payloads)
        record["frames_used"] = len(payloads)
        if failures:
            record["partial_failures"] = failures
        caption = to_caption(payload)
        ok, problems = validate(payload, caption, target["event_text"])
        record.update(
            structured=payload, caption_new=caption, accepted=ok,
            rejected_reasons=problems,
            usage={"input": tokens_in, "output": tokens_out},
        )
        accepted += ok
        results.append(record)
        print(f"    {'NHẬN' if ok else 'LOẠI ' + str(problems)}: {caption[:110]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    tokens = sum(r.get("usage", {}).get("input", 0) for r in results)
    print(f"\nnhận {accepted}/{len(results)} caption mới  (~{tokens} input token)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
