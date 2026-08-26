"""Chẩn đoán lỗi nạp jina-clip-v2 — in ĐÚNG dữ kiện quyết định, không đoán.

    python -m scripts.diag_jina_load                       # đọc AIC_CAPTION_DENSE_MODEL
    python -m scripts.diag_jina_load --model storage/models/jina-clip-v2

Chạy trong CÙNG shell (cùng biến môi trường) với lệnh uvicorn đang lỗi — cờ
offline và HF_HOME là một phần của bệnh, đổi shell là mất chứng cứ.

## Vì sao cần script riêng thay vì đọc traceback

`AutoModel.from_pretrained` có hai đường thất bại trông rất khác nhau:

  A. OSError "We couldn't connect to huggingface.co"
     -> `auto_map` trỏ repo code KHÁC (dấu `--`), thiếu $HF_HOME/hub/models--jinaai--*
     -> vá bằng `python -m scripts.prepare_jina_offline`

  B. ValueError "Unrecognized configuration class ... Model type should be one of ..."
     -> config tới tay AutoModel KHÔNG mang `auto_map["AutoModel"]`, nên transformers
        bỏ qua hẳn đường remote code và tra bảng kiến trúc chuẩn — `jina_clip` không
        có trong đó nên nó liệt kê toàn bộ Config class nó biết.
     -> nguyên nhân là config.json sai/thiếu, KHÔNG phải thiếu mạng.

Hai bệnh, hai thuốc. Script này phân biệt bằng cách kiểm từng tầng theo thứ tự.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_REPOS = ("jinaai--jina-clip-implementation", "jinaai--xlm-roberta-flash-implementation")


def log(message: str) -> None:
    print(message, flush=True)


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None,
                        help="đường dẫn/repo id. Mặc định lấy AIC_CAPTION_DENSE_MODEL")
    arguments = parser.parse_args()

    model = arguments.model or os.environ.get("AIC_CAPTION_DENSE_MODEL") \
        or "storage/models/jina-clip-v2"

    log("=== 1. môi trường ===")
    log(f"  cwd                     {Path.cwd()}")
    log(f"  model                   {model}")
    log(f"  AIC_CAPTION_DENSE_ENCODER {os.environ.get('AIC_CAPTION_DENSE_ENCODER', '(chưa đặt)')}")
    log(f"  HF_HOME                 {hf_home()}")
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        log(f"  {flag:<23} {os.environ.get(flag, '(tắt)')}")
    try:
        import transformers, torch  # noqa: E401
        log(f"  transformers            {transformers.__version__}")
        log(f"  torch                   {torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        log(f"  [LỖI] không import được transformers/torch: {exc}")
        return 1
    for module in ("einops", "timm", "torchvision"):
        try:
            __import__(module)
            log(f"  {module:<23} ok")
        except Exception:  # noqa: BLE001
            log(f"  {module:<23} THIẾU  <- code remote của jina import cái này")

    log("\n=== 2. model nằm ở đâu ===")
    path = Path(model)
    is_local = path.exists()
    log(f"  là thư mục local?       {is_local}")
    if not is_local:
        log("  -> đây là REPO ID trên HuggingFace. Với HF_HUB_OFFLINE=1 nó chỉ chạy")
        log("     nếu $HF_HOME/hub đã có sẵn model. Máy mới thuê thì KHÔNG có.")

    log("\n=== 3. cache code repo (đường hỏng thường gặp nhất) ===")
    hub = hf_home() / "hub"
    missing_repo = False
    for repo in CODE_REPOS:
        present = (hub / f"models--{repo}").exists()
        log(f"  {'ok    ' if present else 'THIẾU '} {hub / ('models--' + repo)}")
        missing_repo |= not present
    if missing_repo:
        log("  -> auto_map dùng dấu `--` = code mô hình ở repo KHÁC. Thiếu thư mục")
        log("     trên là chết dù trọng số đã đủ trên đĩa.")
        log("     Vá: python -m scripts.prepare_jina_offline   (cần mạng, ~350 KB)")

    log("\n=== 4. config.json ===")
    config_json = path / "config.json" if is_local else None
    auto_map: dict[str, str] = {}
    if config_json and config_json.exists():
        raw = json.loads(config_json.read_text(encoding="utf-8"))
        auto_map = raw.get("auto_map") or {}
        log(f"  model_type              {raw.get('model_type')}")
        log(f"  architectures           {raw.get('architectures')}")
        log(f"  auto_map keys           {sorted(auto_map) or 'KHÔNG CÓ'}")
        text_cfg = (raw.get("text_config") or {}).get("hf_model_name_or_path")
        log(f"  text_config.hf_model_..  {text_cfg}")
        if text_cfg and str(text_cfg).startswith("jinaai/"):
            log("  -> text tower vẫn trỏ HuggingFace. Offline sẽ chết ở tầng này.")
            log("     Vá: python -m scripts.prepare_jina_offline (gọi patch_jina_config)")
        if "AutoModel" not in auto_map:
            log("  -> THIẾU auto_map['AutoModel']. Đây CHÍNH LÀ bệnh B: transformers")
            log("     bỏ qua remote code và liệt kê mọi Config class nó biết.")
            log("     Vá: khôi phục config.json.orig hoặc tải lại repo model.")
    elif is_local:
        log(f"  [LỖI] không có {config_json}")
        return 1
    else:
        log("  (bỏ qua — không phải thư mục local)")

    log("\n=== 5. AutoConfig (nhẹ, không đọc 1,7 GB trọng số) ===")
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        log(f"  THẤT BẠI {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        log("  -> chưa qua được tầng config thì AutoModel không bao giờ chạy. Sửa tầng này trước.")
        return 1
    log(f"  ok — class {type(config).__module__}.{type(config).__name__}")
    resolved = getattr(config, "auto_map", {}) or {}
    log(f"  auto_map trên config    {sorted(resolved) or 'KHÔNG CÓ'}")
    if "AutoModel" not in resolved:
        log("  -> AutoModel.from_pretrained SẼ ném 'Unrecognized configuration class'.")
        log("     has_remote_code=False vì đúng khoá này vắng mặt.")

    log("")
    log("=== 5b. TEXT TOWER — tầng hay vỡ nhất ===")
    log("  modeling_clip.py dựng text tower với pretrained=False, nên hf_model.py chạy:")
    log("      AutoConfig.from_pretrained(<hf_model_name_or_path>, trust_remote_code=True)")
    log("      AutoModel.from_config(config, trust_remote_code=True, add_pooling_layer=False)")
    log("  `from_config` CHÍNH LÀ hàm ném 'Unrecognized configuration class ...'.")
    log("  Nó chỉ đi đường remote code khi config mang auto_map['AutoModel'].")
    text_ref = None
    if config_json and config_json.exists():
        raw2 = json.loads(config_json.read_text(encoding="utf-8"))
        text_ref = (raw2.get("text_config") or {}).get("hf_model_name_or_path")
    if not text_ref:
        log("  (không đọc được text_config — bỏ qua)")
    else:
        log(f"  hf_model_name_or_path   {text_ref}")
        text_path = Path(text_ref)
        if not text_path.is_absolute():
            # Đường dẫn TƯƠNG ĐỐI: transformers resolve theo CWD của tiến trình.
            # `prepare_jina_offline.verify()` chạy subprocess với cwd=root nên
            # LUÔN xanh; uvicorn khởi động từ thư mục khác là vỡ ngay ở đây.
            log(f"  -> TƯƠNG ĐỐI, resolve theo CWD hiện tại = {Path.cwd()}")
            log(f"  -> thành {Path.cwd() / text_path}")
        exists = text_path.exists()
        log(f"  tồn tại?                {exists}")
        if not exists:
            log("  -> KHÔNG tồn tại theo CWD này. transformers sẽ coi chuỗi là REPO ID")
            log("     trên HuggingFace; với HF_HUB_OFFLINE=1 là chết.")
            log("     Vá: chạy uvicorn với CWD = gốc repo, hoặc sửa config.json thành")
            log("     đường dẫn TUYỆT ĐỐI.")
        else:
            tcfg = json.loads((text_path / "config.json").read_text(encoding="utf-8"))
            tmap = tcfg.get("auto_map") or {}
            log(f"  model_type              {tcfg.get('model_type')}")
            log(f"  auto_map keys           {sorted(tmap) or 'KHÔNG CÓ'}")
            if "AutoModel" not in tmap:
                log("  -> ĐÂY LÀ NGUYÊN NHÂN. Thiếu auto_map['AutoModel'] nên")
                log("     AutoModel.from_config bỏ qua remote code và liệt kê mọi Config.")
                log("     Vá: tải lại storage/models/jina-embeddings-v3 (config.json hỏng).")

        log("  thử lại ĐÚNG hai lời gọi của jina:")
        from transformers import AutoModel as _AM
        try:
            tconf = AutoConfig.from_pretrained(text_ref, trust_remote_code=True)
            log(f"    AutoConfig ok -> {type(tconf).__name__}")
            log(f"    auto_map trên đó -> {sorted(getattr(tconf, 'auto_map', {}) or {}) or 'KHÔNG CÓ'}")
            _AM.from_config(tconf, trust_remote_code=True, add_pooling_layer=False)
            log("    AutoModel.from_config ok")
        except Exception as exc:  # noqa: BLE001
            log(f"    THẤT BẠI {type(exc).__name__}: {str(exc).splitlines()[0][:220]}")
            log("    -> tái hiện được lỗi của backend NGAY TẠI ĐÂY.")
            return 1

    log("\n=== 6. AutoModel (đọc trọng số — chậm) ===")
    from transformers import AutoModel
    try:
        model_object = AutoModel.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        log(f"  THẤT BẠI {type(exc).__name__}")
        for line in str(exc).splitlines()[:3]:
            log(f"    {line[:200]}")
        return 1
    total = sum(p.numel() for p in model_object.parameters())
    log(f"  ok — {total / 1e6:.0f}M params, class {type(model_object).__name__}")
    log(f"  có encode_text?         {hasattr(model_object, 'encode_text')}")
    log(f"  có encode_image?        {hasattr(model_object, 'encode_image')}")

    log("\nKẾT LUẬN: nạp được. Nếu backend vẫn lỗi thì khác biệt nằm ở biến môi")
    log("trường của tiến trình uvicorn, không nằm ở model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
