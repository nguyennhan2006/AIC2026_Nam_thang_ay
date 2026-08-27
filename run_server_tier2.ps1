# Server Tier 2: LLM soạn dữ liệu riêng cho từng search engine.
#
# Khác baseline (.env.fpt.local) đúng MỘT biến có ảnh hưởng hành vi:
#     AIC_ENABLE_LLM_QUERY_BUNDLE=true
# nên chênh lệch điểm giữa hai lần chạy quy được về đúng thay đổi đó.
#
# Baseline thuần rule:  .\run_server_full.ps1
# Tier 2 (bản này)   :  .\run_server_tier2.ps1

# HF offline PHẢI đặt ở ĐÂY, không phải trong file .env.
#
# `huggingface_hub` đọc HF_HUB_OFFLINE MỘT LẦN lúc import module rồi đóng băng
# thành hằng số. `Settings.from_env()` nạp file .env SAU khi transformers đã
# import xong, nên biến đặt trong .env không còn tác dụng — đã kiểm chứng:
# đặt trong .env thì `hub.constants.HF_HUB_OFFLINE` vẫn là False.
#
# Không có nó, transformers gọi HF Hub check revision cho model CLIP local;
# máy này chặn SSL tới HF nên lệnh đó TREO (boot đứng ở phase "encoder" 15+
# phút, RAM 4GB, CPU = 0). Có nó thì model nạp trong 0.5 giây.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# stdout của Windows mặc định là cp1258 (Vietnamese ANSI), KHÔNG encode nổi
# nhiều ký tự tiếng Việt tổ hợp — vd 'ẫ' trong "vẫn". Container in cảnh báo
# bằng tiếng Việt (container.py:501, cảnh báo coverage caption_dense), nên
# thiếu dòng này thì một CẢNH BÁO làm chết cả server:
#     UnicodeEncodeError: 'charmap' codec can't encode character 'ẫ'
$env:PYTHONIOENCODING = "utf-8"

$env:AIC_ENV_FILE = ".env.tier2.local"
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8001
