# Baseline cho phép so sánh Tier 2: CÙNG cấu hình .env.fpt.local, chỉ TẮT
# LLM query bundle. Hai lần chạy khi đó khác nhau đúng MỘT biến, nên chênh
# lệch điểm quy được về đúng thay đổi đó.
#
#   .\run_server_baseline.ps1   ->  AIC_ENABLE_LLM_QUERY_BUNDLE mac dinh = false
#   .\run_server_tier2.ps1      ->  AIC_ENABLE_LLM_QUERY_BUNDLE = true
#
# KHÔNG dùng run_server_full.ps1 (.env mặc định) làm baseline: nó chỉ có 5
# nhánh, còn .env.fpt.local có 11 (thêm OCR_FUZZY, EVENT/OBJECT/ACTION/COLOR,
# QUERY_TRANSLATION, EXPANSION, RULES). So hai cái đó là trộn hai nguyên nhân.

# Xem run_server_tier2.ps1 để biết vì sao hai biến này PHẢI ở đây, không phải
# trong file .env: huggingface_hub đóng băng HF_HUB_OFFLINE lúc import module.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# stdout Windows là cp1258, không encode nổi ký tự tiếng Việt tổ hợp.
$env:PYTHONIOENCODING = "utf-8"

$env:AIC_ENV_FILE = ".env.fpt.local"
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8001
