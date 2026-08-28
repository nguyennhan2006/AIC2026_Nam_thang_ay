#!/usr/bin/env python3
"""Từ câu truy vấn tới file CSV nộp bài, không đi qua UI.

Sinh ra vì UI không dùng được khi truy vấn chạy lâu: `fetch` của trình duyệt
không có timeout do ta đặt, nên một truy vấn TRAKE vài trăm giây sẽ treo rồi
đứt kết nối — `sequences` và `trake` không bao giờ tới nơi, bảng kết quả trống,
và không có dòng nào để dựng file nộp. Script này chạy TRÊN MÁY SERVER nên
không phụ thuộc trình duyệt, và chờ bao lâu cũng được.

Đi đúng hai bước mà UI vẫn đi, không tự chế format:
    POST /v1/search/<task>      -> lấy mảng kết quả (kis | qa | trake)
    POST /v1/submissions/build  -> BTC-format CSV + danh sách lỗi

    python scripts/make_submission.py --task TRAKE \
        --query-file query-p1-16-trake.txt --out query-p1-16-trake.csv

    python scripts/make_submission.py --task TEXTUAL_KIS \
        --query "người đàn ông mặc áo xanh" --out q1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


# Tên task (đúng enum TaskType) -> đuôi endpoint. Hai thứ này KHÔNG trùng nhau
# nên không suy ra được bằng lower(): TEXTUAL_KIS đi tới /kis, QA đi tới /qa.
ENDPOINT = {
    "TEXTUAL_KIS": "kis",
    "QA": "qa",
    "TRAKE": "trake",
}
# Tên trường chứa kết quả trong SearchResponse, theo từng task.
RESULT_FIELD = {"TEXTUAL_KIS": "kis", "QA": "qa", "TRAKE": "trake"}


def post(url: str, payload: dict, timeout: float, api_key: str | None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"HTTP {error.code} từ {url}\n{detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Không gọi được {url}: {error.reason}") from error
    except TimeoutError as error:
        raise SystemExit(
            f"Quá {timeout:.0f}s mà server chưa trả lời {url}.\n"
            "Truy vấn TRAKE chạy retrieval một lần mỗi step nên rất lâu — "
            "tăng --timeout, hoặc tắt bớt nhánh chậm (AIC_ENABLE_OCR_FUZZY=false)."
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo CSV nộp bài từ một truy vấn")
    parser.add_argument("--task", required=True, choices=sorted(ENDPOINT))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="nội dung truy vấn")
    group.add_argument("--query-file", help="đọc truy vấn từ file (giữ nguyên xuống dòng)")
    parser.add_argument("--out", required=True, help="đường dẫn file CSV ghi ra")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=100,
                        help="BTC nhận tối đa 100 dòng; mặc định lấy đủ 100")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="giây chờ mỗi lượt gọi (mặc định 30 phút)")
    parser.add_argument("--api-key", default=None,
                        help="AIC_ONLINE_API_KEY nếu box có đặt khoá")
    arguments = parser.parse_args()

    if arguments.query_file:
        # Giữ nguyên xuống dòng: format đề thi là "E1 ...\nE2 ...\nE3 ...", và
        # planner tách step dựa vào chính các mốc đó.
        with open(arguments.query_file, encoding="utf-8") as handle:
            query = handle.read().strip()
    else:
        query = arguments.query
    if not query:
        raise SystemExit("Truy vấn rỗng.")

    base = arguments.base.rstrip("/")
    task = arguments.task

    print(f"[1/2] POST /v1/search/{ENDPOINT[task]}  (chờ tối đa {arguments.timeout:.0f}s)")
    print(f"      truy vấn: {query[:90]}{'...' if len(query) > 90 else ''}")
    started = time.time()
    search = post(
        f"{base}/v1/search/{ENDPOINT[task]}",
        {"query": query, "task": task, "top_k": arguments.top_k},
        arguments.timeout,
        arguments.api_key,
    )
    elapsed = time.time() - started
    items = search.get(RESULT_FIELD[task]) or []
    print(f"      xong sau {elapsed:.0f}s — {len(items)} kết quả, "
          f"status={search.get('status')}")
    for warning in search.get("warnings") or []:
        print(f"      [cảnh báo] {warning}")

    if not items:
        # Dừng ở đây thay vì ghi ra file rỗng: file 0 dòng nộp lên trông y hệt
        # một file hợp lệ nhưng sai, và sẽ chỉ lộ ra khi đã hết lượt nộp.
        print("\nKHÔNG CÓ KẾT QUẢ — không ghi file.", file=sys.stderr)
        statuses = search.get("branch_status") or []
        if statuses:
            print("Nhánh chậm nhất:", file=sys.stderr)
            for status in sorted(statuses, key=lambda x: -x.get("latency_ms", 0))[:3]:
                print(f"  {status['execution_id']:24s} "
                      f"{status.get('latency_ms', 0):8.0f} ms  {status.get('state')}",
                      file=sys.stderr)
        return 1

    print("[2/2] POST /v1/submissions/build")
    built = post(
        f"{base}/v1/submissions/build",
        {"task": task, RESULT_FIELD[task]: items},
        arguments.timeout,
        arguments.api_key,
    )

    issues = built.get("issues") or []
    for issue in issues:
        row = issue.get("row_index")
        where = f" (dòng {row})" if row is not None else ""
        print(f"      [{issue.get('severity')}] {issue.get('code')}{where}: "
              f"{issue.get('message')}")

    if built.get("has_errors"):
        # `build` cố ý KHÔNG tự sửa hay tự cắt dòng sai — nó chỉ báo. Ghi đè
        # quyết định đó ở đây là làm hỏng đúng cái lưới an toàn ấy.
        print("\nCÓ LỖI trong dòng nộp — không ghi file. Sửa rồi chạy lại.",
              file=sys.stderr)
        return 2

    csv_text = built.get("csv") or ""
    # Bảo đảm có newline CUỐI FILE. `/v1/submissions/build` trả về chuỗi không
    # kết thúc bằng "\n", nên dòng nộp cuối cùng nằm trần ở cuối file. File
    # nhóm từng nộp thật (`query-p1-16-trake.csv` trong bộ SoTuyen1) thì CÓ, và
    # một bộ chấm đọc bằng `split("\n")` hay đòi RFC4180 chặt có thể bỏ hoặc
    # hiểu sai đúng dòng cuối. Thêm một byte để khỏi phải cược vào chuyện đó.
    if csv_text and not csv_text.endswith("\n"):
        csv_text += "\n"
    with open(arguments.out, "w", encoding="utf-8", newline="") as handle:
        handle.write(csv_text)
    rows = csv_text.count("\n")
    print(f"\n-> {arguments.out}  ({built.get('item_count')} dòng, "
          f"{rows} dòng trong file)")
    print("   3 dòng đầu:")
    for line in csv_text.splitlines()[:3]:
        print(f"     {line}")
    if rows > 100:
        print("   CẢNH BÁO: quá 100 dòng, BTC chỉ nhận tối đa 100.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
