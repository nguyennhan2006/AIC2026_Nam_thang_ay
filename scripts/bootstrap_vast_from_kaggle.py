#!/usr/bin/env python3
"""Kéo dữ liệu thi đấu từ Kaggle về ĐÚNG chỗ trên máy Vast.ai.

Bản trước ánh xạ theo DATASET: "dataset A đổ vào gốc repo, dataset B đổ vào
`storage/exports_competition`". Cách đó hỏng vì các archive do
`scripts/build_kaggle_dataset.py` sinh ra đã mang sẵn đường dẫn tính TỪ GỐC
REPO (`storage/exports_competition/scenes.jsonl`, `storage/models/...`). Chỉ
cần một archive nằm ở dataset "sai" là toàn bộ nội dung của nó lệch một tầng
(`storage/exports_competition/storage/exports_competition/...`) — và lệch trong
im lặng, vì bsdtar tạo thư mục mới không hỏi gì.

Bản này ánh xạ theo TỪNG FILE. Mỗi file tự khai nó là gì qua tên, `ROUTES`
quyết định đích, và file không khớp luật nào thì DỪNG chứ không đoán. Nhờ vậy
xếp archive nào vào dataset nào cũng không còn quan trọng.

    python scripts/bootstrap_vast_from_kaggle.py --plan   # xem đích, chưa tải
    python scripts/bootstrap_vast_from_kaggle.py          # tải thật
    python scripts/bootstrap_vast_from_kaggle.py --verify-only

Cần: `apt-get install -y libarchive-tools` (bsdtar) và `pip install requests tqdm`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterable, Optional
from urllib.parse import quote


try:
    from tqdm import tqdm
except Exception:  # tqdm chỉ để nhìn tiến độ, thiếu vẫn chạy được
    tqdm = None

DATASETS = [
    "trongnhantran25/aic-nam-thang-ay",
    "nguyenchonnhan/data-for-namthangay-competition",
]

DEFAULT_ROOT = Path("/workspace/AIC2026_Nam_thang_ay")
API = "https://www.kaggle.com/api/v1"
CHUNK_SIZE = 8 * 1024 * 1024

# Zip có magic `PK\x03\x04`. Cần nhận diện vì Kaggle đôi khi bọc thêm một lớp
# zip quanh file lẻ khi tải từng file, đôi khi trả nguyên bản — không báo trước.
ZIP_MAGIC = b"PK\x03\x04"

EXPORT_JSONL = {
    "scenes.jsonl", "keyframes.jsonl", "videos.jsonl",
    "events.jsonl", "clips.jsonl", "dataset_manifest.json",
}


def log(message: str) -> None:
    print(f"[kaggle] {message}", flush=True)


def human_bytes(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if count < 1024 or unit == "TiB":
            return f"{count:.2f} {unit}"
        count /= 1024
    return f"{count:.2f} TiB"


def hf_modules_root() -> Path:
    """Thư mục cache `trust_remote_code` của HuggingFace.

    Trên Vast.ai `HF_HOME` thường đã bị trỏ sang `/workspace/.hf_home` để nằm
    trên volume bền vững, nên KHÔNG hard-code `~/.cache/huggingface`: đặt nhầm
    chỗ thì jina-clip-v2 không tìm thấy module và container chết lúc dựng.
    """

    home = os.environ.get("HF_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "modules"


# --- Bảng định tuyến ---------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """Một luật: file tên thế nào thì về đâu, và giải nén hay ghi thẳng."""

    label: str
    matches: Callable[[str], bool]
    destination: Callable[[Path], Path]
    is_archive: bool


def _named(*names: str) -> Callable[[str], bool]:
    wanted = {name.casefold() for name in names}
    return lambda name: name.casefold() in wanted


ROUTES: list[Route] = [
    # Năm archive dưới đây có arcname tính từ gốc repo, nên giải nén tại gốc
    # repo là ra đúng cây thư mục — không phải chọn thư mục con nào cả.
    Route("export (5 JSONL + manifest)", _named("01_export.zip"),
          lambda root: root, True),
    Route("vector jina (873 .npy)", _named("02_vectors.zip"),
          lambda root: root, True),
    Route("model jina-clip-v2 + e-v3", _named("03_models.zip"),
          lambda root: root, True),
    Route("env đã bỏ khoá + docs", _named("05_config.zip"),
          lambda root: root, True),
    Route("ảnh keyframe (28,6 GB)", _named("06_keyframes.zip"),
          lambda root: root, True),
    # Ngoại lệ duy nhất: arcname của nó là `transformers_modules/...`, thuộc
    # cache HuggingFace NGOÀI repo. Giải nén vào gốc repo là hỏng âm thầm —
    # file có mặt, nhưng transformers không bao giờ nhìn tới chỗ đó.
    Route("cache trust_remote_code", _named("04_hf_modules.zip"),
          lambda root: hf_modules_root(), True),
    # File lẻ (dataset không đóng gói qua build_kaggle_dataset.py).
    Route("export lẻ", lambda name: name in EXPORT_JSONL,
          lambda root: root / "storage" / "exports_competition", False),
    Route("vector lẻ", lambda name: name.endswith(".npy"),
          lambda root: root / "storage" / "processed" / "embeddings_pack", False),
]


def resolve_route(path_in_dataset: str) -> Optional[Route]:
    """Định tuyến theo TÊN FILE, bỏ qua thư mục bọc ngoài của Kaggle.

    Kaggle giữ nguyên cây thư mục lúc upload, nhưng người upload có thể đã bọc
    thêm một cấp. Tên file thì không đổi, nên khớp theo tên file là ổn định hơn
    khớp theo đường dẫn đầy đủ.
    """

    name = path_in_dataset.replace("\\", "/").rsplit("/", 1)[-1]
    for route in ROUTES:
        if route.matches(name):
            return route
    return None


# --- Kaggle API --------------------------------------------------------------


def ensure_bsdtar() -> str:
    executable = shutil.which("bsdtar")
    if not executable:
        raise RuntimeError(
            "Không tìm thấy bsdtar. Cài bằng:\n"
            "  apt-get update && apt-get install -y libarchive-tools"
        )
    return executable


def load_auth() -> tuple[dict, Optional[tuple[str, str]]]:
    headers = {
        "User-Agent": "AIC2026-Vast-Kaggle-Streamer/2.0",
        "Accept": "application/octet-stream,*/*",
    }

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        return headers, None

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return headers, (username, key)

    credentials = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    credentials = credentials / "kaggle.json"
    if credentials.exists():
        try:
            data = json.loads(credentials.read_text(encoding="utf-8"))
            username = str(data.get("username", "")).strip()
            key = str(data.get("key", "")).strip()
            if username and key:
                return headers, (username, key)
        except Exception as error:
            log(f"Cảnh báo: không đọc được {credentials}: {error}")

    return headers, None


def dataset_files(dataset: str) -> list[dict]:
    """Liệt kê file trong dataset để định tuyến TRƯỚC khi tải một byte nào."""

    # `requests` import TRONG hàm, không ở đầu file: `patch_jina_config` của
    # module này được `scripts/prepare_jina_offline.py` (và qua đó là
    # `run_competition.sh`) gọi lại, mà đường khởi động server không có lý do gì
    # phải chết chỉ vì máy chưa cài thư viện tải Kaggle.
    import requests

    owner, slug = dataset.split("/", 1)
    headers, basic_auth = load_auth()
    response = requests.get(
        f"{API}/datasets/list/{owner}/{slug}",
        headers={**headers, "Accept": "application/json"},
        auth=basic_auth,
        timeout=(30, 120),
    )
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Kaggle trả HTTP {response.status_code} cho {dataset}. Cấu hình "
            "KAGGLE_API_TOKEN, hoặc KAGGLE_USERNAME/KAGGLE_KEY, hoặc "
            "~/.kaggle/kaggle.json."
        )
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("datasetFiles") or payload.get("files") or []
    return [
        {
            "name": str(entry.get("name") or entry.get("ref") or ""),
            "size": int(entry.get("totalBytes") or entry.get("size") or 0),
        }
        for entry in entries
        if (entry.get("name") or entry.get("ref"))
    ]


# --- Tải + giải nén ----------------------------------------------------------


@dataclass
class Item:
    dataset: str
    name: str
    size: int
    route: Route

    def destination(self, root: Path) -> Path:
        return self.route.destination(root)


def stream_item(item: Item, root: Path, *, retries: int, overwrite: bool) -> None:
    """Tải một file và đổ thẳng vào đích, không lưu bản trung gian.

    35 GB đi qua SSD hai lần (lưu rồi mới giải nén) là lãng phí cả disk lẫn
    thời gian trên máy thuê tính theo giờ, nên HTTP stream nối thẳng vào stdin
    của bsdtar.
    """

    bsdtar = ensure_bsdtar()
    destination = item.destination(root)
    destination.mkdir(parents=True, exist_ok=True)

    owner, slug = item.dataset.split("/", 1)
    url = f"{API}/datasets/download/{owner}/{slug}/{quote(item.name)}"
    headers, basic_auth = load_auth()

    command = [bsdtar, "-x", "-f", "-", "-C", str(destination),
               "--no-same-owner", "--no-same-permissions"]
    if not overwrite:
        command.append("--keep-old-files")

    for attempt in range(1, retries + 1):
        log("=" * 72)
        log(f"File   : {item.name}  ({human_bytes(item.size)})")
        log(f"Từ     : {item.dataset}")
        log(f"Về     : {destination}   [{item.route.label}]")
        log(f"Lần    : {attempt}/{retries}")

        import requests

        response = None
        process = None
        handle = None
        bar = None

        try:
            response = requests.get(url, headers=headers, auth=basic_auth,
                                    stream=True, allow_redirects=True,
                                    timeout=(30, 180))
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"Kaggle trả HTTP {response.status_code} — thiếu quyền hoặc "
                    "chưa cấu hình khoá."
                )
            response.raise_for_status()

            total = int(response.headers.get("Content-Length", "0") or 0)
            stream = response.iter_content(chunk_size=CHUNK_SIZE)

            # Đọc trước một mẩu để biết Kaggle trả zip hay file nguyên bản. Cả
            # hai đều xảy ra với cùng một endpoint tuỳ kích thước file, nên đoán
            # theo đuôi tên là sai — phải nhìn magic.
            head = b""
            for chunk in stream:
                if chunk:
                    head = chunk
                    break
            is_zip = head.startswith(ZIP_MAGIC)

            if is_zip:
                process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stdout=sys.stdout, stderr=sys.stderr)
                if process.stdin is None:
                    raise RuntimeError("Không mở được stdin của bsdtar.")
                sink = process.stdin
            else:
                if item.route.is_archive:
                    raise RuntimeError(
                        f"{item.name} lẽ ra là archive nhưng nội dung không phải "
                        "zip — dừng thay vì ghi bừa."
                    )
                target = destination / Path(item.name).name
                handle = target.open("wb")
                sink = handle

            if tqdm is not None:
                bar = tqdm(total=total or None, unit="B", unit_scale=True,
                           unit_divisor=1024, desc=Path(item.name).name,
                           dynamic_ncols=True)

            sink.write(head)
            if bar is not None:
                bar.update(len(head))
            for chunk in stream:
                if not chunk:
                    continue
                if process is not None and process.poll() not in (None, 0):
                    raise RuntimeError(f"bsdtar dừng sớm, exit {process.poll()}")
                sink.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))

            sink.close()
            if process is not None:
                code = process.wait()
                if code != 0:
                    raise RuntimeError(f"bsdtar thất bại, exit {code}")

            if bar is not None:
                bar.close()
                bar = None
            log(f"OK: {item.name} -> {destination}")
            return

        except KeyboardInterrupt:
            if process is not None:
                process.kill()
            raise
        except Exception as error:
            if bar is not None:
                bar.close()
            if handle is not None and not handle.closed:
                handle.close()
            if process is not None and process.poll() is None:
                process.kill()
            log(f"LỖI: {error}")
            if attempt >= retries:
                raise
            delay = 3 * attempt
            log(f"Thử lại sau {delay}s...")
            time.sleep(delay)
        finally:
            if response is not None:
                response.close()


# --- Sau khi tải -------------------------------------------------------------


def patch_jina_config(root: Path) -> None:
    """Trỏ text tower của jina-clip-v2 sang bản local.

    `config.json` gốc khai `hf_model_name_or_path: "jinaai/jina-embeddings-v3"`
    — một repo TRÊN HuggingFace. Với `HF_HUB_OFFLINE=1` thì `model_info` NÉM LỖI
    thay vì rơi về cache, còn bỏ cờ đó thì phụ thuộc mạng. Vá ở đây vì mỗi lần
    tải lại model là ghi đè, và lỗi quay lại mà không ai nhớ vì sao.
    Cùng cách xử lý với `scripts/run_competition.ps1`.
    """

    config_path = root / "storage" / "models" / "jina-clip-v2" / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config") or {}
    current = str(text_config.get("hf_model_name_or_path", ""))
    if not current.startswith("jinaai/"):
        return
    backup = config_path.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy(config_path, backup)
    text_config["hf_model_name_or_path"] = "storage/models/jina-embeddings-v3"
    config["text_config"] = text_config
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log("[vá] jina-clip-v2/config.json: text tower -> storage/models/"
        "jina-embeddings-v3 (bản gốc ở config.json.orig)")


def verify(root: Path) -> int:
    """Kiểm ĐÚNG những đường dẫn container thật sự mở, không kiểm thư mục suông.

    Thư mục tồn tại không có nghĩa là dữ liệu nằm đúng tầng — đó chính là kiểu
    hỏng mà bản script trước để lọt.
    """

    export = root / "storage" / "exports_competition"
    vectors = root / "storage" / "processed" / "embeddings_pack"
    models = root / "storage" / "models"

    checks: list[tuple[str, bool, str]] = []
    for name in sorted(EXPORT_JSONL):
        checks.append((f"export/{name}", (export / name).exists(), str(export / name)))

    npy_count = len(list(vectors.glob("*.npy"))) if vectors.exists() else 0
    checks.append((f"vector .npy ({npy_count}/873)", npy_count == 873, str(vectors)))

    for model in ("jina-clip-v2", "jina-embeddings-v3"):
        target = models / model / "config.json"
        checks.append((f"model/{model}", target.exists(), str(target)))

    modules = hf_modules_root() / "transformers_modules"
    checks.append(("hf trust_remote_code", modules.exists(), str(modules)))

    # `transformers_modules` KHÔNG đủ: nó là bản sao dẫn xuất, sinh ra SAU khi
    # `cached_file()` lấy được file .py từ repo code. `cached_file()` chỉ nhìn
    # `$HF_HOME/hub`, nên thiếu hai thư mục dưới đây là container chết lúc dựng
    # với `OSError: We couldn't connect to 'https://huggingface.co'` — dù model
    # đã nằm đủ trên đĩa. Vá bằng `python -m scripts.prepare_jina_offline`.
    hub = hf_modules_root().parent / "hub"
    for repo in ("jinaai--jina-clip-implementation",
                 "jinaai--xlm-roberta-flash-implementation"):
        target = hub / f"models--{repo}"
        checks.append((f"hf code {repo.split('--', 1)[1]}", target.exists(), str(target)))

    keyframes = root / "storage" / "processed" / "keyframes"
    if keyframes.exists():
        log(f"[thông tin] keyframes: {len(list(keyframes.iterdir()))} thư mục video")

    print("\n[verify] Đường dẫn container sẽ mở:")
    failures = 0
    for label, ok, path in checks:
        print(f"  [{'OK' if ok else '--'}] {label:28s} {path}")
        failures += 0 if ok else 1
    return failures


# --- Điều phối ---------------------------------------------------------------


def build_plan(datasets: Iterable[str], *, skip_keyframes: bool) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    unrouted: list[str] = []
    for dataset in datasets:
        for entry in dataset_files(dataset):
            route = resolve_route(entry["name"])
            if route is None:
                unrouted.append(f"{dataset}: {entry['name']}")
                continue
            if skip_keyframes and "keyframe" in route.label:
                continue
            items.append(Item(dataset, entry["name"], entry["size"], route))
    return items, unrouted


def print_plan(items: list[Item], root: Path) -> None:
    print(f"\n{'file':28s} {'GB':>8s}  đích")
    for item in items:
        print(f"{Path(item.name).name:28s} {item.size / (1 << 30):8.2f}  "
              f"{item.destination(root)}")
    print(f"{'TỔNG':28s} {sum(i.size for i in items) / (1 << 30):8.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu thi đấu từ Kaggle về đúng thư mục trên Vast.ai."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help=f"gốc repo AIC. Mặc định {DEFAULT_ROOT}")
    parser.add_argument("--dataset", action="append", default=None,
                        help="owner/slug — lặp lại được. Mặc định: cả hai dataset đã biết.")
    parser.add_argument("--plan", action="store_true",
                        help="chỉ in bảng file -> đích rồi dừng")
    parser.add_argument("--verify-only", action="store_true",
                        help="không tải, chỉ kiểm layout hiện có")
    parser.add_argument("--skip-keyframes", action="store_true",
                        help="bỏ 28,6 GB ảnh; vẫn đủ chạy backend + eval")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="giữ file đã có, không ghi đè")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    root = Path(arguments.root).expanduser().resolve()
    datasets = arguments.dataset or DATASETS

    log(f"Gốc repo   : {root}")
    log(f"HF modules : {hf_modules_root()}")

    if arguments.verify_only:
        patch_jina_config(root)
        return 1 if verify(root) else 0

    ensure_bsdtar()
    items, unrouted = build_plan(datasets, skip_keyframes=arguments.skip_keyframes)

    if unrouted:
        # Đoán đích cho file lạ chính là cách bản trước làm hỏng mọi thứ.
        print("\nKHÔNG BIẾT ĐỔ ĐI ĐÂU — thêm luật vào ROUTES rồi chạy lại:")
        for name in unrouted:
            print(f"  - {name}")
        return 2
    if not items:
        log("Không có file nào để tải (dataset rỗng hoặc đã lọc hết).")
        return 2

    print_plan(items, root)
    if arguments.plan:
        print("\n(--plan — chưa tải gì)")
        return 0

    needed = sum(item.size for item in items)
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    if free < needed * 1.1:
        log(f"DỪNG: cần ~{human_bytes(needed * 1.1)} nhưng chỉ còn "
            f"{human_bytes(free)}. Dùng --skip-keyframes hoặc gắn thêm disk.")
        return 2

    for item in items:
        stream_item(item, root, retries=arguments.retries,
                    overwrite=not arguments.no_overwrite)

    patch_jina_config(root)
    failures = verify(root)

    print("\nTiếp theo:")
    print("  ./scripts/run_competition_linux.sh")
    print("  curl -s http://127.0.0.1:8000/v1/health | python -m json.tool")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
