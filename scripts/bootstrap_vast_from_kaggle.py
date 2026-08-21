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
from collections import Counter
from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Callable, Iterable, Optional
from urllib.parse import quote


try:
    from tqdm import tqdm
except Exception:  # tqdm chỉ để nhìn tiến độ, thiếu vẫn chạy được
    tqdm = None

# Chỉ dataset pack thi đấu. `trongnhantran25/aic-nam-thang-ay` CỐ Ý không nằm
# đây: nó là mirror thô của ban tổ chức, ~176k ảnh lẻ đặt tên theo số thứ tự
# keyframe — lệch quy ước `frame_%06d.jpg` mà export trỏ tới, nên tải về cũng
# không dùng được (xem `_is_mirror_keyframe`). Mà chỉ để LIỆT KÊ nó đã tốn
# ~8.800 lượt gọi API vì Kaggle trả 20 file mỗi trang. Cần thì truyền tay:
#     --dataset trongnhantran25/aic-nam-thang-ay
DATASETS = [
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
    """Một luật: file tên thế nào thì về đâu, và giải nén hay ghi thẳng.

    `matches` nhận ĐƯỜNG DẪN đầy đủ trong dataset (đã chuẩn hoá dấu `/`), không
    phải mỗi tên file: có luật chỉ phân biệt được bằng thư mục cha.
    `skip=True` là "nhận ra nhưng cố ý không tải" — khác hẳn không khớp luật nào,
    vốn phải dừng cả lượt.
    """

    label: str
    matches: Callable[[str], bool]
    destination: Callable[[Path], Path]
    is_archive: bool
    skip: bool = False
    # Tính đường dẫn ĐẦY ĐỦ của file đích khi `destination / tên file` là sai.
    # Ảnh keyframe cần giữ thư mục video (`.../keyframes/L21_V001/frame_x.jpg`);
    # làm phẳng thì 176k ảnh dồn vào một thư mục và không ảnh nào tra được.
    target: Optional[Callable[[Path, str], Path]] = None

    def target_path(self, root: Path, path_in_dataset: str) -> Path:
        if self.target is not None:
            return self.target(root, path_in_dataset)
        return self.destination(root) / _basename(path_in_dataset)


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _named(*names: str) -> Callable[[str], bool]:
    wanted = {name.casefold() for name in names}
    return lambda path: _basename(path).casefold() in wanted


def _is_mirror_keyframe(path: str) -> bool:
    """Ảnh theo bố cục mirror ban tổ chức: `.../keyframes/<video>/001.jpg`.

    Mirror đặt tên theo SỐ THỨ TỰ keyframe, còn export trỏ theo FRAME INDEX
    (`frame_000000.jpg`). Chép thẳng thì ảnh nằm trên đĩa mà API không bao giờ
    tra tới — 404 y hệt lúc không có, chỉ tốn 28 GB. Muốn dùng phải dịch qua
    `canonical/keyframe_scene_mapping.csv`, đúng việc `scripts/fetch_aic_data.py`
    làm; đó là đường riêng, không phải việc của script này.
    """

    lowered = path.casefold()
    if "keyframes/" not in lowered:
        return False
    return re.fullmatch(r"\d+\.jpe?g", _basename(lowered)) is not None


# Hai thư mục model, nhận diện bằng CHÍNH TÊN THƯ MỤC chứ không phải tên file:
# model gồm hàng chục file tên chung chung (`config.json`, `tokenizer.json`,
# `model.safetensors`) mà mỗi cái đứng riêng thì không nói lên nó thuộc về đâu.
MODEL_DIRS = ("jina-clip-v2", "jina-embeddings-v3")


def _model_directory_index(path: str) -> Optional[int]:
    parts = path.replace("\\", "/").split("/")
    found = [i for i, part in enumerate(parts[:-1]) if part in MODEL_DIRS]
    return max(found) if found else None


def _is_model_file(path: str) -> bool:
    return _model_directory_index(path) is not None


def _model_target(root: Path, path_in_dataset: str) -> Path:
    """Giữ nguyên cây từ tên thư mục model trở đi.

    Lấy lần xuất hiện CUỐI nên `.../storage/models/jina-clip-v2/x` và
    `03_models/jina-clip-v2/x` ra cùng một đích, không lồng thêm tầng nào.
    """

    parts = path_in_dataset.replace("\\", "/").split("/")
    index = _model_directory_index(path_in_dataset)
    assert index is not None  # đã lọc bằng _is_model_file
    return root.joinpath("storage", "models", *parts[index:])


def _is_frame_image(path: str) -> bool:
    """Ảnh đúng quy ước repo: `frame_000123.jpg` — khớp thẳng `image_path` trong
    export, không cần bảng tra nào."""

    return re.fullmatch(r"frame_\d+\.jpe?g", _basename(path).casefold()) is not None


def _frame_target(root: Path, path_in_dataset: str) -> Path:
    """Giữ nguyên thư mục video: `.../keyframes/<video_id>/frame_x.jpg`."""

    parts = path_in_dataset.replace("\\", "/").split("/")
    video = parts[-2] if len(parts) >= 2 else "unknown"
    return root / "storage" / "processed" / "keyframes" / video / parts[-1]


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
    # Trước các luật theo tên file: model chứa `config.json`, `tokenizer.json`...
    # là những tên quá chung, phải xét thư mục cha mới biết chúng đi đâu.
    Route("file model jina", _is_model_file,
          lambda root: root / "storage" / "models", False,
          target=_model_target),
    # File lẻ: dataset chứa thư mục ĐÃ giải nén (`02_vectors/L21_V001.npy`)
    # thay vì các .zip. Khớp theo tên file nên thư mục bọc ngoài không ảnh hưởng.
    Route("export lẻ", lambda path: _basename(path) in EXPORT_JSONL,
          lambda root: root / "storage" / "exports_competition", False),
    Route("vector lẻ", lambda path: path.casefold().endswith(".npy"),
          lambda root: root / "storage" / "processed" / "embeddings_pack", False),
    # Ảnh ĐÚNG quy ước (`frame_%06d.jpg`) — dùng được ngay, nhưng phải giữ thư
    # mục video nên cần `target` riêng thay vì `destination / tên file`.
    Route("ảnh keyframe (frame_%06d.jpg)", _is_frame_image,
          lambda root: root / "storage" / "processed" / "keyframes", False,
          target=_frame_target),
    # Nhận ra để KHÔNG dừng cả lượt, nhưng cũng không tải. Xem lý do ở
    # `_is_mirror_keyframe`.
    Route("ảnh mirror (tên lệch quy ước — bỏ)", _is_mirror_keyframe,
          lambda root: root, False, True),
]


def resolve_route(path_in_dataset: str) -> Optional[Route]:
    """Định tuyến theo ĐƯỜNG DẪN trong dataset, đã chuẩn hoá dấu `/`.

    Phần lớn luật chỉ nhìn tên file nên thư mục bọc ngoài của Kaggle không ảnh
    hưởng; riêng luật nhận diện ảnh mirror cần thư mục cha mới phân biệt được.
    """

    path = path_in_dataset.replace("\\", "/")
    for route in ROUTES:
        if route.matches(path):
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


def dataset_files(dataset: str, *, max_pages: int = 2000) -> list[dict]:
    """Liệt kê TOÀN BỘ file trong dataset để định tuyến trước khi tải byte nào.

    PHẢI lặp phân trang: endpoint này trả mặc định 20 file một trang và không
    báo gì khi còn nữa. Đọc mỗi trang đầu thì `--plan` in ra một bảng trông
    hoàn toàn hợp lệ, rồi tải về 20/873 vector — hỏng đúng kiểu im lặng mà cả
    script này sinh ra để chặn. Đã dính một lần.

    Phân trang bằng `page_token` — endpoint này KHÔNG nhận `page` (gửi kèm là
    HTTP 400). `page_size` thì tuỳ phiên bản, nên gặp 400 sẽ bỏ nó ra và chạy
    lại với mặc định của Kaggle.
    """

    # `requests` import TRONG hàm, không ở đầu file: `patch_jina_config` của
    # module này được `scripts/prepare_jina_offline.py` (và qua đó là
    # `run_competition.sh`) gọi lại, mà đường khởi động server không có lý do gì
    # phải chết chỉ vì máy chưa cài thư viện tải Kaggle.
    import requests

    owner, slug = dataset.split("/", 1)
    headers, basic_auth = load_auth()
    url = f"{API}/datasets/list/{owner}/{slug}"

    files: list[dict] = []
    seen: set[str] = set()
    token: Optional[str] = None
    page_size: Optional[int] = 200

    for _ in range(max_pages):
        parameters: dict[str, object] = {}
        if page_size:
            parameters["page_size"] = page_size
        if token:
            parameters["page_token"] = token

        response = requests.get(
            url, params=parameters, auth=basic_auth,
            headers={**headers, "Accept": "application/json"},
            timeout=(30, 120),
        )
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Kaggle trả HTTP {response.status_code} cho {dataset}. Cấu hình "
                "KAGGLE_API_TOKEN, hoặc KAGGLE_USERNAME/KAGGLE_KEY, hoặc "
                "~/.kaggle/kaggle.json."
            )
        if response.status_code == 400 and page_size is not None:
            log("API không nhận page_size, dùng mặc định của Kaggle (chậm hơn).")
            page_size = None
            continue
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("datasetFiles") or payload.get("files") or []

        fresh = 0
        for entry in entries:
            name = str(entry.get("name") or entry.get("ref") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            files.append({
                "name": name,
                "size": int(entry.get("totalBytes") or entry.get("size") or 0),
            })
            fresh += 1

        token = payload.get("nextPageToken") or payload.get("next_page_token")
        if not token:
            if fresh and len(entries) >= (page_size or 20):
                # Trang đầy mà không có token tiếp: nhiều khả năng bản API này
                # phân trang kiểu khác và ta đang cắt cụt. Không tự đoán tiếp —
                # nói ra, vì `verify()` sau đó sẽ báo thiếu mà không rõ vì sao.
                log(f"CẢNH BÁO: {dataset} trả trang đầy ({len(entries)} file) "
                    "nhưng không có nextPageToken — danh sách có thể bị cắt.")
            break
        # Chỉ báo mỗi 500 file: trang 20-file cho dataset lớn sẽ in hàng trăm
        # dòng, đẩy hết phần cần đọc ra khỏi màn hình.
        if len(files) % 500 < (page_size or 20):
            log(f"  ... đã liệt kê {len(files)} file")
    else:
        raise RuntimeError(
            f"{dataset}: quá {max_pages} trang khi liệt kê — nghi phân trang lặp "
            "vô tận, dừng thay vì kéo mãi."
        )

    return files


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


def stream_dataset_bulk(
    dataset: str, root: Path, *, skip_keyframes: bool, overwrite: bool,
) -> tuple[Counter, Counter, list[str], int]:
    """Tải NGUYÊN dataset một lần và giải nén chọn lọc ngay trong lúc chảy.

    Vì sao không đi đường liệt-kê-rồi-tải-từng-file: endpoint liệt kê phân trang
    20 file/trang và trả 404 khi `page_token` đi sâu (đã gặp thật ở ~24.000
    file), mà dataset này có ~180.000 entry. Đường đó vừa chậm vừa gãy.

    Vì sao không tải .zip xuống rồi mới giải nén: 35 GB nằm chờ trên đĩa, trong
    khi cùng ổ đó còn phải chứa video sau này. Ở đây zip KHÔNG bao giờ chạm đĩa
    — entry nào không cần thì đọc qua rồi bỏ, nên chỉ ~7 GB thực sự được ghi.

    Đánh đổi phải nói rõ: không có pha `--plan` nữa vì không liệt kê trước được.
    Bù lại tính chất an toàn vẫn giữ — entry không khớp luật nào thì KHÔNG ghi
    đi đâu cả, chỉ được báo lại ở cuối.
    """

    import requests
    try:
        from stream_unzip import stream_unzip
    except ImportError as error:  # noqa: TRY003 - thông báo cần đủ dài để làm theo
        raise RuntimeError(
            "Thiếu stream-unzip (đọc zip trong lúc tải, không lưu xuống đĩa).\n"
            "  pip install stream-unzip"
        ) from error

    owner, slug = dataset.split("/", 1)
    url = f"{API}/datasets/download/{owner}/{slug}"
    headers, basic_auth = load_auth()

    written: Counter = Counter()
    skipped: Counter = Counter()
    unrouted: list[str] = []
    written_bytes = 0

    log(f"Tải nguyên khối: {dataset}")
    log("Zip KHÔNG lưu xuống đĩa; chỉ entry cần mới được ghi.")

    with requests.get(url, headers=headers, auth=basic_auth, stream=True,
                      allow_redirects=True, timeout=(30, 300)) as response:
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Kaggle trả HTTP {response.status_code} cho {dataset} — thiếu "
                "quyền hoặc chưa cấu hình khoá."
            )
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", "0") or 0)
        bar = tqdm(total=total or None, unit="B", unit_scale=True,
                   unit_divisor=1024, desc=slug, dynamic_ncols=True) \
            if tqdm is not None else None

        def counted(iterator):
            for chunk in iterator:
                if bar is not None:
                    bar.update(len(chunk))
                yield chunk

        for raw_name, _size, chunks in stream_unzip(
                counted(response.iter_content(chunk_size=CHUNK_SIZE))):
            name = raw_name.decode("utf-8", "replace") if isinstance(raw_name, bytes) \
                else str(raw_name)
            if name.endswith("/"):
                for _ in chunks:  # entry thư mục: vẫn phải rút cạn để đi tiếp
                    pass
                continue

            route = resolve_route(name)
            if route is None:
                unrouted.append(f"{dataset}: {name}")
                for _ in chunks:
                    pass
                continue
            if route.skip or (skip_keyframes and "keyframe" in route.label):
                skipped[route.label] += 1
                for _ in chunks:
                    pass
                continue

            target = route.target_path(root, name)
            if not overwrite and target.exists():
                # Cho phép chạy lại sau khi đứt giữa chừng mà không ghi đè phần
                # đã xong. Vẫn phải rút cạn chunks: stream chỉ tiến được khi
                # entry hiện tại đã đọc hết.
                skipped["đã có sẵn (--no-overwrite)"] += 1
                for _ in chunks:
                    pass
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if route.is_archive:
                # Zip lồng trong zip: phải có file thật mới mở được mục lục.
                # Ghi tạm cạnh đích rồi xoá, không để lại rác.
                destination = route.destination(root)
                destination.mkdir(parents=True, exist_ok=True)
                handle = tempfile.NamedTemporaryFile(
                    delete=False, dir=str(destination), suffix=".part")
                try:
                    for chunk in chunks:
                        handle.write(chunk)
                        written_bytes += len(chunk)
                    handle.close()
                    with zipfile.ZipFile(handle.name) as archive:
                        archive.extractall(destination)
                finally:
                    handle.close()
                    Path(handle.name).unlink(missing_ok=True)
            else:
                with target.open("wb") as sink:
                    for chunk in chunks:
                        sink.write(chunk)
                        written_bytes += len(chunk)
            written[route.label] += 1

        if bar is not None:
            bar.close()

    return written, skipped, unrouted, written_bytes


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


def build_plan(
    datasets: Iterable[str], *, skip_keyframes: bool,
) -> tuple[list[Item], list[str], Counter]:
    items: list[Item] = []
    unrouted: list[str] = []
    skipped: Counter = Counter()
    for dataset in datasets:
        for entry in dataset_files(dataset):
            route = resolve_route(entry["name"])
            if route is None:
                unrouted.append(f"{dataset}: {entry['name']}")
                continue
            if route.skip or (skip_keyframes and "keyframe" in route.label):
                skipped[route.label] += 1
                continue
            items.append(Item(dataset, entry["name"], entry["size"], route))
    return items, unrouted, skipped


def _group(paths: Iterable[str]) -> Counter:
    """Gộp theo (dataset, thư mục cấp một, đuôi file).

    Dataset ảnh có 176k file; in hết ra thì thông tin duy nhất cần đọc — CÓ
    NHỮNG LOẠI GÌ — trôi mất khỏi màn hình, mà terminal trên máy thuê thường
    không cuộn lại được.
    """

    groups: Counter = Counter()
    for line in paths:
        dataset, _, path = line.partition(": ")
        parts = path.replace("\\", "/").split("/")
        head = parts[0] if len(parts) > 1 else "(gốc)"
        suffix = Path(parts[-1]).suffix.casefold() or "(không đuôi)"
        groups[f"{dataset}: {head}/**/*{suffix}"] += 1
    return groups


def print_plan(items: list[Item], root: Path, skipped: Counter) -> None:
    by_destination: Counter = Counter()
    sizes: Counter = Counter()
    for item in items:
        key = f"{item.route.label} -> {item.destination(root)}"
        by_destination[key] += 1
        sizes[key] += item.size

    print(f"\n{'nhóm':56s} {'file':>7s} {'GB':>8s}")
    for key in sorted(by_destination):
        print(f"{key:56s} {by_destination[key]:7d} {sizes[key] / (1 << 30):8.2f}")
    print(f"{'TỔNG':56s} {len(items):7d} "
          f"{sum(sizes.values()) / (1 << 30):8.2f}")

    if skipped:
        print("\nCố ý BỎ QUA:")
        for label, count in sorted(skipped.items()):
            print(f"  {label:52s} {count:7d} file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu thi đấu từ Kaggle về đúng thư mục trên Vast.ai."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help=f"gốc repo AIC. Mặc định {DEFAULT_ROOT}")
    parser.add_argument("--dataset", action="append", default=None,
                        help="owner/slug — lặp lại được. Mặc định: pack thi đấu "
                             "(nguyenchonnhan/data-for-namthangay-competition).")
    parser.add_argument("--per-file", action="store_true",
                        help="liệt kê rồi tải từng file (chỉ hợp dataset NHỎ — "
                             "endpoint liệt kê 404 khi đi sâu quá ~24.000 file)")
    parser.add_argument("--plan", action="store_true",
                        help="chỉ in bảng file -> đích rồi dừng (cần --per-file)")
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

    if not arguments.per_file:
        if arguments.plan:
            log("--plan cần --per-file: chế độ nguyên khối không liệt kê trước "
                "được (xem docstring stream_dataset_bulk).")
            return 2
        written: Counter = Counter()
        skipped: Counter = Counter()
        unrouted: list[str] = []
        total_bytes = 0
        for dataset in datasets:
            one_written, one_skipped, one_unrouted, one_bytes = stream_dataset_bulk(
                dataset, root, skip_keyframes=arguments.skip_keyframes,
                overwrite=not arguments.no_overwrite)
            written += one_written
            skipped += one_skipped
            unrouted += one_unrouted
            total_bytes += one_bytes

        print(f"\n{'đã ghi':56s} {'file':>7s}")
        for label, count in sorted(written.items()):
            print(f"{label:56s} {count:7d}")
        print(f"{'—— tổng ghi ra đĩa':56s} {sum(written.values()):7d}  "
              f"({human_bytes(total_bytes)})")
        if skipped:
            print("\nBỏ qua:")
            for label, count in sorted(skipped.items()):
                print(f"  {label:52s} {count:7d}")
        if unrouted:
            # Không ghi đi đâu cả — chỉ báo. Khác chế độ --per-file ở chỗ không
            # dừng được nữa (stream đã chạy), nhưng tính an toàn vẫn nguyên.
            print("\nKHÔNG BIẾT ĐỔ ĐI ĐÂU (đã BỎ, không ghi):")
            for group, count in sorted(_group(unrouted).items()):
                print(f"  {group:60s} {count:7d}")
            print("\n  Ví dụ:")
            for name in unrouted[:5]:
                print(f"    - {name}")

        patch_jina_config(root)
        failures = verify(root)
        print("\nTiếp theo:")
        print("  python -m scripts.prepare_jina_offline")
        print("  ./scripts/run_competition_linux.sh")
        return 1 if failures else 0

    ensure_bsdtar()
    for dataset in datasets:
        log(f"Liệt kê    : {dataset} ...")
    items, unrouted, skipped = build_plan(
        datasets, skip_keyframes=arguments.skip_keyframes)

    if unrouted:
        # Đoán đích cho file lạ chính là cách bản trước làm hỏng mọi thứ.
        print("\nKHÔNG BIẾT ĐỔ ĐI ĐÂU — thêm luật vào ROUTES rồi chạy lại:")
        for group, count in sorted(_group(unrouted).items()):
            print(f"  {group:60s} {count:7d} file")
        print(f"  {'—— tổng':60s} {len(unrouted):7d} file")
        print("\n  Ví dụ:")
        for name in unrouted[:5]:
            print(f"    - {name}")
        return 2
    if not items:
        log("Không có file nào để tải (dataset rỗng hoặc đã lọc hết).")
        return 2

    print_plan(items, root, skipped)
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
