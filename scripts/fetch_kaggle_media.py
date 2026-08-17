"""Tải keyframe + video từ Kaggle dataset về đúng chỗ hệ đọc.

Nguồn: `kaggle.com/datasets/trongnhantran25/aic-nam-thang-ay` (115,75 GB) —
`Keyframes_L21/keyframes/<video_id>/<n>.jpg` và `Videos_L21_a/video/<id>.mp4`.

VIỆC THẬT SỰ CỦA SCRIPT NÀY LÀ ĐỔI TÊN, không phải tải.

    Kaggle đặt tên ảnh theo SỐ THỨ TỰ keyframe:  1, 2, 3, 4, ...
    export đặt tên theo FRAME INDEX:             0, 90, 261, 351, ...

Chép thẳng thư mục Kaggle vào `storage/processed/keyframes/` là 176.707 ảnh nằm
sai chỗ: `image_path` trong export trỏ `frame_000090.jpg` còn trên đĩa là
`2.jpg`. Hệ **không báo lỗi** — `/media` trả 404, UI hiện ô trống, và cả tầng
xếp hạng vẫn chạy bình thường vì dense/BM25 không đụng tới ảnh. Đúng kiểu hỏng
im lặng mà repo này dành phần lớn công sức để chống.

Bảng đổi tên lấy từ `keyframe_scene_mapping.csv` của pack thi đấu (cột
`source_keyframe_index` -> `frame_idx`), nên `--pack` là tham số BẮT BUỘC cho
keyframe. Cùng bảng mà `scripts/import_competition_pack.py` đã dùng để sinh
`image_path` — hai bên buộc phải đọc từ một nguồn, nếu không sẽ lệch.

KHÔNG ghép theo THỨ TỰ file. 192/873 video có `source_keyframe_index` KHÔNG liên
tục (pack loại vài frame vì conflict/orphan), nên thư mục Kaggle có nhiều ảnh
hơn export cần — vd L21_V006 có 257 file nhưng export chỉ dùng 256. Ghép theo
thứ tự thì mọi ảnh sau chỗ khuyết đều lệch một nấc, và ảnh vẫn hiện ra bình
thường, chỉ là SAI ảnh. Ghép theo SỐ đọc từ tên file.

Quy ước tên (số lượng chữ số) và thư mục cha của từng batch được DÒ lúc chạy —
không đoán: script thử vài dạng trên một file thật rồi khoá lại dạng nào trả về
đúng JPEG. Không dạng nào chạy thì dừng và in ra thứ đã thử.

    python -m scripts.fetch_kaggle_media --what keyframes --batch L21 --pack <zip>
    python -m scripts.fetch_kaggle_media --what keyframes --pack <zip> --workers 12
    python -m scripts.fetch_kaggle_media --what videos --batch L21
    python -m scripts.fetch_kaggle_media --verify --export storage/exports_competition
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Iterable
import zipfile

DATASET = "trongnhantran25/aic-nam-thang-ay"
API_ROOT = "https://www.kaggle.com/api/v1"

# Dạng tên file được thử, theo thứ tự. AIC phát hành keyframe đánh số từ 1.
NAME_FORMATS = ("{n:03d}.jpg", "{n:04d}.jpg", "{n}.jpg", "{n:05d}.jpg", "{n:06d}.jpg", "{n:03d}.jpeg")

def load_env_file_from_environ() -> None:
    """Nạp `AIC_ENV_FILE` như phần còn lại của repo, để khoá Kaggle nằm chung
    một chỗ với khoá FPT thay vì rải thêm một file ở `~/.kaggle/`.

    Dùng lại `online.config.load_env_file` (`override=False`, tức biến đặt sẵn
    trên dòng lệnh vẫn thắng file). Trỏ sai đường dẫn là lỗi DỪNG — cùng lý do
    đã ghi ở `online/config.py`: gõ sai mà vẫn chạy được nghĩa là chạy không có
    khoá và không ai biết.
    """

    raw = os.getenv("AIC_ENV_FILE")
    if not raw:
        return
    from online.config import load_env_file

    path = Path(raw)
    if not path.exists():
        raise SystemExit(f"AIC_ENV_FILE tro toi {path} nhung file khong ton tai")
    load_env_file(path)


# Thư mục cha ứng với mỗi batch. L26 bị chẻ làm 5 phần trên Kaggle nên phải dò;
# các batch khác chỉ có một khả năng nhưng vẫn đi qua cùng đường dò để một bản
# dataset sau có chẻ thêm thì script không chết.
def _keyframe_parents(batch: str) -> list[str]:
    suffixes = ["", "_a", "_b", "_c", "_d", "_e"]
    return [f"Keyframes_{batch}{suffix}/keyframes" for suffix in suffixes]


def _video_parents(batch: str) -> list[str]:
    suffixes = ["_a", "_b", "_c", "_d", "_e", ""]
    return [f"Videos_{batch}{suffix}/video" for suffix in suffixes]


def system_ca_bundle(data_root: Path) -> str | None:
    """Xuất kho chứng chỉ gốc của Windows ra một file PEM, trả về đường dẫn.

    Máy này chặn TLS tới cả `kaggle.com` lẫn `huggingface.co` với
    `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`, trong
    khi `curl.exe` của Windows vào cùng địa chỉ đó trả 200. Chênh lệch đó nói
    đúng một điều: CA đang ký lại kết nối (proxy/antivirus của mạng) CÓ trong
    kho của hệ điều hành nhưng KHÔNG có trong `certifi`.

    Vì sao xuất ra PEM chứ không dùng `truststore.inject_into_ssl()` — đã thử
    và nó hỏng ở đúng chỗ script này cần: `truststore` bắt Windows SChannel làm
    việc bắt tay, và với 8 luồng bắt tay đồng thời nó đổ
    `SSL record layer failure`. Một luồng thì chạy. Xuất ra PEM giữ phần xác
    thực ở OpenSSL — vốn an toàn với đa luồng — mà vẫn chỉ tin đúng những gì
    Windows đã tin.

    KHÔNG dùng `verify=False`: nó tắt xác thực với MỌI host, biến một vấn đề
    cấu hình thành lỗ hổng thường trực trong script chạy hàng giờ.
    """

    import ssl

    if not hasattr(ssl, "enum_certificates"):  # không phải Windows
        return None
    lines: list[str] = []
    seen: set[bytes] = set()
    for store in ("ROOT", "CA"):
        try:
            entries = ssl.enum_certificates(store)
        except OSError:
            continue
        for der, encoding, trust in entries:
            # `trust=True` = tin cho mọi mục đích; set() = tin cho một số OID.
            # Chỉ lấy chứng chỉ dùng được cho xác thực server.
            if not (trust is True or (isinstance(trust, set) and trust)):
                continue
            if encoding != "x509_asn" or der in seen:
                continue
            seen.add(der)
            lines.append(ssl.DER_cert_to_PEM_cert(der))
    if not lines:
        return None

    import certifi

    # Gộp thêm certifi: kho của Windows đủ cho CA nội bộ nhưng có thể thiếu vài
    # CA công cộng mà máy chưa từng gặp, và thiếu một CA là hỏng cả lượt tải.
    bundle = Path(certifi.where()).read_text(encoding="utf-8") + "\n" + "".join(lines)
    destination = data_root / "state" / "ca_bundle_windows.pem"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_text(encoding="utf-8") != bundle:
        destination.write_text(bundle, encoding="utf-8")
    return str(destination)


class KaggleClient:
    """Tải một file trong dataset qua REST API, dùng lại kết nối.

    Không phụ thuộc gói `kaggle`: chỗ này chỉ cần đúng một endpoint, còn gói đó
    kéo theo cả CLI và một tầng cấu hình riêng. Khoá vẫn đọc từ
    `~/.kaggle/kaggle.json` như bình thường.
    """

    def __init__(self, workers: int, data_root: Path) -> None:
        self._auth = self._credentials()
        self.username = self._auth[0]
        self._verify = system_ca_bundle(data_root) or True
        # MỘT Session cho MỖI luồng. `requests.Session` không an toàn đa luồng —
        # tài liệu của chính requests nói vậy — và dùng chung một Session cho 8
        # luồng đã đổ `SSL record layer failure` giữa lượt tải, tức hỏng SAU khi
        # đã tải được vài trăm ảnh chứ không phải hỏng ngay.
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = requests.Session()
            session.auth = self._auth
            session.verify = self._verify
            self._local.session = session
        return session

    @staticmethod
    def _credentials() -> tuple[str, str]:
        """`KAGGLE_USERNAME`/`KAGGLE_KEY` trước, `kaggle.json` sau.

        Biến môi trường thắng vì `AIC_ENV_FILE` đã nạp chúng từ `.env.*.local`
        vào `os.environ` — cùng một chỗ chứa mọi khoá khác của dự án, thay vì
        thêm một file khoá thứ hai ở `~/.kaggle/`.
        """

        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if username and key:
            return username.strip(), key.strip()
        if username or key:
            raise SystemExit(
                "chi co MOT trong hai bien KAGGLE_USERNAME / KAGGLE_KEY duoc dat. "
                "Can ca hai — dien not o file env dang dung (AIC_ENV_FILE)."
            )
        for candidate in (
            Path(os.getenv("KAGGLE_CONFIG_DIR", "")) / "kaggle.json" if os.getenv("KAGGLE_CONFIG_DIR") else None,
            Path.home() / ".kaggle" / "kaggle.json",
        ):
            if candidate and candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return str(data["username"]), str(data["key"])
        raise SystemExit(
            "khong tim thay khoa Kaggle.\n"
            "  Cach dang dung — dien vao file env roi tro AIC_ENV_FILE vao no:\n"
            "      KAGGLE_USERNAME=<username>\n"
            "      KAGGLE_KEY=<key>\n"
            "      $env:AIC_ENV_FILE = '.env.fpt.local'\n"
            "  Lay key: kaggle.com -> Settings -> API -> Create New Token\n"
            "  (username la truong \"username\" trong kaggle.json tai ve, khong phai email).\n"
            "  Cach thay the: chep kaggle.json vao ~/.kaggle/kaggle.json."
        )

    def fetch(self, remote_path: str, *, timeout: int = 120, retry_404: int = 4) -> bytes | None:
        """Nội dung file, hoặc None nếu Kaggle trả 404 sau khi đã thử lại.

        **404 của Kaggle KHÔNG có nghĩa là "không có file".** Đo được: khi vượt
        hạn mức tải, Kaggle trả 404 cho MỌI đường dẫn — kể cả đường vừa tải xong
        một giây trước — chứ không trả 429. Hạn mức đó dùng chung cho cả endpoint
        tải-từng-file lẫn tải-cả-kho.

        Hậu quả nếu tin 404 ngay: một video đang bị chặn tạm thời sẽ bị ghi là
        "không dò được bố cục" và cả lượt tải dừng, dù đường dẫn hoàn toàn đúng.
        Đã dính đúng lỗi này với L23_V003.

        Nên 404 được thử lại có giãn cách như 429; chỉ khi hết lượt mới coi là
        thật sự không có. `retry_404=0` cho đường DÒ bố cục, nơi 404 mới đúng là
        tín hiệu "dạng tên này sai".

        Kaggle đóng gói .zip cho một số file — mở luôn ở đây để caller chỉ phải
        nghĩ về nội dung thật.
        """

        import requests

        url = f"{API_ROOT}/datasets/download/{DATASET}"
        not_found = 0
        for attempt in range(5):
            try:
                response = self._session().get(
                    url, params={"file_name": remote_path}, timeout=timeout, stream=False
                )
            except requests.exceptions.SSLError as exc:
                raise SystemExit(
                    f"TLS bi tu choi khi noi toi kaggle.com: {exc}\n"
                    "  May nay co CA ky lai ket noi (proxy/antivirus) — no CO trong kho\n"
                    "  chung chi cua Windows nhung KHONG co trong certifi.\n"
                    "  Script da tu xuat kho chung chi Windows ra\n"
                    "  storage/state/ca_bundle_windows.pem — loi nay nghia la ca cach do\n"
                    "  cung khong du. Kiem tra proxy/antivirus dang chan.\n"
                    "  Kiem chung: curl.exe -sS -o NUL -w \"%{http_code}\" https://www.kaggle.com/\n"
                    "  ra 200 nghia la kho cua Windows dung, chi Python chua thay."
                ) from exc
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code == 404:
                not_found += 1
                if not_found > retry_404:
                    return None
                time.sleep(min(2**not_found, 30))
                continue
            if response.status_code in (401, 403):
                # Không thử lại: sai khoá thì thử 5 lần vẫn sai, mà thông báo
                # mặc định của requests ("403 Client Error") không nói được là
                # sai khoá hay chưa bấm đồng ý điều khoản dataset.
                raise SystemExit(
                    f"Kaggle tu choi ({response.status_code}) khi lay {remote_path}.\n"
                    "  - Sai KAGGLE_USERNAME/KAGGLE_KEY? username la truong \"username\" trong\n"
                    "    kaggle.json, KHONG phai email dang nhap.\n"
                    "  - Token cu da bi thu hoi khi tao token moi — dung ban moi nhat.\n"
                    "  - Chua mo trang dataset va bam dong y dieu khoan bang chinh tai khoan do?"
                )
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 4:
                    response.raise_for_status()
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            payload = response.content
            if payload[:4] == b"PK\x03\x04":
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    names = archive.namelist()
                    if len(names) != 1:
                        raise RuntimeError(f"{remote_path}: zip co {len(names)} thanh phan, doi 1")
                    payload = archive.read(names[0])
            return payload
        return None


def load_index_map(pack: Path) -> dict[str, dict[int, int]]:
    """`{video_id: {source_keyframe_index: frame_idx}}` từ pack thi đấu."""

    mapping: dict[str, dict[int, int]] = defaultdict(dict)
    name = "canonical/keyframe_scene_mapping.csv"
    try:
        if pack.is_dir():
            handle: Iterable[str] = (pack / name).open(encoding="utf-8")
        else:
            archive = zipfile.ZipFile(pack)
            handle = io.TextIOWrapper(archive.open(name), encoding="utf-8")
    except (FileNotFoundError, KeyError, IsADirectoryError, zipfile.BadZipFile) as exc:
        raise SystemExit(
            f"{pack}: khong doc duoc {name} ({exc!r}).\n"
            "  --pack phai tro vao AIC2026_competition_clean_v3.zip hoac thu muc da giai nen."
        ) from exc
    for row in csv.DictReader(handle):
        mapping[row["video_id"]][int(row["source_keyframe_index"])] = int(row["frame_idx"])
    if not mapping:
        raise SystemExit(f"{pack}: khong doc duoc {name}")
    return dict(mapping)


def _is_image(payload: bytes) -> bool:
    return payload[:3] == b"\xff\xd8\xff" or payload[:8] == b"\x89PNG\r\n\x1a\n"


class Layout:
    """Thư mục cha + dạng tên file, dò một lần rồi dùng lại.

    Dò trên MỘT ảnh thật của mỗi video mới gặp; dạng tên đã tìm được thì khoá
    cho cả dataset (nó đồng nhất), chỉ còn thư mục cha là phải dò lại khi batch
    bị chẻ (L26 có 5 phần).
    """

    def __init__(self, client: KaggleClient) -> None:
        self._client = client
        self._format: str | None = None
        self._parent: dict[str, str] = {}
        self._lock = threading.Lock()
        self.probes = 0

    def resolve(self, video_id: str, probe_index: int) -> tuple[str, str]:
        batch = video_id.split("_")[0]
        with self._lock:
            known_parent = self._parent.get(batch)
            known_format = self._format
        if known_parent and known_format:
            return known_parent, known_format

        parents = [known_parent] if known_parent else _keyframe_parents(batch)
        formats = [known_format] if known_format else list(NAME_FORMATS)
        for retry_404 in (0, 4):
            for parent in parents:
                for name_format in formats:
                    remote = f"{parent}/{video_id}/{name_format.format(n=probe_index)}"
                    with self._lock:
                        self.probes += 1
                    payload = self._client.fetch(remote, retry_404=retry_404)
                    if payload is not None and _is_image(payload):
                        with self._lock:
                            self._parent[batch] = parent
                            self._format = name_format
                        return parent, name_format
                    if retry_404:
                        # Lượt hai chỉ để phân biệt chặn-tốc-độ với sai-đường-dẫn.
                        # Một tổ hợp đã đủ trả lời; thử tiếp 35 tổ hợp nữa với
                        # giãn cách là đốt hạn mức vô ích.
                        break
                if retry_404:
                    break
        raise SystemExit(
            f"{video_id}: khong do duoc bo cuc tren Kaggle.\n"
            f"  da thu thu muc : {parents}\n"
            f"  da thu dang ten: {formats}\n"
            "  Mo Data Explorer cua dataset, xem ten that cua mot file anh, roi bao lai."
        )


def fetch_keyframes(arguments: argparse.Namespace) -> int:
    client = KaggleClient(arguments.workers, Path(arguments.data_root))
    index_map = load_index_map(Path(arguments.pack))
    layout = Layout(client)
    destination_root = Path(arguments.data_root) / "processed" / "keyframes"

    videos = _selected_videos(sorted(index_map), arguments)
    print(f"video: {len(videos)}   keyframe can lay: {sum(len(index_map[v]) for v in videos)}")

    report = Counter()
    missing_on_kaggle: list[str] = []
    lock = threading.Lock()

    def one_video(video_id: str) -> None:
        frames = index_map[video_id]
        target_dir = destination_root / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        pending = {
            source_index: frame_idx
            for source_index, frame_idx in frames.items()
            if not (target_dir / f"frame_{frame_idx:06d}.jpg").exists()
        }
        with lock:
            report["da_co"] += len(frames) - len(pending)
        if not pending:
            return
        parent, name_format = layout.resolve(video_id, min(pending))
        for source_index, frame_idx in sorted(pending.items()):
            remote = f"{parent}/{video_id}/{name_format.format(n=source_index)}"
            payload = client.fetch(remote)
            if payload is None or not _is_image(payload):
                with lock:
                    report["thieu_tren_kaggle"] += 1
                    if len(missing_on_kaggle) < 20:
                        missing_on_kaggle.append(remote)
                continue
            destination = target_dir / f"frame_{frame_idx:06d}.jpg"
            temporary = destination.with_suffix(".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            with lock:
                report["da_tai"] += 1
                report["byte"] += len(payload)

    if arguments.dry_run:
        video_id = videos[0]
        parent, name_format = layout.resolve(video_id, min(index_map[video_id]))
        print(f"bo cuc do duoc: {parent}/<video_id>/{name_format}")
        for source_index, frame_idx in sorted(index_map[video_id].items())[:5]:
            print(
                f"  {parent}/{video_id}/{name_format.format(n=source_index)}"
                f"  ->  processed/keyframes/{video_id}/frame_{frame_idx:06d}.jpg"
            )
        print(f"... tong {sum(len(index_map[v]) for v in videos)} anh cho {len(videos)} video")
        return 0

    started = time.time()
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(one_video, video_id): video_id for video_id in videos}
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 10 == 0 or done == len(videos):
                elapsed = time.time() - started
                rate = report["da_tai"] / elapsed if elapsed else 0
                print(
                    f"  [{done}/{len(videos)} video] tai {report['da_tai']} anh "
                    f"({report['byte']/1e9:.2f} GB, {rate:.1f} anh/s)",
                    flush=True,
                )

    print()
    print(f"  da tai            {report['da_tai']}  ({report['byte']/1e9:.2f} GB)")
    print(f"  da co san         {report['da_co']}")
    print(f"  thieu tren Kaggle {report['thieu_tren_kaggle']}")
    print(f"  lan do bo cuc     {layout.probes}")
    for item in missing_on_kaggle:
        print(f"    thieu: {item}")
    return 1 if report["thieu_tren_kaggle"] else 0


def fetch_videos(arguments: argparse.Namespace) -> int:
    client = KaggleClient(arguments.workers, Path(arguments.data_root))
    destination_root = Path(arguments.data_root) / "raw" / "videos"
    destination_root.mkdir(parents=True, exist_ok=True)

    if arguments.pack:
        video_ids = sorted(load_index_map(Path(arguments.pack)))
    elif arguments.video:
        video_ids = sorted(arguments.video)
    else:
        raise SystemExit("--what videos can --pack (de biet danh sach video) hoac --video")
    videos = _selected_videos(video_ids, arguments)
    print(f"video: {len(videos)}")

    report = Counter()
    lock = threading.Lock()

    def one_video(video_id: str) -> None:
        destination = destination_root / f"{video_id}.mp4"
        if destination.exists() and destination.stat().st_size > 0:
            with lock:
                report["da_co"] += 1
            return
        batch = video_id.split("_")[0]
        for parent in _video_parents(batch):
            payload = client.fetch(f"{parent}/{video_id}.mp4", timeout=1800)
            if payload is None:
                continue
            temporary = destination.with_suffix(".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            with lock:
                report["da_tai"] += 1
                report["byte"] += len(payload)
            return
        with lock:
            report["khong_thay"] += 1

    if arguments.dry_run:
        print(f"se tai {len(videos)} video vao {destination_root}")
        print(f"  vd: Videos_{videos[0].split('_')[0]}_a/video/{videos[0]}.mp4 -> raw/videos/{videos[0]}.mp4")
        return 0

    with ThreadPoolExecutor(max_workers=min(arguments.workers, 4)) as pool:
        futures = {pool.submit(one_video, video_id): video_id for video_id in videos}
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            print(
                f"  [{done}/{len(videos)}] tai {report['da_tai']} video "
                f"({report['byte']/1e9:.1f} GB)",
                flush=True,
            )

    print()
    print(f"  da tai      {report['da_tai']}  ({report['byte']/1e9:.1f} GB)")
    print(f"  da co san   {report['da_co']}")
    print(f"  khong thay  {report['khong_thay']}")
    return 1 if report["khong_thay"] else 0


def download_archive(arguments: argparse.Namespace) -> int:
    """Tải nguyên kho .zip của dataset, nối tiếp được khi đứt.

    Vì sao phải có đường này: endpoint tải-TỪNG-FILE có hạn ngạch rất chặt — đo
    trên chính tài khoản này, **112 ảnh rồi bị 404 toàn bộ**, không hồi phục sau
    5 phút. 176.707 ảnh qua đường đó là không khả thi. Endpoint tải-CẢ-BỘ thì
    tính là MỘT lần tải, và nó hỗ trợ `Range` nên đứt giữa chừng chạy lại được.

    Đổi lại: kho là 106,13 GB vì gói kèm cả video, trong khi riêng keyframe chỉ
    ~32,5 GB. Không tách được ở phía máy chủ.
    """

    client = KaggleClient(1, Path(arguments.data_root))
    destination = Path(arguments.download_archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    done = destination.stat().st_size if destination.exists() else 0

    session = client._session()
    url = f"{API_ROOT}/datasets/download/{DATASET}"
    probe = session.get(url, headers={"Range": "bytes=0-0"}, timeout=300, stream=True)
    probe.raise_for_status()
    total = int(probe.headers.get("Content-Range", "/0").rsplit("/", 1)[-1])
    probe.close()
    print(f"kho: {total/2**30:.2f} GB   da co: {done/2**30:.2f} GB")
    if done >= total > 0:
        print("da tai xong tu truoc")
        return 0

    started = time.time()
    response = session.get(
        url, headers={"Range": f"bytes={done}-"}, timeout=3600, stream=True
    )
    response.raise_for_status()
    with destination.open("ab") as sink:
        for chunk in response.iter_content(8 << 20):
            sink.write(chunk)
            done += len(chunk)
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"\r  {done/2**30:7.2f}/{total/2**30:.2f} GB "
                f"({done/total*100:5.1f}%, {(done)/elapsed/2**20:.1f} MB/s)",
                end="",
                flush=True,
            )
    print()
    if done < total:
        print(f"  CHUA XONG ({done}/{total} byte) — chay lai lenh nay de noi tiep")
        return 1
    return 0


def _archive_members(archive: zipfile.ZipFile) -> tuple[dict[str, dict[int, str]], dict[str, str]]:
    """`({video_id: {source_index: ten_trong_zip}}, {video_id: duong_dan_mp4})`.

    Chỉ số đọc từ TÊN FILE, không phải thứ tự trong zip — cùng lý do đã ghi ở
    đầu module.
    """

    keyframes: dict[str, dict[int, str]] = defaultdict(dict)
    videos: dict[str, str] = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[0].startswith("Keyframes_") and parts[1] == "keyframes":
            stem = Path(parts[3]).stem
            if stem.isdigit():
                keyframes[parts[2]][int(stem)] = name
        elif len(parts) == 3 and parts[0].startswith("Videos_") and parts[1] == "video":
            if parts[2].lower().endswith(".mp4"):
                videos[Path(parts[2]).stem] = name
    return dict(keyframes), videos


def extract_archive(arguments: argparse.Namespace) -> int:
    """Giải nén + ĐỔI TÊN từ kho .zip đã tải về. Không cần mạng."""

    archive_path = Path(arguments.archive)
    if not archive_path.exists():
        raise SystemExit(f"khong thay kho: {archive_path}")
    data_root = Path(arguments.data_root)
    report = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        in_zip_keyframes, in_zip_videos = _archive_members(archive)
        print(f"kho co {len(in_zip_keyframes)} video keyframe, {len(in_zip_videos)} file mp4")

        if arguments.what in (None, "keyframes"):
            index_map = load_index_map(Path(arguments.pack))
            videos = _selected_videos(sorted(index_map), arguments)
            for video_id in videos:
                available = in_zip_keyframes.get(video_id)
                if not available:
                    report["video_thieu_trong_kho"] += 1
                    continue
                target_dir = data_root / "processed" / "keyframes" / video_id
                target_dir.mkdir(parents=True, exist_ok=True)
                for source_index, frame_idx in sorted(index_map[video_id].items()):
                    member = available.get(source_index)
                    if member is None:
                        report["thieu_trong_kho"] += 1
                        continue
                    destination = target_dir / f"frame_{frame_idx:06d}.jpg"
                    if destination.exists():
                        report["da_co"] += 1
                        continue
                    payload = archive.read(member)
                    if not _is_image(payload):
                        report["khong_phai_anh"] += 1
                        continue
                    temporary = destination.with_suffix(".part")
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
                    report["da_giai_nen"] += 1
                    report["byte"] += len(payload)
                if report["da_giai_nen"] % 2000 < len(index_map[video_id]):
                    print(
                        f"\r  {report['da_giai_nen']} anh ({report['byte']/2**30:.2f} GB)",
                        end="",
                        flush=True,
                    )
            print()

        if arguments.what == "videos":
            wanted = _selected_videos(sorted(in_zip_videos), arguments)
            target_dir = data_root / "raw" / "videos"
            target_dir.mkdir(parents=True, exist_ok=True)
            for video_id in wanted:
                destination = target_dir / f"{video_id}.mp4"
                if destination.exists() and destination.stat().st_size > 0:
                    report["da_co"] += 1
                    continue
                with archive.open(in_zip_videos[video_id]) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink, 8 << 20)
                report["da_giai_nen"] += 1
                report["byte"] += destination.stat().st_size
                print(f"  {video_id} ({report['byte']/2**30:.1f} GB)", flush=True)

    print()
    print(f"  da giai nen        {report['da_giai_nen']}  ({report['byte']/2**30:.2f} GB)")
    print(f"  da co san          {report['da_co']}")
    print(f"  thieu trong kho    {report['thieu_trong_kho']}")
    print(f"  video thieu han    {report['video_thieu_trong_kho']}")
    print(f"  khong phai anh     {report['khong_phai_anh']}")
    return 1 if report["thieu_trong_kho"] or report["video_thieu_trong_kho"] else 0


def verify(arguments: argparse.Namespace) -> int:
    """Đối chiếu ảnh trên đĩa với `image_path` mà export ĐANG trỏ tới.

    Đây mới là phép kiểm đúng, không phải đếm số file trong thư mục: export chỉ
    dùng một tập con các ảnh Kaggle phát hành, và cái hỏng cần bắt là ảnh nằm
    SAI TÊN chứ không phải thiếu ảnh.
    """

    export = Path(arguments.export)
    keyframes_path = export / "keyframes.jsonl" if export.is_dir() else export
    data_root = Path(arguments.data_root)

    total = 0
    present = 0
    per_video_missing: Counter = Counter()
    examples: list[str] = []
    with keyframes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            if (data_root / record["image_path"]).exists():
                present += 1
            else:
                per_video_missing[record["video_id"]] += 1
                if len(examples) < 5:
                    examples.append(record["image_path"])

    print(f"  keyframe trong export  {total}")
    print(f"  co anh dung ten        {present}  ({present/total*100:.1f}%)")
    print(f"  thieu                  {total - present}  o {len(per_video_missing)} video")
    for video_id, count in per_video_missing.most_common(10):
        print(f"    {video_id}: thieu {count}")
    for item in examples:
        print(f"    vd thieu: {item}")
    if per_video_missing:
        print()
        print("  Neu thu muc CO file nhung ten khac (vd 1.jpg thay vi frame_000000.jpg)")
        print("  thi do la loi chep thang thay vi chay script nay — xoa roi tai lai.")
    return 1 if per_video_missing else 0


def _selected_videos(video_ids: list[str], arguments: argparse.Namespace) -> list[str]:
    batches = {item.upper() for item in (arguments.batch or [])}
    wanted = set(arguments.video or [])
    selected = [
        video_id
        for video_id in video_ids
        if (not batches or video_id.split("_")[0].upper() in batches)
        and (not wanted or video_id in wanted)
    ]
    if arguments.limit_videos:
        selected = selected[: arguments.limit_videos]
    if not selected:
        raise SystemExit("khong co video nao khop bo loc --batch/--video")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--what", choices=["keyframes", "videos"], help="loai media can tai")
    parser.add_argument("--verify", action="store_true", help="doi chieu anh tren dia voi export")
    parser.add_argument("--pack", help="AIC2026_competition_clean_v3.zip — BAT BUOC cho keyframes")
    parser.add_argument("--export", default="storage/exports_competition", help="dung cho --verify")
    parser.add_argument("--data-root", default="storage")
    parser.add_argument("--batch", action="append", help="vd L21 (lap lai duoc)")
    parser.add_argument("--video", action="append", help="vd L21_V001 (lap lai duoc)")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="chi do bo cuc va in ke hoach")
    parser.add_argument(
        "--download-archive",
        help="tai NGUYEN kho .zip cua dataset ve duong dan nay (106 GB, noi tiep duoc). "
        "Duong nay ton tai vi endpoint tai-tung-file co han ngach rat chat — xem docs/35 §4",
    )
    parser.add_argument(
        "--archive",
        help="giai nen + doi ten TU kho .zip da tai ve, khong can mang",
    )
    arguments = parser.parse_args()
    load_env_file_from_environ()

    if arguments.download_archive:
        sys.exit(download_archive(arguments))
    if arguments.archive:
        if arguments.what in (None, "keyframes") and not arguments.pack:
            raise SystemExit("--archive voi keyframes can --pack (bang doi ten)")
        sys.exit(extract_archive(arguments))
    if arguments.verify:
        sys.exit(verify(arguments))
    if arguments.what == "keyframes":
        if not arguments.pack:
            raise SystemExit(
                "--what keyframes can --pack: bang doi ten (source_keyframe_index -> frame_idx) "
                "nam trong canonical/keyframe_scene_mapping.csv cua pack."
            )
        sys.exit(fetch_keyframes(arguments))
    if arguments.what == "videos":
        sys.exit(fetch_videos(arguments))
    parser.error("can --what keyframes | --what videos | --verify")


if __name__ == "__main__":
    main()
