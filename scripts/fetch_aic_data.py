"""Tải dữ liệu thi đấu từ mirror của ban tổ chức (`aic-data.ledo.io.vn`).

Đây là nguồn NÊN DÙNG, thay cho `scripts/fetch_kaggle_media.py`. Ba lý do, đều
đo được:

1.  **Tách theo loại.** Keyframe là 28,69 GB nằm trong 14 file .zip riêng; kho
    Kaggle gói chung cả video nên phải kéo 106,13 GB để lấy cùng ngần ấy ảnh.
2.  **Không bị chặn tốc độ.** Kaggle trả 404 cho mọi đường dẫn sau ~112 file và
    không hồi phục hàng chục phút (xem `docs/35` §4). Mirror này đo được
    29,5 MB/s liên tục.
3.  **Có thêm thứ Kaggle không có**: `objects-aic25-b1.zip` (178.195 file phát
    hiện vật thể Open Images) — đúng thứ nhánh `object_search` đang thiếu.

Danh sách link nằm trong CSV ban tổ chức phát ("Dữ liệu cho vòng Sơ Tuyển
AIC 2026 - Batch1.csv", hai cột `Filenames` / `Download link`). Truyền bằng
`--csv`; không truyền thì dùng bảng dựng sẵn ở `MIRROR_ROOT` dưới đây.

VIỆC ĐỔI TÊN vẫn y như `fetch_kaggle_media.py` và dùng CHUNG một bảng tra
(`load_index_map` đọc `canonical/keyframe_scene_mapping.csv` của pack): mirror
đặt tên ảnh theo số thứ tự keyframe (`002.jpg`), export trỏ theo frame index
(`frame_000090.jpg`). Hai script buộc phải đọc từ một nguồn, nếu không sẽ lệch.

    python -m scripts.fetch_aic_data --what keyframes --pack <pack.zip>
    python -m scripts.fetch_aic_data --what keyframes --batch L23 --pack <pack.zip>
    python -m scripts.fetch_aic_data --what objects
    python -m scripts.fetch_aic_data --what videos --batch L21
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

# Dùng chung với đường Kaggle: một bảng đổi tên duy nhất, một cách xuất kho
# chứng chỉ duy nhất. Trùng lặp hai bên là cách chắc chắn để chúng lệch nhau.
from scripts.fetch_kaggle_media import _is_image, load_index_map, system_ca_bundle

MIRROR_ROOT = "https://aic-data.ledo.io.vn"

KEYFRAME_ZIPS = [f"Keyframes_L{n}.zip" for n in (21, 22, 23, 24, 25)] + [
    f"Keyframes_L26_{s}.zip" for s in "abcde"
] + [f"Keyframes_L{n}.zip" for n in (27, 28, 29, 30)]
VIDEO_ZIPS = [f"Videos_L{n}_a.zip" for n in (21, 22, 23, 24, 25)] + [
    f"Videos_L26_{s}.zip" for s in "abcde"
] + [f"Videos_L{n}_a.zip" for n in (27, 28, 29, 30)]
EXTRA_ZIPS = [
    "objects-aic25-b1.zip",
    "map-keyframes-aic25-b1.zip",
    "media-info-aic25-b1.zip",
    "clip-features-32-aic25-b1.zip",
]


def _session(data_root: Path):
    import requests

    session = requests.Session()
    # Cùng lý do đã ghi ở `fetch_kaggle_media.system_ca_bundle`: máy này có CA
    # ký lại kết nối, nằm trong kho của Windows nhưng không có trong certifi.
    session.verify = system_ca_bundle(data_root) or True
    return session


def _links_from_csv(path: Path) -> dict[str, str]:
    """`{ten_file: url}` từ CSV của ban tổ chức.

    `utf-8-sig` vì bản xuất từ Google Sheets có BOM, và cột đầu không tên.
    """

    links: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Filenames") or "").strip()
            url = (row.get("Download link") or "").strip()
            if name and url:
                links[name] = url
    if not links:
        raise SystemExit(f"{path}: khong doc duoc cot Filenames / Download link")
    return links


def download(session, url: str, destination: Path, *, label: str) -> int:
    """Tải `url` về `destination`, nối tiếp nếu đã có một phần.

    Mirror hỗ trợ `Range` (đã kiểm trên cả 32 file), nên đứt giữa chừng chạy lại
    là tiếp chứ không tải lại từ đầu.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    done = destination.stat().st_size if destination.exists() else 0
    head = session.head(url, timeout=120, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length") or 0)
    if total and done >= total:
        print(f"  {label:32s} da co du ({total/2**30:.2f} GB)")
        return 0

    response = session.get(
        url, headers={"Range": f"bytes={done}-"} if done else {}, timeout=3600, stream=True
    )
    response.raise_for_status()
    started, started_at = time.time(), done
    with destination.open("ab" if done else "wb") as sink:
        for chunk in response.iter_content(8 << 20):
            sink.write(chunk)
            done += len(chunk)
            elapsed = max(time.time() - started, 1e-9)
            rate = (done - started_at) / elapsed
            print(
                f"\r  {label:32s} {done/2**30:6.2f}/{total/2**30:.2f} GB "
                f"({rate/2**20:5.1f} MB/s)   ",
                end="",
                flush=True,
            )
    print()
    return done - started_at


def extract_keyframes(archive_path: Path, index_map, data_root: Path, report: Counter) -> None:
    """Giải nén ảnh + ĐỔI TÊN sang `frame_%06d.jpg`.

    Bố cục mirror là `keyframes/<video_id>/<n>.jpg` — nông hơn kho Kaggle một
    cấp (`Keyframes_L23/keyframes/...`), nên khớp theo ĐUÔI đường dẫn chứ không
    theo độ sâu cố định.
    """

    with zipfile.ZipFile(archive_path) as archive:
        by_video: dict[str, dict[int, str]] = defaultdict(dict)
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) < 3 or parts[-3] != "keyframes" or not parts[-1].endswith(".jpg"):
                continue
            stem = Path(parts[-1]).stem
            if stem.isdigit():
                by_video[parts[-2]][int(stem)] = name

        for video_id, available in sorted(by_video.items()):
            wanted = index_map.get(video_id)
            if not wanted:
                report["video_khong_co_trong_export"] += 1
                continue
            target = data_root / "processed" / "keyframes" / video_id
            target.mkdir(parents=True, exist_ok=True)
            for source_index, frame_idx in sorted(wanted.items()):
                member = available.get(source_index)
                if member is None:
                    report["thieu_trong_zip"] += 1
                    continue
                destination = target / f"frame_{frame_idx:06d}.jpg"
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
                report["anh"] += 1
                report["byte"] += len(payload)


def extract_videos(archive_path: Path, data_root: Path, report: Counter) -> None:
    target = data_root / "raw" / "videos"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".mp4"):
                continue
            destination = target / Path(name).name
            if destination.exists() and destination.stat().st_size > 0:
                report["da_co"] += 1
                continue
            with archive.open(name) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink, 8 << 20)
            report["video"] += 1
            report["byte"] += destination.stat().st_size


def extract_objects(archive_path: Path, index_map, data_root: Path, report: Counter) -> None:
    """Giải nén phát hiện vật thể, ĐỔI TÊN theo cùng bảng như ảnh.

    `objects/<video_id>/<n>.json` dùng CÙNG số thứ tự keyframe với ảnh, nên phải
    đi qua đúng bảng tra đó. Ghi ra `storage/processed/objects/<video>/frame_%06d.json`
    để `scripts/import_competition_pack.py --objects` nạp vào export.
    """

    with zipfile.ZipFile(archive_path) as archive:
        by_video: dict[str, dict[int, str]] = defaultdict(dict)
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) < 3 or parts[-3] != "objects" or not parts[-1].endswith(".json"):
                continue
            stem = Path(parts[-1]).stem
            if stem.isdigit():
                by_video[parts[-2]][int(stem)] = name

        for video_id, available in sorted(by_video.items()):
            wanted = index_map.get(video_id)
            if not wanted:
                continue
            target = data_root / "processed" / "objects" / video_id
            target.mkdir(parents=True, exist_ok=True)
            for source_index, frame_idx in sorted(wanted.items()):
                member = available.get(source_index)
                if member is None:
                    report["thieu_trong_zip"] += 1
                    continue
                destination = target / f"frame_{frame_idx:06d}.json"
                if destination.exists():
                    report["da_co"] += 1
                    continue
                destination.write_bytes(archive.read(member))
                report["objects"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--what", required=True, choices=["keyframes", "videos", "objects", "extras"]
    )
    parser.add_argument("--pack", help="pack thi dau — BAT BUOC cho keyframes/objects (bang doi ten)")
    parser.add_argument("--csv", help="CSV link cua ban to chuc; bo qua thi dung bang dung san")
    parser.add_argument("--archive-dir", default="D:/aic_archive", help="noi luu .zip da tai")
    parser.add_argument("--data-root", default="storage")
    parser.add_argument("--batch", action="append", help="vd L23 (lap lai duoc)")
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="giu .zip sau khi giai nen (mac dinh XOA de khoi ton gap doi cho)",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    links = _links_from_csv(Path(arguments.csv)) if arguments.csv else {}
    if arguments.what == "keyframes":
        wanted_zips = KEYFRAME_ZIPS
    elif arguments.what == "videos":
        wanted_zips = VIDEO_ZIPS
    elif arguments.what == "objects":
        wanted_zips = ["objects-aic25-b1.zip"]
    else:
        wanted_zips = EXTRA_ZIPS

    batches = {item.upper() for item in (arguments.batch or [])}
    if batches:
        wanted_zips = [z for z in wanted_zips if any(b in z.upper() for b in batches)]
        if not wanted_zips:
            raise SystemExit(f"khong co file nao khop --batch {sorted(batches)}")

    index_map = {}
    if arguments.what in ("keyframes", "objects"):
        if not arguments.pack:
            raise SystemExit(
                f"--what {arguments.what} can --pack: bang doi ten "
                "(source_keyframe_index -> frame_idx) nam trong pack thi dau."
            )
        index_map = load_index_map(Path(arguments.pack))

    data_root = Path(arguments.data_root)
    archive_dir = Path(arguments.archive_dir)

    if arguments.dry_run:
        print(f"se tai {len(wanted_zips)} file vao {archive_dir}:")
        for name in wanted_zips:
            print(f"  {links.get(name, f'{MIRROR_ROOT}/{name}')}")
        return

    session = _session(data_root)
    report = Counter()
    started = time.time()
    for name in wanted_zips:
        url = links.get(name, f"{MIRROR_ROOT}/{name}")
        archive_path = archive_dir / name
        download(session, url, archive_path, label=name)
        if arguments.what == "keyframes":
            extract_keyframes(archive_path, index_map, data_root, report)
        elif arguments.what == "videos":
            extract_videos(archive_path, data_root, report)
        elif arguments.what == "objects":
            extract_objects(archive_path, index_map, data_root, report)
        else:
            print(f"  {name}: da tai, khong giai nen (--what extras chi tai ve)")
            continue
        if not arguments.keep_zip:
            archive_path.unlink(missing_ok=True)
        print(
            f"    -> anh {report['anh']}  video {report['video']}  objects {report['objects']}"
            f"  ({report['byte']/2**30:.2f} GB)  [{time.time()-started:.0f}s]"
        )

    print()
    print(f"  anh keyframe        {report['anh']}")
    print(f"  video               {report['video']}")
    print(f"  file objects        {report['objects']}")
    print(f"  da co san           {report['da_co']}")
    print(f"  thieu trong zip     {report['thieu_trong_zip']}")
    print(f"  khong phai anh      {report['khong_phai_anh']}")
    print(f"  video la trong zip  {report['video_khong_co_trong_export']}")
    print(f"  tong               {report['byte']/2**30:.2f} GB, {time.time()-started:.0f}s")
    if arguments.what == "keyframes":
        print()
        print("  Kiem lai:  python -m scripts.fetch_kaggle_media --verify "
              "--export storage/exports_competition")
    sys.exit(1 if report["thieu_trong_zip"] else 0)


if __name__ == "__main__":
    main()
