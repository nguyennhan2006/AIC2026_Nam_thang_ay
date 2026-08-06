"""Đo đánh đổi khi chuyển dense visual sang ANN, ở quy mô thi đấu.

Vì sao không đo thẳng trên dữ liệu hiện có: 855 vector thì ANN và quét tuyến
tính trả về CÙNG kết quả và cùng nhanh, nên phép đo đó không nói gì về 876
video. Đánh đổi chỉ xuất hiện khi corpus lớn.

Nên corpus được nhân lên từ **vector CLIP thật**, không phải vector ngẫu nhiên:
mỗi vector gốc được nhân bản kèm nhiễu Gaussian nhỏ rồi chuẩn hoá lại. Cách này
giữ đúng hình học cục bộ của CLIP (dị hướng, tập trung) và tạo ra cấu trúc
gần-trùng — vốn đúng là thứ một corpus 876 bản tin thời sự sẽ có, khi hàng trăm
video cùng quay "hiện trường tai nạn ban đêm".

Vector ngẫu nhiên Gaussian sẽ cho ANN điểm cao giả tạo, vì các điểm nằm cách
đều nhau trong không gian nhiều chiều và bài toán láng giềng trở nên dễ.

    python -m scripts.bench_ann --videos 876
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

EMB_DIR = Path("storage/processed/embeddings")


def load_real_vectors() -> np.ndarray:
    vectors: list[list[float]] = []
    for video_dir in sorted(EMB_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        for path in sorted(video_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            # File là mảng số thẳng; chấp cả dạng bọc dict phòng khi đổi format.
            vec = payload if isinstance(payload, list) else (
                payload.get("embedding") or payload.get("vector") or payload.get("values")
            )
            if vec:
                vectors.append(vec)
    if not vectors:
        raise SystemExit(f"không đọc được vector nào từ {EMB_DIR}")
    array = np.asarray(vectors, dtype="float32")
    array /= np.linalg.norm(array, axis=1, keepdims=True) + 1e-9
    return array


def grow(base: np.ndarray, target: int, noise: float, seed: int = 0) -> np.ndarray:
    """Nhân corpus lên `target` điểm, giữ hình học cục bộ của vector gốc."""

    rng = np.random.default_rng(seed)
    reps = int(np.ceil(target / len(base)))
    out = np.repeat(base, reps, axis=0)[:target].copy()
    # Bản gốc giữ nguyên để truy vấn luôn có láng giềng đúng; phần nhân bản
    # được làm nhiễu để không trùng khít.
    out[len(base):] += rng.normal(0, noise, size=out[len(base):].shape).astype("float32")
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=int, default=876)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.05)
    args = parser.parse_args()

    import faiss

    base = load_real_vectors()
    dim = base.shape[1]
    per_video = len(base) / 3.0
    target = int(per_video * args.videos)
    print(f"{len(base)} vector CLIP thật, {dim} chiều")
    print(f"mô phỏng {args.videos} video -> {target:,} vector "
          f"({target * dim * 4 / 1e9:.2f} GB float32)\n")

    corpus = grow(base, target, args.noise)
    rng = np.random.default_rng(1)
    queries = base[rng.choice(len(base), size=min(args.queries, len(base)), replace=False)]

    # Chuẩn vàng: tích vô hướng chính xác, không xấp xỉ.
    flat = faiss.IndexFlatIP(dim)
    flat.add(corpus)
    start = time.perf_counter()
    _, truth = flat.search(queries, args.k)
    exact_ms = (time.perf_counter() - start) / len(queries) * 1000
    print(f"{'index':28s} {'xây':>8} {'truy vấn':>10} {'recall@%d' % args.k:>10} {'RAM':>8}")
    print("-" * 68)
    print(f"{'FlatIP (chính xác)':28s} {'—':>8} {exact_ms:>9.1f}ms {1.0:>10.3f} "
          f"{target * dim * 4 / 1e9:>7.2f}G")

    def report(name: str, index, build_s: float, extra_bytes: int) -> None:
        start = time.perf_counter()
        _, got = index.search(queries, args.k)
        ms = (time.perf_counter() - start) / len(queries) * 1000
        hit = sum(len(set(a) & set(b)) for a, b in zip(truth, got))
        recall = hit / (len(queries) * args.k)
        print(f"{name:28s} {build_s:>7.1f}s {ms:>9.1f}ms {recall:>10.3f} "
              f"{extra_bytes / 1e9:>7.2f}G")

    for m, ef in ((16, 64), (32, 128)):
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef
        start = time.perf_counter()
        index.add(corpus)
        build = time.perf_counter() - start
        report(f"HNSW m={m} efSearch={ef}", index, build,
               target * dim * 4 + target * m * 2 * 4)

    nlist = max(int(np.sqrt(target)), 64)
    quant = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    start = time.perf_counter()
    ivf.train(corpus)
    ivf.add(corpus)
    build = time.perf_counter() - start
    for probe in (8, 32):
        ivf.nprobe = probe
        report(f"IVF nlist={nlist} nprobe={probe}", ivf, build, target * dim * 4)

    # PQ: nén 768 chiều xuống 96 byte, tức 32 lần.
    ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, 96, 8,
                             faiss.METRIC_INNER_PRODUCT)
    start = time.perf_counter()
    ivfpq.train(corpus)
    ivfpq.add(corpus)
    build = time.perf_counter() - start
    for probe in (8, 32):
        ivfpq.nprobe = probe
        report(f"IVFPQ 96B nprobe={probe}", ivfpq, build, target * 96)

    print(f"\nquét tuyến tính hiện tại ở {target:,} vector: {exact_ms:.0f}ms/truy vấn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
