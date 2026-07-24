from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "08_local_multibranch_search.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# 08 — Local Multi-branch Search (05 + 06 + Metadata)

Notebook này build search engine local từ output 05, output 06 và metadata:

- SQLite FTS5/BM25 cho semantic, OCR, speech, tag và temporal event;
- FAISS HNSW cho scene/frame vector;
- query planner + weighted RRF + quality penalty;
- không cần Elasticsearch, Docker hoặc PyTorch.

PyTorch chỉ là tùy chọn nếu muốn encode text mới sang OpenCLIP vector."""
    ),
    markdown(
        """## 1. Mapping kỹ thuật Elastic → local

| Elastic | Local |
|---|---|
| multi-field BM25 | 5 bảng FTS5 riêng |
| `dense_vector` / HNSW | 2 FAISS HNSW index |
| metadata + range filter | SQLite tables/indexes |
| RRF retriever | weighted RRF Python |
| ingest quality gate | output 06 + status output 05 |
| temporal retrieval | event FTS + beam search |"""
    ),
    code(
        """# 2. Tìm project; hỗ trợ cả thư mục và engine ZIP trên Kaggle
from pathlib import Path
import io, json, os, shutil, sys, zipfile
import numpy as np

def find_project_root():
    candidates = [Path.cwd(), Path.cwd().parent]
    if Path('/kaggle/input').exists():
        candidates += [p.parent for p in Path('/kaggle/input').rglob('pyproject.toml')]
        archives = list(Path('/kaggle/input').rglob('aic_local_search_engine_v2.zip'))
        if archives:
            destination = Path('/kaggle/working/aic_local_search_engine_v2')
            if not (destination / 'pyproject.toml').exists():
                destination.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archives[0]) as zf:
                    zf.extractall(destination)
            candidates.append(destination)
    for candidate in candidates:
        if (candidate / 'src' / 'aic_local_search').exists():
            return candidate.resolve()
        nested = candidate / 'aic_local_search'
        if (nested / 'src' / 'aic_local_search').exists():
            return nested.resolve()
    raise FileNotFoundError('Không tìm thấy source aic_local_search v2.')

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from aic_local_search import EngineConfig, LocalHybridSearchEngine, build_index
print('PROJECT_ROOT =', PROJECT_ROOT)"""
    ),
    markdown(
        """## 3. Input/output

`INPUT_ROOT` chứa ZIP hoặc thư mục output 05, output 06 và metadata. Metadata
rich (`scene_docs/frame_docs`) đã đủ vector scene + frame. Metadata compact
(`scene_metadata`) đủ scene vector; output 01 là tùy chọn để thêm frame vector."""
    ),
    code(
        """IS_KAGGLE = Path('/kaggle/working').exists()
default_input = Path('/kaggle/input') if IS_KAGGLE else PROJECT_ROOT / 'data'
default_index = Path('/kaggle/working/08_local_search_index') if IS_KAGGLE else PROJECT_ROOT / 'artifacts' / '08_local_search_index'

INPUT_ROOT = Path(os.environ.get('AIC_INPUT_ROOT', default_input)).expanduser().resolve()
INDEX_DIR = Path(os.environ.get('AIC_INDEX_DIR', default_index)).expanduser().resolve()

print('INPUT_ROOT =', INPUT_ROOT)
print('INDEX_DIR  =', INDEX_DIR)
if not INPUT_ROOT.exists():
    raise FileNotFoundError(f'Chưa có input: {INPUT_ROOT}')"""
    ),
    markdown(
        """## 4. Build index

`auto` dùng FAISS khi đã cài. `keep_numpy_fallback=False` giữ output tối giản:
không lưu lại ma trận Numpy khi FAISS đã build thành công."""
    ),
    code(
        """config = EngineConfig(
    vector_backend='auto',
    hnsw_m=32,
    hnsw_ef_construction=200,
    hnsw_ef_search=64,
    lexical_candidates=100,
    vector_candidates=100,
    needs_review_penalty=0.75,
    exclude_invalid=True,
    keep_numpy_fallback=False,
)

report = build_index(INPUT_ROOT, INDEX_DIR, config)
print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))"""
    ),
    code(
        """# 5. Kiểm tra manifest và đúng số file cần thiết
manifest = json.loads((INDEX_DIR / 'index_manifest.json').read_text(encoding='utf-8'))
summary = {
    'stats': manifest['stats'],
    'scene_embedding_model': manifest['scene_embedding_model'],
    'scene_vector_index': manifest['scene_vector_index'],
    'frame_vector_index': manifest['frame_vector_index'],
    'warnings': manifest['warnings'],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('Index files:', sorted(p.name for p in INDEX_DIR.iterdir() if p.is_file()))"""
    ),
    markdown(
        """## 6. Multi-branch search — không cần PyTorch

Kết quả cho biết scene xuất hiện ở nhánh nào qua `branch_ranks`. Query planner
tự nhận biết OCR/speech/temporal hints."""
    ),
    code(
        """def show_hits(hits):
    for hit in hits:
        print({
            'rank': hit['rank'],
            'scene_id': hit['scene_id'],
            'time': [hit['start_sec'], hit['end_sec']],
            'branches': hit['branch_ranks'],
            'quality': hit['quality_status'],
            'rrf': round(hit['rrf_score'], 6),
        })

with LocalHybridSearchEngine(INDEX_DIR, asset_root=INPUT_ROOT) as engine:
    text_hits = engine.search(
        'Công viên địa chất Lạng Sơn được UNESCO công nhận',
        use_vector=False,
        task='scene',
        top_k=5,
    )
show_hits(text_hits)"""
    ),
    code(
        """# OCR-oriented query: planner tự boost nhánh OCR
with LocalHybridSearchEngine(INDEX_DIR, asset_root=INPUT_ROOT) as engine:
    ocr_hits = engine.search(
        'chữ CỤC THÚ Y dấu vuông trên màn hình',
        use_vector=False,
        task='frame',
        top_k=5,
    )
show_hits(ocr_hits)"""
    ),
    markdown(
        """## 7. Kiểm tra FAISS bằng vector đã có — vẫn không cần PyTorch

Cell đọc vector đầu tiên ngay từ metadata ZIP/thư mục và dùng nó làm query.
Scene tương ứng phải đứng đầu với cosine gần 1."""
    ),
    code(
        """def load_first_scene_vector(root: Path) -> np.ndarray:
    names = {'scene_visual_embeddings.npy', 'scene_embeddings.npy'}
    for name in names:
        direct = next(root.rglob(name), None)
        if direct:
            return np.asarray(np.load(direct, allow_pickle=False)[0], dtype=np.float32)
    for archive in root.rglob('*.zip'):
        try:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    if Path(member).name in names:
                        return np.asarray(np.load(io.BytesIO(zf.read(member)), allow_pickle=False)[0], dtype=np.float32)
        except zipfile.BadZipFile:
            pass
    raise FileNotFoundError('Không tìm thấy scene vector trong input')

query_vector = load_first_scene_vector(INPUT_ROOT)
with LocalHybridSearchEngine(INDEX_DIR, asset_root=INPUT_ROOT) as engine:
    vector_hits = engine.search('', query_vector=query_vector, task='scene', top_k=3)

show_hits(vector_hits)
assert vector_hits and vector_hits[0]['branch_scores']['scene_vector'] > 0.99
print('Scene vector mapping: PASSED')"""
    ),
    markdown("""## 8. Temporal sequence search"""),
    code(
        """with LocalHybridSearchEngine(INDEX_DIR, asset_root=INPUT_ROOT) as engine:
    sequences = engine.search_sequence(
        ['Cục Thú y nói về thịt heo', 'Lạng Sơn nhận bằng UNESCO'],
        use_vector=False,
        per_step_k=30,
        top_k=5,
        max_gap_sec=180,
    )

for item in sequences:
    print(item['rank'], item['scene_ids'], round(item['score'], 6))"""
    ),
    markdown(
        """## 9. Đóng gói đúng index

Cache giải nén nằm ngoài `INDEX_DIR`, vì vậy ZIP cuối chỉ chứa database, FAISS
index (hoặc Numpy fallback) và manifest."""
    ),
    code(
        """zip_base = Path('/kaggle/working/08_local_search_index') if IS_KAGGLE else INDEX_DIR.parent / '08_local_search_index'
zip_path = Path(shutil.make_archive(str(zip_base), 'zip', root_dir=INDEX_DIR))
print('ZIP:', zip_path, f'({zip_path.stat().st_size / 1024 / 1024:.2f} MiB)')"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
