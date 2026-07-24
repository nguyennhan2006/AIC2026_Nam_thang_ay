from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aic_local_search import EngineConfig, LocalHybridSearchEngine, build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    args = parser.parse_args()

    report = build_index(
        args.input_root,
        args.index_dir,
        EngineConfig(vector_backend="numpy"),
    )
    assert report.scene_count > 0
    assert report.keyframe_count > 0

    with LocalHybridSearchEngine(args.index_dir) as engine:
        lexical = engine.search("Bộ Công an", use_vector=False, top_k=5)
        assert lexical and lexical[0]["lexical_rank"] == 1

        matrix = np.load(args.index_dir / "frame_embeddings.npy")
        hybrid = engine.search("", query_vector=matrix[0], top_k=3)
        assert hybrid and hybrid[0]["vector_score"] > 0.99
    print("SMOKE TEST PASSED", report.to_dict())


if __name__ == "__main__":
    main()

