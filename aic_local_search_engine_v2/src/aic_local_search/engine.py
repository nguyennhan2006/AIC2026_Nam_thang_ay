from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EngineConfig
from .fusion import reciprocal_rank_fusion
from .planner import plan_query
from .storage import (
    connect_database,
    fetch_frames_by_vector_rows,
    fetch_scenes,
    fetch_scenes_by_vector_rows,
    representative_frame,
    search_branch,
    search_event_branch,
)
from .vector_index import OpenClipTextEncoder, VectorIndex


class LocalHybridSearchEngine:
    """Local scene/frame retrieval with six parallel branches and RRF."""

    def __init__(
        self,
        index_dir: str | Path,
        device: str | None = None,
        asset_root: str | Path | None = None,
    ):
        self.index_dir = Path(index_dir).expanduser().resolve()
        manifest_path = self.index_dir / "index_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing index manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", 0)) != 2:
            raise ValueError("This engine expects search index schema_version=2; rebuild the index.")
        self.config = EngineConfig.from_dict(self.manifest.get("config", {}))
        self.connection = connect_database(
            self.index_dir / self.manifest["files"]["database"], readonly=True
        )
        self.scene_vector_index = VectorIndex(
            self.index_dir, self.manifest["scene_vector_index"]
        )
        frame_manifest = self.manifest.get("frame_vector_index")
        self.frame_vector_index = VectorIndex(self.index_dir, frame_manifest) if frame_manifest else None
        self.embedding_model = str(self.manifest["scene_embedding_model"])
        self.device = device
        self.asset_root = Path(asset_root).expanduser().resolve() if asset_root else None
        self._text_encoder: OpenClipTextEncoder | None = None
        self._text_encoder_error: str | None = None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LocalHybridSearchEngine":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def encode_visual_query(self, text: str) -> np.ndarray:
        if self._text_encoder_error is not None:
            raise RuntimeError(self._text_encoder_error)
        if self._text_encoder is None:
            try:
                self._text_encoder = OpenClipTextEncoder(self.embedding_model, self.device)
            except RuntimeError as exc:
                self._text_encoder_error = str(exc)
                raise
        return self._text_encoder.encode(text)

    def _asset_path(self, value: str) -> str:
        if not value or Path(value).is_absolute() or self.asset_root is None:
            return value
        direct = self.asset_root / value
        if direct.exists():
            return str(direct.resolve())
        matches = list(self.asset_root.rglob(Path(value).name))
        return str(matches[0].resolve()) if len(matches) == 1 else value

    def _resolve_frame(self, frame: dict | None) -> dict | None:
        if frame is None:
            return None
        output = dict(frame)
        output["image_path"] = self._asset_path(output.get("image_path", ""))
        return output

    def _scene_vector_candidates(
        self,
        query_vector: np.ndarray,
        limit: int,
        video_id: str | None,
        start_sec: float | None,
        end_sec: float | None,
    ) -> list[dict]:
        hits = self.scene_vector_index.search(query_vector, max(limit * 2, limit))
        scenes = fetch_scenes_by_vector_rows(self.connection, [row for row, _ in hits])
        output = []
        for row, score in hits:
            scene = scenes.get(row)
            if scene is None:
                continue
            if self.config.exclude_invalid and scene["quality_status"] == "invalid":
                continue
            if video_id and scene["video_id"] != video_id:
                continue
            if start_sec is not None and scene["end_sec"] < start_sec:
                continue
            if end_sec is not None and scene["start_sec"] > end_sec:
                continue
            output.append(
                {"scene_id": scene["scene_id"], "branch": "scene_vector", "score": float(score)}
            )
            if len(output) >= limit:
                break
        return output

    def _frame_vector_candidates(
        self,
        query_vector: np.ndarray,
        limit: int,
        video_id: str | None,
        start_sec: float | None,
        end_sec: float | None,
    ) -> list[dict]:
        if self.frame_vector_index is None:
            return []
        hits = self.frame_vector_index.search(query_vector, max(limit * 5, limit))
        frames = fetch_frames_by_vector_rows(self.connection, [row for row, _ in hits])
        scene_ids = list({frame["scene_id"] for frame in frames.values()})
        scenes = fetch_scenes(self.connection, scene_ids)
        best: dict[str, dict] = {}
        for row, score in hits:
            frame = frames.get(row)
            if frame is None:
                continue
            scene = scenes.get(frame["scene_id"])
            if scene is None:
                continue
            if self.config.exclude_invalid and scene["quality_status"] == "invalid":
                continue
            if video_id and scene["video_id"] != video_id:
                continue
            if start_sec is not None and scene["end_sec"] < start_sec:
                continue
            if end_sec is not None and scene["start_sec"] > end_sec:
                continue
            previous = best.get(frame["scene_id"])
            if previous is None or score > previous["score"]:
                best[frame["scene_id"]] = {
                    "scene_id": frame["scene_id"],
                    "branch": "frame_vector",
                    "score": float(score),
                    "best_frame": frame,
                }
        return sorted(best.values(), key=lambda item: -item["score"])[:limit]

    def search(
        self,
        text_query: str,
        *,
        visual_query: str | None = None,
        query_vector: np.ndarray | None = None,
        use_vector: bool = True,
        task: str = "auto",
        top_k: int = 10,
        video_id: str | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        match_all_terms: bool = False,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        plan = plan_query(text_query, self.config, task=task)
        branch_results: dict[str, list[dict]] = {}
        for branch in ("semantic", "ocr", "speech", "tags"):
            branch_results[branch] = search_branch(
                self.connection,
                branch,
                text_query,
                self.config.lexical_candidates,
                self.config,
                video_id=video_id,
                start_sec=start_sec,
                end_sec=end_sec,
                match_all=match_all_terms,
            )
        branch_results["event"] = search_event_branch(
            self.connection,
            text_query,
            self.config.lexical_candidates,
            self.config,
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        vector_status = "disabled"
        if use_vector:
            vector_status = "requested"
            if query_vector is None:
                vector_text = visual_query if visual_query is not None else text_query
                if vector_text.strip():
                    try:
                        query_vector = self.encode_visual_query(vector_text)
                    except RuntimeError as exc:
                        vector_status = "skipped: " + str(exc).splitlines()[0]
            if query_vector is not None:
                vector_status = "used"
                branch_results["scene_vector"] = self._scene_vector_candidates(
                    query_vector, self.config.vector_candidates, video_id, start_sec, end_sec
                )
                branch_results["frame_vector"] = self._frame_vector_candidates(
                    query_vector, self.config.vector_candidates, video_id, start_sec, end_sec
                )
        branch_results.setdefault("scene_vector", [])
        branch_results.setdefault("frame_vector", [])

        ranked_lists: list[tuple[list[str], float]] = []
        all_scene_ids: set[str] = set()
        for branch, results in branch_results.items():
            ids = [item["scene_id"] for item in results]
            all_scene_ids.update(ids)
            if ids and plan.branch_weights.get(branch, 0.0) > 0:
                ranked_lists.append((ids, plan.branch_weights[branch]))
        scenes = fetch_scenes(self.connection, sorted(all_scene_ids))
        multipliers = {scene_id: float(scene["quality_penalty"]) for scene_id, scene in scenes.items()}
        fused = reciprocal_rank_fusion(
            ranked_lists, self.config.rrf_k, item_multipliers=multipliers
        )[:top_k]
        if not fused:
            return []

        by_branch = {
            branch: {item["scene_id"]: item for item in results}
            for branch, results in branch_results.items()
        }
        ranks = {
            branch: {item["scene_id"]: rank for rank, item in enumerate(results, 1)}
            for branch, results in branch_results.items()
        }
        output: list[dict[str, Any]] = []
        for final_rank, (scene_id, rrf_score) in enumerate(fused, 1):
            scene = scenes[scene_id]
            frame_item = by_branch["frame_vector"].get(scene_id, {})
            best_frame = frame_item.get("best_frame") or representative_frame(self.connection, scene_id)
            branch_ranks = {
                branch: branch_rank[scene_id]
                for branch, branch_rank in ranks.items()
                if scene_id in branch_rank
            }
            branch_scores = {
                branch: by_branch[branch][scene_id]["score"]
                for branch in by_branch
                if scene_id in by_branch[branch]
            }
            snippets = {
                branch: by_branch[branch][scene_id].get("snippet", "")
                for branch in ("semantic", "ocr", "speech", "tags", "event")
                if scene_id in by_branch[branch] and by_branch[branch][scene_id].get("snippet")
            }
            matched_event = by_branch["event"].get(scene_id, {}).get("matched_event")
            lexical_ranks = [rank for branch, rank in branch_ranks.items() if branch not in {"scene_vector", "frame_vector"}]
            vector_ranks = [rank for branch, rank in branch_ranks.items() if branch in {"scene_vector", "frame_vector"}]
            output.append(
                {
                    "rank": final_rank,
                    "scene_id": scene_id,
                    "video_id": scene["video_id"],
                    "scene_no": scene["scene_no"],
                    "start_sec": scene["start_sec"],
                    "end_sec": scene["end_sec"],
                    "rrf_score": float(rrf_score),
                    "query_plan": {
                        "task": plan.task,
                        "hints": plan.hints,
                        "vector_status": vector_status,
                    },
                    "branch_ranks": branch_ranks,
                    "branch_scores": branch_scores,
                    "lexical_rank": min(lexical_ranks) if lexical_ranks else None,
                    "vector_rank": min(vector_ranks) if vector_ranks else None,
                    "snippets": snippets,
                    "matched_event": matched_event,
                    "caption_vi": scene["caption_vi"],
                    "caption_en": scene["caption_en"],
                    "ocr_text": scene["ocr_text"],
                    "transcript": scene["transcript"],
                    "keywords": scene["keywords"],
                    "entities": scene["entities"],
                    "actions": scene["actions"],
                    "quality_status": scene["quality_status"],
                    "quality_penalty": scene["quality_penalty"],
                    "quality_errors": scene["quality_errors"],
                    "clip_path": self._asset_path(scene["clip_path"]),
                    "best_frame": self._resolve_frame(best_frame),
                }
            )
        return output

    def search_sequence(
        self,
        steps: list[str],
        *,
        per_step_k: int = 30,
        top_k: int = 5,
        max_gap_sec: float | None = 120.0,
        use_vector: bool = True,
        visual_steps: list[str] | None = None,
        beam_width: int = 200,
    ) -> list[dict[str, Any]]:
        if len(steps) < 2:
            raise ValueError("Temporal search needs at least two ordered steps")
        if visual_steps is not None and len(visual_steps) != len(steps):
            raise ValueError("visual_steps must have the same length as steps")
        candidates = [
            self.search(
                step,
                visual_query=visual_steps[index] if visual_steps else None,
                use_vector=use_vector,
                task="temporal",
                top_k=per_step_k,
            )
            for index, step in enumerate(steps)
        ]
        if any(not group for group in candidates):
            return []

        def anchor(item: dict) -> tuple[float, int | None]:
            event = item.get("matched_event")
            if event:
                return float(event["absolute_start_sec"]), int(event["event_order"])
            return float(item["start_sec"]), None

        beams: list[tuple[float, list[dict]]] = [
            (item["rrf_score"], [item]) for item in candidates[0]
        ]
        for group in candidates[1:]:
            next_beams: list[tuple[float, list[dict]]] = []
            for score, sequence in beams:
                previous = sequence[-1]
                previous_time, previous_event_order = anchor(previous)
                for item in group:
                    if item["video_id"] != previous["video_id"]:
                        continue
                    current_time, current_event_order = anchor(item)
                    same_scene_event_order = (
                        item["scene_id"] == previous["scene_id"]
                        and previous_event_order is not None
                        and current_event_order is not None
                        and current_event_order > previous_event_order
                    )
                    later_scene = item["scene_no"] > previous["scene_no"]
                    if not (same_scene_event_order or later_scene):
                        continue
                    gap = max(0.0, current_time - previous_time)
                    if max_gap_sec is not None and gap > max_gap_sec:
                        continue
                    next_beams.append(
                        (score + item["rrf_score"] - gap * 0.0001, [*sequence, item])
                    )
            beams = sorted(next_beams, key=lambda pair: -pair[0])[:beam_width]
            if not beams:
                return []
        return [
            {
                "rank": rank,
                "score": float(score),
                "video_id": sequence[0]["video_id"],
                "start_sec": anchor(sequence[0])[0],
                "end_sec": anchor(sequence[-1])[0],
                "scene_ids": [item["scene_id"] for item in sequence],
                "steps": sequence,
            }
            for rank, (score, sequence) in enumerate(beams[:top_k], 1)
        ]
