"""Chấm điểm cả bốn task theo đúng luật thi (PR-02).

`scripts/eval_kis.py` chỉ chấm KIS ở mức *scene chồng lấn interval*. Luật thi
chấm ở mức **frame**: KIS nộp `(video_id, frame_idx)`, QA nộp thêm answer,
TRAKE nộp một danh sách frame theo thứ tự và sai video là 0 điểm. Chấm ở mức
scene sẽ cho điểm cao giả tạo so với điểm thật.

Gold benchmark: `examples/AIC2026_L21_V001_queries_4tasks.jsonl` (40 query —
12 KIS, 12 VQA, 8 AVS, 8 TRAKE), schema mô tả ở
`examples/AIC2026_L21_V001_query_schema.json`.

    python -m scripts.eval_tasks --gold examples/AIC2026_L21_V001_queries_4tasks.jsonl \
        --metadata storage/exports/scenes.jsonl --tasks all

PR-07 đã thêm bốn task processor thật (`online.services.{kis,qa,trake,avs}`),
nên harness này chấm trên `response.kis/qa/trake/avs` — kết quả của processor
— chứ không phải `response.results/sequences` thô.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest
from online.domain.search_config import FusionOptions, SearchOptions
from online.domain.tasks import TaskType, normalize_task
from online.services.qa import answer_matches, normalize_answer  # noqa: F401 - re-export cho test cũ
from scripts.eval_kis import build_service

K_VALUES = (1, 5, 20, 50, 100)

# Dedup nhận số nguyên, không nhận None để nói 'không giới hạn' (None = dùng
# mặc định của task). Dùng một số đủ lớn để không policy nào chạm tới.
_UNLIMITED = 1_000_000


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    start_frame: int
    end_frame: int  # inclusive, đúng như gold ghi
    relevance_grade: int = 3
    event_id: str | None = None

    def contains(self, frame_idx: int, tolerance_frames: float = 0.0) -> bool:
        """`tolerance_frames` nới cửa sổ ra hai phía.

        TRAKE cần nó: cửa sổ trong file gold chỉ rộng 9 frame (±4) quanh mốc
        ngữ nghĩa, còn luật chấm thật chấp nhận lệch 3–6 GIÂY tuỳ độ dài
        scene. Chấm bằng ±4 frame làm r_score thấp giả tạo — keyframe được
        trích cách nhau ~123 frame nên gần như không bao giờ rơi trúng.
        """

        return (
            self.start_frame - tolerance_frames
            <= frame_idx
            <= self.end_frame + tolerance_frames
        )


@dataclass(frozen=True, slots=True)
class GoldQuery:
    query_id: str
    task: TaskType
    query: str
    video_id: str
    intervals: tuple[Interval, ...] = ()
    accepted_answers: tuple[str, ...] = ()
    steps: tuple[Interval, ...] = ()
    difficulty: str = ""


def _query_text(raw: dict) -> str:
    """Lấy câu truy vấn tiếng Việt; QA ghép mô tả sự kiện + câu hỏi."""

    for key in ("query_vi", "query_en"):
        if raw.get(key):
            return str(raw[key])
    parts = [raw.get("event_description_vi"), raw.get("question_vi") or raw.get("question_en")]
    text = " ".join(str(item) for item in parts if item)
    if not text:
        raise ValueError(f"gold {raw.get('query_id')!r} không có trường query nào dùng được")
    return text


def load_gold(path: Path, tasks: set[TaskType] | None = None) -> list[GoldQuery]:
    items: list[GoldQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        task = normalize_task(raw["task"])
        if tasks and task not in tasks:
            continue
        intervals = tuple(
            Interval(
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                relevance_grade=int(item.get("relevance_grade", 3)),
                event_id=item.get("event_id"),
            )
            for item in raw.get("target_intervals", []) + raw.get("relevant_intervals", [])
        )
        steps = tuple(
            Interval(
                start_frame=int(item["gt_start_frame"]),
                end_frame=int(item["gt_end_frame"]),
                event_id=str(item.get("event_order", "")),
            )
            for item in sorted(raw.get("events", []), key=lambda x: int(x["event_order"]))
        )
        answers = [raw["answer_canonical"]] if raw.get("answer_canonical") else []
        answers += list(raw.get("accepted_answers", []))
        items.append(
            GoldQuery(
                query_id=str(raw["query_id"]),
                task=task,
                query=_query_text(raw),
                video_id=str(raw["target_video"]),
                intervals=intervals,
                accepted_answers=tuple(dict.fromkeys(answers)),
                steps=steps,
                difficulty=str(raw.get("difficulty", "")),
            )
        )
    if not items:
        raise SystemExit(f"không có gold query nào khớp bộ lọc trong {path}")
    return items


# --------------------------------------------------------------------------
# Chấm điểm
# --------------------------------------------------------------------------
# normalize_answer/answer_matches giờ sống ở online.services.qa (import ở
# đầu file) — dùng chung với QaProcessor.verify_answer và
# online/competition/scorer.py, tránh hai định nghĩa "đúng" khác nhau.


@dataclass(slots=True)
class RankedMetrics:
    hits_at: dict[int, int] = field(default_factory=lambda: {k: 0 for k in K_VALUES})
    reciprocal_ranks: list[float] = field(default_factory=list)
    video_hits_at_100: int = 0
    total: int = 0

    def add(self, rank: int | None, video_rank: int | None) -> None:
        self.total += 1
        for k in K_VALUES:
            if rank is not None and rank <= k:
                self.hits_at[k] += 1
        self.reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if video_rank is not None and video_rank <= 100:
            self.video_hits_at_100 += 1

    def as_row(self) -> dict[str, float]:
        total = max(self.total, 1)
        row = {f"R@{k}": self.hits_at[k] / total for k in K_VALUES}
        row["MRR"] = sum(self.reciprocal_ranks) / total
        row["vidR@100"] = self.video_hits_at_100 / total
        return row


def _first_rank(results, predicate) -> int | None:
    return next((index for index, hit in enumerate(results, start=1) if predicate(hit)), None)


def _frame_hit(hit, gold: GoldQuery) -> bool:
    """Đúng luật: cùng video VÀ frame nộp nằm trong một interval GT."""

    return hit.video_id == gold.video_id and any(
        interval.contains(hit.best_frame_idx) for interval in gold.intervals
    )


def ndcg_at_k(grades: list[int], ideal: list[int], k: int) -> float:
    def dcg(values: list[int]) -> float:
        return sum(
            (2 ** value - 1) / math.log2(position + 1)
            for position, value in enumerate(values[:k], start=1)
        )

    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best else 0.0


@dataclass(frozen=True, slots=True)
class WindowPolicy:
    """Cửa sổ chấp nhận của MỘT step TRAKE.

    Luật chấm: tổng cửa sổ 2–7 giây tuỳ độ dài scene, KHÔNG phải một hằng số.
    Cửa sổ ±4 frame ghi trong file gold là mốc ngữ nghĩa của khoảnh khắc, không
    phải dung sai chấm — nhầm hai thứ này từng dẫn tới một chẩn đoán sai
    (xem docs/20_EXPERIMENT_LOG.md § TRAKE).
    """

    min_sec: float = 2.0
    max_sec: float = 7.0
    ratio: float = 0.5

    def half_window_sec(self, scene_duration_sec: float | None) -> float:
        if scene_duration_sec is None:
            width = self.min_sec
        else:
            width = min(max(scene_duration_sec * self.ratio, self.min_sec), self.max_sec)
        return width / 2.0


def resolve_step_window(
    step: "Interval",
    scenes: list,
    fps: float,
    policy: WindowPolicy,
) -> tuple[float, str]:
    """Trả `(tolerance_frames_mỗi_phía, nguồn_interval)`.

    Thứ tự ưu tiên:
      1. interval event tường minh trong gold — nếu nó rộng hơn mốc ngữ nghĩa
         (>1s) thì đó là dung sai thật, dùng thẳng;
      2. scene chứa mốc đó — suy cửa sổ từ độ dài scene;
      3. fallback tối thiểu.
    """

    gold_width_sec = (step.end_frame - step.start_frame + 1) / max(fps, 1e-9)
    if gold_width_sec > 1.0:
        return gold_width_sec / 2.0 * fps, "explicit_gold_interval"

    centre = (step.start_frame + step.end_frame) // 2
    scene = next(
        (item for item in scenes if item.start_frame <= centre < item.end_frame_exclusive),
        None,
    )
    if scene is not None:
        duration = scene.end_sec - scene.start_sec
        return policy.half_window_sec(duration) * fps, "derived_scene_window"
    return policy.half_window_sec(None) * fps, "fallback_min_window"


def load_fps(metadata: Path, default: float = 30.0) -> float:
    """FPS thật của video, đọc từ `videos.jsonl` cạnh file scene.

    KHÔNG giả định 30 fps: dung sai của TRAKE tính bằng giây nên quy đổi sai
    fps là sai thẳng vào điểm.
    """

    path = metadata.with_name("videos.jsonl")
    if not path.exists():
        return default
    rates = [
        float(json.loads(line)["fps"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "fps" in json.loads(line)
    ]
    return rates[0] if rates else default


def _search_options(args: argparse.Namespace) -> SearchOptions | None:
    """Chỉ đặt option khi người dùng thật sự yêu cầu — để mặc định của harness
    trùng khít mặc định production, không âm thầm đo một cấu hình khác."""

    if args.max_per_video is None:
        return None
    return SearchOptions(fusion=FusionOptions(max_results_per_video=_UNLIMITED if args.max_per_video == 0 else args.max_per_video))


async def evaluate(
    gold: list[GoldQuery], repository: JsonlSceneRepository, args: argparse.Namespace
) -> dict:
    service = await build_service(
        "fusion",
        repository,
        backend=args.backend,
        use_rules=args.use_rules,
        use_expansion=args.use_expansion,
        use_query_prep=args.use_query_prep,
        use_rerank=args.use_rerank,
        candidate_limit=max(args.top_k, 100),
    )
    # Cần scene để suy cửa sổ chấp nhận của TRAKE từ độ dài scene.
    scenes = await repository.all()
    kis = RankedMetrics()
    qa_evidence = RankedMetrics()
    qa_answer_correct = 0
    qa_joint_correct = 0
    qa_total = 0
    trake_video_correct = 0
    trake_r_scores: list[float] = []
    trake_total = 0
    avs_ndcg: list[float] = []
    avs_precision: list[float] = []
    avs_event_coverage: list[float] = []
    per_query: list[dict] = []

    for item in gold:
        response = await service.search(
            SearchRequest(
                query=item.query, task=item.task, top_k=args.top_k,
                search_options=_search_options(args),
            )
        )
        record: dict = {"query_id": item.query_id, "task": item.task.value}

        if item.task == TaskType.TEXTUAL_KIS:
            # KIS chấm trên response.kis (PR-07: KisProcessor.rank), không phải
            # response.results thô — signature/safe-frame có thể xếp hạng khác
            # với điểm fusion thuần.
            rank = next(
                (
                    index
                    for index, row in enumerate(response.kis, start=1)
                    if row.video_id == item.video_id
                    and any(interval.contains(row.frame_idx) for interval in item.intervals)
                ),
                None,
            )
            video_rank = next(
                (index for index, row in enumerate(response.kis, start=1)
                 if row.video_id == item.video_id),
                None,
            )
            record["first_frame_hit_rank"] = rank
            record["first_video_rank"] = video_rank
            kis.add(rank, video_rank)

        elif item.task == TaskType.QA:
            qa_total += 1
            # Đúng bộ ba (video, frame trong interval, answer đúng) — khớp
            # nguyên văn luật chấm QA, không chỉ evidence rank.
            rank = next(
                (
                    index
                    for index, row in enumerate(response.qa, start=1)
                    if row.video_id == item.video_id
                    and any(interval.contains(row.frame_idx) for interval in item.intervals)
                ),
                None,
            )
            video_rank = next(
                (index for index, row in enumerate(response.qa, start=1)
                 if row.video_id == item.video_id),
                None,
            )
            qa_evidence.add(rank, video_rank)
            answer_ok = any(
                row.video_id == item.video_id
                and any(interval.contains(row.frame_idx) for interval in item.intervals)
                and answer_matches(row.answer, item.accepted_answers)
                for row in response.qa
            )
            qa_answer_correct += int(answer_ok)
            qa_joint_correct += int(answer_ok and rank == 1)
            record["answer_correct"] = answer_ok
            record["predicted_answers"] = [row.answer for row in response.qa[:3]]

        elif item.task == TaskType.TRAKE:
            trake_total += 1
            # response.trake (PR-07: TrakeProcessor, video-first) thay cho
            # response.sequences (bản nối scene cũ, không khóa video).
            best = response.trake[0] if response.trake else None
            if best is None or best.video_id != item.video_id:
                trake_r_scores.append(0.0)
                record["r_score"] = 0.0
                record["video_correct"] = bool(best and best.video_id == item.video_id)
            else:
                trake_video_correct += 1
                frame_ids = best.frame_ids
                policy = WindowPolicy(
                    min_sec=args.trake_window_min_sec,
                    max_sec=args.trake_window_max_sec,
                    ratio=args.trake_window_ratio,
                )
                fps = load_fps(args.metadata)
                windows = [resolve_step_window(gt, scenes, fps, policy) for gt in item.steps]
                hits = sum(
                    1
                    for step, step_gt, (tol, _src) in zip(frame_ids, item.steps, windows, strict=False)
                    if step_gt.contains(step, tol)
                )
                record["windows"] = [
                    {
                        "window_width_sec": round(tol * 2 / fps, 3),
                        "tolerance_before_sec": round(tol / fps, 3),
                        "tolerance_after_sec": round(tol / fps, 3),
                        "interval_source": src,
                    }
                    for tol, src in windows
                ]
                # Sai video = 0; đúng video = tỷ lệ step rơi đúng cửa sổ GT.
                r_score = hits / len(item.steps) if item.steps else 0.0
                trake_r_scores.append(r_score)
                record["r_score"] = r_score
                record["video_correct"] = True
                record["predicted_frames"] = frame_ids
                record["expected_steps"] = len(item.steps)

        elif item.task == TaskType.AVS:
            # response.avs (PR-07: AvsProcessor) đã tự chấm relevance_grade
            # 0-3 theo inclusion/exclusion của chính nó; ở đây chỉ đối chiếu
            # với gold để tính nDCG/precision/coverage thật, KHÔNG dùng lại
            # relevance_grade do processor tự gán (đó là điểm nó tự tin, không
            # phải điểm đúng theo gold).
            gold_grade_by_frame = [
                (interval, interval.relevance_grade) for interval in item.intervals
            ]

            def _gold_grade(row) -> int:
                if row.video_id != item.video_id:
                    return 0
                return max(
                    (
                        grade
                        for interval, grade in gold_grade_by_frame
                        if row.best_frame_idx is not None and interval.contains(row.best_frame_idx)
                    ),
                    default=0,
                )

            grades = [_gold_grade(row) for row in response.avs]
            ideal = [interval.relevance_grade for interval in item.intervals]
            ndcg = ndcg_at_k(grades, ideal, args.top_k)
            precision = sum(1 for value in grades[: args.top_k] if value > 0) / max(
                min(args.top_k, len(grades)), 1
            )
            covered = {
                interval.event_id
                for row in response.avs
                for interval, _grade in gold_grade_by_frame
                if row.video_id == item.video_id
                and row.best_frame_idx is not None
                and interval.contains(row.best_frame_idx)
            }
            expected_events = {interval.event_id for interval in item.intervals}
            avs_ndcg.append(ndcg)
            avs_precision.append(precision)
            avs_event_coverage.append(
                len(covered) / len(expected_events) if expected_events else 0.0
            )
            record.update({"ndcg": ndcg, "precision": precision, "result_count": len(response.avs)})

        per_query.append(record)
        if args.verbose:
            print(f"  {record}")

    summary: dict = {"per_query": per_query}
    if kis.total:
        summary["TEXTUAL_KIS"] = kis.as_row() | {"queries": kis.total}
    if qa_total:
        summary["QA"] = qa_evidence.as_row() | {
            "queries": qa_total,
            "answer_accuracy": qa_answer_correct / qa_total,
            "joint_top1": qa_joint_correct / qa_total,
        }
    if trake_total:
        summary["TRAKE"] = {
            "queries": trake_total,
            "correct_video_rate": trake_video_correct / trake_total,
            "mean_r_score": sum(trake_r_scores) / trake_total,
            "complete_chain_rate": sum(1 for value in trake_r_scores if value == 1.0) / trake_total,
        }
    if avs_ndcg:
        summary["AVS"] = {
            "queries": len(avs_ndcg),
            f"nDCG@{args.top_k}": sum(avs_ndcg) / len(avs_ndcg),
            f"P@{args.top_k}": sum(avs_precision) / len(avs_precision),
            "event_coverage": sum(avs_event_coverage) / len(avs_event_coverage),
        }
    return summary


def print_summary(summary: dict) -> None:
    for task in ("TEXTUAL_KIS", "QA", "TRAKE", "AVS"):
        row = summary.get(task)
        if not row:
            continue
        print(f"\n=== {task} ({row['queries']} query) ===")
        for key, value in row.items():
            if key == "queries":
                continue
            print(f"  {key:22s} {value:.3f}" if isinstance(value, float) else f"  {key:22s} {value}")


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Chấm 4 task AIC 2026 ở mức frame")
    parser.add_argument(
        "--gold", type=Path, default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl")
    )
    parser.add_argument("--metadata", type=Path, default=Path("storage/exports/scenes.jsonl"))
    parser.add_argument("--tasks", default="all", help="all | TEXTUAL_KIS,QA,TRAKE,AVS")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--backend", choices=("local", "qdrant"), default="local")
    parser.add_argument("--use-rules", action="store_true")
    parser.add_argument("--use-expansion", action="store_true")
    parser.add_argument("--use-query-prep", action="store_true")
    parser.add_argument("--trake-window-min-sec", type=float, default=2.0,
                         help="tổng cửa sổ chấp nhận tối thiểu của một step TRAKE (giây)")
    parser.add_argument("--trake-window-max-sec", type=float, default=7.0,
                         help="tổng cửa sổ chấp nhận tối đa của một step TRAKE (giây)")
    parser.add_argument("--trake-window-ratio", type=float, default=0.5,
                         help="cửa sổ = clamp(độ_dài_scene * ratio, min, max)")
    parser.add_argument("--max-per-video", type=int, default=None,
                         help="ghi đè fusion.max_results_per_video; 0 = BỎ HẲN giới hạn. "
                              "Lưu ý: KHÔNG truyền cờ này không có nghĩa là không giới hạn — "
                              "khi đó dedup dùng mặc định của task (KIS = 5/video). BẮT BUỘC "
                              "đặt khi dataset chỉ có "
                              "1 video: dedup KIS mặc định giữ 5 kết quả/video nên R@20/50/100 "
                              "sẽ bằng hệt R@5 và mọi số trên K=5 là vô nghĩa")
    parser.add_argument("--use-rerank", action="store_true",
                         help="bật text rerank + QA answer generation qua FPT thật (xem "
                              "scripts/eval_kis.py::build_service) — không có cờ này, answer_accuracy "
                              "chỉ đo rule-based ANSWER_TOOLS, không phản ánh FPT QA LLM")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    selected = (
        None
        if args.tasks == "all"
        else {normalize_task(item) for item in args.tasks.split(",") if item.strip()}
    )
    gold = load_gold(args.gold, selected)
    repository = await JsonlSceneRepository.load(args.metadata)
    scenes = await repository.all()
    gold_videos = {item.video_id for item in gold}
    indexed_videos = {scene.video_id for scene in scenes}
    if not gold_videos & indexed_videos:
        print(
            f"WARNING: gold nói về {sorted(gold_videos)} nhưng metadata chỉ có "
            f"{sorted(indexed_videos)} — mọi điểm sẽ bằng 0. Chạy "
            "`python -m offline assemble` cho đúng video trước.",
            file=sys.stderr,
        )
    print(f"gold={len(gold)} query  scenes={len(scenes)}  backend={args.backend}  top_k={args.top_k}")
    summary = await evaluate(gold, repository, args)
    print_summary(summary)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nchi tiết -> {args.json_out}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
