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
import time

from online.adapters.json_metadata import JsonlSceneRepository
from online.competition.gold_text import resolve_gold_text
from online.domain.models import SearchRequest
from online.domain.search_config import (
    BranchRuntimeOptions,
    FusionOptions,
    SearchOptions,
    TemporalOptions,
)
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


# DIAG-01: hàm này từng là bản private của riêng file này. Nó vẫn đúng, nhưng
# vì không ai ngoài đây dùng được nên một script chẩn đoán đã tự viết bản rút
# gọn chỉ đọc `query_vi` — và bỏ sót lặng lẽ toàn bộ 36 truy vấn QA. Nay dùng
# chung `online.competition.gold_text`.
_query_text = resolve_gold_text


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


def _frame_oracle_coverage(gold: list[GoldQuery], args) -> float:
    """Tỉ lệ bước TRAKE mà CORPUS có ít nhất một keyframe trong cửa sổ GT.

    Đây là TRẦN của `mean_r_score`: không thuật toán chọn frame nào vượt được
    một bước mà corpus không có ứng viên hợp lệ nào.

    Tính ở mức corpus (đọc thẳng keyframes.jsonl) chứ không ở mức candidate
    của từng truy vấn — hai thứ khác nhau, và cái này trả lời đúng câu "dữ liệu
    có cho phép ăn điểm bước này không", độc lập với retrieval.

    Đo được ở đây: 0.800 với cửa sổ ±2s, 1.000 với ±4s — trong khi mean_r_score
    chỉ 0.263. Khoảng cách đó chính là thứ `frame_selection_accuracy` đo.
    """

    import bisect
    import json as _json

    keyframes_path = args.metadata.with_name("keyframes.jsonl")
    if not keyframes_path.exists():
        return 0.0
    by_video: dict[str, list[int]] = {}
    with keyframes_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = _json.loads(line)
            by_video.setdefault(row["video_id"], []).append(row["frame_idx"])
    for frames in by_video.values():
        frames.sort()

    fps = load_fps(args.metadata)
    half_window = int(args.trake_window_min_sec * fps)
    total = covered = 0
    for item in gold:
        if item.task != TaskType.TRAKE:
            continue
        frames = by_video.get(item.video_id, [])
        for step in item.steps:
            total += 1
            if not frames:
                continue
            target = (step.start_frame + step.end_frame) // 2
            index = bisect.bisect_left(frames, target)
            nearest = min(
                (abs(target - frames[j]) for j in (index - 1, index, index + 1)
                 if 0 <= j < len(frames)),
                default=None,
            )
            if nearest is not None and nearest <= half_window:
                covered += 1
    return covered / total if total else 0.0


def _scene_in_gold(scene_id: str, item: "GoldQuery") -> bool:
    """Scene này có chứa frame gold không.

    Dùng khoảng frame của scene chứ không so scene_id: gold ghi frame, còn
    candidate ở tầng fusion là scene. Suy scene_idx từ id (`..._S0042`) rồi đối
    chiếu qua `scene_bounds` đã nạp sẵn.
    """

    bounds = _SCENE_BOUNDS.get(scene_id)
    if bounds is None:
        return False
    start, end = bounds
    targets = [i for i in item.intervals] + [s for s in item.steps]
    for interval in targets:
        mid = (interval.start_frame + interval.end_frame) // 2
        if start <= mid < end:
            return True
    return False


_SCENE_BOUNDS: dict[str, tuple[int, int]] = {}


def _first_rank(results, predicate) -> int | None:
    return next((index for index, hit in enumerate(results, start=1) if predicate(hit)), None)


def _frame_hit(hit, gold: GoldQuery) -> bool:
    """Đúng luật: cùng video VÀ frame nộp nằm trong một interval GT."""

    return hit.video_id == gold.video_id and any(
        interval.contains(hit.best_frame_idx) for interval in gold.intervals
    )


def dedup_grades_by_event(hits: list[tuple[int, str | None]]) -> list[int]:
    """Mỗi sự kiện gold chỉ được tính điểm MỘT lần; lần sau tính 0.

    Gold AVS ghi `dedup_requirement: at most one representative segment per
    news report`, mà một interval gold trải qua 15–64 scene. Tính điểm cho mọi
    scene khớp thì tử số của nDCG cộng trên nhiều vị trí hơn mẫu số và chỉ số
    VƯỢT 1 — đo được 2/24 truy vấn ở baseline P2 (max 1.429) và 14/24 ở lượt
    sau (max 2.357). Chỉ số vượt 1 thì không so được cấu hình nào với cấu hình
    nào, nên mọi kết luận AVS trước đây phải đo lại.
    """

    grades: list[int] = []
    seen: set[str] = set()
    for grade, event_id in hits:
        if event_id is None:
            grades.append(grade)
            continue
        grades.append(0 if event_id in seen else grade)
        seen.add(event_id)
    return grades


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

    # Ba tham số TRAKE là RÀNG BUỘC HÌNH THỨC, không phải độ liên quan:
    # `gap_penalty_per_sec` phạt chuỗi trải rộng theo thời gian, `order_weight`
    # thưởng cho video có thứ tự khớp. Luật TRAKE chỉ đòi ĐÚNG THỨ TỰ và ĐÚNG
    # NGỮ NGHĨA — không điều khoản nào thưởng chuỗi gọn về thời gian. Đây là cờ
    # để đo xem chúng có đang lấn át độ liên quan không (docs/22).
    trake_overrides = {
        key: value
        for key, value in (
            ("gap_penalty_per_sec", args.trake_gap_penalty),
            ("order_weight", args.trake_order_weight),
            ("missing_step_penalty", args.trake_missing_penalty),
            ("sequence_strategy", args.trake_strategy),
            ("video_context_weight", args.trake_context_weight),
            ("video_duplicate_penalty", args.trake_duplicate_penalty),
            ("video_coverage_weight", args.trake_coverage_weight),
        )
        if value is not None
    }
    # Tắt nhánh theo tên. Cần cho phép đo CÔNG BẰNG trên nhiều video: nếu
    # video distractor không có dữ liệu ở một trường (V002/V003 chỉ có caption,
    # không có keyword/ASR/OCR/object/action) thì nhánh tương ứng CHỈ trả về
    # được video gốc và không thể nhầm. Sáu nhánh như vậy làm bài toán dễ đi
    # một cách giả tạo, và số đo "khoẻ khi có distractor" trở nên vô nghĩa.
    disabled = {
        name: BranchRuntimeOptions(enabled=False) for name in (args.disable_branch or [])
    }
    if args.max_per_video is None and not trake_overrides and not disabled:
        return None
    kwargs = {}
    if args.max_per_video is not None:
        kwargs["fusion"] = FusionOptions(
            max_results_per_video=_UNLIMITED if args.max_per_video == 0 else args.max_per_video
        )
    if trake_overrides:
        kwargs["temporal"] = TemporalOptions(**trake_overrides)
    if disabled:
        kwargs["branches"] = disabled
    return SearchOptions(**kwargs)


async def evaluate(
    gold: list[GoldQuery], repository: JsonlSceneRepository, args: argparse.Namespace
) -> dict:
    if args.pipeline == "container":
        # Đo ĐÚNG pipeline server chạy. `build_service` là bản dựng thứ hai:
        # thiếu nhánh object/action/color/event, thiếu VLM rerank, và không
        # bọc encoder bằng TranslatingTextEncoder — số nó in ra trông hợp lệ
        # nhưng không phải số của hệ thống thật.
        from online.api.container import build_container
        from online.config import Settings

        service = (await build_container(Settings.from_env())).search_service
    else:
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
    _SCENE_BOUNDS.clear()
    _SCENE_BOUNDS.update(
        {sc.scene_id: (sc.start_frame, sc.end_frame_exclusive) for sc in scenes}
    )
    kis = RankedMetrics()
    qa_evidence = RankedMetrics()
    qa_answer_correct = 0
    qa_joint_correct = 0
    qa_pairing_correct = 0
    qa_pairing_total = 0
    qa_total = 0
    kis_pairwise_correct = 0
    kis_pairwise_total = 0
    avs_zero_results = 0
    avs_pre_counts: list[int] = []
    avs_post_counts: list[int] = []
    avs_correct_dropped = 0
    trake_video_correct = 0
    trake_r_scores: list[float] = []
    trake_video_ranks: list[int | None] = []
    trake_total = 0
    avs_ndcg: list[float] = []
    avs_precision: list[float] = []
    avs_event_coverage: list[float] = []
    per_query: list[dict] = []

    started_at = time.monotonic()
    for index, item in enumerate(gold, start=1):
        # Tiến trình có FLUSH, không phụ thuộc `--verbose`. Không có nó thì một
        # lần chạy 120 truy vấn là một hộp đen 10 phút: không biết còn sống hay
        # đã kẹt, và không ước lượng được thời gian còn lại. Ghi ra stderr để
        # không lẫn vào bảng kết quả trên stdout.
        if index == 1 or index % 5 == 0 or index == len(gold):
            elapsed = time.monotonic() - started_at
            rate = elapsed / max(index - 1, 1)
            remain = rate * (len(gold) - index)
            print(
                f"[{index}/{len(gold)}] {item.query_id} ({item.task.value}) "
                f"— đã {elapsed:.0f}s, còn ~{remain:.0f}s",
                file=sys.stderr, flush=True,
            )
        response = await service.search(
            SearchRequest(
                query=item.query, task=item.task, top_k=args.top_k,
                search_options=_search_options(args),
                # P2: bật dấu vết từng tầng. Chỉ tốn bộ nhớ, không đổi kết quả
                # — `debug` không tham gia vào bất kỳ quyết định xếp hạng nào.
                debug=True,
            )
        )
        record: dict = {"query_id": item.query_id, "task": item.task.value,
                        "target_video": item.video_id}
        # PR-4A: sức khoẻ từng nhánh cho MỖI truy vấn. Không có nó thì một
        # nhánh timeout lẻ tẻ sẽ hiện ra dưới dạng "metric dao động" và bị quy
        # nhầm cho thuật toán. `branch_status` là thứ duy nhất phân biệt được
        # "nhánh chạy nhưng xếp khác" với "nhánh không chạy".
        record["branches"] = {
            status.execution_id: status.state for status in response.branch_status
        }
        # P2 — năm thứ phải lưu RIÊNG, vì mỗi cái trả lời một câu khác nhau:
        #   candidate_recall_rank  candidate đúng có trong pool không, hạng mấy
        #   prerank_gold_rank      trước rerank nó đứng đâu
        #   postrank_gold_rank     sau rerank nó đứng đâu
        #   branch_latency_ms      nhánh nào chậm
        #   gold_n_branches        bao nhiêu nhánh cùng thấy candidate đúng
        # Điểm cuối gộp cả năm; tách ra mới quy được trách nhiệm cho đúng tầng.
        trace = response.pipeline_trace or {}
        if trace:
            record["branch_latency_ms"] = trace.get("branch_latency_ms")
            record["prefusion_total"] = trace.get("prefusion_total")

            def _gold_rank(rows):
                for index, row in enumerate(rows or (), start=1):
                    if row.get("video_id") != item.video_id:
                        continue
                    scene = row.get("candidate_id") or ""
                    if _scene_in_gold(scene, item):
                        return index, row
                return None, None

            rank, row = _gold_rank(trace.get("fused"))
            record["candidate_recall_rank"] = rank
            record["prerank_gold_rank"] = rank
            record["gold_n_branches"] = row.get("n_branches") if row else None
            record["max_n_branches"] = max(
                (r.get("n_branches", 0) for r in trace.get("fused") or ()), default=0
            )
            post_rank, _ = _gold_rank(trace.get("post_rerank"))
            record["postrank_gold_rank"] = post_rank
        # `disabled` KHÔNG phải hỏng: đó là định tuyến có chủ ý — truy vấn
        # không có manh mối chữ/lời nói thì OCR/ASR nhận trọng số 0 và nhánh
        # không chạy (ROUTE-01, `allow_zero_modality`). `empty` cũng bình
        # thường: nhánh chạy nhưng index không có gì khớp.
        #
        # Gộp chúng vào "degraded" gây báo động giả 40/40 lượt và che mất hỏng
        # THẬT — bản đầu của chỉ số này đã mắc đúng lỗi đó.
        degraded = [
            status.execution_id
            for status in response.branch_status
            if status.state not in ("success", "empty", "disabled")
        ]
        if degraded:
            record["degraded_branches"] = degraded

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
            # `top1_pairwise_accuracy`: TRONG SỐ các truy vấn đã đưa được đáp án
            # vào top-2, bao nhiêu phần trăm xếp nó hạng 1. Tách ra vì đo được
            # 11/12 truy vấn có đáp án ở hạng 1-2 và R@20 đã bằng 1.000 — bài
            # toán còn lại là phân biệt hai ứng viên sát nhau ở đỉnh, không phải
            # tìm kiếm. R@1 gộp chung không phân biệt được "không tìm ra" với
            # "tìm ra nhưng xếp thứ hai".
            if rank is not None and rank <= 2:
                kis_pairwise_total += 1
                kis_pairwise_correct += int(rank == 1)

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
            # `joint_top1` phải là "DÒNG ĐẦU đúng cả ba", không phải "có dòng
            # nào đó đúng answer VÀ dòng đầu đúng frame" — hai vế đó có thể rơi
            # vào HAI DÒNG KHÁC NHAU, và bản cũ đếm cả trường hợp dòng 1 đúng
            # frame nhưng sai answer. Chỉ số bị thổi lên.
            first = response.qa[0] if response.qa else None
            joint_ok = bool(
                first
                and first.video_id == item.video_id
                and any(interval.contains(first.frame_idx) for interval in item.intervals)
                and answer_matches(first.answer, item.accepted_answers)
            )
            # `pairing_accuracy`: trong số truy vấn ĐÃ CÓ CẢ HAI MẢNH (tìm được
            # frame đúng ở đâu đó, và sinh được answer đúng ở đâu đó), bao nhiêu
            # phần trăm ghép chúng vào CÙNG MỘT DÒNG. Tách ra vì đo được
            # answer_accuracy 0.583 mà joint_top1 chỉ 0.083 — hệ có đủ nguyên
            # liệu nhưng ghép sai, và metric gộp không cho thấy điều đó.
            answer_anywhere = any(
                answer_matches(row.answer, item.accepted_answers) for row in response.qa
            )
            evidence_found = rank is not None
            if answer_anywhere and evidence_found:
                qa_pairing_total += 1
                qa_pairing_correct += int(answer_ok)
            qa_answer_correct += int(answer_ok)
            qa_joint_correct += int(joint_ok)
            record["answer_correct"] = answer_ok
            record["joint_top1"] = joint_ok
            record["evidence_rank"] = rank
            record["predicted_answers"] = [row.answer for row in response.qa[:3]]

        elif item.task == TaskType.TRAKE:
            trake_total += 1
            # response.trake (PR-07: TrakeProcessor, video-first) thay cho
            # response.sequences (bản nối scene cũ, không khóa video).
            # PR-4B Stage A: thứ hạng của VIDEO đúng, tách khỏi mọi chuyện
            # chọn frame. `response.trake` được TrakeProcessor sinh theo thứ tự
            # video đã xếp hạng, nên thứ tự video xuất hiện lần đầu chính là
            # bảng xếp hạng Stage A.
            video_order: list[str] = []
            for row in response.trake:
                if row.video_id not in video_order:
                    video_order.append(row.video_id)
            gold_rank = (
                video_order.index(item.video_id) + 1
                if item.video_id in video_order else None
            )
            record["gold_video_rank"] = gold_rank
            record["video_order"] = video_order[:5]
            trake_video_ranks.append(gold_rank)

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

            def _gold_hit(row) -> tuple[int, str | None]:
                """`(điểm, event_id)` của interval khớp; `(0, None)` nếu không."""

                if row.video_id != item.video_id or row.best_frame_idx is None:
                    return 0, None
                matched = [
                    (grade, interval.event_id)
                    for interval, grade in gold_grade_by_frame
                    if interval.contains(row.best_frame_idx)
                ]
                return max(matched, default=(0, None))

            grades = dedup_grades_by_event([_gold_hit(row) for row in response.avs])
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
            # `zero_result_rate`: đo được 3/8 truy vấn trả về ĐÚNG 0 kết quả
            # dù top-100 được phép. nDCG gộp không phân biệt "xếp hạng kém"
            # với "không trả về gì" — hai nguyên nhân cần hai cách sửa khác hẳn.
            avs_zero_results += int(len(response.avs) == 0)
            # AVS-GRADE-01: candidate ĐÚNG có bị chính cổng grade loại không?
            # Đây là câu hỏi quyết định — nó phân biệt "cổng từ vựng vứt mất
            # đáp án" với "pool vốn không có đáp án", hai nguyên nhân của cùng
            # một `zero_result_rate` nhưng cần hai cách sửa khác hẳn.
            diag = response.avs_diagnostics or {}
            if diag:
                avs_pre_counts.append(diag.get("pre_grade_candidate_count", 0))
                avs_post_counts.append(diag.get("post_grade_candidate_count", 0))
                # Chỉ tính candidate đúng bị loại mà LẼ RA đã lọt vào đầu ra.
                # Bản đầu đếm MỌI candidate đúng bị loại, kể cả những cái thừa
                # vốn không bao giờ chen được vào top-`max_per_video` — nên nó
                # báo 0.875 cho một cấu hình đang cho nDCG cao nhất, tức không
                # dùng để ra quyết định được.
                #
                # Xấp xỉ "trong tầm đầu ra": điểm ngữ nghĩa của candidate bị
                # loại cao hơn điểm thấp nhất trong số đã được giữ.
                kept_floor = min(
                    (row.score for row in response.avs), default=0.0
                )
                lost = [
                    row
                    for row in diag.get("dropped", [])
                    if row.get("video_id") == item.video_id
                    and row.get("best_frame_idx") is not None
                    and row.get("semantic_score", 0.0) >= kept_floor
                    and any(
                        interval.contains(row["best_frame_idx"])
                        for interval in item.intervals
                    )
                ]
                avs_correct_dropped += int(bool(lost))
                record["grade_dropped_correct"] = len(lost)
                record["pre_grade"] = diag.get("pre_grade_candidate_count")
                record["post_grade"] = diag.get("post_grade_candidate_count")
            record.update({"ndcg": ndcg, "precision": precision, "result_count": len(response.avs)})

        per_query.append(record)
        if args.verbose:
            print(f"  {record}")

    summary: dict = {"per_query": per_query}
    # Tách theo video: khi gold có truy vấn của nhiều video, cái cần biết là
    # "hệ hoạt động thế nào trên video CHƯA tinh chỉnh", không phải trung bình
    # gộp. Một video được tinh chỉnh kỹ có thể kéo trung bình lên và che hẳn
    # việc hệ không tổng quát được.
    videos = sorted({record.get("target_video") for record in per_query if record.get("target_video")})
    if len(videos) > 1:
        by_video: dict[str, dict] = {}
        for video in videos:
            rows = [r for r in per_query if r.get("target_video") == video]
            block: dict = {"queries": len(rows)}
            kis_rows = [r for r in rows if r.get("task") == "TEXTUAL_KIS"]
            if kis_rows:
                hits = [r for r in kis_rows if r.get("first_frame_hit_rank") == 1]
                block["KIS_R@1"] = len(hits) / len(kis_rows)
            qa_rows = [r for r in rows if r.get("task") == "QA"]
            if qa_rows:
                block["QA_joint_top1"] = sum(
                    1 for r in qa_rows if r.get("joint_top1")
                ) / len(qa_rows)
            trake_rows = [r for r in rows if r.get("task") == "TRAKE"]
            if trake_rows:
                block["TRAKE_mean_r"] = sum(
                    r.get("r_score", 0.0) for r in trake_rows
                ) / len(trake_rows)
                block["TRAKE_video_recall@1"] = sum(
                    1 for r in trake_rows if r.get("gold_video_rank") == 1
                ) / len(trake_rows)
            avs_rows = [r for r in rows if r.get("task") == "AVS"]
            if avs_rows:
                block["AVS_nDCG"] = sum(r.get("ndcg", 0.0) for r in avs_rows) / len(avs_rows)
            by_video[video] = block
        summary["by_video"] = by_video
    if kis.total:
        summary["TEXTUAL_KIS"] = kis.as_row() | {
            "queries": kis.total,
            # Đổi tên cho đúng bản chất: R@20 chính là "đáp án có lọt vào tập
            # ứng viên top-20 không". Giữ cả hai tên để so được với số cũ.
            "candidate_recall@20": kis.hits_at[20] / kis.total,
            "top1_pairwise_accuracy": (
                kis_pairwise_correct / kis_pairwise_total if kis_pairwise_total else 0.0
            ),
            "top1_pairwise_n": kis_pairwise_total,
        }
    if qa_total:
        summary["QA"] = qa_evidence.as_row() | {
            "queries": qa_total,
            "answer_accuracy": qa_answer_correct / qa_total,
            # `evidence_recall`: frame đúng có xuất hiện ở BẤT KỲ dòng nào không.
            # Tách khỏi answer để biết mảnh nào đang thiếu.
            "evidence_recall": qa_evidence.hits_at[100] / qa_total,
            "pairing_accuracy": (
                qa_pairing_correct / qa_pairing_total if qa_pairing_total else 0.0
            ),
            "pairing_n": qa_pairing_total,
            "joint_top1": qa_joint_correct / qa_total,
        }
    if trake_total:
        oracle = _frame_oracle_coverage(gold, args)
        mean_r = sum(trake_r_scores) / trake_total
        correct_scores = [
            record["r_score"]
            for record in per_query
            if record.get("task") == "TRAKE" and record.get("video_correct")
        ]
        mean_r_correct = (
            sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
        )
        summary["TRAKE"] = {
            "queries": trake_total,
            "correct_video_rate": trake_video_correct / trake_total,
            # Stage A tách riêng: `video_recall@1` chính là correct_video_rate,
            # nhưng `@3` và thứ hạng trung vị cho biết video đúng TRƯỢT XA bao
            # nhiêu — "xếp thứ 2" và "không có trong danh sách" cần hai cách
            # sửa khác hẳn nhau.
            "video_recall@1": sum(1 for r in trake_video_ranks if r == 1) / trake_total,
            "video_recall@3": sum(1 for r in trake_video_ranks if r and r <= 3) / trake_total,
            "gold_video_rank_median": (
                sorted(r for r in trake_video_ranks if r)[
                    len([r for r in trake_video_ranks if r]) // 2
                ]
                if any(trake_video_ranks) else None
            ),
            "gold_video_missing": sum(1 for r in trake_video_ranks if r is None) / trake_total,
            "mean_r_score": mean_r,
            # `frame_oracle_coverage`: tỉ lệ bước MÀ CORPUS có ít nhất một
            # keyframe rơi vào cửa sổ GT. Đây là TRẦN của mean_r_score — không
            # cách chọn nào vượt được nó.
            "frame_oracle_coverage": oracle,
            # `frame_selection_accuracy`: trong phần trần đó, hệ chọn đúng bao
            # nhiêu. Tách hai tầng này ra là bắt buộc: đo được nới cửa sổ đẩy
            # trần +0.200 mà điểm thật chỉ +0.056, tức vấn đề nằm ở CHỌN chứ
            # không ở trần — mean_r_score gộp chung không cho thấy điều đó.
            #
            # Tính trên `mean_r_on_correct_video`, KHÔNG trên `mean_r_score`:
            # query chọn sai video thì mọi step đều 0 theo luật, nên trộn chúng
            # vào đây biến chỉ số "chọn frame" thành chỉ số "chọn video" trá
            # hình. Đo trên 3 video: mean_r tụt 0.263 -> 0.144 nhưng riêng phần
            # video đúng chỉ 0.263 -> 0.230 — gần như toàn bộ mất mát nằm ở
            # Stage A chứ không ở chọn frame.
            "mean_r_on_correct_video": mean_r_correct,
            "frame_selection_accuracy": (mean_r_correct / oracle) if oracle else 0.0,
            "complete_chain_rate": sum(1 for value in trake_r_scores if value == 1.0) / trake_total,
        }
    if avs_ndcg:
        summary["AVS"] = {
            "queries": len(avs_ndcg),
            f"nDCG@{args.top_k}": sum(avs_ndcg) / len(avs_ndcg),
            f"P@{args.top_k}": sum(avs_precision) / len(avs_precision),
            "event_coverage": sum(avs_event_coverage) / len(avs_event_coverage),
            "zero_result_rate": avs_zero_results / len(avs_ndcg),
            "pre_grade_candidate_count": (
                sum(avs_pre_counts) / len(avs_pre_counts) if avs_pre_counts else 0.0
            ),
            "post_grade_candidate_count": (
                sum(avs_post_counts) / len(avs_post_counts) if avs_post_counts else 0.0
            ),
            "correct_candidate_dropped_by_grade": avs_correct_dropped / len(avs_ndcg),
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


def _warn_if_nondeterministic() -> None:
    """Cảnh báo khi `PYTHONHASHSEED` chưa cố định.

    Đã đo được: chạy CÙNG lệnh, CÙNG dữ liệu, hai lần liên tiếp cho
    `mean_r_score` 0.212 rồi 0.075. Đặt `PYTHONHASHSEED=0` thì ba lần chạy ra
    số y hệt. Thứ tự lặp `set` chuỗi thay đổi theo tiến trình và rò vào
    tie-break của xếp hạng.

    Hệ quả: mọi so sánh chênh 1 query (8.3 điểm phần trăm trên 12 query) là
    KHÔNG kết luận được nếu chưa cố định seed.
    """

    import os

    if os.environ.get("PYTHONHASHSEED") in (None, "random"):
        print(
            "CẢNH BÁO: PYTHONHASHSEED chưa cố định — kết quả KHÔNG tái lập được.\n"
            "          Chạy lại với:  PYTHONHASHSEED=0 python -m scripts.eval_tasks ...\n",
            file=sys.stderr,
        )


async def _main() -> None:
    _warn_if_nondeterministic()
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
    parser.add_argument("--trake-gap-penalty", type=float, default=None,
                        help="Ghi đè gap_penalty_per_sec (mặc định deployment 0.002). 0 = tắt phạt khoảng cách")
    parser.add_argument("--trake-order-weight", type=float, default=None,
                        help="Ghi đè order_weight (mặc định 0.6). 0 = tắt thưởng thứ tự khi xếp hạng video")
    parser.add_argument("--disable-branch", action="append", default=[],
                        help="Tat mot nhanh theo branch_id. Lap lai duoc. Dung cho phep do cong bang da video")
    parser.add_argument("--trake-context-weight", type=float, default=None,
                        help="Stage A: trong so do lien quan (mac dinh 0.4)")
    parser.add_argument("--trake-duplicate-penalty", type=float, default=None,
                        help="Stage A: phat khi nhieu step tro ve cung scene (mac dinh 0.5)")
    parser.add_argument("--trake-coverage-weight", type=float, default=None,
                        help="Stage A: trong so ti le step co bang chung (mac dinh 1.0)")
    parser.add_argument("--trake-strategy", choices=("beam","dp"), default=None,
                        help="beam (mac dinh) hoac dp — quy hoach dong chinh xac, cung ham muc tieu")
    parser.add_argument("--trake-missing-penalty", type=float, default=None,
                        help="Ghi đè missing_step_penalty (mặc định 0.5)")
    parser.add_argument(
        "--pipeline", choices=("legacy", "container"), default="legacy",
        help="container = dựng qua online/api/container.py, tức đúng pipeline server "
             "(đọc env/AIC_ENV_FILE: VLM rerank, dịch query VI→EN, expansion bằng LLM). "
             "legacy = bản dựng riêng của script, chỉ dùng cho ablation.",
    )
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
        # Tự tạo thư mục cha: thiếu nó thì script chạy xong, in đủ số ra
        # màn hình, rồi ném FileNotFoundError ở dòng cuối và MẤT TRẮNG kết
        # quả của một lần chạy dài. Đã xảy ra thật.
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nchi tiết -> {args.json_out}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
