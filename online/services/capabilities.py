"""Từ chối cấu hình mà backend không thực sự chạy (PR-04).

Vấn đề cụ thể: `SearchOptions` khai báo ~25 field, phần lớn chưa có consumer.
Request đặt `{"branches": {"ocr_fuzzy": {"min_score": 0.8}}}` nhận **200 OK**
rồi backend bỏ qua — người dùng tin là đã chỉnh, ablation ra số vô nghĩa.

Nguyên tắc: **implemented thì chạy thật, chưa implemented thì 422, không bao
giờ nhận rồi lờ đi.**

Chỉ field được caller ĐẶT TƯỜNG MINH mới bị kiểm tra (`model_fields_set`).
Nhờ vậy `search_options: {}` — toàn giá trị mặc định — vẫn hợp lệ và giữ
nguyên hành vi cũ, dù mặc định của một số field (vd `rerank.text.enabled`)
mô tả tính năng chưa có.
"""

from __future__ import annotations

from pydantic import BaseModel

from online.domain.execution import BranchCapabilities
from online.domain.search_config import SearchOptions
from online.errors import OnlineError

# Control áp dụng cho mọi branch, do orchestrator/fusion xử lý chứ không phải
# adapter, nên không cần branch tự khai báo.
GLOBAL_BRANCH_CONTROLS = frozenset(
    {"enabled", "weight", "top_k", "timeout_ms", "min_score", "threshold_space", "threshold_policy"}
)

# (đường dẫn field, giá trị bị từ chối hoặc None nếu từ chối mọi giá trị) -> lý do.
UNSUPPORTED: dict[str, tuple[object, str]] = {
    "query.enable_hyde": (
        True,
        "HyDE cần một LLM sinh mô tả giả định; chưa có model server nào được nối",
    ),
    "query.generate_english_variant": (
        True,
        "chưa có bộ dịch/encoder đa ngữ; query tiếng Anh sẽ không được sinh",
    ),
    "query.generate_bilingual_variant": (
        True,
        "chưa có bộ dịch/encoder đa ngữ; biến thể song ngữ sẽ không được sinh",
    ),
    "query.preserve_raw_query": (
        False,
        "query gốc luôn được giữ (QueryPlan.original_query); không tắt được",
    ),
    "query.normalize_query": (
        False,
        "chuẩn hóa NFC/khoảng trắng luôn chạy; không tắt được",
    ),
    "rerank.temporal_verifier": (
        True,
        "temporal verifier chưa được cài đặt",
    ),
    "results.group_by": (
        None,
        "response chưa hỗ trợ nhóm kết quả; chỉ có danh sách phẳng",
    ),
    "fusion.dedup_similarity": (
        None,
        "dedup theo độ tương đồng thị giác (embedding) chưa được cài đặt; "
        "dùng fusion.dedup_scope thay thế",
    ),
    # Field khai báo nhưng không có consumer nào đọc — phát hiện khi audit UI
    # competition studio (rà từng field trong SearchOptions xem có code nào
    # thật sự dùng không). Từ chối tường minh thay vì nhận rồi bỏ qua.
    "fusion.fusion_top_k": (
        None,
        "chưa có consumer nào đọc field này; giới hạn candidate thực tế là "
        "candidate_limit của SearchService, không đổi được qua request",
    ),
    "rerank.enable_rules": (
        None,
        "bật/tắt rule bonus/penalty là cấu hình deployment (AIC_ENABLE_RULES), "
        "không đổi được theo từng request",
    ),
    "rerank.text.weight": (
        None,
        "text reranker hiện thay thế thứ hạng trực tiếp (re-order), không "
        "blend theo trọng số nên field này chưa có tác dụng",
    ),
    "rerank.vlm.weight": (
        None,
        "VLM reranker hiện thay thế thứ hạng trực tiếp (re-order), không "
        "blend theo trọng số nên field này chưa có tác dụng",
    ),
    "temporal.enabled": (
        False,
        "TRAKE luôn chạy căn chỉnh chuỗi khi query có >= 2 event; không tắt được",
    ),
    "temporal.same_video_required": (
        False,
        "Stage A luôn khóa đúng một video trước khi căn chỉnh (luật TRAKE: sai "
        "video = 0 điểm); không tắt được",
    ),
    "temporal.ordered_steps_required": (
        False,
        "beam search luôn yêu cầu frame tăng dần giữa các step; không tắt được",
    ),
    "temporal.minimum_gap_sec": (
        None,
        "chưa có override theo giây; SequenceConfig chỉ có min_gap_frames cố định",
    ),
    "temporal.neighbor_before_sec": (
        None,
        "chưa nối vào NeighborContext của evidence pack",
    ),
    "temporal.neighbor_after_sec": (
        None,
        "chưa nối vào NeighborContext của evidence pack",
    ),
}

# Giá trị được chấp nhận cho field vốn bị từ chối "mọi giá trị".
ALLOWED_VALUES: dict[str, set[object]] = {
    "results.group_by": {"none"},
    # PR-05: dedup service hỗ trợ thật các scope này.
    "fusion.dedup_scope": {"none", "frame", "scene", "event"},
}

UNSUPPORTED_BRANCH_CONTROLS: dict[str, str] = {
    "field_weights": "chưa có branch lexical đa field; mỗi branch BM25 hiện chỉ phục vụ một field",
    "model_id": "chưa có model registry; branch dùng đúng model đã đăng ký lúc build",
    "index_id": "chưa có index registry; branch dùng đúng index đã đăng ký lúc build",
    "query_variant": "biến thể query chọn qua branch riêng (vd bm25_caption.expanded), không qua field này",
}


class UnsupportedSearchOptionError(OnlineError):
    """Caller đặt option mà backend chưa chạy thật."""

    code = "unsupported_search_option"


def _walk(model: BaseModel, prefix: str = "") -> dict[str, object]:
    """Trả về các field được đặt TƯỜNG MINH, phẳng theo đường dẫn `a.b.c`."""

    explicit: dict[str, object] = {}
    for name in model.model_fields_set:
        value = getattr(model, name)
        path = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            explicit.update(_walk(value, f"{path}."))
        else:
            explicit[path] = value
    return explicit


def validate_search_options(
    options: SearchOptions | None,
    capabilities: list[BranchCapabilities],
    *,
    rerank_stages: dict[str, bool] | None = None,
) -> None:
    """Ném `UnsupportedSearchOptionError` nếu có option không chạy thật.

    `rerank_stages` cho biết tầng rerank nào ĐANG có model server. Bật một
    tầng chưa cấu hình là lỗi tường minh chứ không phải "bật rồi nhưng không
    thấy gì khác".
    """

    if options is None:
        return
    problems: list[str] = []
    explicit = _walk(options)
    stages = rerank_stages or {}

    for stage, path in (("text", "rerank.text.enabled"), ("vlm", "rerank.vlm.enabled")):
        if explicit.get(path) is True and not stages.get(stage):
            problems.append(
                f"{path}=True: chưa cấu hình {stage} reranker "
                f"(đặt AIC_RERANK_{stage.upper()}_URL) nên tầng này sẽ không chạy"
            )

    for path, value in explicit.items():
        if path.startswith("branches"):
            continue
        rule = UNSUPPORTED.get(path)
        if rule is None:
            continue
        rejected, reason = rule
        allowed = ALLOWED_VALUES.get(path, set())
        if value in allowed:
            continue
        if rejected is None or value == rejected:
            problems.append(f"{path}={value!r}: {reason}")

    known_branches = {item.branch_id for item in capabilities}
    known_executions = {
        execution for item in capabilities for execution in item.execution_ids
    }
    controls_by_branch = {item.branch_id: set(item.supported_controls) for item in capabilities}

    for key, branch_options in options.branches.items():
        branch_id = key.rsplit(".", 1)[0] if "." in key else key
        if key not in known_branches and key not in known_executions:
            problems.append(
                f"branches[{key!r}]: không có branch/execution nào tên này đang chạy; "
                f"xem GET /v1/search/capabilities (đang có: {sorted(known_branches)})"
            )
            continue
        supported = controls_by_branch.get(branch_id, set()) | GLOBAL_BRANCH_CONTROLS
        for control in branch_options.model_fields_set:
            if control in UNSUPPORTED_BRANCH_CONTROLS:
                problems.append(
                    f"branches[{key!r}].{control}: {UNSUPPORTED_BRANCH_CONTROLS[control]}"
                )
            elif control not in supported:
                problems.append(
                    f"branches[{key!r}].{control}: branch {branch_id!r} không đọc control này "
                    f"(hỗ trợ: {sorted(supported)})"
                )

    if problems:
        raise UnsupportedSearchOptionError(
            "search_options chứa cấu hình backend chưa chạy thật:\n- " + "\n- ".join(problems)
        )


__all__ = [
    "GLOBAL_BRANCH_CONTROLS",
    "UNSUPPORTED",
    "UNSUPPORTED_BRANCH_CONTROLS",
    "UnsupportedSearchOptionError",
    "validate_search_options",
]
