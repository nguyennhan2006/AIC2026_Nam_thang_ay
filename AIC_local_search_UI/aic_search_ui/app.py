from __future__ import annotations

import html
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from aic_local_search import LocalHybridSearchEngine
    from aic_local_search.vector_index import OpenClipTextEncoder
except ImportError as exc:
    st.set_page_config(page_title="AIC Local Search", page_icon="🔎", layout="wide")
    st.error(
        "Không import được `aic_local_search`. Hãy chọn đúng interpreter "
        "`aic-search` trong VS Code và cài engine bằng `python -m pip install -e .`."
    )
    st.exception(exc)
    st.stop()


st.set_page_config(
    page_title="AIC Local Video Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


BRANCH_LABELS = {
    "semantic": "Caption",
    "ocr": "OCR",
    "speech": "ASR",
    "tags": "Tags",
    "event": "Event",
    "scene_vector": "Scene FAISS",
    "frame_vector": "Frame FAISS",
}

MEDIA_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
}


st.markdown(
    """
    <style>
    .aic-pipeline {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .55rem;
        margin: .35rem 0 1rem;
    }
    .aic-step {
        border: 1px solid rgba(128, 128, 128, .28);
        border-radius: .65rem;
        padding: .65rem .7rem;
        min-height: 4.25rem;
        background: rgba(90, 120, 255, .055);
    }
    .aic-step strong { display: block; margin-bottom: .15rem; }
    .aic-step span { opacity: .72; font-size: .85rem; }
    .aic-branches {
        display: flex;
        flex-wrap: wrap;
        gap: .35rem;
        margin: .4rem 0 .2rem;
    }
    .aic-chip {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(128, 128, 128, .28);
        border-radius: 999px;
        padding: .18rem .5rem;
        font-size: .82rem;
        background: rgba(90, 120, 255, .07);
    }
    .aic-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        background: rgba(90, 120, 255, .16);
        font-weight: 700;
        margin-right: .4rem;
    }
    .aic-time {
        opacity: .72;
        font-variant-numeric: tabular-nums;
    }
    .aic-placeholder {
        min-height: 12rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px dashed rgba(128, 128, 128, .35);
        border-radius: .65rem;
        opacity: .65;
        text-align: center;
        padding: 1rem;
    }
    @media (max-width: 900px) {
        .aic-pipeline { grid-template-columns: 1fr; }
        .aic-step { min-height: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clean_path(value: str) -> Path:
    return Path(value.strip().strip('"')).expanduser()


def _fmt_time(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _marked_snippet(value: str) -> str:
    escaped = html.escape(value or "")
    return escaped.replace("[", "<mark>").replace("]", "</mark>")


@st.cache_data(show_spinner=False)
def load_manifest(index_dir: str, manifest_mtime: float) -> dict[str, Any]:
    del manifest_mtime
    return json.loads(
        (Path(index_dir) / "index_manifest.json").read_text(encoding="utf-8")
    )


@st.cache_data(show_spinner=False)
def load_video_catalog(database_path: str, database_mtime: float) -> list[dict[str, Any]]:
    del database_mtime
    connection = sqlite3.connect(
        f"file:{Path(database_path).resolve()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT video_id,
                   COUNT(*) AS scene_count,
                   MIN(start_sec) AS start_sec,
                   MAX(end_sec) AS end_sec
            FROM scenes
            GROUP BY video_id
            ORDER BY video_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def build_asset_catalog(asset_root: str, root_mtime: float) -> dict[str, Any]:
    del root_mtime
    root = Path(asset_root)
    by_name: dict[str, tuple[str, str, str]] = {}
    by_exact: dict[str, tuple[str, str, str]] = {}
    if not root.exists():
        return {"by_name": by_name, "by_exact": by_exact}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix in MEDIA_SUFFIXES:
            record = ("file", str(path.resolve()), "")
            by_name.setdefault(path.name.casefold(), record)
            try:
                relative = path.relative_to(root).as_posix().casefold()
                by_exact.setdefault(relative, record)
            except ValueError:
                pass
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        member_path = Path(member)
                        if member_path.suffix.casefold() not in MEDIA_SUFFIXES:
                            continue
                        record = ("zip", str(path.resolve()), member)
                        normalized = member.replace("\\", "/").lstrip("/").casefold()
                        by_exact.setdefault(normalized, record)
                        by_name.setdefault(member_path.name.casefold(), record)
            except (zipfile.BadZipFile, OSError):
                continue
    return {"by_name": by_name, "by_exact": by_exact}


@st.cache_data(show_spinner=False)
def read_zipped_asset(zip_path: str, member: str, zip_mtime: float) -> bytes:
    del zip_mtime
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member)


@st.cache_resource(show_spinner=False)
def get_text_encoder(model_spec: str, device: str) -> OpenClipTextEncoder:
    selected_device = None if device == "auto" else device
    return OpenClipTextEncoder(model_spec, selected_device)


def resolve_asset(
    reference: str | None,
    asset_root: str,
    catalog: dict[str, Any],
) -> tuple[str, str | bytes] | None:
    if not reference:
        return None
    raw = str(reference).strip()
    path = Path(raw)
    if path.is_file():
        return path.suffix.casefold(), str(path)

    root = Path(asset_root) if asset_root else None
    if root and root.exists():
        direct = root / raw
        if direct.is_file():
            return direct.suffix.casefold(), str(direct)

    normalized = raw.replace("\\", "/").lstrip("/").casefold()
    basename = normalized.rsplit("/", 1)[-1]
    record = catalog.get("by_exact", {}).get(normalized)
    if record is None:
        record = catalog.get("by_name", {}).get(basename)
    if record is None:
        return None

    kind, location, member = record
    suffix = Path(member or location).suffix.casefold()
    if kind == "file":
        return suffix, location
    data = read_zipped_asset(location, member, Path(location).stat().st_mtime)
    return suffix, data


def show_image(reference: str | None, asset_root: str, catalog: dict[str, Any]) -> bool:
    asset = resolve_asset(reference, asset_root, catalog)
    if asset is None:
        st.markdown(
            '<div class="aic-placeholder">Không tìm thấy keyframe.<br>'
            "Chọn đúng thư mục chứa output 01/04 ở thanh bên.</div>",
            unsafe_allow_html=True,
        )
        return False
    suffix, source = asset
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return False
    st.image(source, width="stretch")
    return True


def show_video(reference: str | None, asset_root: str, catalog: dict[str, Any]) -> bool:
    asset = resolve_asset(reference, asset_root, catalog)
    if asset is None:
        return False
    suffix, source = asset
    if suffix not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        return False
    st.video(source)
    return True


def run_search(
    *,
    index_dir: Path,
    query: str,
    visual_query: str,
    use_vector: bool,
    task: str,
    top_k: int,
    video_id: str | None,
    start_sec: float | None,
    end_sec: float | None,
    match_all: bool,
    device: str,
    embedding_model: str,
) -> tuple[list[dict[str, Any]], float, str | None]:
    started = time.perf_counter()
    vector_error: str | None = None
    query_vector = None
    effective_vector = use_vector

    if use_vector:
        try:
            encoder = get_text_encoder(embedding_model, device)
            query_vector = encoder.encode(visual_query.strip() or query)
        except Exception as exc:  # Show the issue and keep the lexical result usable.
            vector_error = f"{type(exc).__name__}: {exc}"
            effective_vector = False

    with LocalHybridSearchEngine(index_dir) as engine:
        hits = engine.search(
            query,
            visual_query=visual_query.strip() or None,
            query_vector=query_vector,
            use_vector=effective_vector,
            task=task,
            top_k=top_k,
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            match_all_terms=match_all,
        )
    return hits, time.perf_counter() - started, vector_error


def result_branch_html(hit: dict[str, Any]) -> str:
    ranks = hit.get("branch_ranks", {})
    chips = []
    for branch, rank in sorted(ranks.items(), key=lambda item: item[1]):
        label = BRANCH_LABELS.get(branch, branch)
        chips.append(
            f'<span class="aic-chip">{html.escape(label)} · #{int(rank)}</span>'
        )
    return '<div class="aic-branches">' + "".join(chips) + "</div>"


def render_evidence(hit: dict[str, Any]) -> None:
    snippets = hit.get("snippets") or {}
    if snippets:
        st.markdown("**Đoạn khớp theo từng nhánh**")
        for branch, snippet in snippets.items():
            label = BRANCH_LABELS.get(branch, branch)
            st.markdown(
                f"**{html.escape(label)}:** {_marked_snippet(str(snippet))}",
                unsafe_allow_html=True,
            )

    evidence = [
        ("Caption", hit.get("caption_vi") or hit.get("caption_en")),
        ("OCR", hit.get("ocr_text")),
        ("ASR", hit.get("transcript")),
        ("Keywords", hit.get("keywords")),
        ("Entities", hit.get("entities")),
        ("Actions", hit.get("actions")),
    ]
    for label, value in evidence:
        if value:
            st.markdown(f"**{label}:** {value}")

    event = hit.get("matched_event")
    if event:
        st.markdown(
            "**Event:** "
            + str(event.get("description_vi") or event.get("description_en") or "")
        )


def render_hit(
    hit: dict[str, Any],
    asset_root: str,
    catalog: dict[str, Any],
    max_rrf: float,
) -> None:
    with st.container(border=True):
        heading = (
            f'<span class="aic-rank">{int(hit["rank"])}</span>'
            f'<strong>{html.escape(str(hit["scene_id"]))}</strong> '
            f'<span class="aic-time">{_fmt_time(hit.get("start_sec"))} → '
            f'{_fmt_time(hit.get("end_sec"))}</span>'
        )
        st.markdown(heading, unsafe_allow_html=True)

        media_col, detail_col = st.columns([1.05, 2.35], gap="large")
        with media_col:
            frame = hit.get("best_frame") or {}
            show_image(frame.get("image_path"), asset_root, catalog)
            st.caption(
                f"{hit.get('video_id', '')} · "
                f"keyframe {frame.get('keyframe_id', '—')}"
            )
        with detail_col:
            score = float(hit.get("rrf_score", 0.0))
            score_ratio = score / max_rrf if max_rrf > 0 else 0.0
            metric_col, quality_col = st.columns(2)
            metric_col.metric("RRF score", f"{score:.5f}")
            quality_col.metric(
                "Quality",
                str(hit.get("quality_status", "unknown")),
                f"×{float(hit.get('quality_penalty', 1.0)):.2f}",
            )
            st.progress(min(1.0, max(0.0, score_ratio)))
            st.markdown(result_branch_html(hit), unsafe_allow_html=True)
            caption = hit.get("caption_vi") or hit.get("caption_en")
            if caption:
                st.write(caption)

            with st.expander("Xem bằng chứng OCR / ASR / caption / tags"):
                render_evidence(hit)

            if hit.get("clip_path"):
                with st.expander("Phát scene clip"):
                    if not show_video(hit.get("clip_path"), asset_root, catalog):
                        st.info(
                            "Không tìm thấy clip. UI vẫn hiển thị đầy đủ metadata và điểm search."
                        )


def render_search_results(
    run: dict[str, Any],
    asset_root: str,
    catalog: dict[str, Any],
) -> None:
    hits = run["hits"]
    if run.get("vector_error"):
        st.warning(
            "Nhánh FAISS không được bật vì OpenCLIP chưa encode được query. "
            "Kết quả bên dưới đang dùng BM25/FTS5.\n\n"
            + run["vector_error"]
        )
    if not hits:
        st.info("Không có scene phù hợp. Thử query ngắn hơn hoặc bỏ “khớp tất cả từ”.")
        return

    vector_status = hits[0].get("query_plan", {}).get("vector_status", "disabled")
    plan_hints = ", ".join(hits[0].get("query_plan", {}).get("hints", [])) or "none"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kết quả", len(hits))
    c2.metric("Thời gian", f"{run['elapsed']:.3f} s")
    c3.metric("Vector", vector_status)
    c4.metric("Planner hints", plan_hints)

    branch_counts = {
        label: sum(
            1 for hit in hits if branch in (hit.get("branch_ranks") or {})
        )
        for branch, label in BRANCH_LABELS.items()
    }
    chart_data = pd.DataFrame(
        {"Nhánh": list(branch_counts.keys()), "Số kết quả có đóng góp": list(branch_counts.values())}
    ).set_index("Nhánh")
    st.bar_chart(chart_data, horizontal=True)

    max_rrf = max(float(hit.get("rrf_score", 0.0)) for hit in hits)
    for hit in hits:
        render_hit(hit, asset_root, catalog, max_rrf)


def render_sequence_results(
    sequences: list[dict[str, Any]],
    asset_root: str,
    catalog: dict[str, Any],
) -> None:
    if not sequences:
        st.info(
            "Không tìm được chuỗi scene đúng thứ tự. Thử tăng max gap hoặc tắt vector."
        )
        return
    for sequence in sequences:
        with st.container(border=True):
            st.markdown(
                f"### #{sequence['rank']} · {sequence['video_id']} "
                f"· score {float(sequence['score']):.5f}"
            )
            st.caption(
                f"{_fmt_time(sequence['start_sec'])} → "
                f"{_fmt_time(sequence['end_sec'])}"
            )
            steps = sequence.get("steps", [])
            columns = st.columns(min(len(steps), 4))
            for index, hit in enumerate(steps):
                with columns[index % len(columns)]:
                    st.markdown(f"**Bước {index + 1}**")
                    st.code(hit["scene_id"], language=None)
                    frame = hit.get("best_frame") or {}
                    show_image(frame.get("image_path"), asset_root, catalog)
                    st.caption(
                        f"{_fmt_time(hit.get('start_sec'))} · "
                        f"{(hit.get('caption_vi') or hit.get('caption_en') or '')[:160]}"
                    )


default_project_root = Path.home() / "Documents" / "AIC2026"
default_index = os.environ.get(
    "AIC_INDEX_DIR", str(default_project_root / "08_local_search_index_050607")
)
default_assets = os.environ.get(
    "AIC_ASSET_ROOT", str(default_project_root / "component_outputs")
)

with st.sidebar:
    st.header("Kết nối dữ liệu")
    index_input = st.text_input(
        "Index directory",
        value=default_index,
        help="Thư mục có index_manifest.json và aic_search.db.",
    )
    asset_input = st.text_input(
        "Asset root",
        value=default_assets,
        help="Có thể trỏ vào thư mục chứa ZIP output 01/04; UI đọc ảnh/clip trực tiếp từ ZIP.",
    )
    device = st.selectbox(
        "Thiết bị OpenCLIP",
        options=["auto", "cpu", "cuda"],
        help="Auto ưu tiên CUDA nếu PyTorch nhận GPU.",
    )
    st.caption("Đổi đường dẫn xong, UI tự nạp lại index.")


index_dir = _clean_path(index_input)
manifest_path = index_dir / "index_manifest.json"
if not manifest_path.is_file():
    st.title("AIC Local Video Search")
    st.error(f"Không tìm thấy `{manifest_path}`.")
    st.code(
        'python -m aic_local_search.cli inspect --index-dir "'
        + str(index_dir)
        + '"',
        language="powershell",
    )
    st.stop()

manifest = load_manifest(str(index_dir), manifest_path.stat().st_mtime)
database_path = index_dir / manifest["files"]["database"]
if not database_path.is_file():
    st.error(f"Thiếu database: `{database_path}`")
    st.stop()

video_catalog = load_video_catalog(str(database_path), database_path.stat().st_mtime)
video_ids = [item["video_id"] for item in video_catalog]
asset_root = str(_clean_path(asset_input)) if asset_input.strip() else ""
asset_root_path = Path(asset_root) if asset_root else None
if asset_root_path and asset_root_path.exists():
    with st.spinner("Đang lập danh mục ảnh và clip…"):
        asset_catalog = build_asset_catalog(
            asset_root, asset_root_path.stat().st_mtime
        )
else:
    asset_catalog = {"by_name": {}, "by_exact": {}}

stats = manifest.get("stats", {})
scene_index = manifest.get("scene_vector_index") or {}
frame_index = manifest.get("frame_vector_index") or {}

st.title("AIC Local Video Search")
st.caption(
    "SQLite FTS5/BM25 + OpenCLIP/FAISS + RRF · chạy hoàn toàn trên máy local"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenes", int(stats.get("scene_count", scene_index.get("count", 0))))
m2.metric("Keyframes", int(stats.get("keyframe_count", frame_index.get("count", 0))))
m3.metric("Videos", int(stats.get("video_count", len(video_ids))))
m4.metric(
    "Vector backend",
    str(scene_index.get("backend", "none")),
    "validation passed" if stats.get("validation_passed") else "check validation",
)

st.markdown(
    """
    <div class="aic-pipeline" role="img" aria-label="Luồng tìm kiếm local gồm năm bước">
      <div class="aic-step"><strong>1 · Query</strong><span>Mô tả cảnh hoặc chuỗi sự kiện</span></div>
      <div class="aic-step"><strong>2 · Planner</strong><span>Nhận diện OCR, ASR, action, temporal</span></div>
      <div class="aic-step"><strong>3 · Parallel search</strong><span>BM25 + scene/frame FAISS</span></div>
      <div class="aic-step"><strong>4 · RRF fusion</strong><span>Gộp rank và quality penalty</span></div>
      <div class="aic-step"><strong>5 · Results</strong><span>Scene, keyframe, clip và evidence</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

search_tab, sequence_tab, index_tab = st.tabs(
    ["🔎 Scene / frame search", "⏱️ Temporal sequence", "🗂️ Index inspector"]
)

with search_tab:
    mode_col, task_col = st.columns(2)
    with mode_col:
        search_mode = st.radio(
            "Cơ chế",
            options=["BM25 / FTS5", "Hybrid BM25 + FAISS"],
            horizontal=True,
        )
    with task_col:
        task = st.selectbox(
            "Loại truy vấn",
            options=["auto", "scene", "frame", "qa", "temporal"],
            index=1,
        )

    with st.form("search_form"):
        query = st.text_input(
            "Query",
            value="Công viên địa chất Lạng Sơn được UNESCO công nhận",
            placeholder="Mô tả cảnh bạn cần tìm…",
        )
        visual_query = st.text_input(
            "Visual query (không bắt buộc)",
            value="",
            placeholder="Ví dụ: a news report showing a UNESCO ceremony",
            help="Dùng riêng cho OpenCLIP. Để trống thì OpenCLIP encode Query ở trên.",
        )
        f1, f2, f3, f4 = st.columns(4)
        top_k = f1.slider("Top K", 1, 50, 10)
        selected_video = f2.selectbox("Video", ["Tất cả", *video_ids])
        use_time_filter = f3.checkbox("Lọc thời gian")
        match_all = f4.checkbox("Khớp tất cả từ")
        t1, t2 = st.columns(2)
        start_sec = t1.number_input(
            "Từ giây", min_value=0.0, value=0.0, disabled=not use_time_filter
        )
        end_sec = t2.number_input(
            "Đến giây", min_value=0.0, value=300.0, disabled=not use_time_filter
        )
        submitted = st.form_submit_button(
            "Tìm kiếm", type="primary", width="stretch"
        )

    if submitted:
        if not query.strip():
            st.warning("Hãy nhập query.")
        else:
            with st.spinner("Đang chạy các nhánh search và RRF…"):
                hits, elapsed, vector_error = run_search(
                    index_dir=index_dir,
                    query=query.strip(),
                    visual_query=visual_query,
                    use_vector=search_mode == "Hybrid BM25 + FAISS",
                    task=task,
                    top_k=top_k,
                    video_id=None if selected_video == "Tất cả" else selected_video,
                    start_sec=float(start_sec) if use_time_filter else None,
                    end_sec=float(end_sec) if use_time_filter else None,
                    match_all=match_all,
                    device=device,
                    embedding_model=str(manifest["scene_embedding_model"]),
                )
            st.session_state["aic_search_run"] = {
                "hits": hits,
                "elapsed": elapsed,
                "vector_error": vector_error,
            }

    if "aic_search_run" in st.session_state:
        render_search_results(
            st.session_state["aic_search_run"], asset_root, asset_catalog
        )

with sequence_tab:
    with st.form("sequence_form"):
        steps_text = st.text_area(
            "Mỗi dòng là một bước, theo đúng thứ tự thời gian",
            value=(
                "Cục Thú y nói về thịt heo\n"
                "Lạng Sơn nhận bằng công nhận của UNESCO"
            ),
            height=120,
        )
        s1, s2, s3 = st.columns(3)
        sequence_top_k = s1.slider("Top K chuỗi", 1, 10, 5)
        max_gap_sec = s2.number_input(
            "Khoảng cách tối đa (giây)", min_value=1.0, value=120.0
        )
        sequence_vector = s3.checkbox("Dùng OpenCLIP + FAISS", value=False)
        sequence_submitted = st.form_submit_button(
            "Tìm chuỗi sự kiện", type="primary", width="stretch"
        )

    if sequence_submitted:
        steps = [line.strip() for line in steps_text.splitlines() if line.strip()]
        if len(steps) < 2:
            st.warning("Temporal search cần ít nhất hai dòng.")
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Đang tìm từng bước và ghép theo timeline…"):
                    selected_device = None if device == "auto" else device
                    with LocalHybridSearchEngine(
                        index_dir, device=selected_device
                    ) as engine:
                        sequences = engine.search_sequence(
                            steps,
                            top_k=sequence_top_k,
                            max_gap_sec=float(max_gap_sec),
                            use_vector=sequence_vector,
                        )
                st.session_state["aic_sequence_run"] = {
                    "sequences": sequences,
                    "elapsed": time.perf_counter() - started,
                }
            except Exception as exc:
                st.error(f"Temporal search lỗi: {type(exc).__name__}: {exc}")

    if "aic_sequence_run" in st.session_state:
        st.caption(
            f"Thời gian: {st.session_state['aic_sequence_run']['elapsed']:.3f} s"
        )
        render_sequence_results(
            st.session_state["aic_sequence_run"]["sequences"],
            asset_root,
            asset_catalog,
        )

with index_tab:
    st.subheader("Video trong index")
    if video_catalog:
        table = pd.DataFrame(video_catalog)
        table["duration"] = table["end_sec"].map(_fmt_time)
        table = table[["video_id", "scene_count", "duration"]]
        table.columns = ["Video ID", "Scenes", "Thời lượng"]
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("Database chưa có video.")

    st.subheader("Các nhánh đang dùng")
    branch_table = pd.DataFrame(
        [
            ("Caption", "SQLite FTS5 / BM25", "caption_vi, caption_en, event_text"),
            ("OCR", "SQLite FTS5 / BM25", "ocr_text, visible_text"),
            ("ASR", "SQLite FTS5 / BM25", "transcript, speech_summary"),
            ("Tags", "SQLite FTS5 / BM25", "entities, actions, attributes"),
            ("Event", "SQLite FTS5 / BM25", "temporal event + timestamp"),
            ("Scene vector", scene_index.get("backend", "none"), "scene embedding"),
            ("Frame vector", frame_index.get("backend", "none"), "keyframe embedding"),
        ],
        columns=["Nhánh", "Backend", "Dữ liệu"],
    )
    st.dataframe(branch_table, hide_index=True, width="stretch")

    with st.expander("Xem index_manifest.json"):
        st.json(manifest)
