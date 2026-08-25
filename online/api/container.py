"""Composition root: the only module that selects concrete infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from online.adapters.bm25 import LexicalRetriever
from online.adapters.object_lexicon import ObjectQueryTransform
from online.adapters.color_search import ColorSearchRetriever
from online.adapters.dense_retriever import DenseRetriever
from online.adapters.dense_text import CaptionDenseRetriever, build_text_encoder, JinaClipV2Encoder
from online.adapters.encoders import HashingTextEncoder, LocalTextEncoder, RemoteTextEncoder
from online.adapters.event_search import EventSearchRetriever, JsonlEventRepository
from online.adapters.fpt_advisor import FptEvidenceSelector, FptWeightRecommender
from online.adapters.fpt_client import FptClient
from online.adapters.fpt_query import (
    FptQueryExpander,
    FptQueryTranslator,
    TranslatingTextEncoder,
)
from online.adapters.frame_vector_store import (
    build_frame_vector_rows,
    build_frame_vector_rows_by_index,
)
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.ocr_fuzzy import OcrFuzzyRetriever
from online.adapters.qa_llm import FptQaAnswerer
from online.adapters.rerank import (
    BgeTextReranker,
    FptTextReranker,
    FptVlmReranker,
    QwenVlReranker,
)
from online.adapters.draft_store import JsonlDraftStore
from online.adapters.session_store import InMemorySessionStore
from online.adapters.vector_stores import InMemoryVectorStore, QdrantVectorStore
from online.config import Settings
from online.services.keyword_extraction import keyword_query
from online.services.query_expansion import QueryExpansionRetriever
from online.services.query_prep import PreparedQueryPlanner
from online.services.evidence_builder import EvidenceBuilder
from online.services.qa import QaProcessor
from online.services.avs import AvsConfig
from online.services.keyword_extraction import CorpusIdf
from online.services.rerank_pipeline import RerankPipeline
from online.services.rules import RuleConfig
from online.services.search import SearchService
from online.services.vqa import VQAService


_ENCODER_KIND_SUFFIXES = (":clip", ":siglip", ":jina")


def parse_dense_indexes(spec: str) -> list[tuple[str, str, str | None]]:
    """`name:model_path[:kind]` (ngăn cách bằng dấu phẩy) -> `[(name, path, kind)]`.

    KHÔNG tách bừa theo mọi dấu hai chấm: đường dẫn Windows (`D:/models/x`) có
    dấu hai chấm ở giữa, và tách bừa sẽ cắt nát nó. Tách đúng MỘT lần cho tên,
    rồi chỉ bóc `kind` khi phần còn lại kết thúc bằng một hậu tố hợp lệ.
    """

    out: list[tuple[str, str, str | None]] = []
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        name, separator, rest = item.partition(":")
        if not separator or not rest:
            raise ValueError(
                f"AIC_DENSE_INDEXES: {item!r} sai định dạng, cần <name>:<model_path>[:<kind>]"
            )
        kind: str | None = None
        for suffix in _ENCODER_KIND_SUFFIXES:
            if rest.casefold().endswith(suffix):
                kind = suffix[1:]
                rest = rest[: -len(suffix)]
                break
        out.append((name.strip(), rest.strip(), kind))
    return out


def parse_branch_weights(spec: str) -> dict[str, float]:
    """`bm25_ocr:0.2,color_search:0.1` -> `{"bm25_ocr": 0.2, ...}`.

    Trọng số phải nằm trong [0, 10] để khớp ràng buộc của `BranchRuntimeOptions`;
    ngoài khoảng thì báo ngay thay vì để pydantic ném ra giữa request đầu tiên.
    """

    out: dict[str, float] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        name, separator, raw = item.partition(":")
        if not separator:
            raise ValueError(
                f"AIC_BRANCH_WEIGHTS: {item!r} sai định dạng, cần <branch_id>:<weight>"
            )
        try:
            weight = float(raw)
        except ValueError as exc:
            raise ValueError(f"AIC_BRANCH_WEIGHTS: {raw!r} không phải số") from exc
        if not 0.0 <= weight <= 10.0:
            raise ValueError(
                f"AIC_BRANCH_WEIGHTS: trọng số {weight} của {name!r} ngoài [0, 10]"
            )
        out[name.strip()] = weight
    return out


def _assert_dimension_matches(
    name: str, probe: list[float], rows: list, *, hint: str = "AIC_DENSE_INDEXES"
) -> None:
    """Vector text và vector ảnh phải cùng chiều, nếu không cosine là rác.

    Đây đúng loại lỗi hệ này hay dính: sai model thì phép nhân VẪN chạy, VẪN ra
    số, thứ hạng VẪN có — chỉ là vô nghĩa. Chặn ngay lúc khởi động thay vì để
    phát hiện lúc đọc kết quả.
    """

    if not rows:
        return
    stored = len(rows[0][2])
    if len(probe) != stored:
        raise ValueError(
            f"index dense {name!r}: text encoder cho vector {len(probe)} chiều nhưng "
            f"vector ảnh trong export là {stored} chiều. Gần như chắc chắn là khai "
            f"sai model trong {hint} — phải ĐÚNG model đã sinh vector ảnh."
        )


def _read_dataset_manifest(metadata_jsonl: Path) -> dict | None:
    """Nội dung `dataset_manifest.json` cạnh export — dùng cho `dataset_version`
    (session trace) và dataset stats (`/v1/health`, UI competition studio).

    Không lỗi nếu thiếu file (vd metadata trỏ thẳng tới một .jsonl không đi
    kèm manifest) — caller khi đó chỉ nhận `None` và tự quyết định fallback.
    """

    manifest_path = metadata_jsonl.with_name("dataset_manifest.json")
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    repository: JsonlSceneRepository
    search_service: SearchService
    vqa_service: VQAService
    vector_store: object
    event_repository: JsonlEventRepository | None = None
    dataset_manifest: dict | None = None
    draft_store: JsonlDraftStore | None = None


async def build_container(
    settings: Settings, progress: "Callable[[str], None] | None" = None
) -> AppContainer:
    """Dựng toàn bộ hệ.

    `progress` nhận tên chặng đang chạy. Nó tồn tại vì việc này mất ~4 phút
    trên corpus thi đấu và người dùng nhìn màn hình trắng suốt quãng đó không
    phân biệt được "đang nạp" với "chết rồi" — xem `Boot` trong api/app.py.
    """

    def phase(name: str) -> None:
        if progress is not None:
            progress(name)

    phase("metadata")
    repository = await JsonlSceneRepository.load(settings.metadata_jsonl)
    dataset_manifest = _read_dataset_manifest(settings.metadata_jsonl)
    # LLM mở rộng query bằng đồng nghĩa TIẾNG VIỆT — chỉ dựng một lần rồi dùng
    # chung cho cả caption lẫn keyword để hai nhánh chia sẻ cache, không gọi
    # LLM hai lần cho cùng một câu.
    # Việc ngắn/máy móc dùng model "nhanh" (trả thẳng `content`); model chính
    # để dành cho QA. Chọn nhầm model reasoning ở đây thì mỗi lần dịch một câu
    # tốn ~1650 token và chỉ ra kết quả nếu max_tokens đủ lớn.
    fast_llm_model = settings.fpt_fast_llm_model or settings.fpt_llm_model
    query_expander = None
    if settings.enable_llm_expansion:
        if not (settings.fpt_enabled and fast_llm_model):
            raise ValueError(
                "AIC_ENABLE_LLM_EXPANSION=true nhưng chưa có LLM: cần "
                "AIC_FPT_ENABLED=true và AIC_FPT_FAST_LLM_MODEL (hoặc AIC_FPT_LLM_MODEL)"
            )
        query_expander = FptQueryExpander(
            FptClient.from_settings(settings), model_id=fast_llm_model
        )

    # Chỉ nhánh `keyword` được biến đổi truy vấn: caption/OCR/ASR là văn bản
    # tự nhiên nên khớp cả câu vẫn hợp lý, còn keyword là nhãn object ngắn.
    keyword_transform = keyword_query if settings.enable_keyword_extraction else None

    # `bm25_ocr` tách được ra khỏi nhóm bốn nhánh lexical vì nó là nhánh DUY NHẤT
    # đo được là gây hại khi bật mặc định (xem AIC_ENABLE_OCR_BRANCH trong
    # .env.fpt.local). Tắt nhánh KHÔNG làm mất dữ liệu OCR: `ocr_texts` vẫn đi
    # vào evidence pack mà QA đọc để TRẢ LỜI, và vẫn bật lại được cho từng
    # request qua `search_options.branches.bm25_ocr.weight`.
    lexical_fields = ["caption", "asr", "keyword"]
    if settings.enable_ocr_branch:
        lexical_fields.insert(1, "ocr")
    phase("lexical")
    lexical = []
    for field in lexical_fields:
        retriever = await LexicalRetriever.build(
            field, repository,
            query_transform=keyword_transform if field == "keyword" else None,
            drop_overlay_df=settings.ocr_overlay_df,
            overlay_max_words=settings.ocr_overlay_max_words,
        )
        # Phương án K: chỉ wrap caption/keyword — OCR/ASR phải giữ nguyên văn.
        if settings.enable_expansion and field in ("caption", "keyword"):
            retriever = QueryExpansionRetriever(retriever, expander=query_expander)
        lexical.append(retriever)

    local_frame_rows: list[tuple[str, str, list[float], dict]] = []
    has_real_embeddings = False
    # Chỉ khác None ở đường local + có embedding thật; backend qdrant không có
    # vector nào tại chỗ để so chiều.
    visual_encoder: LocalTextEncoder | None = None
    if settings.backend == "qdrant":
        encoder = RemoteTextEncoder(
            settings.embedding_url or "", settings.request_timeout_sec, settings.embedding_api_key
        )
        vector_store = QdrantVectorStore(
            settings.qdrant_url or "",
            settings.qdrant_scene_collection,
            settings.qdrant_vector_name,
            api_key=settings.qdrant_api_key,
            timeout_sec=settings.request_timeout_sec,
        )
    else:
        phase("vectors")
        local_frame_rows, has_real_embeddings = await build_frame_vector_rows(
            repository, settings.data_root, embedding_name=settings.visual_embedding_name or None
        )
        # Khai tên mà lọc ra RỖNG là lỗi cấu hình, không phải "không có
        # embedding": `has_real_embeddings` xét trước khi lọc nên vẫn True, và
        # để chạy tiếp thì `dense_visual` được đăng ký với vector store rỗng —
        # /capabilities quảng cáo `backend_kind: vector`, nhánh trả `empty` ở
        # MỌI truy vấn, không có gì báo lỗi. Đúng kiểu hỏng âm thầm mà
        # AIC_DENSE_INDEXES đã chặn sẵn; chặn ở đây cùng cách.
        if settings.visual_embedding_name and has_real_embeddings and not local_frame_rows:
            available = sorted(
                await build_frame_vector_rows_by_index(repository, settings.data_root)
            )
            raise ValueError(
                f"AIC_VISUAL_EMBEDDING_NAME={settings.visual_embedding_name!r} nhưng export không "
                f"có vector nào mang tên đó. Tên CÓ trong export: {available or '(không có)'}. "
                "Sinh vector bằng scripts/embed_export_keyframes.py, hoặc sửa tên cho khớp."
            )
        if has_real_embeddings:
            # Export có embedding thật (PR-13, scripts/embed_keyframes_local.py):
            # text encoder PHẢI cùng model CLIP để chung không gian embedding
            # với vector ảnh, nếu không cosine similarity vô nghĩa.
            encoder = LocalTextEncoder(
                settings.visual_embedding_model, revision=settings.visual_embedding_model_revision
            )
            # Giữ tham chiếu tới encoder CHƯA bọc: dùng để dò chiều vector sau
            # warmup. Dò qua bản đã bọc dịch sẽ tốn một lời gọi LLM cho chuỗi
            # "probe" — vô ích, và làm khởi động phụ thuộc mạng.
            visual_encoder = encoder
            # Text tower của CLIP chỉ biết tiếng Anh, mà truy vấn thi đấu là
            # tiếng Việt — không dịch thì nhánh này vẫn trả số nhưng số đó gần
            # như vô nghĩa. Bọc encoder chứ không sửa DenseRetriever: mọi thứ
            # phía sau (vector store, fusion) không cần biết có bước dịch.
            #
            # Trừ `jina` — cùng lý do đã ghi ở nhánh AIC_DENSE_INDEXES bên dưới:
            # text tower của nó đa ngữ sẵn, dịch là mất thông tin chứ không thêm.
            if settings.enable_query_translation and encoder.kind != "jina":
                if not (settings.fpt_enabled and fast_llm_model):
                    raise ValueError(
                        "AIC_ENABLE_QUERY_TRANSLATION=true nhưng chưa có LLM: cần "
                        "AIC_FPT_ENABLED=true và AIC_FPT_FAST_LLM_MODEL (hoặc AIC_FPT_LLM_MODEL)"
                    )
                encoder = TranslatingTextEncoder(
                    encoder,
                    FptQueryTranslator(
                        FptClient.from_settings(settings),
                        model_id=fast_llm_model,
                        cache_dir=settings.query_translation_cache_dir,
                    ),
                )
            vector_store = InMemoryVectorStore(local_frame_rows)
        else:
            encoder = HashingTextEncoder()
            rows = []
            for scene in await repository.all():
                search_text = " ".join(scene.captions + scene.keywords)
                rows.append(
                    (
                        scene.scene_id,
                        scene.video_id,
                        await encoder.encode(search_text),
                        {
                            "scene_id": scene.scene_id,
                            "video_id": scene.video_id,
                            "scene_idx": scene.scene_idx,
                            # start/end_frame đi kèm payload để candidate từ vector
                            # store cũng truy ngược được về khoảng frame, không phải
                            # chờ hydrate mới biết (PR-01 frame contract).
                            "start_frame": scene.start_frame,
                            "end_frame": scene.end_frame_exclusive - 1,
                            "start_sec": scene.start_sec,
                            "end_sec": scene.end_sec,
                            "has_ocr": bool(scene.ocr_texts),
                            "has_asr": bool(scene.asr_texts),
                        },
                    )
                )
            vector_store = InMemoryVectorStore(rows)

    # Backend local KHÔNG PHẢI lúc nào cũng dense visual thật: nếu export
    # chưa có embedding pack (PR-13 chưa chạy), HashingTextEncoder +
    # InMemoryVectorStore hash trên text caption — đăng ký đúng tên
    # `lexical_hash_fallback` để /capabilities không quảng cáo nhầm và
    # ablation không bị đọc sai (PR-03). Có embedding thật thì dùng cùng tên
    # `dense_visual` như nhánh qdrant vì bản chất giờ giống nhau (chỉ khác
    # backend lưu vector).
    if settings.backend == "qdrant" or has_real_embeddings:
        dense = DenseRetriever(
            encoder, vector_store, branch_id="dense_visual", backend_kind="vector"
        )
    else:
        dense = DenseRetriever(
            encoder,
            vector_store,
            branch_id="lexical_hash_fallback",
            backend_kind="lexical_fallback",
        )
    # Nạp model NGAY, ngoài request path. Không làm thì truy vấn đầu tiên
    # nuốt trọn ~3s thời gian nạp, vượt deadline nhánh và dense_visual bị bỏ
    # qua trong im lặng — đo được: 1-2 truy vấn đầu mỗi tiến trình cho ranking
    # khác hẳn các truy vấn sau.
    #
    # Warmup hỏng thì PHẢI chặn khởi động, KHÔNG được chỉ cảnh báo: export này
    # có embedding thật nên `dense_visual` đã được đăng ký và `/capabilities`
    # quảng cáo nó là available. Encoder không nạp được nghĩa là nhánh đó
    # `failed` ở MỌI request, trong khi server vẫn trả 200 — đúng kiểu hỏng
    # âm thầm khiến người đo tưởng đang chạy đủ nhánh. Cùng quy ước fail-fast
    # đã áp cho AIC_ENABLE_EVENT_SEARCH bên dưới.
    if has_real_embeddings and hasattr(encoder, "warmup"):
        phase("encoder")
        try:
            encoder.warmup()
        except Exception as exc:  # noqa: BLE001 - đổi thành lỗi cấu hình có ngữ cảnh
            raise ValueError(
                f"không nạp được text encoder {settings.visual_embedding_model!r}: {exc}. "
                "Export có embedding thật nên nhánh dense_visual bắt buộc cần encoder này. "
                "Trên máy không tải được từ HuggingFace, trỏ AIC_VISUAL_EMBEDDING_MODEL vào "
                "thư mục model đã tải sẵn (vd storage/models/clip-vit-large-patch14) — xem "
                "scripts/download_hf_model.py. Nếu lỗi trên nói về PHIÊN BẢN thư viện "
                "(vd 'huggingface-hub>=0.34.0,<1.0 is required ... but found 1.x') thì đó là "
                "xung đột dependency chứ không phải thiếu model: pip install "
                "'huggingface_hub>=0.34,<1' vào đúng venv đang chạy."
            ) from exc
        # Model và bộ vector giờ khai báo ĐỘC LẬP (AIC_VISUAL_EMBEDDING_MODEL và
        # AIC_VISUAL_EMBEDDING_NAME), nên khai lệch nhau là chuyện xảy ra được —
        # vd trỏ model CLIP 768 chiều vào bộ vector jina 1024 chiều. Cùng phép
        # chặn mà AIC_DENSE_INDEXES đã có, vì cùng một hậu quả: cosine vẫn ra số.
        #
        # CHỈ khi tên được khai tường minh: đó đúng là bậc tự do mới mà phép
        # chặn này canh. Đường cũ (không khai tên) giữ nguyên hành vi — và giữ
        # nguyên việc `encode()` không bị gọi lúc dựng container, thứ mà test
        # wiring dựa vào khi nó mock `warmup` để khỏi tải model thật về.
        if visual_encoder is not None and settings.visual_embedding_name:
            _assert_dimension_matches(
                settings.visual_embedding_name,
                await visual_encoder.encode("probe"),
                local_frame_rows,
                hint="AIC_VISUAL_EMBEDDING_MODEL/AIC_VISUAL_EMBEDDING_NAME",
            )

    dense_branches = [dense]

    # ---- Nhiều index dense song song (AIC_DENSE_INDEXES) ----
    # Mỗi model có KHÔNG GIAN EMBEDDING RIÊNG, nên phải là một vector store
    # riêng + một text encoder riêng, không gộp chung. Gộp lại thì cosine giữa
    # hai không gian vẫn ra số nhưng vô nghĩa.
    #
    # Đặt tên nhánh đồng nhất `dense_<name>` khi cờ này bật — KHÔNG giữ lẫn
    # `dense_visual` cho cái đầu rồi `dense_x` cho phần còn lại. Bật cờ là đổi
    # topology có chủ đích; tên lẫn lộn sẽ làm mọi cấu hình trọng số và mọi
    # `--disable-branch` đã lưu trỏ sai chỗ mà không báo.
    if settings.dense_indexes and settings.backend != "qdrant":
        declared = parse_dense_indexes(settings.dense_indexes)
        rows_by_index = await build_frame_vector_rows_by_index(
            repository, settings.data_root,
            embedding_names=[name for name, _path, _kind in declared],
        )
        missing = [name for name, _p, _k in declared if not rows_by_index.get(name)]
        if missing:
            # Quét LẠI không lọc để liệt kê tên thật sự có. Dùng `rows_by_index`
            # cho phần này là vô dụng: nó đã bị lọc theo đúng danh sách vừa
            # không khớp, nên luôn rỗng — mà "có những tên nào" mới là thông tin
            # người đọc cần để sửa.
            available = sorted(
                await build_frame_vector_rows_by_index(repository, settings.data_root)
            )
            raise ValueError(
                f"AIC_DENSE_INDEXES khai index {missing} nhưng export không có vector nào "
                f"mang `embedding_name` đó. Tên CÓ trong export: {available or '(không có index nào)'}. "
                "Sinh vector ở offline trước, hoặc sửa tên cho khớp."
            )
        dense_branches = []
        for name, model_path, kind in declared:
            rows = rows_by_index[name]
            index_encoder = LocalTextEncoder(
                model_path, revision=settings.visual_embedding_model_revision, kind=kind
            )
            try:
                index_encoder.warmup()
            except Exception as exc:  # noqa: BLE001 - đổi thành lỗi cấu hình có ngữ cảnh
                raise ValueError(
                    f"index dense {name!r}: không nạp được text encoder {model_path!r}: {exc}"
                ) from exc
            _assert_dimension_matches(name, await index_encoder.encode("probe"), rows)
            wrapped = index_encoder
            # Dịch CHỈ cho text tower tiếng Anh. `jina` (jina-clip) dùng
            # jina-XLM-RoBERTa đa ngữ: đo được nó phân biệt tiếng Việt ngang
            # tiếng Anh (cosine giữa các câu khác nghĩa 0.260 so với 0.262),
            # trong khi CLIP là 0.912 so với 0.448. Dịch cho nó là mất thông tin
            # (ghép vi↔en chỉ 0.820, tức bản dịch KHÔNG bằng bản gốc), cộng thêm
            # 0.39s và một phụ thuộc mạng cho đúng cái nhánh sinh ra để khỏi cần
            # mạng. Xem docs/20 § VISUAL-01.
            if settings.enable_query_translation and kind != "jina":
                wrapped = TranslatingTextEncoder(
                    index_encoder,
                    FptQueryTranslator(
                        FptClient.from_settings(settings),
                        model_id=fast_llm_model,
                        cache_dir=settings.query_translation_cache_dir,
                    ),
                )
            dense_branches.append(
                DenseRetriever(
                    wrapped,
                    InMemoryVectorStore(rows),
                    branch_id=f"dense_{name}",
                    backend_kind="vector",
                )
            )

    retrievers = [*dense_branches, *lexical]
    if settings.enable_ocr_fuzzy:
        retrievers.append(await OcrFuzzyRetriever.build(repository))
    if settings.enable_object_search:
        # Nhãn Open Images là tiếng Anh, truy vấn thi đấu là tiếng Việt, nên
        # không có bước dịch này nhánh khớp đúng 0 token — đo được 0,0% trên
        # 96/120 truy vấn gold. Tra bảng dựng sẵn, không gọi LLM lúc truy vấn.
        # Thiếu từ điển thì `load()` trả None và nhánh giữ nguyên hành vi cũ.
        retrievers.append(
            await LexicalRetriever.build(
                "object", repository, query_transform=ObjectQueryTransform.load()
            )
        )
    if settings.enable_action_search:
        retrievers.append(await LexicalRetriever.build("action", repository))
    if settings.enable_color_search:
        retrievers.append(await ColorSearchRetriever.build(repository))

    # ---- Nhánh dense trên TEXT caption (DENSE-TEXT-01) ----
    # Rỗng = tắt, cùng quy ước với AIC_DENSE_INDEXES. Nhánh này khớp truy vấn
    # với embedding VĂN BẢN của caption/tag, khác hẳn `dense_visual` khớp với
    # embedding ẢNH — nó là nhánh duy nhất chịu được "cùng nghĩa, khác từ" ở
    # phía lexical, chỗ mà 6 nhánh BM25 không giúp được gì.
    if settings.caption_dense_index:
        # `for_passages=False`: phía online luôn là query side. Script dựng
        # index gọi CÙNG factory với `for_passages=True`, nên cơ chế bất đối
        # xứng (prefix của E5 / LoRA adapter của jina) không lệch được.
        caption_dense = CaptionDenseRetriever(
            Path(settings.caption_dense_index),
            build_text_encoder(
                settings.caption_dense_encoder,
                settings.caption_dense_model,
                for_passages=False,
            ),
        )
        # Bốn chốt, cùng tinh thần fail-fast đã áp cho AIC_DENSE_INDEXES: mọi
        # ca dưới đây đều KHÔNG tự báo lỗi nếu để chạy tiếp — nhánh vẫn trả
        # candidate, `branch_status` vẫn `success`, chỉ có kết quả là vô nghĩa.
        corpus_scene_ids = [scene.scene_id for scene in await repository.all()]
        coverage = caption_dense.assert_covers(corpus_scene_ids)
        if coverage < 0.98:
            missing_videos = sorted({
                sid.rsplit("_S", 1)[0]
                for sid in corpus_scene_ids
                if sid not in set(caption_dense.scene_ids)
            })
            print(
                f"[WARN] caption_dense coverage {coverage:.1%} < 98% "
                f"({len(caption_dense.scene_ids):,} index / {len(corpus_scene_ids):,} corpus). "
                f"Missing videos: {missing_videos[:5]}. "
                "Nhánh vẫn chạy với index hiện có."
            )
        caption_dense.assert_encoder_kind(caption_dense.encoder)
        try:
            caption_dense.encoder.warmup()
        except Exception as exc:  # noqa: BLE001 - đổi thành lỗi cấu hình có ngữ cảnh
            raise ValueError(
                f"không nạp được text encoder caption dense "
                f"{settings.caption_dense_model!r} (kind={settings.caption_dense_encoder!r}): "
                f"{exc}. Trên máy không tải được từ HuggingFace, trỏ "
                "AIC_CAPTION_DENSE_MODEL vào thư mục đã tải sẵn "
                "(vd storage/models/multilingual-e5-large)."
            ) from exc
        probe = caption_dense.encoder.encode([caption_dense.query_prefix + "probe"])[0]
        caption_dense.assert_dimension(probe)
        retrievers.append(caption_dense)

    phase("events")
    events_path = settings.metadata_jsonl.with_name("events.jsonl")
    event_repository = await JsonlEventRepository.load(events_path) if events_path.exists() else None
    if settings.enable_event_search:
        if event_repository is None:
            raise ValueError(
                f"AIC_ENABLE_EVENT_SEARCH is set but {events_path} does not exist — "
                "run the offline pipeline/exporter (which now always writes events.jsonl) first"
            )
        retrievers.append(await EventSearchRetriever.build(event_repository))

    evidence_builder = EvidenceBuilder(
        repository,
        model_versions={
            key: value
            for key, value in (
                # Ghi ĐÚNG model sẽ chạy thật, không phải model được cấu hình:
                # bật FPT thì đường tự-host bị bỏ qua, mà trace vẫn ghi tên cũ
                # thì so hai run sẽ tưởng cùng model trong khi khác hẳn.
                (
                    "text_reranker",
                    settings.fpt_rerank_model
                    if (settings.fpt_enabled and settings.fpt_rerank_model)
                    else (settings.rerank_text_model if settings.rerank_text_url else ""),
                ),
                (
                    "vlm_reranker",
                    settings.fpt_vlm_model
                    if (settings.fpt_enabled and settings.fpt_vlm_model)
                    else (settings.rerank_vlm_model if settings.rerank_vlm_url else ""),
                ),
                ("dense_backend", settings.backend),
            )
            if value
        },
    )
    # Ưu tiên FPT khi bật (PR-15) — cùng chiến lược "chỉ đổi tên model qua env"
    # đã áp dụng cho embedding/enrichment: production tự chuyển về
    # BgeTextReranker (worker tự host) chỉ bằng cách tắt AIC_FPT_ENABLED.
    text_reranker = None
    if settings.fpt_enabled and settings.fpt_rerank_model:
        text_reranker = FptTextReranker(FptClient.from_settings(settings), model_id=settings.fpt_rerank_model)
    elif settings.rerank_text_url:
        text_reranker = BgeTextReranker(
            settings.rerank_text_url,
            model_id=settings.rerank_text_model,
            timeout_sec=settings.request_timeout_sec,
            api_key=settings.rerank_api_key,
        )
    # VLM rerank: cùng chiến lược "FPT ưu tiên" như text rerank ở trên, nhưng
    # KHÔNG dùng chung adapter được — QwenVlReranker nói contract của worker tự
    # host, FPT chỉ có /chat/completions (xem FptVlmReranker).
    vlm_reranker = None
    if settings.enable_vlm_rerank and settings.fpt_enabled and settings.fpt_vlm_model:
        vlm_reranker = FptVlmReranker(
            FptClient.from_settings(settings),
            model_id=settings.fpt_vlm_model,
            data_root=settings.data_root,
            frames_per_candidate=settings.rerank_vlm_frames_per_candidate,
            max_concurrency=settings.fpt_max_concurrency,
        )
    elif settings.enable_vlm_rerank and settings.rerank_vlm_url:
        vlm_reranker = QwenVlReranker(
            settings.rerank_vlm_url,
            model_id=settings.rerank_vlm_model,
            timeout_sec=max(settings.request_timeout_sec, 30.0),
            api_key=settings.rerank_api_key,
        )
    rerank_pipeline = RerankPipeline(
        evidence_builder,
        text_reranker=text_reranker,
        vlm_reranker=vlm_reranker,
    )
    # QA answer generation: FPT LLM ưu tiên khi bật (cùng chiến lược với
    # rerank ở trên) — rule-based ANSWER_TOOLS vẫn luôn chạy làm baseline vì
    # score_qa chấm bất kỳ dòng nào trong submission, không chỉ rank 1.
    qa_llm_answerer = None
    if settings.fpt_enabled and settings.fpt_llm_model:
        qa_llm_answerer = FptQaAnswerer(
            FptClient.from_settings(settings),
            model_id=settings.fpt_llm_model,
            max_tokens=settings.fpt_qa_max_tokens,
        )
    # IDF dựng MỘT LẦN rồi dùng chung cho AVS lẫn verifier của QA — hai nơi hỏi
    # cùng một câu ("cụm này có phân biệt được gì không") trên cùng một corpus.
    corpus_idf = CorpusIdf.from_scenes(await repository.all())
    qa_processor = QaProcessor(
        llm_answerer=qa_llm_answerer,
        llm_top_n=settings.fpt_qa_top_n,
        llm_rank_mode=settings.qa_llm_rank_mode,
        idf=corpus_idf,
        min_answer_idf=settings.qa_min_answer_idf,
    )

    # Cố vấn LLM — chỉ dựng khi có model reasoning. Cả hai đều là tuỳ chọn
    # theo từng request (`recommend_weights` / `select_evidence`), nên dựng sẵn
    # không tốn gì cho các request không hỏi tới.
    weight_recommender = None
    evidence_selector = None
    if settings.fpt_enabled and settings.fpt_llm_model:
        client = FptClient.from_settings(settings)
        weight_recommender = FptWeightRecommender(client, model_id=settings.fpt_llm_model)
        evidence_selector = FptEvidenceSelector(client, model_id=settings.fpt_llm_model)

    search_service = SearchService(
        repository,
        retrievers,
        planner=PreparedQueryPlanner() if settings.enable_query_prep else None,
        candidate_limit=settings.candidate_limit,
        rrf_k=settings.rrf_k,
        rule_config=RuleConfig() if settings.enable_rules else None,
        rerank_pipeline=rerank_pipeline,
        evidence_builder=evidence_builder,
        qa_processor=qa_processor,
        # PR-09: mọi search đi qua SearchService.search() đều được ghi trace —
        # kể cả gọi qua endpoint convenience /search/kis, không chỉ /v1/search.
        session_store=InMemorySessionStore(),
        dataset_version=(dataset_manifest or {}).get("build_id"),
        branch_timeout_ms=settings.branch_timeout_ms,
        avs_config=AvsConfig(
            grade_mode=settings.avs_grade_mode,
            soft_lambda=settings.avs_soft_lambda,
            semantic_tau=settings.avs_semantic_tau,
            max_per_video=settings.avs_max_per_video,
        ),
        # AVS-CRITERIA-01: `AvsCriteria.grade` chấm bằng độ phủ token có trọng
        # số IDF. Dựng IDF một lần lúc khởi động từ chính văn bản mà cổng grade
        # sẽ đọc; thiếu nó thì rơi về độ phủ không trọng số và mất khả năng
        # phân biệt `người`/`đang` với `thợ lặn`/`rùa biển`.
        avs_idf=corpus_idf,
        playback_pad_sec=settings.playback_pad_sec,
        fusion_method=settings.fusion_method,
        branch_weights=parse_branch_weights(settings.branch_weights),
        fusion_method_qa=settings.fusion_method_qa,
        trake_engine=settings.trake_engine,
        trake_solver=settings.trake_solver,
        trake_gap_penalty=settings.trake_gap_penalty,
        trake_candidate_limit=settings.trake_candidate_limit,
        trake_allow_missing_steps=settings.trake_allow_missing_steps,
        trake_min_covered_steps=settings.trake_min_covered_steps,
        trake_missing_step_penalty=settings.trake_missing_step_penalty,
        trake_beam_size=settings.trake_beam_size,
        trake_per_video_beam=settings.trake_per_video_beam,
        trake_max_chains_per_video=settings.trake_max_chains_per_video,
        trake_free_gap_sec=settings.trake_free_gap_sec,
        trake_gap_cap=settings.trake_gap_cap,
        media_root=settings.data_root,
        weight_recommender=weight_recommender,
        evidence_selector=evidence_selector,
    )
    return AppContainer(
        settings=settings,
        repository=repository,
        search_service=search_service,
        vqa_service=VQAService(search_service, repository),
        vector_store=vector_store,
        event_repository=event_repository,
        dataset_manifest=dataset_manifest,
        draft_store=JsonlDraftStore(settings.draft_store_path),
    )
