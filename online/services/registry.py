"""Sổ đăng ký retrieval branch (PR-03).

`/v1/search/capabilities` phải mô tả đúng những gì đang chạy, bằng đúng
những id mà cấu hình per-branch dùng. Trước PR-03 hai thứ đó lệch nhau
(`bm25_caption_expanded` được công bố, nhưng fusion tra `bm25_caption`), nên
UI chỉnh weight cho branch đó thì không có gì xảy ra.

Registry là nơi duy nhất trả lời: có những branch nào, mỗi branch có những
execution nào, chạy trên backend gì, và điều khiển được bằng control nào.
"""

from __future__ import annotations

from collections import defaultdict

from online.domain.execution import BranchCapabilities

# Control mà MỌI branch đều hiểu (orchestrator/fusion xử lý, không phải adapter).
UNIVERSAL_CONTROLS = ("enabled", "weight", "top_k", "timeout_ms")


class RetrieverRegistry:
    """Gom retriever đang hoạt động và trả về capability đã introspect."""

    def __init__(self, retrievers: list) -> None:
        self._retrievers = list(retrievers)

    @property
    def retrievers(self) -> list:
        return list(self._retrievers)

    def resolve(self, execution_id: str):
        for retriever in self._retrievers:
            if getattr(retriever, "execution_id", getattr(retriever, "name", None)) == execution_id:
                return retriever
        raise KeyError(f"không có execution nào tên {execution_id!r}")

    def capabilities(self) -> list[BranchCapabilities]:
        grouped: dict[str, list] = defaultdict(list)
        for retriever in self._retrievers:
            branch_id = getattr(retriever, "branch_id", None) or getattr(
                retriever, "name", "unknown"
            )
            grouped[str(branch_id)].append(retriever)

        result: list[BranchCapabilities] = []
        for branch_id, members in sorted(grouped.items()):
            first = members[0]
            backend_kind = getattr(first, "backend_kind", "lexical")
            controls = sorted(
                set(getattr(first, "supported_controls", ())) | set(UNIVERSAL_CONTROLS)
            )
            result.append(
                BranchCapabilities(
                    branch_id=branch_id,
                    execution_ids=sorted(
                        str(getattr(item, "execution_id", branch_id)) for item in members
                    ),
                    modality=getattr(first, "modality", None),
                    backend_kind=backend_kind,
                    available=True,
                    # Backend fallback vẫn "available" nhưng phải bị đánh dấu
                    # degraded: nó trả kết quả, chỉ là không phải kết quả mà
                    # tên branch gợi ý.
                    degraded=backend_kind == "lexical_fallback",
                    degraded_reason=(
                        "hashing/BM25 trên text thay cho vector thật — không dùng để đo ablation dense"
                        if backend_kind == "lexical_fallback"
                        else None
                    ),
                    model_id=getattr(first, "model_id", None),
                    index_id=getattr(first, "index_id", None),
                    supported_controls=controls,
                )
            )
        return result


__all__ = ["UNIVERSAL_CONTROLS", "RetrieverRegistry"]
