from aic_local_search.fusion import reciprocal_rank_fusion


def test_rrf_rewards_documents_present_in_both_lists():
    fused = reciprocal_rank_fusion(
        [(["scene_a", "scene_b"], 1.0), (["scene_b", "scene_c"], 1.0)],
        rrf_k=60,
    )
    assert fused[0][0] == "scene_b"


def test_rrf_allows_branch_weights():
    fused = reciprocal_rank_fusion(
        [(["lexical"], 2.0), (["vector"], 1.0)], rrf_k=60
    )
    assert fused[0][0] == "lexical"

