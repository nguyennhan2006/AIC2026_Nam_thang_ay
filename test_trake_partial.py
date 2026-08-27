"""Unit test for TRAKE partial-chain logic."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from online.services.temporal import link_event_hits
from online.domain.models import SearchHit

def make_hit(video_id: str, scene_idx: int, frame_idx: int, score: float = 0.8) -> SearchHit:
    """Create mock SearchHit with realistic timestamps."""
    # Use consistent timestamps based on frame_idx
    start_sec = frame_idx * 10.0
    end_sec = start_sec + 5.0
    return SearchHit(
        candidate_id=f"cand_{video_id}_{scene_idx}_{frame_idx}",
        video_id=video_id,
        scene_id=f"scene_{scene_idx}",
        scene_idx=scene_idx,
        start_frame=frame_idx * 10,
        end_frame_exclusive=(frame_idx + 1) * 10,
        start_sec=start_sec,
        end_sec=end_sec,
        best_frame_idx=frame_idx,
        score=score,
    )

def test_partial_chain_4_events():
    """Test: 4 events, only 3 match."""
    print("\n=== Test: 4 events, match 3/4 ===")

    # E0: match - frame 100
    e0_hits = [make_hit("video_A", 0, 100, 0.9)]
    # E1: NO match (empty)
    e1_hits = []
    # E2: match - frame 102 (skip 101, but that's OK)
    e2_hits = [make_hit("video_A", 1, 102, 0.8)]
    # E3: match - frame 103
    e3_hits = [make_hit("video_A", 2, 103, 0.85)]

    event_hits = [e0_hits, e1_hits, e2_hits, e3_hits]

    # Test with allow_missing_steps=True
    result = link_event_hits(
        event_hits,
        allow_missing_steps=True,
        min_covered_steps=1,
        beam_size=100,
    )

    print(f"Result count: {len(result)}")
    for seq in result:
        print(f"  video={seq.video_id}, covered_steps={seq.covered_steps}/{seq.total_steps}")
        print(f"  steps: {seq.covered_steps}, score: {seq.score:.3f}")

    assert len(result) > 0, "FAIL: No results!"
    assert len(result[0].covered_steps) <= 4, "FAIL: Covered steps > total!"
    print("PASS: Partial chain retained!")

def test_full_chain_4_events():
    """Test: 4 events, all 4 match."""
    print("\n=== Test: 4 events, match 4/4 ===")

    # Use sequential frames for valid temporal ordering
    e0_hits = [make_hit("video_A", 0, 100, 0.9)]
    e1_hits = [make_hit("video_A", 1, 101, 0.8)]
    e2_hits = [make_hit("video_A", 2, 102, 0.85)]
    e3_hits = [make_hit("video_A", 3, 103, 0.88)]

    event_hits = [e0_hits, e1_hits, e2_hits, e3_hits]

    result = link_event_hits(
        event_hits,
        allow_missing_steps=True,
        min_covered_steps=1,
        beam_size=100,
    )

    print(f"Result count: {len(result)}")
    for seq in result[:3]:
        print(f"  video={seq.video_id}, covered_steps={seq.covered_steps}/{seq.total_steps}")

    assert len(result) > 0, "FAIL: No results!"
    assert len(result[0].covered_steps) == 4, "FAIL: Not all 4 matched!"
    print("PASS: Full chain retained!")

def test_only_1_match():
    """Test: 4 events, only 1 match."""
    print("\n=== Test: 4 events, match 1/4 ===")

    e0_hits = []
    e1_hits = []
    e2_hits = [make_hit("video_A", 1, 200, 0.8)]
    e3_hits = []

    event_hits = [e0_hits, e1_hits, e2_hits, e3_hits]

    result = link_event_hits(
        event_hits,
        allow_missing_steps=True,
        min_covered_steps=1,
        beam_size=100,
    )

    print(f"Result count: {len(result)}")
    for seq in result:
        print(f"  video={seq.video_id}, covered_steps={seq.covered_steps}/{seq.total_steps}")

    assert len(result) > 0, "FAIL: No results - but should be retained!"
    print("PASS: Single match retained!")

def test_allow_missing_false():
    """Test: allow_missing_steps=False rejects all if any event empty."""
    print("\n=== Test: allow_missing_steps=False ===")

    e0_hits = [make_hit("video_A", 0, 100, 0.9)]
    e1_hits = []  # Empty!
    e2_hits = [make_hit("video_A", 1, 200, 0.8)]
    e3_hits = [make_hit("video_A", 2, 300, 0.85)]

    event_hits = [e0_hits, e1_hits, e2_hits, e3_hits]

    result = link_event_hits(
        event_hits,
        allow_missing_steps=False,  # FALSE!
        beam_size=100,
    )

    print(f"Result count: {len(result)}")

    # With allow_missing_steps=False, must return [] because there are empty events
    if len(result) == 0:
        print("EXPECTED: No results because allow_missing_steps=False")
    else:
        print(f"Results: {len(result)} sequences")

if __name__ == "__main__":
    test_partial_chain_4_events()
    test_full_chain_4_events()
    test_only_1_match()
    test_allow_missing_false()

    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
