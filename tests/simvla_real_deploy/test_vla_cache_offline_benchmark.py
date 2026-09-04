from architectures.simvla.adapters.vla_cache.offline_benchmark import (
    build_balanced_query_plan,
)


def test_balanced_query_plan_preserves_episode_and_execution_cadence():
    samples = [
        *(('episode_a', index) for index in range(0, 51)),
        *(('episode_b', index) for index in range(10, 61)),
    ]
    plan = build_balanced_query_plan(samples, query_count=12, execution_horizon=5)
    assert len(plan) == 12
    assert {item['episode_id'] for item in plan} == {'episode_a', 'episode_b'}
    by_segment = {}
    for item in plan:
        by_segment.setdefault(item['segment_id'], []).append(item['frame_index'])
    assert all(
        right - left == 5
        for frames in by_segment.values()
        for left, right in zip(frames, frames[1:])
    )


def test_balanced_query_plan_rejects_an_oversized_request():
    samples = [('episode_a', index) for index in range(10)]
    try:
        build_balanced_query_plan(samples, query_count=4, execution_horizon=5)
    except ValueError as error:
        assert 'requested 4 sequential queries' in str(error)
    else:
        raise AssertionError('oversized query request should fail')
