import numpy as np

from tools.seer.analyze_independent_k4_confirmation import (
    hierarchical_teacher_bootstrap_ci,
    paired_bootstrap_ci,
)


def test_paired_bootstrap_preserves_difference_sign():
    positive = {task: [1.0] * 8 + [0.0] * 2 for task in range(10)}
    low, high = paired_bootstrap_ci(
        positive, n_bootstrap=1000, seed=7
    )
    assert 0.0 < low <= high <= 1.0


def test_hierarchical_bootstrap_is_deterministic():
    groups = {
        f"repeat_{repeat}": {
            task: [float((episode + task + repeat) % 2) for episode in range(10)]
            for task in range(10)
        }
        for repeat in range(1, 4)
    }
    first = hierarchical_teacher_bootstrap_ci(
        groups, n_bootstrap=500, seed=11
    )
    second = hierarchical_teacher_bootstrap_ci(
        groups, n_bootstrap=500, seed=11
    )
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
