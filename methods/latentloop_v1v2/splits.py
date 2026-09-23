"""Split-integrity helpers shared by V1 and V2 gates."""

from __future__ import annotations

from collections.abc import Mapping


def _keys(manifest: Mapping[str, object]) -> set[str]:
    values = manifest.get("episode_keys") or manifest.get("validation_episode_keys") or []
    return {str(value) for value in values}


def assert_disjoint_split_manifests(named: Mapping[str, Mapping[str, object]]) -> None:
    names = list(named)
    for index, left_name in enumerate(names):
        left = _keys(named[left_name])
        if not left:
            raise ValueError(f"split {left_name} has no episode keys")
        for right_name in names[index + 1 :]:
            overlap = left & _keys(named[right_name])
            if overlap:
                raise ValueError(
                    f"episode overlap between {left_name} and {right_name}: {sorted(overlap)[:5]}"
                )
