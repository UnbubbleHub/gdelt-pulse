"""Running-average centroid computation for clusters."""

from __future__ import annotations


def compute_new_centroid(
    current_centroid: list[float],
    new_embedding: list[float],
    current_count: int,
) -> list[float]:
    """Compute an updated centroid as a running average.

    Given the current centroid (average of `current_count` vectors) and a new
    embedding, returns the new centroid that is the average of all
    `current_count + 1` vectors — without needing to re-read them all.
    """
    n = current_count + 1
    return [
        (old * current_count + new) / n
        for old, new in zip(current_centroid, new_embedding)
    ]
