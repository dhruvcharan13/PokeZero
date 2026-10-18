from __future__ import annotations

import math


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def fit_elo(
    names: list[str],
    results: dict[tuple[str, str], list[float]],
    anchor: str | None = None,
    iters: int = 200,
) -> dict[str, float]:
    """Maximum-likelihood Elo from pairwise scores (1 win, 0.5 draw, 0 loss for the first name).

    Plain gradient ascent on the Bradley-Terry log-likelihood; ratings are shifted so the
    ``anchor`` (default: first name) sits at 1000.
    """
    r = {n: 0.0 for n in names}
    k = 400.0 / math.log(10)
    for _ in range(iters):
        grad = {n: 0.0 for n in names}
        for (a, b), scores in results.items():
            if not scores:
                continue
            for s in scores:
                p = expected_score(r[a], r[b])
                grad[a] += s - p
                grad[b] -= s - p
        for n in names:
            r[n] += 0.5 * k * grad[n] / max(1, sum(len(v) for v in results.values()) / len(names))
            r[n] = max(-1500.0, min(1500.0, r[n]))
    shift = 1000.0 - r[anchor or names[0]]
    return {n: r[n] + shift for n in names}
