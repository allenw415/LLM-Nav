from __future__ import annotations

import math

DEFAULT_EVIDENCE_SCORE_TEMPERATURE = 3.0


def normalize_positive_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in scores.values() if value > 0.0)
    if total <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: value / total for key, value in scores.items()}


def evidence_scores_to_distribution(
    scores: dict[str, float],
    *,
    temperature: float = DEFAULT_EVIDENCE_SCORE_TEMPERATURE,
) -> dict[str, float]:
    if not scores:
        return {}
    finite_scores = {
        key: float(value)
        for key, value in scores.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    if not finite_scores:
        return {key: 0.0 for key in scores}

    effective_temperature = max(float(temperature), 1e-6)
    max_score = max(finite_scores.values())
    weights = {
        key: math.exp((score - max_score) / effective_temperature)
        for key, score in finite_scores.items()
    }
    for key in scores:
        weights.setdefault(key, 0.0)
    return normalize_positive_scores(weights)
