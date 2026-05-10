"""
Compatibility analysis between PLAUD checkpoints.

Two checkpoints are *fully compatible* if every parameter in one has a
corresponding parameter of the same shape in the other. They are *partially
compatible* if some parameters match and some don't.

Full compatibility → blend all parameters.
Partial compatibility → blend matching parameters, fall back for the rest.
"""

from __future__ import annotations
import pandas as pd
import torch


def param_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(p.shape) for k, p in model.named_parameters()}


def pairwise_compatibility(models: dict[str, object]) -> pd.DataFrame:
    """Return a DataFrame with pairwise compatibility metrics.

    Columns: model_a, model_b, matching_params, total_params, fully_compatible
    """
    names = list(models.keys())
    shapes = {n: param_shapes(m) for n, m in models.items()}
    rows = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            s1, s2 = shapes[n1], shapes[n2]
            matching = sum(1 for k, v in s1.items() if k in s2 and s2[k] == v)
            total = max(len(s1), len(s2))
            fully = matching == len(s1) == len(s2)
            rows.append(
                dict(model_a=n1, model_b=n2,
                     matching_params=matching, total_params=total,
                     fully_compatible=fully)
            )
    return pd.DataFrame(rows)


def mismatched_params(model_a, model_b) -> list[tuple[str, tuple | None, tuple | None]]:
    """List parameters whose shapes differ (or are absent in one model)."""
    s1 = param_shapes(model_a)
    s2 = param_shapes(model_b)
    result = []
    for k in sorted(set(s1) | set(s2)):
        sha, shb = s1.get(k), s2.get(k)
        if sha != shb:
            result.append((k, sha, shb))
    return result


def compatible_keys(model_a, model_b) -> list[str]:
    """Return parameter names that can be blended (same shape in both models)."""
    s1 = param_shapes(model_a)
    s2 = param_shapes(model_b)
    return [k for k, v in s1.items() if k in s2 and s2[k] == v]
