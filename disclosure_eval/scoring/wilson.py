"""Wilson score confidence interval for a binomial proportion.

The Wilson interval behaves sensibly at the extremes (0 of n, n of n) and
for the small sample sizes typical of an evaluation run, which is why it is
preferred over the normal approximation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class WilsonInterval:
    """A proportion with its Wilson score interval.

    Attributes:
        successes: Number of positive outcomes.
        n: Number of trials.
        point: ``successes / n`` (``0.0`` when ``n == 0``).
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence: Confidence level used, for example ``0.95``.
    """

    successes: int
    n: int
    point: float
    lower: float
    upper: float
    confidence: float

    def format(self) -> str:
        """Render as ``"x/n (p%, CI lo-hi%)"``."""
        pct = self.point * 100
        return (
            f"{self.successes}/{self.n} ({pct:.1f}%, "
            f"{int(self.confidence * 100)}% CI {self.lower * 100:.1f}-{self.upper * 100:.1f}%)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "successes": self.successes,
            "n": self.n,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
        }


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> WilsonInterval:
    """Compute the Wilson score interval for ``successes`` out of ``n``.

    Args:
        successes: Count of positive outcomes, ``0 <= successes <= n``.
        n: Number of trials. When ``0`` the interval is ``[0, 1]`` (no
            information) with a point estimate of ``0.0``.
        confidence: Two-sided confidence level in ``(0, 1)``.

    Raises:
        ValueError: If the inputs are out of range.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"invalid counts: successes={successes}, n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return WilsonInterval(0, 0, 0.0, 0.0, 1.0, confidence)

    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half_width = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    return WilsonInterval(
        successes=successes,
        n=n,
        point=round(p, 6),
        lower=round(lower, 6),
        upper=round(upper, 6),
        confidence=confidence,
    )
