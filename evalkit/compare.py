"""Сравнение двух прогонов.

Среднее по набору скрывает размен: одно починили, другое сломали, цифра та же.
Поэтому сравниваются и средние, и каждый случай отдельно.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import RunResult
from .runner import UPPER_BOUND


@dataclass
class MetricDelta:
    name: str
    base: float
    new: float

    @property
    def delta(self) -> float:
        return self.new - self.base

    @property
    def improved(self) -> bool:
        """Для задержки, стоимости и ошибок рост — это ухудшение."""
        if any(word in self.name for word in UPPER_BOUND):
            return self.delta < 0
        return self.delta > 0


@dataclass
class CaseChange:
    case_id: str
    metric: str
    base: float
    new: float


@dataclass
class Comparison:
    metrics: list[MetricDelta] = field(default_factory=list)
    regressions: list[CaseChange] = field(default_factory=list)
    fixes: list[CaseChange] = field(default_factory=list)
    only_in_base: list[str] = field(default_factory=list)
    only_in_new: list[str] = field(default_factory=list)
    new_errors: list[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions or self.new_errors)


def compare(base: RunResult, new: RunResult, eps: float = 1e-9) -> Comparison:
    out = Comparison()

    for name in sorted(set(base.aggregates) | set(new.aggregates)):
        out.metrics.append(MetricDelta(name, base.aggregates.get(name, 0.0), new.aggregates.get(name, 0.0)))

    base_cases = {c.case_id: c for c in base.cases}
    new_cases = {c.case_id: c for c in new.cases}

    out.only_in_base = sorted(set(base_cases) - set(new_cases))
    out.only_in_new = sorted(set(new_cases) - set(base_cases))

    for case_id in sorted(set(base_cases) & set(new_cases)):
        b, n = base_cases[case_id], new_cases[case_id]
        if n.error and not b.error:
            out.new_errors.append(f"{case_id}: {n.error}")
        for metric in sorted(set(b.scores) | set(n.scores)):
            bv, nv = b.scores.get(metric, 0.0), n.scores.get(metric, 0.0)
            if nv < bv - eps:
                out.regressions.append(CaseChange(case_id, metric, bv, nv))
            elif nv > bv + eps:
                out.fixes.append(CaseChange(case_id, metric, bv, nv))
    return out
