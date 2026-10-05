"""Прогон набора: вызов задачи, замер времени, стоимость, пороги.

Повторные прогоны одного случая нужны, чтобы увидеть разброс ответов модели.
Оценка берётся средняя по повторам, устойчивость считается отдельно.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import metrics as metrics_mod
from .core import Case, CaseResult, Config, Prediction, RunResult, load_cases, load_task


def cost_of(pred: Prediction, pricing: dict[str, dict[str, float]]) -> float:
    """Стоимость одного вызова. Цены задаются в долларах за миллион токенов."""
    if not pred.model or pred.model not in pricing:
        return 0.0
    prices = pricing[pred.model]
    cost = 0.0
    if pred.input_tokens:
        cost += pred.input_tokens / 1_000_000 * float(prices.get("input", 0.0))
    if pred.output_tokens:
        cost += pred.output_tokens / 1_000_000 * float(prices.get("output", 0.0))
    return cost


def _fingerprint(output: Any) -> str:
    try:
        return json.dumps(metrics_mod.normalize(metrics_mod.as_data(output)), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(output)


def _stability(outputs: list[Any]) -> float | None:
    """Доля самого частого ответа среди повторов."""
    if len(outputs) < 2:
        return None
    counts = Counter(_fingerprint(o) for o in outputs)
    return counts.most_common(1)[0][1] / len(outputs)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def run_case(
    case: Case,
    task: Callable[[Case], Prediction],
    scorers: list[tuple[str, metrics_mod.Metric]],
    pricing: dict[str, dict[str, float]],
    repeats: int,
) -> CaseResult:
    outputs: list[Any] = []
    latencies: list[float] = []
    cost = 0.0
    per_metric: dict[str, list[float]] = {name: [] for name, _ in scorers}
    error: str | None = None

    for _ in range(repeats):
        started = time.perf_counter()
        try:
            pred = task(case)
        except Exception as e:  # падение задачи — это результат прогона, а не повод всё уронить
            latencies.append((time.perf_counter() - started) * 1000)
            error = f"{type(e).__name__}: {e}"
            break
        latencies.append((time.perf_counter() - started) * 1000)
        if pred.error:
            error = pred.error
            break
        outputs.append(pred.output)
        cost += cost_of(pred, pricing)
        for name, scorer in scorers:
            try:
                per_metric[name].append(float(scorer(pred.output, case.expected)))
            except Exception as e:
                error = f"метрика {name}: {type(e).__name__}: {e}"
                per_metric[name].append(0.0)

    scores = {name: (statistics.fmean(vals) if vals else 0.0) for name, vals in per_metric.items()}
    return CaseResult(
        case_id=case.id,
        tags=case.tags,
        scores=scores,
        latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        cost_usd=cost / max(len(outputs), 1),
        stability=_stability(outputs),
        outputs=outputs,
        error=error,
    )


def aggregate(results: list[CaseResult], metric_names: list[str]) -> dict[str, float]:
    agg: dict[str, float] = {}
    for name in metric_names:
        agg[name] = statistics.fmean([r.scores.get(name, 0.0) for r in results]) if results else 0.0
    latencies = [r.latency_ms for r in results]
    agg["mean_latency_ms"] = statistics.fmean(latencies) if latencies else 0.0
    agg["p95_latency_ms"] = percentile(latencies, 0.95)
    agg["cost_per_case_usd"] = statistics.fmean([r.cost_usd for r in results]) if results else 0.0
    agg["total_cost_usd"] = sum(r.cost_usd for r in results)
    stabilities = [r.stability for r in results if r.stability is not None]
    if stabilities:
        agg["stability"] = statistics.fmean(stabilities)
    agg["pass_rate"] = (sum(1 for r in results if r.passed) / len(results)) if results else 0.0
    agg["errors"] = float(sum(1 for r in results if r.error))
    return agg


# Для этих величин порог означает «не выше», для остальных — «не ниже».
UPPER_BOUND = ("latency", "cost", "errors")


def check_thresholds(agg: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for key, limit in thresholds.items():
        if key not in agg:
            failures.append(f"{key}: порог задан, но такой величины в прогоне нет")
            continue
        value = agg[key]
        upper = any(word in key for word in UPPER_BOUND)
        if upper and value > limit:
            failures.append(f"{key}: {value:.4g} выше порога {limit:.4g}")
        elif not upper and value < limit:
            failures.append(f"{key}: {value:.4g} ниже порога {limit:.4g}")
    return failures


def run(config: Config) -> RunResult:
    cases = load_cases(config.dataset_path())
    task = load_task(config.task)
    scorers = [metrics_mod.build(spec) for spec in config.metrics]
    names = [n for n, _ in scorers]
    if len(set(names)) != len(names):
        raise ValueError("имена метрик повторяются — задайте name в конфиге")

    results = [run_case(c, task, scorers, config.pricing, config.repeats) for c in cases]
    agg = aggregate(results, names)
    return RunResult(
        name=config.name,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_path=str(config.base_dir),
        cases=results,
        aggregates=agg,
        threshold_failures=check_thresholds(agg, config.thresholds),
    )


def save(result: RunResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path: str | Path) -> RunResult:
    return RunResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
