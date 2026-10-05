"""Типы, конфигурация и загрузка наборов.

Набор — это jsonl, одна строка на случай. Обязательных полей два: id и input.
Всё остальное, включая эталон, задача и метрики трактуют сами.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class Case:
    """Один случай из набора."""

    id: str
    input: Any
    expected: Any = None
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Ответ задачи на один случай.

    Токены задача сообщает сама: только она знает, какой был вызов.
    Без них стоимость не считается, но прогон не падает.
    """

    output: Any
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


@dataclass
class CaseResult:
    case_id: str
    tags: list[str]
    scores: dict[str, float]
    latency_ms: float
    cost_usd: float
    stability: float | None
    outputs: list[Any]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(v >= 1.0 for v in self.scores.values())


@dataclass
class RunResult:
    name: str
    started_at: str
    config_path: str
    cases: list[CaseResult]
    aggregates: dict[str, float]
    threshold_failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "config_path": self.config_path,
            "aggregates": self.aggregates,
            "threshold_failures": self.threshold_failures,
            "cases": [asdict(c) for c in self.cases],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RunResult":
        return RunResult(
            name=d["name"],
            started_at=d["started_at"],
            config_path=d.get("config_path", ""),
            cases=[CaseResult(**c) for c in d["cases"]],
            aggregates=d["aggregates"],
            threshold_failures=d.get("threshold_failures", []),
        )


@dataclass
class Config:
    name: str
    task: str
    dataset: str
    metrics: list[dict[str, Any]]
    thresholds: dict[str, float] = field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    repeats: int = 1
    base_dir: Path = field(default_factory=Path)

    @staticmethod
    def load(path: str | Path) -> "Config":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        missing = [k for k in ("name", "task", "dataset", "metrics") if k not in raw]
        if missing:
            raise ValueError(f"в конфиге {path} нет обязательных полей: {', '.join(missing)}")
        repeats = int(raw.get("repeats", 1))
        if repeats < 1:
            raise ValueError("repeats должен быть не меньше 1")
        return Config(
            name=raw["name"],
            task=raw["task"],
            dataset=raw["dataset"],
            metrics=raw["metrics"],
            thresholds=raw.get("thresholds", {}) or {},
            pricing=raw.get("pricing", {}) or {},
            repeats=repeats,
            base_dir=path.parent,
        )

    def dataset_path(self) -> Path:
        p = Path(self.dataset)
        return p if p.is_absolute() else (self.base_dir / p)


def load_cases(path: str | Path) -> list[Case]:
    """Читает jsonl. Пустые строки пропускаются, дубли id — ошибка."""
    path = Path(path)
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno} — строка не разбирается как json: {e}") from e
        if "id" not in raw or "input" not in raw:
            raise ValueError(f"{path}:{lineno} — нужны поля id и input")
        case_id = str(raw["id"])
        if case_id in seen:
            raise ValueError(f"{path}:{lineno} — id {case_id} уже встречался")
        seen.add(case_id)
        cases.append(
            Case(
                id=case_id,
                input=raw["input"],
                expected=raw.get("expected"),
                tags=list(raw.get("tags", [])),
                meta=raw.get("meta", {}) or {},
            )
        )
    if not cases:
        raise ValueError(f"{path} — набор пустой")
    return cases


def load_task(ref: str) -> Callable[[Case], Prediction]:
    """Загружает функцию задачи по ссылке вида 'пакет.модуль:функция'."""
    if ":" not in ref:
        raise ValueError(f"ссылка на задачу должна быть вида 'модуль:функция', получено {ref!r}")
    module_name, func_name = ref.split(":", 1)
    import importlib

    module = importlib.import_module(module_name)
    try:
        func = getattr(module, func_name)
    except AttributeError as e:
        raise ValueError(f"в модуле {module_name} нет функции {func_name}") from e
    if not callable(func):
        raise ValueError(f"{ref} — не функция")
    return func
