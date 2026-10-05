"""Метрики.

Каждая метрика получает ответ задачи и случай и возвращает число от 0 до 1.
Единица — случай зачтён, ноль — нет, промежуточное значение допустимо там,
где совпадение частичное.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

Metric = Callable[[Any, Any], float]

_REGISTRY: dict[str, Callable[..., Metric]] = {}


def register(name: str):
    def wrap(factory: Callable[..., Metric]) -> Callable[..., Metric]:
        _REGISTRY[name] = factory
        return factory

    return wrap


def build(spec: dict[str, Any]) -> tuple[str, Metric]:
    """Собирает метрику из описания в конфиге."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind is None:
        raise ValueError("у метрики не указан type")
    if kind not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"неизвестная метрика {kind!r}, доступны: {known}")
    name = spec.pop("name", kind)
    return name, _REGISTRY[kind](**spec)


def as_data(output: Any) -> Any:
    """Приводит ответ к данным: строку пробует разобрать как json."""
    if isinstance(output, str):
        text = output.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return output
    return output


def normalize(value: Any) -> Any:
    """Приводит значение к виду, в котором сравнение не зависит от мелочей."""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().lower().replace("ё", "е")
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


# --- метрики -----------------------------------------------------------------


@register("json_valid")
def json_valid() -> Metric:
    """Ответ разбирается как json-объект или массив."""

    def metric(output: Any, expected: Any) -> float:
        data = as_data(output)
        return 1.0 if isinstance(data, (dict, list)) else 0.0

    return metric


@register("exact_match")
def exact_match(strict: bool = False) -> Metric:
    """Ответ совпадает с эталоном целиком."""

    def metric(output: Any, expected: Any) -> float:
        if strict:
            return 1.0 if output == expected else 0.0
        return 1.0 if normalize(as_data(output)) == normalize(expected) else 0.0

    return metric


@register("contains")
def contains(value: str | None = None, case_sensitive: bool = False) -> Metric:
    """Ответ содержит подстроку: либо заданную в конфиге, либо эталон."""

    def metric(output: Any, expected: Any) -> float:
        needle = value if value is not None else expected
        if needle is None:
            raise ValueError("метрике contains нужен value или expected в наборе")
        hay, needle_s = str(output), str(needle)
        if not case_sensitive:
            hay, needle_s = normalize(hay), normalize(needle_s)
        return 1.0 if needle_s in hay else 0.0

    return metric


@register("regex")
def regex(pattern: str, flags: str = "") -> Metric:
    """Ответ соответствует регулярному выражению."""
    f = 0
    for ch in flags:
        f |= {"i": re.I, "m": re.M, "s": re.S}.get(ch, 0)
    compiled = re.compile(pattern, f)

    def metric(output: Any, expected: Any) -> float:
        return 1.0 if compiled.search(str(output)) else 0.0

    return metric


@register("fields_match")
def fields_match(fields: list[str], partial: bool = True, field: str | None = None) -> Metric:
    """Сравнивает перечисленные поля ответа с эталоном.

    partial=True — доля совпавших полей, иначе всё или ничего. Если ответ
    и эталон списки, они сопоставляются по порядку, а разница в длине
    штрафуется. field указывает, из какого ключа брать сравниваемую часть.
    """
    if not fields:
        raise ValueError("метрике fields_match нужен непустой список fields")

    def score_one(got: Any, want: Any) -> float:
        if not isinstance(got, dict) or not isinstance(want, dict):
            return 0.0
        hits = sum(1 for f in fields if normalize(got.get(f)) == normalize(want.get(f)))
        return hits / len(fields)

    def pick(value: Any) -> Any:
        if field is None:
            return value
        return value.get(field) if isinstance(value, dict) else None

    def metric(output: Any, expected: Any) -> float:
        got, want = pick(as_data(output)), pick(expected)
        if isinstance(want, list):
            if not isinstance(got, list) or not want:
                return 0.0
            pairs = min(len(got), len(want))
            total = sum(score_one(got[i], want[i]) for i in range(pairs))
            score = total / max(len(got), len(want))
        else:
            score = score_one(got, want)
        if not partial:
            return 1.0 if score >= 1.0 else 0.0
        return score

    return metric


@register("numeric_close")
def numeric_close(field: str | None = None, tol: float = 0.01, relative: bool = False) -> Metric:
    """Числовое поле отличается от эталона не больше, чем на допуск."""

    def pick(value: Any) -> Any:
        if field is None:
            return value
        data = as_data(value)
        return data.get(field) if isinstance(data, dict) else None

    def metric(output: Any, expected: Any) -> float:
        try:
            got = float(pick(output))
            want = float(pick(expected) if field else expected)
        except (TypeError, ValueError):
            return 0.0
        limit = abs(want) * tol if relative else tol
        return 1.0 if abs(got - want) <= limit else 0.0

    return metric


@register("set_overlap")
def set_overlap(field: str | None = None) -> Metric:
    """Доля совпадения двух множеств значений (мера Жаккара)."""

    def pick(value: Any) -> list[Any]:
        data = as_data(value)
        if field is not None and isinstance(data, dict):
            data = data.get(field)
        return data if isinstance(data, list) else []

    def metric(output: Any, expected: Any) -> float:
        got = {json.dumps(normalize(v), ensure_ascii=False, sort_keys=True) for v in pick(output)}
        want = {json.dumps(normalize(v), ensure_ascii=False, sort_keys=True) for v in pick(expected)}
        if not got and not want:
            return 1.0
        if not got or not want:
            return 0.0
        return len(got & want) / len(got | want)

    return metric


def available() -> list[str]:
    return sorted(_REGISTRY)
