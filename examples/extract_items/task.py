"""Пример задачи: извлечь позиции заказа из письма клиента.

Задача — обычная функция: получает случай, возвращает ответ и расход токенов.
Что внутри, харнесс не знает: вызов модели, цепочка, агент — безразлично.

По умолчанию работает демо-режим: вместо модели подставляется разбор на
регулярных выражениях, который ведёт себя как модель — на части писем
ошибается. Так репозиторий запускается без ключей и показывает осмысленный
результат. Переменная EXTRACT_VARIANT переключает две версии разбора,
чтобы было на чём показать сравнение прогонов.

Реальный вызов включается переменной LLM_API_KEY.
"""

from __future__ import annotations

import json
import os
import re

from evalkit.core import Case, Prediction

SYSTEM_PROMPT = (
    "Ты разбираешь письма клиентов оптового поставщика. "
    "Верни json вида {\"items\": [{\"name\": ..., \"qty\": ..., \"unit\": ...}]} "
    "без пояснений. Количество — число, единица измерения — строка."
)

UNITS = {
    "шт": "шт", "шт.": "шт", "штук": "шт", "штуки": "шт",
    "м": "м", "метр": "м", "метра": "м", "метров": "м", "м.": "м",
    "кг": "кг", "килограмм": "кг", "компл": "компл", "комплект": "компл",
    "уп": "уп", "упаковка": "уп", "упак": "уп",
}

# Строка позиции: название, затем количество и единица измерения.
LINE_V1 = re.compile(r"^[ \t]*[-–•*]?[ \t]*(?P<name>[^\d\n]+?)[ \t]+(?P<qty>\d+)[ \t]*(?P<unit>[а-яa-z.]+)[ \t]*$", re.I | re.M)
LINE_V2 = re.compile(
    r"^[ \t]*[-–•*]?[ \t]*(?P<name>[^\d\n]+?)[ \t—-]+(?P<qty>\d+(?:[.,]\d+)?)[ \t]*(?P<unit>[а-яa-z.]+)[ \t]*\.?$",
    re.I | re.M,
)


def _clean_name(raw: str) -> str:
    return re.sub(r"[\s,:—-]+$", "", raw.strip())


def _unit(raw: str) -> str:
    return UNITS.get(raw.strip().lower().rstrip("."), raw.strip().lower())


def _extract_demo(text: str, variant: str) -> dict:
    """Разбор, изображающий две версии промпта с разными слабыми местами."""
    if variant == "v1":
        # Первая версия обрывает разбор на итоговой строке и не берёт дробные количества.
        head = re.split(r"(?mi)^\s*итого\b", text)[0]
        items = [
            {"name": _clean_name(m.group("name")), "qty": int(m.group("qty")), "unit": _unit(m.group("unit"))}
            for m in LINE_V1.finditer(head)
        ]
        return {"items": items}

    # Вторая версия видит дробные количества и строки после «итого»,
    # но стала сокращать длинные названия — на части писем это ухудшение.
    items = []
    for m in LINE_V2.finditer(text):
        qty_raw = m.group("qty")
        qty: float | int = float(qty_raw.replace(",", ".")) if {",", "."} & set(qty_raw) else int(qty_raw)
        if isinstance(qty, float) and qty.is_integer():
            qty = int(qty)
        name = _clean_name(m.group("name"))
        words = name.split()
        if len(words) > 3:
            name = " ".join(words[:3])
        items.append({"name": name, "qty": qty, "unit": _unit(m.group("unit"))})
    return {"items": items}


def _call_model(text: str) -> Prediction:
    """Вызов модели через совместимый с OpenAI endpoint."""
    import httpx

    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    response = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    return Prediction(
        output=data["choices"][0]["message"]["content"],
        model=model,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def run(case: Case) -> Prediction:
    text = case.input if isinstance(case.input, str) else json.dumps(case.input, ensure_ascii=False)

    if os.environ.get("LLM_API_KEY"):
        return _call_model(text)

    variant = os.environ.get("EXTRACT_VARIANT", "v1")
    output = _extract_demo(text, variant)
    # Токены в демо-режиме оцениваются грубо: нужны, чтобы работал расчёт стоимости.
    return Prediction(
        output=output,
        model=f"demo-{variant}",
        input_tokens=len(text) // 4 + 60,
        output_tokens=len(json.dumps(output, ensure_ascii=False)) // 4,
    )
