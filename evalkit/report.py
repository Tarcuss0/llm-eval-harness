"""Вывод в консоль. Без зависимостей: обычный текст, выровненный по колонкам."""

from __future__ import annotations

from .compare import Comparison
from .core import RunResult

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _fmt(value: float, name: str) -> str:
    if "cost" in name:
        return f"${value:.5f}"
    if "latency" in name:
        return f"{value:.0f} мс"
    if name == "errors":
        return f"{value:.0f}"
    return f"{value:.3f}"


def _color(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{RESET}" if enabled else text


def render_run(result: RunResult, color: bool = True, show_cases: bool = False) -> str:
    lines = [f"Прогон: {result.name}   {result.started_at}", ""]

    width = max((len(k) for k in result.aggregates), default=10)
    lines.append("Итоги по набору")
    for name, value in result.aggregates.items():
        lines.append(f"  {name:<{width}}  {_fmt(value, name)}")
    lines.append("")

    failed = [c for c in result.cases if not c.passed]
    lines.append(f"Случаев: {len(result.cases)}, не прошли: {len(failed)}")

    if show_cases or failed:
        shown = result.cases if show_cases else failed
        for c in shown:
            mark = _color("ok  ", GREEN, color) if c.passed else _color("fail", RED, color)
            scores = "  ".join(f"{k}={v:.2f}" for k, v in c.scores.items())
            tail = f"  {_color(c.error, YELLOW, color)}" if c.error else ""
            stab = f"  стабильность={c.stability:.2f}" if c.stability is not None else ""
            lines.append(f"  {mark}  {c.case_id:<20} {scores}{stab}{tail}")
    lines.append("")

    if result.threshold_failures:
        lines.append(_color("Пороги не пройдены:", RED, color))
        for f in result.threshold_failures:
            lines.append(f"  - {f}")
    else:
        lines.append(_color("Все пороги пройдены.", GREEN, color))
    return "\n".join(lines)


def render_comparison(cmp: Comparison, color: bool = True) -> str:
    lines = ["Сравнение с базой", ""]

    width = max((len(m.name) for m in cmp.metrics), default=10)
    for m in cmp.metrics:
        # Стрелка показывает направление изменения, цвет — стало лучше или хуже.
        if abs(m.delta) < 1e-9:
            arrow, c = "=", DIM
        else:
            arrow = "↑" if m.delta > 0 else "↓"
            c = GREEN if m.improved else RED
        change = _color(f"{arrow} {_fmt(abs(m.delta), m.name)}", c, color)
        lines.append(f"  {m.name:<{width}}  {_fmt(m.base, m.name)}  →  {_fmt(m.new, m.name)}   {change}")
    lines.append("")

    if cmp.regressions:
        lines.append(_color(f"Сломалось ({len(cmp.regressions)}):", RED, color))
        for r in cmp.regressions:
            lines.append(f"  - {r.case_id}  {r.metric}: {r.base:.2f} → {r.new:.2f}")
    if cmp.new_errors:
        lines.append(_color("Новые ошибки:", RED, color))
        for e in cmp.new_errors:
            lines.append(f"  - {e}")
    if cmp.fixes:
        lines.append(_color(f"Починилось ({len(cmp.fixes)}):", GREEN, color))
        for f in cmp.fixes:
            lines.append(f"  - {f.case_id}  {f.metric}: {f.base:.2f} → {f.new:.2f}")
    if cmp.only_in_new:
        lines.append(f"Только в новом прогоне: {', '.join(cmp.only_in_new)}")
    if cmp.only_in_base:
        lines.append(f"Пропали из набора: {', '.join(cmp.only_in_base)}")

    if not cmp.has_regressions:
        lines.append("")
        lines.append(_color("Регрессий нет.", GREEN, color))
    return "\n".join(lines)
