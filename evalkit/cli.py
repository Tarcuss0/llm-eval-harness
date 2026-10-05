"""Командная строка.

    python -m evalkit run examples/extract_items/config.yaml --save runs/new.json
    python -m evalkit compare runs/base.json runs/new.json

Код возврата отличен от нуля, если пороги не пройдены или нашлись регрессии,
поэтому харнесс можно ставить шагом в CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import compare
from .core import Config
from .metrics import available
from .report import render_comparison, render_run
from . import runner


def _add_run(sub: argparse._SubParsersAction) -> None:
    pr = sub.add_parser("run", help="прогнать набор по конфигу")
    pr.add_argument("config", help="путь к yaml-конфигу")
    pr.add_argument("--save", help="куда записать результат прогона (json)")
    pr.add_argument("--baseline", help="сравнить с этим прогоном сразу после запуска")
    pr.add_argument("--repeats", type=int, help="переопределить число повторов")
    pr.add_argument("--all", action="store_true", help="показать все случаи, не только упавшие")
    pr.add_argument("--no-color", action="store_true")
    pr.add_argument("--fail-on-regression", action="store_true", help="ненулевой код возврата при регрессиях")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="прогон и сравнение оценок качества LLM-задач")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)

    pc = sub.add_parser("compare", help="сравнить два сохранённых прогона")
    pc.add_argument("base")
    pc.add_argument("new")
    pc.add_argument("--no-color", action="store_true")
    pc.add_argument("--fail-on-regression", action="store_true")

    sub.add_parser("metrics", help="показать доступные метрики")

    args = parser.parse_args(argv)
    color = not getattr(args, "no_color", False)

    if args.command == "metrics":
        print("\n".join(available()))
        return 0

    if args.command == "compare":
        cmp = compare(runner.load(args.base), runner.load(args.new))
        print(render_comparison(cmp, color=color))
        return 1 if (args.fail_on_regression and cmp.has_regressions) else 0

    config = Config.load(args.config)
    if args.repeats:
        config.repeats = args.repeats
    # конфиг может ссылаться на задачу рядом с собой
    root = str(Path(config.base_dir).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    result = runner.run(config)
    print(render_run(result, color=color, show_cases=args.all))

    if args.save:
        path = runner.save(result, args.save)
        print(f"\nПрогон сохранён: {path}")

    code = 1 if result.threshold_failures else 0
    if args.baseline:
        cmp = compare(runner.load(args.baseline), result)
        print()
        print(render_comparison(cmp, color=color))
        if args.fail_on_regression and cmp.has_regressions:
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
