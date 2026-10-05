"""Проверки сквозного сценария: харнесс должен годиться шагом в CI."""

import json

from evalkit.cli import main


def test_example_runs_and_passes_thresholds(capsys):
    code = main(["run", "examples/extract_items/config.yaml", "--no-color"])
    out = capsys.readouterr().out
    assert "extract_items" in out
    assert code == 0


def test_failed_threshold_gives_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXTRACT_VARIANT", "v2")
    code = main(["run", "examples/extract_items/config.yaml", "--no-color"])
    assert code == 1
    assert "Пороги не пройдены" in capsys.readouterr().out


def test_regression_between_variants_is_reported(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.json"
    monkeypatch.setenv("EXTRACT_VARIANT", "v1")
    main(["run", "examples/extract_items/config.yaml", "--no-color", "--save", str(base)])

    monkeypatch.setenv("EXTRACT_VARIANT", "v2")
    code = main([
        "run", "examples/extract_items/config.yaml", "--no-color",
        "--baseline", str(base), "--fail-on-regression",
    ])
    out = capsys.readouterr().out
    assert "Сломалось" in out and "long-names" in out
    assert "Починилось" in out and "fractional-qty" in out
    assert code == 1


def test_saved_run_is_readable_json(tmp_path):
    path = tmp_path / "r.json"
    main(["run", "examples/extract_items/config.yaml", "--no-color", "--save", str(path)])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "extract_items"
    assert len(data["cases"]) == 10


def test_compare_command_without_regressions_exits_zero(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    main(["run", "examples/extract_items/config.yaml", "--no-color", "--save", str(a)])
    main(["run", "examples/extract_items/config.yaml", "--no-color", "--save", str(b)])
    code = main(["compare", str(a), str(b), "--no-color", "--fail-on-regression"])
    assert "Регрессий нет" in capsys.readouterr().out
    assert code == 0


def test_metrics_command_lists_available(capsys):
    assert main(["metrics"]) == 0
    assert "json_valid" in capsys.readouterr().out
