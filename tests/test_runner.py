import json

import pytest

from evalkit.compare import compare
from evalkit.core import Case, Config, Prediction, RunResult, load_cases, load_task
from evalkit import runner


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --- набор -------------------------------------------------------------------


def test_load_cases_skips_blank_lines(tmp_path):
    p = write(tmp_path, "d.jsonl", '{"id": "1", "input": "a"}\n\n{"id": "2", "input": "b"}\n')
    assert [c.id for c in load_cases(p)] == ["1", "2"]


def test_load_cases_rejects_duplicate_ids(tmp_path):
    p = write(tmp_path, "d.jsonl", '{"id": "1", "input": "a"}\n{"id": "1", "input": "b"}\n')
    with pytest.raises(ValueError, match="уже встречался"):
        load_cases(p)


def test_load_cases_reports_line_number(tmp_path):
    p = write(tmp_path, "d.jsonl", '{"id": "1", "input": "a"}\nне json\n')
    with pytest.raises(ValueError, match=":2"):
        load_cases(p)


def test_load_cases_requires_input(tmp_path):
    p = write(tmp_path, "d.jsonl", '{"id": "1"}\n')
    with pytest.raises(ValueError, match="id и input"):
        load_cases(p)


def test_empty_dataset_is_an_error(tmp_path):
    p = write(tmp_path, "d.jsonl", "\n\n")
    with pytest.raises(ValueError, match="пустой"):
        load_cases(p)


# --- стоимость ---------------------------------------------------------------


def test_cost_counts_input_and_output_separately():
    pred = Prediction(output="x", model="m", input_tokens=1_000_000, output_tokens=500_000)
    assert runner.cost_of(pred, {"m": {"input": 0.15, "output": 0.60}}) == pytest.approx(0.45)


def test_cost_is_zero_for_unknown_model():
    pred = Prediction(output="x", model="неизвестная", input_tokens=1000)
    assert runner.cost_of(pred, {"m": {"input": 1.0}}) == 0.0


def test_cost_survives_missing_token_counts():
    assert runner.cost_of(Prediction(output="x", model="m"), {"m": {"input": 1.0}}) == 0.0


# --- пороги ------------------------------------------------------------------


def test_latency_and_cost_thresholds_are_upper_bounds():
    agg = {"p95_latency_ms": 5000.0, "cost_per_case_usd": 0.02}
    failures = runner.check_thresholds(agg, {"p95_latency_ms": 3000, "cost_per_case_usd": 0.01})
    assert len(failures) == 2
    assert all("выше порога" in f for f in failures)


def test_quality_thresholds_are_lower_bounds():
    failures = runner.check_thresholds({"accuracy": 0.8}, {"accuracy": 0.9})
    assert "ниже порога" in failures[0]


def test_threshold_for_unknown_metric_is_reported():
    failures = runner.check_thresholds({"accuracy": 1.0}, {"опечатка": 0.5})
    assert "такой величины в прогоне нет" in failures[0]


def test_threshold_on_the_boundary_passes():
    assert runner.check_thresholds({"accuracy": 0.9}, {"accuracy": 0.9}) == []


# --- прогон ------------------------------------------------------------------


def make_config(tmp_path, task_ref, **over):
    write(
        tmp_path,
        "d.jsonl",
        '{"id": "a", "input": "раз", "expected": "раз"}\n{"id": "b", "input": "два", "expected": "два"}\n',
    )
    cfg = {
        "name": "t",
        "task": task_ref,
        "dataset": "d.jsonl",
        "metrics": [{"type": "exact_match"}],
        **over,
    }
    import yaml

    write(tmp_path, "c.yaml", yaml.safe_dump(cfg, allow_unicode=True))
    return Config.load(tmp_path / "c.yaml")


def echo(case: Case) -> Prediction:
    return Prediction(output=case.input, model="m", input_tokens=100, output_tokens=10)


def broken(case: Case) -> Prediction:
    raise RuntimeError("модель недоступна")


def test_run_scores_every_case(tmp_path):
    result = runner.run(make_config(tmp_path, "tests.test_runner:echo"))
    assert result.aggregates["exact_match"] == 1.0
    assert result.aggregates["pass_rate"] == 1.0
    assert len(result.cases) == 2


def test_task_failure_does_not_stop_the_run(tmp_path):
    result = runner.run(make_config(tmp_path, "tests.test_runner:broken"))
    assert result.aggregates["errors"] == 2.0
    assert all("модель недоступна" in c.error for c in result.cases)
    assert all(not c.passed for c in result.cases)


def test_repeats_measure_stability(tmp_path):
    calls = {"n": 0}

    def flaky(case: Case) -> Prediction:
        calls["n"] += 1
        return Prediction(output=case.input if calls["n"] % 2 else "мимо", model="m")

    import tests.test_runner as mod

    mod.flaky = flaky
    result = runner.run(make_config(tmp_path, "tests.test_runner:flaky", repeats=4))
    assert all(c.stability == 0.5 for c in result.cases)


def test_single_repeat_has_no_stability(tmp_path):
    result = runner.run(make_config(tmp_path, "tests.test_runner:echo"))
    assert all(c.stability is None for c in result.cases)


def test_cost_per_case_is_not_multiplied_by_repeats(tmp_path):
    pricing = {"m": {"input": 1.0, "output": 1.0}}
    one = runner.run(make_config(tmp_path, "tests.test_runner:echo", pricing=pricing))
    three = runner.run(make_config(tmp_path, "tests.test_runner:echo", repeats=3, pricing=pricing))
    assert one.aggregates["cost_per_case_usd"] == pytest.approx(three.aggregates["cost_per_case_usd"])


def test_duplicate_metric_names_are_rejected(tmp_path):
    cfg = make_config(
        tmp_path,
        "tests.test_runner:echo",
        metrics=[{"type": "exact_match"}, {"type": "exact_match"}],
    )
    with pytest.raises(ValueError, match="повторяются"):
        runner.run(cfg)


def test_percentile_interpolates():
    assert runner.percentile([10, 20, 30, 40], 0.5) == pytest.approx(25.0)
    assert runner.percentile([5], 0.95) == 5.0
    assert runner.percentile([], 0.95) == 0.0


# --- конфиг ------------------------------------------------------------------


def test_config_requires_mandatory_fields(tmp_path):
    write(tmp_path, "c.yaml", "name: t\n")
    with pytest.raises(ValueError, match="task"):
        Config.load(tmp_path / "c.yaml")


def test_dataset_path_is_relative_to_config(tmp_path):
    cfg = make_config(tmp_path, "tests.test_runner:echo")
    assert cfg.dataset_path() == tmp_path / "d.jsonl"


def test_bad_task_reference_is_explained():
    with pytest.raises(ValueError, match="модуль:функция"):
        load_task("просто_модуль")


# --- сохранение и сравнение --------------------------------------------------


def test_result_survives_save_and_load(tmp_path):
    result = runner.run(make_config(tmp_path, "tests.test_runner:echo"))
    path = runner.save(result, tmp_path / "runs" / "r.json")
    back = runner.load(path)
    assert back.aggregates == result.aggregates
    assert [c.case_id for c in back.cases] == [c.case_id for c in result.cases]


def fake_run(name, cases):
    return RunResult(
        name=name,
        started_at="2026-01-01T00:00:00+00:00",
        config_path="",
        cases=cases,
        aggregates={"accuracy": sum(c.scores["accuracy"] for c in cases) / len(cases), "cost_per_case_usd": 0.001},
        threshold_failures=[],
    )


def case(cid, score, error=None):
    from evalkit.core import CaseResult

    return CaseResult(
        case_id=cid, tags=[], scores={"accuracy": score}, latency_ms=10,
        cost_usd=0.001, stability=None, outputs=["x"], error=error,
    )


def test_compare_separates_regressions_from_fixes():
    base = fake_run("base", [case("a", 1.0), case("b", 0.0)])
    new = fake_run("new", [case("a", 0.0), case("b", 1.0)])
    cmp = compare(base, new)
    assert [r.case_id for r in cmp.regressions] == ["a"]
    assert [f.case_id for f in cmp.fixes] == ["b"]
    assert cmp.has_regressions


def test_average_can_hide_a_regression():
    """Среднее одинаковое, но один случай сломался — ради этого всё и затевалось."""
    base = fake_run("base", [case("a", 1.0), case("b", 0.0)])
    new = fake_run("new", [case("a", 0.0), case("b", 1.0)])
    cmp = compare(base, new)
    accuracy = next(m for m in cmp.metrics if m.name == "accuracy")
    assert accuracy.delta == 0
    assert cmp.regressions


def test_new_error_counts_as_regression():
    base = fake_run("base", [case("a", 1.0)])
    new = fake_run("new", [case("a", 0.0, error="таймаут")])
    cmp = compare(base, new)
    assert cmp.new_errors and cmp.has_regressions


def test_cost_growth_is_not_an_improvement():
    base = fake_run("base", [case("a", 1.0)])
    new = fake_run("new", [case("a", 1.0)])
    new.aggregates["cost_per_case_usd"] = 0.002
    cost = next(m for m in compare(base, new).metrics if m.name == "cost_per_case_usd")
    assert cost.delta > 0 and not cost.improved


def test_cases_added_and_removed_are_listed():
    base = fake_run("base", [case("a", 1.0), case("old", 1.0)])
    new = fake_run("new", [case("a", 1.0), case("new", 1.0)])
    cmp = compare(base, new)
    assert cmp.only_in_base == ["old"]
    assert cmp.only_in_new == ["new"]
