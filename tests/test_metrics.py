import pytest

from evalkit import metrics


def build(spec):
    return metrics.build(spec)[1]


def test_json_valid_accepts_fenced_block():
    m = build({"type": "json_valid"})
    assert m('```json\n{"a": 1}\n```', None) == 1.0
    assert m("просто текст", None) == 0.0


def test_exact_match_ignores_case_and_spacing():
    m = build({"type": "exact_match"})
    assert m("  Кондиционер   настенный ", "кондиционер настенный") == 1.0
    assert m("другое", "кондиционер") == 0.0


def test_exact_match_strict_compares_as_is():
    m = build({"type": "exact_match", "strict": True})
    assert m("Текст", "Текст") == 1.0
    assert m("текст", "Текст") == 0.0


def test_yo_and_ye_are_the_same_letter():
    m = build({"type": "exact_match"})
    assert m("шлёпанцы", "шлепанцы") == 1.0


def test_contains_uses_expected_when_value_absent():
    m = build({"type": "contains"})
    assert m("в ответе есть счёт на оплату", "Счёт") == 1.0


def test_contains_requires_something_to_look_for():
    m = build({"type": "contains"})
    with pytest.raises(ValueError):
        m("текст", None)


def test_regex_with_flags():
    m = build({"type": "regex", "pattern": "^итого", "flags": "im"})
    assert m("Позиции\nИтого 5", None) == 1.0


def test_fields_match_partial_score():
    m = build({"type": "fields_match", "fields": ["name", "qty", "unit"]})
    got = {"name": "труба", "qty": 10, "unit": "шт"}
    want = {"name": "труба", "qty": 10, "unit": "м"}
    assert m(got, want) == pytest.approx(2 / 3)


def test_fields_match_all_or_nothing():
    m = build({"type": "fields_match", "fields": ["name", "qty"], "partial": False})
    assert m({"name": "a", "qty": 1}, {"name": "a", "qty": 2}) == 0.0


def test_fields_match_penalizes_extra_items():
    m = build({"type": "fields_match", "fields": ["name"], "field": "items"})
    got = {"items": [{"name": "труба"}, {"name": "лишнее"}]}
    want = {"items": [{"name": "труба"}]}
    assert m(got, want) == pytest.approx(0.5)


def test_fields_match_reads_nested_field():
    m = build({"type": "fields_match", "fields": ["name", "qty"], "field": "items"})
    got = '{"items": [{"name": "труба", "qty": 3}]}'
    want = {"items": [{"name": "Труба", "qty": 3}]}
    assert m(got, want) == 1.0


def test_numeric_close_within_tolerance():
    m = build({"type": "numeric_close", "field": "total", "tol": 0.5})
    assert m({"total": 10.4}, {"total": 10.0}) == 1.0
    assert m({"total": 11.0}, {"total": 10.0}) == 0.0


def test_numeric_close_relative():
    m = build({"type": "numeric_close", "field": "total", "tol": 0.1, "relative": True})
    assert m({"total": 105}, {"total": 100}) == 1.0
    assert m({"total": 120}, {"total": 100}) == 0.0


def test_numeric_close_on_garbage_returns_zero():
    m = build({"type": "numeric_close", "field": "total"})
    assert m({"total": "много"}, {"total": 10}) == 0.0


def test_set_overlap_is_jaccard():
    m = build({"type": "set_overlap", "field": "items"})
    got = {"items": ["а", "б", "в"]}
    want = {"items": ["б", "в", "г"]}
    assert m(got, want) == pytest.approx(0.5)


def test_set_overlap_two_empty_sets_match():
    m = build({"type": "set_overlap", "field": "items"})
    assert m({"items": []}, {"items": []}) == 1.0


def test_unknown_metric_names_alternatives():
    with pytest.raises(ValueError, match="json_valid"):
        metrics.build({"type": "нет такой"})


def test_metric_requires_type():
    with pytest.raises(ValueError):
        metrics.build({"fields": ["a"]})


def test_custom_name_is_kept():
    name, _ = metrics.build({"type": "json_valid", "name": "схема"})
    assert name == "схема"
