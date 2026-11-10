"""Tests for Paraguay Registro Unico de Contribuyentes (RUC) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import PyRucRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Paraguay RUC recognizer instance for testing."""
    return PyRucRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Paraguay RUC entity type for testing."""
    return ["PY_RUC"]


# Fixtures are numbers whose check digits were
# computed with the pinned mod-11 rule and
# cross-checked against python-stdnum, except for
# 80000519-8, 80028061-0, 9991603 and 80000035-8,
# which are published examples: the first is
# Telefonica Celular del Paraguay S.A.E. (DNIT
# General Resolution 105/2021 and public
# procurement records) and the other three are the
# python-stdnum py.ruc docstring examples. No
# fixture is a real person's number beyond those
# published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid RUC numbers ---
        # published examples, plain
        ("800005198", 1, (0, 9), 0.5),  # Telefonica, 80000519-8
        ("800280610", 1, (0, 9), 0.5),  # python-stdnum docstring, remainder-1 edge
        ("800000358", 1, (0, 9), 0.5),  # python-stdnum format example
        ("9991603", 1, (0, 7), 0.5),  # python-stdnum docstring, 6 digit base
        # published example, hyphenated
        ("80000519-8", 1, (0, 10), 0.5),  # Telefonica
        ("80028061-0", 1, (0, 10), 0.5),  # remainder-1 edge, hyphenated
        ("80000035-8", 1, (0, 10), 0.5),
        ("999160-3", 1, (0, 8), 0.5),  # 6 digit base, hyphenated
        # synthetic numbers, check digits computed with the pinned rule
        ("800000005", 1, (0, 9), 0.5),  # 80000000-5
        ("80000000-5", 1, (0, 10), 0.5),
        ("123456789", 1, (0, 9), 0.5),  # sequential base, deliberately accepted
        ("12345678-9", 1, (0, 10), 0.5),
        ("45678901", 1, (0, 8), 0.5),  # 7 digit base
        ("4567890-1", 1, (0, 9), 0.5),
        ("1000004", 1, (0, 7), 0.5),  # 6 digit base
        ("100000-4", 1, (0, 8), 0.5),
        ("500000000", 1, (0, 9), 0.5),  # remainder-1 edge
        ("50000000-0", 1, (0, 10), 0.5),
        ("800005635", 1, (0, 9), 0.5),
        ("801234565", 1, (0, 9), 0.5),
        ("500000018", 1, (0, 9), 0.5),  # refugee series base
        ("12345679", 1, (0, 8), 0.5),  # 7 digit base
        ("76543218", 1, (0, 8), 0.5),
        ("800000013", 1, (0, 9), 0.5),
        ("456789014", 1, (0, 9), 0.5),
        ("600000001", 1, (0, 9), 0.5),
        ("1234560", 1, (0, 7), 0.5),  # 6 digit base
        ("876543212", 1, (0, 9), 0.5),  # descending sequential base
        # a number embedded in a sentence
        ("RUC 800005198 on file", 1, (4, 13), 0.5),
        # a sentence period after the check digit does not block the match
        ("RUC 800005198.", 1, (4, 13), 0.5),
        ("RUC 80000519-8.", 1, (4, 14), 0.5),
        # --- Invalid: wrong check digit ---
        ("800005197", 0, (0, 0), 0),
        ("800280611", 0, (0, 0), 0),
        ("123456780", 0, (0, 0), 0),
        ("80000519-7", 0, (0, 0), 0),  # hyphenated form, wrong check digit
        # --- Invalid: all digits identical ---
        # 00000000-0 and 11111111-0 both pass the
        # checksum, so only the all-same-digit rules
        # reject them: the full number rule catches
        # 000000000, and the identifier rule catches
        # 11111111-0 and 999999990, whose 8 digit
        # identifiers are all the same digit
        ("000000000", 0, (0, 0), 0),
        ("111111111", 0, (0, 0), 0),
        ("999999990", 0, (0, 0), 0),
        ("00000000-0", 0, (0, 0), 0),
        ("11111111-0", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("123456", 0, (0, 0), 0),  # 6 digits, no check digit
        ("8000051988", 0, (0, 0), 0),  # 10 digits
        ("12345", 0, (0, 0), 0),  # 5 digits
        # --- Lookalikes: same shape, but not a RUC number ---
        ("180000519-8", 0, (0, 0), 0),  # a leading digit
        ("80000519-88", 0, (0, 0), 0),  # a longer hyphenated run
        ("80-000519-8", 0, (0, 0), 0),  # wrong hyphen grouping
        ("A80000519-8", 0, (0, 0), 0),  # a leading letter
        ("80000519-8A", 0, (0, 0), 0),  # a trailing letter
        ("80000 519 8", 0, (0, 0), 0),  # spaced form is not accepted
        ("80000519 8", 0, (0, 0), 0),  # space instead of a hyphen
        # fmt: on
    ],
)
def test_when_ruc_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid numbers and rejects invalid ones."""
    results = recognizer.analyze(text, entities)

    assert len(results) == expected_len
    if expected_len > 0:
        assert_result(
            results[0],
            entities[0],
            expected_position[0],
            expected_position[1],
            expected_score,
        )


def test_multiple_ids_positions(recognizer, entities):
    """Both RUC numbers in one text are found at their exact positions."""
    results = recognizer.analyze("RUCs: 800005198 and 800280610", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 15, 0.5)
    assert_result(results[1], entities[0], 20, 29, 0.5)


sequential_test_set = [
    ["12345678", True],
    ["87654321", True],
    ["123456789", True],
    ["987654321", True],
    ["80000519", False],
    ["80028061", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check kept for symmetry with other recognizers."""
    assert PyRucRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("80000519") == "8"
    assert recognizer._check_digit("80028061") == "0"
    assert recognizer._check_digit("999160") == "3"
    assert recognizer._check_digit("11111111") == "0"
    assert recognizer._check_digit("50000000") == "0"
    assert recognizer._check_digit("12345678") == "9"
    assert recognizer._check_digit("100000") == "4"


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("800005198") is True
    assert recognizer._has_valid_checksum("800280610") is True
    assert recognizer._has_valid_checksum("9991603") is True
    assert recognizer._has_valid_checksum("800000358") is True
    assert recognizer._has_valid_checksum("111111110") is True
    assert recognizer._has_valid_checksum("800005197") is False
    assert recognizer._has_valid_checksum("800280611") is False
    assert recognizer._has_valid_checksum("123456780") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("800005198") is False
    assert recognizer.invalidate_result("800280610") is False
    assert recognizer.invalidate_result("9991603") is False
    assert recognizer.invalidate_result("800000358") is False
    assert recognizer.invalidate_result("80000519-8") is False
    assert recognizer.invalidate_result("80028061-0") is False
    assert recognizer.invalidate_result("800005197") is True
    assert recognizer.invalidate_result("80000519-7") is True
    assert recognizer.invalidate_result("000000000") is True
    assert recognizer.invalidate_result("11111111-0") is True
    assert recognizer.invalidate_result("99999999-0") is True
    assert recognizer.invalidate_result("123456") is True
    assert recognizer.invalidate_result("8000051988") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = PyRucRecognizer(name="CustomPyRuc")
    assert recognizer.name == "CustomPyRuc"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a RUC number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(PyRucRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="800005198", language="en", entities=["PY_RUC"])
    with_context = analyzer.analyze(
        text="800005198",
        language="en",
        entities=["PY_RUC"],
        context=["ruc"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(PyRucRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="RUC 800005197",
        language="en",
        entities=["PY_RUC"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "PyRucRecognizer"]
    assert len(entries) == 1, "PyRucRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "py"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["PyRucRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "PY_RUC" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My RUC is 800005198",
        language="en",
        entities=["PY_RUC"],
    )
    assert [result.entity_type for result in results] == ["PY_RUC"]
