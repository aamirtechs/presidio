"""Tests for Dominican Republic Cedula de Identidad y Electoral recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import DoCedulaRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Dominican Republic Cedula recognizer instance for testing."""
    return DoCedulaRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Dominican Republic Cedula entity type for testing."""
    return ["DO_CEDULA"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned Luhn algorithm, except for
# 00113918205, 22400022111, and 00114905656, which are
# published examples: the first two are the
# python-stdnum docstring examples, and the third is
# the MicroFocus Eduction grammar example 001-1490565-6.
# No fixture is a real person's number beyond those
# published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid cedula numbers ---
        # published examples, plain
        ("00113918205", 1, (0, 11), 0.5),  # python-stdnum docstring example
        ("22400022111", 1, (0, 11), 0.5),  # python-stdnum format example
        ("00114905656", 1, (0, 11), 0.5),  # MicroFocus example 001-1490565-6
        # synthetic numbers, check digits computed with Luhn
        ("40200459978", 1, (0, 11), 0.5),  # 402 series
        ("03100293814", 1, (0, 11), 0.5),  # Santiago series
        ("00119345676", 1, (0, 11), 0.5),
        ("05600012347", 1, (0, 11), 0.5),
        ("22300045675", 1, (0, 11), 0.5),
        # canonical hyphenated form
        ("001-1391820-5", 1, (0, 13), 0.5),  # python-stdnum example, hyphenated
        ("001-1490565-6", 1, (0, 13), 0.5),  # MicroFocus example, hyphenated
        ("402-0045997-8", 1, (0, 13), 0.5),  # 402 series, hyphenated
        # a number embedded in a sentence
        ("Cedula 00113918205 on file", 1, (7, 18), 0.5),
        # a sentence period after the check digit does not block the match
        ("Cedula 00113918205.", 1, (7, 18), 0.5),
        ("Cedula 001-1391820-5.", 1, (7, 20), 0.5),
        # --- Invalid: wrong check digit ---
        ("00113918204", 0, (0, 0), 0),
        ("00114905657", 0, (0, 0), 0),
        ("22400022112", 0, (0, 0), 0),
        ("001-1391820-4", 0, (0, 0), 0),  # hyphenated form, wrong check digit
        # --- Invalid: all digits identical ---
        # 00000000000 passes the Luhn check, so only the
        # all-same-digit rule rejects it
        ("00000000000", 0, (0, 0), 0),
        ("11111111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        # 34567890123 and 32109876543 both pass the Luhn
        # check, so only the sequential rule rejects them
        ("34567890123", 0, (0, 0), 0),
        ("32109876543", 0, (0, 0), 0),
        ("01234567890", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("0011391820", 0, (0, 0), 0),  # 10 digits, no check digit
        ("001139182055", 0, (0, 0), 0),  # 12 digits
        # --- Lookalikes: same shape, but not a cedula number ---
        ("100113918205", 0, (0, 0), 0),  # a leading digit
        ("001-1391820-55", 0, (0, 0), 0),  # a longer hyphenated run
        ("00-11391820-5", 0, (0, 0), 0),  # wrong hyphen grouping
        ("A00113918205", 0, (0, 0), 0),  # a leading letter
        ("00113918205A", 0, (0, 0), 0),  # a trailing letter
        ("001 1391820 5", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_cedula_in_text_then_all_ids_found(
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
    """Both cedula numbers in one text are found at their exact positions."""
    results = recognizer.analyze("Cedulas: 00113918205 and 22400022111", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 9, 20, 0.5)
    assert_result(results[1], entities[0], 25, 36, 0.5)


sequential_test_set = [
    ["01234567890", True],
    ["98765432109", True],
    ["34567890123", True],
    ["32109876543", True],
    ["00113918205", False],
    ["22400022111", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the cedula digits."""
    assert DoCedulaRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("0011391820") == "5"
    assert recognizer._check_digit("2240002211") == "1"
    assert recognizer._check_digit("0011490565") == "6"
    assert recognizer._check_digit("4020045997") == "8"
    assert recognizer._check_digit("0310029381") == "4"
    assert recognizer._check_digit("0011934567") == "6"


def test_has_valid_checksum(recognizer):
    """The Luhn checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("00113918205") is True
    assert recognizer._has_valid_checksum("22400022111") is True
    assert recognizer._has_valid_checksum("00114905656") is True
    assert recognizer._has_valid_checksum("40200459978") is True
    assert recognizer._has_valid_checksum("00113918204") is False
    assert recognizer._has_valid_checksum("00114905657") is False
    assert recognizer._has_valid_checksum("22400022112") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("00113918205") is False
    assert recognizer.invalidate_result("22400022111") is False
    assert recognizer.invalidate_result("00114905656") is False
    assert recognizer.invalidate_result("40200459978") is False
    assert recognizer.invalidate_result("001-1391820-5") is False
    assert recognizer.invalidate_result("001-1490565-6") is False
    assert recognizer.invalidate_result("00113918204") is True
    assert recognizer.invalidate_result("001-1391820-4") is True
    assert recognizer.invalidate_result("00000000000") is True
    assert recognizer.invalidate_result("34567890123") is True
    assert recognizer.invalidate_result("0011391820") is True
    assert recognizer.invalidate_result("001139182055") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = DoCedulaRecognizer(name="CustomDoCedula")
    assert recognizer.name == "CustomDoCedula"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a cedula number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(DoCedulaRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="00113918205", language="en", entities=["DO_CEDULA"])
    with_context = analyzer.analyze(
        text="00113918205",
        language="en",
        entities=["DO_CEDULA"],
        context=["cedula"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(DoCedulaRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="Cedula 00113918204",
        language="en",
        entities=["DO_CEDULA"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "DoCedulaRecognizer"]
    assert len(entries) == 1, "DoCedulaRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "do"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["DoCedulaRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "DO_CEDULA" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My cedula is 00113918205",
        language="en",
        entities=["DO_CEDULA"],
    )
    assert [result.entity_type for result in results] == ["DO_CEDULA"]
