"""Tests for Colombia NIT tax identifier recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import CoNitRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Colombia NIT recognizer instance for testing."""
    return CoNitRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Colombia NIT entity type for testing."""
    return ["CO_NIT"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned DIAN mod-11 algorithm, except
# for 8909039388, 8001972684, 8999990681, 8300952130,
# 8909000501, 8909006089, 8909039395, and 8603242181,
# which are published examples. No fixture is a real
# person's number: the published examples are company
# NITs (Bancolombia, DIAN itself, Ecopetrol, Organizacion
# Terpel, Grupo Nutresa, Almacenes Exito, Postobon, and
# the DIAN Orden Administrativa worked example).
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid NIT numbers ---
        # published examples, plain
        ("8909039388", 1, (0, 10), 0.5),  # Bancolombia
        ("8001972684", 1, (0, 10), 0.5),  # DIAN itself
        ("8999990681", 1, (0, 10), 0.5),  # Ecopetrol, remainder 1: check digit 1
        ("8300952130", 1, (0, 10), 0.5),  # Terpel, remainder 0: check digit 0
        ("8909000501", 1, (0, 10), 0.5),  # Nutresa, remainder 1: check digit 1
        ("8909006089", 1, (0, 10), 0.5),  # Almacenes Exito
        ("8909039395", 1, (0, 10), 0.5),  # Postobon
        ("8603242181", 1, (0, 10), 0.5),  # DIAN Orden Administrativa example
        # synthetic numbers, check digits computed
        ("9012345677", 1, (0, 10), 0.5),
        ("8001234565", 1, (0, 10), 0.5),
        ("9008765431", 1, (0, 10), 0.5),
        ("8123456781", 1, (0, 10), 0.5),  # remainder 1: check digit 1
        ("1234567800", 1, (0, 10), 0.5),  # remainder 0: check digit 0
        ("7001234562", 1, (0, 10), 0.5),
        # canonical dotted form and hyphenated form
        ("890.903.938-8", 1, (0, 13), 0.5),  # Bancolombia, dotted
        ("830.095.213-0", 1, (0, 13), 0.5),  # Terpel, dotted
        ("890903938-8", 1, (0, 11), 0.5),  # Bancolombia, hyphenated
        # a number embedded in a sentence
        ("NIT 8909039388 on file", 1, (4, 14), 0.5),
        # a sentence period after the check digit does not block the match
        ("NIT 8909039388.", 1, (4, 14), 0.5),
        ("NIT 890.903.938-8.", 1, (4, 17), 0.5),
        # --- Invalid: wrong check digit ---
        ("8909039389", 0, (0, 0), 0),
        ("8300952131", 0, (0, 0), 0),  # remainder 0 takes 0, not 1
        ("8999990680", 0, (0, 0), 0),  # remainder 1 takes 1, not 0
        ("890.903.938-9", 0, (0, 0), 0),  # dotted form, wrong check digit
        # --- Invalid: all digits identical ---
        ("1111111111", 0, (0, 0), 0),
        ("0000000000", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        # 2345678901 has a valid checksum, so only the
        # sequential rule rejects it
        ("2345678901", 0, (0, 0), 0),
        ("0123456789", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("890903938", 0, (0, 0), 0),  # 9 digits, no check digit
        ("89090393881", 0, (0, 0), 0),  # 11 digits
        # --- Lookalikes: same shape, but not a NIT number ---
        ("18909039388", 0, (0, 0), 0),  # a leading digit
        ("890903938-81", 0, (0, 0), 0),  # a longer hyphenated run
        ("890.903.938-8-1", 0, (0, 0), 0),  # a longer dotted run
        ("89.090.3938-8", 0, (0, 0), 0),  # wrong dot grouping
        ("890.903.9388", 0, (0, 0), 0),  # dotted form without the hyphen
        ("A8909039388", 0, (0, 0), 0),  # a leading letter
        ("8909039388A", 0, (0, 0), 0),  # a trailing letter
        ("890 9039388", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_nit_in_text_then_all_ids_found(
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
    """Both NIT numbers in one text are found at their exact positions."""
    results = recognizer.analyze("NITs: 8909039388 and 8001972684", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 16, 0.5)
    assert_result(results[1], entities[0], 21, 31, 0.5)


sequential_test_set = [
    ["0123456789", True],
    ["9876543210", True],
    ["2345678901", True],
    ["8765432109", True],
    ["8909039388", False],
    ["8300952130", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the NIT digits."""
    assert CoNitRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("890903938") == "8"
    assert recognizer._check_digit("800197268") == "4"
    assert recognizer._check_digit("901234567") == "7"
    assert recognizer._check_digit("900876543") == "1"
    # remainder 1 gives a check digit of 1
    assert recognizer._check_digit("899999068") == "1"
    assert recognizer._check_digit("860324218") == "1"
    assert recognizer._check_digit("812345678") == "1"
    # remainder 0 gives a check digit of 0
    assert recognizer._check_digit("830095213") == "0"
    assert recognizer._check_digit("123456780") == "0"


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("8909039388") is True
    assert recognizer._has_valid_checksum("8001972684") is True
    assert recognizer._has_valid_checksum("8999990681") is True
    assert recognizer._has_valid_checksum("8300952130") is True
    assert recognizer._has_valid_checksum("8603242181") is True
    assert recognizer._has_valid_checksum("8909039389") is False
    assert recognizer._has_valid_checksum("8300952131") is False
    # the remainder-0 edge case takes 0, not 1
    assert recognizer._has_valid_checksum("8300952131") is False
    # the remainder-1 edge case takes 1, not 0
    assert recognizer._has_valid_checksum("8999990680") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("8909039388") is False
    assert recognizer.invalidate_result("8999990681") is False
    assert recognizer.invalidate_result("8300952130") is False
    assert recognizer.invalidate_result("9012345677") is False
    assert recognizer.invalidate_result("890.903.938-8") is False
    assert recognizer.invalidate_result("890903938-8") is False
    assert recognizer.invalidate_result("8909039389") is True
    assert recognizer.invalidate_result("890.903.938-9") is True
    assert recognizer.invalidate_result("1111111111") is True
    assert recognizer.invalidate_result("2345678901") is True
    assert recognizer.invalidate_result("890903938") is True
    assert recognizer.invalidate_result("89090393881") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = CoNitRecognizer(name="CustomCoNit")
    assert recognizer.name == "CustomCoNit"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a NIT number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(CoNitRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="8909039388", language="en", entities=["CO_NIT"])
    with_context = analyzer.analyze(
        text="8909039388",
        language="en",
        entities=["CO_NIT"],
        context=["nit"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(CoNitRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="NIT 8909039389",
        language="en",
        entities=["CO_NIT"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "CoNitRecognizer"]
    assert len(entries) == 1, "CoNitRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "co"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["CoNitRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "CO_NIT" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My NIT is 8909039388",
        language="en",
        entities=["CO_NIT"],
    )
    assert [result.entity_type for result in results] == ["CO_NIT"]
