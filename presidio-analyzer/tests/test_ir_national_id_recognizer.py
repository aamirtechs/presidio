"""Tests for Iran Code Melli national code recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import IrNationalIdRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Iran Code Melli recognizer instance for testing."""
    return IrNationalIdRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Iran Code Melli entity type for testing."""
    return ["IR_NATIONAL_ID"]


# Fixtures are synthetic numbers whose check digits were
# computed with the published mod-11 algorithm, except for
# 472-171992-2 and 4721719922, which are the published
# worked example (MicroFocus Iran entity documentation).
# No fixture is a real person's number.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid Code Melli numbers ---
        # published worked example, grouped and plain
        ("472-171992-2", 1, (0, 12), 0.5),
        ("4721719922", 1, (0, 10), 0.5),
        # synthetic examples, check digits computed
        ("4561237895", 1, (0, 10), 0.5),
        ("0491234562", 1, (0, 10), 0.5),
        ("3216549879", 1, (0, 10), 0.5),
        # remainder below 2: check digit equals the remainder (1)
        ("1234567891", 1, (0, 10), 0.5),
        # remainder 0: check digit 0
        ("0490020550", 1, (0, 10), 0.5),
        (
            "Her code melli 472-171992-2 was verified",
            1,
            (15, 27),
            0.5,
        ),
        # a sentence period after the check digit does not block the match
        ("Code melli 4721719922.", 1, (11, 21), 0.5),
        # --- Invalid: wrong check digit ---
        ("4721719923", 0, (0, 0), 0),
        ("4561237896", 0, (0, 0), 0),
        # --- Invalid: first three digits 000 ---
        # checksum valid, so only the prefix rule rejects it
        ("0001234560", 0, (0, 0), 0),
        # --- Invalid: digits 4 to 9 all zeros ---
        # checksum valid, so only the serial rule rejects it
        ("1230000003", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        # checksum valid, so only the all-same rule rejects it
        ("1111111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential runs ---
        ("1234567890", 0, (0, 0), 0),
        ("9876543210", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("472171992", 0, (0, 0), 0),  # 9 digits
        ("47217199222", 0, (0, 0), 0),  # 11 digits
        # --- Lookalikes: same shape, but not a Code Melli number ---
        # an 11 digit run is not a 10 digit Code Melli
        ("04721719922", 0, (0, 0), 0),
        ("Order 472-171992-2X shipped", 0, (0, 0), 0),
        ("ID 472-171992-2-3 on file", 0, (0, 0), 0),
        ("472 171992 2", 0, (0, 0), 0),  # space form is not accepted
        ("472-171992-23", 0, (0, 0), 0),  # extra digit after the hyphen
        # fmt: on
    ],
)
def test_when_code_melli_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid codes and rejects invalid ones."""
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
    """Both Code Melli numbers in one text are found at their exact positions."""
    results = recognizer.analyze("IDs: 472-171992-2 and 4561237895", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 17, 0.5)
    assert_result(results[1], entities[0], 22, 32, 0.5)


sequential_test_set = [
    ["1234567890", True],
    ["9876543210", True],
    ["4721719922", False],
    ["4561237895", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the Code Melli digits."""
    assert IrNationalIdRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked example."""
    assert recognizer._check_digit("472171992") == "2"
    assert recognizer._check_digit("123456789") == "1"
    assert recognizer._check_digit("049002055") == "0"


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("4721719922") is True
    assert recognizer._has_valid_checksum("4561237895") is True
    assert recognizer._has_valid_checksum("4721719923") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("472-171992-2") is False
    assert recognizer.invalidate_result("0001234560") is True
    assert recognizer.invalidate_result("1230000003") is True
    assert recognizer.invalidate_result("1111111111") is True
    assert recognizer.invalidate_result("1234567890") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = IrNationalIdRecognizer(name="CustomIrNationalId")
    assert recognizer.name == "CustomIrNationalId"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a Code Melli number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(IrNationalIdRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="4721719922", language="en", entities=["IR_NATIONAL_ID"]
    )
    with_context = analyzer.analyze(
        text="4721719922",
        language="en",
        entities=["IR_NATIONAL_ID"],
        context=["code melli"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(IrNationalIdRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="Code melli 4721719923",
        language="en",
        entities=["IR_NATIONAL_ID"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "IrNationalIdRecognizer"]
    assert len(entries) == 1, "IrNationalIdRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ir"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "IrNationalIdRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "IR_NATIONAL_ID" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My code melli is 472-171992-2",
        language="en",
        entities=["IR_NATIONAL_ID"],
    )
    assert [result.entity_type for result in results] == ["IR_NATIONAL_ID"]
