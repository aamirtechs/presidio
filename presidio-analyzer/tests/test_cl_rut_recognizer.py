"""Tests for Chile Rol Unico Tributario (RUT) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import ClRutRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Chile RUT recognizer instance for testing."""
    return ClRutRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Chile RUT entity type for testing."""
    return ["CL_RUT"]


# Fixtures are synthetic numbers whose verification digits were
# computed with the published mod-11 algorithm, except for
# 12345678-5 and 12.345.678-5, which are the published tutorial
# example for that algorithm. No fixture is a real person's number.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid RUT numbers ---
        # published tutorial example, plain and dotted
        ("12345678-5", 1, (0, 10), 0.5),
        ("12.345.678-5", 1, (0, 12), 0.5),
        # 7-digit body, plain and dotted
        ("2012345-1", 1, (0, 9), 0.5),
        ("2.012.345-1", 1, (0, 11), 0.5),
        # 8-digit body, synthetic computed example
        ("20123456-5", 1, (0, 10), 0.5),
        # K verification digit, upper and lower case
        ("1000005-K", 1, (0, 9), 0.5),
        ("1000005-k", 1, (0, 9), 0.5),
        # zero verification digit
        ("1000013-0", 1, (0, 9), 0.5),
        (
            "Her RUT 12.345.678-5 was verified",
            1,
            (8, 20),
            0.5,
        ),
        # a sentence period after the check digit does not block the match
        ("RUT 12345678-5.", 1, (4, 14), 0.5),
        # --- Invalid: wrong verification digit ---
        ("12345678-6", 0, (0, 0), 0),
        ("12.345.678-4", 0, (0, 0), 0),
        # --- Invalid: all digits identical (passes the checksum) ---
        ("11111111-1", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit run ---
        ("12345678-9", 0, (0, 0), 0),
        # --- Invalid: wrong body lengths ---
        ("123456-7", 0, (0, 0), 0),  # 6 digits
        ("123456789-5", 0, (0, 0), 0),  # 9 digits
        # --- Lookalikes: same shape, but not a RUT number ---
        ("Order 12345678-5X shipped", 0, (0, 0), 0),
        ("ID 12345678-5-6 on file", 0, (0, 0), 0),
        ("12.345678-5", 0, (0, 0), 0),  # partial dot separators
        ("12345678 -5", 0, (0, 0), 0),  # space before the hyphen
        # fmt: on
    ],
)
def test_when_rut_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid RUT numbers and rejects invalid ones."""
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
    """Both RUT numbers in one text are found at their exact positions."""
    results = recognizer.analyze("IDs: 12345678-5 and 12.345.678-5", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 15, 0.5)
    assert_result(results[1], entities[0], 20, 32, 0.5)


sequential_test_set = [
    ["123456789", True],
    ["987654321", True],
    ["123456785", False],
    ["20123451", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the RUT digits."""
    assert ClRutRecognizer._is_sequential(input_text) == expected_output


def test_verification_digit(recognizer):
    """The mod-11 verification digit matches the published worked example."""
    assert recognizer._verification_digit("12345678") == "5"
    assert recognizer._verification_digit("20123456") == "5"
    assert recognizer._verification_digit("2012345") == "1"
    assert recognizer._verification_digit("1000005") == "K"
    assert recognizer._verification_digit("1000013") == "0"


def test_invalidate_result(recognizer):
    """Checksum and sanity checks reject invalid numbers and accept valid ones."""
    assert recognizer.invalidate_result("12345678-5") is False
    assert recognizer.invalidate_result("12.345.678-5") is False
    assert recognizer.invalidate_result("2012345-1") is False
    assert recognizer.invalidate_result("2.012.345-1") is False
    assert recognizer.invalidate_result("1000005-K") is False
    assert recognizer.invalidate_result("1000005-k") is False
    assert recognizer.invalidate_result("1000013-0") is False
    assert recognizer.invalidate_result("12345678-6") is True
    assert recognizer.invalidate_result("12.345.678-4") is True
    assert recognizer.invalidate_result("11111111-1") is True
    assert recognizer.invalidate_result("12345678-9") is True
    assert recognizer.invalidate_result("123456-7") is True
    assert recognizer.invalidate_result("123456789-5") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "chile",
        "chilean",
        "rut",
        "rol unico tributario",
        "rol unico nacional",
        "rol unico",
        "tax identification number",
        "identity card",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["CL_RUT"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = ClRutRecognizer(name="CustomClRut")
    assert recognizer.name == "CustomClRut"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a RUT number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(ClRutRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="12345678-5", language="en", entities=["CL_RUT"])
    with_context = analyzer.analyze(
        text="12345678-5",
        language="en",
        entities=["CL_RUT"],
        context=["rut"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(ClRutRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="RUT 12345678-6",
        language="en",
        entities=["CL_RUT"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "ClRutRecognizer"]
    assert len(entries) == 1, "ClRutRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "cl"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["ClRutRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "CL_RUT" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My RUT is 12.345.678-5",
        language="en",
        entities=["CL_RUT"],
    )
    assert [result.entity_type for result in results] == ["CL_RUT"]
