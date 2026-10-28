"""Tests for Ukraine individual taxpayer number (RNOKPP) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import UaRnokppRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Ukraine RNOKPP recognizer instance for testing."""
    return UaRnokppRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Ukraine RNOKPP entity type for testing."""
    return ["UA_RNOKPP"]


# Fixtures are synthetic numbers built to the published structure
# (5-digit day count from 31 December 1899 + 3-digit serial +
# sex digit + check digit), because the recognizer implements no
# checksum: no authoritative Ukrainian source with the weights and
# a worked valid example was found for the published check digit
# algorithm (see the recognizer docstring). The trailing digit in
# each fixture is arbitrary.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid RNOKPPs ---
        # male, born 1990-07-01 (day count 33054, sex digit 1 = odd)
        ("3305412315", 1, (0, 10), 0.5),
        # female, born 1985-03-15 (day count 31120, sex digit 2 = even)
        ("3112045627", 1, (0, 10), 0.5),
        (
            "His RNOKPP 3305412315 was verified",
            1,
            (11, 21),
            0.5,
        ),
        # day 29 Feb is valid in a leap year (2000-02-29, day count 36584)
        ("3658400139", 1, (0, 10), 0.5),
        # female, born 1978-12-31 (day count 28854, sex digit 4 = even)
        ("2885478942", 1, (0, 10), 0.5),
        # --- Invalid: day count out of range (sanity check) ---
        ("0000012315", 0, (0, 0), 0),  # day count 0, before 1900-01-01
        ("9999912315", 0, (0, 0), 0),  # day count 99999, year 2173
        ("7286612315", 0, (0, 0), 0),  # 2099-07-01, in the future
        # --- Invalid: wrong digit counts ---
        ("330541231", 0, (0, 0), 0),  # 9 digits
        ("33054123155", 0, (0, 0), 0),  # 11 digits
        # --- Invalid: all digits identical (sanity check) ---
        ("1111111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential runs (sanity check) ---
        ("0123456789", 0, (0, 0), 0),
        ("9876543210", 0, (0, 0), 0),
        # --- Lookalikes: same shape, but not an RNOKPP ---
        ("Order number 330541231501 was shipped", 0, (0, 0), 0),
        ("abcdefghij", 0, (0, 0), 0),
        # fmt: on
    ],
)
def test_when_rnokpp_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid RNOKPPs and rejects invalid ones."""
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
    """Both RNOKPPs in one text are found at their exact positions."""
    results = recognizer.analyze(
        "IDs: 3305412315 and 3112045627", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 15, 0.5)
    assert_result(results[1], entities[0], 20, 30, 0.5)


sequential_test_set = [
    ["0123456789", True],
    ["9876543210", True],
    ["3305412315", False],
    ["3112045627", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the RNOKPP digits."""
    assert UaRnokppRecognizer._is_sequential(input_text) == expected_output


birth_date_test_set = [
    # (rnokpp, expected)
    ["3305412315", True],  # 1990-07-01, male
    ["3112045627", True],  # 1985-03-15, female
    ["3658400139", True],  # 2000-02-29, leap year
    ["0000012315", False],  # day count 0
    ["9999912315", False],  # year 2173, in the future
    ["7286612315", False],  # 2099-07-01, in the future
]


@pytest.mark.parametrize("input_text, expected_output", birth_date_test_set)
def test_valid_birth_date(input_text, expected_output):
    """Test the day-count range check on the birth date part."""
    assert UaRnokppRecognizer._valid_birth_date(input_text) == expected_output


def test_invalidate_result(recognizer):
    """Sanity checks reject trivial numbers and accept real RNOKPP shapes."""
    assert recognizer.invalidate_result("3305412315") is False
    assert recognizer.invalidate_result("3112045627") is False
    assert recognizer.invalidate_result("3658400139") is False
    assert recognizer.invalidate_result("0000012315") is True
    assert recognizer.invalidate_result("9999912315") is True
    assert recognizer.invalidate_result("7286612315") is True
    assert recognizer.invalidate_result("1111111111") is True
    assert recognizer.invalidate_result("0123456789") is True
    assert recognizer.invalidate_result("330541231") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "ukraine",
        "ukrainian",
        "rnokpp",
        "tax identification",
        "taxpayer",
        "individual tax number",
        "identification number",
        "inn",
        "ipn",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["UA_RNOKPP"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = UaRnokppRecognizer(name="CustomUaRnokpp")
    assert recognizer.name == "CustomUaRnokpp"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a Ukraine RNOKPP raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(UaRnokppRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="3305412315", language="en", entities=["UA_RNOKPP"]
    )
    with_context = analyzer.analyze(
        text="3305412315",
        language="en",
        entities=["UA_RNOKPP"],
        context=["rnokpp"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the sanity checks reject."""
    registry = RecognizerRegistry()
    registry.add_recognizer(UaRnokppRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="RNOKPP 7286612315",
        language="en",
        entities=["UA_RNOKPP"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "UaRnokppRecognizer"]
    assert len(entries) == 1, "UaRnokppRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ua"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "UaRnokppRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "UA_RNOKPP" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My RNOKPP is 3305412315",
        language="en",
        entities=["UA_RNOKPP"],
    )
    assert [result.entity_type for result in results] == ["UA_RNOKPP"]
