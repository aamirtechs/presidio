"""Tests for Uzbekistan Personal Identification Number (PINFL) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import UzPinflRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Uzbekistan PINFL recognizer instance for testing."""
    return UzPinflRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Uzbekistan PINFL entity type for testing."""
    return ["UZ_PINFL"]


# Fixtures are synthetic numbers built to the published structure
# (gender/century digit + DDMMYY birth date + 3-digit district code +
# 3-digit serial + check digit), because the recognizer implements no
# checksum: no authoritative Uzbek source with a worked valid example
# was found for the published check digit (see the recognizer
# docstring). The trailing digit in each fixture is arbitrary.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid PINFLs ---
        # male, born 1990-07-15 (digit 3 = male, 1900s)
        ("31507901234567", 1, (0, 14), 0.5),
        # female, born 1995-12-31 (digit 4 = female, 1900s)
        ("43112950010023", 1, (0, 14), 0.5),
        (
            "His PINFL 31507901234567 was verified",
            1,
            (10, 24),
            0.5,
        ),
        # day 29 Feb is valid in a leap year (2000, digit 5 = male, 2000s)
        ("52902000100209", 1, (0, 14), 0.5),
        # 1800s century digit (digit 2 = female, 1800s)
        ("21003959999991", 1, (0, 14), 0.5),
        # --- Invalid: month or day out of range (sanity check) ---
        ("31313901234567", 0, (0, 0), 0),  # month 13
        ("33102901234567", 0, (0, 0), 0),  # 31 Feb
        ("30007901234567", 0, (0, 0), 0),  # day 00
        # day 29 Feb in a non-leap year (2001, digit 6 = female, 2000s)
        ("62902010100209", 0, (0, 0), 0),
        # --- Invalid: gender/century digit out of range ---
        ("71507901234567", 0, (0, 0), 0),  # digit 7
        ("01507901234567", 0, (0, 0), 0),  # digit 0
        # --- Invalid: district code 000 ---
        ("31507900004567", 0, (0, 0), 0),
        # --- Invalid: serial number 000 ---
        ("31507901230007", 0, (0, 0), 0),
        # --- Invalid: birth date in the future ---
        ("50107991234567", 0, (0, 0), 0),  # 2099-07-01
        # --- Invalid: wrong digit counts ---
        ("3150790123456", 0, (0, 0), 0),  # 13 digits
        ("315079012345678", 0, (0, 0), 0),  # 15 digits
        # --- Invalid: all digits identical (sanity check) ---
        ("11111111111111", 0, (0, 0), 0),
        # --- Lookalikes: same shape, but not a PINFL ---
        ("Order number 3150790123456701 was shipped", 0, (0, 0), 0),
        ("abcdefghij", 0, (0, 0), 0),
        # fmt: on
    ],
)
def test_when_pinfl_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid PINFLs and rejects invalid ones."""
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
    """Both PINFLs in one text are found at their exact positions."""
    results = recognizer.analyze(
        "IDs: 31507901234567 and 43112950010023", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 19, 0.5)
    assert_result(results[1], entities[0], 24, 38, 0.5)


sequential_test_set = [
    ["12345678901234", True],
    ["43210987654321", True],
    ["31507901234567", False],
    ["43112950010023", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the PINFL digits."""
    assert UzPinflRecognizer._is_sequential(input_text) == expected_output


birth_date_test_set = [
    # (pinfl, expected)
    ["31507901234567", True],  # 1990-07-15, male 1900s
    ["43112950010023", True],  # 1995-12-31, female 1900s
    ["52902000100209", True],  # 2000-02-29, leap year
    ["62902010100209", False],  # 2001-02-29, not a leap year
    ["31313901234567", False],  # month 13
    ["33102901234567", False],  # 31 Feb
    ["50107991234567", False],  # 2099, in the future
    ["71507901234567", False],  # century digit 7
]


@pytest.mark.parametrize("input_text, expected_output", birth_date_test_set)
def test_valid_birth_date(input_text, expected_output):
    """Test the calendar check on the birth date part."""
    assert UzPinflRecognizer._valid_birth_date(input_text) == expected_output


def test_invalidate_result(recognizer):
    """Sanity checks reject trivial numbers and accept real PINFL shapes."""
    assert recognizer.invalidate_result("31507901234567") is False
    assert recognizer.invalidate_result("43112950010023") is False
    assert recognizer.invalidate_result("52902000100209") is False
    assert recognizer.invalidate_result("21003959999991") is False
    assert recognizer.invalidate_result("31313901234567") is True
    assert recognizer.invalidate_result("33102901234567") is True
    assert recognizer.invalidate_result("62902010100209") is True
    assert recognizer.invalidate_result("71507901234567") is True
    assert recognizer.invalidate_result("31507900004567") is True
    assert recognizer.invalidate_result("31507901230007") is True
    assert recognizer.invalidate_result("50107991234567") is True
    assert recognizer.invalidate_result("11111111111111") is True
    assert recognizer.invalidate_result("3150790123456") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "uzbekistan",
        "uzbek",
        "pinfl",
        "pini",
        "personal identification number",
        "individual identification number",
        "national identification",
        "uzbekistani",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["UZ_PINFL"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = UzPinflRecognizer(name="CustomUzPinfl")
    assert recognizer.name == "CustomUzPinfl"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a Uzbekistan PINFL raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(UzPinflRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="31507901234567", language="en", entities=["UZ_PINFL"]
    )
    with_context = analyzer.analyze(
        text="31507901234567",
        language="en",
        entities=["UZ_PINFL"],
        context=["pinfl"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the sanity checks reject."""
    registry = RecognizerRegistry()
    registry.add_recognizer(UzPinflRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="PINFL 31313901234567",
        language="en",
        entities=["UZ_PINFL"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "UzPinflRecognizer"]
    assert len(entries) == 1, "UzPinflRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "uz"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "UzPinflRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "UZ_PINFL" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My PINFL is 31507901234567",
        language="en",
        entities=["UZ_PINFL"],
    )
    assert [result.entity_type for result in results] == ["UZ_PINFL"]
