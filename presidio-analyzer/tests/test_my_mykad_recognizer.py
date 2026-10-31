"""Tests for Malaysia National Registration Identity Card (MyKad) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import MyMykadRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Malaysia MyKad recognizer instance for testing."""
    return MyMykadRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Malaysia MyKad entity type for testing."""
    return ["MY_MYKAD"]


# Fixtures are synthetic numbers built to the published structure
# (YYMMDD birth date + 2-digit place of birth code + 3-digit serial
# + gender digit, with or without hyphens), because the MyKad
# number has no checksum: Microsoft Learn's Malaysia identity card
# entity definition states the checksum explicitly as "No" (see
# the recognizer docstring).
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid MyKad numbers ---
        # male born 16-09-1963, Johor (01), serial 123, gender digit 5
        ("630916-01-1235", 1, (0, 14), 0.5),
        # same number without hyphens
        ("630916011235", 1, (0, 12), 0.5),
        # female born 29-02-2000 (leap day), Kuala Lumpur (14),
        # serial 456, gender digit 2
        ("000229-14-4562", 1, (0, 14), 0.5),
        # male born 05-03-2010, Penang (07), serial 789, gender digit 1
        ("100305-07-7891", 1, (0, 14), 0.5),
        # only the first hyphen present still matches
        ("630916-011235", 1, (0, 13), 0.5),
        (
            "Her MyKad 630916-01-1235 was verified",
            1,
            (10, 24),
            0.5,
        ),
        # --- Invalid: impossible birth dates ---
        ("631316-01-1235", 0, (0, 0), 0),  # month 13
        ("630932-01-1235", 0, (0, 0), 0),  # day 32
        ("010229-01-1235", 0, (0, 0), 0),  # 29 Feb 2001 is not a leap day
        ("261231-01-1235", 0, (0, 0), 0),  # 31 Dec 2026 is in the future
        # --- Invalid: zero place of birth code or serial part ---
        ("630916-00-1235", 0, (0, 0), 0),
        ("630916-01-0000", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("111111111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("123456789012", 0, (0, 0), 0),
        ("654321098765", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("63091601123", 0, (0, 0), 0),  # 11 digits
        ("6309160112355", 0, (0, 0), 0),  # 13 digits
        # --- Lookalikes: same shape, but not a MyKad number ---
        ("Order 630916-01-1235X shipped", 0, (0, 0), 0),
        ("ID 630916-01-1235-67 on file", 0, (0, 0), 0),
        # fmt: on
    ],
)
def test_when_mykad_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid MyKad numbers and rejects invalid ones."""
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
    """Both MyKad numbers in one text are found at their exact positions."""
    results = recognizer.analyze(
        "IDs: 630916-01-1235 and 000229-14-4562", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 19, 0.5)
    assert_result(results[1], entities[0], 24, 38, 0.5)


sequential_test_set = [
    ["123456789012", True],
    ["654321098765", True],
    ["630916011235", False],
    ["000229144562", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the MyKad digits."""
    assert MyMykadRecognizer._is_sequential(input_text) == expected_output


def test_birth_date_check(recognizer):
    """The YYMMDD part must decode to a real past calendar date."""
    assert recognizer._birth_date_is_invalid("630916") is False
    assert recognizer._birth_date_is_invalid("000229") is False
    assert recognizer._birth_date_is_invalid("100305") is False
    assert recognizer._birth_date_is_invalid("631316") is True
    assert recognizer._birth_date_is_invalid("630932") is True
    assert recognizer._birth_date_is_invalid("010229") is True
    assert recognizer._birth_date_is_invalid("261231") is True


def test_invalidate_result(recognizer):
    """Sanity checks reject trivial numbers and accept real MyKad shapes."""
    assert recognizer.invalidate_result("630916-01-1235") is False
    assert recognizer.invalidate_result("630916011235") is False
    assert recognizer.invalidate_result("000229-14-4562") is False
    assert recognizer.invalidate_result("100305-07-7891") is False
    assert recognizer.invalidate_result("631316-01-1235") is True
    assert recognizer.invalidate_result("630916-00-1235") is True
    assert recognizer.invalidate_result("630916-01-0000") is True
    assert recognizer.invalidate_result("111111111111") is True
    assert recognizer.invalidate_result("123456789012") is True
    assert recognizer.invalidate_result("63091601123") is True
    assert recognizer.invalidate_result("6309160112355") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "malaysia",
        "malaysian",
        "mykad",
        "kad pengenalan",
        "nric",
        "national registration identity card",
        "identity card",
        "ic number",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["MY_MYKAD"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = MyMykadRecognizer(name="CustomMyMykad")
    assert recognizer.name == "CustomMyMykad"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a MyKad number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MyMykadRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="630916-01-1235", language="en", entities=["MY_MYKAD"]
    )
    with_context = analyzer.analyze(
        text="630916-01-1235",
        language="en",
        entities=["MY_MYKAD"],
        context=["mykad"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the sanity checks reject."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MyMykadRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="MyKad 631316-01-1235",
        language="en",
        entities=["MY_MYKAD"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "MyMykadRecognizer"]
    assert len(entries) == 1, "MyMykadRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "my"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "MyMykadRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "MY_MYKAD" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My MyKad is 630916-01-1235",
        language="en",
        entities=["MY_MYKAD"],
    )
    assert [result.entity_type for result in results] == ["MY_MYKAD"]
