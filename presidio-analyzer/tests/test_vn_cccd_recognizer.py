"""Tests for Vietnam citizen identification number (CCCD) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import VnCccdRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Vietnam CCCD recognizer instance for testing."""
    return VnCccdRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Vietnam CCCD entity type for testing."""
    return ["VN_CCCD"]


# Fixtures are synthetic numbers built to the published structure
# (3-digit province code + gender and century digit + 2-digit
# birth year + 6-digit random serial, with or without hyphens),
# because the CCCD number has no checksum: the last six digits
# are a random serial, not a check value (see the recognizer
# docstring).
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid CCCD numbers ---
        # male born 1987, Hanoi (001), serial 123456
        ("001087123456", 1, (0, 12), 0.5),
        # same number with hyphens between the structural parts
        ("001-0-87-123456", 1, (0, 15), 0.5),
        # female born 2005, Ho Chi Minh City (079), serial 654321
        ("079305654321", 1, (0, 12), 0.5),
        # male born 2010, Hanoi (001), serial 789123
        ("001210789123", 1, (0, 12), 0.5),
        # only the first hyphen present still matches
        ("001-087123456", 1, (0, 13), 0.5),
        (
            "Her CCCD 079305654321 was verified",
            1,
            (9, 21),
            0.5,
        ),
        # --- Invalid: gender and century digit outside 0 to 3 ---
        ("001487123456", 0, (0, 0), 0),
        ("001987123456", 0, (0, 0), 0),
        # --- Invalid: birth year in the future (2027) ---
        ("001227123456", 0, (0, 0), 0),
        # --- Invalid: zero province code or serial part ---
        ("000087123456", 0, (0, 0), 0),
        ("001087000000", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("111111111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("123456789012", 0, (0, 0), 0),
        ("654321098765", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("00108712345", 0, (0, 0), 0),  # 11 digits
        ("0010871234567", 0, (0, 0), 0),  # 13 digits
        # --- Lookalikes: same shape, but not a CCCD number ---
        ("Order 001087123456X shipped", 0, (0, 0), 0),
        ("ID 001087123456-78 on file", 0, (0, 0), 0),
        # fmt: on
    ],
)
def test_when_cccd_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid CCCD numbers and rejects invalid ones."""
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
    """Both CCCD numbers in one text are found at their exact positions."""
    results = recognizer.analyze("IDs: 001087123456 and 079305654321", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 17, 0.5)
    assert_result(results[1], entities[0], 22, 34, 0.5)


sequential_test_set = [
    ["123456789012", True],
    ["654321098765", True],
    ["001087123456", False],
    ["079305654321", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the CCCD digits."""
    assert VnCccdRecognizer._is_sequential(input_text) == expected_output


def test_birth_year_check(recognizer):
    """The decoded birth year must not be in the future."""
    assert recognizer._birth_year_is_invalid("0", "87") is False
    assert recognizer._birth_year_is_invalid("1", "87") is False
    assert recognizer._birth_year_is_invalid("2", "05") is False
    assert recognizer._birth_year_is_invalid("3", "05") is False
    assert recognizer._birth_year_is_invalid("2", "27") is True
    assert recognizer._birth_year_is_invalid("3", "99") is True


def test_invalidate_result(recognizer):
    """Sanity checks reject trivial numbers and accept real CCCD shapes."""
    assert recognizer.invalidate_result("001087123456") is False
    assert recognizer.invalidate_result("001-0-87-123456") is False
    assert recognizer.invalidate_result("079305654321") is False
    assert recognizer.invalidate_result("001210789123") is False
    assert recognizer.invalidate_result("001487123456") is True
    assert recognizer.invalidate_result("001227123456") is True
    assert recognizer.invalidate_result("000087123456") is True
    assert recognizer.invalidate_result("001087000000") is True
    assert recognizer.invalidate_result("111111111111") is True
    assert recognizer.invalidate_result("123456789012") is True
    assert recognizer.invalidate_result("00108712345") is True
    assert recognizer.invalidate_result("0010871234567") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "vietnam",
        "vietnamese",
        "cccd",
        "can cuoc",
        "can cuoc cong dan",
        "citizen identity card",
        "identity card",
        "personal identification number",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["VN_CCCD"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = VnCccdRecognizer(name="CustomVnCccd")
    assert recognizer.name == "CustomVnCccd"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a CCCD number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(VnCccdRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="001087123456", language="en", entities=["VN_CCCD"])
    with_context = analyzer.analyze(
        text="001087123456",
        language="en",
        entities=["VN_CCCD"],
        context=["cccd"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the sanity checks reject."""
    registry = RecognizerRegistry()
    registry.add_recognizer(VnCccdRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="CCCD 001487123456",
        language="en",
        entities=["VN_CCCD"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "VnCccdRecognizer"]
    assert len(entries) == 1, "VnCccdRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "vn"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["VnCccdRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "VN_CCCD" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My CCCD is 001087123456",
        language="en",
        entities=["VN_CCCD"],
    )
    assert [result.entity_type for result in results] == ["VN_CCCD"]
