"""Tests for Indonesia National Identity Number (NIK) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import IdNikRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Indonesia NIK recognizer instance for testing."""
    return IdNikRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Indonesia NIK entity type for testing."""
    return ["ID_NIK"]


# Fixtures are synthetic numbers built to the published structure
# (6-digit area code + DDMMYY birth date, day plus 40 for women +
# 4-digit serial), because the NIK has no checksum: Microsoft Learn's
# Indonesia identity card entity definition states the checksum
# explicitly as "No" (see the recognizer docstring).
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid NIKs ---
        # area 317101, male born 17-08-1945, serial 0001
        ("3171011708450001", 1, (0, 16), 0.5),
        # area 317101, female born 17-08-1945 (day 57), serial 0002
        ("3171015708450002", 1, (0, 16), 0.5),
        # area 317101, male born 05-03-2010, serial 0003
        ("3171010503100003", 1, (0, 16), 0.5),
        # area 317101, female born 29-02-2000 (leap day, day 69), serial 0004
        ("3171016902000004", 1, (0, 16), 0.5),
        (
            "His NIK 3171011708450001 was verified",
            1,
            (8, 24),
            0.5,
        ),
        # --- Invalid: impossible birth dates ---
        ("3171011713450001", 0, (0, 0), 0),  # month 13
        ("3171013208450001", 0, (0, 0), 0),  # male day 32
        ("3171014008450001", 0, (0, 0), 0),  # day 40 is neither sex
        ("3171012902010001", 0, (0, 0), 0),  # 29 Feb 2001 is not a leap day
        ("3171011511260001", 0, (0, 0), 0),  # 15 Nov 2026 is in the future
        # --- Invalid: zero area code or serial ---
        ("0000001708450001", 0, (0, 0), 0),
        ("3171011708450000", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("1111111111111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("1234567890123456", 0, (0, 0), 0),
        ("6543210987654321", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("317101170845000", 0, (0, 0), 0),  # 15 digits
        ("31710117084500011", 0, (0, 0), 0),  # 17 digits
        # --- Lookalikes: same shape, but not a NIK ---
        ("Order 3171011708450001X shipped", 0, (0, 0), 0),
        ("1234567890123456 is a counter", 0, (0, 0), 0),
        # fmt: on
    ],
)
def test_when_nik_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid NIKs and rejects invalid ones."""
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
    """Both NIKs in one text are found at their exact positions."""
    results = recognizer.analyze(
        "IDs: 3171011708450001 and 3171015708450002", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 21, 0.5)
    assert_result(results[1], entities[0], 26, 42, 0.5)


sequential_test_set = [
    ["1234567890123456", True],
    ["6543210987654321", True],
    ["3171011708450001", False],
    ["3171015708450002", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the NIK digits."""
    assert IdNikRecognizer._is_sequential(input_text) == expected_output


def test_birth_date_check(recognizer):
    """The DDMMYY part must decode to a real past calendar date."""
    assert recognizer._birth_date_is_invalid("170845") is False
    assert recognizer._birth_date_is_invalid("570845") is False
    assert recognizer._birth_date_is_invalid("690200") is False
    assert recognizer._birth_date_is_invalid("171345") is True
    assert recognizer._birth_date_is_invalid("320845") is True
    assert recognizer._birth_date_is_invalid("400845") is True
    assert recognizer._birth_date_is_invalid("290201") is True
    assert recognizer._birth_date_is_invalid("151126") is True


def test_invalidate_result(recognizer):
    """Sanity checks reject trivial numbers and accept real NIK shapes."""
    assert recognizer.invalidate_result("3171011708450001") is False
    assert recognizer.invalidate_result("3171015708450002") is False
    assert recognizer.invalidate_result("3171010503100003") is False
    assert recognizer.invalidate_result("3171016902000004") is False
    assert recognizer.invalidate_result("3171011713450001") is True
    assert recognizer.invalidate_result("0000001708450001") is True
    assert recognizer.invalidate_result("3171011708450000") is True
    assert recognizer.invalidate_result("1111111111111111") is True
    assert recognizer.invalidate_result("1234567890123456") is True
    assert recognizer.invalidate_result("317101170845000") is True
    assert recognizer.invalidate_result("31710117084500011") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "indonesia",
        "indonesian",
        "nik",
        "nomor induk kependudukan",
        "ktp",
        "kartu tanda penduduk",
        "national identity",
        "identity card",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["ID_NIK"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = IdNikRecognizer(name="CustomIdNik")
    assert recognizer.name == "CustomIdNik"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around an Indonesia NIK raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(IdNikRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="3171011708450001", language="en", entities=["ID_NIK"]
    )
    with_context = analyzer.analyze(
        text="3171011708450001",
        language="en",
        entities=["ID_NIK"],
        context=["nik"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the sanity checks reject."""
    registry = RecognizerRegistry()
    registry.add_recognizer(IdNikRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="NIK 3171011713450001",
        language="en",
        entities=["ID_NIK"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "IdNikRecognizer"]
    assert len(entries) == 1, "IdNikRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "id"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "IdNikRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "ID_NIK" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My NIK is 3171011708450001",
        language="en",
        entities=["ID_NIK"],
    )
    assert [result.entity_type for result in results] == ["ID_NIK"]
