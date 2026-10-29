"""Tests for Morocco National Identity Card (CNIE) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import MaCnieRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Morocco CNIE recognizer instance for testing."""
    return MaCnieRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Morocco CNIE entity type for testing."""
    return ["MA_CNIE"]


# Fixtures are synthetic numbers built to the published structure
# (1 to 3 region letters + 6-digit serial + 2-digit control number),
# because the recognizer implements no checksum: published sources
# describe the last two digits as a Luhn control number, but no
# authoritative Moroccan source with a worked valid example was found
# to verify the variant (see the recognizer docstring).
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid CNIEs ---
        # two letters, serial 482910, control 37
        ("AB48291037", 1, (0, 10), 0.5),
        # one letter (older province code), serial 791204, control 65
        ("C79120465", 1, (0, 9), 0.5),
        # three letters, serial 103847, control 29
        ("XYZ10384729", 1, (0, 11), 0.5),
        (
            "Her CNIE AB48291037 was verified",
            1,
            (9, 19),
            0.5,
        ),
        # matching is case-insensitive
        ("ab48291037", 1, (0, 10), 0.5),
        # --- Invalid: serial part 000000 ---
        ("AB00000037", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("AB11111111", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("AB12345678", 0, (0, 0), 0),
        ("AB87654321", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("AB1234567", 0, (0, 0), 0),  # 7 digits
        ("ABCD12345678", 0, (0, 0), 0),  # 4 letters
        ("AB482910371", 0, (0, 0), 0),  # 9 digits
        # --- Lookalikes: same shape, but not a CNIE ---
        ("Order AB48291037X shipped", 0, (0, 0), 0),
        ("abcdefghij", 0, (0, 0), 0),
        # fmt: on
    ],
)
def test_when_cnie_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid CNIEs and rejects invalid ones."""
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
    """Both CNIEs in one text are found at their exact positions."""
    results = recognizer.analyze(
        "IDs: AB48291037 and C79120465", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 15, 0.5)
    assert_result(results[1], entities[0], 20, 29, 0.5)


sequential_test_set = [
    ["12345678", True],
    ["87654321", True],
    ["48291037", False],
    ["79120465", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the CNIE digits."""
    assert MaCnieRecognizer._is_sequential(input_text) == expected_output


def test_invalidate_result(recognizer):
    """Sanity checks reject trivial numbers and accept real CNIE shapes."""
    assert recognizer.invalidate_result("AB48291037") is False
    assert recognizer.invalidate_result("C79120465") is False
    assert recognizer.invalidate_result("XYZ10384729") is False
    assert recognizer.invalidate_result("ab48291037") is False
    assert recognizer.invalidate_result("AB00000037") is True
    assert recognizer.invalidate_result("AB11111111") is True
    assert recognizer.invalidate_result("AB12345678") is True
    assert recognizer.invalidate_result("AB87654321") is True
    assert recognizer.invalidate_result("AB1234567") is True
    assert recognizer.invalidate_result("ABCD12345678") is True
    assert recognizer.invalidate_result("AB482910371") is True


def test_context_words(recognizer):
    """Test that context words are properly set."""
    expected_context = [
        "morocco",
        "moroccan",
        "cnie",
        "cin",
        "carte nationale",
        "national identity card",
        "identity card",
    ]
    assert recognizer.context == expected_context


def test_supported_entity(recognizer):
    """Test that supported entity is correctly set."""
    assert recognizer.supported_entities == ["MA_CNIE"]


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = MaCnieRecognizer(name="CustomMaCnie")
    assert recognizer.name == "CustomMaCnie"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a Morocco CNIE raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MaCnieRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="AB48291037", language="en", entities=["MA_CNIE"]
    )
    with_context = analyzer.analyze(
        text="AB48291037",
        language="en",
        entities=["MA_CNIE"],
        context=["cnie"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the sanity checks reject."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MaCnieRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="CNIE AB00000037",
        language="en",
        entities=["MA_CNIE"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "MaCnieRecognizer"]
    assert len(entries) == 1, "MaCnieRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ma"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "MaCnieRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "MA_CNIE" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My CNIE is AB48291037",
        language="en",
        entities=["MA_CNIE"],
    )
    assert [result.entity_type for result in results] == ["MA_CNIE"]
