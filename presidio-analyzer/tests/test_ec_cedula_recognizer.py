"""Tests for Ecuador Cedula de identidad recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import EcCedulaRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Ecuador cedula recognizer instance for testing."""
    return EcCedulaRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Ecuador cedula entity type for testing."""
    return ["EC_CEDULA"]


# Fixtures are synthetic numbers whose check digits were
# computed with the published mod-10 algorithm, except for
# 171430710-3 and 1714307103, which are the published worked
# example (python-stdnum doctest and MicroFocus example), and
# 0102260098, which is the Verifik documentation example.
# No fixture is a real person's number.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid cedula numbers ---
        # published worked example, hyphenated and plain
        ("171430710-3", 1, (0, 11), 0.5),
        ("1714307103", 1, (0, 10), 0.5),
        # Verifik documentation example, plain and hyphenated
        ("0102260098", 1, (0, 10), 0.5),
        ("010226009-8", 1, (0, 11), 0.5),
        # synthetic examples, check digits computed
        ("0921234563", 1, (0, 10), 0.5),
        # zero check digit
        ("1711000040", 1, (0, 10), 0.5),
        # special province codes 30 and 50
        ("3012345678", 1, (0, 10), 0.5),
        ("5012345673", 1, (0, 10), 0.5),
        (
            "Her cedula 171430710-3 was verified",
            1,
            (11, 22),
            0.5,
        ),
        # a sentence period after the check digit does not block the match
        ("Cedula 1714307103.", 1, (7, 17), 0.5),
        # --- Invalid: wrong check digit ---
        ("1714307104", 0, (0, 0), 0),
        ("0102260099", 0, (0, 0), 0),
        # --- Invalid: province code outside 01 to 24, 30, 50 ---
        # both have a valid checksum, so only the province rule rejects them
        ("0012345674", 0, (0, 0), 0),
        ("2512345675", 0, (0, 0), 0),
        # --- Invalid: third digit above 6 ---
        # both have a valid checksum, so only the third digit rule rejects them
        ("1771234562", 0, (0, 0), 0),
        ("1791234568", 0, (0, 0), 0),
        # --- Invalid: serial of all zeros (checksum valid) ---
        ("1700000001", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("0000000000", 0, (0, 0), 0),
        # --- Invalid: third digit above 6 on a special province code ---
        # checksum valid, so only the third digit rule rejects it
        ("3071234565", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("171430710", 0, (0, 0), 0),  # 9 digits
        ("17143071033", 0, (0, 0), 0),  # 11 digits
        # --- Lookalikes: same shape, but not a cedula number ---
        # a 13 digit personal RUC is not a 10 digit cedula
        ("1714307103001", 0, (0, 0), 0),
        ("Order 171430710-3X shipped", 0, (0, 0), 0),
        ("ID 171430710-3-4 on file", 0, (0, 0), 0),
        ("171430710 3", 0, (0, 0), 0),  # space form is not accepted
        ("171430710-34", 0, (0, 0), 0),  # extra digit after the hyphen
        # fmt: on
    ],
)
def test_when_cedula_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid cedulas and rejects invalid ones."""
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
    results = recognizer.analyze("IDs: 171430710-3 and 0102260098", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 16, 0.5)
    assert_result(results[1], entities[0], 21, 31, 0.5)


sequential_test_set = [
    ["1234567890", True],
    ["9876543210", True],
    ["1714307103", False],
    ["0921234563", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the cedula digits."""
    assert EcCedulaRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("171430710") == "3"
    assert recognizer._check_digit("010226009") == "8"
    assert recognizer._check_digit("171100004") == "0"


def test_has_valid_checksum(recognizer):
    """The mod-10 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("1714307103") is True
    assert recognizer._has_valid_checksum("0102260098") is True
    assert recognizer._has_valid_checksum("1714307104") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("171430710-3") is False
    assert recognizer.invalidate_result("0012345674") is True
    assert recognizer.invalidate_result("1771234562") is True
    assert recognizer.invalidate_result("1700000001") is True
    assert recognizer.invalidate_result("3071234565") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = EcCedulaRecognizer(name="CustomEcCedula")
    assert recognizer.name == "CustomEcCedula"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a cedula number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(EcCedulaRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="1714307103", language="en", entities=["EC_CEDULA"])
    with_context = analyzer.analyze(
        text="1714307103",
        language="en",
        entities=["EC_CEDULA"],
        context=["cedula"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(EcCedulaRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="Cedula 1714307104",
        language="en",
        entities=["EC_CEDULA"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "EcCedulaRecognizer"]
    assert len(entries) == 1, "EcCedulaRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ec"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["EcCedulaRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "EC_CEDULA" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My cedula is 171430710-3",
        language="en",
        entities=["EC_CEDULA"],
    )
    assert [result.entity_type for result in results] == ["EC_CEDULA"]
