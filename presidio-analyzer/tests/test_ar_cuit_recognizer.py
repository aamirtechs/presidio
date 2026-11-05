"""Tests for Argentina CUIT/CUIL tax identifier recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import ArCuitRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Argentina CUIT/CUIL recognizer instance for testing."""
    return ArCuitRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Argentina CUIT/CUIL entity type for testing."""
    return ["AR_CUIT"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned mod-11 algorithm, except for
# 20-12345678-6 and 30-12345678-1, which are the worked
# examples from the task specification. No fixture is a
# real person's number.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid CUIT/CUIL numbers ---
        # worked examples, grouped and plain
        ("20-12345678-6", 1, (0, 13), 0.5),
        ("20123456786", 1, (0, 11), 0.5),
        ("30-12345678-1", 1, (0, 13), 0.5),
        ("30123456781", 1, (0, 11), 0.5),
        # every other allowed prefix, synthetic, check digits computed
        ("23123456785", 1, (0, 11), 0.5),
        ("24123456781", 1, (0, 11), 0.5),
        ("27123456780", 1, (0, 11), 0.5),  # remainder 0: check digit 0
        ("33123456780", 1, (0, 11), 0.5),
        ("34123456787", 1, (0, 11), 0.5),
        ("20987654326", 1, (0, 11), 0.5),
        ("27111111117", 1, (0, 11), 0.5),
        ("20111111112", 1, (0, 11), 0.5),
        (
            "Her CUIT 20-12345678-6 was verified",
            1,
            (9, 22),
            0.5,
        ),
        # a sentence period after the check digit does not block the match
        ("CUIT 20123456786.", 1, (5, 16), 0.5),
        # a company number embedded in a sentence
        ("Company CUIT 30-12345678-1 on file", 1, (13, 26), 0.5),
        # --- Invalid: wrong check digit ---
        ("20123456787", 0, (0, 0), 0),
        ("30123456782", 0, (0, 0), 0),
        # --- Invalid: remainder-1 combinations are never issued ---
        # body 2787654321 sums to 188, 188 mod 11 = 1, so no
        # check digit can validate it, not 0 and not 5
        ("27876543210", 0, (0, 0), 0),
        ("27876543215", 0, (0, 0), 0),
        # --- Invalid: prefixes outside the allowed set ---
        # each has a valid checksum, so only the prefix rule rejects it
        ("25123456788", 0, (0, 0), 0),  # 25 is not issued
        ("26123456784", 0, (0, 0), 0),  # 26 is not issued
        ("99123456781", 0, (0, 0), 0),
        ("10123456780", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("00000000000", 0, (0, 0), 0),
        ("22222222222", 0, (0, 0), 0),
        # --- Invalid: fully sequential runs ---
        ("23456789012", 0, (0, 0), 0),
        ("98765432109", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("2012345678", 0, (0, 0), 0),  # 10 digits
        ("201234567861", 0, (0, 0), 0),  # 12 digits
        # --- Lookalikes: same shape, but not a CUIT/CUIL number ---
        ("120123456786", 0, (0, 0), 0),  # a 12 digit run, not an 11 digit CUIT
        ("Order 20-12345678-6X shipped", 0, (0, 0), 0),
        ("ID 20-12345678-6-7 on file", 0, (0, 0), 0),
        ("20 12345678 6", 0, (0, 0), 0),  # space form is not accepted
        ("20-12345678-65", 0, (0, 0), 0),  # extra digit after the hyphen
        ("20.12345678.6", 0, (0, 0), 0),  # dot form is not accepted
        ("A20123456786", 0, (0, 0), 0),  # a letter run is not a CUIT
        # fmt: on
    ],
)
def test_when_cuit_in_text_then_all_ids_found(
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
    """Both CUIT numbers in one text are found at their exact positions."""
    results = recognizer.analyze("IDs: 20-12345678-6 and 30123456781", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 5, 18, 0.5)
    assert_result(results[1], entities[0], 23, 34, 0.5)


sequential_test_set = [
    ["12345678901", True],
    ["98765432109", True],
    ["20123456786", False],
    ["27111111117", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential digit run check used on the CUIT digits."""
    assert ArCuitRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the two worked examples."""
    assert recognizer._check_digit("2012345678") == "6"
    assert recognizer._check_digit("3012345678") == "1"
    assert recognizer._check_digit("2712345678") == "0"
    # remainder 1 means the combination is never issued
    assert recognizer._check_digit("2787654321") is None
    assert recognizer._check_digit("2000000001") is None


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("20123456786") is True
    assert recognizer._has_valid_checksum("30123456781") is True
    assert recognizer._has_valid_checksum("20123456787") is False
    assert recognizer._has_valid_checksum("27876543210") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("20-12345678-6") is False
    assert recognizer.invalidate_result("30-12345678-1") is False
    assert recognizer.invalidate_result("25123456788") is True
    assert recognizer.invalidate_result("26123456784") is True
    assert recognizer.invalidate_result("00000000000") is True
    assert recognizer.invalidate_result("23456789012") is True
    assert recognizer.invalidate_result("27876543210") is True
    assert recognizer.invalidate_result("2012345678") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = ArCuitRecognizer(name="CustomArCuit")
    assert recognizer.name == "CustomArCuit"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a CUIT number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(ArCuitRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="20123456786", language="en", entities=["AR_CUIT"])
    with_context = analyzer.analyze(
        text="20123456786",
        language="en",
        entities=["AR_CUIT"],
        context=["cuit"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(ArCuitRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="CUIT 20123456787",
        language="en",
        entities=["AR_CUIT"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "ArCuitRecognizer"]
    assert len(entries) == 1, "ArCuitRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ar"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["ArCuitRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "AR_CUIT" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My CUIT is 20-12345678-6",
        language="en",
        entities=["AR_CUIT"],
    )
    assert [result.entity_type for result in results] == ["AR_CUIT"]
