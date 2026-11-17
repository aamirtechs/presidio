"""Tests for Croatian personal identification number recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import HrOibRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Croatian personal identification number recognizer for testing."""
    return HrOibRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Croatian personal identification number entity type for testing."""
    return ["HR_OIB"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned ISO 7064 Mod 11, 10 check,
# except for 33392005961, the python-stdnum docstring
# example, 69435151530, the FastCalc validator worked
# example, and 89018712265, the VAT number example
# HR89018712265. No fixture is a real person's number
# beyond those published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid personal identification numbers ---
        # published examples, plain
        ("33392005961", 1, (0, 11), 0.5),  # stdnum docstring, checksum ends 1
        ("69435151530", 1, (0, 11), 0.5),  # FastCalc worked example, checksum ends 1
        ("89018712265", 1, (0, 11), 0.5),  # VAT example HR89018712265, checksum ends 1
        # synthetic numbers, check digits computed with the Mod 11, 10 check
        ("12345678903", 1, (0, 11), 0.5),  # check digit 3
        ("00000000010", 1, (0, 11), 0.5),  # leading zeros, check digit 0
        ("10000000000", 1, (0, 11), 0.5),  # check digit 0
        ("99999999994", 1, (0, 11), 0.5),  # check digit 4
        ("55555555551", 1, (0, 11), 0.5),  # check digit 1
        ("20261117016", 1, (0, 11), 0.5),  # check digit 6
        ("98765432106", 1, (0, 11), 0.5),  # check digit 6
        ("11122233343", 1, (0, 11), 0.5),  # check digit 3
        # a number embedded in a sentence
        ("OIB 33392005961 on file", 1, (4, 15), 0.5),
        ("HR 69435151530", 1, (3, 14), 0.5),
        # a sentence period after the check digit does not block the match
        ("OIB 33392005961.", 1, (4, 15), 0.5),
        # --- Invalid: wrong check digit ---
        ("33392005962", 0, (0, 0), 0),  # stdnum docstring invalid example
        ("69435151531", 0, (0, 0), 0),
        ("89018712266", 0, (0, 0), 0),
        ("12345678904", 0, (0, 0), 0),
        ("39074615284", 0, (0, 0), 0),  # TaxDo fictitious example, fails the check
        # --- Invalid: all digits identical ---
        # no all-same-digit string passes the checksum
        ("11111111111", 0, (0, 0), 0),
        ("00000000000", 0, (0, 0), 0),
        ("99999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        # 90123456789 passes the checksum but is rejected by the sequential rule
        ("01234567890", 0, (0, 0), 0),
        ("98765432109", 0, (0, 0), 0),
        ("90123456789", 0, (0, 0), 0),
        ("12345678901", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("3339200596", 0, (0, 0), 0),  # 10 digits
        ("333920059611", 0, (0, 0), 0),  # 12 digits
        # --- Lookalikes: same shape, but not a personal identification number ---
        ("133392005961", 0, (0, 0), 0),  # a leading digit
        ("A33392005961", 0, (0, 0), 0),  # a leading letter
        ("33392005961A", 0, (0, 0), 0),  # a trailing letter
        ("HR89018712265", 0, (0, 0), 0),  # HR prefixed VAT form, not matched
        ("3339 2005 961", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("3339-2005-961", 0, (0, 0), 0),  # dashed form, stdnum accepts, not matched
        # fmt: on
    ],
)
def test_when_oib_in_text_then_all_ids_found(
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
    """Both personal identification numbers in one text are found."""
    results = recognizer.analyze("OIBs: 33392005961 and 69435151530", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 17, 0.5)
    assert_result(results[1], entities[0], 22, 33, 0.5)


sequential_test_set = [
    ["01234567890", True],
    ["98765432109", True],
    ["90123456789", True],
    ["12345678901", True],
    ["33392005961", False],
    ["69435151530", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert HrOibRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("3339200596") == "1"
    assert recognizer._check_digit("6943515153") == "0"
    assert recognizer._check_digit("8901871226") == "5"
    assert recognizer._check_digit("1234567890") == "3"
    assert recognizer._check_digit("0000000001") == "0"
    assert recognizer._check_digit("9999999999") == "4"
    # every 10 digit body has a valid check digit
    assert recognizer._check_digit("5555555555") == "1"
    assert recognizer._check_digit("9876543210") == "6"


def test_has_valid_checksum(recognizer):
    """The Mod 11, 10 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("33392005961") is True
    assert recognizer._has_valid_checksum("69435151530") is True
    assert recognizer._has_valid_checksum("89018712265") is True
    assert recognizer._has_valid_checksum("12345678903") is True
    assert recognizer._has_valid_checksum("00000000010") is True
    assert recognizer._has_valid_checksum("10000000000") is True
    assert recognizer._has_valid_checksum("99999999994") is True
    assert recognizer._has_valid_checksum("33392005962") is False
    assert recognizer._has_valid_checksum("69435151531") is False
    assert recognizer._has_valid_checksum("89018712266") is False
    assert recognizer._has_valid_checksum("12345678904") is False
    assert recognizer._has_valid_checksum("39074615284") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("33392005961") is False
    assert recognizer.invalidate_result("69435151530") is False
    assert recognizer.invalidate_result("89018712265") is False
    assert recognizer.invalidate_result("12345678903") is False
    assert recognizer.invalidate_result("00000000010") is False
    assert recognizer.invalidate_result("10000000000") is False
    assert recognizer.invalidate_result("33392005962") is True
    assert recognizer.invalidate_result("69435151531") is True
    assert recognizer.invalidate_result("39074615284") is True
    assert recognizer.invalidate_result("11111111111") is True
    assert recognizer.invalidate_result("00000000000") is True
    assert recognizer.invalidate_result("99999999999") is True
    assert recognizer.invalidate_result("01234567890") is True
    assert recognizer.invalidate_result("98765432109") is True
    assert recognizer.invalidate_result("90123456789") is True
    assert recognizer.invalidate_result("3339200596") is True
    assert recognizer.invalidate_result("333920059611") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = HrOibRecognizer(name="CustomHrOib")
    assert recognizer.name == "CustomHrOib"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(HrOibRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="33392005961", language="en", entities=["HR_OIB"])
    with_context = analyzer.analyze(
        text="33392005961",
        language="en",
        entities=["HR_OIB"],
        context=["oib"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(HrOibRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="oib 33392005962",
        language="en",
        entities=["HR_OIB"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "HrOibRecognizer"]
    assert len(entries) == 1, "HrOibRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "hr"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["HrOibRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "HR_OIB" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My OIB is 33392005961",
        language="en",
        entities=["HR_OIB"],
    )
    assert [result.entity_type for result in results] == ["HR_OIB"]


def test_registered_disabled_in_subset_yamls():
    """Recognizer is listed with enabled false in the subset YAMLs."""
    analyzer_dir = Path(presidio_analyzer.__file__).parent
    repo_root = analyzer_dir.parents[1]
    yaml_paths = [
        analyzer_dir / "conf" / "slim.yaml",
        analyzer_dir / "conf" / "default_analyzer_full.yaml",
        repo_root / "e2e-tests" / "resources" / "test_ollama_enabled_recognizers.yaml",
    ]
    for yaml_path in yaml_paths:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        recognizers = content.get("recognizers")
        if recognizers is None:
            recognizers = content["recognizer_registry"]["recognizers"]
        entries = [r for r in recognizers if r.get("name") == "HrOibRecognizer"]
        assert len(entries) == 1, f"HrOibRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"HrOibRecognizer enabled in {yaml_path}"
