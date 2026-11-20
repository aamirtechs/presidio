"""Tests for Greek tax identification number recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import GrAfmRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Greek tax identification number recognizer for testing."""
    return GrAfmRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Greek tax identification number entity type for testing."""
    return ["GR_AFM"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned weighted mod 11 mod 10 check,
# except for 094259216, the python-stdnum docstring
# example, and 090000045, the FastCalc worked example.
# The 8 digit old format, EL and GR prefixed forms, and
# spaced and dashed forms are accepted by python-stdnum
# but are not matched by this recognizer, which matches
# plain 9 digits only. No fixture is a real person's
# number beyond those published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid tax identification numbers ---
        # published examples, plain
        ("094259216", 1, (0, 9), 0.5),  # stdnum docstring, sum 1634, check 6
        ("090000045", 1, (0, 9), 0.5),  # FastCalc example, sum 1160, check 5
        # synthetic numbers, check digits computed with the mod 11 mod 10 check
        ("123456783", 1, (0, 9), 0.5),  # sum 1004, check 3
        ("000000050", 1, (0, 9), 0.5),  # remainder 10, check digit 0, sum 10
        ("110704340", 1, (0, 9), 0.5),  # remainder 0, check digit 0, sum 660
        ("631706690", 1, (0, 9), 0.5),  # remainder 10, check digit 0, sum 2298
        ("800000002", 1, (0, 9), 0.5),  # sum 2048, check 2
        ("555555559", 1, (0, 9), 0.5),  # sum 2550, check 9
        ("400000001", 1, (0, 9), 0.5),  # sum 1024, check 1
        ("999999993", 1, (0, 9), 0.5),  # sum 4590, check 3
        ("200000006", 1, (0, 9), 0.5),  # sum 512, check 6
        ("700000000", 1, (0, 9), 0.5),  # remainder 10, check digit 0, sum 1792
        ("987654324", 1, (0, 9), 0.5),  # near sequential body, sum 4096, check 4
        ("012345670", 1, (0, 9), 0.5),  # 7 to 0 breaks the run, sum 494, check 0
        # a number embedded in a sentence
        ("AFM 094259216 on file", 1, (4, 13), 0.5),
        ("AFM 090000045", 1, (4, 13), 0.5),
        # a sentence period after the check digit does not block the match
        ("AFM 094259216.", 1, (4, 13), 0.5),
        # --- Invalid: wrong check digit ---
        ("094259217", 0, (0, 0), 0),
        ("090000046", 0, (0, 0), 0),
        ("123456781", 0, (0, 0), 0),  # stdnum docstring invalid example
        ("000000051", 0, (0, 0), 0),
        ("110704341", 0, (0, 0), 0),
        ("631706691", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("000000000", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("111111111", 0, (0, 0), 0),
        ("999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("012345678", 0, (0, 0), 0),
        ("987654321", 0, (0, 0), 0),
        ("123456789", 0, (0, 0), 0),
        ("321098765", 0, (0, 0), 0),  # passes the checksum, rejected as sequential
        # --- Invalid: wrong lengths ---
        ("09425921", 0, (0, 0), 0),  # 8 digits, the old format stdnum pads
        ("0942592160", 0, (0, 0), 0),  # 10 digits
        # --- Lookalikes: same shape, but not a tax identification number ---
        ("1094259216", 0, (0, 0), 0),  # a leading digit
        ("A094259216", 0, (0, 0), 0),  # a leading letter
        ("094259216A", 0, (0, 0), 0),  # a trailing letter
        ("094 259 216", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("094-259-216", 0, (0, 0), 0),  # dashed form, not matched
        ("EL094259216", 0, (0, 0), 0),  # EL prefixed VAT form, not matched
        ("GR094259216", 0, (0, 0), 0),  # GR prefixed form, not matched
        # fmt: on
    ],
)
def test_when_afm_in_text_then_all_ids_found(
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
    """Both tax identification numbers in one text are found."""
    results = recognizer.analyze("AFMs: 094259216 and 090000045", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 15, 0.5)
    assert_result(results[1], entities[0], 20, 29, 0.5)


sequential_test_set = [
    ["012345678", True],
    ["987654321", True],
    ["123456789", True],
    ["321098765", True],
    ["876543210", True],
    ["094259216", False],
    ["090000045", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert GrAfmRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("09425921") == "6"
    assert recognizer._check_digit("09000004") == "5"
    assert recognizer._check_digit("12345678") == "3"
    assert recognizer._check_digit("80000000") == "2"
    assert recognizer._check_digit("55555555") == "9"
    assert recognizer._check_digit("40000000") == "1"
    assert recognizer._check_digit("99999999") == "3"
    # remainder 10 and remainder 0 both give a check digit of 0
    assert recognizer._check_digit("00000005") == "0"
    assert recognizer._check_digit("11070434") == "0"
    assert recognizer._check_digit("63170669") == "0"
    assert recognizer._check_digit("70000000") == "0"


def test_has_valid_checksum(recognizer):
    """The mod 11 mod 10 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("094259216") is True
    assert recognizer._has_valid_checksum("090000045") is True
    assert recognizer._has_valid_checksum("123456783") is True
    assert recognizer._has_valid_checksum("000000050") is True
    assert recognizer._has_valid_checksum("110704340") is True
    assert recognizer._has_valid_checksum("631706690") is True
    assert recognizer._has_valid_checksum("094259217") is False
    assert recognizer._has_valid_checksum("090000046") is False
    assert recognizer._has_valid_checksum("123456781") is False
    assert recognizer._has_valid_checksum("000000051") is False
    assert recognizer._has_valid_checksum("110704341") is False
    assert recognizer._has_valid_checksum("012345678") is False
    assert recognizer._has_valid_checksum("987654321") is False
    assert recognizer._has_valid_checksum("123456789") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("094259216") is False
    assert recognizer.invalidate_result("090000045") is False
    assert recognizer.invalidate_result("123456783") is False
    assert recognizer.invalidate_result("000000050") is False
    assert recognizer.invalidate_result("110704340") is False
    assert recognizer.invalidate_result("631706690") is False
    assert recognizer.invalidate_result("094259217") is True
    assert recognizer.invalidate_result("090000046") is True
    assert recognizer.invalidate_result("123456781") is True
    assert recognizer.invalidate_result("000000000") is True
    assert recognizer.invalidate_result("111111111") is True
    assert recognizer.invalidate_result("999999999") is True
    assert recognizer.invalidate_result("321098765") is True
    assert recognizer.invalidate_result("876543210") is True
    assert recognizer.invalidate_result("012345678") is True
    assert recognizer.invalidate_result("987654321") is True
    assert recognizer.invalidate_result("09425921") is True
    assert recognizer.invalidate_result("0942592160") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = GrAfmRecognizer(name="CustomGrAfm")
    assert recognizer.name == "CustomGrAfm"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(GrAfmRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="094259216", language="en", entities=["GR_AFM"])
    with_context = analyzer.analyze(
        text="094259216",
        language="en",
        entities=["GR_AFM"],
        context=["afm"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(GrAfmRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="afm 123456781",
        language="en",
        entities=["GR_AFM"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "GrAfmRecognizer"]
    assert len(entries) == 1, "GrAfmRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "gr"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["GrAfmRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "GR_AFM" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My AFM is 094259216",
        language="en",
        entities=["GR_AFM"],
    )
    assert [result.entity_type for result in results] == ["GR_AFM"]


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
        entries = [r for r in recognizers if r.get("name") == "GrAfmRecognizer"]
        assert len(entries) == 1, f"GrAfmRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"GrAfmRecognizer enabled in {yaml_path}"
