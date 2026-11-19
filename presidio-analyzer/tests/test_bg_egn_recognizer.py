"""Tests for Bulgarian uniform civil number recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import BgEgnRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Bulgarian uniform civil number recognizer for testing."""
    return BgEgnRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Bulgarian uniform civil number entity type for testing."""
    return ["BG_EGN"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned weighted mod 11 check,
# except for 8032056031, the python-stdnum docstring
# example, 1547121580, the Wikipedia example, and
# 7523169263, the python-stdnum compact example.
# The first Wikipedia example prints 7901050018, but
# its own products sum to 106 with remainder 7, so
# the number that follows from the stated algorithm
# is 7901050017, used here, and the printed 7901050018
# is used below as a wrong check digit case. No fixture
# is a real person's number beyond those published
# test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid uniform civil numbers ---
        # published examples, plain
        ("8032056031", 1, (0, 10), 0.5),  # stdnum docstring, sum 155, check 1
        ("1547121580", 1, (0, 10), 0.5),  # Wikipedia example, sum 187, check 0
        ("7523169263", 1, (0, 10), 0.5),  # stdnum compact example, sum 234, check 3
        ("7901050017", 1, (0, 10), 0.5),  # corrected Wikipedia ex 1, sum 106, check 7
        # synthetic numbers, check digits computed with the mod 11 check
        ("9002281238", 1, (0, 10), 0.5),  # male 1990-02-28, sum 151, check 8
        ("8802294567", 1, (0, 10), 0.5),  # male 1988-02-29 leap day, sum 238, check 7
        ("0042297896", 1, (0, 10), 0.5),  # female 2000-02-29 leap day, sum 270, check 6
        ("9912310001", 1, (0, 10), 0.5),  # 1999-12-31, serial 000, sum 111, check 1
        ("7501019995", 1, (0, 10), 0.5),  # female 1975-01-01, serial 999, check 5
        ("1547121596", 1, (0, 10), 0.5),  # female 2015-07-12, serial 159, check 6
        ("9912311230", 1, (0, 10), 0.5),  # remainder 10, check digit 0, sum 142
        ("6101047500", 1, (0, 10), 0.5),  # remainder 0, check digit 0, sum 121
        ("0101010012", 1, (0, 10), 0.5),  # female 1901-01-01, serial 001, check 2
        ("8032056026", 1, (0, 10), 0.5),  # male 1880-12-05, serial 602, check 6
        ("9921011235", 1, (0, 10), 0.5),  # 1899-01-01, month code 21, sum 115, check 5
        ("2601011237", 1, (0, 10), 0.5),  # male 2026-01-01, month code 41, check 7
        # a number embedded in a sentence
        ("EGN 8032056031 on file", 1, (4, 14), 0.5),
        ("EGN 1547121580", 1, (4, 14), 0.5),
        # a sentence period after the check digit does not block the match
        ("EGN 8032056031.", 1, (4, 14), 0.5),
        # --- Invalid: wrong check digit ---
        ("7901050018", 0, (0, 0), 0),  # Wikipedia printed number, check should be 7
        ("8032056032", 0, (0, 0), 0),
        ("1547121581", 0, (0, 0), 0),
        ("7523169264", 0, (0, 0), 0),
        # --- Invalid: impossible birth date ---
        ("3212991233", 0, (0, 0), 0),  # day 32
        ("2902991230", 0, (0, 0), 0),  # 29 Feb 1999, not a leap year
        ("9913991234", 0, (0, 0), 0),  # month 13
        ("0000991234", 0, (0, 0), 0),  # day 00
        ("9900991233", 0, (0, 0), 0),  # month 00
        ("0020291235", 0, (0, 0), 0),  # 29 Feb 1900, century non-leap
        ("9920991238", 0, (0, 0), 0),  # month code 20, no century
        ("9940991232", 0, (0, 0), 0),  # month code 40, no century
        # --- Invalid: all digits identical ---
        ("1111111111", 0, (0, 0), 0),
        ("0000000000", 0, (0, 0), 0),
        ("9999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("0123456789", 0, (0, 0), 0),
        ("9876543210", 0, (0, 0), 0),
        ("1234567890", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("803205603", 0, (0, 0), 0),  # 9 digits
        ("80320560311", 0, (0, 0), 0),  # 11 digits
        # --- Lookalikes: same shape, but not a uniform civil number ---
        ("18032056031", 0, (0, 0), 0),  # a leading digit
        ("A8032056031", 0, (0, 0), 0),  # a leading letter
        ("8032056031A", 0, (0, 0), 0),  # a trailing letter
        ("8032 0560 31", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("8032-0560-31", 0, (0, 0), 0),  # dashed form, not matched
        # fmt: on
    ],
)
def test_when_egn_in_text_then_all_ids_found(
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
    """Both uniform civil numbers in one text are found."""
    results = recognizer.analyze("EGNs: 8032056031 and 1547121580", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 16, 0.5)
    assert_result(results[1], entities[0], 21, 31, 0.5)


sequential_test_set = [
    ["0123456789", True],
    ["9876543210", True],
    ["1234567890", True],
    ["8032056031", False],
    ["1547121580", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert BgEgnRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("803205603") == "1"
    assert recognizer._check_digit("154712158") == "0"
    assert recognizer._check_digit("752316926") == "3"
    assert recognizer._check_digit("790105001") == "7"
    assert recognizer._check_digit("900228123") == "8"
    assert recognizer._check_digit("004229789") == "6"
    assert recognizer._check_digit("992101123") == "5"
    # remainder 10 and remainder 0 both give a check digit of 0
    assert recognizer._check_digit("991231123") == "0"
    assert recognizer._check_digit("610104750") == "0"


def test_has_valid_checksum(recognizer):
    """The mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("8032056031") is True
    assert recognizer._has_valid_checksum("1547121580") is True
    assert recognizer._has_valid_checksum("7523169263") is True
    assert recognizer._has_valid_checksum("7901050017") is True
    assert recognizer._has_valid_checksum("9912311230") is True
    assert recognizer._has_valid_checksum("6101047500") is True
    assert recognizer._has_valid_checksum("7901050018") is False
    assert recognizer._has_valid_checksum("8032056032") is False
    assert recognizer._has_valid_checksum("1547121581") is False
    assert recognizer._has_valid_checksum("7523169264") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("8032056031") is True
    assert recognizer._has_valid_birth_date("1547121580") is True
    assert recognizer._has_valid_birth_date("0042297896") is True
    assert recognizer._has_valid_birth_date("8802294567") is True
    assert recognizer._has_valid_birth_date("0101010012") is True
    assert recognizer._has_valid_birth_date("7523169263") is True
    assert recognizer._has_valid_birth_date("9921011235") is True
    assert recognizer._has_valid_birth_date("2601011237") is True
    assert recognizer._has_valid_birth_date("3212991233") is False
    assert recognizer._has_valid_birth_date("2902991230") is False
    assert recognizer._has_valid_birth_date("9913991234") is False
    assert recognizer._has_valid_birth_date("0000991234") is False
    assert recognizer._has_valid_birth_date("9900991233") is False
    assert recognizer._has_valid_birth_date("0020291235") is False
    assert recognizer._has_valid_birth_date("9920991238") is False
    assert recognizer._has_valid_birth_date("9940991232") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    if tomorrow.year >= 2000:
        month_code = tomorrow.month + 40
    elif tomorrow.year >= 1900:
        month_code = tomorrow.month
    else:
        month_code = tomorrow.month + 20
    body = f"{tomorrow.year % 100:02d}{month_code:02d}{tomorrow.day:02d}001"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("8032056031") is False
    assert recognizer.invalidate_result("1547121580") is False
    assert recognizer.invalidate_result("7523169263") is False
    assert recognizer.invalidate_result("7901050017") is False
    assert recognizer.invalidate_result("9912311230") is False
    assert recognizer.invalidate_result("6101047500") is False
    assert recognizer.invalidate_result("0042297896") is False
    assert recognizer.invalidate_result("9921011235") is False
    assert recognizer.invalidate_result("7901050018") is True
    assert recognizer.invalidate_result("8032056032") is True
    assert recognizer.invalidate_result("3212991233") is True
    assert recognizer.invalidate_result("2902991230") is True
    assert recognizer.invalidate_result("1111111111") is True
    assert recognizer.invalidate_result("0000000000") is True
    assert recognizer.invalidate_result("9999999999") is True
    assert recognizer.invalidate_result("0123456789") is True
    assert recognizer.invalidate_result("9876543210") is True
    assert recognizer.invalidate_result("803205603") is True
    assert recognizer.invalidate_result("80320560311") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = BgEgnRecognizer(name="CustomBgEgn")
    assert recognizer.name == "CustomBgEgn"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(BgEgnRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="8032056031", language="en", entities=["BG_EGN"])
    with_context = analyzer.analyze(
        text="8032056031",
        language="en",
        entities=["BG_EGN"],
        context=["egn"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(BgEgnRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="egn 7901050018",
        language="en",
        entities=["BG_EGN"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "BgEgnRecognizer"]
    assert len(entries) == 1, "BgEgnRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "bg"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["BgEgnRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "BG_EGN" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My EGN is 8032056031",
        language="en",
        entities=["BG_EGN"],
    )
    assert [result.entity_type for result in results] == ["BG_EGN"]


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
        entries = [r for r in recognizers if r.get("name") == "BgEgnRecognizer"]
        assert len(entries) == 1, f"BgEgnRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"BgEgnRecognizer enabled in {yaml_path}"
