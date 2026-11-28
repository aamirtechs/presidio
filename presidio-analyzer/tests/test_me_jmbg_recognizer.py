"""Tests for Montenegrin unique master citizen number recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import MeJmbgRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Montenegrin unique master citizen number recognizer."""
    return MeJmbgRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Montenegrin unique master citizen number entity type."""
    return ["ME_JMBG"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned weighted mod 11 check,
# except for 2003988215003, the number the Treeet
# Montenegro JMBG checker uses as its worked example
# (a woman born 20 March 1988 in Podgorica, region
# 21, serial 500). No fixture is a real person's
# number beyond that published test vector.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid unique master citizen numbers ---
        # synthetic numbers, check digits computed with the mod 11 check
        ("2003988215003", 1, (0, 13), 0.5),  # female 1988-03-20, r21, sum 162, check 3
        ("0101990210005", 1, (0, 13), 0.5),  # male 1990-01-01, r21, sum 72, check 5
        ("1503985225120", 1, (0, 13), 0.5),  # female 1985-03-15, r22, sum 176, fold 0
        ("2902000232506", 1, (0, 13), 0.5),  # male, leap day 2000-02-29, r23, sum 126
        ("2007975246002", 1, (0, 13), 0.5),  # female 1975-07-20, r24, sum 174, check 2
        ("0505001251230", 1, (0, 13), 0.5),  # male 2001-05-05, r25, sum 110, fold 0
        ("1212899264992", 1, (0, 13), 0.5),  # male 1899-12-12, r26, sum 240, check 2
        ("0101990210030", 1, (0, 13), 0.5),  # remainder 1, check digit 0, sum 78
        ("0101990210080", 1, (0, 13), 0.5),  # remainder 0, check digit 0, sum 88
        ("0101990020001", 1, (0, 13), 0.5),  # male 1990-01-01, foreigner r02, sum 65
        ("0101990299995", 1, (0, 13), 0.5),  # female 1990-01-01, r29, sum 193, check 5
        # a number embedded in a sentence
        ("JMBG 2003988215003 on file", 1, (5, 18), 0.5),
        ("JMBG 0101990210005", 1, (5, 18), 0.5),
        # a sentence period after the check digit does not block the match
        ("JMBG 2003988215003.", 1, (5, 18), 0.5),
        # --- Invalid: wrong check digit ---
        ("2003988215004", 0, (0, 0), 0),
        ("0101990210006", 0, (0, 0), 0),
        ("2902000232507", 0, (0, 0), 0),
        ("2007975246003", 0, (0, 0), 0),
        # --- Invalid: region code of another registry ---
        ("2007990501230", 0, (0, 0), 0),  # region 50, Slovenia
        ("2007990411230", 0, (0, 0), 0),  # region 41, Skopje, Macedonia
        ("2007990011232", 0, (0, 0), 0),  # region 01, foreigners in Bosnia
        ("2007990031233", 0, (0, 0), 0),  # region 03, foreigners in Croatia
        ("2007990711234", 0, (0, 0), 0),  # region 71, Belgrade, Serbia
        ("2007990201236", 0, (0, 0), 0),  # region 20, not in use
        ("2007990401235", 0, (0, 0), 0),  # region 40, not in use
        ("2007990601234", 0, (0, 0), 0),  # region 60, temporary residence
        ("2007990001237", 0, (0, 0), 0),  # region 00, naturalized citizens
        ("2007990091236", 0, (0, 0), 0),  # region 09, naturalized citizens
        ("2007990101231", 0, (0, 0), 0),  # region 10, Bosnia and Herzegovina
        # --- Invalid: impossible birth date ---
        ("3201990210000", 0, (0, 0), 0),  # day 32
        ("2902199021002", 0, (0, 0), 0),  # 29 Feb 1999, not a leap year
        ("3113199021001", 0, (0, 0), 0),  # month 13
        ("0001990210000", 0, (0, 0), 0),  # day 00
        ("0100199021006", 0, (0, 0), 0),  # month 00
        ("2902200021003", 0, (0, 0), 0),  # 29 Feb 2200, century non-leap
        # --- Invalid: all digits identical ---
        ("1111111111111", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("0000000000000", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("9999999999999", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        # --- Invalid: fully sequential digit runs ---
        ("0123456789012", 0, (0, 0), 0),
        ("9876543210987", 0, (0, 0), 0),
        ("1234567890123", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("200398821500", 0, (0, 0), 0),  # 12 digits
        ("20039882150033", 0, (0, 0), 0),  # 14 digits
        # --- Lookalikes: same shape, but not a unique master citizen number ---
        ("12003988215003", 0, (0, 0), 0),  # a leading digit
        ("A2003988215003", 0, (0, 0), 0),  # a leading letter
        ("2003988215003A", 0, (0, 0), 0),  # a trailing letter
        ("2003 9882 15003", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("2003-9882-15003", 0, (0, 0), 0),  # dashed form, not matched
        # fmt: on
    ],
)
def test_when_jmbg_in_text_then_all_ids_found(
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
    """Both unique master citizen numbers in one text are found."""
    results = recognizer.analyze("JMBGs: 2003988215003 and 0101990210005", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 7, 20, 0.5)
    assert_result(results[1], entities[0], 25, 38, 0.5)


sequential_test_set = [
    ["0123456789012", True],
    ["9876543210987", True],
    ["1234567890123", True],
    ["2003988215003", False],
    ["0101990210005", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert MeJmbgRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("200398821500") == "3"
    assert recognizer._check_digit("010199021000") == "5"
    assert recognizer._check_digit("150398522512") == "0"
    assert recognizer._check_digit("290200023250") == "6"
    assert recognizer._check_digit("200797524600") == "2"
    assert recognizer._check_digit("050500125123") == "0"
    assert recognizer._check_digit("121289926499") == "2"
    assert recognizer._check_digit("010199002000") == "1"
    assert recognizer._check_digit("010199029999") == "5"
    # remainder 0 and remainder 1 both give a check digit of 0
    assert recognizer._check_digit("010199021008") == "0"
    assert recognizer._check_digit("010199021003") == "0"


def test_has_valid_checksum(recognizer):
    """The mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("2003988215003") is True
    assert recognizer._has_valid_checksum("0101990210005") is True
    assert recognizer._has_valid_checksum("1503985225120") is True
    assert recognizer._has_valid_checksum("2902000232506") is True
    assert recognizer._has_valid_checksum("2007975246002") is True
    assert recognizer._has_valid_checksum("0505001251230") is True
    assert recognizer._has_valid_checksum("1212899264992") is True
    assert recognizer._has_valid_checksum("0101990210030") is True
    assert recognizer._has_valid_checksum("0101990210080") is True
    assert recognizer._has_valid_checksum("2003988215004") is False
    assert recognizer._has_valid_checksum("0101990210006") is False
    assert recognizer._has_valid_checksum("2902000232507") is False
    assert recognizer._has_valid_checksum("2007975246003") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("2003988215003") is True
    assert recognizer._has_valid_birth_date("1503985225120") is True
    assert recognizer._has_valid_birth_date("2902000232506") is True
    assert recognizer._has_valid_birth_date("1212899264992") is True
    assert recognizer._has_valid_birth_date("0101800210004") is True
    assert recognizer._has_valid_birth_date("3201990210000") is False
    assert recognizer._has_valid_birth_date("2902199021002") is False
    assert recognizer._has_valid_birth_date("3113199021001") is False
    assert recognizer._has_valid_birth_date("0001990210000") is False
    assert recognizer._has_valid_birth_date("0100199021006") is False
    assert recognizer._has_valid_birth_date("2902200021003") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    body = f"{tomorrow.day:02d}{tomorrow.month:02d}{tomorrow.year % 1000:03d}21000"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("2003988215003") is False
    assert recognizer.invalidate_result("0101990210005") is False
    assert recognizer.invalidate_result("1503985225120") is False
    assert recognizer.invalidate_result("2902000232506") is False
    assert recognizer.invalidate_result("2007975246002") is False
    assert recognizer.invalidate_result("1212899264992") is False
    assert recognizer.invalidate_result("2003988215004") is True
    assert recognizer.invalidate_result("0101990210006") is True
    assert recognizer.invalidate_result("2007990501230") is True
    assert recognizer.invalidate_result("2007990411230") is True
    assert recognizer.invalidate_result("2007990711234") is True
    assert recognizer.invalidate_result("2007990201236") is True
    assert recognizer.invalidate_result("3201990210000") is True
    assert recognizer.invalidate_result("2902199021002") is True
    assert recognizer.invalidate_result("1111111111111") is True
    assert recognizer.invalidate_result("0000000000000") is True
    assert recognizer.invalidate_result("9999999999999") is True
    assert recognizer.invalidate_result("0123456789012") is True
    assert recognizer.invalidate_result("9876543210987") is True
    assert recognizer.invalidate_result("200398821500") is True
    assert recognizer.invalidate_result("20039882150033") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = MeJmbgRecognizer(name="CustomMeJmbg")
    assert recognizer.name == "CustomMeJmbg"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MeJmbgRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="2003988215003", language="en", entities=["ME_JMBG"])
    with_context = analyzer.analyze(
        text="2003988215003",
        language="en",
        entities=["ME_JMBG"],
        context=["jmbg"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MeJmbgRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="jmbg 2003988215004",
        language="en",
        entities=["ME_JMBG"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "MeJmbgRecognizer"]
    assert len(entries) == 1, "MeJmbgRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "me"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["MeJmbgRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "ME_JMBG" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My JMBG is 2003988215003",
        language="en",
        entities=["ME_JMBG"],
    )
    assert [result.entity_type for result in results] == ["ME_JMBG"]


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
        entries = [r for r in recognizers if r.get("name") == "MeJmbgRecognizer"]
        assert len(entries) == 1, f"MeJmbgRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"MeJmbgRecognizer enabled in {yaml_path}"
        )
