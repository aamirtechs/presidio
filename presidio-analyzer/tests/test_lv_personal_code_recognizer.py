"""Tests for Latvian personal code (personas kods) recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import LvPersonalCodeRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Latvian personal code recognizer for testing."""
    return LvPersonalCodeRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Latvian personal code entity type for testing."""
    return ["LV_PK"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned mod 11 check, except for
# 16117519997 and 32867300679 (the python-stdnum
# lv.pvn module docstring examples) and 11111111111
# (the PEAR Validate_LV personId example). No fixture
# is a real person's number beyond those published
# examples.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid personal codes ---
        # synthetic numbers, check digits computed with the mod 11 check
        ("16117519997", 1, (0, 11), 0.5),  # 16 Nov 1975, c1, s999, sum 291
        ("01010000000", 1, (0, 11), 0.5),  # 1 Jan 1800, c0, s000, fold, check 0
        ("01010020000", 1, (0, 11), 0.5),  # 1 Jan 2000, c2, s000, sum 23, check 0
        ("29020020009", 1, (0, 11), 0.5),  # leap day 2000-02-29, sum 80
        ("31129900002", 1, (0, 11), 0.5),  # 31 Dec 1899, c0, s000, sum 197
        ("31129910008", 1, (0, 11), 0.5),  # 31 Dec 1999, c1, s000, sum 202
        ("32867300679", 1, (0, 11), 0.5),  # new format, starts 32, sum 212
        ("38123456782", 1, (0, 11), 0.5),  # new format, starts 38, sum 252
        ("39123456787", 1, (0, 11), 0.5),  # new format, starts 39, sum 258
        # a number embedded in a sentence
        ("Personas kods 16117519997 on file", 1, (14, 25), 0.5),
        ("Personas kods 32867300679", 1, (14, 25), 0.5),
        # a sentence period after the check digit does not block the match
        ("Personas kods 16117519997.", 1, (14, 25), 0.5),
        # --- Invalid: wrong check digit ---
        ("16117519998", 0, (0, 0), 0),
        ("32867300670", 0, (0, 0), 0),
        ("11111111112", 0, (0, 0), 0),
        ("01010020001", 0, (0, 0), 0),
        # --- Invalid: century digit 3 or 9, or a legal entity PVN ---
        # each carries a valid checksum and a real date; stdnum accepts them
        ("01010030006", 0, (0, 0), 0),  # century 3, year 2100
        ("01010090009", 0, (0, 0), 0),  # century 9, year 2700
        ("40003521600", 0, (0, 0), 0),  # legal entity PVN, stdnum valid
        # --- Invalid: impossible birth date ---
        ("16137519997", 0, (0, 0), 0),  # month 13, stdnum invalid example
        ("29020010003", 0, (0, 0), 0),  # 29 Feb 1900, not a leap year
        ("30023310000", 0, (0, 0), 0),  # 30 Feb 1933
        ("00127510009", 0, (0, 0), 0),  # day 00
        ("01007510009", 0, (0, 0), 0),  # month 00
        # --- Invalid: future birth date ---
        ("01012720000", 0, (0, 0), 0),  # 1 Jan 2027, stdnum accepts, rejected here
        # --- Invalid: all digits identical ---
        ("00000000000", 0, (0, 0), 0),
        ("11111111111", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("99999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("01234567890", 0, (0, 0), 0),  # passes the checksum, rejected as sequential
        ("98765432109", 0, (0, 0), 0),
        ("12345678901", 0, (0, 0), 0),
        ("65432109876", 0, (0, 0), 0),  # passes the checksum, rejected as sequential
        # --- Invalid: wrong lengths ---
        ("1611751999", 0, (0, 0), 0),  # 10 digits
        ("161175199970", 0, (0, 0), 0),  # 12 digits
        # --- Lookalikes: same shape, but not a personal code ---
        ("116117519997", 0, (0, 0), 0),  # a leading digit
        ("A16117519997", 0, (0, 0), 0),  # a leading letter
        ("16117519997A", 0, (0, 0), 0),  # a trailing letter
        ("161175-19997", 0, (0, 0), 0),  # dashed form, stdnum accepts, not matched
        ("1611 7519 997", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        # fmt: on
    ],
)
def test_when_personal_code_in_text_then_all_ids_found(
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
    """Both personal codes in one text are found."""
    results = recognizer.analyze("Personas kods: 16117519997 and 32867300679", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 15, 26, 0.5)
    assert_result(results[1], entities[0], 31, 42, 0.5)


sequential_test_set = [
    ["01234567890", True],
    ["98765432109", True],
    ["12345678901", True],
    ["65432109876", True],
    ["16117519997", False],
    ["32867300679", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert LvPersonalCodeRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("1611751999") == "7"
    assert recognizer._check_digit("1111111111") == "1"
    # the fold case: the remainder is 10, so the check digit is 0
    assert recognizer._check_digit("0101000000") == "0"
    assert recognizer._check_digit("0101002000") == "0"
    assert recognizer._check_digit("2902002000") == "9"
    assert recognizer._check_digit("3112990000") == "2"
    assert recognizer._check_digit("3112991000") == "8"
    assert recognizer._check_digit("3286730067") == "9"
    assert recognizer._check_digit("3812345678") == "2"
    assert recognizer._check_digit("3912345678") == "7"


def test_has_valid_checksum(recognizer):
    """The mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("16117519997") is True
    assert recognizer._has_valid_checksum("11111111111") is True
    assert recognizer._has_valid_checksum("01010000000") is True
    assert recognizer._has_valid_checksum("01010020000") is True
    assert recognizer._has_valid_checksum("29020020009") is True
    assert recognizer._has_valid_checksum("32867300679") is True
    assert recognizer._has_valid_checksum("38123456782") is True
    assert recognizer._has_valid_checksum("16117519998") is False
    assert recognizer._has_valid_checksum("32867300670") is False
    assert recognizer._has_valid_checksum("11111111112") is False
    assert recognizer._has_valid_checksum("01010020001") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("16117519997") is True
    assert recognizer._has_valid_birth_date("11111111111") is True
    assert recognizer._has_valid_birth_date("01010000000") is True
    assert recognizer._has_valid_birth_date("29020020009") is True
    assert recognizer._has_valid_birth_date("31129900002") is True
    assert recognizer._has_valid_birth_date("16137519997") is False
    assert recognizer._has_valid_birth_date("29020010003") is False
    assert recognizer._has_valid_birth_date("30023310000") is False
    assert recognizer._has_valid_birth_date("00127510009") is False
    assert recognizer._has_valid_birth_date("01007510009") is False


def test_is_new_format(recognizer):
    """New format codes start with 3 and a second digit from 2 to 9."""
    assert recognizer._is_new_format("32867300679") is True
    assert recognizer._is_new_format("38123456782") is True
    assert recognizer._is_new_format("39123456787") is True
    assert recognizer._is_new_format("16117519997") is False
    # day 31, an old format code, not a new format code
    assert recognizer._is_new_format("31129910008") is False
    # day 30, an old format code, not a new format code
    assert recognizer._is_new_format("30023310000") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    body = f"{tomorrow.day:02d}{tomorrow.month:02d}{tomorrow.year % 100:02d}2000"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("16117519997") is False
    assert recognizer.invalidate_result("01010000000") is False
    assert recognizer.invalidate_result("29020020009") is False
    assert recognizer.invalidate_result("32867300679") is False
    assert recognizer.invalidate_result("38123456782") is False
    assert recognizer.invalidate_result("16117519998") is True
    assert recognizer.invalidate_result("01010030006") is True
    assert recognizer.invalidate_result("01010090009") is True
    assert recognizer.invalidate_result("40003521600") is True
    assert recognizer.invalidate_result("16137519997") is True
    assert recognizer.invalidate_result("29020010003") is True
    assert recognizer.invalidate_result("01012720000") is True
    assert recognizer.invalidate_result("00000000000") is True
    assert recognizer.invalidate_result("11111111111") is True
    assert recognizer.invalidate_result("99999999999") is True
    assert recognizer.invalidate_result("01234567890") is True
    assert recognizer.invalidate_result("65432109876") is True
    assert recognizer.invalidate_result("1611751999") is True
    assert recognizer.invalidate_result("161175199970") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = LvPersonalCodeRecognizer(name="CustomLvPersonalCode")
    assert recognizer.name == "CustomLvPersonalCode"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(LvPersonalCodeRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="16117519997", language="en", entities=["LV_PK"])
    with_context = analyzer.analyze(
        text="16117519997",
        language="en",
        entities=["LV_PK"],
        context=["personas kods"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(LvPersonalCodeRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="personas kods 16117519998",
        language="en",
        entities=["LV_PK"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "LvPersonalCodeRecognizer"]
    assert len(entries) == 1, "LvPersonalCodeRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "lv"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "LvPersonalCodeRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "LV_PK" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My personas kods is 16117519997",
        language="en",
        entities=["LV_PK"],
    )
    assert [result.entity_type for result in results] == ["LV_PK"]


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
        entries = [
            r for r in recognizers if r.get("name") == "LvPersonalCodeRecognizer"
        ]
        assert len(entries) == 1, f"LvPersonalCodeRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"LvPersonalCodeRecognizer enabled in {yaml_path}"
        )
