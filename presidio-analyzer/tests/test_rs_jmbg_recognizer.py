"""Tests for Serbian unique master citizen number recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import RsJmbgRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Serbian unique master citizen number recognizer for testing."""
    return RsJmbgRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Serbian unique master citizen number entity type for testing."""
    return ["RS_JMBG"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned weighted mod 11 check,
# except for 2007990711234, the number the Slovenian
# EMSO tests use as their Serbia region fixture (a man
# born 20 July 1990 in Belgrade, region 71, serial 123).
# No fixture is a real person's number beyond that
# published test vector.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid unique master citizen numbers ---
        # synthetic numbers, check digits computed with the mod 11 check
        ("0101990710008", 1, (0, 13), 0.5),  # male 1990-01-01, r71, sum 102, check 8
        ("1503985735129", 1, (0, 13), 0.5),  # female 1985-03-15, r73, sum 211, check 9
        ("2902000802507", 1, (0, 13), 0.5),  # male, leap day 2000-02-29, r80, sum 147
        ("2007975066004", 1, (0, 13), 0.5),  # female 1975-07-20, foreigner r06, sum 172
        ("0505001701236", 1, (0, 13), 0.5),  # male 2001-05-05, r70, sum 115, check 6
        ("1212899724993", 1, (0, 13), 0.5),  # male 1899-12-12, r72, sum 250, check 3
        ("3011995829994", 1, (0, 13), 0.5),  # female 1995-11-30, r82, sum 249, check 4
        ("1006988910017", 1, (0, 13), 0.5),  # male 1988-06-10, r91, sum 191, check 7
        ("0101990710040", 1, (0, 13), 0.5),  # remainder 0, check digit 0, sum 110
        ("0101990716090", 1, (0, 13), 0.5),  # remainder 1, check digit 0, sum 144
        ("2007990711234", 1, (0, 13), 0.5),  # SI tests Serbia fixture, sum 150, check 4
        # a number embedded in a sentence
        ("JMBG 0101990710008 on file", 1, (5, 18), 0.5),
        ("JMBG 1503985735129", 1, (5, 18), 0.5),
        # a sentence period after the check digit does not block the match
        ("JMBG 0101990710008.", 1, (5, 18), 0.5),
        # --- Invalid: wrong check digit ---
        ("0101990710009", 0, (0, 0), 0),
        ("1503985735120", 0, (0, 0), 0),
        ("2902000802508", 0, (0, 0), 0),
        ("2007975066005", 0, (0, 0), 0),
        # --- Invalid: region code of another registry ---
        ("2007990501230", 0, (0, 0), 0),  # region 50, Slovenia
        ("2007990411230", 0, (0, 0), 0),  # region 41, Skopje, Macedonia
        ("2007990101231", 0, (0, 0), 0),  # region 01, foreigners in Bosnia
        ("2007990201236", 0, (0, 0), 0),  # region 02, foreigners in Montenegro
        ("2007990301230", 0, (0, 0), 0),  # region 03, foreigners in Croatia
        ("2007992011230", 0, (0, 0), 0),  # region 20, Montenegro
        ("2007994011237", 0, (0, 0), 0),  # region 40, not in use
        ("2007996011234", 0, (0, 0), 0),  # region 60, temporary residence
        ("2007990001237", 0, (0, 0), 0),  # region 00, naturalized citizens
        ("2007990091236", 0, (0, 0), 0),  # region 09, naturalized citizens
        # --- Invalid: impossible birth date ---
        ("3201990710003", 0, (0, 0), 0),  # day 32
        ("2902199071000", 0, (0, 0), 0),  # 29 Feb 1999, not a leap year
        ("3113199071009", 0, (0, 0), 0),  # month 13
        ("0001990710003", 0, (0, 0), 0),  # day 00
        ("0100199071003", 0, (0, 0), 0),  # month 00
        ("2902200071000", 0, (0, 0), 0),  # 29 Feb 2200, century non-leap
        # --- Invalid: all digits identical ---
        ("1111111111111", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("0000000000000", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("9999999999999", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        # --- Invalid: fully sequential digit runs ---
        ("0123456789012", 0, (0, 0), 0),
        ("9876543210987", 0, (0, 0), 0),
        ("1234567890123", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("010199071000", 0, (0, 0), 0),  # 12 digits
        ("01019907100088", 0, (0, 0), 0),  # 14 digits
        # --- Lookalikes: same shape, but not a unique master citizen number ---
        ("10101990710008", 0, (0, 0), 0),  # a leading digit
        ("A0101990710008", 0, (0, 0), 0),  # a leading letter
        ("0101990710008A", 0, (0, 0), 0),  # a trailing letter
        ("0101 9907 10008", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("0101-9907-10008", 0, (0, 0), 0),  # dashed form, not matched
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
    results = recognizer.analyze("JMBGs: 0101990710008 and 1503985735129", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 7, 20, 0.5)
    assert_result(results[1], entities[0], 25, 38, 0.5)


sequential_test_set = [
    ["0123456789012", True],
    ["9876543210987", True],
    ["1234567890123", True],
    ["0101990710008", False],
    ["1503985735129", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert RsJmbgRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("010199071000") == "8"
    assert recognizer._check_digit("150398573512") == "9"
    assert recognizer._check_digit("290200080250") == "7"
    assert recognizer._check_digit("200797506600") == "4"
    assert recognizer._check_digit("050500170123") == "6"
    assert recognizer._check_digit("121289972499") == "3"
    assert recognizer._check_digit("301199582999") == "4"
    assert recognizer._check_digit("100698891001") == "7"
    assert recognizer._check_digit("200799071123") == "4"
    # remainder 0 and remainder 1 both give a check digit of 0
    assert recognizer._check_digit("010199071004") == "0"
    assert recognizer._check_digit("010199071609") == "0"


def test_has_valid_checksum(recognizer):
    """The mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("0101990710008") is True
    assert recognizer._has_valid_checksum("1503985735129") is True
    assert recognizer._has_valid_checksum("2902000802507") is True
    assert recognizer._has_valid_checksum("2007975066004") is True
    assert recognizer._has_valid_checksum("0505001701236") is True
    assert recognizer._has_valid_checksum("1212899724993") is True
    assert recognizer._has_valid_checksum("0101990710040") is True
    assert recognizer._has_valid_checksum("0101990716090") is True
    assert recognizer._has_valid_checksum("0101990710009") is False
    assert recognizer._has_valid_checksum("1503985735120") is False
    assert recognizer._has_valid_checksum("2902000802508") is False
    assert recognizer._has_valid_checksum("2007975066005") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("0101990710008") is True
    assert recognizer._has_valid_birth_date("1503985735129") is True
    assert recognizer._has_valid_birth_date("2902000802507") is True
    assert recognizer._has_valid_birth_date("1212899724993") is True
    assert recognizer._has_valid_birth_date("0101800710007") is True
    assert recognizer._has_valid_birth_date("3201990710003") is False
    assert recognizer._has_valid_birth_date("2902199071000") is False
    assert recognizer._has_valid_birth_date("3113199071009") is False
    assert recognizer._has_valid_birth_date("0001990710003") is False
    assert recognizer._has_valid_birth_date("0100199071003") is False
    assert recognizer._has_valid_birth_date("2902200071000") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    body = f"{tomorrow.day:02d}{tomorrow.month:02d}{tomorrow.year % 1000:03d}71000"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("0101990710008") is False
    assert recognizer.invalidate_result("1503985735129") is False
    assert recognizer.invalidate_result("2902000802507") is False
    assert recognizer.invalidate_result("2007975066004") is False
    assert recognizer.invalidate_result("0505001701236") is False
    assert recognizer.invalidate_result("1212899724993") is False
    assert recognizer.invalidate_result("0101990710009") is True
    assert recognizer.invalidate_result("1503985735120") is True
    assert recognizer.invalidate_result("2007990501230") is True
    assert recognizer.invalidate_result("2007990411230") is True
    assert recognizer.invalidate_result("3201990710003") is True
    assert recognizer.invalidate_result("2902199071000") is True
    assert recognizer.invalidate_result("1111111111111") is True
    assert recognizer.invalidate_result("0000000000000") is True
    assert recognizer.invalidate_result("9999999999999") is True
    assert recognizer.invalidate_result("0123456789012") is True
    assert recognizer.invalidate_result("9876543210987") is True
    assert recognizer.invalidate_result("010199071000") is True
    assert recognizer.invalidate_result("01019907100088") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = RsJmbgRecognizer(name="CustomRsJmbg")
    assert recognizer.name == "CustomRsJmbg"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(RsJmbgRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="0101990710008", language="en", entities=["RS_JMBG"])
    with_context = analyzer.analyze(
        text="0101990710008",
        language="en",
        entities=["RS_JMBG"],
        context=["jmbg"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(RsJmbgRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="jmbg 0101990710009",
        language="en",
        entities=["RS_JMBG"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "RsJmbgRecognizer"]
    assert len(entries) == 1, "RsJmbgRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "rs"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["RsJmbgRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "RS_JMBG" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My JMBG is 0101990710008",
        language="en",
        entities=["RS_JMBG"],
    )
    assert [result.entity_type for result in results] == ["RS_JMBG"]


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
        entries = [r for r in recognizers if r.get("name") == "RsJmbgRecognizer"]
        assert len(entries) == 1, f"RsJmbgRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"RsJmbgRecognizer enabled in {yaml_path}"
        )
