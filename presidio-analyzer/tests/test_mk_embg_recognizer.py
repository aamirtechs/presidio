"""Tests for North Macedonian unique master citizen number recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import MkEmbgRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a North Macedonian unique master citizen number recognizer."""
    return MkEmbgRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the North Macedonian unique master citizen number entity type."""
    return ["MK_EMBG"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned weighted mod 11 check,
# except for 0807992455005, the Treeet EMBG checker
# worked example (a woman born 8 July 1992 in Skopje,
# region 45, serial 500). No fixture is a real
# person's number beyond that published test vector.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid unique master citizen numbers ---
        # synthetic numbers, check digits computed with the mod 11 check
        ("0807992455005", 1, (0, 13), 0.5),  # female 1992-07-08, r45, sum 204, check 5
        ("0101990410004", 1, (0, 13), 0.5),  # male 1990-01-01, r41, sum 84, check 4
        ("1503985435125", 1, (0, 13), 0.5),  # female 1985-03-15, r43, sum 193, check 5
        ("2902000472507", 1, (0, 13), 0.5),  # male, leap day 2000-02-29, r47, sum 158
        ("1212899494998", 1, (0, 13), 0.5),  # male 1899-12-12, r49, sum 267, check 8
        ("2007975046003", 1, (0, 13), 0.5),  # female 1975-07-20, foreigner r04, sum 162
        ("1006988420010", 1, (0, 13), 0.5),  # male 1988-06-10, r42, sum 166, check 0
        ("1006988440010", 1, (0, 13), 0.5),  # male 1988-06-10, r44, sum 176, check 0
        ("3011995469998", 1, (0, 13), 0.5),  # female 1995-11-30, r46, sum 245, check 8
        ("0505001481236", 1, (0, 13), 0.5),  # male 2001-05-05, r48, sum 137, check 6
        ("0101990491233", 1, (0, 13), 0.5),  # male 1990-01-01, r49, sum 140, check 3
        ("0101990410020", 1, (0, 13), 0.5),  # remainder 0, check digit 0, sum 88
        ("0101990410080", 1, (0, 13), 0.5),  # remainder 1, check digit 0, sum 100
        ("0807992450070", 1, (0, 13), 0.5),  # remainder 0, check digit 0, sum 198
        ("0807992450020", 1, (0, 13), 0.5),  # remainder 1, check digit 0, sum 188
        # a number embedded in a sentence
        ("EMBG 0807992455005 on file", 1, (5, 18), 0.5),
        ("EMBG 0101990410004", 1, (5, 18), 0.5),
        # a sentence period after the check digit does not block the match
        ("EMBG 0807992455005.", 1, (5, 18), 0.5),
        # --- Invalid: wrong check digit ---
        ("0807992455006", 0, (0, 0), 0),
        ("0101990410005", 0, (0, 0), 0),
        ("1503985435126", 0, (0, 0), 0),
        ("2902000472508", 0, (0, 0), 0),
        # --- Invalid: region code of another registry ---
        ("2007990501230", 0, (0, 0), 0),  # region 50, Slovenia
        ("2007990401235", 0, (0, 0), 0),  # region 40, not in use
        ("2007990011232", 0, (0, 0), 0),  # region 00, naturalized citizens
        ("2007990091236", 0, (0, 0), 0),  # region 09, naturalized citizens
        ("2007990601234", 0, (0, 0), 0),  # region 60, temporary residence
        ("2007990101231", 0, (0, 0), 0),  # region 01, foreigners in Bosnia
        ("2007990201236", 0, (0, 0), 0),  # region 02, foreigners in Montenegro
        ("2007990301230", 0, (0, 0), 0),  # region 03, foreigners in Croatia
        ("2007990701239", 0, (0, 0), 0),  # region 70, Serbia
        # --- Invalid: impossible birth date ---
        ("3201990410000", 0, (0, 0), 0),  # day 32
        ("2902199041003", 0, (0, 0), 0),  # 29 Feb 1999, not a leap year
        ("3113199041002", 0, (0, 0), 0),  # month 13
        ("0001990410000", 0, (0, 0), 0),  # day 00
        ("0100199041007", 0, (0, 0), 0),  # month 00
        ("2902200041004", 0, (0, 0), 0),  # 29 Feb 2200, century non-leap
        # --- Invalid: all digits identical ---
        ("1111111111111", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("0000000000000", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("9999999999999", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        # --- Invalid: fully sequential digit runs ---
        ("0123456789012", 0, (0, 0), 0),
        ("9876543210987", 0, (0, 0), 0),
        ("1234567890123", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("010199041000", 0, (0, 0), 0),  # 12 digits
        ("01019904100044", 0, (0, 0), 0),  # 14 digits
        # --- Lookalikes: same shape, but not a unique master citizen number ---
        ("10101990410004", 0, (0, 0), 0),  # a leading digit
        ("A0101990410004", 0, (0, 0), 0),  # a leading letter
        ("0101990410004A", 0, (0, 0), 0),  # a trailing letter
        ("0101 9904 10004", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("0101-9904-10004", 0, (0, 0), 0),  # dashed form, not matched
        # fmt: on
    ],
)
def test_when_embg_in_text_then_all_ids_found(
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
    results = recognizer.analyze("EMBGs: 0807992455005 and 0101990410004", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 7, 20, 0.5)
    assert_result(results[1], entities[0], 25, 38, 0.5)


sequential_test_set = [
    ["0123456789012", True],
    ["9876543210987", True],
    ["1234567890123", True],
    ["0807992455005", False],
    ["0101990410004", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert MkEmbgRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("080799245500") == "5"
    assert recognizer._check_digit("010199041000") == "4"
    assert recognizer._check_digit("150398543512") == "5"
    assert recognizer._check_digit("290200047250") == "7"
    assert recognizer._check_digit("121289949499") == "8"
    assert recognizer._check_digit("200797504600") == "3"
    assert recognizer._check_digit("100698842001") == "0"
    assert recognizer._check_digit("100698844001") == "0"
    assert recognizer._check_digit("301199546999") == "8"
    assert recognizer._check_digit("050500148123") == "6"
    assert recognizer._check_digit("010199049123") == "3"
    # remainder 0 and remainder 1 both give a check digit of 0
    assert recognizer._check_digit("010199041002") == "0"
    assert recognizer._check_digit("010199041008") == "0"
    assert recognizer._check_digit("080799245007") == "0"
    assert recognizer._check_digit("080799245002") == "0"


def test_has_valid_checksum(recognizer):
    """The mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("0807992455005") is True
    assert recognizer._has_valid_checksum("0101990410004") is True
    assert recognizer._has_valid_checksum("1503985435125") is True
    assert recognizer._has_valid_checksum("2902000472507") is True
    assert recognizer._has_valid_checksum("1212899494998") is True
    assert recognizer._has_valid_checksum("2007975046003") is True
    assert recognizer._has_valid_checksum("0101990410020") is True
    assert recognizer._has_valid_checksum("0101990410080") is True
    assert recognizer._has_valid_checksum("0807992455006") is False
    assert recognizer._has_valid_checksum("0101990410005") is False
    assert recognizer._has_valid_checksum("1503985435126") is False
    assert recognizer._has_valid_checksum("2902000472508") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("0807992455005") is True
    assert recognizer._has_valid_birth_date("0101990410004") is True
    assert recognizer._has_valid_birth_date("2902000472507") is True
    assert recognizer._has_valid_birth_date("1212899494998") is True
    assert recognizer._has_valid_birth_date("0101800410003") is True
    assert recognizer._has_valid_birth_date("3201990410000") is False
    assert recognizer._has_valid_birth_date("2902199041003") is False
    assert recognizer._has_valid_birth_date("3113199041002") is False
    assert recognizer._has_valid_birth_date("0001990410000") is False
    assert recognizer._has_valid_birth_date("0100199041007") is False
    assert recognizer._has_valid_birth_date("2902200041004") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    body = f"{tomorrow.day:02d}{tomorrow.month:02d}{tomorrow.year % 1000:03d}45000"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("0807992455005") is False
    assert recognizer.invalidate_result("0101990410004") is False
    assert recognizer.invalidate_result("2902000472507") is False
    assert recognizer.invalidate_result("2007975046003") is False
    assert recognizer.invalidate_result("1212899494998") is False
    assert recognizer.invalidate_result("0101990410020") is False
    assert recognizer.invalidate_result("0101990410080") is False
    assert recognizer.invalidate_result("0807992455006") is True
    assert recognizer.invalidate_result("0101990410005") is True
    assert recognizer.invalidate_result("2007990501230") is True
    assert recognizer.invalidate_result("2007990401235") is True
    assert recognizer.invalidate_result("3201990410000") is True
    assert recognizer.invalidate_result("2902199041003") is True
    assert recognizer.invalidate_result("1111111111111") is True
    assert recognizer.invalidate_result("0000000000000") is True
    assert recognizer.invalidate_result("9999999999999") is True
    assert recognizer.invalidate_result("0123456789012") is True
    assert recognizer.invalidate_result("9876543210987") is True
    assert recognizer.invalidate_result("010199041000") is True
    assert recognizer.invalidate_result("01019904100044") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = MkEmbgRecognizer(name="CustomMkEmbg")
    assert recognizer.name == "CustomMkEmbg"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MkEmbgRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="0807992455005", language="en", entities=["MK_EMBG"])
    with_context = analyzer.analyze(
        text="0807992455005",
        language="en",
        entities=["MK_EMBG"],
        context=["embg"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MkEmbgRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="embg 0807992455006",
        language="en",
        entities=["MK_EMBG"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "MkEmbgRecognizer"]
    assert len(entries) == 1, "MkEmbgRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "mk"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["MkEmbgRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "MK_EMBG" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My EMBG is 0807992455005",
        language="en",
        entities=["MK_EMBG"],
    )
    assert [result.entity_type for result in results] == ["MK_EMBG"]


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
        entries = [r for r in recognizers if r.get("name") == "MkEmbgRecognizer"]
        assert len(entries) == 1, f"MkEmbgRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"MkEmbgRecognizer enabled in {yaml_path}"
        )
