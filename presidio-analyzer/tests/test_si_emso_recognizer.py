"""Tests for Slovenian unique master citizen number recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import SiEmsoRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Slovenian unique master citizen number recognizer for testing."""
    return SiEmsoRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Slovenian unique master citizen number entity type for testing."""
    return ["SI_EMSO"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned weighted mod 11 check,
# except for 0101006500006, the python-stdnum docstring
# example and Wikipedia example, and 1809987504991,
# the MicroFocus national ID example. No fixture is a
# real person's number beyond those published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid unique master citizen numbers ---
        # published examples, plain
        ("0101006500006", 1, (0, 13), 0.5),  # stdnum docstring, sum 82, check 6
        ("1809987504991", 1, (0, 13), 0.5),  # MicroFocus example, sum 274, check 1
        # synthetic numbers, check digits computed with the mod 11 check
        ("2007990501230", 1, (0, 13), 0.5),  # male 1990-07-20, sum 133, check 0
        ("1503985507002", 1, (0, 13), 0.5),  # female 1985-03-15, sum 185, check 2
        ("1501025510006", 1, (0, 13), 0.5),  # male 2025-01-15, region 51, check 6
        ("3006995052503", 1, (0, 13), 0.5),  # male 1995-06-30, region 05, check 3
        ("0101006505008", 1, (0, 13), 0.5),  # female 2006-01-01, serial 500, check 8
        ("3112999509995", 1, (0, 13), 0.5),  # female 1999-12-31, serial 999, check 5
        ("2902000501230", 1, (0, 13), 0.5),  # leap day 2000-02-29, check 0
        ("0101000500004", 1, (0, 13), 0.5),  # male 2000-01-01, serial 000, check 4
        ("2007990500020", 1, (0, 13), 0.5),  # remainder 0, check digit 0
        ("2007990500080", 1, (0, 13), 0.5),  # remainder 1, check digit 0
        # a number embedded in a sentence
        ("EMSO 0101006500006 on file", 1, (5, 18), 0.5),
        ("EMSO 1809987504991", 1, (5, 18), 0.5),
        # a sentence period after the check digit does not block the match
        ("EMSO 0101006500006.", 1, (5, 18), 0.5),
        # --- Invalid: wrong check digit ---
        ("0101006500007", 0, (0, 0), 0),  # stdnum docstring invalid example
        ("1809987504992", 0, (0, 0), 0),
        ("2007990501231", 0, (0, 0), 0),
        ("1503985507003", 0, (0, 0), 0),
        # --- Invalid: region code of another registry ---
        ("2007990711234", 0, (0, 0), 0),  # region 71, Belgrade, Serbia
        ("2007990411230", 0, (0, 0), 0),  # region 41, Skopje, Macedonia
        ("2007990001237", 0, (0, 0), 0),  # region 00, naturalized citizens
        ("2007990991237", 0, (0, 0), 0),  # region 09, naturalized citizens
        ("2007990601234", 0, (0, 0), 0),  # region 60, temporary residence
        # --- Invalid: impossible birth date ---
        ("3201990501234", 0, (0, 0), 0),  # day 32
        ("2902199050126", 0, (0, 0), 0),  # 29 Feb 1999, not a leap year
        ("3113199050125", 0, (0, 0), 0),  # month 13
        ("0001990501234", 0, (0, 0), 0),  # day 00
        ("0100199050120", 0, (0, 0), 0),  # month 00
        ("2902200501234", 0, (0, 0), 0),  # 29 Feb 2200, century non-leap
        # --- Invalid: all digits identical ---
        ("1111111111111", 0, (0, 0), 0),
        ("0000000000000", 0, (0, 0), 0),
        ("9999999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("0123456789012", 0, (0, 0), 0),
        ("9876543210987", 0, (0, 0), 0),
        ("1234567890123", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("010100650000", 0, (0, 0), 0),  # 12 digits
        ("01010065000066", 0, (0, 0), 0),  # 14 digits
        # --- Lookalikes: same shape, but not a unique master citizen number ---
        ("10101006500006", 0, (0, 0), 0),  # a leading digit
        ("A0101006500006", 0, (0, 0), 0),  # a leading letter
        ("0101006500006A", 0, (0, 0), 0),  # a trailing letter
        ("0101 0065 00006", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("0101-0065-00006", 0, (0, 0), 0),  # dashed form, not matched
        # fmt: on
    ],
)
def test_when_emso_in_text_then_all_ids_found(
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
    results = recognizer.analyze("EMSOS: 0101006500006 and 1809987504991", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 7, 20, 0.5)
    assert_result(results[1], entities[0], 25, 38, 0.5)


sequential_test_set = [
    ["0123456789012", True],
    ["9876543210987", True],
    ["1234567890123", True],
    ["0101006500006", False],
    ["1809987504991", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert SiEmsoRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("010100650000") == "6"
    assert recognizer._check_digit("180998750499") == "1"
    assert recognizer._check_digit("200799050123") == "0"
    assert recognizer._check_digit("150398550700") == "2"
    assert recognizer._check_digit("150102551000") == "6"
    assert recognizer._check_digit("300699505250") == "3"
    assert recognizer._check_digit("010100650500") == "8"
    # remainder 0 and remainder 1 both give a check digit of 0
    assert recognizer._check_digit("200799050002") == "0"
    assert recognizer._check_digit("200799050008") == "0"


def test_has_valid_checksum(recognizer):
    """The mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("0101006500006") is True
    assert recognizer._has_valid_checksum("1809987504991") is True
    assert recognizer._has_valid_checksum("2007990501230") is True
    assert recognizer._has_valid_checksum("1503985507002") is True
    assert recognizer._has_valid_checksum("2007990500020") is True
    assert recognizer._has_valid_checksum("2007990500080") is True
    assert recognizer._has_valid_checksum("0101006500007") is False
    assert recognizer._has_valid_checksum("1809987504992") is False
    assert recognizer._has_valid_checksum("2007990501231") is False
    assert recognizer._has_valid_checksum("1503985507003") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("0101006500006") is True
    assert recognizer._has_valid_birth_date("1809987504991") is True
    assert recognizer._has_valid_birth_date("2902000501230") is True
    assert recognizer._has_valid_birth_date("0101800500002") is True
    assert recognizer._has_valid_birth_date("3201990501234") is False
    assert recognizer._has_valid_birth_date("2902199050126") is False
    assert recognizer._has_valid_birth_date("3113199050125") is False
    assert recognizer._has_valid_birth_date("0001990501234") is False
    assert recognizer._has_valid_birth_date("0100199050120") is False
    assert recognizer._has_valid_birth_date("2902200501234") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    body = f"{tomorrow.day:02d}{tomorrow.month:02d}{tomorrow.year % 1000:03d}50000"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("0101006500006") is False
    assert recognizer.invalidate_result("1809987504991") is False
    assert recognizer.invalidate_result("2007990501230") is False
    assert recognizer.invalidate_result("1503985507002") is False
    assert recognizer.invalidate_result("1501025510006") is False
    assert recognizer.invalidate_result("3006995052503") is False
    assert recognizer.invalidate_result("0101006500007") is True
    assert recognizer.invalidate_result("1809987504992") is True
    assert recognizer.invalidate_result("2007990711234") is True
    assert recognizer.invalidate_result("2007990411230") is True
    assert recognizer.invalidate_result("3201990501234") is True
    assert recognizer.invalidate_result("2902199050126") is True
    assert recognizer.invalidate_result("1111111111111") is True
    assert recognizer.invalidate_result("0000000000000") is True
    assert recognizer.invalidate_result("9999999999999") is True
    assert recognizer.invalidate_result("0123456789012") is True
    assert recognizer.invalidate_result("9876543210987") is True
    assert recognizer.invalidate_result("010100650000") is True
    assert recognizer.invalidate_result("01010065000066") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = SiEmsoRecognizer(name="CustomSiEmso")
    assert recognizer.name == "CustomSiEmso"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(SiEmsoRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="0101006500006", language="en", entities=["SI_EMSO"])
    with_context = analyzer.analyze(
        text="0101006500006",
        language="en",
        entities=["SI_EMSO"],
        context=["emso"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(SiEmsoRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="emso 0101006500007",
        language="en",
        entities=["SI_EMSO"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "SiEmsoRecognizer"]
    assert len(entries) == 1, "SiEmsoRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "si"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["SiEmsoRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "SI_EMSO" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My EMSO is 0101006500006",
        language="en",
        entities=["SI_EMSO"],
    )
    assert [result.entity_type for result in results] == ["SI_EMSO"]


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
        entries = [r for r in recognizers if r.get("name") == "SiEmsoRecognizer"]
        assert len(entries) == 1, f"SiEmsoRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"SiEmsoRecognizer enabled in {yaml_path}"
        )
