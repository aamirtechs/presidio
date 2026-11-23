"""Tests for Estonian personal identification code (isikukood) recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import EeIsikukoodRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Estonian personal identification code recognizer for testing."""
    return EeIsikukoodRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Estonian personal identification code entity type for testing."""
    return ["EE_IK"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned two round mod 11 check,
# except for 36805280109 (the python-stdnum ee.ik
# module docstring example). No fixture is a real
# person's number beyond that published example.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid personal codes ---
        # synthetic numbers, check digits computed with the two round check
        ("36805280109", 1, (0, 11), 0.5),  # male 1968-05-28, s010, sum1 141
        ("37605030299", 1, (0, 11), 0.5),  # male 1976-05-03, s029, sum 108
        ("49002020002", 1, (0, 11), 0.5),  # female 1990-02-02, s000, sum 46
        ("60002290003", 1, (0, 11), 0.5),  # female, leap day 2000-02-29, sum 91
        ("50001010006", 1, (0, 11), 0.5),  # male 2000-01-01, s000, sum 17, check 6
        ("10101010005", 1, (0, 11), 0.5),  # male 1801-01-01, s000, sum 16, check 5
        ("29912310009", 1, (0, 11), 0.5),  # female 1899-12-31, s000, sum 86
        ("19912310008", 1, (0, 11), 0.5),  # male 1899-12-31, s000, sum 85, check 8
        ("23103250882", 1, (0, 11), 0.5),  # female 1831-03-25, s088, 2nd round
        ("26909016130", 1, (0, 11), 0.5),  # female 1869-09-01, s613, both rounds 10
        # a number embedded in a sentence
        ("Isikukood 36805280109 on file", 1, (10, 21), 0.5),
        ("Isikukood 37605030299", 1, (10, 21), 0.5),
        # a sentence period after the check digit does not block the match
        ("Isikukood 36805280109.", 1, (10, 21), 0.5),
        # --- Invalid: wrong check digit ---
        ("36805280108", 0, (0, 0), 0),  # stdnum docstring invalid example
        ("37605030290", 0, (0, 0), 0),
        ("49002020003", 0, (0, 0), 0),
        ("60002290004", 0, (0, 0), 0),
        # --- Invalid: first digit 7, 8, 9 or 0 ---
        # each carries a valid checksum and a real date; stdnum accepts 7 and 8
        ("76805280102", 0, (0, 0), 0),  # first digit 7, 2100 century
        ("86805280103", 0, (0, 0), 0),  # first digit 8, 2100 century
        ("96805280104", 0, (0, 0), 0),  # first digit 9, stdnum rejects too
        ("06805280106", 0, (0, 0), 0),  # first digit 0, stdnum rejects too
        # --- Invalid: impossible birth date ---
        ("33302300068", 0, (0, 0), 0),  # 30 Feb 1933
        ("40002290001", 0, (0, 0), 0),  # 29 Feb 1900, not a leap year
        ("33313310062", 0, (0, 0), 0),  # month 13
        ("33301000067", 0, (0, 0), 0),  # day 00
        ("33300010069", 0, (0, 0), 0),  # month 00
        # --- Invalid: future birth date ---
        ("52701010009", 0, (0, 0), 0),  # 1 Jan 2027, stdnum accepts, rejected here
        # --- Invalid: all digits identical ---
        ("00000000000", 0, (0, 0), 0),  # passes the checksum, rejected as implausible
        ("11111111111", 0, (0, 0), 0),
        ("99999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("01234567890", 0, (0, 0), 0),
        ("98765432109", 0, (0, 0), 0),
        ("12345678901", 0, (0, 0), 0),
        ("87654321098", 0, (0, 0), 0),  # passes the checksum, rejected as sequential
        # --- Invalid: wrong lengths ---
        ("3680528010", 0, (0, 0), 0),  # 10 digits
        ("368052801099", 0, (0, 0), 0),  # 12 digits
        # --- Lookalikes: same shape, but not a personal code ---
        ("136805280109", 0, (0, 0), 0),  # a leading digit
        ("A36805280109", 0, (0, 0), 0),  # a leading letter
        ("36805280109A", 0, (0, 0), 0),  # a trailing letter
        ("3680 5280 109", 0, (0, 0), 0),  # spaced form, stdnum accepts, not matched
        ("3680-5280-109", 0, (0, 0), 0),  # dashed form, not matched
        # fmt: on
    ],
)
def test_when_isikukood_in_text_then_all_ids_found(
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
    results = recognizer.analyze("Isikukood: 36805280109 and 37605030299", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 11, 22, 0.5)
    assert_result(results[1], entities[0], 27, 38, 0.5)


sequential_test_set = [
    ["01234567890", True],
    ["98765432109", True],
    ["12345678901", True],
    ["87654321098", True],
    ["36805280109", False],
    ["37605030299", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert EeIsikukoodRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("3680528010") == "9"
    assert recognizer._check_digit("3760503029") == "9"
    assert recognizer._check_digit("4900202000") == "2"
    assert recognizer._check_digit("6000229000") == "3"
    assert recognizer._check_digit("5000101000") == "6"
    assert recognizer._check_digit("1010101000") == "5"
    assert recognizer._check_digit("1991231000") == "8"
    assert recognizer._check_digit("2991231000") == "9"
    assert recognizer._check_digit("2310325088") == "2"
    # both rounds leave remainder 10, so the check digit is 0
    assert recognizer._check_digit("2690901613") == "0"


def test_has_valid_checksum(recognizer):
    """The two round mod 11 check accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("36805280109") is True
    assert recognizer._has_valid_checksum("37605030299") is True
    assert recognizer._has_valid_checksum("49002020002") is True
    assert recognizer._has_valid_checksum("60002290003") is True
    assert recognizer._has_valid_checksum("50001010006") is True
    assert recognizer._has_valid_checksum("23103250882") is True
    assert recognizer._has_valid_checksum("26909016130") is True
    assert recognizer._has_valid_checksum("36805280108") is False
    assert recognizer._has_valid_checksum("37605030290") is False
    assert recognizer._has_valid_checksum("49002020003") is False
    assert recognizer._has_valid_checksum("60002290004") is False


def test_has_valid_birth_date(recognizer):
    """The birth date rule accepts real past dates and rejects the rest."""
    assert recognizer._has_valid_birth_date("36805280109") is True
    assert recognizer._has_valid_birth_date("37605030299") is True
    assert recognizer._has_valid_birth_date("49002020002") is True
    assert recognizer._has_valid_birth_date("60002290003") is True
    assert recognizer._has_valid_birth_date("19912310008") is True
    assert recognizer._has_valid_birth_date("33302300068") is False
    assert recognizer._has_valid_birth_date("40002290001") is False
    assert recognizer._has_valid_birth_date("33313310062") is False
    assert recognizer._has_valid_birth_date("33301000067") is False
    assert recognizer._has_valid_birth_date("33300010069") is False


def test_future_birth_date_is_rejected(recognizer, entities):
    """A number with tomorrow as its birth date is never detected."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    body = f"5{tomorrow.year % 100:02d}{tomorrow.month:02d}{tomorrow.day:02d}000"
    number = body + recognizer._check_digit(body)
    assert recognizer.analyze(number, entities) == []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("36805280109") is False
    assert recognizer.invalidate_result("37605030299") is False
    assert recognizer.invalidate_result("49002020002") is False
    assert recognizer.invalidate_result("60002290003") is False
    assert recognizer.invalidate_result("26909016130") is False
    assert recognizer.invalidate_result("36805280108") is True
    assert recognizer.invalidate_result("76805280102") is True
    assert recognizer.invalidate_result("86805280103") is True
    assert recognizer.invalidate_result("96805280104") is True
    assert recognizer.invalidate_result("06805280106") is True
    assert recognizer.invalidate_result("33302300068") is True
    assert recognizer.invalidate_result("40002290001") is True
    assert recognizer.invalidate_result("52701010009") is True
    assert recognizer.invalidate_result("00000000000") is True
    assert recognizer.invalidate_result("11111111111") is True
    assert recognizer.invalidate_result("99999999999") is True
    assert recognizer.invalidate_result("01234567890") is True
    assert recognizer.invalidate_result("87654321098") is True
    assert recognizer.invalidate_result("3680528010") is True
    assert recognizer.invalidate_result("368052801099") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = EeIsikukoodRecognizer(name="CustomEeIsikukood")
    assert recognizer.name == "CustomEeIsikukood"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(EeIsikukoodRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="36805280109", language="en", entities=["EE_IK"])
    with_context = analyzer.analyze(
        text="36805280109",
        language="en",
        entities=["EE_IK"],
        context=["isikukood"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(EeIsikukoodRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="isikukood 36805280108",
        language="en",
        entities=["EE_IK"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "EeIsikukoodRecognizer"]
    assert len(entries) == 1, "EeIsikukoodRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ee"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["EeIsikukoodRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "EE_IK" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My isikukood is 36805280109",
        language="en",
        entities=["EE_IK"],
    )
    assert [result.entity_type for result in results] == ["EE_IK"]


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
        entries = [r for r in recognizers if r.get("name") == "EeIsikukoodRecognizer"]
        assert len(entries) == 1, f"EeIsikukoodRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"EeIsikukoodRecognizer enabled in {yaml_path}"
        )
