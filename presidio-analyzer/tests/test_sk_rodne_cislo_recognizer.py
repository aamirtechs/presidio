"""Tests for Slovak birth number recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import SkRodneCisloRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Slovak birth number recognizer instance for testing."""
    return SkRodneCisloRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Slovak birth number entity type for testing."""
    return ["SK_RODNE_CISLO"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned divisible-by-11 rule, except for
# 9007203117, which is the @ppasmik/birth-number-utils
# worked example, and 7103192745, which is the
# python-stdnum sk.rc docstring example. No fixture is a
# real person's number beyond those published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid birth numbers ---
        # published examples, plain
        ("9007203117", 1, (0, 10), 0.5),  # @ppasmik example, man 1990-07-20
        ("7103192745", 1, (0, 10), 0.5),  # stdnum sk.rc docstring example
        # synthetic numbers, check digits computed with the div-11 rule
        ("7552121236", 1, (0, 10), 0.5),  # woman, 1975-02-12, month code 52
        ("0523154566", 1, (0, 10), 0.5),  # man, 2005-03-15, month code 23 (+20)
        ("1076081237", 1, (0, 10), 0.5),  # woman, 2010-06-08, month code 76 (+70)
        ("8511301238", 1, (0, 10), 0.5),  # man, 1985-11-30
        ("9057201230", 1, (0, 10), 0.5),  # woman, 1990-07-20, check digit 0
        ("8001011238", 1, (0, 10), 0.5),  # man, 1980-01-01
        ("0002291234", 1, (0, 10), 0.5),  # leap day, 29/02/2000
        ("9912311233", 1, (0, 10), 0.5),  # man, 1999-12-31
        # canonical slashed form
        ("900720/3117", 1, (0, 11), 0.5),  # @ppasmik example, slashed
        ("755212/1236", 1, (0, 11), 0.5),  # woman, slashed
        ("052315/4566", 1, (0, 11), 0.5),  # man, +20 form, slashed
        ("000229/1234", 1, (0, 11), 0.5),  # leap day, slashed
        ("710319/2745", 1, (0, 11), 0.5),  # stdnum example, slashed
        # a number embedded in a sentence
        ("rodne cislo 9007203117 on file", 1, (12, 22), 0.5),
        ("SK 9007203117", 1, (3, 13), 0.5),
        # a sentence period after the check digit does not block the match
        ("rodne cislo 9007203117.", 1, (12, 22), 0.5),
        ("rodne cislo 900720/3117.", 1, (12, 23), 0.5),
        # --- Invalid: wrong check digit ---
        ("9007203118", 0, (0, 0), 0),
        ("7552121230", 0, (0, 0), 0),
        ("0523154560", 0, (0, 0), 0),
        ("7103192746", 0, (0, 0), 0),  # stdnum docstring invalid example
        ("900720/3118", 0, (0, 0), 0),  # slashed form, wrong check digit
        # --- Invalid: remainder-10 historical form ---
        # the first 9 digits 900720006 are 10 mod 11, so no
        # digit makes the whole number divisible by 11;
        # historically a 0 check digit was written instead
        ("9007200060", 0, (0, 0), 0),
        # --- Invalid: 9 digit pre-1954 numbers ---
        # python-stdnum validates these as 1890 and 1899
        # numbers, but they have no checksum and are not
        # accepted here
        ("900720311", 0, (0, 0), 0),
        ("991231123", 0, (0, 0), 0),
        # --- Invalid: date in the future ---
        # checksum is valid, so only the future rule rejects this
        ("5301011232", 0, (0, 0), 0),  # 2053-01-01
        # --- Invalid: +20 / +70 codes before 2004 ---
        # checksums are valid and python-stdnum accepts both,
        # so only the post-2004 rule rejects these
        ("9923151238", 0, (0, 0), 0),  # man, 1999-03-15, month code 23
        ("9976081236", 0, (0, 0), 0),  # woman, 1999-06-08, month code 76
        # --- Invalid: not a real calendar date ---
        # checksums are valid, so only the date rule rejects these
        ("9013011238", 0, (0, 0), 0),  # month 13
        ("9001001240", 0, (0, 0), 0),  # day 00
        ("9001321230", 0, (0, 0), 0),  # day 32
        ("9002301231", 0, (0, 0), 0),  # February 30
        ("9902291234", 0, (0, 0), 0),  # February 29 in 1999, not a leap year
        ("9083011234", 0, (0, 0), 0),  # month code 83
        ("9063011232", 0, (0, 0), 0),  # month code 63
        ("9000151237", 0, (0, 0), 0),  # month code 00
        ("9033151237", 0, (0, 0), 0),  # month code 33
        ("9020151239", 0, (0, 0), 0),  # month code 20
        ("9050151231", 0, (0, 0), 0),  # month code 50
        ("9070151233", 0, (0, 0), 0),  # month code 70
        ("9052001233", 0, (0, 0), 0),  # day 00, woman
        ("1742196214", 0, (0, 0), 0),  # month code 42: python-stdnum folds
        # it to February with month = MM mod 50 mod 20, but no source
        # defines code 42, so it is rejected here
        # --- Invalid: serial part is 0000 ---
        # checksum is valid, so only the serial rule rejects this
        ("5411010000", 0, (0, 0), 0),  # 1954-11-01, tail 0000
        # --- Invalid: all digits identical ---
        ("1111111111", 0, (0, 0), 0),
        ("9999999999", 0, (0, 0), 0),
        ("0000000000", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("0123456789", 0, (0, 0), 0),
        ("1234567890", 0, (0, 0), 0),
        ("9876543210", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("90072031171", 0, (0, 0), 0),  # 11 digits
        # --- Lookalikes: same shape, but not a birth number ---
        ("19007203117", 0, (0, 0), 0),  # a leading digit
        ("A9007203117", 0, (0, 0), 0),  # a leading letter
        ("9007203117A", 0, (0, 0), 0),  # a trailing letter
        ("90/07203117", 0, (0, 0), 0),  # wrong slash grouping
        ("90072/03117", 0, (0, 0), 0),  # wrong slash grouping
        ("900720-3117", 0, (0, 0), 0),  # hyphen is not the separator
        ("900720 3117", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_birth_number_in_text_then_all_ids_found(
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
    """Both birth numbers in one text are found at their exact positions."""
    results = recognizer.analyze("rodne cisla: 9007203117 and 7552121236", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 13, 23, 0.5)
    assert_result(results[1], entities[0], 28, 38, 0.5)


sequential_test_set = [
    ["0123456789", True],
    ["9876543210", True],
    ["1234567890", True],
    ["9007203117", False],
    ["7552121236", False],
    ["0523154566", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the birth number digits."""
    assert SkRodneCisloRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("900720311") == "7"
    assert recognizer._check_digit("755212123") == "6"
    assert recognizer._check_digit("052315456") == "6"
    assert recognizer._check_digit("710319274") == "5"
    assert recognizer._check_digit("905720123") == "0"
    # no single digit fits this body: the first 9 digits are 10 mod 11
    assert recognizer._check_digit("900720006") == "10"


def test_has_valid_checksum(recognizer):
    """The divisibility-by-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("9007203117") is True
    assert recognizer._has_valid_checksum("7552121236") is True
    assert recognizer._has_valid_checksum("0523154566") is True
    assert recognizer._has_valid_checksum("7103192745") is True
    assert recognizer._has_valid_checksum("9057201230") is True
    assert recognizer._has_valid_checksum("9007203118") is False
    assert recognizer._has_valid_checksum("7552121230") is False
    assert recognizer._has_valid_checksum("0523154560") is False
    assert recognizer._has_valid_checksum("7103192746") is False
    # remainder-10 historical form: no check digit fits the body
    assert recognizer._has_valid_checksum("9007200060") is False
    assert recognizer._has_valid_checksum("9007200061") is False


def test_has_valid_date(recognizer):
    """The date rule accepts real dates and rejects the rest."""
    assert recognizer._has_valid_date("9007203117") is True
    assert recognizer._has_valid_date("0002291234") is True
    assert recognizer._has_valid_date("0523154566") is True
    assert recognizer._has_valid_date("1076081237") is True
    assert recognizer._has_valid_date("7552121236") is True
    assert recognizer._has_valid_date("9912311233") is True
    assert recognizer._has_valid_date("9013011238") is False
    assert recognizer._has_valid_date("9001001240") is False
    assert recognizer._has_valid_date("9001321230") is False
    assert recognizer._has_valid_date("9002301231") is False
    assert recognizer._has_valid_date("9902291234") is False
    assert recognizer._has_valid_date("9083011234") is False
    assert recognizer._has_valid_date("9063011232") is False
    assert recognizer._has_valid_date("9000151237") is False
    assert recognizer._has_valid_date("9033151237") is False
    assert recognizer._has_valid_date("9020151239") is False
    assert recognizer._has_valid_date("9050151231") is False
    assert recognizer._has_valid_date("9070151233") is False
    assert recognizer._has_valid_date("9052001233") is False
    assert recognizer._has_valid_date("1742196214") is False
    # date in the future
    assert recognizer._has_valid_date("5301011232") is False
    # +20 / +70 codes before 2004
    assert recognizer._has_valid_date("9923151238") is False
    assert recognizer._has_valid_date("9976081236") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("9007203117") is False
    assert recognizer.invalidate_result("7552121236") is False
    assert recognizer.invalidate_result("0523154566") is False
    assert recognizer.invalidate_result("7103192745") is False
    assert recognizer.invalidate_result("900720/3117") is False
    assert recognizer.invalidate_result("755212/1236") is False
    assert recognizer.invalidate_result("9007203118") is True
    assert recognizer.invalidate_result("900720/3118") is True
    assert recognizer.invalidate_result("9007200060") is True
    assert recognizer.invalidate_result("900720311") is True
    assert recognizer.invalidate_result("991231123") is True
    assert recognizer.invalidate_result("5301011232") is True
    assert recognizer.invalidate_result("9923151238") is True
    assert recognizer.invalidate_result("9976081236") is True
    assert recognizer.invalidate_result("9013011238") is True
    assert recognizer.invalidate_result("9001001240") is True
    assert recognizer.invalidate_result("9001321230") is True
    assert recognizer.invalidate_result("9002301231") is True
    assert recognizer.invalidate_result("9902291234") is True
    assert recognizer.invalidate_result("9083011234") is True
    assert recognizer.invalidate_result("1742196214") is True
    assert recognizer.invalidate_result("5411010000") is True
    assert recognizer.invalidate_result("1111111111") is True
    assert recognizer.invalidate_result("9999999999") is True
    assert recognizer.invalidate_result("0123456789") is True
    assert recognizer.invalidate_result("90072031171") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = SkRodneCisloRecognizer(name="CustomSkRodneCislo")
    assert recognizer.name == "CustomSkRodneCislo"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a birth number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(SkRodneCisloRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="9007203117", language="en", entities=["SK_RODNE_CISLO"]
    )
    with_context = analyzer.analyze(
        text="9007203117",
        language="en",
        entities=["SK_RODNE_CISLO"],
        context=["rodne cislo"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(SkRodneCisloRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="rodne cislo 9007203118",
        language="en",
        entities=["SK_RODNE_CISLO"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "SkRodneCisloRecognizer"]
    assert len(entries) == 1, "SkRodneCisloRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "sk"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == [
        "SkRodneCisloRecognizer"
    ]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "SK_RODNE_CISLO" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My rodne cislo is 9007203117",
        language="en",
        entities=["SK_RODNE_CISLO"],
    )
    assert [result.entity_type for result in results] == ["SK_RODNE_CISLO"]


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
        entries = [r for r in recognizers if r.get("name") == "SkRodneCisloRecognizer"]
        assert len(entries) == 1, f"SkRodneCisloRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"SkRodneCisloRecognizer enabled in {yaml_path}"
        )
