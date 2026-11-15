"""Tests for Romanian personal numeric code recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import RoCnpRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Romanian personal numeric code recognizer instance for testing."""
    return RoCnpRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Romanian personal numeric code entity type for testing."""
    return ["RO_CNP"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned 279146358279 weighted mod-11
# rule, except for 1960523123457, the worked example
# from the CNP validator write-ups, 1630615123457,
# the python-stdnum docstring example, and
# 1940616346114, 7041218318525, 6201206018078 and
# 5201206346491, the rocnp package examples. No fixture
# is a real person's number beyond those published
# test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid personal numeric codes ---
        # published examples, plain
        ("1960523123457", 1, (0, 13), 0.5),  # worked example, man 1996-05-23, Cluj
        ("1630615123457", 1, (0, 13), 0.5),  # stdnum docstring, man 1963-06-15, Cluj
        ("1940616346114", 1, (0, 13), 0.5),  # rocnp example, man 1994-06-16, Teleorman
        ("7041218318525", 1, (0, 13), 0.5),  # rocnp example, male resident, Salaj
        ("6201206018078", 1, (0, 13), 0.5),  # rocnp example, woman 2020-12-06, Alba
        ("5201206346491", 1, (0, 13), 0.5),  # rocnp example, remainder-10 fold, check 1
        # synthetic numbers, check digits computed with the pinned rule
        ("2851130401234", 1, (0, 13), 0.5),  # woman, 1985-11-30, Bucharest 40
        ("5000229700012", 1, (0, 13), 0.5),  # man, leap day 29/02/2000, county 70
        ("3990115024562", 1, (0, 13), 0.5),  # man, 1899-01-15, Arad 02, first digit 3
        ("8900704237897", 1, (0, 13), 0.5),  # female resident, 1990-07-04, Ilfov 23
        ("7000229121234", 1, (0, 13), 0.5),  # male resident, leap day 29/02/2000
        ("1791219471237", 1, (0, 13), 0.5),  # man, 1979-12-19, county 47, last day
        ("1900101120080", 1, (0, 13), 0.5),  # man, 1990-01-01, check digit 0
        ("1900101120013", 1, (0, 13), 0.5),  # man, 1990-01-01, Cluj 12
        ("2750630059991", 1, (0, 13), 0.5),  # woman, 1975-06-30, Bihor 05
        # a number embedded in a sentence
        ("CNP 1960523123457 on file", 1, (4, 17), 0.5),
        ("RO 1630615123457", 1, (3, 16), 0.5),
        # a sentence period after the check digit does not block the match
        ("CNP 1960523123457.", 1, (4, 17), 0.5),
        # --- Invalid: wrong check digit ---
        ("1960523123458", 0, (0, 0), 0),
        ("1630615123458", 0, (0, 0), 0),  # stdnum docstring invalid example
        ("5201206346490", 0, (0, 0), 0),  # remainder-10 fold, wrong check digit
        ("2851130401230", 0, (0, 0), 0),
        # --- Invalid: first digit 0 or 9 ---
        # 9 is a python-stdnum-only foreigner form, not in the specification
        ("0960523123455", 0, (0, 0), 0),  # first digit 0
        ("9960523123451", 0, (0, 0), 0),  # first digit 9, stdnum accepts this
        # --- Invalid: date in the future ---
        # checksum is valid, so only the future rule rejects this
        ("5990101121235", 0, (0, 0), 0),  # 2099-01-01
        # --- Invalid: not a real calendar date ---
        # checksums are valid, so only the date rule rejects these
        ("1961323123450", 0, (0, 0), 0),  # month 13
        ("1960500123458", 0, (0, 0), 0),  # day 00
        ("1960532123451", 0, (0, 0), 0),  # day 32
        ("1960230123453", 0, (0, 0), 0),  # February 30
        ("1990229123457", 0, (0, 0), 0),  # February 29 in 1999, not a leap year
        # --- Invalid: county code not in the published table ---
        # checksums are valid, so only the county rule rejects these
        ("1960523001235", 0, (0, 0), 0),  # county 00
        ("1960523491239", 0, (0, 0), 0),  # county 49, no such county
        ("1960523501238", 0, (0, 0), 0),  # county 50, no such county
        ("1960523531231", 0, (0, 0), 0),  # county 53, no such county
        ("1960523801231", 0, (0, 0), 0),  # county 80, stdnum-only Unknown
        # --- Invalid: Bucharest sectors 7 and 8 after abolition ---
        # checksums are valid and python-stdnum accepts both,
        # so only the 1979-12-19 rule rejects these
        ("1850310471239", 0, (0, 0), 0),  # county 47, 1985-03-10
        ("1850310481236", 0, (0, 0), 0),  # county 48, 1985-03-10
        # --- Invalid: serial part is 000 ---
        # checksum is valid and python-stdnum accepts this,
        # so only the serial rule rejects it
        ("1960523120005", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        ("1111111111111", 0, (0, 0), 0),
        ("9999999999999", 0, (0, 0), 0),
        ("0000000000000", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("0123456789012", 0, (0, 0), 0),
        ("1234567890123", 0, (0, 0), 0),
        ("9876543210987", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("196052312345", 0, (0, 0), 0),  # 12 digits
        ("19605231234571", 0, (0, 0), 0),  # 14 digits
        # --- Lookalikes: same shape, but not a personal numeric code ---
        ("11960523123457", 0, (0, 0), 0),  # a leading digit
        ("A1960523123457", 0, (0, 0), 0),  # a leading letter
        ("1960523123457A", 0, (0, 0), 0),  # a trailing letter
        ("1960523-123457", 0, (0, 0), 0),  # hyphenated form is not accepted
        ("1960523 123457", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_cnp_in_text_then_all_ids_found(
    text, expected_len, expected_position, expected_score, recognizer, entities
):
    """Test that the recognizer finds valid codes and rejects invalid ones."""
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
    """Both personal numeric codes in one text are found at their exact positions."""
    results = recognizer.analyze("CNPs: 1960523123457 and 1630615123457", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 19, 0.5)
    assert_result(results[1], entities[0], 24, 37, 0.5)


sequential_test_set = [
    ["0123456789012", True],
    ["9876543210987", True],
    ["1234567890123", True],
    ["1960523123457", False],
    ["1630615123457", False],
    ["5201206346491", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the code digits."""
    assert RoCnpRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("196052312345") == "7"
    assert recognizer._check_digit("163061512345") == "7"
    assert recognizer._check_digit("194061634611") == "4"
    assert recognizer._check_digit("704121831852") == "5"
    assert recognizer._check_digit("620120601807") == "8"
    # remainder 10 folds to a check digit of 1
    assert recognizer._check_digit("520120634649") == "1"
    assert recognizer._check_digit("190010112008") == "0"


def test_has_valid_checksum(recognizer):
    """The weighted mod-11 checksum accepts valid codes and rejects the rest."""
    assert recognizer._has_valid_checksum("1960523123457") is True
    assert recognizer._has_valid_checksum("1630615123457") is True
    assert recognizer._has_valid_checksum("1940616346114") is True
    assert recognizer._has_valid_checksum("7041218318525") is True
    assert recognizer._has_valid_checksum("6201206018078") is True
    assert recognizer._has_valid_checksum("5201206346491") is True
    assert recognizer._has_valid_checksum("1900101120080") is True
    assert recognizer._has_valid_checksum("1960523123458") is False
    assert recognizer._has_valid_checksum("1630615123458") is False
    assert recognizer._has_valid_checksum("5201206346490") is False
    assert recognizer._has_valid_checksum("2851130401230") is False


def test_has_valid_date(recognizer):
    """The date rule accepts real dates and rejects the rest."""
    assert recognizer._has_valid_date("1960523123457") is True
    assert recognizer._has_valid_date("5000229700012") is True
    assert recognizer._has_valid_date("3990115024562") is True
    assert recognizer._has_valid_date("8900704237897") is True
    # a resident code for 29 February 2000: 1900 was not a leap year,
    # 2000 was, and the century is not encoded for first digits 7 and 8
    assert recognizer._has_valid_date("7000229121234") is True
    assert recognizer._has_valid_date("1961323123450") is False
    assert recognizer._has_valid_date("1960500123458") is False
    assert recognizer._has_valid_date("1960532123451") is False
    assert recognizer._has_valid_date("1960230123453") is False
    assert recognizer._has_valid_date("1990229123457") is False
    # date in the future
    assert recognizer._has_valid_date("5990101121235") is False


def test_has_valid_county(recognizer):
    """The county rule accepts published codes and rejects the rest."""
    assert recognizer._has_valid_county("1960523123457") is True
    assert recognizer._has_valid_county("5000229700012") is True
    assert recognizer._has_valid_county("1960523001235") is False
    assert recognizer._has_valid_county("1960523491239") is False
    assert recognizer._has_valid_county("1960523501238") is False
    assert recognizer._has_valid_county("1960523531231") is False
    assert recognizer._has_valid_county("1960523801231") is False
    # county 47 on the last day before abolition is accepted
    assert recognizer._has_valid_county("1791219471237") is True
    # county 47 and 48 after 19 December 1979 are rejected
    assert recognizer._has_valid_county("1850310471239") is False
    assert recognizer._has_valid_county("1850310481236") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("1960523123457") is False
    assert recognizer.invalidate_result("1630615123457") is False
    assert recognizer.invalidate_result("1940616346114") is False
    assert recognizer.invalidate_result("7041218318525") is False
    assert recognizer.invalidate_result("6201206018078") is False
    assert recognizer.invalidate_result("5201206346491") is False
    assert recognizer.invalidate_result("5000229700012") is False
    assert recognizer.invalidate_result("7000229121234") is False
    assert recognizer.invalidate_result("1960523123458") is True
    assert recognizer.invalidate_result("1630615123458") is True
    assert recognizer.invalidate_result("0960523123455") is True
    assert recognizer.invalidate_result("9960523123451") is True
    assert recognizer.invalidate_result("5990101121235") is True
    assert recognizer.invalidate_result("1961323123450") is True
    assert recognizer.invalidate_result("1960500123458") is True
    assert recognizer.invalidate_result("1960532123451") is True
    assert recognizer.invalidate_result("1960230123453") is True
    assert recognizer.invalidate_result("1990229123457") is True
    assert recognizer.invalidate_result("1960523001235") is True
    assert recognizer.invalidate_result("1960523491239") is True
    assert recognizer.invalidate_result("1960523801231") is True
    assert recognizer.invalidate_result("1850310471239") is True
    assert recognizer.invalidate_result("1850310481236") is True
    assert recognizer.invalidate_result("1960523120005") is True
    assert recognizer.invalidate_result("1111111111111") is True
    assert recognizer.invalidate_result("9999999999999") is True
    assert recognizer.invalidate_result("0123456789012") is True
    assert recognizer.invalidate_result("196052312345") is True
    assert recognizer.invalidate_result("19605231234571") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = RoCnpRecognizer(name="CustomRoCnp")
    assert recognizer.name == "CustomRoCnp"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a code raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(RoCnpRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="1960523123457", language="en", entities=["RO_CNP"])
    with_context = analyzer.analyze(
        text="1960523123457",
        language="en",
        entities=["RO_CNP"],
        context=["cod numeric personal"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a code the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(RoCnpRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="cod numeric personal 1960523123458",
        language="en",
        entities=["RO_CNP"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "RoCnpRecognizer"]
    assert len(entries) == 1, "RoCnpRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "ro"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["RoCnpRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "RO_CNP" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My CNP is 1960523123457",
        language="en",
        entities=["RO_CNP"],
    )
    assert [result.entity_type for result in results] == ["RO_CNP"]


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
        entries = [r for r in recognizers if r.get("name") == "RoCnpRecognizer"]
        assert len(entries) == 1, f"RoCnpRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"RoCnpRecognizer enabled in {yaml_path}"
