"""Tests for Iceland kennitala recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import IsKennitalaRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Iceland kennitala recognizer instance for testing."""
    return IsKennitalaRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Iceland kennitala entity type for testing."""
    return ["IS_KENNITALA"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned mod-11 algorithm, except for
# 0902862349, 1201743399, and 4504013150, which are
# published examples: the first is the Business::IS::PIN
# worked example, the second is the Wikipedia mockup
# example and the python-stdnum docstring example, and
# the third is the python-stdnum docstring organisation
# example. No fixture is a real person's number beyond
# those published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid kennitala numbers ---
        # published examples, plain
        ("0902862349", 1, (0, 10), 0.5),  # Business::IS::PIN worked example
        ("1201743399", 1, (0, 10), 0.5),  # Wikipedia mockup, stdnum docstring
        ("4504013150", 1, (0, 10), 0.5),  # stdnum docstring, organisation
        # synthetic numbers, check digits computed with the mod-11 algorithm
        ("1505102280", 1, (0, 10), 0.5),  # 2000s individual, 15/05/10
        ("0101804579", 1, (0, 10), 0.5),  # 01/01/80
        ("3112997789", 1, (0, 10), 0.5),  # 31/12/99
        ("2902003340", 1, (0, 10), 0.5),  # leap day, 29/02/2000
        ("4101743329", 1, (0, 10), 0.5),  # entity, day 41 (01 + 40)
        ("7101743349", 1, (0, 10), 0.5),  # entity, day 71 (31 + 40)
        ("0506952049", 1, (0, 10), 0.5),  # random digits 20
        ("0506959999", 1, (0, 10), 0.5),  # random digits 99
        ("1505100900", 1, (0, 10), 0.5),  # check digit 0
        # canonical hyphenated form
        ("090286-2349", 1, (0, 11), 0.5),  # Business::IS::PIN example, hyphenated
        ("120174-3399", 1, (0, 11), 0.5),  # Wikipedia mockup, hyphenated
        ("450401-3150", 1, (0, 11), 0.5),  # stdnum organisation, hyphenated
        ("150510-2280", 1, (0, 11), 0.5),  # 2000s individual, hyphenated
        ("290200-3340", 1, (0, 11), 0.5),  # leap day, hyphenated
        # a number embedded in a sentence
        ("kennitala 1201743399 on file", 1, (10, 20), 0.5),
        ("IS 1201743399", 1, (3, 13), 0.5),
        # a sentence period after the check digit does not block the match
        ("kennitala 1201743399.", 1, (10, 20), 0.5),
        ("kennitala 120174-3399.", 1, (10, 21), 0.5),
        # --- Invalid: wrong check digit ---
        ("0902862359", 0, (0, 0), 0),
        ("1201743389", 0, (0, 0), 0),
        ("4504013160", 0, (0, 0), 0),
        ("120174-3389", 0, (0, 0), 0),  # hyphenated form, wrong check digit
        # --- Invalid: no check digit fits this body ---
        # the body 15051004 has weighted sum 56, 56 mod 11 = 1,
        # so the formula yields 10, which is not a single digit
        ("1505100400", 0, (0, 0), 0),
        ("1505100410", 0, (0, 0), 0),
        ("1505100490", 0, (0, 0), 0),
        # --- Invalid: century digit is not 9 or 0 ---
        # checksums are valid, so only the century rule rejects these
        ("1201743391", 0, (0, 0), 0),
        ("1201743398", 0, (0, 0), 0),
        # --- Invalid: not a real calendar date ---
        # checksums are valid, so only the date rule rejects these
        ("1213743319", 0, (0, 0), 0),  # month 13
        ("0012743339", 0, (0, 0), 0),  # day 00
        ("3201743339", 0, (0, 0), 0),  # day 32
        ("3002743319", 0, (0, 0), 0),  # February 30
        ("2902993309", 0, (0, 0), 0),  # February 29 in 1999, not a leap year
        ("4001743349", 0, (0, 0), 0),  # entity day 40
        ("7201743329", 0, (0, 0), 0),  # entity day 72
        # --- Invalid: all digits identical ---
        ("1111111111", 0, (0, 0), 0),
        ("9999999999", 0, (0, 0), 0),
        ("0000000000", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("0123456789", 0, (0, 0), 0),
        ("1234567890", 0, (0, 0), 0),
        ("9876543210", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("120174339", 0, (0, 0), 0),  # 9 digits
        ("12017433991", 0, (0, 0), 0),  # 11 digits
        # --- Lookalikes: same shape, but not a kennitala number ---
        ("11201743399", 0, (0, 0), 0),  # a leading digit
        ("A1201743399", 0, (0, 0), 0),  # a leading letter
        ("1201743399A", 0, (0, 0), 0),  # a trailing letter
        ("12-017433-99", 0, (0, 0), 0),  # wrong hyphen grouping
        ("12017-43399", 0, (0, 0), 0),  # wrong hyphen grouping
        ("12 0174 3399", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_kennitala_in_text_then_all_ids_found(
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
    """Both kennitala numbers in one text are found at their exact positions."""
    results = recognizer.analyze("kennitalas: 0902862349 and 1201743399", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 12, 22, 0.5)
    assert_result(results[1], entities[0], 27, 37, 0.5)


sequential_test_set = [
    ["0123456789", True],
    ["9876543210", True],
    ["1234567890", True],
    ["1201743399", False],
    ["0902862349", False],
    ["4504013150", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the kennitala digits."""
    assert IsKennitalaRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("09028623") == "4"
    assert recognizer._check_digit("12017433") == "9"
    assert recognizer._check_digit("45040131") == "5"
    assert recognizer._check_digit("15051022") == "8"
    assert recognizer._check_digit("15051009") == "0"
    # no single digit fits this body: the formula yields "10"
    assert recognizer._check_digit("15051004") == "10"


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("0902862349") is True
    assert recognizer._has_valid_checksum("1201743399") is True
    assert recognizer._has_valid_checksum("4504013150") is True
    assert recognizer._has_valid_checksum("1505102280") is True
    assert recognizer._has_valid_checksum("1505100900") is True
    assert recognizer._has_valid_checksum("0902862359") is False
    assert recognizer._has_valid_checksum("1201743389") is False
    assert recognizer._has_valid_checksum("4504013160") is False
    # no check digit fits the body 15051004
    assert recognizer._has_valid_checksum("1505100400") is False
    assert recognizer._has_valid_checksum("1505100410") is False
    assert recognizer._has_valid_checksum("1505100490") is False


def test_has_valid_date(recognizer):
    """The date rule accepts real dates and rejects the rest."""
    assert recognizer._has_valid_date("1201743399") is True
    assert recognizer._has_valid_date("2902003340") is True
    assert recognizer._has_valid_date("4101743329") is True
    assert recognizer._has_valid_date("7101743349") is True
    assert recognizer._has_valid_date("1213743319") is False
    assert recognizer._has_valid_date("0012743339") is False
    assert recognizer._has_valid_date("3201743339") is False
    assert recognizer._has_valid_date("3002743319") is False
    assert recognizer._has_valid_date("2902993309") is False
    assert recognizer._has_valid_date("4001743349") is False
    assert recognizer._has_valid_date("7201743329") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("0902862349") is False
    assert recognizer.invalidate_result("1201743399") is False
    assert recognizer.invalidate_result("4504013150") is False
    assert recognizer.invalidate_result("1505102280") is False
    assert recognizer.invalidate_result("090286-2349") is False
    assert recognizer.invalidate_result("120174-3399") is False
    assert recognizer.invalidate_result("0902862359") is True
    assert recognizer.invalidate_result("120174-3389") is True
    assert recognizer.invalidate_result("1201743391") is True
    assert recognizer.invalidate_result("1201743398") is True
    assert recognizer.invalidate_result("1213743319") is True
    assert recognizer.invalidate_result("0012743339") is True
    assert recognizer.invalidate_result("3201743339") is True
    assert recognizer.invalidate_result("3002743319") is True
    assert recognizer.invalidate_result("2902993309") is True
    assert recognizer.invalidate_result("4001743349") is True
    assert recognizer.invalidate_result("7201743329") is True
    assert recognizer.invalidate_result("1111111111") is True
    assert recognizer.invalidate_result("9999999999") is True
    assert recognizer.invalidate_result("0123456789") is True
    assert recognizer.invalidate_result("120174339") is True
    assert recognizer.invalidate_result("12017433991") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = IsKennitalaRecognizer(name="CustomIsKennitala")
    assert recognizer.name == "CustomIsKennitala"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a kennitala number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(IsKennitalaRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="1201743399", language="en", entities=["IS_KENNITALA"]
    )
    with_context = analyzer.analyze(
        text="1201743399",
        language="en",
        entities=["IS_KENNITALA"],
        context=["kennitala"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(IsKennitalaRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="kennitala 1201743389",
        language="en",
        entities=["IS_KENNITALA"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "IsKennitalaRecognizer"]
    assert len(entries) == 1, "IsKennitalaRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "is"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["IsKennitalaRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "IS_KENNITALA" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My kennitala is 1201743399",
        language="en",
        entities=["IS_KENNITALA"],
    )
    assert [result.entity_type for result in results] == ["IS_KENNITALA"]


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
        entries = [r for r in recognizers if r.get("name") == "IsKennitalaRecognizer"]
        assert len(entries) == 1, f"IsKennitalaRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, (
            f"IsKennitalaRecognizer enabled in {yaml_path}"
        )
