"""Tests for Dutch citizen service number recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import NlBsnRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Dutch citizen service number recognizer instance for testing."""
    return NlBsnRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Dutch citizen service number entity type for testing."""
    return ["NL_BSN"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned eleven test, except for
# 111222333, the python-stdnum docstring example,
# 123456782, the widely published example, and
# 211551557, the worked example from the Dutch
# government eleven-test explanation. No fixture
# is a real person's number beyond those published
# test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid citizen service numbers ---
        # published examples, plain
        ("111222333", 1, (0, 9), 0.5),  # stdnum docstring, sum 66
        ("123456782", 1, (0, 9), 0.5),  # published example, sum 154
        ("211551557", 1, (0, 9), 0.5),  # government worked example, sum 110
        ("010000008", 1, (0, 9), 0.5),  # leading zero, sum 0
        ("999000007", 1, (0, 9), 0.5),  # 999 test range, sum 209
        # synthetic numbers, check digits computed with the eleven test
        ("972997520", 1, (0, 9), 0.5),  # check digit 0, sum 297
        ("646288970", 1, (0, 9), 0.5),  # check digit 0, sum 253
        ("908301662", 1, (0, 9), 0.5),  # sum 187
        ("131860914", 1, (0, 9), 0.5),  # sum 143
        ("390996038", 1, (0, 9), 0.5),  # sum 220
        ("082462811", 1, (0, 9), 0.5),  # leading zero, sum 165
        ("948219932", 1, (0, 9), 0.5),  # sum 253
        # a number embedded in a sentence
        ("BSN 111222333 on file", 1, (4, 13), 0.5),
        ("NL 123456782", 1, (3, 12), 0.5),
        # a sentence period after the check digit does not block the match
        ("BSN 111222333.", 1, (4, 13), 0.5),
        # --- Invalid: wrong check digit ---
        ("111222334", 0, (0, 0), 0),
        ("123456783", 0, (0, 0), 0),
        ("211551558", 0, (0, 0), 0),
        ("010000009", 0, (0, 0), 0),
        ("999000008", 0, (0, 0), 0),
        # --- Invalid: no check digit exists for this body ---
        # the first 8 digits have a weighted sum of 10 mod 11,
        # so every 9th digit fails the eleven test
        ("100000060", 0, (0, 0), 0),
        ("100000065", 0, (0, 0), 0),
        # --- Invalid: all digits identical ---
        # 000000000 passes the eleven test (sum 0) but is not a real BSN
        ("111111111", 0, (0, 0), 0),
        ("000000000", 0, (0, 0), 0),
        ("999999999", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("012345678", 0, (0, 0), 0),
        ("123456789", 0, (0, 0), 0),
        ("987654321", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("11122233", 0, (0, 0), 0),  # 8 digits
        ("1112223334", 0, (0, 0), 0),  # 10 digits
        # --- Lookalikes: same shape, but not a citizen service number ---
        ("1111222333", 0, (0, 0), 0),  # a leading digit
        ("A111222333", 0, (0, 0), 0),  # a leading letter
        ("111222333A", 0, (0, 0), 0),  # a trailing letter
        ("1112.22.333", 0, (0, 0), 0),  # dotted form, stdnum accepts, not matched
        ("111 222 333", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_bsn_in_text_then_all_ids_found(
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
    """Both citizen service numbers in one text are found at their exact positions."""
    results = recognizer.analyze("BSNs: 111222333 and 123456782", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 15, 0.5)
    assert_result(results[1], entities[0], 20, 29, 0.5)


sequential_test_set = [
    ["012345678", True],
    ["987654321", True],
    ["123456789", True],
    ["111222333", False],
    ["123456782", False],
    ["211551557", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert NlBsnRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("11122233") == "3"
    assert recognizer._check_digit("12345678") == "2"
    assert recognizer._check_digit("21155155") == "7"
    assert recognizer._check_digit("01000000") == "8"
    assert recognizer._check_digit("99900000") == "7"
    # check digit 0 cases
    assert recognizer._check_digit("97299752") == "0"
    assert recognizer._check_digit("64628897") == "0"
    # no check digit exists when the weighted sum is 10 mod 11
    assert recognizer._check_digit("10000006") is None


def test_has_valid_checksum(recognizer):
    """The eleven test accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("111222333") is True
    assert recognizer._has_valid_checksum("123456782") is True
    assert recognizer._has_valid_checksum("211551557") is True
    assert recognizer._has_valid_checksum("010000008") is True
    assert recognizer._has_valid_checksum("999000007") is True
    assert recognizer._has_valid_checksum("972997520") is True
    assert recognizer._has_valid_checksum("646288970") is True
    assert recognizer._has_valid_checksum("111222334") is False
    assert recognizer._has_valid_checksum("123456783") is False
    assert recognizer._has_valid_checksum("211551558") is False
    assert recognizer._has_valid_checksum("010000009") is False
    assert recognizer._has_valid_checksum("999000008") is False
    assert recognizer._has_valid_checksum("100000065") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("111222333") is False
    assert recognizer.invalidate_result("123456782") is False
    assert recognizer.invalidate_result("211551557") is False
    assert recognizer.invalidate_result("010000008") is False
    assert recognizer.invalidate_result("999000007") is False
    assert recognizer.invalidate_result("972997520") is False
    assert recognizer.invalidate_result("111222334") is True
    assert recognizer.invalidate_result("123456783") is True
    assert recognizer.invalidate_result("100000065") is True
    assert recognizer.invalidate_result("111111111") is True
    assert recognizer.invalidate_result("000000000") is True
    assert recognizer.invalidate_result("999999999") is True
    assert recognizer.invalidate_result("012345678") is True
    assert recognizer.invalidate_result("123456789") is True
    assert recognizer.invalidate_result("987654321") is True
    assert recognizer.invalidate_result("11122233") is True
    assert recognizer.invalidate_result("1112223334") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = NlBsnRecognizer(name="CustomNlBsn")
    assert recognizer.name == "CustomNlBsn"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(NlBsnRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="111222333", language="en", entities=["NL_BSN"])
    with_context = analyzer.analyze(
        text="111222333",
        language="en",
        entities=["NL_BSN"],
        context=["burgerservicenummer"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(NlBsnRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="burgerservicenummer 111222334",
        language="en",
        entities=["NL_BSN"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "NlBsnRecognizer"]
    assert len(entries) == 1, "NlBsnRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "nl"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["NlBsnRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "NL_BSN" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My BSN is 111222333",
        language="en",
        entities=["NL_BSN"],
    )
    assert [result.entity_type for result in results] == ["NL_BSN"]


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
        entries = [r for r in recognizers if r.get("name") == "NlBsnRecognizer"]
        assert len(entries) == 1, f"NlBsnRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"NlBsnRecognizer enabled in {yaml_path}"
