"""Tests for Uruguay Registro Unico Tributario (RUT) recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import UyRutRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Uruguay RUT recognizer instance for testing."""
    return UyRutRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Uruguay RUT entity type for testing."""
    return ["UY_RUT"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned mod-11 algorithm, except for
# 211003420017, 211406340011, and 214111700011, which are
# published examples: the first two are the python-stdnum
# docstring examples, and the third is the real RUT of
# Deremate.com de Uruguay S.R.L. cited in URSEC
# Resolution 008/2024. No fixture is a real person's
# number beyond those published test vectors.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid RUT numbers ---
        # published examples, plain
        ("211003420017", 1, (0, 12), 0.5),  # python-stdnum docstring example
        ("211406340011", 1, (0, 12), 0.5),  # python-stdnum docstring example
        ("214111700011", 1, (0, 12), 0.5),  # URSEC Resolution 008/2024 example
        # synthetic numbers, check digits computed with the mod-11 algorithm
        ("220000010011", 1, (0, 12), 0.5),  # 22 series, new format
        ("221234560016", 1, (0, 12), 0.5),  # 22 series
        ("011234560016", 1, (0, 12), 0.5),  # 01 prefix
        ("210000010014", 1, (0, 12), 0.5),
        ("220987650014", 1, (0, 12), 0.5),
        ("219999990016", 1, (0, 12), 0.5),
        # canonical hyphenated form
        ("21-100342-001-7", 1, (0, 15), 0.5),  # python-stdnum example, hyphenated
        ("21-140634-001-1", 1, (0, 15), 0.5),  # python-stdnum example, hyphenated
        ("21-411170-001-1", 1, (0, 15), 0.5),  # URSEC example, hyphenated
        ("22-000001-001-1", 1, (0, 15), 0.5),  # 22 series, hyphenated
        ("22-123456-001-6", 1, (0, 15), 0.5),  # 22 series, hyphenated
        # a number embedded in a sentence
        ("RUT 211003420017 on file", 1, (4, 16), 0.5),
        ("UY 211003420017", 1, (3, 15), 0.5),
        # a sentence period after the check digit does not block the match
        ("RUT 211003420017.", 1, (4, 16), 0.5),
        ("RUT 21-100342-001-7.", 1, (4, 19), 0.5),
        # --- Invalid: wrong check digit ---
        ("211003420018", 0, (0, 0), 0),
        ("211406340012", 0, (0, 0), 0),
        ("214111700012", 0, (0, 0), 0),
        ("21-100342-001-8", 0, (0, 0), 0),  # hyphenated form, wrong check digit
        # --- Invalid: no check digit fits this body ---
        # the body 21100006001 has weighted sum 45, 45 mod 11 = 1,
        # so the formula yields 10, which is not a single digit
        ("211000060010", 0, (0, 0), 0),
        ("211000060011", 0, (0, 0), 0),
        ("211000060019", 0, (0, 0), 0),
        # --- Invalid: registration number outside 01 to 22 ---
        # checksums are valid, so only the prefix rule rejects these
        ("231003420011", 0, (0, 0), 0),
        ("001003420017", 0, (0, 0), 0),
        # --- Invalid: sequence number 000000 ---
        # checksum is valid, so only the sequence rule rejects this
        ("210000000019", 0, (0, 0), 0),
        # --- Invalid: digits nine to eleven are not the fixed block 001 ---
        # checksums are valid, so only the block rule rejects these
        ("211003420025", 0, (0, 0), 0),  # block 002
        ("211003420009", 0, (0, 0), 0),  # block 000
        # --- Invalid: all digits identical ---
        ("111111111111", 0, (0, 0), 0),
        ("222222222222", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        ("012345678901", 0, (0, 0), 0),
        ("123456789012", 0, (0, 0), 0),
        ("987654321098", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("21100342001", 0, (0, 0), 0),  # 11 digits, no check digit
        ("2110034200171", 0, (0, 0), 0),  # 13 digits
        # --- Lookalikes: same shape, but not a RUT number ---
        ("1211003420017", 0, (0, 0), 0),  # a leading digit
        ("211-003420-001-7", 0, (0, 0), 0),  # wrong hyphen grouping
        ("21-10034-2001-7", 0, (0, 0), 0),  # wrong hyphen grouping
        ("A211003420017", 0, (0, 0), 0),  # a leading letter
        ("211003420017A", 0, (0, 0), 0),  # a trailing letter
        ("21 100342 001 7", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_rut_in_text_then_all_ids_found(
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
    """Both RUT numbers in one text are found at their exact positions."""
    results = recognizer.analyze("RUTs: 211003420017 and 214111700011", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 18, 0.5)
    assert_result(results[1], entities[0], 23, 35, 0.5)


sequential_test_set = [
    ["012345678901", True],
    ["987654321098", True],
    ["123456789012", True],
    ["211003420017", False],
    ["214111700011", False],
    ["220000010011", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the RUT digits."""
    assert UyRutRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("21100342001") == "7"
    assert recognizer._check_digit("21140634001") == "1"
    assert recognizer._check_digit("21411170001") == "1"
    assert recognizer._check_digit("22000001001") == "1"
    assert recognizer._check_digit("22123456001") == "6"
    # no single digit fits this body: the formula yields "10"
    assert recognizer._check_digit("21100006001") == "10"


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("211003420017") is True
    assert recognizer._has_valid_checksum("211406340011") is True
    assert recognizer._has_valid_checksum("214111700011") is True
    assert recognizer._has_valid_checksum("220000010011") is True
    assert recognizer._has_valid_checksum("211003420018") is False
    assert recognizer._has_valid_checksum("211406340012") is False
    assert recognizer._has_valid_checksum("214111700012") is False
    # no check digit fits the body 21100006001
    assert recognizer._has_valid_checksum("211000060010") is False
    assert recognizer._has_valid_checksum("211000060011") is False
    assert recognizer._has_valid_checksum("211000060019") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("211003420017") is False
    assert recognizer.invalidate_result("211406340011") is False
    assert recognizer.invalidate_result("214111700011") is False
    assert recognizer.invalidate_result("220000010011") is False
    assert recognizer.invalidate_result("21-100342-001-7") is False
    assert recognizer.invalidate_result("21-411170-001-1") is False
    assert recognizer.invalidate_result("211003420018") is True
    assert recognizer.invalidate_result("21-100342-001-8") is True
    assert recognizer.invalidate_result("231003420011") is True
    assert recognizer.invalidate_result("001003420017") is True
    assert recognizer.invalidate_result("210000000019") is True
    assert recognizer.invalidate_result("211003420025") is True
    assert recognizer.invalidate_result("211003420009") is True
    assert recognizer.invalidate_result("111111111111") is True
    assert recognizer.invalidate_result("012345678901") is True
    assert recognizer.invalidate_result("21100342001") is True
    assert recognizer.invalidate_result("2110034200171") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = UyRutRecognizer(name="CustomUyRut")
    assert recognizer.name == "CustomUyRut"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a RUT number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(UyRutRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="211003420017", language="en", entities=["UY_RUT"])
    with_context = analyzer.analyze(
        text="211003420017",
        language="en",
        entities=["UY_RUT"],
        context=["rut"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(UyRutRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="RUT 211003420018",
        language="en",
        entities=["UY_RUT"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "UyRutRecognizer"]
    assert len(entries) == 1, "UyRutRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "uy"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["UyRutRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "UY_RUT" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My RUT is 211003420017",
        language="en",
        entities=["UY_RUT"],
    )
    assert [result.entity_type for result in results] == ["UY_RUT"]


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
        entries = [r for r in recognizers if r.get("name") == "UyRutRecognizer"]
        assert len(entries) == 1, f"UyRutRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"UyRutRecognizer enabled in {yaml_path}"
