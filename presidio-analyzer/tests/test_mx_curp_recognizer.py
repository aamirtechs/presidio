"""Tests for Mexico CURP population registry code recognizer."""

from datetime import date
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import MxCurpRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Mexico CURP recognizer instance for testing."""
    return MxCurpRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Mexico CURP entity type for testing."""
    return ["MX_CURP"]


# Fixtures are synthetic codes whose check digits were
# computed with the pinned mod-10 algorithm, except for
# HEGG560427MVZRRL04, SABC560626MDFLRN01,
# MOTR930411HJCRMN03, GOGR800101HDFRRL07,
# MASM020203MDFRNTA6, LOGJ070423HDFPTSA0, and
# GUMB690113MDFTLT08, which are published examples.
# No fixture is a real person's code.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid CURP codes ---
        # published examples, plain
        ("HEGG560427MVZRRL04", 1, (0, 18), 0.5),
        ("SABC560626MDFLRN01", 1, (0, 18), 0.5),
        ("MOTR930411HJCRMN03", 1, (0, 18), 0.5),
        ("GOGR800101HDFRRL07", 1, (0, 18), 0.5),
        ("MASM020203MDFRNTA6", 1, (0, 18), 0.5),
        ("LOGJ070423HDFPTSA0", 1, (0, 18), 0.5),
        ("GUMB690113MDFTLT08", 1, (0, 18), 0.5),
        ("MOTR930411HJCRMG70", 1, (0, 18), 0.5),  # remainder 0: check digit 0
        # X sex letter, issued since 2022
        ("HEGG560427XVZRRL08", 1, (0, 18), 0.5),
        # NE state code, born abroad
        ("HEGG560427MNERRL09", 1, (0, 18), 0.5),
        # a spread of state codes, synthetic, check digits computed
        ("HEGG800101HBCRRL08", 1, (0, 18), 0.5),  # BC Baja California
        ("HEGG800101HNLRRL00", 1, (0, 18), 0.5),  # NL Nuevo Leon
        ("HEGG800101HQTRRL08", 1, (0, 18), 0.5),  # QT Queretaro
        ("HEGG800101HYNRRL04", 1, (0, 18), 0.5),  # YN Yucatan
        ("HEGG800101HZSRRL01", 1, (0, 18), 0.5),  # ZS Zacatecas
        ("HEGG800101HCHRRL01", 1, (0, 18), 0.5),  # CH Chihuahua
        ("HEGG800101HCMRRL01", 1, (0, 18), 0.5),  # CM Colima, not DF
        ("HEGG800101HMCRRL01", 1, (0, 18), 0.5),  # MC Estado de Mexico
        ("HEGG800101HJCRRL02", 1, (0, 18), 0.5),  # JC Jalisco
        ("HEGG800101HGRRRL07", 1, (0, 18), 0.5),  # GR Guerrero
        # leap day 1956-02-29 is a real date
        ("HEGG560229MVZRRL08", 1, (0, 18), 0.5),
        # born 2026-01-01, letter differentiator for a post-2000 birth
        ("HEGG260101MVZRRLA3", 1, (0, 18), 0.5),
        # spaced forms: single spaces at segment boundaries
        ("HEGG 560427 MVZRRL04", 1, (0, 20), 0.5),
        ("HEGG 560427 M VZ RRL 0 4", 1, (0, 24), 0.5),
        # a code embedded in a sentence
        (
            "Her CURP HEGG560427MVZRRL04 was verified",
            1,
            (9, 27),
            0.5,
        ),
        # a sentence period after the check digit does not block the match
        ("CURP HEGG560427MVZRRL04.", 1, (5, 23), 0.5),
        # a spaced code embedded in a sentence
        ("CURP HEGG 560427 MVZRRL04 on file", 1, (5, 25), 0.5),
        # --- Invalid: wrong check digit ---
        ("HEGG560427MVZRRL05", 0, (0, 0), 0),
        ("MASM020203MDFRNTA7", 0, (0, 0), 0),
        # --- Invalid: state code outside the catalogue ---
        # each has a valid checksum, so only the state rule rejects it
        ("HEGG560427MXXRRL02", 0, (0, 0), 0),
        ("HEGG560427MZZRRL06", 0, (0, 0), 0),
        # --- Invalid: birth date cannot exist ---
        # each has a valid checksum, so only the date rule rejects it
        ("HEGG561327MVZRRL03", 0, (0, 0), 0),  # month 13
        ("HEGG560432MVZRRL09", 0, (0, 0), 0),  # day 32
        ("HEGG560230MVZRRL09", 0, (0, 0), 0),  # Feb 30
        ("HEGG550229MVZRRL01", 0, (0, 0), 0),  # Feb 29 in 1955, not a leap year
        ("HEGG261231MVZRRLA0", 0, (0, 0), 0),  # 2026-12-31 is in the future
        # --- Invalid: sex letter outside H, M, X ---
        # each has a valid checksum, so only the sex rule rejects it
        ("HEGG560427QVZRRL04", 0, (0, 0), 0),
        ("HEGG560427ZVZRRL02", 0, (0, 0), 0),
        # --- Invalid: differentiator breaks the century rule ---
        # each has a valid checksum, so only the differentiator rule rejects it
        ("HEGG560427MVZRRLA4", 0, (0, 0), 0),  # letter for a 1956 birth
        ("MASM020203MDFRNT06", 0, (0, 0), 0),  # digit for a 2002 birth
        # --- Invalid: all characters identical ---
        ("AAAAAAAAAAAAAAAAAA", 0, (0, 0), 0),
        ("000000000000000000", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("HEGG560427MVZRRL0", 0, (0, 0), 0),  # 17 characters
        ("HEGG560427MVZRRL044", 0, (0, 0), 0),  # 19 characters
        # --- Lookalikes: same shape, but not a CURP code ---
        ("XHEGG560427MVZRRL04", 0, (0, 0), 0),  # a leading letter
        ("HEGG560427MVZRRL04X", 0, (0, 0), 0),  # a trailing letter
        ("HEGG560427MVZRRL04-5", 0, (0, 0), 0),  # a longer hyphenated run
        ("HEGG.560427.MVZRRL04", 0, (0, 0), 0),  # dot form is not accepted
        ("hegg560427mvzrrl04", 0, (0, 0), 0),  # lowercase is not accepted
        ("HEGG  560427MVZRRL04", 0, (0, 0), 0),  # double space is not accepted
        ("0123456789ABCDEFGH", 0, (0, 0), 0),  # a fully sequential run
        # fmt: on
    ],
)
def test_when_curp_in_text_then_all_ids_found(
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
    """Both CURP codes in one text are found at their exact positions."""
    results = recognizer.analyze(
        "CURPs: HEGG560427MVZRRL04 and MASM020203MDFRNTA6", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 7, 25, 0.5)
    assert_result(results[1], entities[0], 30, 48, 0.5)


sequential_test_set = [
    ["0123456789ABCDEFGH", True],
    ["HGFEDCBA9876543210", True],
    ["HEGG560427MVZRRL04", False],
    ["MASM020203MDFRNTA6", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the CURP characters."""
    assert MxCurpRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("HEGG560427MVZRRL0") == "4"
    assert recognizer._check_digit("SABC560626MDFLRN0") == "1"
    assert recognizer._check_digit("MOTR930411HJCRMN0") == "3"
    assert recognizer._check_digit("GOGR800101HDFRRL0") == "7"
    assert recognizer._check_digit("MASM020203MDFRNTA") == "6"
    assert recognizer._check_digit("LOGJ070423HDFPTSA") == "0"
    assert recognizer._check_digit("GUMB690113MDFTLT0") == "8"
    # remainder 0 gives a check digit of 0
    assert recognizer._check_digit("MOTR930411HJCRMG7") == "0"


def test_has_valid_checksum(recognizer):
    """The mod-10 checksum accepts valid codes and rejects the rest."""
    assert recognizer._has_valid_checksum("HEGG560427MVZRRL04") is True
    assert recognizer._has_valid_checksum("SABC560626MDFLRN01") is True
    assert recognizer._has_valid_checksum("MASM020203MDFRNTA6") is True
    assert recognizer._has_valid_checksum("HEGG560427MVZRRL05") is False
    # the deepwiki table prints MOTR930411HJCRMG73, but the
    # weighted sum is 2580, so the check digit is 0, not 3
    assert recognizer._has_valid_checksum("MOTR930411HJCRMG73") is False
    assert recognizer._has_valid_checksum("MOTR930411HJCRMG70") is True


def test_birth_date(recognizer):
    """The YYMMDD part decodes to real dates only."""
    assert recognizer._birth_date("560427") == date(1956, 4, 27)
    assert recognizer._birth_date("020203") == date(2002, 2, 3)
    assert recognizer._birth_date("991231") == date(1999, 12, 31)
    assert recognizer._birth_date("260101") == date(2026, 1, 1)
    assert recognizer._birth_date("560229") == date(1956, 2, 29)
    assert recognizer._birth_date("550229") is None
    assert recognizer._birth_date("561327") is None
    assert recognizer._birth_date("560432") is None


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("HEGG560427MVZRRL04") is False
    assert recognizer.invalidate_result("MASM020203MDFRNTA6") is False
    assert recognizer.invalidate_result("HEGG 560427 MVZRRL04") is False
    assert recognizer.invalidate_result("HEGG560427MXXRRL02") is True
    assert recognizer.invalidate_result("HEGG560427MZZRRL06") is True
    assert recognizer.invalidate_result("HEGG561327MVZRRL03") is True
    assert recognizer.invalidate_result("HEGG550229MVZRRL01") is True
    assert recognizer.invalidate_result("HEGG261231MVZRRLA0") is True
    assert recognizer.invalidate_result("HEGG560427QVZRRL04") is True
    assert recognizer.invalidate_result("HEGG560427MVZRRLA4") is True
    assert recognizer.invalidate_result("MASM020203MDFRNT06") is True
    assert recognizer.invalidate_result("AAAAAAAAAAAAAAAAAA") is True
    assert recognizer.invalidate_result("HEGG560427MVZRRL0") is True


def test_all_state_codes_accepted(recognizer):
    """Every one of the 33 catalogue state codes validates."""
    assert len(recognizer.STATE_CODES) == 33
    for code in sorted(recognizer.STATE_CODES):
        prefix = f"HEGG800101H{code}RRL0"
        curp = prefix + recognizer._check_digit(prefix)
        assert recognizer.invalidate_result(curp) is False, code


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = MxCurpRecognizer(name="CustomMxCurp")
    assert recognizer.name == "CustomMxCurp"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a CURP code raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MxCurpRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="HEGG560427MVZRRL04", language="en", entities=["MX_CURP"]
    )
    with_context = analyzer.analyze(
        text="HEGG560427MVZRRL04",
        language="en",
        entities=["MX_CURP"],
        context=["curp"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a code the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(MxCurpRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="CURP HEGG560427MVZRRL05",
        language="en",
        entities=["MX_CURP"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "MxCurpRecognizer"]
    assert len(entries) == 1, "MxCurpRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "mx"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["MxCurpRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "MX_CURP" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My CURP is HEGG560427MVZRRL04",
        language="en",
        entities=["MX_CURP"],
    )
    assert [result.entity_type for result in results] == ["MX_CURP"]
