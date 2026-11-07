"""Tests for Peru RUC tax identifier recognizer."""

from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import PeRucRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create a Peru RUC recognizer instance for testing."""
    return PeRucRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Peru RUC entity type for testing."""
    return ["PE_RUC"]


# Fixtures are synthetic numbers whose check digits were
# computed with the pinned mod-11 algorithm, except for
# 20503644968, 20131312955, 20100070970, 20612504301, and
# 20557912879, which are published examples. No fixture
# is a real person's number: the published examples are
# company RUCs (SUNAT itself, Supermercados Peruanos
# S.A., Negocios Inmobiliarios Dream House S.A.C., and
# the Odoo Peru localization example).
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid RUC numbers ---
        # published examples, plain
        ("20503644968", 1, (0, 11), 0.5),
        ("20131312955", 1, (0, 11), 0.5),  # SUNAT itself
        ("20100070970", 1, (0, 11), 0.5),  # remainder 1: check digit 0
        ("20612504301", 1, (0, 11), 0.5),  # remainder 0: check digit 1
        ("20557912879", 1, (0, 11), 0.5),
        # one synthetic number per issued prefix, check digits computed
        ("10456789019", 1, (0, 11), 0.5),  # 10 natural person with DNI
        ("15123456782", 1, (0, 11), 0.5),  # 15 natural person, other document
        ("17123456785", 1, (0, 11), 0.5),  # 17 foreigner or others
        ("20123456786", 1, (0, 11), 0.5),  # 20 legal entity
        ("20987654326", 1, (0, 11), 0.5),  # 20 legal entity
        # a number embedded in a sentence
        ("RUC 20503644968 on file", 1, (4, 15), 0.5),
        # a sentence period after the check digit does not block the match
        ("RUC 20503644968.", 1, (4, 15), 0.5),
        # --- Invalid: wrong check digit ---
        ("20503644969", 0, (0, 0), 0),
        ("20100070971", 0, (0, 0), 0),
        # --- Invalid: prefix outside the issued set ---
        # each has a valid checksum, so only the prefix rule rejects it
        ("30123456781", 0, (0, 0), 0),  # 30 is not a RUC prefix
        ("16123456789", 0, (0, 0), 0),  # 16 is historical, no longer issued
        ("00123456785", 0, (0, 0), 0),  # 00 is not a RUC prefix
        ("99123456781", 0, (0, 0), 0),  # 99 is not a RUC prefix
        # --- Invalid: all digits identical ---
        ("11111111111", 0, (0, 0), 0),
        ("00000000000", 0, (0, 0), 0),
        # --- Invalid: fully sequential digit runs ---
        # 10987654321 has a valid prefix and a valid
        # checksum, so only the sequential rule rejects it
        ("10987654321", 0, (0, 0), 0),
        ("01234567890", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("2050364496", 0, (0, 0), 0),  # 10 digits
        ("205036449680", 0, (0, 0), 0),  # 12 digits
        # --- Lookalikes: same shape, but not a RUC number ---
        ("120503644968", 0, (0, 0), 0),  # a leading digit
        ("205036449681", 0, (0, 0), 0),  # a trailing digit
        ("20503644968-1", 0, (0, 0), 0),  # a longer hyphenated run
        ("20503.644968", 0, (0, 0), 0),  # dot form is not accepted
        ("A20503644968", 0, (0, 0), 0),  # a leading letter
        ("20503644968A", 0, (0, 0), 0),  # a trailing letter
        ("205 03644968", 0, (0, 0), 0),  # spaced form is not accepted
        # fmt: on
    ],
)
def test_when_ruc_in_text_then_all_ids_found(
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
    """Both RUC numbers in one text are found at their exact positions."""
    results = recognizer.analyze("RUCs: 20503644968 and 20131312955", entities)

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 17, 0.5)
    assert_result(results[1], entities[0], 22, 33, 0.5)


sequential_test_set = [
    ["01234567890", True],
    ["98765432109", True],
    ["10987654321", True],
    ["20503644968", False],
    ["20100070970", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the RUC digits."""
    assert PeRucRecognizer._is_sequential(input_text) == expected_output


def test_check_digit(recognizer):
    """The check digit matches the published worked examples."""
    assert recognizer._check_digit("2050364496") == "8"
    assert recognizer._check_digit("2013131295") == "5"
    assert recognizer._check_digit("2055791287") == "9"
    assert recognizer._check_digit("1045678901") == "9"
    # remainder 1 gives a check digit of 0
    assert recognizer._check_digit("2010007097") == "0"
    # remainder 0 gives a check digit of 1
    assert recognizer._check_digit("2061250430") == "1"


def test_has_valid_checksum(recognizer):
    """The mod-11 checksum accepts valid numbers and rejects the rest."""
    assert recognizer._has_valid_checksum("20503644968") is True
    assert recognizer._has_valid_checksum("20131312955") is True
    assert recognizer._has_valid_checksum("20100070970") is True
    assert recognizer._has_valid_checksum("20612504301") is True
    assert recognizer._has_valid_checksum("20557912879") is True
    assert recognizer._has_valid_checksum("20503644969") is False
    assert recognizer._has_valid_checksum("20100070971") is False
    # the remainder-0 edge case takes 1, not 0
    assert recognizer._has_valid_checksum("20612504300") is False
    # the remainder-1 edge case takes 0, not 1
    assert recognizer._has_valid_checksum("20100070971") is False


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("20503644968") is False
    assert recognizer.invalidate_result("20100070970") is False
    assert recognizer.invalidate_result("20612504301") is False
    assert recognizer.invalidate_result("10456789019") is False
    assert recognizer.invalidate_result("20503644969") is True
    assert recognizer.invalidate_result("30123456781") is True
    assert recognizer.invalidate_result("16123456789") is True
    assert recognizer.invalidate_result("11111111111") is True
    assert recognizer.invalidate_result("10987654321") is True
    assert recognizer.invalidate_result("2050364496") is True
    assert recognizer.invalidate_result("205036449680") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = PeRucRecognizer(name="CustomPeRuc")
    assert recognizer.name == "CustomPeRuc"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a RUC number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(PeRucRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(text="20503644968", language="en", entities=["PE_RUC"])
    with_context = analyzer.analyze(
        text="20503644968",
        language="en",
        entities=["PE_RUC"],
        context=["ruc"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the checksum rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(PeRucRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="RUC 20503644969",
        language="en",
        entities=["PE_RUC"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "PeRucRecognizer"]
    assert len(entries) == 1, "PeRucRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "pe"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["PeRucRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "PE_RUC" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My RUC is 20503644968",
        language="en",
        entities=["PE_RUC"],
    )
    assert [result.entity_type for result in results] == ["PE_RUC"]
