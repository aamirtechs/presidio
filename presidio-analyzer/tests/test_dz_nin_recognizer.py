"""Tests for Algerian national identification number recognizer."""

import datetime
from pathlib import Path

import presidio_analyzer
import pytest
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import DzNinRecognizer
from presidio_analyzer.recognizer_registry import (
    RecognizerRegistry,
    RecognizerRegistryProvider,
)

from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    """Create an Algerian national identification number recognizer."""
    return DzNinRecognizer()


@pytest.fixture(scope="module")
def entities():
    """Return the Algerian national identification number entity type."""
    return ["DZ_NIN"]


# Fixtures are synthetic numbers built from the pinned structure in
# decret executif 23-316, article 6: two digits for sex, act mention
# and birth in Algeria or abroad; three digits, the last three digits
# of the birth registration year; four digits, the commune or country
# code of birth registration; five digits, the birth act number; two
# digits, the register serial per year; and two digits, the control
# key. The control key formula is not published, so the fixture keys
# are arbitrary two digit values, and no fixture is a real person's
# number. Commune codes were checked against the syscli-algeria
# commune data through dz-format.
@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # fmt: off
        # --- Valid national identification numbers ---
        # synthetic numbers, keys arbitrary (no published key formula)
        ("100041601000010147", 1, (0, 18), 0.5),  # male 2004 Alger-centre act 1
        ("119953101001230208", 1, (0, 18), 0.5),  # female 1995 Oran act 123
        ("100230901543211299", 1, (0, 18), 0.5),  # male 2023 Blida act 54321
        ("209832301000070300", 1, (0, 18), 0.5),  # male dual 1983 Annaba act 7
        ("110000601999999955", 1, (0, 18), 0.5),  # female 2000 Bejaia act 99999
        ("109991501123450731", 1, (0, 18), 0.5),  # male 1999 Tizi Ouzou act 12345
        ("210260101000020173", 1, (0, 18), 0.5),  # female dual 2026 Adrar act 2
        ("100043501004561019", 1, (0, 18), 0.5),  # male 2004 Boumerdes act 456
        ("119831101777772542", 1, (0, 18), 0.5),  # female 1983 Tamenghasset
        ("100161609000030466", 1, (0, 18), 0.5),  # male 2016 Bir Mourad Rais act 3
        ("110091602000110523", 1, (0, 18), 0.5),  # female 2009 Sidi M'Hamed act 11
        ("100181603000990981", 1, (0, 18), 0.5),  # male 2018 El Madania act 99
        ("209120901002341107", 1, (0, 18), 0.5),  # male dual 1912 Blida act 234
        ("119752301005552090", 1, (0, 18), 0.5),  # female 1975 Annaba act 555
        ("100049999000010147", 1, (0, 18), 0.5),  # male 2004 country code block
        # a number embedded in a sentence
        ("NIN 100041601000010147 on file", 1, (4, 22), 0.5),
        ("NIN 119953101001230208", 1, (4, 22), 0.5),
        ("NIN:100041601000010147", 1, (4, 22), 0.5),
        # a sentence period after the key does not block the match
        ("NIN 100041601000010147.", 1, (4, 22), 0.5),
        # --- Invalid: implausible registration year ---
        ("100991601000010147", 0, (0, 0), 0),  # year 099, 2099, in the future
        ("100501601000010147", 0, (0, 0), 0),  # year 050, 2050, in the future
        ("101501601000010147", 0, (0, 0), 0),  # year 150, no plausible reading
        ("105001601000010147", 0, (0, 0), 0),  # year 500, no plausible reading
        ("108501601000010147", 0, (0, 0), 0),  # year 850, 1850, no living holder
        ("102001601000010147", 0, (0, 0), 0),  # year 200, no plausible reading
        ("107771601000010147", 0, (0, 0), 0),  # year 777, no plausible reading
        # --- Invalid: zero blocks ---
        ("100040000000010147", 0, (0, 0), 0),  # commune or country code 0000
        ("100041601000000147", 0, (0, 0), 0),  # birth act number 00000
        ("100041601000010047", 0, (0, 0), 0),  # register serial 00
        # --- Invalid: all digits identical ---
        ("111111111111111111", 0, (0, 0), 0),  # rejected as implausible
        ("000000000000000000", 0, (0, 0), 0),  # rejected as implausible
        ("999999999999999999", 0, (0, 0), 0),  # rejected as implausible
        # --- Invalid: fully sequential digit runs ---
        ("012345678901234567", 0, (0, 0), 0),
        ("987654321098765432", 0, (0, 0), 0),
        ("123456789012345678", 0, (0, 0), 0),
        # --- Invalid: wrong lengths ---
        ("10004160100001014", 0, (0, 0), 0),  # 17 digits
        ("1000416010000101470", 0, (0, 0), 0),  # 19 digits
        # --- Lookalikes: same shape, but not a national identification number ---
        ("1100041601000010147", 0, (0, 0), 0),  # a leading digit
        ("A100041601000010147", 0, (0, 0), 0),  # a leading letter
        ("100041601000010147A", 0, (0, 0), 0),  # a trailing letter
        ("1000 4160 1000 0101 47", 0, (0, 0), 0),  # spaced form, not matched
        ("1000-4160-1000-0101-47", 0, (0, 0), 0),  # dashed form, not matched
        ("100041601000010147/2", 0, (0, 0), 0),  # slashed form, not matched
        # fmt: on
    ],
)
def test_when_nin_in_text_then_all_ids_found(
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
    """Both national identification numbers in one text are found."""
    results = recognizer.analyze(
        "NINs: 100041601000010147 and 119953101001230208", entities
    )

    assert len(results) == 2
    assert_result(results[0], entities[0], 6, 24, 0.5)
    assert_result(results[1], entities[0], 29, 47, 0.5)


def test_two_invalid_numbers_in_text(recognizer, entities):
    """Two numbers with implausible years give no results."""
    results = recognizer.analyze(
        "NINs: 100991601000010147 and 101501601000010147", entities
    )

    assert results == []


sequential_test_set = [
    ["012345678901234567", True],
    ["987654321098765432", True],
    ["123456789012345678", True],
    ["100041601000010147", False],
    ["119953101001230208", False],
]


@pytest.mark.parametrize("input_text, expected_output", sequential_test_set)
def test_is_sequential(input_text, expected_output):
    """Test the sequential run check used on the number digits."""
    assert DzNinRecognizer._is_sequential(input_text) == expected_output


def _number_with_year(year_code):
    """Build a structurally valid number with the given 3 digit year."""
    return f"10{year_code}1601000010147"


def test_has_valid_registration_year(recognizer):
    """The registration year rule accepts plausible years and rejects the rest."""
    assert recognizer._has_valid_registration_year(_number_with_year("004")) is True
    assert recognizer._has_valid_registration_year(_number_with_year("000")) is True
    assert recognizer._has_valid_registration_year(_number_with_year("026")) is True
    assert recognizer._has_valid_registration_year(_number_with_year("999")) is True
    assert recognizer._has_valid_registration_year(_number_with_year("900")) is True
    assert recognizer._has_valid_registration_year(_number_with_year("983")) is True
    assert recognizer._has_valid_registration_year(_number_with_year("099")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("050")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("100")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("150")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("500")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("850")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("899")) is False
    assert recognizer._has_valid_registration_year(_number_with_year("777")) is False


def test_current_registration_year_is_accepted(recognizer):
    """The current year as registration year is valid, next year is not."""
    this_year = datetime.date.today().year % 1000
    assert (
        recognizer._has_valid_registration_year(_number_with_year(f"{this_year:03d}"))
        is True
    )
    next_year = (datetime.date.today().year + 1) % 1000
    assert (
        recognizer._has_valid_registration_year(_number_with_year(f"{next_year:03d}"))
        is False
    )


def test_future_registration_year_is_rejected(recognizer, entities):
    """A number with next year as its registration year is never detected."""
    next_year = datetime.date.today().year + 1
    number = _number_with_year(f"{next_year % 1000:03d}")
    assert recognizer.analyze(number, entities) == []


def test_control_key_is_not_validated(recognizer, entities):
    """The last two digits are not checked: the key formula is unpublished."""
    for key in ["00", "47", "99"]:
        number = f"1000416010000101{key}"
        assert recognizer.analyze(number, entities) != []


def test_invalidate_result_rules(recognizer):
    """Each structural rule rejects its own case directly."""
    assert recognizer.invalidate_result("100041601000010147") is False
    assert recognizer.invalidate_result("119953101001230208") is False
    assert recognizer.invalidate_result("209832301000070300") is False
    assert recognizer.invalidate_result("110000601999999955") is False
    assert recognizer.invalidate_result("100049999000010147") is False
    assert recognizer.invalidate_result("100991601000010147") is True
    assert recognizer.invalidate_result("100501601000010147") is True
    assert recognizer.invalidate_result("101501601000010147") is True
    assert recognizer.invalidate_result("108501601000010147") is True
    assert recognizer.invalidate_result("100040000000010147") is True
    assert recognizer.invalidate_result("100041601000000147") is True
    assert recognizer.invalidate_result("100041601000010047") is True
    assert recognizer.invalidate_result("111111111111111111") is True
    assert recognizer.invalidate_result("000000000000000000") is True
    assert recognizer.invalidate_result("999999999999999999") is True
    assert recognizer.invalidate_result("012345678901234567") is True
    assert recognizer.invalidate_result("987654321098765432") is True
    assert recognizer.invalidate_result("10004160100001014") is True
    assert recognizer.invalidate_result("1000416010000101470") is True


def test_supported_language(recognizer):
    """Test that supported language is correctly set."""
    assert recognizer.supported_language == "en"


def test_accepts_name_kwarg():
    """Constructor must accept the ``name`` kwarg the YAML loader passes.

    Without it, loading the recognizer from a registry YAML raises
    ``TypeError``, taking down construction of the whole registry.
    """
    recognizer = DzNinRecognizer(name="CustomDzNin")
    assert recognizer.name == "CustomDzNin"


def test_context_word_boosts_score(mock_nlp_engine):
    """A context word around a number raises its pattern score."""
    registry = RecognizerRegistry()
    registry.add_recognizer(DzNinRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    plain = analyzer.analyze(
        text="100041601000010147", language="en", entities=["DZ_NIN"]
    )
    with_context = analyzer.analyze(
        text="100041601000010147",
        language="en",
        entities=["DZ_NIN"],
        context=["nin"],
    )

    assert len(plain) == 1
    assert plain[0].score == 0.5
    assert len(with_context) == 1
    assert with_context[0].score > plain[0].score


def test_context_does_not_rescue_invalid_id(mock_nlp_engine):
    """A context word never rescues a number the year rule rejects."""
    registry = RecognizerRegistry()
    registry.add_recognizer(DzNinRecognizer())
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)

    results = analyzer.analyze(
        text="nin 100991601000010147",
        language="en",
        entities=["DZ_NIN"],
    )
    assert results == []


def test_loads_from_default_recognizers_yaml(mock_nlp_engine):
    """Recognizer is registered in the default YAML and detects once enabled."""
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    recognizers = yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
    entries = [r for r in recognizers if r.get("name") == "DzNinRecognizer"]
    assert len(entries) == 1, "DzNinRecognizer missing from YAML"
    entry = entries[0]
    assert entry["country_code"] == "dz"
    assert entry["enabled"] is False
    assert "en" in entry["supported_languages"]

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["en"],
            "recognizers": [dict(entry, enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["DzNinRecognizer"]
    entities = {e for rec in registry.recognizers for e in rec.supported_entities}
    assert "DZ_NIN" in entities

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=mock_nlp_engine)
    results = analyzer.analyze(
        text="My NIN is 100041601000010147",
        language="en",
        entities=["DZ_NIN"],
    )
    assert [result.entity_type for result in results] == ["DZ_NIN"]


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
        entries = [r for r in recognizers if r.get("name") == "DzNinRecognizer"]
        assert len(entries) == 1, f"DzNinRecognizer missing from {yaml_path}"
        assert entries[0]["enabled"] is False, f"DzNinRecognizer enabled in {yaml_path}"
