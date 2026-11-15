from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class RoCnpRecognizer(PatternRecognizer):
    """
    Recognize Romanian personal numeric codes (Cod Numeric Personal, CNP).

    The CNP is the national identification number issued
    in Romania by the General Directorate for Persons'
    Record to every person born in Romania and to every
    Romanian resident. It has 13 digits in seven parts,
    S AA LL ZZ JJ NNN C:

    - S, one digit for sex and century of birth: 1 male
      and 2 female born 1900 to 1999, 3 male and 4 female
      born 1800 to 1899, 5 male and 6 female born 2000 to
      2099, and 7 male and 8 female resident, for whom
      the century is not encoded.
    - AA LL ZZ, six digits for the date of birth, YY MM DD.
    - JJ, two digits for the county code: 01 to 39 for
      the counties in alphabetical order, 40 for
      Bucharest, 41 to 46 for its six sectors, 47 and 48
      for the former sectors 7 and 8, 51 and 52 for
      Calarasi and Giurgiu, and 70 for the new civil
      status system, in use since May 2024, which uses
      70 in place of the county for every registration.
      There is no 49 or 50.
    - NNN, three digits, a serial number from 001 to 999.
    - C, one check digit.

    The CNP is written plain, 13 digits with no
    separators, for example 1960523123457.

    Checksum: the first 12 digits are multiplied by
    the matching digits of the constant 279146358279,
    the products are added, and the sum is divided by
    11. The remainder is the check digit, except that
    a remainder of 10 gives a check digit of 1. Worked
    examples, verified by hand and by computation:
    1960523123457 (a man born 1996-05-23, county 12
    Cluj): 1x2 + 9x7 + 6x9 + 0x1 + 5x4 + 2x6 + 3x3 +
    1x5 + 2x8 + 3x2 + 4x7 + 5x9 = 260, 260 mod 11 = 7;
    1630615123457 (the python-stdnum docstring example,
    a man born 1963-06-15, county 12 Cluj): the
    products add to 216, 216 mod 11 = 7; 1940616346114
    (a man born 1994-06-16, county 34 Teleorman): the
    products add to 224, 224 mod 11 = 4; 7041218318525
    (a male resident, born 2004-12-18, county 31
    Salaj): the products add to 181, 181 mod 11 = 5;
    6201206018078 (a woman born 2020-12-06, county 01
    Alba): the products add to 140, 140 mod 11 = 8;
    and 5201206346491 (a man born 2020-12-06, county
    34 Teleorman): the products add to 219, 219 mod
    11 = 10, so the check digit is 1. The constant
    279146358279 and the remainder-10 fold are
    confirmed by the CNP specification
    (vimishor/cnp-spec, published on Zenodo as the
    Personal Identification Number Specification),
    by Wikipedia (Romanian identity card), and by
    the python-stdnum ro.cnp module. The
    recognizer's checksum was also cross-checked
    against python-stdnum on all fixtures, with
    zero mismatches on the numbers both accept.

    Five honest caveats. First, the first digit 9
    appears in python-stdnum only, as another
    foreigner form, and is not in the specification,
    so it is rejected here, for example
    9960523123451, which python-stdnum accepts.
    Second, county codes 80 to 83 appear in
    python-stdnum only, as Unknown, and are not in
    the specification, so they are rejected here,
    for example 1960523801231, which python-stdnum
    accepts. Third, Bucharest sectors 7 and 8,
    county codes 47 and 48, were abolished on
    19 December 1979, so they are rejected for
    birth dates after that day when the century is
    known, for example 1850310471239, which
    python-stdnum accepts; for residents (first
    digits 7 and 8) the century is not encoded,
    so the rule is applied only when the date is
    after 19 December 1979 under every plausible
    century. Fourth, the remainder-10 fold means
    remainders 1 and 10 both produce a check digit
    of 1, so the checksum is weaker than a strict
    mod-11 check: a single digit typo that moves
    the sum from one to the other still passes.
    Fifth, for residents (first digits 7 and 8)
    the century is not encoded, so the date is
    accepted when it is real and not in the future
    in either the 1900s or the 2000s; a resident
    code for 29 February 2000, 7000229121234, is
    therefore accepted here although python-stdnum
    rejects it, because python-stdnum assumes 1900
    for those digits and 1900 was not a leap year.
    Two further differences from python-stdnum are
    deliberate: a serial of 000 is rejected here,
    although python-stdnum accepts it, because
    the specification gives the serial range as
    001 to 999; and a birth date in the future is
    rejected here, although python-stdnum does not
    check it.

    The recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because
    a mod-11 check passes about one time in eleven
    for an arbitrary same-shape token, and the
    remainder-10 fold weakens it further.

    Invalidation is based on: the number must be
    exactly 13 digits; the number must not be all
    the same digit; a fully sequential digit run is
    rejected; the first digit must be 1 to 8; the
    next six digits must be a real calendar date
    (month 01 to 12, day 01 to 31, February 29 only
    in a leap year), with the century taken from
    the first digit (1900s for 1 and 2, 1800s for
    3 and 4, 2000s for 5 and 6, and either the
    1900s or the 2000s for residents with 7 and 8),
    and the date must not be in the future; the
    county code must be 01 to 48, 51, 52, or 70,
    and codes 47 and 48 are rejected for birth
    dates after 19 December 1979 when the century
    is known; the serial must be 001 to 999; and
    the check digit must match the 279146358279
    weighted sum mod 11, with a remainder of 10
    giving 1. This follows the approach of the
    Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh
    NID, Sri Lanka NIC, Ghana Card, Kazakhstan
    IIN, Ukraine RNOKPP, Uzbekistan PINFL, Morocco
    CNIE, Indonesia NIK, Malaysia MyKad, Vietnam
    CCCD, Chile RUT, Ecuador Cedula, Iran Code
    Melli, Argentina CUIT/CUIL, Mexico CURP, Peru
    RUC, Colombia NIT, Dominican Republic Cedula,
    Paraguay RUC, Uruguay RUT, Iceland kennitala,
    and Czech birth number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ro"

    PATTERNS = [
        Pattern(
            "RO_CNP",
            r"(?<![A-Za-z0-9./-])\d{13}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "romania",
        "romanian",
        "cnp",
        "cod numeric personal",
        "personal numeric code",
        "numerical personal code",
    ]

    # The weighted-sum constant for the check digit.
    _WEIGHTS = (2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9)

    # County codes in the published table: 01 to 48,
    # 51 and 52, and 70 for the new civil status system.
    _VALID_COUNTIES = frozenset(
        [f"{code:02d}" for code in range(1, 49)] + ["51", "52", "70"]
    )

    # Bucharest sectors 7 and 8 were abolished on this day.
    _SECTOR_CUTOFF = date(1979, 12, 19)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "RO_CNP",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        """
        Check if the pattern text cannot be a RO_CNP entity.

        The number must be 13 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, the first digit must be 1 to 8, the
        next six digits must be a real calendar date
        not in the future with the century taken from
        the first digit, the county code must be in
        the published table with codes 47 and 48
        rejected for birth dates after 19 December
        1979 when the century is known, the serial
        must be 001 to 999, and the check digit must
        match the weighted sum mod 11, with a
        remainder of 10 giving 1.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 13 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued number is never one fully sequential run
            return True
        if text[0] not in "12345678":
            # the first digit is the sex and century, 1 to 8
            return True
        if not self._has_valid_date(text):
            # the next six digits must be a real date, not in the future
            return True
        if not self._has_valid_county(text):
            # the county code must be in the published table
            return True
        if text[9:12] == "000":
            # the serial number runs from 001 to 999
            return True
        return not self._has_valid_checksum(text)

    def _candidate_dates(self, digits: str) -> List[date]:
        """
        Return the plausible birth dates for a CNP body.

        The century comes from the first digit: 1900s
        for 1 and 2, 1800s for 3 and 4, 2000s for 5
        and 6. For residents, first digits 7 and 8,
        the century is not encoded, so both the 1900s
        and the 2000s are tried. Only real calendar
        dates that are not in the future are returned.

        :param digits: The 13 digit personal numeric code
        :return: The list of plausible birth dates, possibly empty
        """
        first = int(digits[0])
        if first in (1, 2):
            centuries = [1900]
        elif first in (3, 4):
            centuries = [1800]
        elif first in (5, 6):
            centuries = [2000]
        else:
            centuries = [1900, 2000]
        year = int(digits[1:3])
        month = int(digits[3:5])
        day = int(digits[5:7])
        today = date.today()
        candidates = []
        for century in centuries:
            try:
                birth_date = date(century + year, month, day)
            except ValueError:
                continue
            if birth_date <= today:
                candidates.append(birth_date)
        return candidates

    def _has_valid_date(self, digits: str) -> bool:
        """
        Check the digits 2 to 7 form a real calendar date.

        The century is taken from the first digit, and
        for residents (first digits 7 and 8) either
        the 1900s or the 2000s counts. The date must
        be real, February 29 only in a leap year, and
        must not be in the future.

        :param digits: The 13 digit personal numeric code
        :return: True when the date is a real calendar date
        """
        return len(self._candidate_dates(digits)) > 0

    def _has_valid_county(self, digits: str) -> bool:
        """
        Check the county code is in the published table.

        County codes are 01 to 48, 51, 52, and 70 for
        the new civil status system. Codes 47 and 48,
        Bucharest sectors 7 and 8, were abolished on
        19 December 1979, so they are rejected when
        every plausible birth date is after that day.

        :param digits: The 13 digit personal numeric code
        :return: True when the county code is valid
        """
        county = digits[7:9]
        if county not in self._VALID_COUNTIES:
            return False
        if county in ("47", "48"):
            candidates = self._candidate_dates(digits)
            if candidates and all(
                birth_date > self._SECTOR_CUTOFF for birth_date in candidates
            ):
                return False
        return True

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the weighted mod-11 checksum of a 13 digit code.

        The first 12 digits are multiplied by the
        matching digits of the constant 279146358279,
        the products are added, and the sum mod 11 is
        the check digit, except that a remainder of
        10 gives a check digit of 1.

        :param digits: The 13 digit personal numeric code
        :return: True when the checksum holds
        """
        check_digit = RoCnpRecognizer._check_digit(digits[:12])
        return check_digit == digits[12]

    @staticmethod
    def _check_digit(first_twelve: str) -> str:
        """
        Compute the check digit for the first 12 digits.

        The weighted sum of the first 12 digits with
        the constant 279146358279, mod 11, is the
        check digit, except that a remainder of 10
        gives a check digit of 1.

        :param first_twelve: The first 12 digits of the code
        :return: The check digit, 0 to 9
        """
        total = sum(
            weight * int(digit)
            for weight, digit in zip(RoCnpRecognizer._WEIGHTS, first_twelve)
        )
        remainder = total % 11
        return "1" if remainder == 10 else str(remainder)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 0123456789012 ascends by one each step, and
        9876543210987 descends by one each step.

        :param digits: Numeric string to check
        :return: True if every digit follows the previous one sequentially
        """
        values = [int(digit) for digit in digits]
        ascending = all(
            (values[i] + 1) % 10 == values[i + 1] for i in range(len(values) - 1)
        )
        descending = all(
            (values[i] - 1) % 10 == values[i + 1] for i in range(len(values) - 1)
        )
        return ascending or descending
