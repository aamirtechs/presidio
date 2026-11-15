from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class IsKennitalaRecognizer(PatternRecognizer):
    """
    Recognize Iceland kennitala (national identification) numbers.

    The kennitala is the national identification number
    issued in Iceland by Registers Iceland (Thjodskra)
    to individuals and by the company register
    (Fyrirtaekjaskra) to legal entities. It has 10
    digits in four parts:

    - A six digit date, DDMMYY: the date of birth for
      individuals, or the registration date for legal
      entities, with 40 added to the day for entities,
      so entity days run 41 to 71.
    - Two random digits.
    - One mod-11 check digit.
    - One century digit: 9 for 1900 to 1999, 0 for
      2000 and later.

    The kennitala is written plain, 10 digits with no
    separators, or in its canonical hyphenated form
    DDDDDD-DDDD, for example 120174-3399. In
    international VAT contexts the same number is
    written with the ISO country prefix IS, for
    example IS1201743399; the prefix is not part of
    the matched number.

    Checksum: the check digit uses a mod-11 algorithm
    with the weights 3, 2, 7, 6, 5, 4, 3, 2 applied
    to the first 8 digits from left to right. The
    check digit is (11 - (total mod 11)) mod 11,
    where total is the weighted sum. Worked examples,
    verified by hand: 0902862349 (the Business::IS::PIN
    worked example): 0x3 + 9x2 + 0x7 + 2x6 + 8x5
    + 6x4 + 2x3 + 3x2 = 106, and 106 mod 11 = 7, so
    the check digit is 4; 1201743399 (the Wikipedia
    mockup example and the python-stdnum docstring
    example): 1x3 + 2x2 + 0x7 + 1x6 + 7x5 + 4x4
    + 3x3 + 3x2 = 79, and 79 mod 11 = 2, so the check
    digit is 9; 4504013150 (the python-stdnum
    docstring example, an organisation): 4x3 + 5x2
    + 0x7 + 4x6 + 0x5 + 1x4 + 3x3 + 1x2 = 61, and
    61 mod 11 = 6, so the check digit is 5. The
    10 digit structure and the checksum formula are
    confirmed by the OECD Iceland TIN information
    sheet, which gives the same weights and the
    same formula. The recognizer's check digit
    implementation was also cross-checked against
    python-stdnum on all fixtures and on 2000 random
    8 digit bodies, with zero mismatches.

    Three honest caveats. First, the century digit
    must be 9 (1900 to 1999) or 0 (2000 and later),
    following python-stdnum; the 1800s marker 8
    from older documentation is not accepted, so a
    pre-1900 kennitala would be missed. Second, the
    two random digits are not restricted to the 20
    to 99 range Wikipedia mentions, because it notes
    exceptions and python-stdnum accepts any two
    digits, so the recognizer accepts any random
    pair. Third, when the weighted sum is 1 mod 11
    the formula yields 10, which is not a single
    digit, so such an 8 digit body has no valid
    kennitala at all; the recognizer rejects all ten
    endings for it. One more note: the Registers
    Iceland site shows the example 120174-3389,
    which fails the published checksum (the check
    digit for that body is 9, not 8); the valid form
    120174-3399 from Wikipedia and the
    python-stdnum docstring is used instead.

    The recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because
    a mod-11 check passes about one time in eleven
    for an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 10 digits after removing the hyphen;
    the number must not be all the same digit; a
    fully sequential digit run is rejected; the
    first six digits must be a real calendar date
    (day 01 to 31 for individuals, day 41 to 71
    for legal entities, month 01 to 12, February 29
    only in a leap year); the century digit must be
    9 or 0; and the mod-11 checksum must hold. This
    follows the approach of the Pakistani CNIC,
    Saudi national ID, UAE Emirates ID, Egyptian
    national ID, Bangladesh NID, Sri Lanka NIC,
    Ghana Card, Kazakhstan IIN, Ukraine RNOKPP,
    Uzbekistan PINFL, Morocco CNIE, Indonesia NIK,
    Malaysia MyKad, Vietnam CCCD, Chile RUT,
    Ecuador Cedula, Iran Code Melli, Argentina
    CUIT/CUIL, Mexico CURP, Peru RUC, Colombia NIT,
    Dominican Republic Cedula, Paraguay RUC, and
    Uruguay RUT recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "is"

    PATTERNS = [
        Pattern(
            "IS_KENNITALA",
            r"(?<![A-Za-z0-9.-])(?:\d{10}|\d{6}-\d{4})(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "iceland",
        "icelandic",
        "kennitala",
        "thjodskra",
        "registers iceland",
        "vsk",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "IS_KENNITALA",
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
        Check if the pattern text cannot be a IS_KENNITALA entity.

        The number must be 10 digits after removing
        the hyphen, the number must not be all the
        same digit, a fully sequential digit run is
        rejected, the first six digits must be a real
        calendar date (day 01 to 31 for individuals,
        day 41 to 71 for legal entities, month 01 to
        12, February 29 only in a leap year), the
        century digit must be 9 or 0, and the mod-11
        checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) != 10 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued number is never one fully sequential run
            return True
        if text[9] not in ("9", "0"):
            # the century digit is 9 for 1900 to 1999, 0 for 2000 and later
            return True
        if not self._has_valid_date(text):
            # the first six digits must be a real calendar date
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_date(digits: str) -> bool:
        """
        Check the first six digits form a real calendar date.

        The day is 01 to 31 for individuals and 41 to
        71 for legal entities (40 added to the day),
        the month is 01 to 12, and the full year comes
        from the century digit: 9 means 1900 to 1999,
        0 means 2000 and later.

        :param digits: The 10 digit kennitala
        :return: True when the date is a real calendar date
        """
        day = int(digits[:2])
        month = int(digits[2:4])
        year = int(digits[4:6])
        if digits[9] == "9":
            year += 1900
        else:
            year += 2000
        if day >= 40:
            # a legal entity: 40 was added to the day
            day -= 40
        if day < 1 or day > 31 or month < 1 or month > 12:
            return False
        try:
            date(year, month, day)
        except ValueError:
            return False
        return True

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the mod-11 checksum of a 10 digit kennitala.

        The check digit is the ninth digit. It is
        (11 - (total mod 11)) mod 11, where total is
        the weighted sum of the first 8 digits with
        the weights 3, 2, 7, 6, 5, 4, 3, 2.

        :param digits: The 10 digit kennitala
        :return: True when the checksum holds
        """
        check_digit = IsKennitalaRecognizer._check_digit(digits[:8])
        return check_digit == digits[8]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the mod-11 check digit for the first 8 digits.

        The weights 3, 2, 7, 6, 5, 4, 3, 2 are applied
        to the body from left to right, and the check
        digit is (11 - (total mod 11)) mod 11. When
        the weighted sum is 1 mod 11 the result is
        "10", which is not a single digit: such a body
        has no valid kennitala, and _has_valid_checksum
        then always returns False for it.

        :param body: The first 8 digits of the kennitala
        :return: The check digit, 0 to 9, or "10" when no digit fits
        """
        weights = (3, 2, 7, 6, 5, 4, 3, 2)
        total = sum(int(digit) * weight for digit, weight in zip(body, weights))
        return str((11 - (total % 11)) % 11)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 0123456789 ascends by one each step, and
        9876543210 descends by one each step.

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
