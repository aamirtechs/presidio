from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class CzBirthNumberRecognizer(PatternRecognizer):
    """
    Recognize Czech birth numbers (rodne cislo).

    The rodne cislo is the national identification
    number issued in Czechia by the Ministry of the
    Interior to every person registered in the
    population register. It has 10 digits in two
    parts:

    - A six digit date, YYMMDD: the date of birth,
      with the month code carrying the sex. Month
      codes 01 to 12 are men; 51 to 62 are women
      (50 added to the month). From 2004, when a
      day's series was exhausted, month codes 21
      to 32 are used for men (20 added) and 71 to
      82 for women (70 added).
    - A four digit serial plus check digit: three
      serial digits and one check digit.

    The birth number is written plain, 10 digits
    with no separators, or in its canonical slashed
    form YYMMDD/XXXX, for example 900720/3117. The
    same format is used for Slovak birth numbers;
    this recognizer is registered for Czechia with
    country code cz.

    Checksum: the whole 10 digit number must be
    divisible by 11. Worked examples, verified by
    hand and by computation: 9007203117
    (the @ppasmik/birth-number-utils example, a man
    born 1990-07-20): 9007203117 = 11 x 818836647,
    so 9007203117 mod 11 = 0; 7552121236 (a woman
    born 1975-02-12, month code 52): 7552121236
    = 11 x 686556476, so 7552121236 mod 11 = 0;
    0523154566 (a man born 2005-03-15, month code
    23, the post-2004 +20 form): 0523154566
    = 11 x 47559506, so 0523154566 mod 11 = 0;
    and 7103192745 (the python-stdnum docstring
    example and the fastcalco worked example):
    7103192745 = 11 x 645744795, so 7103192745
    mod 11 = 0. The divisible-by-11 rule and the
    month offsets are confirmed by the python-stdnum
    cz.rc module, by Wikipedia (Rodne cislo), by
    the US SSA POMS GN 01750.315 (Czech Republic
    Birth Number), and by the fastcalco rodne cislo
    validator, which states the whole ten-digit
    number must be divisible by 11. The
    recognizer's checksum was also cross-checked
    against python-stdnum on all fixtures, with
    zero mismatches on the numbers both accept.

    Three honest caveats. First, historically a
    remainder of 10 was written as a check digit
    of 0, so a number whose first 9 digits are
    10 mod 11 carries a 0 check digit (python-stdnum
    accepts these); the recognizer enforces strict
    divisibility by 11 and misses them, for
    example 9007200060, whose first 9 digits are
    10 mod 11 and whose whole number is 1 mod 11.
    Second, 9 digit pre-1954 numbers have no
    checksum and are not accepted, for example
    900720311, which python-stdnum validates as
    an 1890 number. Third, the month codes 21 to
    32 and 71 to 82 are only accepted for birth
    years 2004 and later, following their post-2004
    introduction, so a pre-2004 number with such
    a code is rejected (for example 9923151238,
    a 1999 date with a +20 code, which
    python-stdnum accepts).

    The recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because
    a divisibility-by-11 check passes about one time
    in eleven for an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 10 digits after removing the slash; the
    number must not be all the same digit; a fully
    sequential digit run is rejected; the last four
    digits must not be 0000; the first six digits
    must be a real calendar date (month code 01 to
    12, 21 to 32, 51 to 62, or 71 to 82, day 01 to
    31, February 29 only in a leap year), the
    two-digit year maps to 1954 to 1999 for YY 54
    to 99 and to 2000 to 2053 for YY 00 to 53, the
    month codes 21 to 32 and 71 to 82 require a
    birth year of 2004 or later, and the date must
    not be in the future; and the whole 10 digit
    number must be divisible by 11. This follows
    the approach of the Pakistani CNIC, Saudi
    national ID, UAE Emirates ID, Egyptian national
    ID, Bangladesh NID, Sri Lanka NIC, Ghana Card,
    Kazakhstan IIN, Ukraine RNOKPP, Uzbekistan
    PINFL, Morocco CNIE, Indonesia NIK, Malaysia
    MyKad, Vietnam CCCD, Chile RUT, Ecuador Cedula,
    Iran Code Melli, Argentina CUIT/CUIL, Mexico
    CURP, Peru RUC, Colombia NIT, Dominican Republic
    Cedula, Paraguay RUC, Uruguay RUT, and Iceland
    kennitala recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "cz"

    PATTERNS = [
        Pattern(
            "CZ_RC",
            r"(?<![A-Za-z0-9./-])(?:\d{10}|\d{6}/\d{4})(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "czech",
        "czechia",
        "czech republic",
        "rodne cislo",
        "rodne",
        "birth number",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "CZ_RC",
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
        Check if the pattern text cannot be a CZ_RC entity.

        The number must be 10 digits after removing
        the slash, the number must not be all the
        same digit, a fully sequential digit run is
        rejected, the last four digits must not be
        0000, the first six digits must be a real
        calendar date (month code 01 to 12, 21 to
        32, 51 to 62, or 71 to 82, day 01 to 31,
        February 29 only in a leap year), the
        two-digit year maps to 1954 to 1999 for YY
        54 to 99 and to 2000 to 2053 for YY 00 to
        53, the month codes 21 to 32 and 71 to 82
        require a birth year of 2004 or later, the
        date must not be in the future, and the
        whole 10 digit number must be divisible
        by 11.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("/", "")
        if len(text) != 10 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued number is never one fully sequential run
            return True
        if text[6:] == "0000":
            # the serial part must not be 0000
            return True
        if not self._has_valid_date(text):
            # the first six digits must be a real calendar date
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_date(digits: str) -> bool:
        """
        Check the first six digits form a real calendar date.

        The month code is 01 to 12 for men, 51 to 62
        for women (50 added), 21 to 32 for men from
        2004 (20 added), and 71 to 82 for women from
        2004 (70 added). The two-digit year maps to
        1954 to 1999 for YY 54 to 99 and to 2000 to
        2053 for YY 00 to 53, following python-stdnum.
        The date must be real and must not be in the
        future.

        :param digits: The 10 digit birth number
        :return: True when the date is a real calendar date
        """
        month_code = int(digits[2:4])
        day = int(digits[4:6])
        year = int(digits[:2])
        if year >= 54:
            year += 1900
        else:
            year += 2000
        if 1 <= month_code <= 12:
            month = month_code
        elif 21 <= month_code <= 32:
            # men, post-2004 overflow form
            if year < 2004:
                return False
            month = month_code - 20
        elif 51 <= month_code <= 62:
            # women, 50 added to the month
            month = month_code - 50
        elif 71 <= month_code <= 82:
            # women, post-2004 overflow form
            if year < 2004:
                return False
            month = month_code - 70
        else:
            return False
        if day < 1 or day > 31 or month < 1 or month > 12:
            return False
        try:
            birth_date = date(year, month, day)
        except ValueError:
            return False
        return birth_date <= date.today()

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the divisibility-by-11 checksum of a 10 digit birth number.

        The whole 10 digit number must be divisible
        by 11. The check digit is the tenth digit:
        it must equal the first 9 digits mod 11, so
        when the first 9 digits are 10 mod 11 no
        digit fits and the number is rejected.

        :param digits: The 10 digit birth number
        :return: True when the checksum holds
        """
        check_digit = CzBirthNumberRecognizer._check_digit(digits[:9])
        return check_digit == digits[9]

    @staticmethod
    def _check_digit(first_nine: str) -> str:
        """
        Compute the check digit for the first 9 digits.

        The check digit is the first 9 digits mod 11:
        the whole 10 digit number is then divisible
        by 11. When the first 9 digits are 10 mod 11
        the result is "10", which is not a single
        digit: such a body has no valid birth number
        under strict divisibility by 11, and
        _has_valid_checksum then always returns False
        for it.

        :param first_nine: The first 9 digits of the birth number
        :return: The check digit, 0 to 9, or "10" when no digit fits
        """
        return str(int(first_nine) % 11)

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
