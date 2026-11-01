from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class VnCccdRecognizer(PatternRecognizer):
    """
    Recognize Vietnam citizen identification numbers (CCCD numbers).

    The number printed on the Vietnam identity card (Can cuoc,
    formerly Can cuoc cong dan, CCCD) is a 12-digit number with
    four parts, written with or without hyphens:

    - Digits 1 to 3: the code for the province, centrally
      governed city, or country where the citizen registered
      their birth, for example 001 for Hanoi and 079 for
      Ho Chi Minh City.
    - Digit 4: the gender and birth century code: 0 for a man
      and 1 for a woman born in the 20th century (1901 to 2000),
      2 for a man and 3 for a woman born in the 21st century
      (2001 to 2100).
    - Digits 5 and 6: the last two digits of the birth year.
    - Digits 7 to 12: a random serial number from 000001 to
      999999.

    This structure is set out in Decree 137/2015/ND-CP and in
    the Vietnam identity card structure summary, and the
    gender and century mapping is consistent across the
    published breakdowns.

    Checksum note: the CCCD number has no checksum. No
    published source describes a check digit or a check
    algorithm for it: the last six digits are a random serial,
    not a check value, so this recognizer implements no
    checksum and deliberately does not promote matches on an
    unverified one. Invalidation is based on structural
    sanity checks instead: the value must be exactly 12
    digits once hyphens are removed, the province code must
    not be 000, the gender and century digit must be 0, 1, 2
    or 3, the decoded birth year must not be in the future,
    the serial part must not be 000000, and trivial numbers
    such as all-same-digit or fully sequential digit runs
    are rejected. The full province code list is deliberately
    not enforced: province codes were reorganized in the
    2025 mergers, so a pinned list would reject valid new
    codes. The older 9-digit CMND format is not covered:
    it was abolished on 1 January 2025.
    This follows the approach of the Pakistani CNIC, Saudi
    national ID, UAE Emirates ID, Egyptian national ID,
    Bangladesh NID, Sri Lanka NIC, Ghana Card, Kazakhstan IIN,
    Ukraine RNOKPP, Uzbekistan PINFL, Morocco CNIE, Indonesia
    NIK and Malaysia MyKad recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "vn"

    PATTERNS = [
        Pattern(
            "VN_CCCD",
            r"(?<![A-Za-z0-9-])\d{3}-?\d{1}-?\d{2}-?\d{6}(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "vietnam",
        "vietnamese",
        "cccd",
        "can cuoc",
        "can cuoc cong dan",
        "citizen identity card",
        "identity card",
        "personal identification number",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "VN_CCCD",
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
        Check if the pattern text cannot be an VN_CCCD entity.

        No checksum exists for this recognizer (see the class
        docstring), so invalidation is based on sanity checks: the
        value must be exactly 12 digits once hyphens are removed,
        the province code must not be 000, the gender and century
        digit must be 0, 1, 2 or 3, the decoded birth year must
        not be in the future, the serial part must not be
        000000, and trivial numbers such as all-same-digit or
        fully sequential digit runs are rejected.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) != 12 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued CCCD number is never one fully sequential run
            return True
        if text[0:3] == "000":
            return True
        if text[3] not in "0123":
            return True
        if text[6:] == "000000":
            return True
        return self._birth_year_is_invalid(text[3], text[4:6])

    @staticmethod
    def _birth_year_is_invalid(gender_century_digit: str, year_part: str) -> bool:
        """
        Check the birth year decoded from the gender and century
        digit plus the two-digit year part.


        Digit 0 or 1 maps to the 1900s, digit 2 or 3 maps to
        the 2000s. The decoded year must not be in the future.

        :param gender_century_digit: Digit 4 of the number
        :param year_part: Digits 5 and 6 of the number
        :return: True if the decoded year cannot be a real past birth year
        """
        if gender_century_digit in "01":
            year = 1900 + int(year_part)
        else:
            year = 2000 + int(year_part)
        return year > date.today().year

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 123456789012 ascends by one each step, and
        654321098765 descends by one each step.

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
