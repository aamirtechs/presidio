import calendar
from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class UzPinflRecognizer(PatternRecognizer):
    """
    Recognize Uzbekistan Personal Identification Numbers (PINFL).

    The PINFL, also written PINI, is a 14-digit number issued to every
    individual in Uzbekistan. Its structure is: the first digit encodes
    gender and century of birth (1 male born in the 1800s, 2 female
    born in the 1800s, 3 male born in the 1900s, 4 female born in the
    1900s, 5 male born in the 2000s, 6 female born in the 2000s), the
    next six digits encode the date of birth in DDMMYY form, the next
    three digits are a district and city code (001 to 999), the next
    three digits are a serial number (001 to 999), and the last digit
    is a check digit.

    Checksum note: no authoritative Uzbek source with the check digit
    algorithm and a worked valid example was found, so this recognizer
    implements no checksum and deliberately does not promote matches
    on an unverified one. Invalidation is based on structural sanity
    checks instead: the first digit must be 1 to 6, the DDMMYY date
    must be a real calendar day in the century implied by the first
    digit and must not be in the future, the district code and the
    serial number must not be 000, and trivial numbers such as
    all-same-digit or fully sequential runs are rejected. This follows
    the approach of the Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh NID, Sri Lanka NIC,
    Ghana Card, Kazakhstan IIN and Ukraine RNOKPP recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "uz"

    PATTERNS = [
        Pattern(
            "UZ_PINFL",
            r"(?<![A-Za-z0-9])\d{14}(?![A-Za-z0-9])",
            0.5,
        ),
    ]

    CONTEXT = [
        "uzbekistan",
        "uzbek",
        "pinfl",
        "pini",
        "personal identification number",
        "individual identification number",
        "national identification",
        "uzbekistani",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "UZ_PINFL",
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
        Check if the pattern text cannot be a UZ_PINFL entity.

        No verified checksum exists for this recognizer (see the class
        docstring), so invalidation is based on sanity checks: the
        value must be 14 digits, the first digit must be 1 to 6, the
        DDMMYY date must be a real calendar day in the century implied
        by the first digit and must not be in the future, the district
        code and the serial number must not be 000, and trivial
        numbers such as all-same-digit or fully sequential digit runs
        are rejected.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        if len(pattern_text) != 14 or not pattern_text.isdigit():
            return True
        digits = pattern_text
        if len(set(digits)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(digits):
            # a real issued PINFL is never one fully sequential run
            return True
        if int(digits[0]) not in (1, 2, 3, 4, 5, 6):
            return True
        if digits[7:10] == "000":
            return True
        if digits[10:13] == "000":
            return True
        if not self._valid_birth_date(digits):
            return True
        return False

    @staticmethod
    def _valid_birth_date(digits: str) -> bool:
        """
        Check the DDMMYY date against the century of the 1st digit.

        Century mapping: 1 and 2 mean the 1800s, 3 and 4 the 1900s,
        5 and 6 the 2000s. The date must be a real calendar day
        (leap-aware) and must not be in the future.

        :param digits: 14-digit PINFL candidate
        :return: True if the encoded date is real and not in the future
        """
        century_map = {1: 1800, 2: 1800, 3: 1900, 4: 1900, 5: 2000, 6: 2000}
        century = century_map.get(int(digits[0]))
        if century is None:
            return False
        day = int(digits[1:3])
        month = int(digits[3:5])
        year = century + int(digits[5:7])
        if month < 1 or month > 12:
            return False
        if day < 1 or day > calendar.monthrange(year, month)[1]:
            return False
        if date(year, month, day) > date.today():
            return False
        return True

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 12345678901234 ascends by one each step, and
        43210987654321 descends by one each step.

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
