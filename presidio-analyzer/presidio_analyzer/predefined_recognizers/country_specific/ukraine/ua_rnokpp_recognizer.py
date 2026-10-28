from datetime import date, timedelta
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class UaRnokppRecognizer(PatternRecognizer):
    """
    Recognize Ukraine individual taxpayer numbers (RNOKPP).

    The RNOKPP (registration number of the taxpayer's account card,
    also called the individual tax number, INN or IPN) is a 10-digit
    number issued by the State Tax Service of Ukraine to every
    individual taxpayer. Its structure is: the first five digits
    encode the holder's date of birth as the number of days from
    31 December 1899 (so 1 encodes 1900-01-01), the next three digits
    are a serial number, the ninth digit encodes sex (odd for men,
    even for women), and the tenth digit is a check digit.

    Checksum note: published secondary sources describe a weighted
    modulo 11 check digit algorithm for the RNOKPP, with a second
    weight cycle when the first remainder is 10 or more, but no
    authoritative Ukrainian source with the weights and a worked
    valid example was found, so this recognizer implements no
    checksum and deliberately does not promote matches on an
    unverified one. Invalidation is based on structural sanity
    checks instead: the first five digits must decode to a real
    date on or after 1900-01-01 and not in the future, and trivial
    numbers such as all-same-digit or fully sequential runs are
    rejected. This follows the approach of the Pakistani CNIC,
    Saudi national ID, UAE Emirates ID, Egyptian national ID,
    Bangladesh NID, Sri Lanka NIC, Ghana Card and Kazakhstan IIN
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ua"

    PATTERNS = [
        Pattern(
            "UA_RNOKPP",
            r"(?<![A-Za-z0-9])\d{10}(?![A-Za-z0-9])",
            0.5,
        ),
    ]

    CONTEXT = [
        "ukraine",
        "ukrainian",
        "rnokpp",
        "tax identification",
        "taxpayer",
        "individual tax number",
        "identification number",
        "inn",
        "ipn",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "UA_RNOKPP",
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
        Check if the pattern text cannot be a UA_RNOKPP entity.

        No verified checksum exists for this recognizer (see the class
        docstring), so invalidation is based on sanity checks: the
        value must be 10 digits, the first five digits must decode
        to a real date on or after 1900-01-01 and not in the future,
        and trivial numbers such as all-same-digit or fully
        sequential digit runs are rejected.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        if len(pattern_text) != 10 or not pattern_text.isdigit():
            return True
        digits = pattern_text
        if len(set(digits)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(digits):
            # a real issued RNOKPP is never one fully sequential run
            return True
        if not self._valid_birth_date(digits):
            return True
        return False

    @staticmethod
    def _valid_birth_date(digits: str) -> bool:
        """
        Check the date encoded in the first five digits.

        The first five digits count days from 31 December 1899, so
        the decoded date is that epoch plus the five-digit count.
        It must fall on or after 1900-01-01 and must not be in the
        future. Because the encoding is a day count rather than a
        calendar field, every in-range count decodes to a real date;
        the check is therefore a range check, not a month/day check.

        :param digits: 10-digit RNOKPP candidate
        :return: True if the encoded date is in range and not future
        """
        day_count = int(digits[0:5])
        if day_count < 1:
            return False
        birth_date = date(1899, 12, 31) + timedelta(days=day_count)
        if birth_date < date(1900, 1, 1):
            return False
        if birth_date > date.today():
            return False
        return True

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
