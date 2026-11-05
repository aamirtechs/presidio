from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class ArCuitRecognizer(PatternRecognizer):
    """
    Recognize Argentina CUIT/CUIL tax identifier numbers.

    The CUIT (Clave Unica de Identificacion Tributaria)
    and CUIL (Clave Unica de Identificacion Laboral)
    are the 11 digit tax identifiers issued in
    Argentina. They have three parts:

    - The first two digits are a type prefix: 20, 23,
      24, and 27 for people, and 30, 33, and 34 for
      companies.
    - The next eight digits are the identifier.
    - The eleventh digit is a check digit.

    Checksum: the check digit uses a mod-11 algorithm.
    The first ten digits are multiplied from left to
    right by weights 5, 4, 3, 2, 7, 6, 5, 4, 3, 2. The
    weighted sum is taken modulo 11. The check digit
    is 11 minus the remainder, except that a remainder
    of 0 gives a check digit of 0, and a remainder of 1
    would give 10, so that combination is never issued
    and is rejected rather than mapped to another
    prefix. The worked examples are 20-12345678-6:
    2*5 + 0*4 + 1*3 + 2*2 + 3*7 + 4*6 + 5*5 + 6*4 +
    7*3 + 8*2 = 148, 148 mod 11 = 5, and 11 - 5 = 6;
    and 30-12345678-1: 3*5 + 0*4 + 1*3 + 2*2 + 3*7 +
    4*6 + 5*5 + 6*4 + 7*3 + 8*2 = 153, 153 mod 11 = 10,
    and 11 - 10 = 1. The recognizer validates the
    checksum in invalidate_result and keeps the pattern
    score at 0.5: it does not promote a match to 1.0,
    because a mod-11 check passes about one time in
    eleven for an arbitrary same-shape token.

    Invalidation is based on: the number must be exactly
    11 digits after removing the hyphens; the first two
    digits must be one of 20, 23, 24, 27, 30, 33, or 34
    (25 and 26 are not issued and are rejected); the
    number must not be all the same digit; a fully
    sequential digit run is rejected; and the mod-11
    checksum must hold, including rejection of the
    remainder-1 combinations that are never issued.
    This follows the approach of the Pakistani CNIC,
    Saudi national ID, UAE Emirates ID, Egyptian
    national ID, Bangladesh NID, Sri Lanka NIC, Ghana
    Card, Kazakhstan IIN, Ukraine RNOKPP, Uzbekistan
    PINFL, Morocco CNIE, Indonesia NIK, Malaysia
    MyKad, Vietnam CCCD, Chile RUT, and Iran Code
    Melli recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ar"

    PATTERNS = [
        Pattern(
            "AR_CUIT",
            r"(?<![A-Za-z0-9.-])(?:\d{11}|\d{2}-\d{8}-\d)(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "argentina",
        "argentine",
        "cuit",
        "cuil",
        "clave unica",
        "tax id",
        "afip",
        "arca",
    ]

    ALLOWED_PREFIXES = {"20", "23", "24", "27", "30", "33", "34"}

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "AR_CUIT",
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
        Check if the pattern text cannot be an AR_CUIT entity.

        The number must be 11 digits after removing the
        hyphens, the first two digits must be an allowed
        prefix, the number must not be all the same
        digit, a fully sequential digit run is rejected,
        and the mod-11 checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) != 11 or not text.isdigit():
            return True
        if text[:2] not in self.ALLOWED_PREFIXES:
            # 25 and 26 are not issued prefixes, and neither
            # is any other prefix outside the allowed set
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued number is never one fully sequential run
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the mod-11 checksum of an 11 digit CUIT/CUIL.

        The first ten digits are weighted 5, 4, 3, 2, 7,
        6, 5, 4, 3, 2. A remainder of 1 means the
        combination is never issued, so no check digit
        can validate it.

        :param digits: The 11 digit CUIT/CUIL, hyphens removed
        :return: True when the checksum holds
        """
        check_digit = ArCuitRecognizer._check_digit(digits[:10])
        return check_digit is not None and check_digit == digits[10]

    @staticmethod
    def _check_digit(body: str) -> Optional[str]:
        """
        Compute the check digit for the first 10 digits.

        The weighted sum uses weights 5, 4, 3, 2, 7, 6,
        5, 4, 3, 2. A remainder of 0 gives a check digit
        of 0; any other remainder gives 11 minus the
        remainder, except a remainder of 1, which would
        give 10 and is never issued, so None is returned.

        :param body: The first 10 digits of the CUIT/CUIL
        :return: The check digit, 0 to 9, or None when never issued
        """
        weights = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
        total = sum(int(digit) * weight for digit, weight in zip(body, weights))
        remainder = total % 11
        if remainder == 0:
            return "0"
        if remainder == 1:
            return None
        return str(11 - remainder)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 12345678901 ascends by one each step, and
        98765432109 descends by one each step.

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
