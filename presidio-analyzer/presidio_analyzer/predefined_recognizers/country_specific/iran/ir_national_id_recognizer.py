from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class IrNationalIdRecognizer(PatternRecognizer):
    """
    Recognize Iran Code Melli national code numbers.

    The Code Melli is the 10 digit national code issued
    to Iranian citizens and printed on the national
    identity card. It has three parts:

    - The first three digits are a geographic or city
      code.
    - The next six digits are a sequential registration
      number.
    - The tenth digit is a check digit.

    Checksum: the check digit uses a mod-11 algorithm.
    The first nine digits are multiplied from left to
    right by weights 10 down to 2. The weighted sum is
    taken modulo 11. A remainder below 2 means the check
    digit equals the remainder; otherwise the check digit
    equals 11 minus the remainder. The published worked
    example is 472-171992-2: 4*10 + 7*9 + 2*8 + 1*7 +
    7*6 + 1*5 + 9*4 + 9*3 + 2*2 = 240, 240 mod 11 = 9,
    and 11 - 9 = 2, which matches the check digit. The
    recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because a
    mod-11 check passes about one time in eleven for an
    arbitrary same-shape token.

    Invalidation is based on: the number must be exactly
    10 digits after removing the hyphens; the first three
    digits must not be 000; digits 4 to 9 must not be
    000000; the number must not be all the same digit;
    a fully sequential digit run is rejected; and the
    mod-11 checksum must hold. No specific city-code list
    is enforced, because no authoritative complete list
    was pinned for this recognizer. This follows the
    approach of the Pakistani CNIC, Saudi national ID,
    UAE Emirates ID, Egyptian national ID, Bangladesh
    NID, Sri Lanka NIC, Ghana Card, Kazakhstan IIN,
    Ukraine RNOKPP, Uzbekistan PINFL, Morocco CNIE,
    Indonesia NIK, Malaysia MyKad, Vietnam CCCD, and Chile
    RUT recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ir"

    PATTERNS = [
        Pattern(
            "IR_NATIONAL_ID",
            r"(?<![A-Za-z0-9.-])(?:\d{10}|\d{3}-\d{6}-\d)(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "iran",
        "iranian",
        "code melli",
        "melli code",
        "national code",
        "national id",
        "identity card",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "IR_NATIONAL_ID",
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
        Check if the pattern text cannot be an IR_NATIONAL_ID entity.

        The number must be 10 digits after removing the
        hyphens, the first three digits must not be 000,
        digits 4 to 9 must not be 000000, the number must
        not be all the same digit, a fully sequential
        digit run is rejected, and the mod-11 checksum
        must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) != 10 or not text.isdigit():
            return True
        if text[:3] == "000":
            # a geographic code of all zeros is never issued
            return True
        if text[3:9] == "000000":
            # a serial of all zeros is never issued
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued code is never one fully sequential run
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the mod-11 checksum of a 10 digit Code Melli.

        The first nine digits are weighted 10 down to 2.
        The remainder of the weighted sum modulo 11 is
        below 2 when the check digit equals the remainder,
        and otherwise the check digit equals 11 minus
        the remainder.

        :param digits: The 10 digit Code Melli, hyphens removed
        :return: True when the checksum holds
        """
        return IrNationalIdRecognizer._check_digit(digits[:9]) == digits[9]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the check digit for the first 9 digits.

        The weighted sum uses weights 10 down to 2. A
        remainder below 2 gives that remainder as the
        check digit; otherwise the check digit is 11
        minus the remainder.

        :param body: The first 9 digits of the Code Melli
        :return: The check digit, 0 to 9
        """
        total = sum(int(digit) * (10 - index) for index, digit in enumerate(body))
        remainder = total % 11
        if remainder < 2:
            return str(remainder)
        return str(11 - remainder)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 1234567890 ascends by one each step, and
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
