from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class MaCnieRecognizer(PatternRecognizer):
    """
    Recognize Morocco National Identity Card numbers (CNIE / CIN).

    The number printed on the Carte Nationale d'Identite Electronique
    (CNIE) is made of a region code of 1 to 3 letters (the issuing
    office or province where the card was first issued), followed by
    a 6-digit serial number, followed by a 2-digit control number:
    1 to 3 letters + 8 digits in total. Older cards carry a single
    province letter, newer cards carry two letters for the issuing
    office, and the registry structure allows up to three letters.

    Checksum note: published sources describe the last two digits as
    a Luhn control number, but no authoritative Moroccan source with
    a worked valid example was found to verify the Luhn variant, so
    this recognizer implements no checksum and deliberately does not
    promote matches on an unverified one. Invalidation is based on
    structural sanity checks instead: the value must be 1 to 3
    letters followed by exactly 8 digits, the 6-digit serial part
    must not be 000000, and trivial numbers such as all-same-digit
    or fully sequential digit runs are rejected. This follows the
    approach of the Pakistani CNIC, Saudi national ID, UAE Emirates
    ID, Egyptian national ID, Bangladesh NID, Sri Lanka NIC, Ghana
    Card, Kazakhstan IIN, Ukraine RNOKPP and Uzbekistan PINFL
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ma"

    PATTERNS = [
        Pattern(
            "MA_CNIE",
            r"(?<![A-Za-z0-9])[A-Z]{1,3}\d{8}(?![A-Za-z0-9])",
            0.5,
        ),
    ]

    CONTEXT = [
        "morocco",
        "moroccan",
        "cnie",
        "cin",
        "carte nationale",
        "national identity card",
        "identity card",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "MA_CNIE",
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
        Check if the pattern text cannot be a MA_CNIE entity.

        No verified checksum exists for this recognizer (see the class
        docstring), so invalidation is based on sanity checks: the
        value must be 1 to 3 letters followed by exactly 8 digits,
        the 6-digit serial part must not be 000000, and trivial
        numbers such as all-same-digit or fully sequential digit runs
        are rejected.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.upper()
        letters = ""
        digits = ""
        for char in text:
            if char.isalpha():
                if digits:
                    return True
                letters += char
            elif char.isdigit():
                digits += char
            else:
                return True
        if not 1 <= len(letters) <= 3 or len(digits) != 8:
            return True
        if len(set(digits)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(digits):
            # a real issued CNIE is never one fully sequential run
            return True
        if digits[:6] == "000000":
            return True
        return False

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 12345678 ascends by one each step, and 87654321
        descends by one each step.

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
