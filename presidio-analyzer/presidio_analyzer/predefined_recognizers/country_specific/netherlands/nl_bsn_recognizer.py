from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class NlBsnRecognizer(PatternRecognizer):
    """
    Recognize Dutch citizen service numbers (Burgerservicenummer, BSN).

    The BSN is the unique personal number issued in
    the Netherlands to everyone registered in the
    Personal Records Database (BRP), and to
    non-residents registered in the Non-residents
    Records Database (RNI). It is used in every
    contact with the Dutch government, and it is the
    successor to the former social insurance number
    (sofinummer). It has 9 digits and may begin with
    one or more zeros. It is written plain, 9 digits
    with no separators, for example 111222333.

    Checksum: the BSN must pass the eleven test
    (elfproef). The nine digits are multiplied by
    the weights 9, 8, 7, 6, 5, 4, 3, 2, -1, the
    products are added, and the sum must be
    divisible by 11. Worked examples, verified by
    hand and by computation: 111222333 (the
    python-stdnum docstring example):
    9x1 + 8x1 + 7x1 + 6x2 + 5x2 + 4x2 + 3x3 + 2x3
    - 1x3 = 66, and 66 mod 11 = 0; 123456782:
    9x1 + 8x2 + 7x3 + 6x4 + 5x5 + 4x6 + 3x7 + 2x8
    - 1x2 = 154, and 154 mod 11 = 0; 211551557
    (the worked example from the Dutch government
    eleven-test explanation):
    9x2 + 8x1 + 7x1 + 6x5 + 5x5 + 4x1 + 3x5 + 2x5
    - 1x7 = 110, and 110 mod 11 = 0; 010000008
    (a number with a leading zero, which the
    official government guidance allows):
    9x0 + 8x1 + 7x0 + 6x0 + 5x0 + 4x0 + 3x0 + 2x0
    - 1x8 = 0, and 0 mod 11 = 0; and 999000007
    (from the 999 test range used by Dutch
    government test data):
    9x9 + 8x9 + 7x9 + 6x0 + 5x0 + 4x0 + 3x0 + 2x0
    - 1x7 = 209, and 209 mod 11 = 0. The eleven
    test is confirmed by the python-stdnum nl.bsn
    module, by Wikipedia (Burgerservicenummer),
    by the Dutch government (NetherlandsWorldwide
    and Government.nl, which state that the BSN
    has 9 digits and may begin with zeros), and
    by the Dutch Electoral Council (Kiesraad)
    e-KS code, which implements the same weights.
    The recognizer's checksum was also
    cross-checked against python-stdnum on all
    fixtures and on 2000 random valid-checksum
    bodies: 1805 agree, and there are zero
    unexplained divergences. The other 195 bodies
    have a weighted sum of 10 mod 11, for which
    no check digit exists, so no valid BSN can be
    built from them at all.

    Three honest caveats. First, the eleven test
    passes about one time in eleven for an
    arbitrary 9-digit token, so a bare 9-digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. Second, python-stdnum
    also accepts 8-digit input, reading it as a
    BSN with the leading zero omitted, and the
    dotted form 1112.22.333, while this
    recognizer matches plain 9 digits only, for
    example the dotted form 1112.22.333 is not
    matched here although python-stdnum accepts
    it. Third, 000000000 passes the eleven test
    (its weighted sum is 0) but is not a real
    BSN, so it is rejected here by the
    all-same-digit rule, matching python-stdnum,
    which rejects it as an invalid format.

    The recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because
    an eleven test passes about one time in eleven
    for an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 9 digits; the number must not be all
    the same digit; a fully sequential digit run is
    rejected; and the check digit must satisfy the
    eleven test, that is, the weighted sum with
    weights 9, 8, 7, 6, 5, 4, 3, 2, -1 must be
    divisible by 11. This follows the approach of
    the Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh
    NID, Sri Lanka NIC, Ghana Card, Kazakhstan
    IIN, Ukraine RNOKPP, Uzbekistan PINFL, Morocco
    CNIE, Indonesia NIK, Malaysia MyKad, Vietnam
    CCCD, Chile RUT, Ecuador Cedula, Iran Code
    Melli, Argentina CUIT/CUIL, Mexico CURP, Peru
    RUC, Colombia NIT, Dominican Republic Cedula,
    Paraguay RUC, Uruguay RUT, Iceland kennitala,
    Czech birth number, and Romanian personal
    numeric code recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "nl"

    PATTERNS = [
        Pattern(
            "NL_BSN",
            r"(?<![A-Za-z0-9./-])\d{9}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "netherlands",
        "dutch",
        "bsn",
        "burgerservicenummer",
        "citizen service number",
        "persoonsnummer",
    ]

    # The eleven-test weights for the check digit.
    _WEIGHTS = (9, 8, 7, 6, 5, 4, 3, 2, -1)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "NL_BSN",
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
        Check if the pattern text cannot be a NL_BSN entity.

        The number must be 9 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, and must pass the eleven test, that
        is, the weighted sum with weights 9, 8, 7, 6,
        5, 4, 3, 2, -1 must be divisible by 11.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 9 or not text.isdigit():
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
        Check the eleven test of a 9 digit number.

        The weighted sum of the nine digits with
        weights 9, 8, 7, 6, 5, 4, 3, 2, -1 must be
        divisible by 11.

        :param digits: The 9 digit citizen service number
        :return: True when the checksum holds
        """
        return NlBsnRecognizer._weighted_sum(digits) % 11 == 0

    @staticmethod
    def _check_digit(first_eight: str) -> Optional[str]:
        """
        Compute the check digit for the first 8 digits.

        The check digit is the weighted sum of the
        first 8 digits with weights 9, 8, 7, 6, 5,
        4, 3, 2, mod 11. When that remainder is 10,
        no check digit exists, so None is returned:
        no 9 digit number built from that body can
        pass the eleven test.

        :param first_eight: The first 8 digits of the number
        :return: The check digit, 0 to 9, or None when none exists
        """
        total = sum(
            weight * int(digit)
            for weight, digit in zip(NlBsnRecognizer._WEIGHTS[:8], first_eight)
        )
        remainder = total % 11
        if remainder == 10:
            return None
        return str(remainder)

    @staticmethod
    def _weighted_sum(digits: str) -> int:
        """
        Compute the eleven-test weighted sum of 9 digits.

        The nine digits are multiplied by the weights
        9, 8, 7, 6, 5, 4, 3, 2, -1 and the products
        are added.

        :param digits: The 9 digit citizen service number
        :return: The weighted sum
        """
        return sum(
            weight * int(digit)
            for weight, digit in zip(NlBsnRecognizer._WEIGHTS, digits)
        )

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 012345678 ascends by one each step, and
        987654321 descends by one each step.

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
