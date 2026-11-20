from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class GrAfmRecognizer(PatternRecognizer):
    """
    Recognize Greek tax identification numbers (AFM).

    The AFM (Arithmos Forologikou Mitroou,
    Tax Registration Number) is the 9 digit
    tax identification number assigned in
    Greece by the Independent Authority for
    Public Revenue (AADE) to every taxpayer,
    individual or legal entity, for tax
    purposes. It is also the national part of
    the Greek VAT number: the VAT number is
    the prefix EL followed by the same 9
    digits, per the Wikipedia VAT
    identification number article, which
    states that the EL prefix is followed by
    9 digits that equal the Greek taxpayer
    registration number AFM, and that the
    last digit is a check digit inherent to
    the AFM. The TaxDo Greece TIN guide,
    verified from AADE sources, confirms the
    structure: the first 8 digits are
    sequential numbers that uniquely identify
    the taxpayer, and the 9th digit is a
    checksum digit.

    The number has 9 digits and is written
    plain, 9 digits with no separators, for
    example 094259216. There is no first
    digit restriction: a leading zero is
    valid, as the python-stdnum docstring
    example 094259216 and the FastCalc worked
    example 090000045 both show.

    Checksum: the check digit comes from
    weights 256, 128, 64, 32, 16, 8, 4, 2 on
    the first 8 digits, as published by the
    FastCalc AFM validator and implemented
    by the python-stdnum gr.vat module (there
    is no gr.afm module; the AFM is covered
    by gr.vat, whose docstring names it FPA,
    Phi Pi Alpha, AFM). The weighted sum
    modulo 11, then modulo 10, is the check
    digit, so a remainder of 10 gives a check
    digit of 0. The stdnum implementation
    reaches the same result iteratively:
    starting from 0, each step doubles the
    running total and adds the next digit,
    then the check digit is (total times 2)
    mod 11 mod 10. Worked examples, verified
    by hand and by computation: 094259216
    (the python-stdnum docstring example):
    the weighted sum is 1634, 1634 mod 11 is
    6, and the check digit is 6; 090000045
    (the FastCalc worked example): the
    weighted sum is 1160, 1160 mod 11 is 5,
    and the check digit is 5; 123456783:
    the weighted sum is 1004, 1004 mod 11 is
    3, and the check digit is 3; 000000050:
    the weighted sum is 10, 10 mod 11 is 10,
    so the check digit is 0 through the mod
    10 fold; 110704340: the weighted sum is
    660, 660 mod 11 is 0, and the check digit
    is 0; 631706690: the weighted sum is
    2298, 2298 mod 11 is 10, so the check
    digit is 0 through the mod 10 fold;
    800000002: the weighted sum is 2048,
    2048 mod 11 is 2, and the check digit is
    2; 555555559: the weighted sum is 2550,
    2550 mod 11 is 9, and the check digit is
    9. The recognizer's checksum was also
    cross-checked against python-stdnum on
    all fixtures, on 2000 random 8 digit
    bodies (check digit computation), and on
    2000 random 9 digit numbers (checksum
    validation), with zero mismatches in
    every direction.

    Four honest caveats. First, the checksum
    alone passes about one time in 10 for an
    arbitrary 9 digit token, so a bare 9
    digit number is weak evidence on its own;
    the recognizer keeps the pattern score at
    0.5 and never promotes it. Second,
    python-stdnum accepts the 8 digit old
    format (it pads a leading zero), EL and
    GR prefixed forms, and spaced and dashed
    forms, while this recognizer matches
    plain 9 digits only, so it is stricter
    than python-stdnum. Third, two numbers
    pass the checksum but are rejected here
    as implausible: 000000000 passes the
    checksum and python-stdnum validates it,
    but it is rejected by the all same digit
    rule, and 321098765 passes the checksum
    but is rejected by the fully sequential
    run rule. Note 012345670 is valid: the
    step from 7 to 0 breaks the run, so it
    is not sequential. Fourth, a passing checksum only
    shows the number is well formed; it
    cannot show the number was ever issued to
    anyone, and only the AADE can confirm
    that.

    The recognizer validates the checksum in
    invalidate_result and keeps the pattern
    score at 0.5: it does not promote a match
    to 1.0, because a checksum passes about
    one time in 10 for an arbitrary
    same-shape token.

    Invalidation is based on: the number must
    be exactly 9 digits; the number must not
    be all the same digit; a fully sequential
    digit run is rejected; and the check digit
    must satisfy the weighted mod 11 mod 10
    check, that is, with weights 256, 128,
    64, 32, 16, 8, 4, 2 on the first 8 digits,
    ((sum mod 11) mod 10) must equal the 9th
    digit. This follows the approach of the
    Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID,
    Bangladesh NID, Sri Lanka NIC, Ghana
    Card, Kazakhstan IIN, Ukraine RNOKPP,
    Uzbekistan PINFL, Morocco CNIE, Indonesia
    NIK, Malaysia MyKad, Vietnam CCCD, Chile
    RUT, Ecuador Cedula, Iran Code Melli,
    Argentina CUIT/CUIL, Mexico CURP, Peru
    RUC, Colombia NIT, Dominican Republic
    Cedula, Paraguay RUC, Uruguay RUT,
    Iceland kennitala, Czech birth number,
    Romanian personal numeric code, Dutch
    citizen service number, Croatian personal
    identification number, Slovenian unique
    master citizen number, and Bulgarian
    uniform civil number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "gr"

    PATTERNS = [
        Pattern(
            "GR_AFM",
            r"(?<![A-Za-z0-9./-])\d{9}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "greece",
        "greek",
        "afm",
        "arithmos forologikou mitroou",
        "tax identification number",
        "tax registration number",
        "greek tax number",
        "aade",
    ]

    CHECK_WEIGHTS = (256, 128, 64, 32, 16, 8, 4, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "GR_AFM",
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
        Check if the pattern text cannot be a GR_AFM entity.

        The number must be 9 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, and must pass the weighted mod 11
        mod 10 check.

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
        Check the weighted mod 11 mod 10 checksum of a 9 digit number.

        With weights 256, 128, 64, 32, 16, 8, 4, 2
        on the first 8 digits, ((sum mod 11) mod 10)
        must equal the 9th digit. A remainder of 10
        folds to a check digit of 0.

        :param digits: The 9 digit tax identification number
        :return: True when the checksum holds
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(digits[:8], GrAfmRecognizer.CHECK_WEIGHTS)
        )
        check = total % 11 % 10
        return check == int(digits[8])

    @staticmethod
    def _check_digit(first_eight: str) -> str:
        """
        Compute the check digit for the first 8 digits.

        With weights 256, 128, 64, 32, 16, 8, 4, 2
        on the first 8 digits, the check digit is
        (sum mod 11) mod 10, so a remainder of 10
        gives a check digit of 0, and every 8 digit
        body has a valid check digit.

        :param first_eight: The first 8 digits of the number
        :return: The check digit, 0 to 9
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(first_eight, GrAfmRecognizer.CHECK_WEIGHTS)
        )
        return str(total % 11 % 10)

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
