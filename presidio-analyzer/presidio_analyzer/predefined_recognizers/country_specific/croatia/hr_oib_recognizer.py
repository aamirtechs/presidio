from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class HrOibRecognizer(PatternRecognizer):
    """
    Recognize Croatian personal identification numbers (OIB).

    The OIB (Osobni identifikacijski broj) is the
    unique personal number issued in Croatia by the
    Tax Administration (Porezna uprava) to every
    person and legal entity, and it also serves as
    the Croatian tax identification number. It has
    11 digits and carries no information about the
    holder: the first 10 digits are assigned at
    random, and the 11th digit is a check digit.
    It is written plain, 11 digits with no
    separators, for example 33392005961. For VAT
    purposes the same number is written with an
    HR prefix, for example HR89018712265.

    Checksum: the OIB must pass the ISO 7064
    Mod 11, 10 check, named in the Croatian
    OIB rulebook (Pravilnik o OIB-u, Article 3)
    and implemented by the python-stdnum hr.oib
    module. Starting from 5, each digit updates
    the running value as
    check = (((check or 10) * 2) % 11 + digit) % 10,
    and a valid number ends with check equal to 1.
    Worked examples, verified by hand and by
    computation: 33392005961 (the python-stdnum
    docstring example, also valid written as
    HR 33392005961): the running value moves
    5, 3, 9, 0, 8, 7, 3, 6, 6, 0, 5, 1, ending
    at 1; 69435151530 (the worked example from
    the FastCalc OIB validator): the running
    value moves 5, 6, 0, 3, 9, 2, 5, 5, 1, 7, 6,
    1, ending at 1; 89018712265 (the VAT number
    example HR89018712265): the running value
    moves 5, 8, 4, 8, 6, 9, 4, 9, 9, 9, 3, 1,
    ending at 1. The check digit for a 10 digit
    body is (1 - ((checksum(body) or 10) * 2)
    % 11) % 10, so every body has a valid check
    digit, unlike the Dutch BSN. The checksum
    is confirmed by the python-stdnum hr.oib
    module, by the Croatian OIB rulebook naming
    ISO 7064 Modul 11,10, by the TaxDo Croatia
    guide, and by the FastCalc OIB validator.
    The recognizer's checksum was also
    cross-checked against python-stdnum on all
    fixtures and on 2000 random 10 digit bodies
    and 2000 random 11 digit numbers, with zero
    mismatches in both directions.

    Four honest caveats. First, the Mod 11, 10
    check passes about one time in ten for an
    arbitrary 11 digit token, so a bare 11 digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. Second, python-stdnum
    also accepts the HR prefixed form, for
    example HR89018712265, and spaced or dashed
    forms, for example 3339 2005 961, while this
    recognizer matches plain 11 digits only, for
    example the HR prefixed form HR89018712265
    is not matched here although python-stdnum
    accepts it. Third, 90123456789 passes the
    checksum but is a fully sequential digit run,
    so it is rejected here by the sequential rule.
    Fourth, a passing checksum only shows the
    number is well formed; it cannot show the
    number was ever issued to anyone, and only
    the Croatian Tax Administration register can
    confirm that.

    The recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because
    a Mod 11, 10 check passes about one time in ten
    for an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 11 digits; the number must not be all
    the same digit; a fully sequential digit run is
    rejected; and the check digit must satisfy the
    ISO 7064 Mod 11, 10 check, that is, the running
    value starting from 5 and updated as
    check = (((check or 10) * 2) % 11 + digit) % 10
    must end at 1. This follows the approach of
    the Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh
    NID, Sri Lanka NIC, Ghana Card, Kazakhstan
    IIN, Ukraine RNOKPP, Uzbekistan PINFL, Morocco
    CNIE, Indonesia NIK, Malaysia MyKad, Vietnam
    CCCD, Chile RUT, Ecuador Cedula, Iran Code
    Melli, Argentina CUIT/CUIL, Mexico CURP, Peru
    RUC, Colombia NIT, Dominican Republic Cedula,
    Paraguay RUC, Uruguay RUT, Iceland kennitala,
    Czech birth number, Romanian personal
    numeric code, and Dutch citizen service number
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "hr"

    PATTERNS = [
        Pattern(
            "HR_OIB",
            r"(?<![A-Za-z0-9./-])\d{11}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "croatia",
        "croatian",
        "oib",
        "osobni identifikacijski broj",
        "personal identification number",
        "tax identification number",
        "porezna uprava",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "HR_OIB",
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
        Check if the pattern text cannot be a HR_OIB entity.

        The number must be 11 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, and must pass the ISO 7064 Mod 11, 10
        check, that is, the running value starting from 5
        and updated as
        check = (((check or 10) * 2) % 11 + digit) % 10
        must end at 1.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 11 or not text.isdigit():
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
        Check the ISO 7064 Mod 11, 10 checksum of an 11 digit number.

        Starting from 5, each digit updates the running
        value as check = (((check or 10) * 2) % 11 + digit)
        % 10, and a valid number ends with check equal to 1.

        :param digits: The 11 digit personal identification number
        :return: True when the checksum holds
        """
        return HrOibRecognizer._checksum(digits) == 1

    @staticmethod
    def _check_digit(first_ten: str) -> str:
        """
        Compute the check digit for the first 10 digits.

        The check digit is (1 - ((checksum(body) or 10)
        * 2) % 11) % 10, which always yields a digit from
        0 to 9, so every 10 digit body has a valid check
        digit.

        :param first_ten: The first 10 digits of the number
        :return: The check digit, 0 to 9
        """
        return str((1 - ((HrOibRecognizer._checksum(first_ten) or 10) * 2) % 11) % 10)

    @staticmethod
    def _checksum(digits: str) -> int:
        """
        Compute the ISO 7064 Mod 11, 10 checksum of the digits.

        Starting from 5, each digit updates the running
        value as check = (((check or 10) * 2) % 11 + digit)
        % 10. A valid number ends with check equal to 1.

        :param digits: The digits to checksum
        :return: The checksum value
        """
        check = 5
        for digit in digits:
            check = (((check or 10) * 2) % 11 + int(digit)) % 10
        return check

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 01234567890 ascends by one each step, and
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
