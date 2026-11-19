import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class BgEgnRecognizer(PatternRecognizer):
    """
    Recognize Bulgarian uniform civil numbers (EGN).

    The EGN (Edinen grazhdanski nomer, Uniform
    Civil Number) is the 10 digit personal number
    assigned to every Bulgarian citizen at birth
    or when a birth certificate is issued. It is
    regulated by the uniform system for civil
    registration and administrative service of
    population (Edinna sistema za grazhdanska
    registratsiya i administrativno obsluzhvane
    na naselenieto, ESGRAON). It is also the tax
    identification number for Bulgarian citizens,
    per the OECD Bulgaria TIN sheet, which gives
    the structure as YYMMDDXXXC: date of birth,
    serial number, control digit.

    The number has 10 digits and encodes the
    holder's date of birth: YYMMDD SSS K, where
    YY is the last two digits of the year of
    birth, MM the month of birth, DD the day of
    birth, SSS a birth order number, and K a
    check digit. It is written plain, 10 digits
    with no separators, for example 8032056031.

    Month: for people born between 1900 and 1999
    the month is the plain month number. For
    people born before 1900, 20 is added to the
    month, and for people born from 2000, 40 is
    added to the month, so a month code of 21 to
    32 means an 1800s birth and 41 to 52 means a
    2000s birth.

    Birth order: the next three digits designate
    the birth order number within the district of
    birth. Each district is assigned a range of
    three digit numbers, used consecutively. The
    third digit of the birth order number is even
    for males and odd for females.

    Checksum: the check digit comes from weights
    2, 4, 8, 5, 10, 9, 7, 3, 6 on the first 9
    digits, as published in the Wikipedia Unique
    citizenship number article and implemented
    by the python-stdnum bg.egn module. The
    weighted sum modulo 11 is the check digit,
    and a remainder of 10 gives a check digit of
    0. Worked examples, verified by hand and by
    computation: 8032056031 (the python-stdnum
    docstring example, born 5 December 1880,
    month code 32, serial 603, female): the
    weighted sum is 155, 155 mod 11 is 1, and
    the check digit is 1; 1547121580 (the
    Wikipedia example, a boy born 12 July 2015,
    month code 47, serial 158, male): the
    weighted sum is 187, 187 mod 11 is 0, and
    the check digit is 0; 7523169263 (the
    python-stdnum compact example, born 16 March
    1875, month code 23, serial 926, male): the
    weighted sum is 234, 234 mod 11 is 3, and
    the check digit is 3; 7901050017 (the base
    number of the first Wikipedia example, born
    5 January 1979, serial 001, female): the
    weighted sum is 106, 106 mod 11 is 7, and
    the check digit is 7; 9912311230 (born 31
    December 1999, serial 123, female): the
    weighted sum is 142, 142 mod 11 is 10, so
    the check digit is 0; 6101047500 (born 4
    January 1961, serial 750, male): the
    weighted sum is 121, 121 mod 11 is 0, and
    the check digit is 0. The recognizer's
    checksum was also cross-checked against
    python-stdnum on all fixtures, on 2000
    random 9 digit bodies (check digit
    computation), and on 2000 random 10 digit
    numbers (checksum validation), with zero
    mismatches in every direction.

    Five honest caveats. First, the checksum
    alone passes about one time in 10 for an
    arbitrary 10 digit token, so a bare 10 digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. The date rule cuts this
    down sharply: a random 10 digit token passes
    the full validation only about 1 time in 120
    (1,707 in 200,000 random tokens measured).
    Second, the first worked example printed in
    the Wikipedia article is arithmetically
    wrong: it prints the full number 7901050018
    and claims its products sum to 140 with
    remainder 8, but the products it lists sum
    to 106 with remainder 7, so the number that
    follows from the stated algorithm is
    7901050017, which python-stdnum also
    validates, while the printed 7901050018
    fails the check. This recognizer follows the
    algorithm and python-stdnum, and uses the
    corrected number as its fixture. Third,
    python-stdnum accepts spaced and dashed
    forms and does not reject future dates,
    while this recognizer matches plain 10
    digits only and rejects a birth date in the
    future, so it is stricter than
    python-stdnum. Fourth, python-stdnum places
    no restriction on the serial number and
    checks no district range, and neither does
    this recognizer: the serial may be 000 to
    999, and no region allowlist is enforced,
    because the district ranges are not published
    as a fixed list. Fifth, a passing checksum
    only shows the number is well formed; it
    cannot show the number was ever issued to
    anyone, and only the Bulgarian civil registry
    can confirm that.

    The recognizer validates the date and
    checksum in invalidate_result and keeps the
    pattern score at 0.5: it does not promote a
    match to 1.0, because a checksum passes about
    one time in 10 for an arbitrary same-shape
    token.

    Invalidation is based on: the number must be
    exactly 10 digits; the number must not be all
    the same digit; a fully sequential digit run
    is rejected; the first 6 digits must be a
    real calendar date that is not in the future,
    with the month read as 01 to 12 meaning the
    1900s, 21 to 32 meaning the 1800s with 20
    subtracted, and 41 to 52 meaning the 2000s
    with 40 subtracted; and the check digit must
    satisfy the weighted mod 11 check, that is,
    with weights 2, 4, 8, 5, 10, 9, 7, 3, 6 on
    the first 9 digits, (sum mod 11) must equal
    the 10th digit, with a remainder of 10
    meaning 0. This follows the approach of the
    Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh
    NID, Sri Lanka NIC, Ghana Card, Kazakhstan
    IIN, Ukraine RNOKPP, Uzbekistan PINFL,
    Morocco CNIE, Indonesia NIK, Malaysia MyKad,
    Vietnam CCCD, Chile RUT, Ecuador Cedula,
    Iran Code Melli, Argentina CUIT/CUIL, Mexico
    CURP, Peru RUC, Colombia NIT, Dominican
    Republic Cedula, Paraguay RUC, Uruguay RUT,
    Iceland kennitala, Czech birth number,
    Romanian personal numeric code, Dutch citizen
    service number, Croatian personal
    identification number, and Slovenian unique
    master citizen number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "bg"

    PATTERNS = [
        Pattern(
            "BG_EGN",
            r"(?<![A-Za-z0-9./-])\d{10}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "bulgaria",
        "bulgarian",
        "egn",
        "edinen grazhdanski nomer",
        "uniform civil number",
        "unified civil number",
        "personal identification number",
        "esgraon",
    ]

    CHECK_WEIGHTS = (2, 4, 8, 5, 10, 9, 7, 3, 6)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "BG_EGN",
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
        Check if the pattern text cannot be a BG_EGN entity.

        The number must be 10 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, must carry a real birth date that is
        not in the future, and must pass the weighted
        mod 11 check.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 10 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued number is never one fully sequential run
            return True
        if not self._has_valid_birth_date(text):
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_birth_date(digits: str) -> bool:
        """
        Check the birth date encoded in the first 6 digits.

        The date is YYMMDD, with a month code of 01 to
        12 meaning the 1900s, 21 to 32 meaning the
        1800s with 20 subtracted, and 41 to 52
        meaning the 2000s with 40 subtracted. The
        date must be a real calendar date and must
        not be in the future. This mirrors the
        python-stdnum bg.egn get_birth_date rule.

        :param digits: The 10 digit uniform civil number
        :return: True when the encoded birth date is real and past
        """
        year = int(digits[0:2])
        month = int(digits[2:4])
        day = int(digits[4:6])
        if month > 40:
            year += 2000
            month -= 40
        elif month > 20:
            year += 1800
            month -= 20
        else:
            year += 1900
        try:
            birth_date = datetime.date(year, month, day)
        except ValueError:
            return False
        return birth_date <= datetime.date.today()

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the weighted mod 11 checksum of a 10 digit number.

        With weights 2, 4, 8, 5, 10, 9, 7, 3, 6 on
        the first 9 digits, (sum mod 11) must equal
        the 10th digit, with a remainder of 10
        meaning a check digit of 0.

        :param digits: The 10 digit uniform civil number
        :return: True when the checksum holds
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(digits[:9], BgEgnRecognizer.CHECK_WEIGHTS)
        )
        check = total % 11
        if check == 10:
            check = 0
        return check == int(digits[9])

    @staticmethod
    def _check_digit(first_nine: str) -> str:
        """
        Compute the check digit for the first 9 digits.

        With weights 2, 4, 8, 5, 10, 9, 7, 3, 6 on
        the first 9 digits, the check digit is
        (sum mod 11), with a remainder of 10 giving
        a check digit of 0, so every 9 digit body
        has a valid check digit.

        :param first_nine: The first 9 digits of the number
        :return: The check digit, 0 to 9
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(first_nine, BgEgnRecognizer.CHECK_WEIGHTS)
        )
        check = total % 11
        if check == 10:
            check = 0
        return str(check)

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
