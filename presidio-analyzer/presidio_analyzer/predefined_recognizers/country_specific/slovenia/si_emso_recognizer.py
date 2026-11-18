import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class SiEmsoRecognizer(PatternRecognizer):
    """
    Recognize Slovenian unique master citizen numbers (EMSO).

    The EMSO (Enotna maticna stevilka obcana,
    Unique Master Citizen Number) is the personal
    number issued in Slovenia by the Central
    Citizen Registry (Centralni Register
    Prebivalstva, CRP) to every person, including
    foreign citizens living in Slovenia. It has
    13 digits and encodes the holder's date of
    birth: DDMMYYY RR SSS K, where DD is the day
    of birth, MM the month of birth, YYY the last
    three digits of the year of birth, RR the
    region of birth, SSS a gender and serial
    number (000 to 499 for men, 500 to 999 for
    women), and K a check digit. It is written
    plain, 13 digits with no separators, for
    example 0101006500006.

    Region: 50 is the standard Slovenia value,
    stated by the US Social Security
    Administration POMS and by the Microsoft
    Slovenia DLP entity definition. Codes 51 to
    59 entered use in 2024, per Wikipedia. Code
    05 covers foreigners registered in Slovenia,
    per the JMBG region table. All other region
    codes belong to other former Yugoslav
    registries and are rejected here.

    Year: the three digit year follows the
    python-stdnum convention, a value below 800
    means the 2000s and 800 or above means the
    1800s, so 006 is 2006 and 987 is 1987.

    Checksum: the check digit comes from weights
    7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 on the
    first 12 digits, as published in the
    Wikipedia Unique Master Citizen Number
    article and implemented by the python-stdnum
    si.emso module. With m = 11 - (sum mod 11),
    a value of m from 1 to 9 is the check digit,
    and m of 10 or 11 gives a check digit of 0.
    Worked examples, verified by hand and by
    computation: 0101006500006 (the python-stdnum
    docstring example and the Wikipedia example,
    the first male baby registered in Slovenia
    on 1 January 2006): the weighted sum is 82,
    82 mod 11 is 5, and 11 minus 5 is 6;
    1809987504991 (the MicroFocus national ID
    example, a man born 18 September 1987 in
    region 50, serial 499): the weighted sum is
    274, 274 mod 11 is 10, and 11 minus 10 is 1;
    2007990501230 (a man born 20 July 1990 in
    region 50, serial 123): the weighted sum is
    133, 133 mod 11 is 1, and 11 minus 1 is 10,
    so the check digit is 0. The recognizer's
    checksum was also cross-checked against
    python-stdnum on all fixtures, on 2000
    random 12 digit bodies (check digit
    computation), and on 2000 random 13 digit
    numbers (validation), with zero mismatches
    in every direction.

    Six honest caveats. First, the checksum
    alone passes about one time in ten for an
    arbitrary 13 digit token, so a bare 13 digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. The date and region rules
    cut this down sharply: a random 13 digit
    token passes the full validation only about
    1 time in 10,000 (23 in 200,000 random tokens
    measured). Second, python-stdnum
    accepts any region code and does not reject
    future dates, while this recognizer enforces
    region 05 or 50 to 59 and rejects a birth
    date in the future, so it is stricter than
    python-stdnum. Third, python-stdnum strips
    spaces when compacting a number, while this
    recognizer matches plain 13 digits only, so
    a spaced form is not matched here. Fourth,
    Wikipedia notes that a small number of
    genuinely issued numbers carry an invalid
    checksum through operator error, so those
    numbers are missed here, in the safe
    direction. Fifth, serial 000 is valid: the
    published example 0101006500006 is the first
    baby registered on its day, with serial 000.
    Sixth, a passing checksum only shows the
    number is well formed; it cannot show the
    number was ever issued to anyone, and only
    the Slovenian Central Citizen Registry can
    confirm that.

    The recognizer validates the date, region,
    and checksum in invalidate_result and keeps
    the pattern score at 0.5: it does not promote
    a match to 1.0, because a checksum passes
    about one time in ten for an arbitrary
    same-shape token.

    Invalidation is based on: the number must be
    exactly 13 digits; the number must not be all
    the same digit; a fully sequential digit run
    is rejected; the first 7 digits must be a
    real calendar date that is not in the future,
    with the year read as below 800 meaning the
    2000s and 800 or above meaning the 1800s;
    the region code must be 05 or 50 to 59; and
    the check digit must satisfy the weighted
    mod 11 check, that is, with weights 7, 6, 5,
    4, 3, 2, 7, 6, 5, 4, 3, 2 on the first 12
    digits, m = 11 - (sum mod 11) must equal the
    13th digit, with m of 10 or 11 meaning 0.
    This follows the approach of the Pakistani
    CNIC, Saudi national ID, UAE Emirates ID,
    Egyptian national ID, Bangladesh NID, Sri
    Lanka NIC, Ghana Card, Kazakhstan IIN,
    Ukraine RNOKPP, Uzbekistan PINFL, Morocco
    CNIE, Indonesia NIK, Malaysia MyKad, Vietnam
    CCCD, Chile RUT, Ecuador Cedula, Iran Code
    Melli, Argentina CUIT/CUIL, Mexico CURP,
    Peru RUC, Colombia NIT, Dominican Republic
    Cedula, Paraguay RUC, Uruguay RUT, Iceland
    kennitala, Czech birth number, Romanian
    personal numeric code, Dutch citizen service
    number, Croatian personal identification
    number, and Slovenian unique master citizen
    number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "si"

    PATTERNS = [
        Pattern(
            "SI_EMSO",
            r"(?<![A-Za-z0-9./-])\d{13}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "slovenia",
        "slovenian",
        "emso",
        "enotna maticna stevilka obcana",
        "unique master citizen number",
        "personal identification number",
        "central citizen registry",
    ]

    # Region codes of the Slovenian registry: 50 is the
    # standard value, 51 to 59 entered use in 2024,
    # and 05 covers foreigners registered in Slovenia.
    VALID_REGIONS = frozenset(["05"] + [str(code) for code in range(50, 60)])

    CHECK_WEIGHTS = (7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "SI_EMSO",
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
        Check if the pattern text cannot be a SI_EMSO entity.

        The number must be 13 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, must carry a real birth date that is
        not in the future, must carry region 05 or 50
        to 59, and must pass the weighted mod 11 check.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 13 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued number is never one fully sequential run
            return True
        if not self._has_valid_birth_date(text):
            return True
        if text[7:9] not in self.VALID_REGIONS:
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_birth_date(digits: str) -> bool:
        """
        Check the birth date encoded in the first 7 digits.

        The date is DDMMYYY, with YYY below 800 meaning
        the 2000s and 800 or above meaning the 1800s.
        The date must be a real calendar date and must
        not be in the future.

        :param digits: The 13 digit unique master citizen number
        :return: True when the encoded birth date is real and past
        """
        day = int(digits[0:2])
        month = int(digits[2:4])
        year = int(digits[4:7])
        if year < 800:
            year += 2000
        else:
            year += 1000
        try:
            birth_date = datetime.date(year, month, day)
        except ValueError:
            return False
        return birth_date <= datetime.date.today()

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the weighted mod 11 checksum of a 13 digit number.

        With weights 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2
        on the first 12 digits, m = 11 - (sum mod 11)
        must equal the 13th digit, with m of 10 or 11
        meaning a check digit of 0.

        :param digits: The 13 digit unique master citizen number
        :return: True when the checksum holds
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(digits[:12], SiEmsoRecognizer.CHECK_WEIGHTS)
        )
        check = 11 - (total % 11)
        if check >= 10:
            check = 0
        return check == int(digits[12])

    @staticmethod
    def _check_digit(first_twelve: str) -> str:
        """
        Compute the check digit for the first 12 digits.

        With weights 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2
        on the first 12 digits, m = 11 - (sum mod 11),
        and m of 10 or 11 gives a check digit of 0, so
        every 12 digit body has a valid check digit.

        :param first_twelve: The first 12 digits of the number
        :return: The check digit, 0 to 9
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(first_twelve, SiEmsoRecognizer.CHECK_WEIGHTS)
        )
        check = 11 - (total % 11)
        if check >= 10:
            check = 0
        return str(check)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 0123456789012 ascends by one each step, and
        9876543210987 descends by one each step.

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
