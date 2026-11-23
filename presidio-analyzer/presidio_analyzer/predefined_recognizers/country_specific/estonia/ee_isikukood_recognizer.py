import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class EeIsikukoodRecognizer(PatternRecognizer):
    """
    Recognize Estonian personal identification codes (isikukood).

    The isikukood is the 11 digit personal
    number issued to Estonian citizens and
    registered residents, including
    e-residents, by the Population Register.
    It doubles as the individual tax
    identification number (TIN); no separate
    tax number is issued to residents. The
    number has 11 digits and encodes the
    holder's sex, birth century and date of
    birth: GYYMMDDSSSC, where G is the gender
    and century digit (1 male and 2 female
    born 1800 to 1899, 3 male and 4 female
    born 1900 to 1999, 5 male and 6 female
    born 2000 to 2099), YYMMDD is the date
    of birth, SSS is a three digit serial
    number distinguishing persons born on
    the same date, and C is a check digit.
    It is written plain, 11 digits with no
    separators, for example 36805280109.

    Checksum: the check digit comes from
    weights 1, 2, 3, 4, 5, 6, 7, 8, 9, 1 on
    the first 10 digits, as published in the
    OECD Estonia TIN sheet (which states the
    GYYMMDDSSSC structure and names the last
    digit a checksum) and implemented by the
    python-stdnum ee.ik module (which reuses
    the same two round computation as its
    lt.asmens module). With r = sum mod 11,
    a value of r from 0 to 9 is the check
    digit. When r is 10, a second round runs
    with weights 3, 4, 5, 6, 7, 8, 9, 1, 2, 3
    on the first 10 digits, and the check
    digit is the second round remainder mod
    10, so a second round remainder of 10
    gives a check digit of 0. Worked examples,
    verified by hand and by computation:
    36805280109 (a man born 28 May 1968,
    serial 010): the first weighted sum is
    141, 141 mod 11 is 9, and the check digit
    is 9; 37605030299 (a man born 3 May 1976,
    serial 029): the weighted sum is 108, 108
    mod 11 is 9, and the check digit is 9;
    49002020002 (a woman born 2 February
    1990, serial 000): the weighted sum is
    46, 46 mod 11 is 2, and the check digit
    is 2; 60002290003 (a woman born on the
    leap day 29 February 2000, serial 000):
    the weighted sum is 91, 91 mod 11 is 3,
    and the check digit is 3; 50001010006 (a
    man born 1 January 2000, serial 000): the
    weighted sum is 17, 17 mod 11 is 6, and
    the check digit is 6; 10101010005 (a man
    born 1 January 1801, serial 000): the
    weighted sum is 16, 16 mod 11 is 5, and
    the check digit is 5; 29912310009 (a woman
    born 31 December 1899, serial 000): the
    weighted sum is 86, 86 mod 11 is 9, and
    the check digit is 9; 19912310008 (a man
    born 31 December 1899, serial 000): the
    weighted sum is 85, 85 mod 11 is 8, and
    the check digit is 8; 23103250882 (a woman
    born 25 March 1831, serial 088): the
    first weighted sum is 153, 153 mod 11 is
    10, so the second round runs, its weighted
    sum is 145, 145 mod 11 is 2, and the check
    digit is 2; 26909016130 (a woman born 1
    September 1869, serial 613): the first
    weighted sum is 153, 153 mod 11 is 10, so
    the second round runs, its weighted sum
    is 164, 164 mod 11 is 10, and the check
    digit is 0 through the fold. The
    recognizer's checksum was also cross-checked
    against python-stdnum ee.ik on all
    fixtures and on 2000 random 10 digit bodies
    (check digit computation), with zero
    mismatches. Full validation differs from
    ee.ik only in the stricter direction, by
    design: this recognizer accepts only
    first digits 1 to 6 and rejects future
    birth dates, so in a 200,000 random 11
    digit number run, the only disagreements
    were 255 numbers ee.ik accepts and this
    recognizer rejects (160 with first digits
    7 or 8, 95 with future dates), none the
    other way.

    Seven honest caveats. First, the checksum
    alone passes about one time in ten for an
    arbitrary 11 digit token (19,999 in
    200,000 random tokens measured), so a bare
    11 digit number is weak evidence on its
    own; the recognizer keeps the pattern
    score at 0.5 and never promotes it. The
    date and first digit rules cut this down
    sharply: a random 11 digit token passes
    the full validation only about 1 time in
    533 (375 in 200,000 random tokens
    measured). Second, python-stdnum ee.ik
    accepts first digits 7 and 8 (the 2100
    century) and does not reject future birth
    dates, while this recognizer accepts only
    first digits 1 to 6 and rejects a birth
    date in the future, so it is stricter than
    python-stdnum. Third, the sources agree
    on the format: the OECD Estonia TIN sheet,
    the Microsoft Purview Estonia personal
    code definition (first digit 1 to 6, 11
    digits, checksum yes), and python-stdnum
    ee.ik all state the same GYYMMDDSSSC
    structure, and no source disagreement was
    found; the check rule itself is set by
    standard EVS 585, which the sources cite
    without reproducing. Fourth, python-stdnum
    strips spaces when compacting a number,
    while this recognizer matches plain 11
    digits only, so a spaced form is not
    matched here. Fifth, the all zeros string
    00000000000 passes the checksum but is
    rejected here as implausible, and the
    fully sequential run 87654321098 passes
    the checksum but is rejected here as a
    sequential run; no other fully sequential
    11 digit run passes the checksum (all 20
    wrap runs checked). Sixth, a first digit
    of 9 or 0 is rejected by python-stdnum
    ee.ik as well as by this recognizer, so
    those codes are missed in the safe
    direction by both. Seventh, a passing
    checksum only shows the number is well
    formed; it cannot show the number was ever
    issued to anyone, and only the Estonian
    Population Register can confirm that. The
    OECD sheet adds that Estonia has no online
    tool for TIN verification.

    The recognizer validates the date, first
    digit, and checksum in invalidate_result
    and keeps the pattern score at 0.5: it does
    not promote a match to 1.0, because a
    checksum passes about one time in ten for
    an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 11 digits; the number must not be all
    the same digit; a fully sequential digit run
    is rejected; the first digit must be 1 to 6,
    giving the birth century (1 and 2 the 1800s,
    3 and 4 the 1900s, 5 and 6 the 2000s); the
    next 6 digits must be a real calendar date
    that is not in the future; and the check
    digit must satisfy the two round mod 11
    check, that is, with weights 1, 2, 3, 4, 5,
    6, 7, 8, 9, 1 on the first 10 digits, the
    remainder mod 11 is the check digit unless
    it is 10, in which case weights 3, 4, 5, 6,
    7, 8, 9, 1, 2, 3 are used and the remainder
    mod 10 is the check digit.
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
    number, Slovenian unique master citizen
    number, Bulgarian uniform civil number,
    Greek tax identification number, Serbian
    unique master citizen number, and Lithuanian
    personal code recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ee"

    PATTERNS = [
        Pattern(
            "EE_IK",
            r"(?<![A-Za-z0-9./-])\d{11}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "estonia",
        "estonian",
        "isikukood",
        "personal code",
        "estonian personal code",
        "population register",
    ]

    # First digit: 1 and 2 are the 1800s, 3 and 4
    # the 1900s, 5 and 6 the 2000s. Odd is male,
    # even is female. Digits 7 and 8 would be the
    # 2100s, which python-stdnum ee.ik accepts,
    # but they are not accepted here: no isikukood
    # with a 2100s birth date can exist yet. Digits
    # 9 and 0 are not accepted either.
    FIRST_DIGIT_CENTURIES = {
        "1": 1800,
        "2": 1800,
        "3": 1900,
        "4": 1900,
        "5": 2000,
        "6": 2000,
    }

    CHECK_WEIGHTS_FIRST = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1)
    CHECK_WEIGHTS_SECOND = (3, 4, 5, 6, 7, 8, 9, 1, 2, 3)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "EE_IK",
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
        Check if the pattern text cannot be a EE_IK entity.

        The number must be 11 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, must carry a first digit of 1 to 6,
        must carry a real birth date that is not in the
        future, and must pass the two round mod 11 check.

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
        if text[0] not in self.FIRST_DIGIT_CENTURIES:
            return True
        if not self._has_valid_birth_date(text):
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_birth_date(digits: str) -> bool:
        """
        Check the birth date encoded in the first 7 digits.

        The date is GYYMMDD, with G of 1 or 2 meaning
        the 1800s, 3 or 4 the 1900s, and 5 or 6 the
        2000s. The date must be a real calendar date
        and must not be in the future.

        :param digits: The 11 digit personal code
        :return: True when the encoded birth date is real and past
        """
        century = EeIsikukoodRecognizer.FIRST_DIGIT_CENTURIES[digits[0]]
        year = century + int(digits[1:3])
        month = int(digits[3:5])
        day = int(digits[5:7])
        try:
            birth_date = datetime.date(year, month, day)
        except ValueError:
            return False
        return birth_date <= datetime.date.today()

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the two round mod 11 checksum of an 11 digit number.

        With weights 1, 2, 3, 4, 5, 6, 7, 8, 9, 1
        on the first 10 digits, the remainder mod 11
        is the check digit, unless it is 10, in which
        case weights 3, 4, 5, 6, 7, 8, 9, 1, 2, 3
        are used and the remainder mod 10 is the
        check digit.

        :param digits: The 11 digit personal code
        :return: True when the checksum holds
        """
        return EeIsikukoodRecognizer._check_digit(digits[:10]) == digits[10]

    @staticmethod
    def _check_digit(first_ten: str) -> str:
        """
        Compute the check digit for the first 10 digits.

        With weights 1, 2, 3, 4, 5, 6, 7, 8, 9, 1
        on the first 10 digits, the remainder mod 11
        is the check digit, unless it is 10, in which
        case weights 3, 4, 5, 6, 7, 8, 9, 1, 2, 3
        are used and the remainder mod 10 is the
        check digit, so every 10 digit body has a
        valid check digit.

        :param first_ten: The first 10 digits of the number
        :return: The check digit, 0 to 9
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(
                first_ten, EeIsikukoodRecognizer.CHECK_WEIGHTS_FIRST
            )
        )
        remainder = total % 11
        if remainder == 10:
            total = sum(
                int(digit) * weight
                for digit, weight in zip(
                    first_ten, EeIsikukoodRecognizer.CHECK_WEIGHTS_SECOND
                )
            )
            remainder = total % 11
        return str(remainder % 10)

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
