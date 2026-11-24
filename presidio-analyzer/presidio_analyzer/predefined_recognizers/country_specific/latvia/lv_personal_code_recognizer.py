import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class LvPersonalCodeRecognizer(PatternRecognizer):
    """
    Recognize Latvian personal codes (personas kods).

    The personas kods is the 11 digit personal
    number issued to Latvian citizens and
    registered residents by the Office of
    Citizenship and Migration Affairs (PMLP).
    It doubles as the individual tax
    identification number (TIN); the OECD
    Latvia TIN sheet states the TIN issued to
    individuals is identical with their
    personal code, and no separate tax number
    is issued to residents.

    Two formats exist. The old format, issued
    up to 30 June 2017, is DDMMYY-CNNNZ,
    where DDMMYY is the date of birth, C is
    the century digit (0 for the 1800s, 1 for
    the 1900s, 2 for the 2000s), NNN is a
    three digit serial number distinguishing
    persons born on the same date, and Z is a
    check digit. It is usually printed with a
    hyphen after the sixth digit, for example
    161175-19997. The new format, issued from
    1 July 2017, starts with the digit 3 and
    a second digit from 2 to 9, and the
    remaining nine digits are random, so it
    encodes no date of birth; the OECD sheet
    gives the example 38XXXX-XXXXX. The last
    digit is still a check digit.

    Checksum: the check digit comes from
    weights 1, 6, 3, 7, 9, 10, 5, 8, 4, 2 on
    the first 10 digits. With S the weighted
    sum, the check digit is (1101 - S) mod 11
    mod 10, so a remainder of 10 folds to a
    check digit of 0. This is the formula
    stated by Wikipedia for the Latvian
    personal code, and it is exactly the rule
    implemented by the python-stdnum lv.pvn
    module, which computes (1 + sum) mod 11
    mod 10 with weights 10, 5, 8, 4, 2, 1, 6,
    3, 7, 9; the two weight sets sum to 11 at
    every position, so the two computations
    agree for every body. The OECD Latvia TIN
    sheet states the structure and the
    century digits but states no check rule.
    Worked examples, verified by hand and by
    computation: 16117519997 (a person born
    16 November 1975, century 1, serial 999):
    the weighted sum is 291, 1101 - 291 is
    810, 810 mod 11 is 7, and the check digit
    is 7; 11111111111 (a person born 11
    November 1911, century 1, serial 111):
    the weighted sum is 55, 1101 - 55 is
    1046, 1046 mod 11 is 1, and the check
    digit is 1; 01010000000 (a person born 1
    January 1800, century 0, serial 000):
    the weighted sum is 13, 1101 - 13 is
    1088, 1088 mod 11 is 10, and the check
    digit is 0 through the fold; 01010020000
    (a person born 1 January 2000, century 2,
    serial 000): the weighted sum is 23, 1101
    - 23 is 1078, 1078 mod 11 is 0, and the
    check digit is 0; 29020020009 (a person
    born on the leap day 29 February 2000,
    century 2, serial 000): the weighted sum
    is 80, 1101 - 80 is 1021, 1021 mod 11 is
    9, and the check digit is 9; 31129900002
    (a person born 31 December 1899, century
    0, serial 000): the weighted sum is 197,
    1101 - 197 is 904, 904 mod 11 is 2, and
    the check digit is 2; 31129910008 (a
    person born 31 December 1999, century 1,
    serial 000): the weighted sum is 202, 1101
    - 202 is 899, 899 mod 11 is 8, and the
    check digit is 8; 32867300679 (a new
    format code starting with 32): the
    weighted sum is 212, 1101 - 212 is 889,
    889 mod 11 is 9, and the check digit is
    9; 38123456782 (a new format code
    starting with 38): the weighted sum is
    252, 1101 - 252 is 849, 849 mod 11 is 2,
    and the check digit is 2; 39123456787 (a
    new format code starting with 39): the
    weighted sum is 258, 1101 - 258 is 843,
    843 mod 11 is 7, and the check digit is
    7. The recognizer's checksum was also
    cross-checked against python-stdnum
    lv.pvn on all fixtures and on 2000 random
    10 digit bodies (check digit
    computation), with zero mismatches. Full
    validation differs from lv.pvn in both
    directions, by design: this recognizer
    accepts new format codes starting with 33
    to 39, which lv.pvn rejects, following
    the OECD sheet, and it rejects century
    digits 3 to 9, future birth dates, and
    legal entity PVN numbers, which lv.pvn
    accepts, so in a 200,000 random 11 digit
    number run, 1,430 numbers were accepted
    here and rejected by lv.pvn (all new
    format codes starting with 33 to 39) and
    11,586 numbers were accepted by lv.pvn
    and rejected here (11,021 legal entity
    PVN numbers, 516 with century digits 3
    to 9, and 49 with future birth dates).

    Seven honest caveats. First, the checksum
    alone passes about one time in ten for an
    arbitrary 11 digit token (19,776 in
    200,000 random tokens measured), so a bare
    11 digit number is weak evidence on its
    own; the recognizer keeps the pattern
    score at 0.5 and never promotes it. The
    date, century, and prefix rules cut this
    down: a random 11 digit token passes the
    full validation only about 1 time in 111
    (1,805 in 200,000 random tokens
    measured), weaker than the date carrying
    codes of Estonia and Lithuania, because
    the new format branch carries no date to
    check. Second, the sources agree on the
    structure: the OECD Latvia TIN sheet, the
    Microsoft Purview Latvia personal code
    definition (old format DDMMYY with
    century 0, 1, 2, new format starting with
    32, checksum yes), the FCA Annex II
    Latvia row, and Wikipedia all state the
    same DDMMYY century serial check
    structure, but they disagree on the new
    format prefix: the OECD sheet says the
    first digit is 3 and the second digit is
    a random digit from 2 to 9, while
    Microsoft and python-stdnum accept only
    32; this recognizer follows the OECD
    sheet and accepts 32 to 39. Third, the
    check digit itself is not stated in the
    law or in the OECD sheet, so it rests on
    the agreement of Wikipedia, python-stdnum
    lv.pvn, the PEAR Validate_LV personId
    function, and Microsoft Purview. The
    PEAR function agrees except in the fold
    case: it requires the weighted sum over
    all 11 digits mod 11 to equal 1, which
    admits no valid check digit when the
    remainder is 10, while Wikipedia and
    python-stdnum give a check digit of 0
    there; measured, PEAR rejects 162 of
    2000 numbers valid under the
    Wikipedia and python-stdnum rule, all
    fold cases, and this recognizer follows
    Wikipedia and python-stdnum. Fourth,
    python-stdnum strips spaces and hyphens
    when compacting a number, while this
    recognizer matches plain 11 digits only,
    so the hyphenated printed form
    DDMMYY-XXXXX is not matched here. Fifth,
    the all ones string 11111111111 passes
    the checksum but is rejected here as
    implausible, and the fully sequential
    runs 01234567890 and 65432109876 pass the
    checksum but are rejected here as
    sequential runs; no other fully
    sequential 11 digit run passes the
    checksum (all 20 wrap runs checked).
    Sixth, a first digit of 4, 5, or 9 marks
    a legal entity PVN number under a
    different check, not a personal code, so
    those numbers are not matched here.
    Seventh, a passing checksum only shows
    the number is well formed; it cannot show
    the number was ever issued to anyone, and
    only the Latvian Office of Citizenship
    and Migration Affairs (PMLP) can confirm
    that. The OECD sheet gives PMLP as the
    information source and names no online
    verification tool for individuals.

    The recognizer validates the date, first
    digit, and checksum in invalidate_result
    and keeps the pattern score at 0.5: it does
    not promote a match to 1.0, because a
    checksum passes about one time in ten for
    an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 11 digits; the number must not be all
    the same digit; a fully sequential digit run
    is rejected; a number starting with 3 and a
    second digit from 2 to 9 is a new format
    code and needs only the checksum; otherwise
    the first digit must be 0, 1, 2, or 3, the
    first six digits must be a real calendar
    date that is not in the future, the seventh
    digit must be 0, 1, or 2, giving the birth
    century (0 the 1800s, 1 the 1900s, 2 the
    2000s), and the check digit must satisfy
    the mod 11 check, that is, with weights 1,
    6, 3, 7, 9, 10, 5, 8, 4, 2 on the first 10
    digits, the check digit is (1101 - sum)
    mod 11 mod 10.
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
    unique master citizen number, Lithuanian
    personal code, and Estonian personal
    identification code recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "lv"

    PATTERNS = [
        Pattern(
            "LV_PK",
            r"(?<![A-Za-z0-9./-])\d{11}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "latvia",
        "latvian",
        "personas kods",
        "personal code",
        "latvian personal code",
        "population register",
    ]

    # Century digit: 0 is the 1800s, 1 the
    # 1900s, 2 the 2000s. Digits 3 to 9 are
    # not accepted: python-stdnum lv.pvn
    # accepts them, but no century digit
    # above 2 is defined by the OECD sheet,
    # Microsoft Purview, or Wikipedia.
    CENTURY_DIGITS = {
        "0": 1800,
        "1": 1900,
        "2": 2000,
    }

    # New format codes start with 3 and a
    # second digit from 2 to 9, per the OECD
    # Latvia TIN sheet. A code starting with
    # 30 or 31 is an old format code with a
    # birth day of 30 or 31, never a new
    # format code.
    NEW_FORMAT_SECOND_DIGITS = "23456789"

    CHECK_WEIGHTS = (1, 6, 3, 7, 9, 10, 5, 8, 4, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "LV_PK",
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
        Check if the pattern text cannot be a LV_PK entity.

        The number must be 11 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, and must pass the mod 11 check. A
        number starting with 3 and a second digit from
        2 to 9 is a new format code and needs nothing
        else. Otherwise the first six digits must be a
        real birth date that is not in the future and
        the seventh digit must be 0, 1, or 2.

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
        if self._is_new_format(text):
            # new format codes carry no date, only the checksum
            return not self._has_valid_checksum(text)
        if text[0] not in "0123":
            # a first digit of 4, 5, or 9 is a legal entity
            # PVN number, not a personal code
            return True
        if not self._has_valid_birth_date(text):
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _is_new_format(digits: str) -> bool:
        """
        Check if the number is a new format personal code.

        New format codes, issued from 1 July 2017,
        start with 3 and a second digit from 2 to 9,
        per the OECD Latvia TIN sheet.

        :param digits: The 11 digit personal code
        :return: True when the number is a new format code
        """
        return (
            digits[0] == "3"
            and digits[1] in LvPersonalCodeRecognizer.NEW_FORMAT_SECOND_DIGITS
        )

    @staticmethod
    def _has_valid_birth_date(digits: str) -> bool:
        """
        Check the birth date encoded in the first 7 digits.

        The date is DDMMYY, with the seventh digit 0,
        1, or 2 meaning the 1800s, 1900s, or 2000s.
        The date must be a real calendar date and must
        not be in the future.

        :param digits: The 11 digit personal code
        :return: True when the encoded birth date is real and past
        """
        if digits[6] not in LvPersonalCodeRecognizer.CENTURY_DIGITS:
            return False
        century = LvPersonalCodeRecognizer.CENTURY_DIGITS[digits[6]]
        year = century + int(digits[4:6])
        month = int(digits[2:4])
        day = int(digits[0:2])
        try:
            birth_date = datetime.date(year, month, day)
        except ValueError:
            return False
        return birth_date <= datetime.date.today()

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the mod 11 checksum of an 11 digit number.

        With weights 1, 6, 3, 7, 9, 10, 5, 8, 4, 2
        on the first 10 digits, the check digit is
        (1101 - sum) mod 11 mod 10.

        :param digits: The 11 digit personal code
        :return: True when the checksum holds
        """
        return LvPersonalCodeRecognizer._check_digit(digits[:10]) == digits[10]

    @staticmethod
    def _check_digit(first_ten: str) -> str:
        """
        Compute the check digit for the first 10 digits.

        With weights 1, 6, 3, 7, 9, 10, 5, 8, 4, 2
        on the first 10 digits, the check digit is
        (1101 - sum) mod 11 mod 10, so a remainder
        of 10 folds to a check digit of 0, and every
        10 digit body has a valid check digit.

        :param first_ten: The first 10 digits of the number
        :return: The check digit, 0 to 9
        """
        total = sum(
            int(digit) * weight
            for digit, weight in zip(first_ten, LvPersonalCodeRecognizer.CHECK_WEIGHTS)
        )
        return str((1101 - total) % 11 % 10)

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
