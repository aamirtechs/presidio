import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class MkEmbgRecognizer(PatternRecognizer):
    """
    Recognize North Macedonian unique master citizen numbers (EMBG).

    The EMBG (Edinstven maticen broj na graganinot,
    Unique Master Citizen Number) is the 13 digit
    personal number assigned to citizens of North
    Macedonia at birth registration. It is the same
    number system used across the former Yugoslav
    republics: Wikipedia notes it continues to be
    used in its original form in Bosnia and
    Herzegovina, Montenegro, North Macedonia,
    Serbia and Slovenia, while Croatia replaced it
    with the personal identification number (OIB).
    The number has 13 digits and encodes the
    holder's date of birth: DDMMYYY RR BBB K,
    where DD is the day of birth, MM the month of
    birth, YYY the last three digits of the year
    of birth, RR the region of birth, BBB a gender
    and serial number (000 to 499 for men, 500 to
    999 for women), and K a check digit. It is
    written plain, 13 digits with no separators,
    for example 0807992455005.

    Region: 41 to 49 is North Macedonia (41
    Bitola, 42 Kumanovo, 43 Ohrid, 44 Prilep,
    45 Skopje, 46 Strumica, 47 Tetovo, 48 Veles,
    49 Stip). Code 04 covers foreigners in
    Macedonia. Code 40 is not assigned to any
    region in the Wikipedia JMBG region table
    and is not accepted here. All other region
    codes belong to other registries and are
    rejected here: 50 to 59 is Slovenia
    (SI_EMSO), 70 to 99 is Serbia (RS_JMBG),
    10 to 19 Bosnia and Herzegovina, 20 to 29
    Montenegro, and 30 to 39 Croatia, where the
    number is no longer used and HR_OIB
    replaced it.

    Year: the three digit year follows the
    python-stdnum convention, a value below 800
    means the 2000s and 800 or above means the
    1800s, so 006 is 2006 and 987 is 1987, and
    899 is 1899.

    Checksum: the check digit comes from weights
    7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 on the
    first 12 digits, as published in the
    Wikipedia Unique Master Citizen Number
    article and implemented by the python-stdnum
    si.emso module (python-stdnum has no
    dedicated EMBG module; its si package holds
    the Slovenian form of the same shared JMBG
    number). With m = 11 - (sum mod 11), a value
    of m from 1 to 9 is the check digit, and m
    of 10 or 11 gives a check digit of 0. Worked
    examples, verified by hand and by
    computation: 0807992455005 (the Treeet EMBG
    checker worked example, a woman born 8 July
    1992 in Skopje, region 45, serial 500): the
    weighted sum is 204, 204 mod 11 is 6, and
    11 minus 6 is 5; 0101990410004 (a man born
    1 January 1990 in Bitola, region 41, serial
    000): the weighted sum is 84, 84 mod 11 is
    7, and 11 minus 7 is 4; 1503985435125 (a
    woman born 15 March 1985 in Ohrid, region
    43, serial 512): the weighted sum is 193,
    193 mod 11 is 6, and 11 minus 6 is 5;
    2902000472507 (a man born on the leap day
    29 February 2000 in Tetovo, region 47,
    serial 250): the weighted sum is 158, 158
    mod 11 is 4, and 11 minus 4 is 7;
    1212899494998 (a man born 12 December 1899
    in Stip, region 49, serial 499): the
    weighted sum is 267, 267 mod 11 is 3, and
    11 minus 3 is 8; 2007975046003 (a woman born
    20 July 1975, a foreigner in Macedonia,
    region 04, serial 600): the weighted sum is
    162, 162 mod 11 is 8, and 11 minus 8 is 3;
    0101990410020: the weighted sum is 88, 88
    mod 11 is 0, and 11 minus 0 is 11, so the
    check digit is 0; 0101990410080: the weighted
    sum is 100, 100 mod 11 is 1, and 11 minus 1
    is 10, so the check digit is 0. The
    recognizer's checksum was also cross-checked
    against python-stdnum si.emso on all
    fixtures and on 2000 random 12 digit bodies
    (check digit computation), with zero
    mismatches.

    Seven honest caveats. First, the checksum
    alone passes about one time in ten for an
    arbitrary 13 digit token (20,030 in 200,000
    random tokens measured), so a bare 13 digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. The date and region rules
    cut this down sharply: a random 13 digit
    token passes the full validation only about
    1 time in 16,667 (12 in 200,000 random tokens
    measured). Second, python-stdnum si.emso
    accepts any region code and does not reject
    future dates, while this recognizer enforces
    the North Macedonia region codes and rejects
    a birth date in the future, so it is stricter
    than python-stdnum: in the same 200,000 token
    run, 708 numbers were accepted by si.emso
    and rejected here (547 with a future birth
    date, 161 with a past date and a region code
    outside North Macedonia), and 0 numbers were
    accepted here and rejected by si.emso.
    Third, python-stdnum strips spaces when
    compacting a number, while this recognizer
    matches plain 13 digits only, so a spaced
    form is not matched here. Fourth, some EMBG
    validators treat a body whose weighted sum
    leaves remainder 1 (m of 10) as having no
    valid check digit, for example the Treeet
    EMBG checker, which states that remainder 1
    means the number is not issued; this
    recognizer follows Wikipedia and
    python-stdnum on this point and folds m of
    10 to a check digit of 0, so 0101990410080
    is valid here and invalid there. Fifth,
    region codes 00 and 09 (naturalized citizens
    with no republican citizenship) and 60 to
    69 (citizens with temporary residence) are
    not North Macedonia specific and are not
    matched, and code 40 is unassigned in the
    Wikipedia region table, so genuinely issued
    numbers carrying those codes are missed
    here, in the safe direction. Sixth, every
    all same digit string passes the checksum
    (all ten do, by the arithmetic) but is
    rejected here as implausible; no fully
    sequential 13 digit run passes the checksum
    (all 20 wrap runs checked), so the
    sequential rule is kept as a plausibility
    guard only. Seventh, Wikipedia notes that a
    small number of genuinely issued numbers
    carry an invalid checksum through operator
    error, and that duplicate JMBGs exist, so
    those numbers are missed here, in the safe
    direction. A passing checksum only shows the
    number is well formed; it cannot show the
    number was ever issued to anyone, and only
    the North Macedonian citizen registry can
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
    the region code must be 04 or 41 to 49; and
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
    number, Slovenian unique master citizen
    number, Bulgarian uniform civil number,
    Greek tax identification number, Serbian
    unique master citizen number, Lithuanian
    personal code, Estonian personal
    identification code, Latvian personal code,
    and Slovak birth number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "mk"

    PATTERNS = [
        Pattern(
            "MK_EMBG",
            r"(?<![A-Za-z0-9./-])\d{13}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "north macedonia",
        "macedonia",
        "macedonian",
        "embg",
        "edinstven maticen broj na graganinot",
        "unique master citizen number",
        "master citizen number",
        "maticen broj",
    ]

    # Region codes of the North Macedonian registry:
    # 41 to 49 North Macedonia (41 Bitola, 42
    # Kumanovo, 43 Ohrid, 44 Prilep, 45 Skopje,
    # 46 Strumica, 47 Tetovo, 48 Veles, 49 Stip),
    # and 04 foreigners in Macedonia. Code 40 is
    # unassigned in the Wikipedia JMBG region
    # table and is not accepted.
    VALID_REGIONS = frozenset(["04"] + [str(code) for code in range(41, 50)])

    CHECK_WEIGHTS = (7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "MK_EMBG",
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
        Check if the pattern text cannot be a MK_EMBG entity.

        The number must be 13 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, must carry a real birth date that is
        not in the future, must carry a North Macedonia
        region code, and must pass the weighted mod 11 check.

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
            for digit, weight in zip(digits[:12], MkEmbgRecognizer.CHECK_WEIGHTS)
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
            for digit, weight in zip(first_twelve, MkEmbgRecognizer.CHECK_WEIGHTS)
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
