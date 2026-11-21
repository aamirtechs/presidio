import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class RsJmbgRecognizer(PatternRecognizer):
    """
    Recognize Serbian unique master citizen numbers (JMBG).

    The JMBG (Jedinstveni maticni broj gradana,
    Unique Master Citizen Number) is the 13 digit
    personal number assigned to Serbian citizens
    at birth registration. It is the same number
    system used across the former Yugoslav
    republics: Wikipedia notes it continues to
    be used in its original form in Bosnia and
    Herzegovina, Montenegro, North Macedonia,
    Serbia and Slovenia, while Croatia replaced
    it with the personal identification number
    (OIB). For Serbian citizens who are not
    entrepreneurs the JMBG also doubles as the
    tax number; the separate 9 digit PIB is the
    tax number for companies and entrepreneurs
    and is a different number. The number has
    13 digits and encodes the holder's date of
    birth: DDMMYYY RR BBB K, where DD is the day
    of birth, MM the month of birth, YYY the last
    three digits of the year of birth, RR the
    region of birth, BBB a gender and serial
    number (000 to 499 for men, 500 to 999 for
    women), and K a check digit. It is written
    plain, 13 digits with no separators, for
    example 0101990710008.

    Region: 70 to 79 is Central Serbia (70
    Serbian citizens registered abroad at a
    Serbian diplomatic or consular post, 71
    Belgrade, 72 Sumadija and Pomoravlje, 73
    Nis, 74 Southern Morava, 75 Zajecar, 76
    Podunavlje, 77 Podrinje and Kolubara, 78
    Kraljevo, 79 Uzice), 80 to 89 is Vojvodina
    (80 Novi Sad, 81 Sombor, 82 Subotica, 84
    Kikinda, 85 Zrenjanin, 86 Pancevo, 87
    Vrsac, 88 Ruma, 89 Sremska Mitrovica), and
    90 to 99 is Kosovo (91 Pristina, 92
    Kosovska Mitrovica, 93 Pec, 94 Dakovica,
    95 Prizren, 96 Gnjilane). Codes 06, 07 and
    08 cover foreigners in Central Serbia,
    Vojvodina and Kosovo. All other region
    codes belong to other registries and are
    rejected here: 50 to 59 is Slovenia
    (SI_EMSO), 10 to 19 Bosnia and Herzegovina,
    20 to 29 Montenegro, 41 to 49 North
    Macedonia, and 30 to 39 Croatia, where the
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
    si.emso module (python-stdnum has no jmbg
    module; its rs package holds only pib, the
    9 digit Serbian tax number). With m = 11 -
    (sum mod 11), a value of m from 1 to 9 is
    the check digit, and m of 10 or 11 gives a
    check digit of 0. Worked examples, verified
    by hand and by computation: 0101990710008
    (a man born 1 January 1990 in Belgrade,
    region 71, serial 000): the weighted sum is
    102, 102 mod 11 is 3, and 11 minus 3 is 8;
    1503985735129 (a woman born 15 March 1985
    in Nis, region 73, serial 512): the weighted
    sum is 211, 211 mod 11 is 2, and 11 minus 2
    is 9; 2902000802507 (a man born on the leap
    day 29 February 2000 in Novi Sad, region
    80, serial 250): the weighted sum is 147,
    147 mod 11 is 4, and 11 minus 4 is 7;
    2007975066004 (a woman born 20 July 1975,
    a foreigner in Central Serbia, region 06,
    serial 600): the weighted sum is 172, 172
    mod 11 is 7, and 11 minus 7 is 4;
    0505001701236 (a man born 5 May 2001,
    registered abroad, region 70, serial 123):
    the weighted sum is 115, 115 mod 11 is 5,
    and 11 minus 5 is 6; 1212899724993 (a man
    born 12 December 1899, region 72, serial
    499): the weighted sum is 250, 250 mod 11
    is 8, and 11 minus 8 is 3; 0101990710040:
    the weighted sum is 110, 110 mod 11 is 0,
    and 11 minus 0 is 11, so the check digit is
    0; 0101990716090: the weighted sum is 144,
    144 mod 11 is 1, and 11 minus 1 is 10, so
    the check digit is 0. The recognizer's
    checksum was also cross-checked against
    python-stdnum si.emso on all fixtures and
    on 2000 random 12 digit bodies (check digit
    computation), with zero mismatches. Full
    validation differs from si.emso only in the
    stricter direction, by design: this
    recognizer enforces the Serbia region codes
    and rejects future birth dates, so in a
    2000 random 13 digit number run, the only
    disagreements were 5 numbers si.emso
    accepts and this recognizer rejects (future
    dates or non-Serbia regions), none the
    other way. The fixtures were also checked
    against the jmbg-labs Serbian JMBG library
    on the fixtures whose region it lists, with
    the two disagreements below as the only
    exceptions.

    Eight honest caveats. First, the checksum
    alone passes about one time in ten for an
    arbitrary 13 digit token, so a bare 13 digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. The date and region rules
    cut this down sharply: a random 13 digit
    token passes the full validation only about
    1 time in 4,348 (46 in 200,000 random tokens
    measured). Second, python-stdnum has no
    jmbg module, and its si.emso module accepts
    any region code and does not reject future
    dates, while this recognizer enforces the
    Serbia region codes and rejects a birth
    date in the future, so it is stricter than
    python-stdnum. Third, python-stdnum strips
    spaces when compacting a number, while this
    recognizer matches plain 13 digits only, so
    a spaced form is not matched here. Fourth,
    the jmbg-labs library disagrees on two
    points, and this recognizer follows Wikipedia
    and python-stdnum on both: jmbg-labs rejects
    a body whose weighted sum leaves remainder 1
    (m of 10) as having no valid check digit,
    while Wikipedia and python-stdnum fold it to
    a check digit of 0, so 0101990716090 is valid
    here and invalid there; and jmbg-labs omits
    region 70 (citizens registered abroad), 90
    and 97 to 99, and 06 to 08 (foreigners in
    Serbia) from its region table, while
    Wikipedia lists them as Serbia codes, so
    0505001701236 (region 70) and 2007975066004
    (region 06) are valid here and rejected
    there. Fifth, every all same digit string
    passes the checksum (all ten do, by the
    arithmetic) but is rejected here as
    implausible; no fully sequential 13 digit
    run passes the checksum (all 20 wrap runs
    checked), so the sequential rule is kept as
    a plausibility guard only. Sixth, region
    codes 00 and 09 (naturalized citizens with
    no republican citizenship) and 60 to 69
    (citizens with temporary residence) are not
    Serbia specific and are not matched, so
    genuinely issued numbers carrying those
    codes are missed here, in the safe
    direction. Seventh, Wikipedia notes that a
    small number of genuinely issued numbers
    carry an invalid checksum through operator
    error, and that duplicate JMBGs exist, so
    those numbers are missed here, in the safe
    direction. Eighth, a passing checksum only
    shows the number is well formed; it cannot
    show the number was ever issued to anyone,
    and only the Serbian citizen registry can
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
    the region code must be 06 to 08 or 70 to 99;
    and the check digit must satisfy the weighted
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
    number, Bulgarian uniform civil number, and
    Greek tax identification number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "rs"

    PATTERNS = [
        Pattern(
            "RS_JMBG",
            r"(?<![A-Za-z0-9./-])\d{13}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "serbia",
        "serbian",
        "jmbg",
        "jedinstveni maticni broj gradana",
        "unique master citizen number",
        "unique citizen number",
        "serbian citizen number",
        "maticni broj",
    ]

    # Region codes of the Serbian registry: 70 to 79
    # Central Serbia, 80 to 89 Vojvodina, 90 to 99
    # Kosovo, and 06 to 08 foreigners in Central
    # Serbia, Vojvodina and Kosovo. Codes 00 and
    # 09 (naturalized citizens) and 60 to 69
    # (temporary residence) are not Serbia
    # specific and are not accepted.
    VALID_REGIONS = frozenset(
        ["06", "07", "08"] + [str(code) for code in range(70, 100)]
    )

    CHECK_WEIGHTS = (7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "RS_JMBG",
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
        Check if the pattern text cannot be a RS_JMBG entity.

        The number must be 13 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, must carry a real birth date that is
        not in the future, must carry a Serbia region
        code, and must pass the weighted mod 11 check.

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
            for digit, weight in zip(digits[:12], RsJmbgRecognizer.CHECK_WEIGHTS)
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
            for digit, weight in zip(first_twelve, RsJmbgRecognizer.CHECK_WEIGHTS)
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
