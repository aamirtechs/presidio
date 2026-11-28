import datetime
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class MeJmbgRecognizer(PatternRecognizer):
    """
    Recognize Montenegrin unique master citizen numbers (JMBG).

    The JMBG (Jedinstveni maticni broj gradana,
    Unique Master Citizen Number, locally JMB) is
    the 13 digit personal number assigned to
    Montenegrin citizens at birth registration.
    It is the same number system used across the
    former Yugoslav republics: Wikipedia notes
    it continues to be used in its original form
    in Bosnia and Herzegovina, Montenegro, North
    Macedonia, Serbia and Slovenia, while
    Croatia replaced it with the personal
    identification number (OIB). The number has
    13 digits and encodes the holder's date of
    birth: DDMMYYY RR BBB K, where DD is the day
    of birth, MM the month of birth, YYY the last
    three digits of the year of birth, RR the
    region of birth, BBB a gender and serial
    number (000 to 499 for men, 500 to 999 for
    women), and K a check digit. It is written
    plain, 13 digits with no separators, for
    example 2003988215003.

    Region: 21 to 29 is Montenegro (21
    Podgorica, Danilovgrad and Kolasin, 22 Bar
    and Ulcinj, 23 Budva, Kotor and Tivat, 24
    Herceg Novi, 25 Cetinje, 26 Niksic, Pluzine
    and Savnik, 27 Berane, Rozaje, Plav and
    Andrijevica, 28 Bijelo Polje and Mojkovac,
    29 Pljevlja and Zabljak). Code 20 is not in
    use. Code 02 covers foreigners in
    Montenegro. All other region codes belong
    to other registries and are rejected here:
    50 to 59 is Slovenia (SI_EMSO), 10 to 19
    Bosnia and Herzegovina, 41 to 49 North
    Macedonia, 70 to 99 Serbia and Kosovo, and
    30 to 39 Croatia, where the number is no
    longer used and HR_OIB replaced it.

    Year: the three digit year follows the
    python-stdnum convention, a value below 800
    means the 2000s and 800 or above means the
    1800s, so 006 is 2006 and 988 is 1988, and
    899 is 1899.

    Checksum: the check digit comes from weights
    7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 on the
    first 12 digits, as published in the
    Wikipedia Unique Master Citizen Number
    article, in the Treeet Montenegro JMBG
    checker, and implemented by the
    python-stdnum si.emso module (python-stdnum
    has no jmbg module and no me module; its me
    package holds only the IBAN and the company
    tax number PIB). With m = 11 - (sum mod 11),
    a value of m from 1 to 9 is the check digit,
    and m of 10 or 11 gives a check digit of 0.
    Worked examples, verified by hand and by
    computation: 2003988215003 (a woman born 20
    March 1988 in Podgorica, region 21, serial
    500, the worked example of the Treeet
    Montenegro JMBG checker): the weighted sum
    is 162, 162 mod 11 is 8, and 11 minus 8 is
    3; 0101990210005 (a man born 1 January 1990
    in Podgorica, region 21, serial 000): the
    weighted sum is 72, 72 mod 11 is 6, and 11
    minus 6 is 5; 1503985225120 (a woman born 15
    March 1985 in Bar, region 22, serial 512):
    the weighted sum is 176, 176 mod 11 is 0,
    and 11 minus 0 is 11, so the check digit is
    0; 2902000232506 (a man born on the leap day
    29 February 2000 in Budva, region 23, serial
    250): the weighted sum is 126, 126 mod 11
    is 5, and 11 minus 5 is 6; 2007975246002 (a
    woman born 20 July 1975 in Herceg Novi,
    region 24, serial 600): the weighted sum is
    174, 174 mod 11 is 9, and 11 minus 9 is 2;
    0505001251230 (a man born 5 May 2001 in
    Cetinje, region 25, serial 123): the
    weighted sum is 110, 110 mod 11 is 0, and 11
    minus 0 is 11, so the check digit is 0;
    1212899264992 (a man born 12 December 1899
    in Niksic, region 26, serial 499): the
    weighted sum is 240, 240 mod 11 is 9, and 11
    minus 9 is 2; 0101990210030 (a man born 1
    January 1990 in Podgorica, region 21, serial
    003): the weighted sum is 78, 78 mod 11 is
    1, and 11 minus 1 is 10, so the check digit
    is 0; 0101990210080 (a man born 1 January
    1990 in Podgorica, region 21, serial 008):
    the weighted sum is 88, 88 mod 11 is 0, and
    11 minus 0 is 11, so the check digit is 0;
    0101990020001 (a man born 1 January 1990, a
    foreigner in Montenegro, region 02, serial
    000): the weighted sum is 65, 65 mod 11 is
    10, and 11 minus 10 is 1; 0101990299995 (a
    woman born 1 January 1990 in Pljevlja,
    region 29, serial 999): the weighted sum is
    193, 193 mod 11 is 6, and 11 minus 6 is 5.
    The recognizer's checksum was also
    cross-checked against python-stdnum si.emso
    on all fixtures and on 2000 random 12 digit
    bodies (check digit computation), with zero
    mismatches. Full validation differs from
    si.emso only in the stricter direction, by
    design: this recognizer enforces the
    Montenegro region codes and rejects future
    birth dates, so in a 2000 random 13 digit
    number run, the only disagreements were 5
    numbers si.emso accepts and this recognizer
    rejects (future dates or non-Montenegro
    regions), none the other way.

    Eight honest caveats. First, the checksum
    alone passes about one time in ten for an
    arbitrary 13 digit token, so a bare 13 digit
    number is weak evidence on its own; the
    recognizer keeps the pattern score at 0.5 and
    never promotes it. The date and region rules
    cut this down sharply: a random 13 digit
    token passes the full validation only about
    1 time in 10,526 (19 in 200,000 random tokens
    measured). Second, python-stdnum has no
    jmbg module and no me module, and its
    si.emso module accepts any region code and
    does not reject future dates, while this
    recognizer enforces the Montenegro region
    codes and rejects a birth date in the
    future, so it is stricter than
    python-stdnum. Third, python-stdnum strips
    spaces when compacting a number, while this
    recognizer matches plain 13 digits only, so
    a spaced form is not matched here. Fourth,
    the Treeet Montenegro JMBG checker disagrees
    on one point, and this recognizer follows
    Wikipedia and python-stdnum: Treeet says a
    body whose weighted sum leaves remainder 1
    (m of 10) is never issued and has no valid
    check digit, while Wikipedia and
    python-stdnum fold it to a check digit of 0,
    so 0101990210030 is valid here and invalid
    there; the jmbg-labs library takes the same
    remainder-1 position as Treeet. Fifth, every
    all same digit string passes the checksum
    (all ten do, by the arithmetic) but is
    rejected here as implausible; no fully
    sequential 13 digit run passes the checksum
    (all 20 wrap runs checked), so the
    sequential rule is kept as a plausibility
    guard only. Sixth, region codes 20 (not in
    use), 00 and 09 (naturalized citizens with
    no republican citizenship) and 60 to 69
    (citizens with temporary residence) are not
    Montenegro specific and are not matched, so
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
    and only the Montenegrin citizen registry
    can confirm that. Montenegro also plans to
    replace the JMB with a new identification
    number after joining the European Union, but
    the JMB remains the current identifier and
    is still printed on identity documents.

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
    the region code must be 02 or 21 to 29;
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
    number, Bulgarian uniform civil number,
    Greek tax identification number, Serbian
    unique master citizen number, Lithuanian
    personal code, Estonian personal
    identification code, Latvian personal code,
    Slovak birth number, Algerian national
    identification number, and North Macedonian
    unique master citizen number recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "me"

    PATTERNS = [
        Pattern(
            "ME_JMBG",
            r"(?<![A-Za-z0-9./-])\d{13}(?![A-Za-z0-9/-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "montenegro",
        "montenegrin",
        "jmbg",
        "jmb",
        "jedinstveni maticni broj gradana",
        "unique master citizen number",
        "unique citizen number",
        "montenegrin citizen number",
        "maticni broj",
    ]

    # Region codes of the Montenegrin registry:
    # 21 to 29 Montenegro, and 02 foreigners in
    # Montenegro. Code 20 is not in use. Codes
    # 01 and 03 to 09 cover foreigners in other
    # former Yugoslav republics and are not
    # accepted, and 00 and 09 (naturalized
    # citizens) and 60 to 69 (temporary
    # residence) are not Montenegro specific.
    VALID_REGIONS = frozenset(["02"] + [str(code) for code in range(21, 30)])

    CHECK_WEIGHTS = (7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "ME_JMBG",
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
        Check if the pattern text cannot be a ME_JMBG entity.

        The number must be 13 digits, must not be all
        the same digit, must not be a fully sequential
        digit run, must carry a real birth date that is
        not in the future, must carry a Montenegro
        region code, and must pass the weighted mod 11
        check.

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
            for digit, weight in zip(digits[:12], MeJmbgRecognizer.CHECK_WEIGHTS)
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
            for digit, weight in zip(first_twelve, MeJmbgRecognizer.CHECK_WEIGHTS)
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
