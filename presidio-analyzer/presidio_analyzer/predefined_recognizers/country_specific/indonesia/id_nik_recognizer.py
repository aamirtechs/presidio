import calendar
from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class IdNikRecognizer(PatternRecognizer):
    """
    Recognize Indonesia National Identity Numbers (NIK).

    The NIK (Nomor Induk Kependudukan) printed on the Kartu Tanda
    Penduduk (KTP) is a 16-digit number with three parts:

    - Digits 1 to 6: the area code where the number was first
      registered (2-digit province, 2-digit regency or city,
      2-digit sub-district), assigned by the Ministry of Home
      Affairs.
    - Digits 7 to 12: the date of birth as DDMMYY. Because the
      number carries no gender code, 40 is added to the day for
      women, so a female day field runs from 41 to 71 while a male
      day field runs from 01 to 31.
    - Digits 13 to 16: a serial number unique within the area and
      birth date.

    Checksum note: the NIK has no checksum. Microsoft Learn's
    Indonesia identity card entity definition states the checksum
    explicitly as "No", and the published structure sources agree
    the last four digits are a serial number, not a check value, so
    this recognizer implements no checksum and deliberately does
    not promote matches on an unverified one. Invalidation is based
    on structural sanity checks instead: the value must be exactly
    16 digits, the area code must not be 000000, the serial must
    not be 0000, the date part must decode to a real calendar date
    that is not in the future (with the two-digit year mapped to
    the 1900s when it is greater than the current two-digit year
    and to the 2000s otherwise), and trivial numbers such as
    all-same-digit or fully sequential digit runs are rejected.
    This follows the approach of the Pakistani CNIC, Saudi
    national ID, UAE Emirates ID, Egyptian national ID,
    Bangladesh NID, Sri Lanka NIC, Ghana Card, Kazakhstan IIN,
    Ukraine RNOKPP, Uzbekistan PINFL and Morocco CNIE
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "id"

    PATTERNS = [
        Pattern(
            "ID_NIK",
            r"(?<![A-Za-z0-9])\d{16}(?![A-Za-z0-9])",
            0.5,
        ),
    ]

    CONTEXT = [
        "indonesia",
        "indonesian",
        "nik",
        "nomor induk kependudukan",
        "ktp",
        "kartu tanda penduduk",
        "national identity",
        "identity card",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "ID_NIK",
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
        Check if the pattern text cannot be an ID_NIK entity.

        No checksum exists for this recognizer (see the class
        docstring), so invalidation is based on sanity checks: the
        value must be exactly 16 digits, the area code must not be
        000000, the serial must not be 0000, the date part must
        decode to a real calendar date that is not in the future,
        and trivial numbers such as all-same-digit or fully
        sequential digit runs are rejected.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 16 or not text.isdigit():
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued NIK is never one fully sequential run
            return True
        if text[:6] == "000000":
            return True
        if text[12:] == "0000":
            return True
        return self._birth_date_is_invalid(text[6:12])

    @staticmethod
    def _birth_date_is_invalid(date_part: str) -> bool:
        """
        Check the DDMMYY birth date part of a NIK.

        The raw day runs 01 to 31 for men and 41 to 71 for women
        (40 added to the day). The two-digit year maps to the
        1900s when it is greater than the current two-digit year
        and to the 2000s otherwise. The decoded date must be a
        real calendar date (leap-aware) and must not be in the
        future.

        :param date_part: Six digits, DDMMYY with the female day offset
        :return: True if the date part cannot be a real past birth date
        """
        raw_day = int(date_part[0:2])
        month = int(date_part[2:4])
        year_two_digits = int(date_part[4:6])
        if 1 <= raw_day <= 31:
            day = raw_day
        elif 41 <= raw_day <= 71:
            day = raw_day - 40
        else:
            return True
        if not 1 <= month <= 12:
            return True
        current_two_digits = date.today().year % 100
        if year_two_digits > current_two_digits:
            year = 1900 + year_two_digits
        else:
            year = 2000 + year_two_digits
        if day > calendar.monthrange(year, month)[1]:
            return True
        try:
            birth_date = date(year, month, day)
        except ValueError:
            return True
        return birth_date > date.today()

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 1234567890123456 ascends by one each step, and
        6543210987654321 descends by one each step.

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
