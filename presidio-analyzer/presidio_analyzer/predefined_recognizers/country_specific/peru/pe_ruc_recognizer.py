from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class PeRucRecognizer(PatternRecognizer):
    """
    Recognize Peru RUC tax identifier numbers.

    The RUC (Registro Unico de Contribuyentes) is the
    11 digit tax identifier issued in Peru by SUNAT
    (Superintendencia Nacional de Aduanas y de
    Administracion Tributaria). It has three parts:

    - The first two digits are a taxpayer type prefix:
      10 for a natural person identified with a DNI,
      15 for a natural person identified with another
      document, 17 for a foreigner or others (also the
      prefix of RUCs issued from 1993 to 2000), and 20
      for a legal entity. Prefix 16 was used in the
      past but is no longer issued, so it is rejected.
    - The next eight digits are the identifier: the DNI
      for a prefix 10 RUC, otherwise a number assigned
      by SUNAT.
    - The eleventh digit is a check digit.

    Checksum: the check digit uses a mod-11 algorithm.
    The first ten digits are multiplied from left to
    right by weights 5, 4, 3, 2, 7, 6, 5, 4, 3, 2. The
    weighted sum is taken modulo 11. The check digit
    is 11 minus the remainder, except that a result of
    10 maps to a check digit of 0 and a result of 11
    maps to a check digit of 1. So a remainder of 1
    gives a check digit of 0, and a remainder of 0
    gives a check digit of 1. Both edge cases are
    confirmed by real SUNAT issued numbers: 20100070970
    (Supermercados Peruanos S.A., remainder 1, check
    digit 0) and 20612504301 (Negocios Inmobiliarios
    Dream House S.A.C., remainder 0, check digit 1).
    The worked examples are 20503644968: 2*5 + 0*4 +
    5*3 + 0*2 + 3*7 + 6*6 + 4*5 + 4*4 + 9*3 + 6*2 =
    157, 157 mod 11 = 3, and 11 - 3 = 8; and
    20131312955 (SUNAT itself): 2*5 + 0*4 + 1*3 + 3*2
    + 1*7 + 3*6 + 1*5 + 2*4 + 9*3 + 5*2 = 94, 94 mod
    11 = 6, and 11 - 6 = 5. The edge cases work out
    as 20100070970: 2*5 + 0*4 + 1*3 + 0*2 + 0*7 + 0*6
    + 7*5 + 0*4 + 9*3 + 7*2 = 89, 89 mod 11 = 1, and
    11 - 1 = 10, mapped to 0; and 20612504301: 2*5 +
    0*4 + 6*3 + 1*2 + 2*7 + 5*6 + 0*5 + 4*4 + 3*3 +
    0*2 = 99, 99 mod 11 = 0, and 11 - 0 = 11, mapped
    to 1. The recognizer validates the checksum in
    invalidate_result and keeps the pattern score at
    0.5: it does not promote a match to 1.0, because
    a mod-11 check passes about one time in eleven
    for an arbitrary same-shape token.

    Invalidation is based on: the number must be
    exactly 11 digits; the first two digits must be
    one of 10, 15, 17, or 20 (16 and every other
    prefix are not issued and are rejected); the
    number must not be all the same digit; a fully
    sequential digit run is rejected; and the mod-11
    checksum must hold. This follows the approach of
    the Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh NID,
    Sri Lanka NIC, Ghana Card, Kazakhstan IIN, Ukraine
    RNOKPP, Uzbekistan PINFL, Morocco CNIE, Indonesia
    NIK, Malaysia MyKad, Vietnam CCCD, Chile RUT,
    Iran Code Melli, Argentina CUIT/CUIL, and Mexico
    CURP recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "pe"

    PATTERNS = [
        Pattern(
            "PE_RUC",
            r"(?<![A-Za-z0-9.-])\d{11}(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "peru",
        "peruvian",
        "ruc",
        "registro unico",
        "contribuyentes",
        "sunat",
        "dni",
    ]

    ALLOWED_PREFIXES = {"10", "15", "17", "20"}

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "PE_RUC",
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
        Check if the pattern text cannot be a PE_RUC entity.

        The number must be 11 digits, the first two
        digits must be an allowed prefix, the number
        must not be all the same digit, a fully
        sequential digit run is rejected, and the
        mod-11 checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip()
        if len(text) != 11 or not text.isdigit():
            return True
        if text[:2] not in self.ALLOWED_PREFIXES:
            # only 10, 15, 17, and 20 are issued prefixes
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
        Check the mod-11 checksum of an 11 digit RUC.

        The first ten digits are weighted 5, 4, 3, 2, 7,
        6, 5, 4, 3, 2. A remainder of 1 gives a check
        digit of 0 and a remainder of 0 gives a check
        digit of 1, both confirmed by real SUNAT issued
        numbers.

        :param digits: The 11 digit RUC
        :return: True when the checksum holds
        """
        check_digit = PeRucRecognizer._check_digit(digits[:10])
        return check_digit == digits[10]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the check digit for the first 10 digits.

        The weighted sum uses weights 5, 4, 3, 2, 7, 6,
        5, 4, 3, 2. The check digit is 11 minus the
        sum modulo 11, except that a result of 10 maps
        to 0 and a result of 11 maps to 1.

        :param body: The first 10 digits of the RUC
        :return: The check digit, 0 to 9
        """
        weights = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
        total = sum(int(digit) * weight for digit, weight in zip(body, weights))
        result = 11 - (total % 11)
        if result == 10:
            return "0"
        if result == 11:
            return "1"
        return str(result)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 01234567890 ascends by one each step, and
        10987654321 descends by one each step.

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
