from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class EcCedulaRecognizer(PatternRecognizer):
    """
    Recognize Ecuador Cedula de identidad numbers (cedulas).

    The cedula is the 10 digit national identity card number
    issued by the Direccion General de Registro Civil,
    Identificacion y Cedulacion. It is also the first 10 digits
    of the personal RUC (Registro Unico de Contribuyentes) for
    individuals, whose RUC appends 001. The number has three
    parts:

    - The first two digits are a province code: 01 to 24 for
      the provinces, plus 30 and 50 for the special codes
      used outside the province list.
    - The third digit is 0 to 6 for a personal identity
      number (7 and above belong to other RUC types).
    - Digits 4 to 9 are a serial, and the tenth digit is a
      check digit.

    Checksum: the check digit uses a mod-10 algorithm. The
    ten digits are multiplied from left to right by weights
    alternating 2, 1, 2, 1, and so on. A product above 9 has
    9 subtracted from it (so 14 folds to 5). The folded
    products are summed, and a valid number has a sum that
    is a multiple of 10. This is the algorithm implemented
    by python-stdnum for the Ecuadorian CI, whose worked
    example is 171430710-3: the folded weighted sum is 30,
    a multiple of 10, so the check digit 3 is valid. The
    recognizer validates the checksum in invalidate_result
    and keeps the pattern score at 0.5: it does not promote
    a match to 1.0, because a mod-10 check passes about
    one time in ten for an arbitrary same-shape token.

    Invalidation is based on: the number must be exactly
    10 digits after removing the hyphen; the province code
    must be 01 to 24, 30, or 50; the third digit must be 0
    to 6; the six digit serial must not be all zeros; the
    number must not be all the same digit; a fully
    sequential digit run is rejected; and the mod-10
    checksum must hold. This follows the approach of the
    Pakistani CNIC, Saudi national ID, UAE Emirates ID,
    Egyptian national ID, Bangladesh NID, Sri Lanka NIC,
    Ghana Card, Kazakhstan IIN, Ukraine RNOKPP, Uzbekistan
    PINFL, Morocco CNIE, Indonesia NIK, Malaysia MyKad,
    Vietnam CCCD, and Chile RUT recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "ec"

    PATTERNS = [
        Pattern(
            "EC_CEDULA",
            r"(?<![A-Za-z0-9.-])(?:\d{10}|\d{9}-\d)(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "ecuador",
        "ecuadorian",
        "cedula",
        "cedula de identidad",
        "cedula de ciudadania",
        "identity card",
        "national id",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "EC_CEDULA",
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
        Check if the pattern text cannot be an EC_CEDULA entity.

        The number must be 10 digits after removing the
        hyphen, the province code must be 01 to 24, 30, or
        50, the third digit must be 0 to 6, the serial must
        not be all zeros, the number must not be all the
        same digit, a fully sequential digit run is
        rejected, and the mod-10 checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) != 10 or not text.isdigit():
            return True
        province = int(text[:2])
        if not (1 <= province <= 24 or province in (30, 50)):
            return True
        if int(text[2]) > 6:
            return True
        if text[3:9] == "000000":
            # a serial of all zeros is never issued
            return True
        if len(set(text)) == 1:
            # cannot be all the same digit
            return True
        if self._is_sequential(text):
            # a real issued cedula is never one fully sequential run
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the mod-10 checksum of a 10 digit cedula.

        Digits are multiplied from left to right by weights
        alternating 2, 1. A product above 9 has 9 subtracted
        from it. The number is valid when the folded sum is
        a multiple of 10.

        :param digits: The 10 digit cedula, hyphen removed
        :return: True when the checksum holds
        """
        total = 0
        for index, digit in enumerate(digits):
            weight = 2 if index % 2 == 0 else 1
            product = int(digit) * weight
            total += product - 9 if product > 9 else product
        return total % 10 == 0

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the check digit for the first 9 digits.

        The check digit is the value that makes the folded
        weighted sum of all 10 digits a multiple of 10:
        10 minus the folded sum of the body modulo 10,
        with 10 mapping to 0.

        :param body: The first 9 digits of the cedula
        :return: The check digit, 0 to 9
        """
        total = 0
        for index, digit in enumerate(body):
            weight = 2 if index % 2 == 0 else 1
            product = int(digit) * weight
            total += product - 9 if product > 9 else product
        return str((10 - (total % 10)) % 10)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 1234567890 ascends by one each step, and
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
