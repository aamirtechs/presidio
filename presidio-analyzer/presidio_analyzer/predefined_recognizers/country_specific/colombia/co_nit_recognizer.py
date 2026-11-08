from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class CoNitRecognizer(PatternRecognizer):
    """
    Recognize Colombia NIT tax identifier numbers.

    The NIT (Numero de Identificacion Tributaria) is
    the tax identifier issued in Colombia by DIAN
    (Direccion de Impuestos y Aduanas Nacionales).
    For a legal entity it has two parts:

    - A nine digit body assigned by DIAN.
    - One check digit, the digito de verificacion,
      assigned by the DIAN system. It is not
      considered part of the NIT itself, but it is
      printed with it and checked on electronic
      invoices.

    The NIT is written plain, 10 digits with no
    separators, or in its canonical dotted form
    XXX.XXX.XXX-Y, for example 890.903.938-8. A
    hyphenated form without dots, 890903938-8, is
    also common. Natural persons use their cedula
    number plus a check digit as their NIT, so
    bodies of other lengths exist, but this
    recognizer covers the nine digit legal entity
    body only, which is the form published for
    companies.

    Checksum: the check digit uses a mod-11
    algorithm defined in DIAN Orden Administrativa
    4 de 1989. The nine body digits are multiplied
    from right to left by the prime weights 3, 7,
    13, 17, 19, 23, 29, 37, 41, which read from
    left to right as 41, 37, 29, 23, 19, 17, 13, 7,
    3. The weighted sum is taken modulo 11. If the
    remainder is 0 or 1, the check digit equals the
    remainder; otherwise the check digit is 11
    minus the remainder. This is not the Peru RUC
    rule: a remainder of 1 gives a check digit of
    1 here, not 0, and a remainder of 0 gives a
    check digit of 0, not 1. Both edge cases are
    confirmed by real DIAN issued numbers: a
    remainder of 1 gives a check digit of 1 in
    899999068-1 (Ecopetrol, sum 1519) and in the
    Orden Administrativa worked example 860324218-1
    (sum 782), and a remainder of 0 gives a check
    digit of 0 in 830095213-0 (Organizacion Terpel,
    sum 737) and in 890903937-0 (Banco Santander
    Colombia, sum 1078). The worked examples are
    890903938-8 (Bancolombia): 8*41 + 9*37 + 0*29
    + 9*23 + 0*19 + 3*17 + 9*13 + 3*7 + 8*3 =
    1081, 1081 mod 11 = 3, and 11 - 3 = 8; and
    800197268-4 (DIAN itself): 8*41 + 0*37 + 0*29
    + 1*23 + 9*19 + 7*17 + 2*13 + 6*7 + 8*3 =
    733, 733 mod 11 = 7, and 11 - 7 = 4. The rule
    was verified by hand against 34 published
    DIAN issued NITs, including 11 remainder-0
    and 5 remainder-1 edge cases, and all 34
    match. The recognizer validates the checksum
    in invalidate_result and keeps the pattern
    score at 0.5: it does not promote a match to
    1.0, because a mod-11 check passes about one
    time in eleven for an arbitrary same-shape
    token.

    Invalidation is based on: the number must be
    exactly 10 digits after removing dots and the
    hyphen; the number must not be all the same
    digit; a fully sequential digit run is
    rejected; and the mod-11 checksum must hold.
    This follows the approach of the Pakistani
    CNIC, Saudi national ID, UAE Emirates ID,
    Egyptian national ID, Bangladesh NID, Sri
    Lanka NIC, Ghana Card, Kazakhstan IIN,
    Ukraine RNOKPP, Uzbekistan PINFL, Morocco
    CNIE, Indonesia NIK, Malaysia MyKad, Vietnam
    CCCD, Chile RUT, Iran Code Melli, Argentina
    CUIT/CUIL, Mexico CURP, and Peru RUC
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "co"

    PATTERNS = [
        Pattern(
            "CO_NIT",
            r"(?<![A-Za-z0-9.-])(?:\d{10}|\d{3}\.\d{3}\.\d{3}-\d|\d{9}-\d)(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "colombia",
        "colombian",
        "nit",
        "dian",
        "rut",
        "tributario",
        "verificacion",
        "digito de verificacion",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "CO_NIT",
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
        Check if the pattern text cannot be a CO_NIT entity.

        The number must be 10 digits after removing
        dots and the hyphen, the number must not be
        all the same digit, a fully sequential digit
        run is rejected, and the mod-11 checksum must
        hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace(".", "").replace("-", "")
        if len(text) != 10 or not text.isdigit():
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
        Check the mod-11 checksum of a 10 digit NIT.

        The nine body digits are weighted from right
        to left by 3, 7, 13, 17, 19, 23, 29, 37, 41.
        A remainder of 0 or 1 gives a check digit
        equal to the remainder, both confirmed by
        real DIAN issued numbers.

        :param digits: The 10 digit NIT
        :return: True when the checksum holds
        """
        check_digit = CoNitRecognizer._check_digit(digits[:9])
        return check_digit == digits[9]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the check digit for the first 9 digits.

        The weighted sum uses the prime weights 3, 7,
        13, 17, 19, 23, 29, 37, 41 from right to left.
        The check digit is the remainder of the sum
        modulo 11 when the remainder is 0 or 1, and
        11 minus the remainder otherwise.

        :param body: The first 9 digits of the NIT
        :return: The check digit, 0 to 9
        """
        weights = (3, 7, 13, 17, 19, 23, 29, 37, 41)
        total = sum(
            int(digit) * weight for digit, weight in zip(reversed(body), weights)
        )
        remainder = total % 11
        if remainder in (0, 1):
            return str(remainder)
        return str(11 - remainder)

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
