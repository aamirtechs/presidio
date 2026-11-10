from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class PyRucRecognizer(PatternRecognizer):
    """
    Recognize Paraguay Registro Unico de Contribuyentes (RUC) numbers.

    The RUC is the tax identification number issued
    in Paraguay by the Direccion Nacional de
    Ingresos Tributarios (DNIT), which took over
    from the Subsecretaria de Estado de Tributacion
    (SET) under Law 7143/2023. It is the holder's
    identifier plus one check digit. For
    individuals the identifier is the holder's
    cedula number; refugee applicants get a number
    in the 50,000,000 series and companies one in
    the 80,000,000 series. The identifier is 6 to
    8 digits, so the full number is 7 to 9 digits,
    written plain or in its canonical hyphenated
    form XXXXXXXX-Y, for example 80000519-8.

    Checksum: the check digit uses the mod-11
    method published by the DNIT. Starting from
    the rightmost identifier digit and moving
    left, each digit is multiplied by 2, 3, 4 and
    so on, restarting at 2 after 11. The products
    are added and the sum is divided by 11. If the
    remainder is above 1, the check digit is 11
    minus the remainder; otherwise it is 0. With
    an identifier of at most 8 digits the weights
    run from 2 up to 9, so the restart after 11
    never triggers in practice. This is confirmed
    by three independent sources that agree on
    the rule: python-stdnum, whose py.ruc module
    implements the same calculation; LookupTax,
    which documents the DNIT method; and the
    nationid catalogue, which cross-validated the
    rule against the paraguay-ruc npm package.
    Worked examples, verified by hand: 80000519-8
    (Telefonica Celular del Paraguay S.A.E.,
    listed in DNIT General Resolution 105/2021
    and in public procurement records): the
    products are 9 times 2, 1 times 3, 5 times 4,
    0 times 5, 0 times 6, 0 times 7, 0 times 8
    and 8 times 9, which is 18 plus 3 plus 20
    plus 0 plus 0 plus 0 plus 0 plus 72, for a
    sum of 113; 113 mod 11 is 3, so the check
    digit is 11 minus 3, which is 8; 80028061-0
    (the python-stdnum docstring example, a
    remainder-1 edge case): the products are 1
    times 2, 6 times 3, 0 times 4, 8 times 5, 2
    times 6, 0 times 7, 0 times 8 and 8 times 9,
    which is 2 plus 18 plus 0 plus 40 plus 12
    plus 0 plus 0 plus 72, for a sum of 144;
    144 mod 11 is 1, so the check digit is 0;
    9991603 (the python-stdnum docstring
    example): the products are 0 times 2, 6
    times 3, 1 times 4, 9 times 5, 9 times 6
    and 9 times 7, which is 0 plus 18 plus 4
    plus 45 plus 54 plus 63, for a sum of 184;
    184 mod 11 is 8, so the check digit is 11
    minus 8, which is 3; 11111111-0 (remainder-0
    edge case): the products are 1 times each of
    2, 3, 4, 5, 6, 7, 8 and 9, for a sum of 44;
    44 mod 11 is 0, so the check digit is 0.
    The recognizer's check digit implementation
    was also cross-checked against python-stdnum
    on all fixtures, with zero mismatches. One
    honest caveat: python-stdnum accepts
    identifiers shorter than 6 digits (its
    docstring validates 2660-3), while this
    recognizer requires a 6 to 8 digit
    identifier, because real Paraguayan cedula
    numbers are 6 digits or longer and a shorter
    token would be far too generic to detect
    safely. The recognizer validates the checksum
    in invalidate_result and keeps the pattern
    score at 0.5: it does not promote a match to
    1.0, because a mod-11 check passes about one
    time in eleven for an arbitrary same-shape
    token.

    Invalidation is based on: the number must be
    7 to 9 digits after removing the hyphen; the
    number must not be all the same digit, and
    the identifier must not be all the same digit
    either (note that 00000000-0 and 11111111-0
    both pass the checksum, so these rules are
    needed); and the mod-11 checksum must hold. Unlike the
    cedula recognizers, a fully sequential digit
    run is not rejected: the identifier is a
    sequential registry number (the holder's
    cedula), so a sequential base such as
    12345678 can in principle be a real issued
    number. This follows the approach of the
    Pakistani CNIC, Saudi national ID, UAE
    Emirates ID, Egyptian national ID, Bangladesh
    NID, Sri Lanka NIC, Ghana Card, Kazakhstan
    IIN, Ukraine RNOKPP, Uzbekistan PINFL,
    Morocco CNIE, Indonesia NIK, Malaysia MyKad,
    Vietnam CCCD, Chile RUT, Ecuador Cedula, Iran
    Code Melli, Argentina CUIT/CUIL, Mexico CURP,
    Peru RUC, Colombia NIT, and Dominican Republic
    Cedula recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "py"

    PATTERNS = [
        Pattern(
            "PY_RUC",
            r"(?<![A-Za-z0-9.-])(?:\d{7,9}|\d{6,8}-\d)(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "paraguay",
        "paraguayan",
        "ruc",
        "registro unico de contribuyentes",
        "registro único de contribuyentes",
        "dnit",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "PY_RUC",
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
        Check if the pattern text cannot be a PY_RUC entity.

        The number must be 7 to 9 digits after
        removing the hyphen, neither the number
        nor its identifier may be all the same
        digit, and the mod-11 checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) < 7 or len(text) > 9 or not text.isdigit():
            return True
        if len(set(text)) == 1 or len(set(text[:-1])) == 1:
            # cannot be all the same digit, in the
            # full number or in the identifier alone
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_checksum(digits: str) -> bool:
        """
        Check the mod-11 checksum of a 7 to 9 digit RUC.

        The check digit is the last digit. The
        identifier digits are weighted 2, 3, 4 and
        so on from the right, the products are
        summed, and the remainder mod 11 decides
        the check digit: 0 when the remainder is
        0 or 1, otherwise 11 minus the remainder.

        :param digits: The 7 to 9 digit RUC
        :return: True when the checksum holds
        """
        check_digit = PyRucRecognizer._check_digit(digits[:-1])
        return check_digit == digits[-1]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the mod-11 check digit for the identifier digits.

        The rightmost identifier digit is weighted
        2, the next 3, and so on, restarting at 2
        after 11. The check digit is 0 when the
        weighted sum mod 11 is 0 or 1, otherwise
        11 minus the remainder.

        :param body: The 6 to 8 digit identifier
        :return: The check digit, 0 to 9
        """
        total = 0
        for position, digit in enumerate(reversed(body)):
            weight = (position % 10) + 2
            total += int(digit) * weight
        remainder = total % 11
        if remainder <= 1:
            return "0"
        return str(11 - remainder)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 012345678 ascends by one each step, and
        987654321 descends by one each step. This is kept for
        symmetry with the other country recognizers, but it is
        not used to invalidate a RUC: the identifier is a
        sequential registry number, so a sequential base can
        in principle be a real issued number.

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
