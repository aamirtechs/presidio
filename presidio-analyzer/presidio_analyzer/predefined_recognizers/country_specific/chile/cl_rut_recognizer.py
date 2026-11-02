from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class ClRutRecognizer(PatternRecognizer):
    """
    Recognize Chile Rol Unico Tributario numbers (RUT numbers).

    The RUT is the tax identification number issued by the
    Servicio de Impuestos Internos (SII). For individuals
    resident in Chile the National Identity Number (RUN) given
    by the Civil Registry serves as the RUT, so the two share
    this format. The number has two parts, written with or
    without dot thousand separators:

    - A body of 7 or 8 digits.
    - A hyphen, then a verification digit: 0 to 9 or the
      letter K, for example 12.345.678-5.

    This structure is set out in the OECD Chile TIN
    information sheet, which describes the TIN as a serial
    number, a dash, and a check digit ranging from 0 to 9 or
    the letter K, in the form xx.xxx.xxx-x, verified with
    the mod-11 algorithm.

    Checksum: the verification digit uses the standard
    mod-11 algorithm. Digits are multiplied from right to
    left by weights cycling 2, 3, 4, 5, 6, 7, then the sum
    is taken modulo 11 and subtracted from 11. A result of
    11 maps to 0 and a result of 10 maps to K. The worked
    example in the published algorithm description confirms
    this: body 12345678 gives sum 138, remainder 6, and
    verification digit 5, so 12345678-5. The recognizer
    validates the checksum in invalidate_result and keeps
    the pattern score at 0.5: it does not promote a match
    to 1.0, because a mod-11 check passes about one time in
    eleven for an arbitrary same-shape token.

    Invalidation is based on: the body must be 7 or 8 digits
    and the check character must be 0 to 9 or K; the body
    must not be all the same digit (11111111-1 passes the
    checksum, so this rule is what rejects it); a fully
    sequential digit run formed by the body plus a digit
    check character is rejected (when the check character
    is K there is no digit run to check, and the checksum
    alone gates the value); and the check character must
    equal the mod-11 value computed from the body. This
    follows the approach of the Pakistani CNIC, Saudi
    national ID, UAE Emirates ID, Egyptian national ID,
    Bangladesh NID, Sri Lanka NIC, Ghana Card, Kazakhstan
    IIN, Ukraine RNOKPP, Uzbekistan PINFL, Morocco CNIE,
    Indonesia NIK, Malaysia MyKad and Vietnam CCCD
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "cl"

    PATTERNS = [
        Pattern(
            "CL_RUT",
            r"(?<![A-Za-z0-9.-])(?:\d{1,2}\.\d{3}\.\d{3}|\d{7,8})-[0-9kK](?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "chile",
        "chilean",
        "rut",
        "rol unico tributario",
        "rol unico nacional",
        "rol unico",
        "tax identification number",
        "identity card",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "CL_RUT",
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
        Check if the pattern text cannot be a CL_RUT entity.

        The body must be 7 or 8 digits, the check character
        must be 0 to 9 or K, the body must not be all the
        same digit, a fully sequential digit run formed by
        the body plus a digit check character is rejected,
        and the check character must equal the mod-11 value
        computed from the body.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace(".", "")
        if "-" not in text:
            return True
        body, check = text.rsplit("-", 1)
        check = check.upper()
        if len(body) not in (7, 8) or not body.isdigit():
            return True
        if len(check) != 1 or check not in "0123456789K":
            return True
        if len(set(body)) == 1:
            # cannot be all the same digit
            return True
        if check.isdigit() and self._is_sequential(body + check):
            # a real issued RUT is never one fully sequential run
            return True
        return check != self._verification_digit(body)

    @staticmethod
    def _verification_digit(body: str) -> str:
        """
        Compute the mod-11 verification digit for a RUT body.

        Digits are multiplied from right to left by weights
        cycling 2, 3, 4, 5, 6, 7. The sum modulo 11 is
        subtracted from 11: 11 maps to 0, 10 maps to K.

        :param body: The 7 or 8 digit RUT body
        :return: The verification digit, 0 to 9 or K
        """
        total = 0
        weight = 2
        for digit in reversed(body):
            total += int(digit) * weight
            weight = weight + 1 if weight < 7 else 2
        result = 11 - (total % 11)
        if result == 11:
            return "0"
        if result == 10:
            return "K"
        return str(result)

    @staticmethod
    def _is_sequential(digits: str) -> bool:
        """
        Check if the digits form a sequential run, allowing 9 to wrap to 0.

        For example, 123456789 ascends by one each step, and
        987654321 descends by one each step.

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
