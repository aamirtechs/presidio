from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class DoCedulaRecognizer(PatternRecognizer):
    """
    Recognize Dominican Republic Cedula de Identidad y Electoral numbers.

    The cedula is the national identity card number
    issued in the Dominican Republic by the Junta
    Central Electoral (JCE). It is also used as the
    holder's taxpayer identification number (an
    individual's RNC) and as a driver licence number.
    It has three parts:

    - A three digit series number, the code of the
      municipality where the holder was registered
      (for example 001 for the Distrito Nacional,
      031 for Santiago; newer cards are assigned
      under the 402 series).
    - A seven digit document number assigned to the
      holder.
    - One Luhn check digit.

    The cedula is written plain, 11 digits with no
    separators, or in its canonical hyphenated form
    XXX-XXXXXXX-X, for example 001-1391820-5. No
    series allowlist is enforced: no authoritative
    complete list of series codes was found, and new
    series are still being assigned.

    Checksum: the check digit uses the standard Luhn
    (mod-10) algorithm. Starting from the rightmost
    digit and moving left, every second digit is
    doubled, and if doubling gives a value above 9
    the two digits of the product are added (which
    is the same as subtracting 9). The number is
    valid when the sum of all the resulting values
    is a multiple of 10. This is confirmed by
    python-stdnum, whose do.cedula module validates
    a cedula with the Luhn algorithm, and by the
    Wikipedia article on the Dominican national
    identity card, which describes the number as a
    series number (3 digits), a document number (7
    digits), and a Luhn checksum. Worked examples,
    verified by hand: 00113918205 (the python-stdnum
    docstring example): the kept digits 5, 2, 1, 3,
    1, 0 sum to 12, and the doubled digits 0, 8, 9,
    1, 0 become 0, 7, 9, 2, 0 and sum to 18, for a
    total of 30; 22400022111 (the python-stdnum
    format example): the kept digits 1, 1, 2, 0, 4,
    2 sum to 10, and the doubled digits 1, 2, 0, 0,
    2 become 2, 4, 0, 0, 4 and sum to 10, for a
    total of 20; 00114905656 (the MicroFocus Eduction
    grammar example 001-1490565-6): the kept digits
    6, 6, 0, 4, 1, 0 sum to 17, and the doubled
    digits 5, 5, 9, 1, 0 become 1, 1, 9, 2, 0 and
    sum to 13, for a total of 30. All three totals
    are multiples of 10. One honest caveat:
    python-stdnum keeps a whitelist of genuinely
    issued cedulas that fail the Luhn check, so
    enforcing the checksum can miss a small set of
    real legacy numbers. That is the safe direction
    for a detector (a missed number rather than a
    false positive), and the whitelist is not
    reproduced here. The recognizer validates the
    checksum in invalidate_result and keeps the
    pattern score at 0.5: it does not promote a
    match to 1.0, because a Luhn check passes about
    one time in ten for an arbitrary same-shape
    token.

    Invalidation is based on: the number must be
    exactly 11 digits after removing the hyphens;
    the number must not be all the same digit (note
    that 00000000000 passes the Luhn check, so this
    rule is needed); a fully sequential digit run is
    rejected; and the Luhn checksum must hold.
    This follows the approach of the Pakistani
    CNIC, Saudi national ID, UAE Emirates ID,
    Egyptian national ID, Bangladesh NID, Sri
    Lanka NIC, Ghana Card, Kazakhstan IIN,
    Ukraine RNOKPP, Uzbekistan PINFL, Morocco
    CNIE, Indonesia NIK, Malaysia MyKad, Vietnam
    CCCD, Chile RUT, Ecuador Cedula, Iran Code
    Melli, Argentina CUIT/CUIL, Mexico CURP, Peru
    RUC, and Colombia NIT recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "do"

    PATTERNS = [
        Pattern(
            "DO_CEDULA",
            r"(?<![A-Za-z0-9.-])(?:\d{11}|\d{3}-\d{7}-\d)(?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "dominican",
        "dominican republic",
        "cedula",
        "cédula",
        "cedula de identidad",
        "jce",
        "junta central electoral",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "DO_CEDULA",
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
        Check if the pattern text cannot be a DO_CEDULA entity.

        The number must be 11 digits after removing
        the hyphens, the number must not be all the
        same digit, a fully sequential digit run is
        rejected, and the Luhn checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace("-", "")
        if len(text) != 11 or not text.isdigit():
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
        Check the Luhn checksum of an 11 digit cedula.

        The check digit is the last digit. Starting
        from the rightmost digit and moving left,
        every second digit is doubled, with products
        above 9 reduced by summing their digits.
        The number is valid when the total is a
        multiple of 10.

        :param digits: The 11 digit cedula
        :return: True when the checksum holds
        """
        check_digit = DoCedulaRecognizer._check_digit(digits[:10])
        return check_digit == digits[10]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the Luhn check digit for the first 10 digits.

        The doubled positions are the 2nd, 4th, 6th,
        8th and 10th digits counting from the right
        of the full number, which are the 1st, 3rd,
        5th, 7th and 9th digits counting from the
        right of this 10 digit body. A product above
        9 is reduced by summing its digits. The
        check digit is (10 - total mod 10) mod 10.

        :param body: The first 10 digits of the cedula
        :return: The check digit, 0 to 9
        """
        total = 0
        for position, digit in enumerate(reversed(body)):
            value = int(digit)
            if position % 2 == 0:
                # doubled position: 1st, 3rd, 5th, 7th, 9th from the right
                value *= 2
                if value > 9:
                    value -= 9
            total += value
        return str((10 - total % 10) % 10)

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
