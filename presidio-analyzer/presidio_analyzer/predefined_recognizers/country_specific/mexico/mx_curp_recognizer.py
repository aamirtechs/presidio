from datetime import date
from typing import List, Optional, Tuple

from presidio_analyzer import Pattern, PatternRecognizer


class MxCurpRecognizer(PatternRecognizer):
    """
    Recognize Mexico CURP population registry codes.

    The CURP (Clave Unica de Registro de Poblacion)
    is the 18 character population registry code
    issued in Mexico by RENAPO. It has seven parts:

    - The first four letters are name initials: the
      paternal surname initial and first internal
      vowel, the maternal surname initial, and the
      first name initial.
    - The next six digits are the birth date as
      YYMMDD.
    - The eleventh character is the sex: H for male,
      M for female, or X for non-binary (issued since
      2022).
    - Characters 12 and 13 are the two letter state
      code of the state of birth, from the catalogue
      of the 32 federal entities plus NE for people
      born abroad (nacido en el extranjero). DF is
      kept for Ciudad de Mexico; CM is Colima.
    - Characters 14 to 16 are the first internal
      consonants of the paternal surname, maternal
      surname, and first name.
    - Character 17 is the homonymy differentiator:
      a digit for people born before 2000, a letter
      for people born since 2000.
    - Character 18 is a check digit.

    Checksum: the check digit uses a mod-10 algorithm.
    Each of the first 17 characters maps to a value
    through the CURP alphabet (0 to 9, then A to Z,
    with N valued 23, the tilde N valued 24, and O
    valued 25 through Z at 36; the tilde N never
    appears in a CURP, since RENAPO replaces it with
    X). Each value is multiplied by a weight
    running from 18 down to 2, and the check digit is
    (10 minus the sum modulo 10) modulo 10. The
    worked examples are HEGG560427MVZRRL04: the
    weighted sum is 2246, 2246 mod 10 is 6, and
    10 minus 6 is 4; and SABC560626MDFLRN01: the
    weighted sum is 1979, 1979 mod 10 is 9, and
    10 minus 9 is 1. Two more published examples
    confirm the algorithm: MOTR930411HJCRMN03 sums
    to 2587, 2587 mod 10 is 7, and 10 minus 7 is 3;
    and GOGR800101HDFRRL07 sums to 2153, 2153 mod 10
    is 3, and 10 minus 3 is 7. The recognizer
    validates the checksum in invalidate_result and
    keeps the pattern score at 0.5: it does not
    promote a match to 1.0, because a mod-10 check
    passes about one time in ten for an arbitrary
    same-shape token.

    Invalidation is based on: the code must be exactly
    18 characters after removing single spaces; the
    characters must sit in the right positions
    (letters, digits, sex, state, consonants,
    differentiator, check digit); the sex letter must
    be H, M, or X; the state code must be one of the
    33 catalogue codes; the birth date must be a real
    calendar date, leap aware, and not in the future;
    the differentiator must be a digit for a pre-2000
    birth and a letter for a birth since 2000; the
    code must not be all the same character; a fully
    sequential run in the ordered 0 to 9 and A to Z
    alphabet is rejected; and the mod-10 checksum
    must hold. This follows the approach of the
    Pakistani CNIC, Saudi national ID, UAE Emirates
    ID, Egyptian national ID, Bangladesh NID, Sri
    Lanka NIC, Ghana Card, Kazakhstan IIN, Ukraine
    RNOKPP, Uzbekistan PINFL, Morocco CNIE, Indonesia
    NIK, Malaysia MyKad, Vietnam CCCD, Chile RUT,
    Iran Code Melli, and Argentina CUIT/CUIL
    recognizers.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    COUNTRY_CODE = "mx"

    PATTERNS = [
        Pattern(
            "MX_CURP",
            r"(?<![A-Za-z0-9.-])[A-Z]{4} ?[0-9]{6} ?[HMX] ?[A-Z]{2} "
            r"?[A-Z]{3} ?[0-9A-Z] ?[0-9](?![A-Za-z0-9-])",
            0.5,
        ),
    ]

    CONTEXT = [
        "mexico",
        "mexican",
        "curp",
        "clave unica",
        "registro de poblacion",
        "renapo",
        "ine",
    ]

    # Two letter state codes of the 32 federal entities,
    # plus NE for people born abroad (nacido en el
    # extranjero). DF is kept for Ciudad de Mexico;
    # CM is Colima. Source: RENAPO state catalogue
    # as published by the Secretaria de Gobernacion.
    STATE_CODES = {
        "AS",  # Aguascalientes
        "BC",  # Baja California
        "BS",  # Baja California Sur
        "CC",  # Campeche
        "CL",  # Coahuila
        "CM",  # Colima
        "CS",  # Chiapas
        "CH",  # Chihuahua
        "DF",  # Ciudad de Mexico
        "DG",  # Durango
        "GT",  # Guanajuato
        "GR",  # Guerrero
        "HG",  # Hidalgo
        "JC",  # Jalisco
        "MC",  # Estado de Mexico
        "MN",  # Michoacan
        "MS",  # Morelos
        "NT",  # Nayarit
        "NL",  # Nuevo Leon
        "OC",  # Oaxaca
        "PL",  # Puebla
        "QT",  # Queretaro
        "QR",  # Quintana Roo
        "SP",  # San Luis Potosi
        "SL",  # Sinaloa
        "SR",  # Sonora
        "TC",  # Tabasco
        "TS",  # Tamaulipas
        "TL",  # Tlaxcala
        "VZ",  # Veracruz
        "YN",  # Yucatan
        "ZS",  # Zacatecas
        "NE",  # Nacido en el extranjero
    }

    SEX_LETTERS = {"H", "M", "X"}

    # Character values for the mod-10 checksum: digits
    # keep their face value, A is 10 through N at 23,
    # the tilde N is 24, and O is 25 through Z at 36.
    _CHECK_ALPHABET = "0123456789ABCDEFGHIJKLMN\xd1OPQRSTUVWXYZ"

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "MX_CURP",
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
        Check if the pattern text cannot be an MX_CURP entity.

        The code must be 18 characters after removing
        single spaces, with letters, digits, sex, state,
        consonants, differentiator, and check digit in
        the right positions. The sex letter must be H,
        M, or X, the state code must be in the 33 code
        catalogue, the birth date must be a real
        calendar date and not in the future, the
        differentiator must be a digit for a pre-2000
        birth and a letter for a birth since 2000, the
        code must not be all the same character, a fully
        sequential run is rejected, and the mod-10
        checksum must hold.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        text = pattern_text.strip().replace(" ", "")
        if len(text) != 18:
            return True
        if not self._has_valid_shape(text):
            return True
        if text[10] not in self.SEX_LETTERS:
            # only H, M, and X are issued as sex letters
            return True
        if text[11:13] not in self.STATE_CODES:
            # the state code must be one of the 33 catalogue codes
            return True
        birth = self._birth_date(text[4:10])
        if birth is None or birth > date.today():
            # the birth date must be real and not in the future
            return True
        differentiator = text[16]
        if birth.year < 2000 and not differentiator.isdigit():
            # pre-2000 births take a digit differentiator
            return True
        if birth.year >= 2000 and differentiator.isdigit():
            # births since 2000 take a letter differentiator
            return True
        if len(set(text)) == 1:
            # cannot be all the same character
            return True
        if self._is_sequential(text):
            # a real issued code is never one fully sequential run
            return True
        return not self._has_valid_checksum(text)

    @staticmethod
    def _has_valid_shape(text: str) -> bool:
        """
        Check the 18 characters sit in the right positions.

        Four letters, six digits, a sex letter, two
        state letters, three consonant letters, an
        alphanumeric differentiator, and a check digit.
        Only plain ASCII capitals are accepted, which
        also keeps the checksum alphabet lookup safe.

        :param text: The 18 character CURP, spaces removed
        :return: True when the shape holds
        """
        letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return (
            all(char in letters for char in text[:4])
            and text[4:10].isdigit()
            and text[10] in letters
            and all(char in letters for char in text[11:16])
            and text[16] in letters | set("0123456789")
            and text[17].isdigit()
        )

    @staticmethod
    def _birth_date(date_part: str) -> Optional[date]:
        """
        Decode the YYMMDD birth date part to a real date.

        A two digit year above the current two digit
        year maps to the 1900s, otherwise to the 2000s,
        matching the other date encoded recognizers.
        A month, day, or leap day that cannot exist
        returns None.

        :param date_part: The six digit YYMMDD birth date
        :return: The decoded date, or None when it cannot exist
        """
        year_two_digits = int(date_part[:2])
        current_two_digits = date.today().year % 100
        if year_two_digits > current_two_digits:
            year = 1900 + year_two_digits
        else:
            year = 2000 + year_two_digits
        try:
            return date(year, int(date_part[2:4]), int(date_part[4:6]))
        except ValueError:
            return None

    @staticmethod
    def _has_valid_checksum(text: str) -> bool:
        """
        Check the mod-10 checksum of an 18 character CURP.

        The first 17 characters are weighted 18 down to
        2 through the CURP alphabet, and the check digit
        is (10 minus the sum modulo 10) modulo 10.

        :param text: The 18 character CURP, spaces removed
        :return: True when the checksum holds
        """
        check_digit = MxCurpRecognizer._check_digit(text[:17])
        return check_digit == text[17]

    @staticmethod
    def _check_digit(body: str) -> str:
        """
        Compute the check digit for the first 17 characters.

        Each character value comes from the CURP alphabet
        (0 to 9, then A at 10 through N at 23, the tilde
        N at 24, O at 25 through Z at 36)
        and is multiplied by a weight running from 18
        down to 2. The check digit is (10 minus the sum
        modulo 10) modulo 10, which is 0 when the sum is
        already a multiple of 10.

        :param body: The first 17 characters of the CURP
        :return: The check digit, 0 to 9
        """
        alphabet = MxCurpRecognizer._CHECK_ALPHABET
        total = sum(
            alphabet.index(char) * (18 - index) for index, char in enumerate(body)
        )
        return str((10 - (total % 10)) % 10)

    @staticmethod
    def _is_sequential(text: str) -> bool:
        """
        Check if the text forms a sequential run in the ordered alphabet.

        The ordered alphabet is digits 0 to 9 followed by
        letters A to Z, so 0123456789ABCDEFGH ascends by
        one each step and HGFEDCBA9876543210 descends.

        :param text: Alphanumeric string to check
        :return: True if every character follows the previous one sequentially
        """
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        try:
            values = [alphabet.index(char) for char in text]
        except ValueError:
            return False
        ascending = all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))
        descending = all(values[i] - 1 == values[i + 1] for i in range(len(values) - 1))
        return ascending or descending
