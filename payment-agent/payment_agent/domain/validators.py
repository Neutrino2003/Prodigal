import math
from datetime import datetime
from enum import Enum


class CardType(Enum):
    AMEX = "amex"
    VISA = "visa"
    MASTERCARD = "mastercard"
    DISCOVER = "discover"
    UNKNOWN = "unknown"


def validate_account_id(account_id: str | None) -> tuple[bool, str | None]:
    if account_id is None or not account_id.strip():
        return False, "Account ID cannot be empty"
    return True, None


def validate_date_format(date_str: str) -> tuple[bool, str | None]:
    """Basic format check — kept for backward compatibility."""
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True, None
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD"


def validate_dob(date_str: str) -> tuple[bool, str | None]:
    """Strict date-of-birth validation with specific failure reasons.

    Handles:
    - Wrong format
    - Structurally invalid dates (month > 12, day > 31)
    - Impossible calendar dates (Feb 29 on non-leap years, Apr 31, etc.)
    - Future dates
    - Implausibly old dates (>120 years ago)

    Returns (True, None) on success, (False, reason) on failure.
    """
    raw = date_str.strip()

    # ── Step 1: format check (YYYY-MM-DD) ────────────────────────────────────
    import re as _re
    if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return False, "Please use the format YYYY-MM-DD (e.g. 1990-05-14)"

    year_s, month_s, day_s = raw.split("-")
    year, month, day = int(year_s), int(month_s), int(day_s)

    # ── Step 2: range sanity before trying to parse ───────────────────────────
    if month < 1 or month > 12:
        return False, f"Month {month} is not valid — it should be between 01 and 12"
    if day < 1 or day > 31:
        return False, f"Day {day} is not valid — it should be between 01 and 31"

    # ── Step 3: actual calendar validity (catches Feb 29 on non-leap years) ───
    try:
        dob = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        # Diagnose the common leap-year case specifically
        if month == 2 and day == 29:
            return (
                False,
                f"{year} is not a leap year, so February 29 didn't exist. "
                f"If your birthday is Feb 29, try Feb 28 or check the year. "
                f"Or you can verify with your Aadhaar last 4 or pincode instead.",
            )
        return False, f"{raw} is not a valid calendar date — please double-check it"

    # ── Step 4: not in the future ─────────────────────────────────────────────
    today = datetime.now().date()
    if dob > today:
        return False, "Date of birth can't be in the future"

    # ── Step 5: plausibility (no one is 120+) ────────────────────────────────
    if (today - dob).days > 120 * 365:
        return False, "That date seems too far in the past — please double-check"

    return True, None


def luhn_check(card_number: str) -> bool:
    digits = card_number.strip().replace(" ", "")
    if not digits.isdigit():
        return False

    nums = [int(c) for c in digits][::-1]
    for i in range(1, len(nums), 2):
        doubled = nums[i] * 2
        nums[i] = doubled - 9 if doubled > 9 else doubled
    return sum(nums) % 10 == 0


def detect_card_type(card_number: str) -> CardType:
    digits = card_number.strip().replace(" ", "")
    if not digits.isdigit():
        return CardType.UNKNOWN

    if digits.startswith("4"):
        return CardType.VISA
    if digits.startswith(("34", "37")):
        return CardType.AMEX
    if len(digits) >= 4:
        first4 = int(digits[:4])
        first2 = int(digits[:2])
        if 51 <= first2 <= 55 or 2221 <= first4 <= 2720:
            return CardType.MASTERCARD
    if digits.startswith("6011") or digits.startswith("65") or digits.startswith("64") or digits.startswith("62"):
        return CardType.DISCOVER
    return CardType.UNKNOWN


def validate_card_length(card_number: str, card_type: CardType) -> tuple[bool, str | None]:
    length = len(card_number.strip().replace(" ", ""))
    if card_type == CardType.AMEX:
        return (length == 15, None if length == 15 else "Amex card must be 15 digits")
    if card_type in (CardType.VISA, CardType.MASTERCARD, CardType.DISCOVER):
        return (length == 16, None if length == 16 else "Card must be 16 digits")
    return False, "Card type not recognized"


def validate_card_number(card_number: str) -> tuple[bool, str | None]:
    digits = card_number.strip().replace(" ", "")
    if not digits.isdigit():
        return False, "Card number must contain only digits"

    card_type = detect_card_type(digits)
    length_ok, length_error = validate_card_length(digits, card_type)
    if not length_ok:
        return False, length_error
    if not luhn_check(digits):
        return False, "Card number failed validation"
    return True, None


def validate_cvv_length(cvv: str, card_type: CardType) -> tuple[bool, str | None]:
    value = cvv.strip()
    if not value.isdigit():
        return False, "CVV must be numeric"
    if card_type == CardType.AMEX:
        return (len(value) == 4, None if len(value) == 4 else "Amex CVV must be 4 digits")
    if card_type in (CardType.VISA, CardType.MASTERCARD, CardType.DISCOVER):
        return (len(value) == 3, None if len(value) == 3 else "CVV must be 3 digits")
    return False, "Unknown card type"


def validate_expiry(month: int, year: int) -> tuple[bool, str | None]:
    if month < 1 or month > 12:
        return False, "Month must be between 1 and 12"

    now = datetime.now()
    if year < now.year:
        return False, "Card has expired"
    if year == now.year and month < now.month:
        return False, "Card has expired"
    if year > now.year + 15:
        return False, f"Expiry year {year} seems too far in the future — please double-check"
    return True, None


def parse_expiry(raw: str) -> tuple[int | None, int | None, str | None]:
    """Parse a raw expiry string into (month, year) integers.

    Accepts common formats: MM/YY, MM/YYYY, MM-YY, MM-YYYY.
    Two-digit years are expanded relative to the current century
    (e.g. '27' → 2027, '99' → 2099).

    Returns:
        (month, year, None)  on success
        (None, None, error)  on failure
    """
    import re as _re
    raw = raw.strip()
    m = _re.fullmatch(r"(\d{1,2})[/\-](\d{2,4})", raw)
    if not m:
        return None, None, (
            f'"{raw}" is not a recognised expiry format. '
            "Please use MM/YY or MM/YYYY (e.g. 06/27 or 06/2027)."
        )

    month_str, year_str = m.group(1), m.group(2)
    month = int(month_str)
    year_raw = int(year_str)

    # Expand 2-digit years: treat as 20XX
    if year_raw < 100:
        year = 2000 + year_raw
    else:
        year = year_raw

    valid, error = validate_expiry(month, year)
    if not valid:
        return None, None, error

    return month, year, None


def validate_amount(amount: float, max_amount: float) -> tuple[bool, str | None]:
    if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
        return False, "Amount must be a valid number"
    if amount <= 0:
        return False, "Amount must be greater than 0"
    if amount > max_amount:
        return False, f"Amount exceeds your balance of ₹{max_amount:.2f}"
    if round(float(amount), 2) != float(amount):
        return False, "Amount must have at most 2 decimal places"
    return True, None
