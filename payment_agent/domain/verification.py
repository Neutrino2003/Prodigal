"""Identity verification with input classification and matching.

This module is the single source of truth for identity verification logic.
It handles:
  1. Factor classification — validates that a value matches its claimed type
  2. Name matching — case-insensitive comparison
  3. Factor matching — type-specific comparison against account data
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from ..services.tools import AccountData


# ── Factor Types ──────────────────────────────────────────────────────────────

class FactorType(str, Enum):
    DOB = "dob"
    AADHAAR = "aadhaar_last4"
    PINCODE = "pincode"

    @classmethod
    def from_string(cls, value: str) -> Optional["FactorType"]:
        """Parse a factor type string, accepting common aliases."""
        mapping = {
            "dob": cls.DOB,
            "date_of_birth": cls.DOB,
            "aadhaar": cls.AADHAAR,
            "aadhaar_last4": cls.AADHAAR,
            "pincode": cls.PINCODE,
        }
        return mapping.get(value.strip().lower())


# ── Classification: validate that a value looks like the claimed type ────────

@dataclass(frozen=True)
class ClassifyResult:
    """Result of classifying a factor value against its claimed type."""
    valid: bool
    cleaned_value: str = ""
    error: str = ""
    suggestion: str = ""
    # Set when the raw value was invalid but was automatically adjusted.
    normalized: bool = False
    normalization_note: str = ""


def classify_factor(factor_type: FactorType, raw_value: str) -> ClassifyResult:
    """Validate that raw_value is structurally correct for the given factor_type.

    This does NOT check whether it matches account data — only whether
    the value is the right shape (a real date, 4 digits, 6 digits, etc.).
    """
    value = raw_value.strip()

    if factor_type == FactorType.DOB:
        return _classify_dob(value)
    if factor_type == FactorType.AADHAAR:
        return _classify_aadhaar(value)
    if factor_type == FactorType.PINCODE:
        return _classify_pincode(value)

    return ClassifyResult(valid=False, error=f"Unknown factor type: {factor_type}")


def _classify_dob(value: str) -> ClassifyResult:
    """Validate a date-of-birth string (YYYY-MM-DD)."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return ClassifyResult(
            valid=False,
            error="Please use the format YYYY-MM-DD (e.g. 1990-05-14).",
            suggestion="You could also verify with Aadhaar last 4 digits or pincode.",
        )

    year, month, day = int(value[:4]), int(value[5:7]), int(value[8:10])

    if month < 1 or month > 12:
        return ClassifyResult(
            valid=False,
            error=f"Month {month:02d} is not valid (must be 01–12).",
        )
    if day < 1 or day > 31:
        return ClassifyResult(
            valid=False,
            error=f"Day {day:02d} is not valid (must be 01–31).",
        )

    # Calendar validity — catches Feb 29 on non-leap years, Apr 31, etc.
    try:
        dob = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        if month == 2 and day == 29:
            return ClassifyResult(
                valid=False,
                error=f"{year} is not a leap year, so February 29 didn't exist.",
                suggestion="If your birthday is Feb 29, try checking the year.",
            )
        if month == 2 and day > 28:
            return ClassifyResult(
                valid=False,
                error=f"February only has 28 days (or 29 in a leap year), not {day}.",
                suggestion="Please double-check your date of birth.",
            )
        return ClassifyResult(
            valid=False,
            error=f"{value} is not a valid calendar date.",
        )

    today = datetime.now().date()
    if dob > today:
        return ClassifyResult(valid=False, error="Date of birth can't be in the future.")
    if (today - dob).days > 120 * 365:
        return ClassifyResult(valid=False, error="That date seems too far in the past.")

    return ClassifyResult(valid=True, cleaned_value=value)


def _classify_aadhaar(value: str) -> ClassifyResult:
    """Validate Aadhaar last-4 digits."""
    digits = value.replace(" ", "")
    if not digits.isdigit():
        return ClassifyResult(valid=False, error="Aadhaar last 4 must be digits only.")
    if len(digits) != 4:
        return ClassifyResult(
            valid=False,
            error=f"Expected exactly 4 digits, got {len(digits)}.",
        )
    return ClassifyResult(valid=True, cleaned_value=digits)


def _classify_pincode(value: str) -> ClassifyResult:
    """Validate an Indian pincode (6 digits)."""
    digits = value.replace(" ", "")
    if not digits.isdigit():
        return ClassifyResult(valid=False, error="Pincode must be digits only.")
    if len(digits) != 6:
        return ClassifyResult(
            valid=False,
            error=f"Indian pincodes are 6 digits, got {len(digits)}.",
        )
    return ClassifyResult(valid=True, cleaned_value=digits)


# ── Matching: compare submitted data against account records ─────────────────

def match_name(submitted: str, expected: str) -> bool:
    """Strict (case-sensitive) name comparison.

    The assignment explicitly requires no fuzzy matching and no case-insensitive
    workarounds for names.  Only leading/trailing whitespace is stripped — every
    other character, including capitalisation, must match exactly.
    """
    return submitted.strip() == expected.strip()


def match_factor(
    factor_type: FactorType,
    submitted_value: str,
    account_data: AccountData,
) -> bool:
    """Compare a submitted factor value against the account's stored value."""
    submitted = submitted_value.strip()

    if factor_type == FactorType.DOB:
        return submitted == account_data.dob.strip()
    if factor_type == FactorType.AADHAAR:
        return submitted == account_data.aadhaar_last4.strip()
    if factor_type == FactorType.PINCODE:
        return submitted == account_data.pincode.strip()

    return False


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a full identity verification attempt."""
    success: bool
    message: str
    name_matched: bool = False
    factor_matched: bool = False
    is_format_error: bool = False  # True if the value failed structural validation
    normalization_note: str = ""  # Non-empty when a date was silently adjusted (e.g. Feb 29)


def verify_identity(
    account_data: AccountData,
    submitted_name: str,
    factor_type: str,
    factor_value: str,
) -> VerifyResult:
    """Full identity verification: name match + secondary factor match.

    Steps:
      1. Parse and validate factor_type
      2. Classify factor_value (structural validation)
      3. Match name against account records
      4. Match factor against account records
      5. Return detailed result

    Returns VerifyResult with success=True only if BOTH name and factor match.
    """
    # Step 1: parse factor type
    ftype = FactorType.from_string(factor_type)
    if ftype is None:
        return VerifyResult(
            success=False,
            message=f"Unknown verification factor: '{factor_type}'. "
                    f"Use 'dob', 'aadhaar_last4', or 'pincode'.",
        )

    # Step 2: classify (structural validation)
    classification = classify_factor(ftype, factor_value)
    if not classification.valid:
        msg = f"Invalid {ftype.value}: {classification.error}"
        if classification.suggestion:
            msg += f" {classification.suggestion}"
        return VerifyResult(
            success=False,
            message=msg,
            is_format_error=True,
        )

    # Step 3: match name
    name_ok = match_name(submitted_name, account_data.full_name)

    # Step 4: match factor
    factor_ok = match_factor(ftype, classification.cleaned_value, account_data)

    # Step 5: result
    # Carry normalization note so callers can inform the customer
    note = classification.normalization_note if classification.normalized else ""

    if name_ok and factor_ok:
        return VerifyResult(
            success=True,
            message="Verification successful.",
            name_matched=True,
            factor_matched=True,
            normalization_note=note,
        )

    return VerifyResult(
        success=False,
        message="The details provided do not match our records.",
        name_matched=name_ok,
        factor_matched=factor_ok,
        normalization_note=note,
    )
