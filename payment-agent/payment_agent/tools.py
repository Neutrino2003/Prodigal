"""LangGraph tools with guardrails.

Each tool validates inputs, enforces preconditions, calls the API, and
returns a human-readable result string for the LLM to interpret.

Business state lives in ``ToolContext`` — a mutable object captured by
closures so tools can read/write it without modifying LangGraph state
directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.tools import tool

from .config import settings
from .domain.validators import (
    detect_card_type,
    validate_amount,
    validate_card_number,
    validate_cvv_length,
    validate_expiry,
)
from .domain.verification import verify_identity as _verify_identity
from .services.tools import (
    LookupNotFound,
    LookupSuccess,
    PaymentFailure,
    PaymentSuccess,
    lookup_account as _api_lookup,
    process_payment as _api_pay,
    AccountData,
)

logger = logging.getLogger(__name__)

MAX_VERIFICATION_ATTEMPTS = 3
MAX_CARD_ATTEMPTS = 3


@dataclass
class ToolContext:
    """Mutable business state shared across all tool closures."""
    account_id: str | None = None
    account_data: AccountData | None = None
    verified: bool = False
    verification_attempts: int = 0
    payment_amount: float | None = None
    transaction_id: str | None = None
    card_attempts: int = 0
    session_closed: bool = False


def make_tools(ctx: ToolContext) -> list:
    """Create tool instances bound to the given context."""

    @tool
    def lookup_account(account_id: str) -> str:
        """Look up a customer account by their account ID (e.g. ACC1001).
        Call this after the customer provides their account ID.

        Args:
            account_id: The customer's account identifier.
        """
        account_id = account_id.strip()
        if not account_id:
            return "Error: account_id is empty. Ask the customer for a valid account ID."

        # ── Same account: return current state ───────────────────────────────
        if ctx.account_data is not None and ctx.account_id == account_id:
            if ctx.verified:
                return "Account loaded and identity already verified. Proceed to payment."
            return "Account already loaded. Identity verification still needed."

        # ── Different account requested: MUST reset all verification state ────
        # Verification for one account NEVER carries over to another account.
        if ctx.account_data is not None and ctx.account_id != account_id:
            logger.warning(
                "Account switch detected: %s → %s. Resetting verification state.",
                ctx.account_id, account_id,
            )
            ctx.account_data = None
            ctx.account_id = None
            ctx.verified = False
            ctx.verification_attempts = 0
            ctx.payment_amount = None
            ctx.transaction_id = None
            ctx.card_attempts = 0
            ctx.session_closed = False

        result = _api_lookup(account_id)

        if isinstance(result, LookupSuccess):
            ctx.account_id = account_id
            ctx.account_data = result.account_data
            return (
                "Account found and loaded. "
                "Do not reveal the customer's name or balance — identity verification needed first. "
                "Previous session verification does NOT apply — customer must re-verify for this account."
            )
        if isinstance(result, LookupNotFound):
            ctx.session_closed = True
            return (
                "No account found for that ID. "
                "Let the customer know and suggest they double-check or contact support. "
                "Close the session."
            )
        return f"System error looking up the account: {result.message}. Ask them to try again."

    @tool
    def verify_identity(full_name: str, factor_type: str, factor_value: str) -> str:
        """Verify the customer's identity using their name and one secondary factor.

        Args:
            full_name: Customer's full name as they stated it.
            factor_type: One of 'dob', 'aadhaar_last4', or 'pincode'.
            factor_value: The value for the chosen factor (e.g. '1990-05-14').
        """
        # ── Preconditions ─────────────────────────────────────────────────────
        if ctx.account_data is None:
            return "Error: account must be looked up first. Call lookup_account."
        if ctx.verified:
            return "Identity already verified. Proceed to payment."
        if ctx.verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
            ctx.session_closed = True
            return (
                "Maximum verification attempts reached. "
                "Apologise and ask the customer to contact support. Close session."
            )

        # ── Guard: reject incomplete calls (not counted as attempts) ──────────
        if not full_name or not full_name.strip():
            return (
                "You still need the customer's full name. "
                "Ask them for it warmly before calling this tool again. "
                "Not counted as an attempt."
            )
        if not factor_type or not factor_type.strip():
            return (
                "You have the customer's name. Now ask them warmly for one verification "
                "factor: date of birth (YYYY-MM-DD), Aadhaar last 4 digits, or pincode. "
                "Not counted as an attempt."
            )
        if not factor_value or not factor_value.strip():
            return (
                "You have the customer's name. Now ask them warmly for one verification "
                "factor: date of birth (YYYY-MM-DD), Aadhaar last 4 digits, or pincode. "
                "Not counted as an attempt."
            )

        # ── Guard: reject placeholder values ──────────────────────────────────
        PLACEHOLDERS = {
            "unknown", "n/a", "na", "none", "not provided", "not available",
            "null", "tbd", "placeholder", "xxx", "0000",
        }
        if factor_value.strip().lower() in PLACEHOLDERS:
            return (
                "The value looks like a placeholder, not real customer data. "
                "Ask the customer warmly for their actual detail. Not counted as an attempt."
            )

        # ── Classify + Verify (single call handles everything) ────────────────
        result = _verify_identity(
            ctx.account_data,
            full_name.strip(),
            factor_type.strip().lower(),
            factor_value.strip(),
        )

        # Classification failed (bad format/invalid date) — not a real attempt
        if result.is_format_error:
            return (
                f"TELL THE CUSTOMER: \"{result.message}\" "
                "Then warmly ask them to try again or offer an alternative factor. "
                "Not counted as an attempt."
            )

        # Real verification attempt — count it
        ctx.verification_attempts += 1

        # Build normalization preamble if the date was auto-adjusted (e.g. Feb 29 on non-leap year)
        norm_prefix = ""
        if result.normalization_note:
            norm_prefix = (
                "IMPORTANT — before sharing the outcome, first tell the customer warmly: "
                f"\"{result.normalization_note}\" "
                "Then share the verification outcome below. "
            )

        if result.success:
            ctx.verified = True
            balance = ctx.account_data.balance
            if balance <= 0:
                ctx.session_closed = True
                return (
                    norm_prefix +
                    "Identity verified! But the account has no outstanding balance. "
                    "Let the customer know there's nothing to pay. Close the session."
                )
            return (
                norm_prefix +
                f"Identity verified successfully! "
                f"Outstanding balance: \u20b9{balance:,.2f}. "
                f"Ask the customer how much they'd like to pay (up to \u20b9{balance:,.2f})."
            )

        # Failed -- report remaining attempts
        remaining = MAX_VERIFICATION_ATTEMPTS - ctx.verification_attempts
        attempt_num = ctx.verification_attempts
        if remaining <= 0:
            ctx.session_closed = True
            return (
                norm_prefix +
                f"Verification failed (attempt {attempt_num} of {MAX_VERIFICATION_ATTEMPTS}) -- "
                "maximum attempts exhausted. "
                "MUST tell the customer: 'I'm sorry, I was unable to verify your identity "
                f"after {MAX_VERIFICATION_ATTEMPTS} attempts. For your security, I cannot "
                "proceed further. Please visit your nearest branch or call our support line.' "
                "Then close the session."
            )
        return (
            norm_prefix +
            f"Verification failed (attempt {attempt_num} of {MAX_VERIFICATION_ATTEMPTS}, "
            f"{remaining} attempt(s) remaining). "
            f"MUST tell the customer: 'I'm sorry, the details you provided don't match our records. "
            f"This was attempt {attempt_num} of {MAX_VERIFICATION_ATTEMPTS}. "
            f"You have {remaining} attempt(s) remaining.' "
            "Then ask them to try again with the correct full name and a verification factor."
        )

    @tool
    def validate_card(
        card_number: str,
        cvv: str,
        expiry_month: int,
        expiry_year: int,
        amount: float,
    ) -> str:
        """Validate card details and payment amount BEFORE calling process_payment.

        Args:
            card_number: The card number (digits only).
            cvv: Card verification value.
            expiry_month: Expiry month (1-12).
            expiry_year: Expiry year (4-digit, e.g. 2027).
            amount: Payment amount in rupees.
        """
        if not ctx.verified:
            return "Error: identity must be verified first."

        balance = ctx.account_data.balance if ctx.account_data else 0.0

        # Validate amount
        valid, error = validate_amount(amount, balance)
        if not valid:
            return f"Amount error: {error}. Ask for a valid amount up to ₹{balance:,.2f}."

        # Validate card number (Luhn + length)
        digits = card_number.strip().replace(" ", "")
        valid, error = validate_card_number(digits)
        if not valid:
            return f"Card number error: {error}. Ask the customer to re-enter."

        # Validate CVV
        card_type = detect_card_type(digits)
        valid, error = validate_cvv_length(cvv.strip(), card_type)
        if not valid:
            return f"CVV error: {error}. Ask the customer to re-enter."

        # Validate expiry
        valid, error = validate_expiry(expiry_month, expiry_year)
        if not valid:
            return f"Expiry error: {error}. Ask the customer for a valid expiry date."

        ctx.payment_amount = amount
        return (
            f"Card validated successfully ({card_type.value.upper()}). "
            f"Amount: ₹{amount:,.2f}. "
            f"Now call process_payment to complete the transaction."
        )

    @tool
    def process_payment(
        card_number: str,
        cvv: str,
        expiry_month: int,
        expiry_year: int,
        cardholder_name: str,
        amount: float,
    ) -> str:
        """Process the actual card payment via the payment API.
        Only call AFTER validate_card succeeds.

        Args:
            card_number: Validated card number.
            cvv: Card CVV.
            expiry_month: Expiry month (1-12).
            expiry_year: Expiry year (4-digit).
            cardholder_name: Name on the card.
            amount: Payment amount in rupees.
        """
        if not ctx.verified:
            return "Error: identity must be verified first."
        if not ctx.account_id:
            return "Error: no account loaded."

        if ctx.card_attempts >= MAX_CARD_ATTEMPTS:
            ctx.session_closed = True
            return (
                "Maximum payment attempts reached. "
                "Apologise and ask the customer to contact support. Close session."
            )

        ctx.card_attempts += 1
        payload = {
            "account_id": ctx.account_id,
            "amount": amount,
            "payment_method": {
                "type": "card",
                "card": {
                    "cardholder_name": cardholder_name,
                    "card_number": card_number.strip().replace(" ", ""),
                    "cvv": cvv.strip(),
                    "expiry_month": expiry_month,
                    "expiry_year": expiry_year,
                },
            },
        }

        result = _api_pay(payload)

        if isinstance(result, PaymentSuccess):
            ctx.transaction_id = result.transaction_id
            ctx.session_closed = True
            return (
                f"Payment successful! Transaction ID: {result.transaction_id}. "
                f"Amount: ₹{amount:,.2f}. "
                "Thank the customer and close the session."
            )

        if isinstance(result, PaymentFailure):
            code = result.error_code
            remaining = MAX_CARD_ATTEMPTS - ctx.card_attempts

            if code == "insufficient_balance":
                balance = ctx.account_data.balance if ctx.account_data else 0.0
                return (
                    f"Payment declined — insufficient balance. "
                    f"Available: ₹{balance:,.2f}. Ask for a lower amount."
                )
            if remaining <= 0:
                ctx.session_closed = True
                return (
                    f"Payment declined ({code}). No attempts remaining. "
                    "Ask the customer to contact support. Close session."
                )
            return (
                f"Payment declined — {code.replace('_', ' ')}. "
                f"{remaining} attempt(s) remaining. Ask for corrected card details."
            )

        # PaymentError (network etc.)
        remaining = MAX_CARD_ATTEMPTS - ctx.card_attempts
        if remaining <= 0:
            ctx.session_closed = True
            return "Payment system error. No attempts remaining. Close session."
        return (
            f"Network error processing payment. "
            f"{remaining} attempt(s) remaining. Ask to re-enter card details."
        )

    return [lookup_account, verify_identity, validate_card, process_payment]
