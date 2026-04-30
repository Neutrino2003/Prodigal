"""Integration tests — test tools directly with mocked APIs."""

from unittest.mock import patch, MagicMock
import httpx

from payment_agent.tools import ToolContext, make_tools
from payment_agent.services.tools import (
    LookupNotFound,
    LookupSuccess,
    PaymentFailure,
    PaymentSuccess,
    AccountData,
)


def _make_account(balance: float = 1250.75) -> AccountData:
    return AccountData(
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",   # actual live API value
        pincode="400001",       # actual live API value
        balance=balance,
    )


def _get_tools(ctx: ToolContext) -> dict:
    tools = make_tools(ctx)
    return {t.name: t for t in tools}


# ── lookup_account ────────────────────────────────────────────────────────────

def test_lookup_account_success():
    ctx = ToolContext()
    tools = _get_tools(ctx)
    account = _make_account()

    with patch("payment_agent.tools._api_lookup", return_value=LookupSuccess(account)):
        result = tools["lookup_account"].invoke({"account_id": "ACC1001"})

    assert "Account found" in result
    assert ctx.account_id == "ACC1001"
    assert ctx.account_data is not None
    assert ctx.account_data.full_name == "Nithin Jain"
    # Guardrail: lookup must NOT leak the customer name or balance
    assert "Nithin" not in result
    assert "1,250" not in result
    assert "1250" not in result


def test_lookup_account_not_found():
    ctx = ToolContext()
    tools = _get_tools(ctx)

    with patch("payment_agent.tools._api_lookup", return_value=LookupNotFound()):
        result = tools["lookup_account"].invoke({"account_id": "INVALID999"})

    assert "No account found" in result
    assert ctx.session_closed


def test_lookup_account_already_loaded():
    ctx = ToolContext()
    ctx.account_data = _make_account()
    ctx.account_id = "ACC1001"  # must match for same-account path
    tools = _get_tools(ctx)

    result = tools["lookup_account"].invoke({"account_id": "ACC1001"})
    assert "already loaded" in result


def test_lookup_account_switch_resets_verification():
    """Security: switching accounts must reset all verification state."""
    ctx = ToolContext()
    ctx.account_data = _make_account()
    ctx.account_id = "ACC1001"
    ctx.verified = True  # was verified on ACC1001
    ctx.verification_attempts = 1
    tools = _get_tools(ctx)

    # Switch to a real but different account
    tools["lookup_account"].invoke({"account_id": "ACC1004"})

    # Verification state must be completely wiped regardless of lookup result
    assert not ctx.verified, "verified must be False after account switch"
    assert ctx.verification_attempts == 0, "attempt counter must reset after account switch"
    assert ctx.account_id == "ACC1004", "account_id must update to new account"


def test_lookup_same_account_preserves_verification():
    """Same-account re-lookup must NOT reset verification state."""
    ctx = ToolContext()
    ctx.account_data = _make_account()
    ctx.account_id = "ACC1001"
    ctx.verified = True
    tools = _get_tools(ctx)

    result = tools["lookup_account"].invoke({"account_id": "ACC1001"})
    assert ctx.verified, "verification must persist for same account"
    assert "already verified" in result.lower()


# ── verify_identity ───────────────────────────────────────────────────────────

def test_verify_identity_success():
    ctx = ToolContext()
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Nithin Jain",
        "factor_type": "dob",
        "factor_value": "1990-05-14",
    })

    assert "verified" in result.lower()
    assert ctx.verified
    assert ctx.verification_attempts == 1


def test_verify_identity_failure():
    ctx = ToolContext()
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Wrong Name",
        "factor_type": "dob",
        "factor_value": "1990-05-14",
    })

    assert "failed" in result.lower()
    assert not ctx.verified
    assert ctx.verification_attempts == 1


def test_verify_identity_exhausted():
    ctx = ToolContext()
    ctx.account_data = _make_account()
    ctx.verification_attempts = 2  # one more attempt left
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Wrong Name",
        "factor_type": "dob",
        "factor_value": "1990-01-01",
    })

    # 3rd attempt failed → session closed
    assert ctx.session_closed
    assert not ctx.verified


def test_verify_identity_no_account():
    ctx = ToolContext()
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Nithin Jain",
        "factor_type": "dob",
        "factor_value": "1990-05-14",
    })
    assert "looked up first" in result.lower()


def test_verify_identity_zero_balance():
    ctx = ToolContext()
    ctx.account_data = _make_account(balance=0.0)
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Nithin Jain",
        "factor_type": "dob",
        "factor_value": "1990-05-14",
    })
    assert "no outstanding balance" in result.lower()
    assert ctx.verified
    assert ctx.session_closed


def test_verify_identity_empty_name():
    """Guardrail: verify must reject empty full_name."""
    ctx = ToolContext()
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "",
        "factor_type": "dob",
        "factor_value": "1990-05-14",
    })
    assert "full name" in result.lower() or "name" in result.lower()
    assert not ctx.verified
    assert ctx.verification_attempts == 0  # not counted


def test_verify_identity_empty_factor_type():
    """Guardrail: verify must reject empty factor_type."""
    ctx = ToolContext()
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Nithin Jain",
        "factor_type": "",
        "factor_value": "1990-05-14",
    })
    assert "factor" in result.lower() or "name" in result.lower()
    assert not ctx.verified
    assert ctx.verification_attempts == 0


def test_verify_identity_empty_factor_value():
    """Guardrail: verify must reject empty factor_value."""
    ctx = ToolContext()
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["verify_identity"].invoke({
        "full_name": "Nithin Jain",
        "factor_type": "dob",
        "factor_value": "",
    })
    assert "factor" in result.lower() or "name" in result.lower()
    assert not ctx.verified
    assert ctx.verification_attempts == 0


# ── validate_card ─────────────────────────────────────────────────────────────

def test_validate_card_success():
    ctx = ToolContext()
    ctx.verified = True
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["validate_card"].invoke({
        "card_number": "4532015112830366",
        "cvv": "123",
        "expiry_month": 12,
        "expiry_year": 2027,
        "amount": 500.0,
    })

    assert "validated" in result.lower()
    assert ctx.payment_amount == 500.0


def test_validate_card_bad_luhn():
    ctx = ToolContext()
    ctx.verified = True
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    result = tools["validate_card"].invoke({
        "card_number": "1234567890123456",
        "cvv": "123",
        "expiry_month": 12,
        "expiry_year": 2027,
        "amount": 500.0,
    })

    assert "error" in result.lower()


def test_validate_card_not_verified():
    ctx = ToolContext()
    tools = _get_tools(ctx)

    result = tools["validate_card"].invoke({
        "card_number": "4532015112830366",
        "cvv": "123",
        "expiry_month": 12,
        "expiry_year": 2027,
        "amount": 500.0,
    })
    assert "verified first" in result.lower()


# ── process_payment ───────────────────────────────────────────────────────────

def test_process_payment_success():
    ctx = ToolContext()
    ctx.verified = True
    ctx.account_id = "ACC1001"
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    with patch("payment_agent.tools._api_pay", return_value=PaymentSuccess("TXN-123")):
        result = tools["process_payment"].invoke({
            "card_number": "4532015112830366",
            "cvv": "123",
            "expiry_month": 12,
            "expiry_year": 2027,
            "cardholder_name": "Nithin Jain",
            "amount": 500.0,
        })

    assert "TXN-123" in result
    assert ctx.transaction_id == "TXN-123"
    assert ctx.session_closed


def test_process_payment_failure_422():
    ctx = ToolContext()
    ctx.verified = True
    ctx.account_id = "ACC1001"
    ctx.account_data = _make_account()
    tools = _get_tools(ctx)

    with patch("payment_agent.tools._api_pay", return_value=PaymentFailure("insufficient_balance")):
        result = tools["process_payment"].invoke({
            "card_number": "4532015112830366",
            "cvv": "123",
            "expiry_month": 12,
            "expiry_year": 2027,
            "cardholder_name": "Nithin Jain",
            "amount": 9999.0,
        })

    assert "insufficient" in result.lower()
    assert not ctx.session_closed  # retryable


def test_process_payment_card_exhausted():
    ctx = ToolContext()
    ctx.verified = True
    ctx.account_id = "ACC1001"
    ctx.account_data = _make_account()
    ctx.card_attempts = 3  # already exhausted
    tools = _get_tools(ctx)

    result = tools["process_payment"].invoke({
        "card_number": "4532015112830366",
        "cvv": "123",
        "expiry_month": 12,
        "expiry_year": 2027,
        "cardholder_name": "Nithin Jain",
        "amount": 500.0,
    })

    assert "maximum" in result.lower()
    assert ctx.session_closed


def test_agent_next_returns_dict():
    """The Agent.next() interface must return {"message": str}."""
    from payment_agent import Agent

    agent = Agent()
    result = agent.next("")
    assert isinstance(result, dict)
    assert "message" in result
    assert isinstance(result["message"], str)
