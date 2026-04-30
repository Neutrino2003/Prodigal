"""Unit tests — validators, tools context, and agent interface."""

from datetime import datetime

from payment_agent import Agent
from payment_agent.domain.validators import (
    CardType,
    detect_card_type,
    luhn_check,
    validate_account_id,
    validate_amount,
    validate_card_number,
    validate_date_format,
    validate_expiry,
)
from payment_agent.tools import MAX_CARD_ATTEMPTS, MAX_VERIFICATION_ATTEMPTS, ToolContext


def test_module_imports():
    from payment_agent.agent import Agent
    from payment_agent.graph import create_graph
    from payment_agent.prompts import SYSTEM_PROMPT
    from payment_agent.tools import ToolContext, make_tools

    assert Agent is not None
    assert create_graph is not None
    assert SYSTEM_PROMPT
    assert ToolContext is not None
    assert make_tools is not None


def test_validators_core():
    assert luhn_check("4532015112830366")
    assert not luhn_check("1234567890123456")
    assert validate_date_format("1988-02-29")[0]
    assert not validate_date_format("1988-02-30")[0]
    assert validate_account_id("ACC1001")[0]
    assert not validate_account_id("  ")[0]
    assert validate_card_number("4532015112830366")[0]
    assert not validate_card_number("1234567890123456")[0]
    assert detect_card_type("4532015112830366") == CardType.VISA
    assert detect_card_type("378282246310005") == CardType.AMEX


def test_amount_and_expiry_validation():
    assert validate_amount(500.0, 5000.0)[0]
    assert not validate_amount(-1.0, 5000.0)[0]
    assert not validate_amount(5000.01, 5000.0)[0]
    now = datetime.now()
    assert validate_expiry(now.month, now.year)[0]
    assert validate_expiry(12, now.year + 1)[0]
    if now.month > 1:
        assert not validate_expiry(now.month - 1, now.year)[0]


def test_tool_context_defaults():
    ctx = ToolContext()
    assert ctx.account_id is None
    assert ctx.account_data is None
    assert not ctx.verified
    assert ctx.verification_attempts == 0
    assert ctx.card_attempts == 0
    assert not ctx.session_closed
    assert ctx.transaction_id is None


def test_tool_context_constants():
    assert MAX_VERIFICATION_ATTEMPTS == 3
    assert MAX_CARD_ATTEMPTS == 3


def test_next_returns_dict():
    """next() must return {"message": str}."""
    agent = Agent()
    result = agent.next("")
    assert isinstance(result, dict)
    assert "message" in result
    assert isinstance(result["message"], str)
    assert len(result["message"]) > 0


def test_agent_exposes_ctx():
    agent = Agent()
    assert hasattr(agent, "ctx")
    assert isinstance(agent.ctx, ToolContext)
    assert not agent.ctx.verified
