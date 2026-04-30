"""LangGraph graph definition for the payment agent."""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import settings
from .domain.verification import FactorType, classify_factor
from .domain.validators import parse_expiry
from .prompts import SYSTEM_PROMPT
from .state import AgentState
from .tools import ToolContext, make_tools

logger = logging.getLogger(__name__)

# ── Response Cleaning ─────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"</?think>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\[[A-Za-z][A-Za-z\s]{1,30}\]")
_GARBAGE_LINE_RE = re.compile(r"^(?:-+|\u2013+|\u2014+|\d+\.\s*|(?:Hello!\s*)+)$")
_JSON_CONTENT_RE = re.compile(
    r'"type"\s*:\s*"text"|"text"\s*:|"tool_call"|"tool_use"',
    re.IGNORECASE,
)
_REASONING_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"So (?:the|we|I|this|it)|"
    r"But (?:the|we|I|this|it)|"
    r"Wait,|Hmm,|"
    r"The (?:key|issue|problem) is|"
    r"Let me (?:check|think|verify|re-|reconsider)|"
    r"Let's (?:check|think|verify|see)|"
    r"I(?:'m going to| should| need to) (?:check|think|verify|ask|reconsider)"
    r")",
    re.IGNORECASE,
)

# Pattern to detect date-like strings in user input
_DATE_PATTERN_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _extract_text_content(content) -> str:
    """Extract plain text from either a str or a list-of-content-blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p)
    return str(content)


def _strip_preamble_reasoning(text: str) -> str:
    """Strip leading chain-of-thought lines from the model's output."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _REASONING_PREAMBLE_RE.match(stripped):
            start = i + 1
        else:
            break
    return "\n".join(lines[start:]).strip()


def _clean_response(text: str) -> str:
    """Clean LLM output to remove all noise before sending to the customer."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    text = _PLACEHOLDER_RE.sub("", text)
    text = _strip_preamble_reasoning(text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] == "":
                continue
            cleaned_lines.append("")
            continue
        if _GARBAGE_LINE_RE.match(stripped):
            continue
        if _JSON_CONTENT_RE.search(stripped):
            continue
        cleaned_lines.append(line)

    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines).strip()
    result = re.sub(r"^[,;:\s]+", "", result)
    return result


# ── Input Pre-validation ──────────────────────────────────────────────────────

_DATE_PATTERN_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# Matches MM/YY, MM/YYYY, MM-YY, MM-YYYY but NOT YYYY-MM-DD (caught by DOB check)
_EXPIRY_PATTERN_RE = re.compile(r"\b(\d{1,2}[/\-]\d{2,4})\b")

def _prevalidate_user_input(text: str) -> str | None:
    """Validate recognisable patterns in user input BEFORE the LLM runs.

    Checks:
    - DOB in YYYY-MM-DD format: invalid dates are blocked; normalised dates
      (Feb 29) pass through silently.
    - Card expiry in MM/YY or MM/YYYY format: expired or malformed dates
      are blocked immediately so the LLM never accepts bad card data.
    """
    # ── DOB check ─────────────────────────────────────────────────────────────
    date_match = _DATE_PATTERN_RE.search(text)
    if date_match:
        date_str = date_match.group(1)
        result = classify_factor(FactorType.DOB, date_str)

        # Normalised dates pass through silently (explanation comes from tool).
        if result.valid:
            return None

        msg = f'The customer provided "{date_str}" as a date, but it is invalid: {result.error}'
        if result.suggestion:
            msg += f" {result.suggestion}"
        msg += (
            " IMPORTANT: Tell the customer exactly what is wrong with the date. "
            "Do NOT accept it. Ask them to correct it or offer alternative "
            "verification (Aadhaar last 4 digits or pincode)."
        )
        return msg

    # -- Card expiry check (format only) --
    # Validates MM/YY or MM/YYYY format and that month is 1-12.
    # Whether the card has expired is checked by the validate_card tool.
    expiry_match = _EXPIRY_PATTERN_RE.search(text)
    if expiry_match:
        expiry_str = expiry_match.group(1)
        parts = re.split(r"[/\-]", expiry_str)
        if len(parts) == 2:
            try:
                month = int(parts[0])
                year_len = len(parts[1])
                if month < 1 or month > 12:
                    return (
                        f'The customer provided expiry "{expiry_str}" -- '
                        f"month {month} is not valid (must be 01-12). "
                        "Tell the customer the month must be between 01 and 12, "
                        "and ask them to re-enter the expiry in MM/YY format (e.g. 06/27)."
                    )
                if year_len not in (2, 4):
                    return (
                        f'The customer provided expiry "{expiry_str}" -- '
                        f'the year part "{parts[1]}" is not a valid 2 or 4-digit year. '
                        "Tell the customer to use MM/YY (e.g. 06/27) or MM/YYYY (e.g. 06/2027)."
                    )
                # Format is valid -- pass through silently
            except ValueError:
                return (
                    f'The customer provided "{expiry_str}" as expiry -- '
                    "it does not look like a valid date. "
                    "Tell the customer to use MM/YY format (e.g. 06/27)."
                )

    return None



# ── Provenance Guards ─────────────────────────────────────────────────────────

def _user_provided_factor(messages: list, factor_value: str) -> bool:
    """Check if the factor_value actually appears in any HumanMessage."""
    if not factor_value:
        return False
    fv = factor_value.strip().lower()
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if fv in content.lower():
                return True
    return False


def _extract_exact_user_name(messages: list, llm_name: str) -> str:
    """Return the exact string the customer typed for their name.

    The LLM often auto-capitalises names (e.g. 'rahul mehta' → 'Rahul Mehta')
    before passing them to verify_identity.  This would silently bypass the
    strict case-sensitive match_name check.

    Strategy: scan HumanMessages for the LLM's value using a case-insensitive
    search.  When found, return the original substring at that position so that
    whatever casing the customer used is preserved exactly.

    If no match is found (LLM completely fabricated the name), return the
    LLM's value unchanged so the downstream guard can reject it.
    """
    if not llm_name or not llm_name.strip():
        return llm_name

    needle = llm_name.strip().lower()
    for msg in reversed(messages):  # most recent message first
        if not isinstance(msg, HumanMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        idx = content.lower().find(needle)
        if idx != -1:
            # Return the slice from the original content — exact user casing
            return content[idx: idx + len(needle)]
    # Name not found in any human message — return as-is (will fail provenance check)
    return llm_name



def _user_provided_amount(messages: list, amount: float) -> bool:
    """Check if the customer stated an explicit numeric payment amount.

    The customer must type a recognisable number (e.g. "3200", "3,200.50",
    "₹3,200.50") before validate_card or process_payment may be called.
    Vague phrases like "full amount" or "yes" do NOT satisfy this check —
    the agent must ask for a specific figure first.
    """
    if not amount or amount <= 0:
        return False

    candidates = {
        str(int(amount)),       # "3200"
        f"{amount:.2f}",        # "3200.50"
        f"{amount:.1f}",        # "3200.5"
        str(amount),            # "3200.5" (Python default)
        f"{amount:,.2f}",       # "3,200.50"  ← covers "₹3,200.50"
        f"{amount:,.0f}",       # "3,200"
    }

    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        for c in candidates:
            if c in content:
                return True

    return False




# ── Graph Construction ────────────────────────────────────────────────────────

def create_graph(ctx: ToolContext):
    """Build and compile the payment agent graph.

    The graph has three nodes:
        1. ``input_guard``  — deterministic pre-validation of user input
        2. ``assistant``    — calls the LLM (with tools bound)
        3. ``tools``        — executes whatever tools the LLM invoked

    Flow: START → input_guard → assistant → (tool calls?) → tools → assistant
                                           (no tool calls) → END
    """
    tools = make_tools(ctx)
    tools_by_name = {t.name: t for t in tools}

    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        temperature=0.3,
        max_tokens=300,
    )
    llm_with_tools = llm.bind_tools(tools)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def input_guard_node(state: AgentState) -> dict:
        """Deterministic pre-validation before the LLM runs.

        Scans the latest user message for recognisable patterns (dates, etc.)
        and injects a SystemMessage with validation feedback if anything is
        wrong. The LLM then sees this guidance and knows to inform the customer.
        """
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg
                break

        if last_human is None:
            return {"messages": []}

        text = last_human.content if isinstance(last_human.content, str) else str(last_human.content)
        warning = _prevalidate_user_input(text)

        if warning:
            return {"messages": [SystemMessage(content=warning)]}
        return {"messages": []}

    def assistant_node(state: AgentState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(msgs)

        raw = _extract_text_content(response.content) if response.content else ""
        if raw:
            response.content = _clean_response(raw)

        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            name = call["name"]
            if name not in tools_by_name:
                results.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{name}'.",
                        tool_call_id=call["id"],
                    )
                )
                continue

            # ── Guard: block fabricated verification data ──
            if name == "verify_identity":
                # Guard 1: factor_value must have been typed by the customer
                factor_val = call["args"].get("factor_value", "")
                if factor_val and not _user_provided_factor(state["messages"], factor_val):
                    results.append(ToolMessage(
                        content=(
                            "You have the customer's name. Now you need one more piece — "
                            "ask them warmly for their date of birth (YYYY-MM-DD), "
                            "the last 4 digits of their Aadhaar, or their pincode. "
                            "Do not call this tool again until the customer provides that. "
                            "This does NOT count as a verification attempt."
                        ),
                        tool_call_id=call["id"],
                    ))
                    continue

                # Guard 2: preserve the EXACT casing the customer typed for their name.
                # LLMs silently auto-capitalise (e.g. 'rahul mehta' → 'Rahul Mehta'),
                # which would bypass the strict case-sensitive match_name check.
                # We overwrite full_name with the original substring from the chat history.
                llm_name = call["args"].get("full_name", "")
                if llm_name:
                    exact_name = _extract_exact_user_name(state["messages"], llm_name)
                    call["args"] = {**call["args"], "full_name": exact_name}
                    logger.debug(
                        "verify_identity full_name casing: llm=%r → user=%r",
                        llm_name, exact_name,
                    )

            # ── Guard: block fabricated payment amounts ──
            if name in ("validate_card", "process_payment"):
                amount = call["args"].get("amount")
                try:
                    amount_f = float(amount) if amount is not None else 0.0
                except (TypeError, ValueError):
                    amount_f = 0.0
                if not _user_provided_amount(state["messages"], amount_f):
                    results.append(ToolMessage(
                        content=(
                            "HARD STOP — do NOT call any more tools this turn. "
                            "The customer has not yet typed a specific payment amount. "
                            "Ask clearly: 'How much would you like to pay today?' "
                            "The customer MUST type a specific numeric amount (e.g. '3200'). "
                            "If the customer says 'full amount', 'yes', or any non-numeric response, "
                            "respond with: 'Could you please type the exact amount you'd like to pay as a number? "
                            "For example: 3200 or 3,200.50.' "
                            "Do not call validate_card or process_payment again until a "
                            "new message from the customer contains a numeric amount."
                        ),
                        tool_call_id=call["id"],
                    ))
                    continue

            try:
                output = tools_by_name[name].invoke(call["args"])
            except Exception as exc:
                logger.error("Tool %s failed: %s", name, exc)
                output = f"Tool error: {exc}"
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    # ── Routing ───────────────────────────────────────────────────────────────

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # ── Build ─────────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "assistant")
    graph.add_conditional_edges("assistant", should_continue, ["tools", END])
    graph.add_edge("tools", "assistant")

    return graph.compile()
