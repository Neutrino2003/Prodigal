"""Agent class — wraps the LangGraph graph and exposes the required next() interface."""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage

from .graph import create_graph
from .tools import ToolContext

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)
_JSON_ENVELOPE_KEYS = ("text", "content", "message", "response")


def _clean_response(text: str) -> str:
    """Strip <think> tags and unwrap JSON envelopes from model output."""
    clean = _THINK_RE.sub("", text).strip()

    # Some model versions wrap output in {"type":"text","text":"..."}
    if clean.startswith("{"):
        import json
        try:
            parsed = json.loads(clean)
            for key in _JSON_ENVELOPE_KEYS:
                if isinstance(parsed.get(key), str):
                    return parsed[key].strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return clean


class Agent:
    """Payment agent with the required evaluator interface.

    Usage::

        agent = Agent()
        result = agent.next("Hi")         # {"message": "Hello! ..."}
        result = agent.next("ACC1001")    # {"message": "Got it. ..."}
    """

    def __init__(self):
        self.ctx = ToolContext()
        self._graph = create_graph(self.ctx)
        self._state = {"messages": []}

    # ── Required interface ────────────────────────────────────────────────────

    def next(self, user_input: str) -> dict:
        """Process one conversation turn.

        Args:
            user_input: The user's message as a plain string.

        Returns:
            ``{"message": str}`` — the agent's response to display.
        """
        # Add user message (skip empty initial trigger)
        if user_input.strip():
            self._state["messages"].append(HumanMessage(content=user_input))
        elif not self._state["messages"]:
            # First call with empty input — seed with a greeting trigger
            self._state["messages"].append(HumanMessage(content="Hi"))

        try:
            result = self._graph.invoke(self._state)
            self._state = result
        except Exception as exc:
            logger.exception("Graph invocation failed")
            return {"message": f"I'm sorry, something went wrong on my end. Please try again. ({exc})"}

        # Extract the last AI message
        response = self._extract_response()
        return {"message": response}

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def state(self):
        """Expose tool context for tests and eval."""
        return self.ctx

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_response(self) -> str:
        """Pull the last assistant message from the graph state."""
        for msg in reversed(self._state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return _clean_response(msg.content)
        return "I'm here to help — could you tell me a bit more?"
