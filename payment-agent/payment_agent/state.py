"""LangGraph state definition for the payment agent."""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Full graph state — messages are append-only via LangGraph's reducer."""
    messages: Annotated[list[BaseMessage], add_messages]
