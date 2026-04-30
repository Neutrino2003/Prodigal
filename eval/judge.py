"""LLM-based scoring of conversations."""

import json
import logging

from payment_agent.config import settings

logger = logging.getLogger(__name__)


def judge_conversation(conversation_result: dict) -> float:
    conversation_text = _format_conversation(conversation_result["conversation_history"])
    rubric = (
        "Score this payment agent conversation on verification, communication, "
        "validation, privacy, and error handling. Return JSON: "
        '{"score": <0-100>, "reasoning": "<brief>"}'
    )
    prompt = (
        f"{rubric}\n\nConversation:\n{conversation_text}\n\n"
        f"Verified: {conversation_result['verified']}\n"
        f"Transaction ID: {conversation_result['transaction_id']}\n"
        f"Session Closed: {conversation_result.get('session_closed', False)}"
    )

    score = _call_judge_llm(prompt)
    return score if score is not None else _fallback_score(conversation_result)


def _format_conversation(history: list[dict]) -> str:
    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Agent: {turn['agent']}")
    return "\n".join(lines)


def _call_judge_llm(prompt: str) -> float | None:
    if not settings.nvidia_api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; eval judge using fallback score")
        return None

    client = OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
    )
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
            # Disable reasoning for deterministic JSON scoring output
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:
        logger.debug("Judge LLM call failed: %s", str(exc))
        return None

    content = response.choices[0].message.content if response.choices else ""
    try:
        parsed = json.loads((content or "").strip())
        return float(parsed.get("score", 50))
    except Exception:
        return None


def _fallback_score(conversation_result: dict) -> float:
    score = 50
    if conversation_result.get("verified"):
        score += 25
    if conversation_result.get("transaction_id"):
        score += 25
    error_count = sum(1 for turn in conversation_result["conversation_history"] if turn.get("error"))
    score -= error_count * 10
    if conversation_result.get("session_closed") and not conversation_result.get("transaction_id"):
        score -= 10
    return max(0, min(100, score))
