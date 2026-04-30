"""Evaluation runner — drives scripted conversations through the agent."""

import json
import logging
from datetime import datetime
from pathlib import Path

from payment_agent import Agent
from .personas import ALL_PERSONAS, Persona

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent / "results"


def run_eval_conversation(persona: Persona, script: list[str]) -> dict:
    agent = Agent()
    history: list[dict] = []

    # Opening greeting
    opening = agent.next("")
    history.append({"user": "", "agent": opening["message"]})

    for user_input in script:
        try:
            result = agent.next(user_input)
            history.append({"user": user_input, "agent": result["message"]})
        except Exception as exc:
            logger.error("Error: %s", exc)
            history.append({"user": user_input, "agent": f"[ERROR] {exc}", "error": True})

    return {
        "persona": persona.name,
        "account_id": persona.account_id,
        "conversation_history": history,
        "verified": agent.ctx.verified,
        "transaction_id": agent.ctx.transaction_id,
        "session_closed": agent.ctx.session_closed,
    }


def run_eval(personas: dict[str, Persona] | None = None) -> dict:
    if personas is None:
        personas = ALL_PERSONAS

    results: dict = {"conversations": [], "aggregate": {}}
    for name, persona in personas.items():
        logger.info("Running: %s", name)
        conv = run_eval_conversation(persona, persona.get_conversation_script())
        results["conversations"].append(conv)

    convs = results["conversations"]
    if convs:
        results["aggregate"]["total"] = len(convs)
        results["aggregate"]["verified"] = sum(1 for c in convs if c["verified"])
        results["aggregate"]["paid"] = sum(1 for c in convs if c["transaction_id"])

    return results


def save_eval_results(results: dict, output_path: str | None = None) -> str:
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(RESULTS_DIR / f"eval_{ts}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_eval()
    path = save_eval_results(results)
    print(f"\nResults saved to {path}")
    agg = results["aggregate"]
    print(f"Total: {agg['total']}  Verified: {agg['verified']}  Paid: {agg['paid']}")
